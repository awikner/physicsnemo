"""
UC S2S rollout CRPS fine-tuning.

Grows autoregressive rollout by 1 step every `rollout_step_interval` iterations.
Only Pangu decoder layers are trainable; SI UNet and encoders are frozen.

Replaces manual dist.init_process_group / YParams / argparse with
physicsnemo DistributedManager + Hydra. All scientific logic is preserved.
"""

import logging
import os
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.distributed as dist
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from utils.data_loader_multifiles import get_data_loader
from utils.losses import Latitude_weighted_CRPSLoss


class RolloutFinetuner(Trainer):
    """Fine-tune Pangu decoder with growing-rollout CRPS loss via the SI ensemble."""

    def __init__(
        self,
        params,
        dist_mgr: DistributedManager,
        max_rollout_steps: int = 10,
        rollout_step_interval: int = 2000,
        initial_rollout_steps: int = 1,
    ):
        super().__init__(params, dist_mgr)

        self.max_rollout_steps = max_rollout_steps
        self.rollout_step_interval = rollout_step_interval
        self.iters_offset = (initial_rollout_steps - 1) * rollout_step_interval

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        for p in self.model_vae.parameters():
            p.requires_grad = False

        self.diff_model = self._build_si_model()
        self._load_si_checkpoint()

        # Freeze SI UNet; only decoder layers receive gradients
        for p in self.diff_model.unet.parameters():
            p.requires_grad = False

        decoder_modules = [
            self.diff_model.model_det.upsample,
            self.diff_model.model_det.layer4,
            self.diff_model.model_det.patchrecovery2d,
            self.diff_model.model_det.patchrecovery3d,
        ]
        for mod in decoder_modules:
            for p in mod.parameters():
                p.requires_grad = True

        decoder_params = [p for mod in decoder_modules for p in mod.parameters()]
        n_decoder = sum(p.numel() for p in decoder_params)
        n_total = sum(p.numel() for p in self.diff_model.parameters())
        logging.info("Decoder params: %s / %s total", f"{n_decoder:,}", f"{n_total:,}")

        self.optimizer = torch.optim.Adam(
            decoder_params,
            lr=float(params.get("finetune_lr", params.lr)),
            weight_decay=params.weight_decay,
        )

        self.get_dataset_rollout()
        self.scaler = GradScaler()
        self.wandb_enabled = bool(params.get("log_to_wandb", False) and wandb.run is not None)
        if self.wandb_enabled and self.world_rank == 0:
            wandb.define_metric("iters")
            for key in ("train/crps_loss", "train/crps_surface", "train/crps_upper_air",
                        "train/crps_diagnostic", "train/rmse_diagnostic",
                        "train/rollout_steps", "lr"):
                wandb.define_metric(key, step_metric="iters")

    # ------------------------------------------------------------------
    # Rolling schedule
    # ------------------------------------------------------------------

    @property
    def current_rollout_steps(self):
        steps = 1 + (self.iters + self.iters_offset) // self.rollout_step_interval
        return min(steps, self.max_rollout_steps)

    # ------------------------------------------------------------------
    # Dataset (validate=True for multi-step targets)
    # ------------------------------------------------------------------

    def get_dataset_rollout(self):
        logging.info("rank %d, rollout data loader init", self.world_rank)
        distributed = self.dist.distributed

        self.params["sel_dates"] = False

        def _make(year_start, year_end):
            return get_data_loader(
                self.params, self.params.data_dir, distributed,
                year_start=year_start, year_end=year_end,
                train=False, validate=True, shuffle=True,
            )

        if self.params.get("train_year_to_year", False):
            self.train_data_loaders, self.train_datasets, self.train_samplers = [], [], []
            for yr in range(self.params.train_year_start, self.params.train_year_end):
                dl, ds, samp = _make(yr, yr + 1)
                self.train_data_loaders.append(dl)
                self.train_datasets.append(ds)
                self.train_samplers.append(samp)
        else:
            dl, ds, samp = _make(self.params.train_year_start, self.params.train_year_end)
            self.train_data_loaders = [dl]
            self.train_datasets = [ds]
            self.train_samplers = [samp]

        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            self.params, self.params.data_dir, distributed,
            year_start=self.params.val_year_start,
            year_end=self.params.val_year_end,
            train=False, num_inferences=self.params.num_inferences, validate=True,
        )

    def _prepare_inputs_batch_rollout(self, data):
        dev, f32 = self.device, torch.float32
        if self.params.get("has_diagnostic", False):
            surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc, _t = data
            tgt_diag = tgt_diag.to(dev, dtype=f32, non_blocking=True)
        else:
            surf, ua, tgt_sfc, tgt_ua, vbc, _t = data
            tgt_diag = None
        return (
            surf.to(dev, dtype=f32, non_blocking=True),
            ua.to(dev, dtype=f32, non_blocking=True),
            tgt_sfc.to(dev, dtype=f32, non_blocking=True),
            tgt_ua.to(dev, dtype=f32, non_blocking=True),
            tgt_diag,
            vbc.to(dev, dtype=f32, non_blocking=True),
        )

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _build_si_model(self):
        vae_m = self.model_vae.module if hasattr(self.model_vae, "module") else self.model_vae
        det_m = self.model_det.module if hasattr(self.model_det, "module") else self.model_det
        return StochasticInterpolant(T=1000, VAEEncoder=vae_m, DETEncoder=det_m,
                                     params=self.params).to(self.device)

    def _load_si_checkpoint(self):
        def _load(path, module):
            ckpt = torch.load(path, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
            state = ckpt["model_state"]
            if any(k.startswith("module.") for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            module.load_state_dict(state, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)

        si_path = self.params.get("checkpoint_path_diff", None)
        if si_path and os.path.isfile(si_path):
            ckpt = torch.load(si_path, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
            state = ckpt["model_state"]
            if any(k.startswith("module.") for k in state):
                state = OrderedDict((k[7:], v) for k, v in state.items())
            unet_state = {k: v for k, v in state.items() if k.startswith("unet.")}
            self.diff_model.load_state_dict(unet_state, strict=False)
            logging.info("Loaded SI UNet weights from %s", si_path)

        self.diff_model.freeze_encoder()
        self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()

    def _get_latitudes(self):
        return torch.from_numpy(np.array(self.params.lat)).float()

    # ------------------------------------------------------------------
    # Rollout CRPS step
    # ------------------------------------------------------------------

    def _rollout_crps_step(self, surface_in, upper_air_in, varying_boundary_data,
                           target_surface_steps, target_upper_air_steps,
                           target_diagnostic_steps, num_rollout, num_samples):
        latitudes = self._get_latitudes().to(self.device)
        crps_loss_fn = Latitude_weighted_CRPSLoss(latitudes, num_ensemble_members=num_samples)

        s_cur, ua_cur = surface_in, upper_air_in
        total_loss = total_crps_sfc = total_crps_ua = total_crps_diag = total_precip_rmse = 0.0

        for step in range(num_rollout):
            vbc_step = varying_boundary_data[:, step]
            tgt_sfc  = target_surface_steps[:, step]
            tgt_ua   = target_upper_air_steps[:, step]
            tgt_diag = target_diagnostic_steps[:, step] if target_diagnostic_steps is not None else None

            with torch.no_grad():
                surface_cat = self.diff_model._prepare_surface(
                    s_cur, self.constant_boundary_data, vbc_step)
                z = self.diff_model._encode_vae(surface_cat, ua_cur)
                x, skip = self.diff_model._encode_det(surface_cat, ua_cur, train=False)
                source_sigma = float(self.params.get("source_sigma", x.std().item())) * 1.5
                x_si = self.diff_model.generate(
                    model_diff=self.diff_model.unet, z=z, x=x,
                    num_samples=num_samples, sample_shape=x.shape,
                    device=self.device, temperature=1.5, source_sigma=source_sigma,
                )

            torch.cuda.empty_cache()
            BS = x_si.shape[0]

            with autocast(device_type="cuda", dtype=torch.float16):
                Pl_est, Lat_est, Lon_est = self.diff_model.model_det.EST_input_resolution
                x_dec = x_si.reshape(BS, -1, 240 * self.params.updown_scale_factor)
                x_dec = self.diff_model.model_det.upsample(x_dec)
                x_dec = self.diff_model.model_det.layer4(x_dec, train=False)
                skip_exp = skip.repeat_interleave(num_samples, dim=0)
                output = torch.cat([x_dec, skip_exp], dim=-1)
                output = output.transpose(1, 2).reshape(BS, -1, Pl_est, Lat_est, Lon_est)

                pred_sfc  = self.diff_model.model_det.patchrecovery2d(output[:, :, -1, :, :])
                pred_ua   = self.diff_model.model_det.patchrecovery3d(output[:, :, :-1, :, :])
                output_2D = self.diff_model.model_det.patchrecovery2d(output[:, :, -1, :, :])
                pred_sfc  = output_2D[:, self.diff_model.surface_prognostic_idxs]
                pred_diag = output_2D[
                    :,
                    self.diff_model.num_surface_vars:
                    self.diff_model.num_surface_vars + self.diff_model.num_diagnostic_vars
                ].reshape(BS, -1, pred_sfc.shape[-2], pred_sfc.shape[-1])

                tgt_sfc_rep = tgt_sfc.repeat_interleave(num_samples, dim=0)
                tgt_ua_rep  = tgt_ua.repeat_interleave(num_samples, dim=0)

                crps_sfc  = crps_loss_fn(pred_sfc, tgt_sfc_rep)
                crps_ua   = crps_loss_fn(pred_ua, tgt_ua_rep)
                step_loss = 0.2 * (crps_sfc + crps_ua)
                crps_diag = torch.tensor(0.0, device=self.device)
                precip_rmse = torch.tensor(0.0, device=self.device)

                if tgt_diag is not None and self.diff_model.num_diagnostic_vars > 0:
                    precip_idxs = [i for i, v in enumerate(self.params.diagnostic_variables)
                                   if "precipitation" in v]
                    tgt_diag_rep = tgt_diag.repeat_interleave(num_samples, dim=0)
                    crps_diag = crps_loss_fn(pred_diag[:, precip_idxs], tgt_diag_rep[:, precip_idxs])
                    step_loss = step_loss + 0.8 * crps_diag

            self.scaler.scale(step_loss / num_rollout).backward()

            total_crps_sfc    += crps_sfc.item()
            total_crps_ua     += crps_ua.item()
            total_crps_diag   += crps_diag.item()
            total_precip_rmse += precip_rmse.item()
            total_loss        += step_loss.item()

            with torch.no_grad():
                s_cur  = pred_sfc.reshape(-1, num_samples, *pred_sfc.shape[1:]).mean(1).detach().float()
                ua_cur = pred_ua.reshape(-1, num_samples, *pred_ua.shape[1:]).mean(1).detach().float()

            del x_si, x_dec, skip, skip_exp, output, pred_sfc, pred_ua, pred_diag, step_loss
            torch.cuda.empty_cache()

        n = num_rollout
        return total_loss / n, total_crps_sfc / n, total_crps_ua / n, total_crps_diag / n, total_precip_rmse / n

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_one_epoch_rollout(self):
        self.epoch += 1
        num_samples = int(self.params.get("crps_ensemble_size", 2))
        loss_val = 0.0

        for year_idx, loader in enumerate(self.train_data_loaders):
            samp = self.train_samplers[year_idx]
            if samp is not None:
                samp.set_epoch(self.epoch)
            for i, data in enumerate(loader):
                if self.params.get("mode") == "test" and i >= self.params.get("test_iterations", 10):
                    break

                self.iters += 1
                num_rollout = self.current_rollout_steps

                surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc = self._prepare_inputs_batch_rollout(data)
                tgt_sfc  = tgt_sfc[:, :num_rollout]
                tgt_ua   = tgt_ua[:, :num_rollout]
                if tgt_diag is not None:
                    tgt_diag = tgt_diag[:, :num_rollout]
                vbc = vbc[:, :num_rollout]

                self.optimizer.zero_grad()
                loss_val, crps_sfc, crps_ua, crps_diag, precip_rmse = self._rollout_crps_step(
                    surface_in=surf, upper_air_in=ua, varying_boundary_data=vbc,
                    target_surface_steps=tgt_sfc, target_upper_air_steps=tgt_ua,
                    target_diagnostic_steps=tgt_diag,
                    num_rollout=num_rollout, num_samples=num_samples,
                )
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.diff_model.parameters() if p.requires_grad], 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if i % 100 == 0:
                    logging.info(
                        "epoch=%d iter=%d rollout=%d loss=%.4f",
                        self.epoch, self.iters, num_rollout, loss_val,
                    )

                if self.wandb_enabled and self.world_rank == 0:
                    wandb.log({
                        "train/crps_loss": loss_val, "train/crps_surface": crps_sfc,
                        "train/crps_upper_air": crps_ua, "train/crps_diagnostic": crps_diag,
                        "train/rmse_diagnostic": precip_rmse,
                        "train/rollout_steps": num_rollout,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }, step=self.iters)

                if self.iters % 200 == 0:
                    self._save(f"rollout_ckpt_{self.iters}.tar")

        return {"train_crps_loss": loss_val, "epoch": self.epoch}

    def _save(self, name):
        if self.world_rank != 0:
            return
        out_dir = os.path.join(self.params.get("experiment_dir", "./checkpoints"), "training_checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        torch.save({
            "model_state": self.diff_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch, "iters": self.iters,
        }, path)
        logging.info("Saved rollout checkpoint: %s", path)

    def run(self, epochs: int = 10):
        logger = PythonLogger("rollout_finetune")
        while self.epoch < epochs:
            with LaunchLogger("train", epoch=self.epoch) as log:
                logs = self.train_one_epoch_rollout()
                log.log_epoch({"train_crps_loss": logs.get("train_crps_loss", 0)})
            if self.wandb_enabled and self.world_rank == 0:
                wandb.log(logs, step=self.epoch)
            self._save("rollout_last.tar")
        logger.info("Rollout fine-tuning done.")


@hydra.main(version_base="1.2", config_path="conf", config_name="config_finetune_rollout")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist_mgr = DistributedManager()

    LaunchLogger.initialize()
    logger = PythonLogger("main")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    trainer = RolloutFinetuner(
        cfg, dist_mgr,
        max_rollout_steps=cfg.get("max_rollout_steps", 10),
        rollout_step_interval=cfg.get("rollout_step_interval", 2000),
        initial_rollout_steps=cfg.get("initial_rollout_steps", 1),
    )

    finetune_ckpt = cfg.get("finetune_ckpt", None)
    if finetune_ckpt and os.path.isfile(finetune_ckpt):
        trainer._resume(finetune_ckpt)

    trainer.run(epochs=cfg.get("max_epochs", 10))
    logger.info("DONE — rank %d", dist_mgr.rank)


if __name__ == "__main__":
    main()
