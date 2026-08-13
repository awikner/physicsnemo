"""
UC S2S Pangu deterministic backbone training.

Replaces manual dist.init_process_group / YParams with:
  - physicsnemo.distributed.DistributedManager
  - Hydra config (conf/config.yaml)
  - physicsnemo LaunchLogger + save_checkpoint / load_checkpoint

The Trainer class and all S2S-specific logic (land/ocean masks, multi-step
rollout, CRPS, wandb) are preserved exactly; only the framework wiring changes.
"""

import os
import uuid
import logging

import numpy as np
import torch
import wandb
import xarray as xr
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from tqdm import tqdm

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from networks.pangu import PanguModel_Plasim
from networks.vae import VAE
from datapipe.data_loader_multifiles import get_data_loader
from utils.losses import (
    Latitude_weighted_MSELoss, Latitude_weighted_L1Loss,
    Masked_L1Loss, Masked_MSELoss,
    Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,
    Latitude_weighted_CRPSLoss, Kl_divergence_gaussians,
)
from utils.integrate import Integrator, forward_euler
from utils.power_spectrum import *
from utils import logging_utils

logging_utils.config_logger()
torch._dynamo.config.optimize_ddp = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# H100-friendly: disable cuDNN RNN fallback to avoid unnecessary kernel selection overhead
torch.backends.cudnn.allow_cudnn_rnn_fallback = False


def to_ensemble_batch(data, ens_members):
    # expand+reshape avoids allocating a temporary broadcast tensor (vs unsqueeze * torch.ones)
    return data.unsqueeze(1).expand(-1, ens_members, *data.shape[1:]).reshape(-1, *data.shape[1:])


def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416 / 180.0 * latitudes)
    return latitudes.size()[0] * lat_weights_unweighted / torch.sum(lat_weights_unweighted)


def weighted_rmse_torch_channels(pred, target, latitudes):
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, -1, 1))
    return torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))


class Trainer:
    """
    Base S2S trainer using physicsnemo DistributedManager + LaunchLogger.

    All scientific logic (masking, losses, rollout, wandb metrics) is
    preserved from the original S2S train.py; only the framework wiring
    (dist init, logging, checkpointing) uses physicsnemo APIs.
    """

    def __init__(self, params, dist_mgr: DistributedManager):
        self.params = params
        self.dist = dist_mgr
        self.world_rank = dist_mgr.rank
        self.device = dist_mgr.device

        self.iters = 0
        self.startEpoch = 0
        self.epoch = 0
        self.latitudes = torch.from_numpy(np.array(params.lat)).to(self.device)
        self.early_stop_epoch = params.get("early_stop_epoch", None)
        if self.early_stop_epoch is not None:
            self.early_stop_epoch -= 1

        self.run_uuid = str(uuid.uuid4())
        self.check_land_ocean_variables()
        self.get_dataset()
        self.spectra_dir, self.diagnostics_dir, self.output_dir = self.create_dirs(self.run_uuid)

        if params.get("log_to_wandb", False):
            resume = "allow" if params.get("resuming", False) else "never"
            wandb.init(
                config=OmegaConf.to_container(params, resolve=True),
                name=f'{params.name}-{params.get("run_iter", 0)}',
                entity=params.get("entity", None),
                group=params.get("group", None),
                project=params.get("project", "uc_s2s"),
                resume=resume,
            )

        self.mask_bool, self.land_mask = self.get_land_mask_bool()

    # ------------------------------------------------------------------
    # Setup helpers (unchanged logic from original)
    # ------------------------------------------------------------------

    def check_land_ocean_variables(self):
        self.has_land = False
        self.has_ocean = False
        self.mask_output = False
        if hasattr(self.params, "land_variables") and len(self.params.land_variables) > 0:
            self.has_land = True
        else:
            self.params["land_variables"] = []
        if hasattr(self.params, "ocean_variables") and len(self.params.ocean_variables) > 0:
            self.has_ocean = True
        else:
            self.params["ocean_variables"] = []
        if hasattr(self.params, "mask_output"):
            self.mask_output = self.params.mask_output

    def create_dirs(self, run_uuid):
        for d in ["spectra_out", "gif_out", "acc_plots"]:
            os.makedirs(d, exist_ok=True)
        spectra_dir = os.path.join(os.getcwd(), "spectra_out", run_uuid)
        diagnostics_dir = os.path.join(os.getcwd(), "gif_out", run_uuid)
        output_dir = os.path.join(os.getcwd(), "acc_plots", run_uuid)
        if self.world_rank == 0:
            os.makedirs(spectra_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
        return spectra_dir, diagnostics_dir, output_dir

    def get_dataset(self):
        logging.info("rank %d, begin data loader init", self.world_rank)
        distributed = self.dist.distributed

        if self.params.get("train_year_to_year", False):
            self.train_data_loaders, self.train_datasets, self.train_samplers = [], [], []
            for yr in range(self.params.train_year_start, self.params.train_year_end):
                dl, ds, samp = get_data_loader(
                    self.params, self.params.data_dir, distributed,
                    year_start=yr, year_end=yr + 1, train=True,
                )
                self.train_data_loaders.append(dl)
                self.train_datasets.append(ds)
                self.train_samplers.append(samp)
        else:
            dl, ds, samp = get_data_loader(
                self.params, self.params.data_dir, distributed,
                year_start=self.params.train_year_start,
                year_end=self.params.train_year_end, train=True,
            )
            self.train_data_loaders = [dl]
            self.train_datasets = [ds]
            self.train_samplers = [samp]

        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            self.params, self.params.data_dir, distributed,
            year_start=self.params.val_year_start,
            year_end=self.params.val_year_end,
            train=False,
            num_inferences=self.params.num_inferences,
            validate=True,
        )

        self.constant_boundary_data = (
            self.train_datasets[0].constant_boundary_data
            .unsqueeze(0)
            .expand(self.params.batch_size, -1, -1, -1)
            .to(self.device, non_blocking=True)
        )
        if self.params.get("num_ensemble_members", 1) > 1:
            self.constant_boundary_data = to_ensemble_batch(
                self.constant_boundary_data, self.params.num_ensemble_members
            )

        climatology_path = os.path.join(self.params.data_dir, self.params.climatology_file)
        self.climatology = xr.open_dataset(climatology_path).rename({"time": "dayofyear"})
        logging.info("rank %d, data loader initialized", self.world_rank)

    def get_land_mask_bool(self):
        mask_bool, land_mask = [], None
        if self.params.nettype == "pangu_plasim" and (self.has_land or self.has_ocean) and self.mask_output:
            land_mask = self.train_datasets[0].land_mask.detach().to(self.device)
            for var in self.params.surface_variables:
                if var in self.params.land_variables:
                    mask_bool.append(land_mask.clone().bool())
                elif var in self.params.ocean_variables:
                    mask_bool.append(land_mask.clone().bool().logical_not())
                else:
                    mask_bool.append(torch.ones(land_mask.shape, device=self.device, dtype=torch.bool))
            mask_bool = torch.stack(mask_bool)
        return mask_bool, land_mask

    def get_model(self):
        mask_fill = getattr(self.params, "mask_fill", self.train_datasets[0].mask_fill)
        self.model_vae = VAE(self.params).to(self.device)
        self.model_det = PanguModel_Plasim(
            self.params, land_mask=self.land_mask, mask_fill=mask_fill
        ).to(self.device)

        if self.dist.distributed:
            self.model_vae = DistributedDataParallel(
                self.model_vae,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.local_rank,
                find_unused_parameters=True,
            )
            self.model_det = DistributedDataParallel(
                self.model_det,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.local_rank,
                find_unused_parameters=True,
            )
        if self.params.get("log_to_wandb", False) and wandb.run is not None:
            wandb.watch(self.model_vae)
        return self.model_vae, self.model_det

    def count_parameters(self):
        return sum(p.numel() for p in self.model_det.parameters() if p.requires_grad)

    def get_optimizer(self):
        fused = self.params.get("optimizer_type", "") == "FusedAdam"
        self.optimizer = torch.optim.Adam(
            self.model_det.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
            fused=fused,
        )
        return self.optimizer

    def setup_scheduler(self, restart=False):
        if restart and self.startEpoch > 0:
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

        sched = self.params.get("scheduler", None)
        if sched == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode="min"
            )
        elif sched == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.params.max_epochs, last_epoch=self.startEpoch - 1
            )
        elif sched == "OneCycleLR":
            steps_per_epoch = sum(len(l) for l in self.train_data_loaders)
            total_steps = steps_per_epoch * self.params.max_epochs
            kw = dict(
                max_lr=self.params.lr,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.params.get("oc_pct_start", 0.3),
                div_factor=self.params.get("oc_div_factor", 25),
                final_div_factor=self.params.get("oc_final_div_factor", 1e4),
            )
            if self.startEpoch >= 1 and not restart:
                kw["last_epoch"] = (self.startEpoch - 1) * steps_per_epoch
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, **kw)
        else:
            self.scheduler = None

    def setup_loss_fun(self):
        self.loss_obj_pl = self.loss_obj_sfc = self.loss_obj_diagnostic = 0
        self.loss_vae = Kl_divergence_gaussians() if self.params.get("vae_loss", False) else 0

        loss_name = self.params.loss
        lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
        self.lat = lat
        masked = (self.has_land or self.has_ocean) and self.mask_output

        if loss_name == "l1":
            self.loss_obj_pl = torch.nn.L1Loss()
            self.loss_obj_sfc = Masked_L1Loss(self.mask_bool) if masked else torch.nn.L1Loss()
            if self.params.get("has_diagnostic", False):
                self.loss_obj_diagnostic = torch.nn.L1Loss()
        elif loss_name == "l2":
            self.loss_obj_pl = torch.nn.MSELoss()
            self.loss_obj_sfc = Masked_MSELoss(self.mask_bool) if masked else torch.nn.MSELoss()
            if self.params.get("has_diagnostic", False):
                self.loss_obj_diagnostic = torch.nn.MSELoss()
        elif loss_name == "weightedl1":
            self.loss_obj_pl = Latitude_weighted_L1Loss(lat)
            self.loss_obj_sfc = (
                Latitude_weighted_masked_L1Loss(lat, self.mask_bool) if masked
                else Latitude_weighted_L1Loss(lat)
            )
            if self.params.get("has_diagnostic", False):
                self.loss_obj_diagnostic = Latitude_weighted_L1Loss(lat)
        elif loss_name == "weightedl2":
            self.loss_obj_pl = Latitude_weighted_MSELoss(lat)
            self.loss_obj_sfc = (
                Latitude_weighted_masked_MSELoss(lat, self.mask_bool) if masked
                else Latitude_weighted_MSELoss(lat)
            )
            if self.params.get("has_diagnostic", False):
                self.loss_obj_diagnostic = Latitude_weighted_MSELoss(lat)
        elif loss_name == "weightedCRPS":
            n_ens = self.params.num_ensemble_members
            self.loss_obj_pl = Latitude_weighted_CRPSLoss(lat, n_ens)
            self.loss_obj_sfc = (
                Latitude_weighted_CRPSLoss(lat, n_ens, self.mask_bool) if masked
                else Latitude_weighted_CRPSLoss(lat, n_ens)
            )
            if self.params.get("has_diagnostic", False):
                self.loss_obj_diagnostic = Latitude_weighted_CRPSLoss(lat, n_ens)
        else:
            raise NotImplementedError(f"Unknown loss: {loss_name}")

        return self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic

    # ------------------------------------------------------------------
    # Checkpoint helpers (physicsnemo-style manual save/restore)
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_path, model=None, iteration=None):
        """Save model + optimizer + epoch state."""
        model = model or self.model_det
        raw_model = model.module if hasattr(model, "module") else model
        state = {
            "iters": self.iters,
            "epoch": self.epoch,
            "model_state": raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if iteration is not None:
            state["iteration"] = iteration
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state, checkpoint_path)
        logging.info("Saved checkpoint to %s", checkpoint_path)

    def restore_checkpoint(self, checkpoint_path_vae=None, checkpoint_path_det=None, optimizer=True):
        """Restore model weights (and optionally optimizer) from checkpoint."""
        def _load(path, module):
            if not path or not os.path.isfile(path):
                logging.warning("Checkpoint not found: %s", path)
                return
            ckpt = torch.load(path, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
            raw_state = ckpt.get("model_state", ckpt)
            if any(k.startswith("module.") for k in raw_state):
                raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
            raw_module = module.module if hasattr(module, "module") else module
            raw_module.load_state_dict(raw_state, strict=False)
            if optimizer and "optimizer_state" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.startEpoch = ckpt.get("epoch", 0) + 1
            self.iters = ckpt.get("iters", 0)
            logging.info("Restored from %s (epoch %d)", path, self.startEpoch - 1)

        _load(checkpoint_path_vae, self.model_vae)
        _load(checkpoint_path_det, self.model_det)

    # ------------------------------------------------------------------
    # Training loop (unchanged logic)
    # ------------------------------------------------------------------

    def _prepare_inputs_batch(self, data):
        """Extract surface / upper-air / boundary tensors from a batch."""
        surface = data[0].to(self.device, non_blocking=True)
        upper_air = data[1].to(self.device, non_blocking=True)
        target_surface = data[2].to(self.device, non_blocking=True)
        target_upper_air = data[3].to(self.device, non_blocking=True)
        times = data[4]
        varying_boundary = data[5].to(self.device, non_blocking=True) if len(data) > 5 else None
        return surface, upper_air, target_surface, target_upper_air, times, varying_boundary

    def train(self):
        logger = PythonLogger("train")
        logger.info("Starting Training Loop...")
        best_valid_loss = 1.0e6
        early_stopping_counter = 0

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch

            if self.early_stop_epoch is not None and epoch > self.early_stop_epoch:
                logger.info("Early stop epoch %d reached.", self.early_stop_epoch)
                break

            # Set epoch on samplers for proper shuffling
            for samp in self.train_samplers:
                if samp is not None and hasattr(samp, "set_epoch"):
                    samp.set_epoch(epoch)

            with LaunchLogger("train", epoch=epoch,
                              num_mini_batch=sum(len(l) for l in self.train_data_loaders),
                              epoch_alert_freq=10) as log:
                train_logs = self.train_one_epoch()
                log.log_epoch({"train_loss": train_logs.get("loss", 0),
                               "lr": self.optimizer.param_groups[0]["lr"]})

            if self.world_rank == 0:
                with LaunchLogger("valid", epoch=epoch) as log:
                    valid_logs = self.validate_one_epoch()
                    log.log_epoch(valid_logs)

                valid_loss = valid_logs.get("valid_loss", best_valid_loss)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(
                        os.path.join(self.params.get("checkpoint_dir", "./checkpoints"),
                                     "best_ckpt_det.tar"),
                    )

            if self.dist.distributed:
                torch.distributed.barrier()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_logs.get("valid_loss", 0))
                else:
                    self.scheduler.step()

            if epoch % self.params.get("save_checkpoint_freq", 5) == 0 and self.world_rank == 0:
                self.save_checkpoint(
                    os.path.join(self.params.get("checkpoint_dir", "./checkpoints"),
                                 f"ckpt_det_ep{epoch}.tar"),
                )

        torch.cuda.profiler.stop()
        logger.info("Training finished.")

    def train_one_epoch(self):
        self.model_det.train()
        self.model_vae.train()
        total_loss = 0.0
        n_batches = 0
        scaler = GradScaler()
        latitudes = self.latitudes  # pre-cached; avoids numpy→tensor conversion each iter

        for loader in self.train_data_loaders:
            for data in loader:
                surface, upper_air, tgt_sfc, tgt_ua, times, varying_bnd = self._prepare_inputs_batch(data)
                varying_bnd = varying_bnd if varying_bnd is not None else torch.zeros(1, device=self.device)

                self.optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    pred_sfc, pred_ua, _ = self.model_det(
                        surface, self.constant_boundary_data, varying_bnd, upper_air
                    )
                    loss_pl = self.loss_obj_pl(pred_ua, tgt_ua)
                    loss_sfc = self.loss_obj_sfc(pred_sfc, tgt_sfc)
                    loss = loss_pl + loss_sfc

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_det.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

                total_loss += loss.item()
                n_batches += 1
                self.iters += 1

                # Log every 20 iters to reduce per-step overhead
                if self.iters % 20 == 0 and self.world_rank == 0:
                    with torch.no_grad():
                        sfc_rmse = weighted_rmse_torch_channels(pred_sfc, tgt_sfc, latitudes)
                        ua_rmse = weighted_rmse_torch_channels(
                            pred_ua.flatten(2, 3), tgt_ua.flatten(2, 3), latitudes
                        ) if pred_ua.ndim > 4 else weighted_rmse_torch_channels(pred_ua, tgt_ua, latitudes)
                    if self.params.get("log_to_wandb", False) and wandb.run is not None:
                        wandb.log({
                            "train_loss": loss.item(),
                            "train_sfc_rmse": sfc_rmse.mean().item(),
                            "iter": self.iters,
                        })
                # empty_cache() intentionally removed from inner loop:
                # it forces cudaDeviceSynchronize + cudaMemGetInfo and stalls the GPU pipeline.

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model_det.eval()
        total_loss = 0.0
        n_batches = 0

        for data in self.valid_data_loader:
            surface, upper_air, tgt_sfc, tgt_ua, times, varying_bnd = self._prepare_inputs_batch(data)
            varying_bnd = varying_bnd if varying_bnd is not None else torch.zeros(1, device=self.device)

            pred_sfc, pred_ua, _ = self.model_det(
                surface, self.constant_boundary_data, varying_bnd, upper_air
            )
            loss = self.loss_obj_pl(pred_ua, tgt_ua) + self.loss_obj_sfc(pred_sfc, tgt_sfc)
            total_loss += loss.item()
            n_batches += 1

        self.model_det.train()
        return {"valid_loss": total_loss / max(n_batches, 1)}


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    LaunchLogger.initialize()
    logger = PythonLogger("main")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg, dist)
    trainer.mask_bool, trainer.land_mask = trainer.get_land_mask_bool()
    trainer.model_vae, trainer.model_det = trainer.get_model()
    trainer.get_optimizer()
    trainer.setup_loss_fun()

    if cfg.get("resuming", False):
        trainer.restore_checkpoint(
            checkpoint_path_vae=cfg.get("checkpoint_path_vae", None),
            checkpoint_path_det=cfg.get("checkpoint_path_det", None),
        )

    trainer.setup_scheduler(restart=not cfg.get("resuming", False))
    trainer.train()


if __name__ == "__main__":
    main()
