"""
UC S2S stochastic interpolant / diffusion training.

Subclasses Trainer from train.py. Replaces manual dist.init_process_group /
YParams / argparse with physicsnemo DistributedManager + Hydra.
All scientific logic (SI training step, scatter plots, checkpoint save/restore)
is preserved from the original train_diffusion.py.
"""

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from networks.diffusion import ConditionalDiffusionModel
from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer


class DiffusionTrainer(Trainer):
    """Train the stochastic interpolant / diffusion model on top of frozen Pangu + VAE."""

    def __init__(self, params, dist_mgr: DistributedManager):
        super().__init__(params, dist_mgr)

        self.model_vae, self.model_det = self.get_model()
        self.mask_bool, self.land_mask = self.get_land_mask_bool()

        # Freeze VAE — only SI/UNet trains
        for p in self.model_vae.parameters():
            p.requires_grad = False

        self.diff_model = self.get_diffusion_model()
        self.temp_path = os.path.split(self.params.checkpoint_path_diff)[0]
        self.optimizer = torch.optim.Adam(
            self.diff_model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

        if self.params.get("checkpoint_path_diff") and os.path.isfile(self.params.checkpoint_path_diff):
            self.restore_diff_checkpoint(self.params.checkpoint_path_diff)
            self.setup_scheduler(restart=False)
        else:
            self.setup_scheduler(restart=True)

        self._reload_vae_checkpoint()
        self.scaler = GradScaler()

        self.wandb_enabled = bool(
            self.params.get("log_to_wandb", False) and wandb.run is not None
        )
        if self.params.get("log_to_wandb", False) and not self.wandb_enabled and self.world_rank == 0:
            logging.warning("W&B enabled in config but wandb.init() not active — skipping wandb.log.")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _reload_vae_checkpoint(self):
        """Force-reload VAE + det weights into diff_model encoder/model_det."""
        def _load(path, module):
            ckpt = torch.load(path, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
            raw_state = ckpt["model_state"]
            if any(k.startswith("module.") for k in raw_state):
                raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
            module.load_state_dict(raw_state, strict=True)

        _load(self.params.checkpoint_path_vae_c1, self.diff_model.encoder)
        logging.info("Loaded VAE weights into diff_model.encoder")
        _load(self.params.checkpoint_path_det, self.diff_model.model_det)
        logging.info("Loaded det weights into diff_model.model_det")
        self.diff_model.freeze_encoder()
        self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()

    def restore_diff_checkpoint(self, checkpoint_path_diff):
        ckpt = torch.load(checkpoint_path_diff, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
        raw_state = ckpt["model_state"]
        if any(k.startswith("module.") for k in raw_state):
            raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
        # Skip frozen encoder/model_det — those come from their own checkpoints
        unet_state = {
            k: v for k, v in raw_state.items()
            if not k.startswith("encoder.") and not k.startswith("model_det.")
        }
        self.diff_model.load_state_dict(unet_state, strict=False)
        self.diff_model._encoder_param_checksums = self.diff_model._snapshot_encoder_params()
        self.iters = ckpt["iters"]
        self.startEpoch = ckpt["epoch"]
        self.epoch = ckpt["epoch"]
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logging.info("Restored diffusion checkpoint from epoch %d, iters %d", self.epoch, self.iters)

    def get_diffusion_model(self):
        model_type = self.params.get("diffusion_model_type", "SI")
        vae_module = self.model_vae.module if hasattr(self.model_vae, "module") else self.model_vae
        det_module = self.model_det.module if hasattr(self.model_det, "module") else self.model_det

        if model_type == "diffusion":
            self.diff_model = ConditionalDiffusionModel(
                T=1000, VAEEncoder=vae_module, DETEncoder=det_module, params=self.params
            ).to(self.device)
        elif model_type == "SI":
            self.diff_model = StochasticInterpolant(
                T=1000, VAEEncoder=vae_module, DETEncoder=det_module, params=self.params
            ).to(self.device)
        else:
            raise ValueError(f"Unknown diffusion_model_type: {model_type}")

        n_params = sum(p.numel() for p in self.diff_model.parameters())
        logging.info("Diffusion model parameters: %s", f"{n_params:,}")
        return self.diff_model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_one_epoch_diffusion(self) -> dict:
        self.epoch += 1
        total_iterations = sum(len(l) for l in self.train_data_loaders)
        logging.info("Expected total batches: %d", total_iterations)
        loss = torch.tensor(0.0, device=self.device)

        for year_idx, train_data_loader in enumerate(self.train_data_loaders):
            for i, data in enumerate(train_data_loader):
                if i % 100 == 0:
                    logging.info("batch %d / year_idx %d", i, year_idx)

                if self.params.get("mode") == "test" and i >= self.params.get("test_iterations", 10):
                    break

                self.iters += 1
                surface, upper_air, tgt_sfc, tgt_ua, _, varying_bnd = self._prepare_inputs_batch(data)
                varying_bnd = varying_bnd if varying_bnd is not None else torch.zeros(1, device=self.device)

                self.optimizer.zero_grad()

                do_scatter = (self.iters % 1000 == 0) and (self.world_rank == 0)
                scatter_path = os.path.join(os.path.dirname(self.params.checkpoint_path_diff), "scatter_tmp.png")

                with autocast(device_type="cuda", dtype=torch.float16):
                    loss = self.diff_model.training_step(
                        surface_in=surface,
                        constant_boundary=self.constant_boundary_data,
                        varying_boundary=varying_bnd,
                        upper_air_in=upper_air,
                        target_surface_in=tgt_sfc,
                        target_upper_air=tgt_ua,
                        plot_freq=200,
                        iter=self.iters,
                        plot_scatter=do_scatter,
                        scatter_path=scatter_path,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.diff_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.params.get("scheduler") == "OneCycleLR":
                    self.scheduler.step()

                logs = {"loss": loss.detach(), "lr": self.optimizer.param_groups[0]["lr"]}
                if self.world_rank == 0:
                    if self.wandb_enabled:
                        wandb.log(logs, step=self.iters)
                        if do_scatter and os.path.isfile(scatter_path):
                            wandb.log({"scatter_pred_vs_gt": wandb.Image(scatter_path)}, step=self.iters)

                # Mid-epoch checkpoint every 2000 iters
                if i >= 2000 and i % 2000 == 0:
                    diff_path = os.path.join(self.temp_path, f"diff_ckpt_{self.iters}.tar")
                    last_path = os.path.join(self.temp_path, "last_ckpt.tar")
                    self.save_checkpoint(diff_path, self.diff_model)
                    self.save_checkpoint(last_path, self.diff_model)
                    if self.wandb_enabled:
                        wandb.save(last_path, base_path=self.temp_path)

        return {"train_loss": loss, "epoch": self.epoch}

    def train_diff(self, epochs: int = None):
        epochs = epochs or self.params.get("max_epochs", 50)
        logger = PythonLogger("train_diffusion")
        for epoch in range(epochs):
            with LaunchLogger("train", epoch=epoch) as log:
                logs = self.training_one_epoch_diffusion()
                log.log_epoch({"train_loss": float(logs.get("train_loss", 0))})

            if self.wandb_enabled:
                wandb.log(logs, step=self.epoch)

            if self.world_rank == 0:
                last_path = os.path.join(self.temp_path, "diff_ckpt.tar")
                self.save_checkpoint(last_path, self.diff_model)
                logger.info("Saved checkpoint: %s", last_path)
                if self.wandb_enabled:
                    wandb.save(last_path, base_path=self.temp_path)

    @torch.no_grad()
    def validation_diffusion(self):
        self.diff_model.eval()
        max_batches = int(self.params.get("valid_max_batches", 30))
        loss_sum = torch.zeros(1, device=self.device)
        step_count = torch.zeros(1, device=self.device)

        for i, data in tqdm.tqdm(enumerate(self.valid_data_loader), total=max_batches):
            if max_batches > 0 and i >= max_batches:
                break
            surface, upper_air, tgt_sfc, tgt_ua, _, varying_bnd = self._prepare_inputs_batch(data)
            varying_bnd = varying_bnd if varying_bnd is not None else torch.zeros(1, device=self.device)
            with autocast(device_type="cuda", dtype=torch.float16):
                val_loss = self.diff_model.training_step(
                    surface_in=surface, constant_boundary=self.constant_boundary_data,
                    varying_boundary=varying_bnd, upper_air_in=upper_air,
                    target_surface_in=tgt_sfc, target_upper_air=tgt_ua,
                    train=False, plot_freq=0, iter=self.iters,
                )
            loss_sum += val_loss.detach().float()
            step_count += 1.0

        if dist.is_initialized():
            dist.all_reduce(loss_sum)
            dist.all_reduce(step_count)

        valid_loss = (loss_sum / torch.clamp_min(step_count, 1.0)).item()
        self.diff_model.train()
        return {"valid_loss": valid_loss}


@hydra.main(version_base="1.2", config_path="conf", config_name="config_diffusion")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist_mgr = DistributedManager()

    LaunchLogger.initialize()
    logger = PythonLogger("main")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    trainer = DiffusionTrainer(cfg, dist_mgr)
    trainer.train_diff(epochs=cfg.get("max_epochs", 50))
    logger.info("DONE — rank %d", dist_mgr.rank)


if __name__ == "__main__":
    main()
