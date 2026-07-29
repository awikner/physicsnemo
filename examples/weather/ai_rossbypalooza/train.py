# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Method 0 (vanilla MoWE) training: a DiT gate over frozen weather experts.

Trains only the gate: per sample the dataset provides each expert's daily
precip (channel 0) + dynamical predictors at a sampled lead ``tau``; the
gate emits per-expert weight + bias fields and the mixture
``P_hat = sum_i w_i (P_i + b_i)`` is scored with a regional loss against
IMERG. Validation reports per-lead regional RMSE / bias / SEEPS for the
gate vs each expert and the equal-weight mean.

Run (single GPU)::

    python train.py

Multi-GPU::

    torchrun --standalone --nproc-per-node=4 train.py [overrides]
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from datapipes.factory import build_dataset
from datapipes.sampler import MixturePairSampler
from losses import (
    build_loss,
    denormalize_precip,
    imd_valid_mask,
    region_weights,
)
from mowe_precip import MoWEPrecipGate, expert_dropout, mix
from seeps import SeepsClimatology
from validation import MixtureValidator

logger = logging.getLogger("mowe_train")


def _maybe_init_wandb(cfg: DictConfig, *, dist) -> bool:
    """wandb on EVERY rank so wandb.run exists everywhere for LaunchLogger;
    only rank 0 is online and drives logging. Non-rank-0 uses mode="disabled"
    (no-op run, no wandb-core service): concurrent offline-mode services on one
    node fail port-file startup on Midway3 (wandb 0.28.1, ServicePollForToken).
    Must run BEFORE LaunchLogger.initialize."""
    wb = cfg.get("wandb", None)
    if wb is None or not bool(wb.get("enabled", False)):
        return False
    try:
        from physicsnemo.utils.logging.wandb import initialize_wandb
    except ImportError:
        if dist.rank == 0:
            PythonLogger("mowe_train").warning(
                "wandb.enabled=True but wandb is not importable; console only."
            )
        return False
    _ent = wb.get("entity", None)
    initialize_wandb(
        project=str(wb.get("project", "ai-rossbypalooza")),
        entity=str(_ent) if _ent else None,
        name=str(wb.get("name", cfg.get("run_name", "mowe"))),
        mode=str(wb.get("mode", "offline")) if dist.rank == 0 else "disabled",
        config=OmegaConf.to_container(cfg, resolve=True),
        init_timeout=int(wb.get("init_timeout", 300)),
    )
    return dist.rank == 0


def _ddp_mean_scalars(values: dict, *, dist) -> dict:
    """Single all-reduce mean over stacked scalars (no per-step host sync)."""
    if not (getattr(dist, "distributed", False) and dist.world_size > 1):
        return values
    import torch.distributed as tdist

    keys = list(values.keys())
    vec = torch.stack(
        [
            (
                values[k].detach()
                if torch.is_tensor(values[k])
                else torch.as_tensor(float(values[k]), device=dist.device)
            )
            .to(device=dist.device, dtype=torch.float32)
            .reshape(())
            for k in keys
        ]
    )
    tdist.all_reduce(vec, op=tdist.ReduceOp.SUM)
    vec = vec / dist.world_size
    return {k: vec[i] for i, k in enumerate(keys)}


def _build_loader(dataset, sampler, loader_cfg) -> DataLoader:
    num_workers = int(loader_cfg.get("num_workers", 4))
    kwargs = dict(
        batch_size=int(loader_cfg.get("batch_size", 4)),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            loader_cfg.get("persistent_workers", True)
        )
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def _build_scheduler(optimizer, *, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        frac = min(1.0, frac)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * frac)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run(cfg: DictConfig) -> None:
    """Training entry point; separated from the hydra wrapper so tests can
    call it with a programmatic config (and repeatedly, for resume)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dist = DistributedManager()
    torch.manual_seed(int(cfg.seed) + dist.rank)

    wandb_active = _maybe_init_wandb(cfg, dist=dist)
    LaunchLogger.initialize(use_wandb=wandb_active)
    plog = PythonLogger("mowe_train")

    # ---------------- data ----------------
    train_ds = build_dataset(cfg.dataset, "train")
    has_val = cfg.dataset.get("val") is not None
    val_ds = build_dataset(cfg.dataset, "val") if has_val else None
    plog.info(
        f"train pairs: {len(train_ds)} | experts: {train_ds.expert_names} | "
        f"channels: {train_ds.channel_names}"
        + (f" | val pairs: {len(val_ds)}" if has_val else "")
    )

    loader_cfg = cfg.dataset.loader
    train_sampler = MixturePairSampler(
        len(train_ds),
        num_samples=loader_cfg.get("num_samples_per_epoch") or None,
        shuffle=bool(loader_cfg.get("shuffle", True)),
        seed=int(cfg.seed),
        rank=dist.rank,
        world_size=dist.world_size,
    )
    train_loader = _build_loader(train_ds, train_sampler, loader_cfg)
    if has_val:
        val_sampler = MixturePairSampler(
            len(val_ds),
            shuffle=False,
            rank=dist.rank,
            world_size=dist.world_size,
        )
        val_loader = _build_loader(val_ds, val_sampler, loader_cfg)

    # ---------------- model ----------------
    h, w = train_ds.lat.size, train_ds.lon.size
    model_kwargs = OmegaConf.to_container(cfg.model.params, resolve=True)
    model = MoWEPrecipGate(
        input_size=(h, w),
        in_channels=train_ds.layout.num_channels,
        n_experts=len(train_ds.experts),
        **model_kwargs,
    ).to(dist.device)
    inner_model = model
    if dist.distributed and dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )

    # ---------------- loss / optimizer ----------------
    region = list(cfg.region.lat) + list(cfg.region.lon)
    box = (region[0], region[1], region[2], region[3])
    # Optional IMD-coverage restriction: with dataset.imd.store set, the
    # training loss (and the monthly validation metrics) only see the
    # gridpoints where the IMD gauge analysis has data.
    imd_mask = None
    imd_cfg = cfg.dataset.get("imd", None)
    if imd_cfg is not None and imd_cfg.get("store"):
        imd_mask = imd_valid_mask(
            str(imd_cfg.store),
            train_ds.lat,
            train_ds.lon,
            min_finite_frac=float(imd_cfg.get("min_finite_frac", 0.99)),
        )
        plog.info(f"IMD-coverage mask: {int(imd_mask.sum())} gridpoints")
    # Space the mixture is formed in. "physical": experts' precip is
    # inverted to mm/day first, so P_hat = sum_i w_i (P_i + b_i) is an
    # ARITHMETIC mean in mm/day and the loss log-transforms it. "log": mix
    # the standardized log channels directly (a weighted GEOMETRIC mean in
    # mm/day, which is structurally dry) -- kept for ablation.
    mix_space = str(cfg.model.get("mix_space", "physical"))
    if mix_space not in ("physical", "log"):
        raise ValueError(f"model.mix_space must be physical|log, got {mix_space!r}")
    plog.info(f"mixture space: {mix_space}")
    loss_fn = build_loss(
        cfg.loss,
        lat=train_ds.lat,
        lon=train_ds.lon,
        box=box,
        precip_mean=train_ds.precip_mean,
        precip_std=train_ds.precip_std,
        precip_transform=train_ds.precip_transform,
        extra_mask=imd_mask,
        pred_space="physical" if mix_space == "physical" else "normalized",
    ).to(dist.device)

    cfg_train = cfg.training
    optimizer = torch.optim.AdamW(
        inner_model.parameters(),
        lr=float(cfg_train.optimizer.lr),
        betas=tuple(cfg_train.optimizer.get("betas", (0.9, 0.999))),
        weight_decay=float(cfg_train.optimizer.get("weight_decay", 0.05)),
    )
    steps_per_epoch = len(train_loader)
    max_epochs = int(cfg_train.max_epochs)
    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=int(cfg_train.get("warmup_epochs", 1)) * steps_per_epoch,
        total_steps=max_epochs * steps_per_epoch,
        min_lr_ratio=float(cfg_train.get("min_lr_ratio", 0.02)),
    )
    amp = str(cfg_train.get("amp", "none"))
    amp_enabled = amp == "bf16" and dist.device.type == "cuda"
    grad_clip = float(cfg_train.get("grad_clip_norm", 0.0) or 0.0)
    dropout_p = float(cfg_train.get("expert_dropout", 0.0))

    # ---------------- validation harness ----------------
    validator = None
    if has_val and cfg.validation.get("enabled", True):
        seeps_clim = SeepsClimatology(
            to_absolute_path(str(cfg.validation.seeps_climatology))
        )
        val_lead_days = tuple(cfg.dataset.val.lead_days)
        # Score exactly where the loss trains: box (x IMD coverage). The gate
        # emits weights globally but is only supervised in this region, so
        # metrics anywhere else would measure untrained extrapolation.
        val_weights = region_weights(
            val_ds.lat, val_ds.lon, box, extra_mask=imd_mask
        )
        plog.info(
            f"validation region: {int((val_weights > 0).sum())} gridpoints"
            + (" (box n IMD coverage)" if imd_mask is not None else " (box)")
        )
        validator = MixtureValidator(
            expert_names=val_ds.expert_names,
            lead_days=(int(val_lead_days[0]), int(val_lead_days[1])),
            region_weights=val_weights,
            seeps_climatology=seeps_clim,
            precip_mean=val_ds.precip_mean,
            precip_std=val_ds.precip_std,
            precip_transform=val_ds.precip_transform,
            device=dist.device,
            monthly=True,
            loss_fn=loss_fn,
            mix_space=mix_space,
        )

    # ---------------- resume ----------------
    ckpt_dir = Path("./checkpoints")
    loaded_epoch = load_checkpoint(
        str(ckpt_dir),
        models=inner_model,
        optimizer=optimizer,
        scheduler=None,
        device=dist.device,
    )
    start_epoch = max(int(cfg.get("start_epoch", 0)), loaded_epoch + 1 if loaded_epoch else 0)
    for _ in range(start_epoch * steps_per_epoch):
        scheduler.step()

    # ---------------- training loop ----------------
    for epoch in range(start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        # LaunchLogger epochs are 1-indexed (iter starts at (epoch-1)*num_mini_batch).
        with LaunchLogger(
            "train", epoch=epoch + 1, num_mini_batch=steps_per_epoch, epoch_alert_freq=1
        ) as log:
            for batch in train_loader:
                x = batch["expert_inputs"].to(dist.device, non_blocking=True)
                mask = batch["expert_mask"].to(dist.device, non_blocking=True)
                target = batch["target"].to(dist.device, non_blocking=True)
                target_mm = batch["target_mm"].to(dist.device, non_blocking=True)
                taus = batch["lead_days"].to(dist.device, non_blocking=True)

                if dropout_p > 0:
                    x, mask = expert_dropout(x, mask, dropout_p)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist.device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    weights, biases = model(x, mask, taus)
                    expert_precip = x[:, :, 0]
                    if mix_space == "physical":
                        expert_precip = denormalize_precip(
                            expert_precip,
                            mean=train_ds.precip_mean,
                            std=train_ds.precip_std,
                            transform=train_ds.precip_transform,
                        )
                    pred = mix(weights, biases, expert_precip)
                    loss = loss_fn(pred.float(), target, target_mm)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        inner_model.parameters(), grad_clip
                    )
                optimizer.step()
                scheduler.step()
                scalars = {"loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                # Composite loss: log the two terms so bias_weight is tunable
                # from the curves rather than by guesswork.
                if getattr(loss_fn, "bias_weight", 0.0) > 0:
                    scalars["mse_term"] = loss_fn.last_mse
                    scalars["bias_mm"] = loss_fn.last_bias_mm
                log.log_minibatch(_ddp_mean_scalars(scalars, dist=dist))

        # ---------------- validation ----------------
        if (
            validator is not None
            and (epoch + 1) % int(cfg.validation.get("every_n_epochs", 1)) == 0
        ):
            with LaunchLogger("valid", epoch=epoch + 1) as vlog:
                metrics, extras = validator.run(model, val_loader)
                vlog.log_epoch(metrics)
                if dist.rank == 0 and extras.get("weight_maps"):
                    import numpy as np

                    np.savez(
                        f"weight_maps_epoch{epoch}.npz", **extras["weight_maps"]
                    )

        # ---------------- checkpoint ----------------
        if dist.distributed and dist.world_size > 1:
            torch.distributed.barrier()
        if (
            dist.rank == 0
            and (epoch + 1) % int(cfg.get("checkpoint_save_interval", 5)) == 0
        ):
            save_checkpoint(
                str(ckpt_dir),
                models=inner_model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
            )

    if dist.rank == 0:
        save_checkpoint(
            str(ckpt_dir),
            models=inner_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=max_epochs - 1,
        )
    plog.info("training complete")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
