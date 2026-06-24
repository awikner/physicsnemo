# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra entrypoint for AMIP diffusion training (Phase 8c).

Sibling of :mod:`train` — same shared helpers (``build_model``,
``build_datapipe``, ``_flatten_optimizer_cfg``,
``_flatten_scheduler_cfg``, ``_resolve_path``) are imported verbatim
from the deterministic recipe. The diffusion-specific knobs live here:

* The loss is a *scheduler instance* built by
  ``hydra.utils.instantiate(cfg.loss)`` — one of the four classes from
  :mod:`physicsnemo.experimental.diffusion`. The train step calls
  ``scheduler.compute_loss(model, …)`` directly; there's no per-channel
  L1/L2 loss path here.
* The model is a *wrapper* (``AmipDiTWrapper`` / ``RollingDiTWrapper`` /
  ``ERDMWrapper``) that handles the structured-dict ↔ flat-tensor pack
  / unpack.
* ``ClimateZarrDataset`` is opened with ``emit_calendar=True`` so each
  sample includes a ``calendar`` tensor for the model's c_scalar input.
* No :class:`RolloutValidator` — Phase 8c skips rollout validation
  during training (it's too expensive for diffusion). The training loss
  itself is the convergence signal.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.optim import make_optimizer, make_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.climate import (
        ClimateNormalizer,
        ClimateZarrDataset,
        NanFillTransform,
    )
    from physicsnemo.experimental.datapipes.climate.samplers import (
        LeadTimePairSampler,
    )

# Reuse helpers from the deterministic train.py rather than re-implementing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ema import ModelEMA  # noqa: E402
from train import (  # noqa: E402
    _flatten_optimizer_cfg,
    _flatten_scheduler_cfg,
    _maybe_init_wandb,
    _resolve_path,
    build_model,
)


def _resolve_amp_dtype(amp: str | None) -> torch.dtype | None:
    if amp in (None, "none", "off", False):
        return None
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(f"unknown amp value: {amp!r}")


# ---------------------------------------------------------------------------
# Diffusion-specific dataloader build (mirror of build_datapipe but lighter —
# no lead-time pair sampling, calendar emit on, optional window stacking for
# rolling models).
# ---------------------------------------------------------------------------


def _build_dataset(cfg: DictConfig) -> ClimateZarrDataset:
    data = cfg.dataset
    ds = ClimateZarrDataset(
        _resolve_path(data.zarr_path),
        boundary_zarr_path=_resolve_path(data.get("boundary_zarr_path")),
        yearly_repeating_boundary=bool(data.get("yearly_repeating_boundary", False)),
        leap_boundary_zarr_path=_resolve_path(data.get("leap_boundary_zarr_path")),
        non_leap_boundary_zarr_path=_resolve_path(
            data.get("non_leap_boundary_zarr_path")
        ),
        emit_calendar=True,
    )
    normalizer = ClimateNormalizer.from_dataset(
        ds,
        mean_path=_resolve_path(data.mean_path),
        std_path=_resolve_path(data.std_path),
        normalize_constant_boundary=bool(
            data.get("normalize_constant_boundary", False)
        ),
        normalize_diagnostic=bool(data.get("normalize_diagnostic", False)),
    )
    nan_fill = NanFillTransform(
        constant_boundary_variables=list(cfg.model.constant_boundary_variables),
        varying_boundary_variables=list(cfg.model.varying_boundary_variables),
        fill_values=dict(OmegaConf.to_container(data.nan_fill_values, resolve=True) or {}),
        default=float(data.nan_fill_default),
    )
    # Compose normalizer → nan_fill as the dataset transform.
    def _compose(sample):
        return nan_fill(normalizer(sample))

    ds.transform = _compose
    return ds


def _window_size_from_loss(cfg: DictConfig) -> int:
    """Pull the rolling-window length from the scheduler config (if any)."""
    return int(cfg.loss.get("window_size", 0) or 0)


# ---------------------------------------------------------------------------
# Per-batch packing helpers — turn the DataLoader's sample dict into the
# flat tensors the scheduler.compute_loss expects.
# ---------------------------------------------------------------------------


def _pack_single_step(model: nn.Module, sample: dict) -> tuple:
    """SI / SI_X: pack (x, y, c_grid, c_scalar) from a 1-step pair sample."""
    inner = model.module if hasattr(model, "module") else model
    x = inner.pack_state({
        "surface_in": sample["surface_in"],
        "upper_air_in": sample["upper_air_in"],
        "diagnostic": sample.get("diagnostic"),
    })
    y = inner.pack_state({
        "surface_in": sample["target_surface"],
        "upper_air_in": sample["target_upper_air"],
        "diagnostic": sample.get("diagnostic"),  # diagnostic in dataset is target-frame already
    })
    c_grid = inner.pack_c_grid({
        "surface_in": sample["surface_in"],
        "constant_boundary": sample["constant_boundary"],
        "varying_boundary": sample["varying_boundary"],
    })
    c_scalar = sample["calendar"]
    return x, y, c_grid, c_scalar


def _pack_window(model: nn.Module, window: dict) -> tuple:
    """ERDM / RFM: pack (y, c_grid, c_scalar) from a (B, W, …) window sample."""
    inner = model.module if hasattr(model, "module") else model
    y = inner.pack_window_state({
        "surface_in": window["surface_in"],
        "upper_air_in": window["upper_air_in"],
        "diagnostic": window.get("diagnostic"),
    })
    c_grid = inner.pack_window_c_grid({
        "surface_in": window["surface_in"],
        "constant_boundary": window["constant_boundary"],
        "varying_boundary": window["varying_boundary"],
    })
    c_scalar = window["calendar"]
    return y, c_grid, c_scalar


# ---------------------------------------------------------------------------
# Per-batch train step.
# ---------------------------------------------------------------------------


def _train_step(
    *,
    model: nn.Module,
    scheduler_loss,
    sample: dict,
    optimizer: torch.optim.Optimizer,
    grad_scaler,
    amp_dtype,
    device,
    window_mode: bool,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    sample = {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
    }

    with torch.autocast(
        device_type=device.type,
        enabled=amp_dtype is not None,
        dtype=amp_dtype or torch.float32,
    ):
        if window_mode:
            y, c_grid, c_scalar = _pack_window(model, sample)
            loss = scheduler_loss.compute_loss(model, c_grid, c_scalar, y)
        else:
            x, y, c_grid, c_scalar = _pack_single_step(model, sample)
            loss = scheduler_loss.compute_loss(model, x, c_grid, c_scalar, y)

    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {"loss": float(loss.detach().cpu())}


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("amip_diffusion_train")

    if dist.rank == 0:
        _maybe_init_wandb(cfg, dist=dist)
        LaunchLogger.initialize(use_wandb=bool(cfg.wandb.enabled))

    torch.manual_seed(int(cfg.seed) + dist.rank)

    # --- Dataset + DataLoader ---------------------------------------------
    raw_ds = _build_dataset(cfg)

    window_size = _window_size_from_loss(cfg)
    window_mode = window_size > 1
    if window_mode:
        # Rolling models: stack W consecutive frames per sample.
        from physicsnemo.experimental.datapipes.climate import SequenceDataset

        dataset = SequenceDataset(
            raw_ds, sequence_length=window_size, lead_time=1
        )
    else:
        dataset = raw_ds

    sampler = LeadTimePairSampler(
        n_time=len(dataset),
        lead_times=list(cfg.dataset.forecast_lead_times),
        shuffle=bool(cfg.dataset.shuffle),
        seed=int(cfg.seed) + dist.rank,
    ) if not window_mode else None

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        sampler=sampler,
        shuffle=(sampler is None and bool(cfg.dataset.shuffle)),
        prefetch_factor=int(cfg.dataset.prefetch_factor),
        persistent_workers=bool(cfg.dataset.persistent_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
    )

    steps_per_epoch = max(1, len(loader))
    cfg_train = cfg.training
    stages = list(cfg_train.stages)
    total_epochs = sum(int(s.num_epochs) for s in stages)
    logger.info(
        f"diffusion train: steps_per_epoch={steps_per_epoch}, stages={len(stages)}, "
        f"total_epochs={total_epochs}, world_size={dist.world_size}, "
        f"device={dist.device}, window_mode={window_mode} (W={window_size})"
    )

    # --- Model + DDP + Loss + Optimizer ----------------------------------
    model = build_model(cfg.model).to(dist.device)
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            output_device=dist.device if dist.device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
        )
    inner_model = model.module if hasattr(model, "module") else model

    scheduler_loss = hydra.utils.instantiate(cfg.loss).to(dist.device)
    optimizer = make_optimizer(
        inner_model, _flatten_optimizer_cfg(cfg_train.optimizer)
    )

    # --- Mixed precision --------------------------------------------------
    amp_dtype = _resolve_amp_dtype(cfg_train.get("amp", None))
    grad_scaler = None
    if amp_dtype == torch.float16 and dist.device.type == "cuda":
        grad_scaler = torch.amp.GradScaler(device="cuda")
        logger.info("AMP enabled with fp16 + GradScaler")
    elif amp_dtype == torch.bfloat16:
        logger.info("AMP enabled with bf16 (no GradScaler)")
    elif amp_dtype is None:
        logger.info(f"AMP disabled (cfg.training.amp={cfg_train.get('amp', None)})")

    # --- EMA --------------------------------------------------------------
    ema = None
    if bool(cfg_train.ema.enabled):
        ema = ModelEMA(
            inner_model,
            decay=float(cfg_train.ema.decay),
            warmup_epochs=int(cfg_train.ema.warmup_epochs),
            steps_per_epoch=steps_per_epoch,
        )

    # --- Checkpoint resume (mirrors train.py) -----------------------------
    ckpt_dir = _resolve_path(cfg.get("checkpoint_dir", "checkpoints"))
    start_epoch = int(cfg.start_epoch)
    start_epoch = max(
        start_epoch,
        load_checkpoint(
            ckpt_dir,
            models=inner_model,
            optimizer=optimizer,
            device=dist.device,
        )
        + 1,
    )

    # --- Stage loop -------------------------------------------------------
    global_epoch = start_epoch
    for stage_idx, stage in enumerate(stages):
        stage_epochs = int(stage.num_epochs)
        sched_cfg = _flatten_scheduler_cfg(
            stage.scheduler,
            lr=float(cfg_train.optimizer.lr),
            steps_per_epoch=steps_per_epoch,
            num_epochs=stage_epochs,
        )
        lr_scheduler = make_scheduler(optimizer, sched_cfg)
        logger.info(
            f"stage {stage_idx} {stage.name!r} starting at global_epoch={global_epoch}"
        )

        for _ in range(stage_epochs):
            for batch_idx, sample in enumerate(loader):
                losses = _train_step(
                    model=model,
                    scheduler_loss=scheduler_loss,
                    sample=sample,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    amp_dtype=amp_dtype,
                    device=dist.device,
                    window_mode=window_mode,
                )
                if ema is not None:
                    ema.update(inner_model, epoch=global_epoch)
                lr_scheduler.step()
                if (
                    dist.rank == 0
                    and (batch_idx % int(cfg.log_every_n_steps) == 0)
                ):
                    logger.info(
                        f"epoch {global_epoch} batch {batch_idx}/{steps_per_epoch} "
                        f"loss={losses['loss']:.4e}"
                    )

            if dist.rank == 0:
                save_checkpoint(
                    ckpt_dir,
                    models=inner_model,
                    optimizer=optimizer,
                    epoch=global_epoch,
                    metadata={"ema": ema.state_dict() if ema is not None else None},
                )
            global_epoch += 1


if __name__ == "__main__":
    main()
