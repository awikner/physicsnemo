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
import time
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger
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
from train_loop import make_optimizer, make_scheduler  # noqa: E402
from validate import Deterministic, GaussianIC, ReplicateOnly  # noqa: E402
from validate_diffusion import DiffusionRolloutValidator  # noqa: E402


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


def _stage_window_size(cfg: DictConfig, stage: DictConfig) -> int:
    """Resolve the rolling-window length for a single stage.

    Multi-stage curricula override ``cfg.loss.window_size`` per stage via
    ``stage.loss_overrides.window_size``. Stages without an override
    inherit the base loss config's window_size.
    """
    overrides = stage.get("loss_overrides", None)
    if overrides is not None and "window_size" in overrides:
        return int(overrides.get("window_size") or 0)
    return _window_size_from_loss(cfg)


def _build_loader(
    cfg: DictConfig,
    raw_ds: ClimateZarrDataset,
    *,
    window_size: int,
    rank: int,
) -> tuple[DataLoader, bool]:
    """Build the per-stage DataLoader.

    Returns ``(loader, window_mode)``. ``raw_ds`` is wrapped in a
    :class:`SequenceDataset` when ``window_size > 1``. The
    :class:`LeadTimePairSampler` is only used in single-step mode —
    rolling stages stride through the SequenceDataset uniformly.
    """
    window_mode = window_size > 1
    if window_mode:
        from physicsnemo.experimental.datapipes.climate import SequenceDataset

        dataset = SequenceDataset(
            raw_ds, sequence_length=window_size, lead_time=1
        )
    else:
        dataset = raw_ds

    sampler = (
        LeadTimePairSampler(
            dataset_length=len(dataset),
            forecast_lead_times=list(cfg.dataset.forecast_lead_times),
            shuffle=bool(cfg.dataset.shuffle),
            seed=int(cfg.seed) + rank,
        )
        if not window_mode
        else None
    )

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
    return loader, window_mode


def _make_perturber(kind: str, scales: dict | None):
    """Return a :class:`Perturber` instance from the YAML config name."""
    kind_l = (kind or "deterministic").lower()
    if kind_l == "deterministic":
        return Deterministic()
    if kind_l in ("replicate_only", "replicateonly", "replicate"):
        return ReplicateOnly()
    if kind_l in ("gaussian_ic", "gaussianic", "gaussian"):
        return GaussianIC(scales=dict(scales or {}))
    raise ValueError(f"unknown perturber kind {kind!r}")


def _build_validator(
    cfg: DictConfig,
    raw_ds: ClimateZarrDataset,
    wrapper,
    inference_scheduler,
    *,
    device,
) -> DiffusionRolloutValidator | None:
    """Build the :class:`DiffusionRolloutValidator` from cfg.validation.

    Returns ``None`` if validation is disabled. Reuses the training-time
    :class:`ClimateZarrDataset` (single-frame layout) — the validator
    drives it directly to stride boundaries forward one step at a time.
    """
    val_cfg = cfg.get("validation", None)
    if val_cfg is None:
        return None
    rollout_cfg = val_cfg.get("rollout", None)
    if rollout_cfg is None or not bool(rollout_cfg.get("enabled", False)):
        return None

    sampler_cfg = rollout_cfg.get("sampler", None) or {}
    sampler_num_steps = sampler_cfg.get("num_steps", None)
    if sampler_num_steps is None:
        pass
    elif OmegaConf.is_config(sampler_num_steps) or isinstance(
        sampler_num_steps, (list, tuple)
    ):
        # Per-emitted-frame schedule (Phase 8f, F4) — one int per frame.
        sampler_num_steps = [int(s) for s in sampler_num_steps]
    else:
        sampler_num_steps = int(sampler_num_steps)

    perturber = _make_perturber(
        rollout_cfg.get("perturber", "deterministic"),
        OmegaConf.to_container(rollout_cfg.get("perturber_scales", {}), resolve=True),
    )

    has_diagnostic = (
        cfg.model.get("diagnostic_variables") is not None
        and len(list(cfg.model.diagnostic_variables)) > 0
    )

    # Normalizer for physical-unit RMSE — pull from the dataset's
    # composed transform if available. The dataset transform is
    # ``nan_fill(normalizer(sample))`` so the normalizer lives on the
    # closure of ``raw_ds.transform``; rebuild a standalone normalizer
    # here so the validator can call ``.denormalize_state``.
    from physicsnemo.experimental.datapipes.climate import ClimateNormalizer

    normalizer = ClimateNormalizer.from_dataset(
        raw_ds,
        mean_path=_resolve_path(cfg.dataset.mean_path),
        std_path=_resolve_path(cfg.dataset.std_path),
        normalize_constant_boundary=bool(
            cfg.dataset.get("normalize_constant_boundary", False)
        ),
        normalize_diagnostic=bool(cfg.dataset.get("normalize_diagnostic", False)),
    ).to(device)

    horizon = rollout_cfg.get("horizon", None)
    horizon = int(horizon) if horizon is not None else None

    return DiffusionRolloutValidator(
        raw_ds,
        wrapper=wrapper,
        inference_scheduler=inference_scheduler,
        log_steps=list(rollout_cfg.log_steps),
        device=device,
        horizon=horizon,
        ensemble_size=int(rollout_cfg.get("ensemble_size", 1)),
        perturber=perturber,
        has_diagnostic=has_diagnostic,
        batch_size=int(rollout_cfg.get("batch_size", 1)),
        max_initial_conditions=int(rollout_cfg.get("max_initial_conditions", 4)),
        ic_stride=int(rollout_cfg.get("ic_stride", 1)),
        normalizer=normalizer,
        sampler_num_steps=sampler_num_steps,
        seed=int(cfg.seed),
    )


def _build_scheduler_loss(
    cfg: DictConfig, stage: DictConfig, device
):
    """Instantiate the diffusion scheduler (training loss) for a stage.

    Per-stage knobs come from ``stage.loss_overrides`` and are merged on
    top of ``cfg.loss`` before :func:`hydra.utils.instantiate`. Common
    overrides: ``window_size``, ``num_steps``, ``noise``.
    """
    overrides = stage.get("loss_overrides", None)
    if overrides is None or len(overrides) == 0:
        loss_cfg = cfg.loss
    else:
        loss_cfg = OmegaConf.merge(cfg.loss, overrides)
    return hydra.utils.instantiate(loss_cfg).to(device)


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

    # --- Dataset (raw) ----------------------------------------------------
    # The DataLoader itself is built *per stage* below so that multi-stage
    # curricula (e.g. W=3 pretrain → W=6 finetune) can swap the window
    # size — but the underlying ClimateZarrDataset is shared and built
    # exactly once.
    raw_ds = _build_dataset(cfg)
    cfg_train = cfg.training
    stages = list(cfg_train.stages)
    total_epochs = sum(int(s.num_epochs) for s in stages)
    logger.info(
        f"diffusion train: stages={len(stages)}, total_epochs={total_epochs}, "
        f"world_size={dist.world_size}, device={dist.device}"
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

    # --- Per-batch loss TSV (benchmarking, F3) -----------------------------
    # Mirrors train.py's ``cfg.bench.per_batch_tsv`` wiring — every
    # minibatch's wall-clock time + loss is appended to a TSV for the
    # fp32-vs-bf16 comparison in
    # benchmarks/physicsnemo/experimental/models/amip_si/RESULTS.md.
    # Only rank 0 writes to avoid file contention.
    bench_tsv_file = None
    _bench_start_wall = None
    if cfg.get("bench") and cfg.bench.get("per_batch_tsv"):
        if dist.rank == 0:
            bench_tsv_path = Path(_resolve_path(cfg.bench.per_batch_tsv))
            bench_tsv_path.parent.mkdir(parents=True, exist_ok=True)
            bench_tsv_file = open(bench_tsv_path, "w", buffering=1)  # line-buffered
            bench_tsv_file.write("epoch\tbatch_idx\twall_s\tloss\n")
            logger.info(f"benchmark per-batch TSV → {bench_tsv_path}")
        _bench_start_wall = time.perf_counter()

    # --- Stage loop -------------------------------------------------------
    # Per-stage rebuilds: the DataLoader (window size may change), the
    # scheduler loss (window_size / num_steps may change), and the LR
    # scheduler (cosine length follows the stage). EMA is built once on
    # the first stage so its shadow weights persist across stages.
    # The rollout validator is also stage-scoped (window mode toggles
    # change the inference scheduler family).
    val_every = int(cfg.validation.get("every_n_epochs", 0) or 0) if "validation" in cfg else 0
    global_epoch = start_epoch
    ema: ModelEMA | None = None
    loader: DataLoader | None = None
    window_mode = False
    window_size = 0
    validator: DiffusionRolloutValidator | None = None
    for stage_idx, stage in enumerate(stages):
        stage_epochs = int(stage.num_epochs)
        stage_window_size = _stage_window_size(cfg, stage)

        # (Re)build the DataLoader when the window size changes — and on
        # the first stage where it has to be built from scratch.
        if loader is None or stage_window_size != window_size:
            window_size = stage_window_size
            loader, window_mode = _build_loader(
                cfg, raw_ds, window_size=window_size, rank=dist.rank
            )

        steps_per_epoch = max(1, len(loader))

        # (Re)build the diffusion scheduler with this stage's overrides.
        scheduler_loss = _build_scheduler_loss(cfg, stage, dist.device)

        # The validator shares the (stage-scoped) scheduler instance —
        # the inference sampler num_steps is overridden inside the
        # validator. Rebuilt at the same boundaries as ``scheduler_loss``.
        validator = _build_validator(
            cfg, raw_ds, inner_model, scheduler_loss, device=dist.device
        )
        if validator is not None and dist.rank == 0:
            logger.info(
                f"validation: every_n_epochs={val_every}, "
                f"max_ic={validator.max_initial_conditions}, "
                f"ensemble_size={validator.ensemble_size}, "
                f"log_steps={validator.log_steps}, horizon={validator.horizon}"
            )

        # EMA needs steps_per_epoch for warmup pacing. Build once and
        # carry the shadow state across stages — the first stage that's
        # reached after a resume will hydrate it from the checkpoint.
        if ema is None and bool(cfg_train.ema.enabled):
            ema = ModelEMA(
                inner_model,
                decay=float(cfg_train.ema.decay),
                warmup_epochs=int(cfg_train.ema.warmup_epochs),
                steps_per_epoch=steps_per_epoch,
            )

        sched_cfg = _flatten_scheduler_cfg(
            stage.scheduler,
            lr=float(cfg_train.optimizer.lr),
            steps_per_epoch=steps_per_epoch,
            num_epochs=stage_epochs,
        )
        lr_scheduler = make_scheduler(
            optimizer, sched_cfg, total_steps=steps_per_epoch * stage_epochs
        )
        logger.info(
            f"stage {stage_idx} {stage.name!r} starting at "
            f"global_epoch={global_epoch}: window_mode={window_mode} "
            f"(W={window_size}), steps_per_epoch={steps_per_epoch}, "
            f"sched={type(scheduler_loss).__name__}"
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
                if bench_tsv_file is not None:
                    bench_tsv_file.write(
                        f"{global_epoch}\t{batch_idx}\t"
                        f"{time.perf_counter() - _bench_start_wall:.4f}\t"
                        f"{losses['loss']:.6f}\n"
                    )
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

            # Rollout validation — runs EMA-applied to match inference
            # weights, restores live weights immediately after. Every
            # rank participates so the streaming metric all-reduce works.
            if (
                validator is not None
                and val_every > 0
                and global_epoch % val_every == 0
            ):
                if ema is not None:
                    ema.apply_to(inner_model)
                try:
                    inner_model.eval()
                    metrics = validator.run(model, epoch=global_epoch)
                finally:
                    inner_model.train()
                    if ema is not None:
                        ema.restore(inner_model)
                if dist.rank == 0:
                    summary = " ".join(
                        f"{k}={v:.4e}" for k, v in metrics.items()
                    )
                    logger.info(f"epoch {global_epoch} valid: {summary}")

            global_epoch += 1

    if bench_tsv_file is not None:
        bench_tsv_file.close()
    if dist.device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(dist.device) / 1e9
        logger.info(f"peak GPU memory: {peak_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
