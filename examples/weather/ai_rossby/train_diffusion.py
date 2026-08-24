# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
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

import logging as _logging
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
        ClimateZarrMultiYearDataset,
        NanFillTransform,
    )
    from physicsnemo.experimental.datapipes.climate.samplers import (
        LeadTimePairSampler,
    )

# Reuse helpers from the deterministic train.py rather than re-implementing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_setup import (  # noqa: E402
    build_forcing_pipeline,
    model_varying_pre_rescaler,
    resolve_varying_subset,
)
from ema import ModelEMA  # noqa: E402
from train import (  # noqa: E402
    _flatten_optimizer_cfg,
    _flatten_scheduler_cfg,
    _maybe_init_wandb,
    _resolve_path,
    build_model,
)
from train_loop import (  # noqa: E402
    GnormLrGovernor,
    RewindBuffer,
    adopt_ocean_contract,
    apply_math_precision,
    assert_checkpoint_dir_contract,
    choose_worker_start_method,
    lead_times_for_sampler,
    load_partial_weights,
    make_optimizer,
    make_scheduler,
    model_step_rows,
    repair_incomplete_slurm_env,
)
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
    zarr_path = _resolve_path(data.zarr_path)
    _ds_kwargs = dict(
        boundary_zarr_path=_resolve_path(data.get("boundary_zarr_path")),
        yearly_repeating_boundary=bool(data.get("yearly_repeating_boundary", False)),
        leap_boundary_zarr_path=_resolve_path(data.get("leap_boundary_zarr_path")),
        non_leap_boundary_zarr_path=_resolve_path(
            data.get("non_leap_boundary_zarr_path")
        ),
        emit_calendar=True,
    )
    # A DIRECTORY of per-year sub-stores is a multi-year archive; a single
    # ``.zarr`` is one year. Same routing ``train.py`` (line ~594) and
    # ``inference.py`` have had since the multi-year port — the diffusion recipe
    # simply never got it, so until 2026-08-17 it could only ever train on ONE
    # year while the upstream SI/ERDM runs trained 1979-2015.
    #
    # Nothing else here changes: ClimateZarrMultiYearDataset exposes the same
    # ``transform`` / ``layout`` / ``n_time`` surface, dispatches ``(start, lead)``
    # by global index across year boundaries, and SequenceDataset's per-frame
    # ``base[(t, 1)]`` reads inherit that, so rolling windows span years too.
    _p = Path(zarr_path)
    if _p.is_dir() and not str(zarr_path).endswith(".zarr"):
        ds = ClimateZarrMultiYearDataset(zarr_path, **_ds_kwargs)
        _logging.getLogger(__name__).info(
            "multi-year archive %s: %d sub-store(s), %d rows",
            zarr_path, len(getattr(ds, "sub_datasets", []) or []), ds.n_time,
        )
    else:
        ds = ClimateZarrDataset(zarr_path, **_ds_kwargs)
    # The store may serve MORE varying-boundary channels than the model consumes
    # (2026-08-14): the real v1 SI checkpoint lists 3 where amip_dailyavg_coarse
    # has 4, because upstream's run never fed global_mean_co2. When that happens
    # the stream is sliced to the model's list before the fill, so the
    # normalizer has to be aligned to the same list — otherwise it would apply
    # store-ordered statistics to a sliced tensor, which changes no shape.
    store_varying = [
        str(v)
        for v in getattr(getattr(ds, "layout", None), "varying_boundary_variables", [])
    ]
    subset = resolve_varying_subset(cfg, store_varying)
    normalizer_kwargs = {}
    if subset is not None:
        # PRE-rescaler, like the slice above: the normalizer sits between the
        # subset and the assembler, so it never sees the derived SST-anomaly
        # channel — and has no stored stats for it either. SSTRescaler then
        # reads the SST mean/std off this same list.
        normalizer_kwargs["varying_boundary_variables"] = model_varying_pre_rescaler(cfg)
    normalizer = ClimateNormalizer.from_dataset(
        ds,
        mean_path=_resolve_path(data.mean_path),
        std_path=_resolve_path(data.std_path),
        normalize_constant_boundary=bool(
            data.get("normalize_constant_boundary", False)
        ),
        normalize_diagnostic=bool(data.get("normalize_diagnostic", False)),
        **normalizer_kwargs,
    )
    # Phase 12d.13: one construction site for fill → normalize → scalar
    # routing. NOTE this replaces a ``nan_fill(normalizer(sample))`` compose,
    # which substituted PHYSICAL-unit fill values (SST 270 K) into an
    # already-z-scored tensor; dataset_setup pins the documented order.
    pipeline = build_forcing_pipeline(
        cfg, normalizer=normalizer, store_varying_variables=store_varying
    )
    ds.transform = pipeline.dataset_transform
    ds.forcing_pipeline = pipeline
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


def _init_worker_threads(worker_id: int) -> None:  # noqa: ARG001
    """Pin each DataLoader worker to a single torch thread.

    See the note at the ``worker_init_fn`` call site: the boundary smoothing's
    OpenMP-parallel ``conv2d`` deadlocks in a forked worker when the parent's
    OpenMP runtime was already initialized with more than one thread.
    """
    torch.set_num_threads(1)


def _build_loader(
    cfg: DictConfig,
    raw_ds: ClimateZarrDataset,
    *,
    window_size: int,
    rank: int,
    world_size: int = 1,
    forcing_lag: int = 0,
    emit_boundary_next: bool = False,
    anchor_frames: int = 0,
    step_stride: int = 1,
) -> tuple[DataLoader, bool]:
    """Build the per-stage DataLoader.

    Returns ``(loader, window_mode)``. ``raw_ds`` is wrapped in a
    :class:`SequenceDataset` when ``window_size > 1`` (a window of W
    frames = ``unroll_steps = W - 1``). The
    :class:`LeadTimePairSampler` is only used in single-step mode —
    rolling stages stride through the SequenceDataset with a plain
    :class:`~torch.utils.data.DistributedSampler` under DDP so ranks
    see disjoint windows.

    ``forcing_lag`` / ``emit_boundary_next`` (Phase 12f) come from the
    model's channel contract, not from a config knob — see
    ``RollingDiTWrapper.forcing_lag``. ``anchor_frames`` comes from the
    *scheduler* instead (``RSIScheduler.anchor_frames``): a data-coupled
    rolling scheduler anchors slot 1 on the frame before the window, so it
    needs that frame emitted. It costs no extra read at ``forcing_lag >= 1``. ``step_stride`` is the model step in
    store rows (``resolve_step_stride``); the rolling window advances one model
    step per frame, which on the 6-hourly AMIP archives is 4 rows, not 1.
    """
    window_mode = window_size > 1
    if window_mode:
        from physicsnemo.experimental.datapipes.climate import SequenceDataset

        dataset = SequenceDataset(
            raw_ds,
            unroll_steps=window_size - 1,
            forcing_lag=forcing_lag,
            emit_boundary_next=emit_boundary_next,
            emit_anchor=bool(anchor_frames),
            step_stride=step_stride,
        )
    else:
        dataset = raw_ds

    if not window_mode:
        sampler = LeadTimePairSampler(
            dataset_length=len(dataset),
            forecast_lead_times=lead_times_for_sampler(cfg, step_stride),
            shuffle=bool(cfg.dataset.shuffle),
            seed=int(cfg.seed) + rank,
        )
    elif world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(cfg.dataset.shuffle),
            seed=int(cfg.seed),
        )
    else:
        sampler = None

    num_workers = int(cfg.dataset.num_workers)
    worker_kwargs = (
        {
            "prefetch_factor": int(cfg.dataset.prefetch_factor),
            "persistent_workers": bool(cfg.dataset.persistent_workers),
            # One torch thread per worker. Two reasons, the first fatal
            # (diagnosed 2026-08-18 on GH200):
            #
            # DEADLOCK. The boundary NaN fill runs `smooth_masked_boundary`, ten
            # iterations of F.conv2d on CPU, inside the worker. conv2d is
            # OpenMP-parallel, and a forked child that inherits an
            # already-initialized OpenMP runtime hangs on its first parallel
            # region. Measured: with OMP_NUM_THREADS=8 and 4 workers, training
            # produced ZERO batches in 240 s, twice, with py-spy showing every
            # worker stopped at the same conv2d line 30 s apart -- while
            # num_workers=0 did the identical work at ~1 s/batch. The shipped HPC
            # scripts export OMP_NUM_THREADS=1 and therefore never hit it, which
            # is exactly why it stayed hidden.
            #
            # OVERSUBSCRIPTION. Even when it does not deadlock, 4 workers x 8
            # threads on a 16-CPU allocation is 32 threads contending.
            "worker_init_fn": _init_worker_threads,
        }
        if num_workers > 0
        else {}
    )
    # Start method for the workers. `fork` is fastest but inherits the parent's
    # OpenMP runtime and zarr v3's asyncio event-loop thread, neither of which
    # survives a fork: with OMP_NUM_THREADS>1 the workers deadlock in the boundary
    # smoothing's conv2d and training produces zero batches. Chosen rather than
    # assumed -- see train_loop.choose_worker_start_method for the measurements.
    mp_context = choose_worker_start_method(
        num_workers,
        cfg.dataset.get("multiprocessing_context", None),
        log=_logging.getLogger(__name__),
    )
    if mp_context:
        worker_kwargs["multiprocessing_context"] = mp_context
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataset.batch_size),
        num_workers=num_workers,
        sampler=sampler,
        shuffle=(sampler is None and bool(cfg.dataset.shuffle)),
        pin_memory=bool(cfg.dataset.pin_memory),
        **worker_kwargs,
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
    step_size: int = 1,
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
        # The model step, in store rows — the same number the training loader
        # strides by, so validation rolls the model at its trained timestep.
        step_size=step_size,
        normalizer=normalizer,
        sampler_num_steps=sampler_num_steps,
        seed=int(cfg.seed),
    )


def _build_scheduler_loss(
    cfg: DictConfig, stage: DictConfig, device, model: nn.Module | None = None
):
    """Instantiate the diffusion scheduler (training loss) for a stage.

    Per-stage knobs come from ``stage.loss_overrides`` and are merged on
    top of ``cfg.loss`` before :func:`hydra.utils.instantiate`. Common
    overrides: ``window_size``, ``num_steps``, ``noise``.

    Phase 12f: ``nocean`` / ``ocean_grid_indices`` are *injected from the
    model* rather than configured. They are two halves of one contract — the
    tail width of the state axis and where in ``c_grid`` its truth lives — and
    the model already derives both from ``ocean_state_variables``. A config
    that restated them could disagree with the pack, and would do so silently.
    """
    overrides = stage.get("loss_overrides", None)
    if overrides is None or len(overrides) == 0:
        loss_cfg = cfg.loss
    else:
        loss_cfg = OmegaConf.merge(cfg.loss, overrides)

    sched = hydra.utils.instantiate(loss_cfg).to(device)
    if model is not None:
        adopt_ocean_contract(sched, model)
    return sched


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


def _pack_window(model: nn.Module, window: dict, *, anchor_frames: int = 0) -> tuple:
    """ERDM / RFM / RSI: pack (y, c_grid, c_scalar) from a (B, W, …) window sample.

    :class:`SequenceDataset` emits the per-frame fields stacked under
    ``{key}_seq`` names (plus an unstacked ``constant_boundary``); the
    constant boundary arrives batched as ``(B, C, H, W)`` and is expanded
    across the window axis here so the wrapper's ``pack_window_c_grid``
    sees ``(B, W, C, H, W)`` streams throughout.

    With ``anchor_frames=1`` (Rolling Stochastic Interpolants) the state stack
    ``y`` is returned with ``W+1`` frames: the loader's ``{key}_prev`` anchor
    frame packed and prepended, so ``y[:, w]`` is slot ``w``'s target and
    ``y[:, w-1]`` its interpolant anchor. ``c_grid`` / ``c_scalar`` stay at
    ``W`` frames aligned to slots 1..W — the anchor is a state, not a slot, and
    is never itself predicted.
    """
    inner = model.module if hasattr(model, "module") else model
    surface_seq = window["surface_in_seq"]
    W = surface_seq.shape[1]
    const = window["constant_boundary"]
    if const.dim() == 4:  # (B, C, H, W) -> (B, W, C, H, W)
        const = const.unsqueeze(1).expand(-1, W, -1, -1, -1)
    y = inner.pack_window_state({
        "surface_in": surface_seq,
        "upper_air_in": window["upper_air_in_seq"],
        "diagnostic": window.get("diagnostic_seq"),
    })
    c_grid = inner.pack_window_c_grid({
        "surface_in": surface_seq,
        "constant_boundary": const,
        "varying_boundary": window["varying_boundary_seq"],
    })
    c_scalar = window["calendar_seq"]
    if anchor_frames:
        if "surface_in_prev" not in window:
            raise KeyError(
                "the scheduler asks for an anchor frame (anchor_frames="
                f"{anchor_frames}) but the loader emitted no 'surface_in_prev'; "
                "build the SequenceDataset with emit_anchor=True"
            )
        y0 = inner.pack_window_state({
            "surface_in": window["surface_in_prev"].unsqueeze(1),
            "upper_air_in": window["upper_air_in_prev"].unsqueeze(1),
            "diagnostic": (
                window["diagnostic_prev"].unsqueeze(1)
                if "diagnostic_prev" in window else None
            ),
        })
        y = torch.cat([y0, y], dim=1)
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
    anchor_frames: int = 0,
    grad_clip_norm: float = 0.0,
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
        ocean_loss = None
        if window_mode:
            y, c_grid, c_scalar = _pack_window(
                model, sample, anchor_frames=anchor_frames)
            if getattr(scheduler_loss, "nocean", 0):
                # The ocean target is the boundary at each state frame's OWN
                # time — a different slice of the loader's read window than the
                # forcing that conditions it. Both slices have the same shape,
                # so passing the forcing window here would silently train an
                # identity copy; the loader emits the shifted view explicitly.
                bnd_next = sample.get("varying_boundary_next_seq")
                if bnd_next is None:
                    raise KeyError(
                        "the scheduler predicts ocean channels but the loader "
                        "emitted no 'varying_boundary_next_seq'; build the "
                        "SequenceDataset with emit_boundary_next=True"
                    )
                if anchor_frames:
                    # y carries the anchor frame, so the ocean target stack
                    # must too: the anchor's own-time boundary is the slot-1
                    # CONDITIONING frame (forcing_lag=1), which the loader
                    # already emits as varying_boundary_seq[:, 0].
                    bnd_next = torch.cat(
                        [sample["varying_boundary_seq"][:, :1], bnd_next], dim=1
                    )
                y = scheduler_loss.append_ocean_target(y, bnd_next)
            loss, ocean_loss = scheduler_loss.compute_loss(
                model, c_grid, c_scalar, y, return_parts=True
            )
        else:
            x, y, c_grid, c_scalar = _pack_single_step(model, sample)
            loss = scheduler_loss.compute_loss(model, x, c_grid, c_scalar, y)

    # A non-finite loss must stop the run, not be stepped on. Without this the
    # 2026-08-21 A2 run ground out 39,314 NaN batches over 8.5 h on a dedicated
    # 4xH100 node before anyone noticed — the optimizer happily propagates NaN
    # into every weight, so nothing after the first one is recoverable.
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"non-finite training loss ({loss.detach().float().item()}). "
            f"Refusing to step: one NaN update poisons every parameter and the "
            f"run cannot recover. Resume from the last finite checkpoint, and "
            f"see training.grad_clip_norm."
        )

    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()

    # Gradient clipping. This path had NONE — train.py clips, train_diffusion
    # did not, and `amip_diffusion*.yaml` shipped a `grad_clip_norm` key that
    # nothing in this recipe read, so it looked configured while being a no-op.
    # Measured consequence (2026-08-21): RSI trained cleanly for 11,700 batches
    # (loss 1.8e5 -> 1.3e3) and then ran away exponentially, e-folding every ~93
    # batches, with nothing to arrest it.
    grad_norm = None
    if grad_clip_norm and grad_clip_norm > 0:
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)      # clip real grads, not scaled ones
        inner = model.module if hasattr(model, "module") else model
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(inner.parameters(), grad_clip_norm)
        )

    if grad_scaler is not None:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()

    out = {"loss": float(loss.detach().cpu())}
    if grad_norm is not None:
        # The PRE-clip norm, so a run that is quietly riding the clip ceiling is
        # visible rather than looking healthy.
        out["grad_norm"] = grad_norm
    if getattr(scheduler_loss, "nocean", 0) and ocean_loss is not None:
        # Logged separately because it is ~1-2% of a channel-summed loss and
        # collapses fast (the target is recoverable from a forcing in the same
        # forward pass) — folded in, "learned it" and "weighted too low to
        # matter" look identical.
        out["loss_ocean"] = float(ocean_loss.detach().cpu())
    return out


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Before the manager looks at the environment: an sbatch shell without srun
    # leaves SLURM's launch IP unset, which sends initialize() down a path that
    # dies on MASTER_ADDR=None. See train_loop.repair_incomplete_slurm_env.
    repair_incomplete_slurm_env()
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("amip_diffusion_train")

    # Create the wandb run first, then bind it to LaunchLogger so all
    # training + validation log_epoch / log_minibatch dicts route to
    # wandb. _maybe_init_wandb returns False (and LaunchLogger stays
    # console-only) if wandb is disabled or the package is missing.
    #
    # Called on EVERY rank — _maybe_init_wandb's contract (thread-jitter
    # symmetry under DDP; rank 0 alone drives LaunchLogger's wandb
    # backend). Pre-Phase-12b this call was gated on ``dist.rank == 0``,
    # which is precisely the asymmetric configuration the every-rank
    # strategy exists to prevent — the W2 primary suspect for the
    # DDP-init NCCL watchdog hang (docs/dev/wandb_ddp_hang_fix_plan.md).
    #
    # ``wandb.init_after_ddp=true`` (W2 cell M4) defers the init until
    # after the DDP wrap so the first NCCL collective completes before
    # any wandb background thread exists.
    _wandb_after_ddp = bool(
        cfg.wandb.get("init_after_ddp", False)
    ) if cfg.get("wandb", None) is not None else False
    if not _wandb_after_ddp:
        wandb_active = _maybe_init_wandb(cfg, dist=dist)
        LaunchLogger.initialize(use_wandb=wandb_active)

    torch.manual_seed(int(cfg.seed) + dist.rank)

    # --- Dataset (raw) ----------------------------------------------------
    # The DataLoader itself is built *per stage* below so that multi-stage
    # curricula (e.g. W=3 pretrain → W=6 finetune) can swap the window
    # size — but the underlying ClimateZarrDataset is shared and built
    # exactly once.
    raw_ds = _build_dataset(cfg)
    cfg_train = cfg.training
    # Opt-in float32 math knobs, logged either way so a run records the settings
    # its throughput was produced under. Off by default: enabling them changes a
    # trained model's numerics. ++training.matmul_precision=high measured 1.80x on
    # the v2 x_DDC config (A100), and is what upstream amip_v2 already trains with.
    apply_math_precision(cfg_train, log=logger)
    # Platform escape hatch, sitting with the other float-math knobs: DeltaAI's
    # cuDNN 9.20 has no working attention plan at the v2 geometries under bf16,
    # and torch's backend priority reaches for cuDNN first. A no-op unless
    # AI_ROSSBY_NO_CUDNN_SDPA is set — see the function's docstring for the
    # measured backend table.
    from physicsnemo.experimental.models.amip_si import maybe_disable_cudnn_sdp

    maybe_disable_cudnn_sdp()
    stages = list(cfg_train.stages)
    total_epochs = sum(int(s.num_epochs) for s in stages)
    logger.info(
        f"diffusion train: stages={len(stages)}, total_epochs={total_epochs}, "
        f"world_size={dist.world_size}, device={dist.device}"
    )

    # --- Model + DDP + Loss + Optimizer ----------------------------------
    model = build_model(cfg.model).to(dist.device)
    # Anti-fork guard (Phase 12d.13): the model must be sized for exactly the
    # c_grid / c_scalar widths the data pipeline emits.
    if getattr(raw_ds, "forcing_pipeline", None) is not None:
        raw_ds.forcing_pipeline.assert_matches(model, name="cfg.model")
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            output_device=dist.device if dist.device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
        )
    if _wandb_after_ddp:
        # W2 cell M4 ordering experiment: the DDP wrap (and its first
        # NCCL collective) completed above with no wandb threads alive.
        wandb_active = _maybe_init_wandb(cfg, dist=dist)
        LaunchLogger.initialize(use_wandb=wandb_active)
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
    # Resuming your own run is the case where the config is "obviously" the same
    # one — which is exactly why relaunching a run_name with an edited
    # channel_layout (or a reordered variable list) is easy to do and impossible
    # to see: the shapes still match, so the resume loads and training continues
    # against differently-packed weights. Same guard as inference/eval, applied
    # on every rank so a mismatch aborts the job rather than deadlocking it.
    # A no-op for a fresh run: there is no .mdlus in the directory yet.
    assert_checkpoint_dir_contract(inner_model, ckpt_dir, log=logger)
    resumed_epoch = load_checkpoint(
        ckpt_dir,
        models=inner_model,
        optimizer=optimizer,
        device=dist.device,
    )
    start_epoch = max(start_epoch, resumed_epoch + 1)

    # --- Partial-checkpoint warm start (Phase 12f) -------------------------
    # ``training.partial_checkpoint`` warm-starts from a DIFFERENT run's
    # weights — the ocean-variant use case: start from a trained no-ocean
    # ERDM and let only the added ocean parameters begin at init. Applied
    # after the resume check and only when the resume found nothing, so this
    # run's own checkpoints always win: otherwise every restart of a
    # long-running job would silently rewind to the warm-start weights.
    partial_ckpt = cfg_train.get("partial_checkpoint", None)
    if partial_ckpt:
        if resumed_epoch > 0:
            logger.info(
                f"training.partial_checkpoint ignored — resumed this run's own "
                f"checkpoint at epoch {resumed_epoch}"
            )
        else:
            load_partial_weights(
                inner_model, _resolve_path(partial_ckpt), log=logger
            )

    # --- Model step (in store rows) ---------------------------------------
    # ``cfg.model.timedelta_hours`` is authoritative (the step is a model-family
    # property — the PLASIM archive feeds a 24-hour Pangu and a 6-hour SFNO from
    # the same rows); the dataset's row-level ``forecast_lead_times`` is
    # cross-checked against it and a disagreement raises. See
    # ``train_loop.model_step_rows``.
    step_stride = model_step_rows(cfg, raw_ds)
    logger.info(
        f"model step: {step_stride} store row(s) "
        f"({step_stride * int(getattr(raw_ds.layout, 'data_timedelta_hours', 0) or 0)} h)"
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
    anchor_frames = 0
    validator: DiffusionRolloutValidator | None = None
    prior_stage_epochs = 0
    for stage_idx, stage in enumerate(stages):
        stage_epochs = int(stage.num_epochs)
        # Resume bookkeeping. Two pre-existing resume flaws, both harmless
        # under the flat 50-epoch cosine and both live once the schedule
        # actually decays (CosineToFloor): (a) the epoch loop always ran the
        # FULL stage_epochs after a resume, overshooting the total; (b) the
        # per-stage LR scheduler restarted from step 0, snapping a decayed lr
        # back to its peak mid-training — on the RSI objective that is an
        # excursion invitation. ``done_in_stage`` is 0 on a fresh run, so
        # fresh behavior is bit-identical.
        done_in_stage = min(
            stage_epochs, max(0, global_epoch - 1 - prior_stage_epochs)
        )
        prior_stage_epochs += stage_epochs
        if done_in_stage >= stage_epochs:
            continue
        stage_window_size = _stage_window_size(cfg, stage)

        # (Re)build the diffusion scheduler with this stage's overrides FIRST:
        # whether the loader has to emit the pre-window anchor frame is a
        # property of the loss family (RSIScheduler.anchor_frames), so the
        # loader below cannot be built until the scheduler exists. Nothing in
        # _build_scheduler_loss depends on the loader.
        scheduler_loss = _build_scheduler_loss(
            cfg, stage, dist.device, model=inner_model
        )
        stage_anchor_frames = int(getattr(scheduler_loss, "anchor_frames", 0) or 0)

        # (Re)build the DataLoader when the window size or the anchor contract
        # changes — and on the first stage where it has to be built from scratch.
        if (loader is None or stage_window_size != window_size
                or stage_anchor_frames != anchor_frames):
            window_size = stage_window_size
            anchor_frames = stage_anchor_frames
            loader, window_mode = _build_loader(
                cfg,
                raw_ds,
                window_size=window_size,
                rank=dist.rank,
                world_size=dist.world_size,
                step_stride=step_stride,
                # Phase 12f: both come from the model's channel contract, so
                # the loader cannot be aligned differently from the pack.
                forcing_lag=int(getattr(inner_model, "forcing_lag", 0) or 0),
                emit_boundary_next=bool(
                    getattr(inner_model, "num_ocean", 0) or 0
                ),
                # …and this one from the scheduler's.
                anchor_frames=anchor_frames,
            )

        steps_per_epoch = max(1, len(loader))
        if anchor_frames and not int(getattr(inner_model, "forcing_lag", 0) or 0):
            raise ValueError(
                f"{type(scheduler_loss).__name__} needs the pre-window anchor "
                f"frame (anchor_frames={anchor_frames}), which only exists when "
                f"the forcings lag the state. This model's channel_layout gives "
                f"forcing_lag=0 — use a v1/v2 layout."
            )

        # The validator shares the (stage-scoped) scheduler instance —
        # the inference sampler num_steps is overridden inside the
        # validator. Rebuilt at the same boundaries as ``scheduler_loss``.
        validator = _build_validator(
            cfg, raw_ds, inner_model, scheduler_loss,
            device=dist.device, step_size=step_stride,
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
        if done_in_stage > 0:
            # Fast-forward to the resume point so the schedule continues
            # where it left off instead of restarting at peak lr.
            for _ in range(done_in_stage * steps_per_epoch):
                lr_scheduler.step()
            logger.info(
                f"resume: lr scheduler fast-forwarded "
                f"{done_in_stage * steps_per_epoch} steps "
                f"(lr now {optimizer.param_groups[0]['lr']:.3e})"
            )
        # Optional gnorm-triggered lr ratchet (off unless training.lr_drop.
        # enabled) — the RSI A2 floor destabilizes at ANY constant lr, so the
        # governor steps lr down permanently whenever the gradient norm leaves
        # its healthy band. Needs grad_clip_norm > 0 (that is what measures
        # the pre-clip gnorm it feeds on). See GnormLrGovernor.
        lr_governor = None
        lr_rewinder = None
        _drop_cfg = cfg_train.get("lr_drop", None)
        if _drop_cfg is not None and bool(_drop_cfg.get("enabled", False)):
            if not float(cfg_train.get("grad_clip_norm", 0.0) or 0.0) > 0:
                raise ValueError(
                    "training.lr_drop.enabled needs training.grad_clip_norm "
                    "> 0: the governor feeds on the pre-clip gradient norm "
                    "that clipping measures."
                )
            lr_governor = GnormLrGovernor(
                factor=float(_drop_cfg.get("factor", 4.0)),
                drop=float(_drop_cfg.get("drop", 0.5)),
                cooldown=int(_drop_cfg.get("cooldown", 100)),
                warmup=int(_drop_cfg.get("warmup", 100)),
                ema_beta=float(_drop_cfg.get("ema_beta", 0.98)),
                min_lr=float(_drop_cfg.get("min_lr", 1.0e-7)),
                freeze_factor=float(_drop_cfg.get("freeze_factor", 2.0)),
            )
            # Rewind-on-drop: cutting lr AFTER an excursion freezes the
            # damage; restoring a healthy snapshot makes the drop
            # restorative. On by default with the governor.
            if bool(_drop_cfg.get("rewind", True)):
                lr_rewinder = RewindBuffer(
                    every=int(_drop_cfg.get("rewind_every", 500)),
                    keep=int(_drop_cfg.get("rewind_keep", 2)),
                )
            else:
                lr_rewinder = None
            logger.info(
                f"lr governor armed: factor={lr_governor.factor}, "
                f"drop={lr_governor.drop}, cooldown={lr_governor.cooldown}, "
                f"freeze_factor={lr_governor.freeze_factor}, "
                f"rewind={'on' if lr_rewinder is not None else 'off'}"
            )
        logger.info(
            f"stage {stage_idx} {stage.name!r} starting at "
            f"global_epoch={global_epoch}: window_mode={window_mode} "
            f"(W={window_size}), steps_per_epoch={steps_per_epoch}, "
            f"sched={type(scheduler_loss).__name__}"
        )

        # Per-stage iteration cap (mirrors train.py's handling — this key
        # was silently ignored by the diffusion trainer before Phase 12b;
        # found when the v2-layout smoke ran a full 729-batch epoch past
        # its 20-iteration override).
        max_iterations = stage.get("max_iterations", float("inf"))
        max_iterations = (
            int(max_iterations) if max_iterations != float("inf") else None
        )
        stage_iter = 0

        for _ in range(stage_epochs - done_in_stage):
            for batch_idx, sample in enumerate(loader):
                if max_iterations is not None and stage_iter >= max_iterations:
                    break
                stage_iter += 1
                losses = _train_step(
                    model=model,
                    scheduler_loss=scheduler_loss,
                    sample=sample,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    amp_dtype=amp_dtype,
                    device=dist.device,
                    window_mode=window_mode,
                    anchor_frames=anchor_frames,
                    grad_clip_norm=float(cfg_train.get("grad_clip_norm", 0.0) or 0.0),
                )
                if ema is not None:
                    ema.update(inner_model, epoch=global_epoch)
                if lr_governor is not None:
                    dropped = lr_governor.update(
                        losses.get("grad_norm"), optimizer, lr_scheduler
                    )
                    if lr_rewinder is not None:
                        if dropped:
                            lr_rewinder.restore(inner_model, optimizer)
                        else:
                            lr_rewinder.maybe_snapshot(
                                stage_iter, inner_model, optimizer,
                                healthy=lr_governor.healthy,
                            )
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
                    # ``loss_ocean`` (Phase 12f) is the predicted-ocean part of
                    # the same total, reported next to it rather than folded in.
                    extra = (
                        f" loss_ocean={losses['loss_ocean']:.4e}"
                        if "loss_ocean" in losses
                        else ""
                    )
                    # The PRE-clip gradient norm. Logged because the failure this
                    # guards against is silent: a run riding the clip ceiling
                    # looks identical to a healthy one in the loss alone, and the
                    # 2026-08-21 runaway was only visible in hindsight.
                    if "grad_norm" in losses:
                        extra += f" gnorm={losses['grad_norm']:.3e}"
                    logger.info(
                        f"epoch {global_epoch} batch {batch_idx}/{steps_per_epoch} "
                        f"loss={losses['loss']:.4e}{extra}"
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
