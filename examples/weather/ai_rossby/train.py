# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra entrypoint for Pangu_Plasim training (Phase 3 v1: PanguPlasimLegacy).

Wires together:

* :class:`physicsnemo.experimental.datapipes.plasim.PlasimClimateDatapipe`
  (Zarr-backed PLASIM datapipe with normalizer + NaN fill)
* :class:`physicsnemo.experimental.models.pangu_plasim.PanguPlasimLegacy`
* :func:`.train_loop.train_step` + :func:`.train_loop.make_optimizer/_scheduler`
* :class:`.loss.PanguPlasimLoss`
* :class:`.ema.ModelEMA`
* :class:`physicsnemo.distributed.DistributedManager` for DDP
* :func:`physicsnemo.utils.save_checkpoint` /
  :func:`physicsnemo.utils.load_checkpoint`
* :class:`physicsnemo.utils.logging.LaunchLogger`

Launch single-GPU:

.. code-block:: bash

    python train.py data.zarr_path=/path/to/store.zarr \\
                    data.mean_path=... data.std_path=...

Launch multi-GPU with torchrun (Delta convention):

.. code-block:: bash

    torchrun --standalone --nproc-per-node=2 train.py [overrides]
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.plasim import (
        ClimateZarrMultiYearDataset,
        ComposeTransform,
        NanFillTransform,
        PlasimClimateDatapipe,
        PlasimClimateDataset,
        PlasimNormalizer,
    )
    # Model classes are imported by name at runtime via Module.instantiate
    # — no direct import needed here, but we keep the namespace warning
    # silencer active for the datapipe + transforms imports above.

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from typing import Optional

from data_staging import _STAGEABLE_KEYS, resolve_stage_root, stage_store
from ema import ModelEMA
from loss import PanguPlasimLoss
from train_loop import (
    _resolve_amp_dtype,
    make_optimizer,
    make_scheduler,
    multistep_train_step,
    train_step,
)
from validate import (
    Deterministic,
    GaussianIC,
    Perturber,
    ReplicateOnly,
    RolloutValidator,
)


def _resolve_path(p: str | None) -> str | None:
    return to_absolute_path(p) if p else None


class _PressureLevelSubsetTransform:
    """Select a subset of pressure levels from upper-air tensors.

    Applied (CPU-side, at the dataset ``__getitem__`` boundary) when the model
    requests fewer / different pressure levels than the data store carries —
    e.g. a 17-level model trained against the 18-level ERA5 archive. It slices
    the level axis (dim 1: (C_var, C_level, H, W)) of every upper-air tensor
    down to the requested levels, so the tensors reaching the normalizer and
    the model match ``cfg.model.levels``. Chain it BEFORE the NaN-fill transform
    (``ComposeTransform(_PressureLevelSubsetTransform(idx), nan_fill)``).
    """

    def __init__(self, indices: list[int]) -> None:
        self._idx = torch.tensor(indices, dtype=torch.long)

    def __call__(self, sample: dict) -> dict:
        out = dict(sample)
        for key in (
            "upper_air_in",
            "target_upper_air",
            "upper_air_pressure_in",
            "target_upper_air_pressure",
        ):
            if key in out and out[key] is not None:
                out[key] = out[key][:, self._idx]
        return out


def _ddp_mean_scalars(values: dict, *, dist) -> dict:
    """All-reduce scalar metrics across ranks (mean), staying on-device.

    Aggregates per-GPU metrics to a global mean before rank 0 logs them, so the
    single wandb run reflects ALL GPUs rather than rank 0's local shard. Does a
    SINGLE all-reduce over the stacked scalars per call and keeps the results as
    on-device tensors — no ``.item()``/``.cpu()``, so there is no added per-step
    host sync (the existing logging cadence does the only host transfer); for
    the compute-bound SFNO models the extra async collective is negligible.

    No-op in single-process runs (returns ``values`` unchanged). All ranks must
    call this with the same keys in the same order — guaranteed here since every
    rank builds the dict identically.
    """
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
    tdist.all_reduce(vec, op=tdist.ReduceOp.AVG)
    return {k: vec[i] for i, k in enumerate(keys)}


def _maybe_stage_data(cfg: DictConfig, *, dist, logger) -> None:
    """Stage the dataset's Zarr stores to node-local disk (once per node).

    Gated by ``cfg.dataset.stage_to_local`` (default ``False``). When on,
    ``local_rank == 0`` on each node copies every configured store
    (``zarr_path`` / ``val_zarr_path`` / normalization stats / boundary
    stores) into a node-local directory in parallel (see
    :mod:`data_staging`); then *all* ranks rewrite their dataset paths to the
    local copies. A global barrier holds the non-copying ranks until every
    node's copy is complete, so no rank reads a store mid-copy.

    The local path is a pure function of ``(stage_root, store basename)``, so
    every rank computes the same rewritten path without any communication —
    the barrier is the only synchronization needed.
    """
    data = cfg.dataset
    if not bool(data.get("stage_to_local", False)):
        return

    stage_root = resolve_stage_root(data.get("stage_dir", None))
    num_workers = int(data.get("stage_num_workers", 256))

    # Deterministic, de-duplicated set of source stores. mean_path/std_path
    # usually resolve to the SAME normalization store — copy it once.
    srcs: list[str] = []
    for key in _STAGEABLE_KEYS:
        raw = data.get(key, None)
        if not raw:
            continue
        src = _resolve_path(raw)
        if src not in srcs:
            srcs.append(src)
    if not srcs:
        return

    def _local_of(src: str) -> str:
        return str(stage_root / Path(src).name)

    if dist.local_rank == 0:
        logger.info(
            f"[stage] staging {len(srcs)} store(s) to {stage_root} "
            f"({num_workers} concurrent copies)"
        )
        prefix = (lambda m: logger.info(f"[stage r{dist.rank}] {m}"))
        for src in srcs:
            stage_store(src, stage_root, num_workers=num_workers, log=prefix)

    if dist.world_size > 1:
        torch.distributed.barrier()

    # All ranks rewrite their dataset paths to the local copies.
    for key in _STAGEABLE_KEYS:
        raw = data.get(key, None)
        if not raw:
            continue
        data[key] = _local_of(_resolve_path(raw))
    logger.info(f"[stage] dataset paths rewritten to node-local {stage_root}")


def _maybe_init_wandb(cfg: DictConfig, *, dist) -> bool:
    """Create a wandb run on EVERY rank; return whether THIS rank drives
    :class:`LaunchLogger`'s wandb backend (rank 0 only).

    Why init on every rank (not just rank 0): wandb spins up background
    threads (service IPC / console capture / GPU-stats monitor) that grab the
    GIL inside CUDA calls. If only rank 0 runs wandb, that jitter is
    asymmetric and desyncs DDP's NCCL collectives — the gradient all-reduce
    deadlocks mid-epoch. Initializing wandb on ALL ranks makes the jitter
    symmetric so the ranks stay in lockstep. This mirrors the PanguWeather
    reference trainer, which calls ``init_wandb`` unconditionally on every
    rank and logs only from rank 0. To avoid one online run PER rank, only
    rank 0 uses the configured wandb ``mode`` (e.g. ``online``); every other
    rank is forced to ``mode="offline"`` so it still spins up wandb's
    background threads (preserving the timing symmetry) but writes a throwaway
    LOCAL run instead of syncing a second run to the server. Net effect:
    exactly ONE run in the wandb UI, DDP stays alive. (The offline ranks leave
    harmless ``./wandb/offline-run-*`` dirs; don't ``wandb sync`` those.)

    Returns ``True`` only on rank 0 (so ``LaunchLogger`` routes metrics to
    wandb from rank 0 alone — avoiding duplicate data across ranks), and
    ``False`` on the other ranks even though they DID create a run. Returns
    ``False`` everywhere when wandb is disabled or not importable.

    IMPORTANT: this must run **before** ``LaunchLogger.initialize`` — the
    logger binds the wandb backend at initialize time and only when a run
    already exists.
    """
    wb = cfg.get("wandb", None)
    if wb is None or not bool(wb.get("enabled", False)):
        return False
    try:
        from physicsnemo.utils.logging.wandb import initialize_wandb
    except ImportError:
        if dist.rank == 0:
            PythonLogger("pangu_plasim_train").warning(
                "wandb.enabled=True but the `wandb` package is not importable; "
                "continuing with console logging only."
            )
        return False

    # Called on ALL ranks (see docstring). Only rank 0 syncs to the server
    # (configured mode); other ranks go offline so there is exactly ONE online
    # run while their background threads still provide the DDP timing symmetry.
    configured_mode = str(wb.get("mode", "offline"))
    rank_mode = configured_mode if dist.rank == 0 else "offline"
    # entity: a null/empty config value must become Python ``None`` (use the
    # account's default entity). NOT ``str(...)`` — OmegaConf null -> the literal
    # string "None", which wandb's server rejects online with
    # "CommError: permission denied" (offline never validates it, so the bug
    # only bites online runs).
    _ent = wb.get("entity", None)
    entity = str(_ent) if _ent else None
    # init_timeout: the 90 s default can trip rank 0 during a full-job launch
    # while other ranks are still loading data; 300 s is safe (configurable).
    initialize_wandb(
        project=str(wb.get("project", "ai-rossby")),
        entity=entity,
        name=str(wb.get("name", cfg.get("run_name", "train"))),
        mode=rank_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_timeout=int(wb.get("init_timeout", 300)),
    )
    # Rank 0 alone routes LaunchLogger metrics to wandb; the other ranks
    # created an offline run only for background-thread symmetry (see docstring).
    return dist.rank == 0


def _build_captured_train_step(
    *,
    inner_model,
    optimizer,
    loss_fn,
    has_diagnostic: bool,
    unroll_steps: int,
    amp_dtype,
    grad_clip_norm: float,
    label: str,
):
    """Return a :class:`StaticCaptureTraining`-wrapped train-step closure.

    The captured closure handles forward + loss + backward + optimizer
    step + optional gradient clipping + optional AMP autocast inside one
    CUDA graph (after the warmup iterations). Caller is responsible for
    stepping the scheduler and the EMA after each call.

    Loss-component breakdowns are NOT exposed by this path — only the
    total scalar loss is returned. The cost of recomputing per-component
    losses outside the graph would defeat the throughput win, so when
    static capture is on the trainer logs `loss` only and zero-fills the
    other component slots.

    Multi-step rollout: each stage builds its own captured function with
    ``unroll_steps`` baked in. The graph topology depends on
    ``unroll_steps`` and batch shape; changing either at runtime forces
    a re-capture, which is why we bind these at stage start.
    """
    from physicsnemo.utils import StaticCaptureTraining

    use_amp = amp_dtype is not None
    amp_type = amp_dtype if amp_dtype is not None else torch.bfloat16
    grad_clip = grad_clip_norm if grad_clip_norm > 0 else None

    @StaticCaptureTraining(
        model=inner_model,
        optim=optimizer,
        use_amp=use_amp,
        amp_type=amp_type,
        use_graphs=True,
        gradient_clip_norm=grad_clip,
        label=label,
    )
    def captured_step(model, batch):
        if unroll_steps == 1:
            out = model(
                batch["surface_in"],
                batch["constant_boundary"],
                batch["varying_boundary"],
                batch["upper_air_in"],
            )
            if has_diagnostic:
                o_s, o_u, o_d = out[0], out[1], out[2]
            else:
                o_s, o_u = out[0], out[1]
                o_d = None
            losses = loss_fn(
                o_s,
                o_u,
                batch["target_surface"],
                batch["target_upper_air"],
                out_diagnostic=o_d,
                target_diagnostic=batch.get("diagnostic") if has_diagnostic else None,
            )
            return losses["loss"]
        # Multi-step rollout — mirrors multistep_train_step but returns only the
        # scalar so it composes with the static-capture graph.
        surface_seq = batch["surface_in_seq"]
        upper_seq = batch["upper_air_in_seq"]
        varying_seq = batch["varying_boundary_seq"]
        diag_seq = batch.get("diagnostic_seq") if has_diagnostic else None
        const_boundary = batch.get("constant_boundary")
        state_surface = surface_seq[:, 0]
        state_upper = upper_seq[:, 0]
        accum = torch.zeros(
            (), device=state_surface.device, dtype=state_surface.dtype
        )
        for k in range(int(unroll_steps)):
            out = model(
                state_surface,
                const_boundary,
                varying_seq[:, k],
                state_upper,
            )
            if has_diagnostic:
                next_surface, next_upper, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper = out[0], out[1]
                next_diag = None
            target_surface_k = surface_seq[:, k + 1]
            target_upper_k = upper_seq[:, k + 1]
            target_diag_k = diag_seq[:, k + 1] if diag_seq is not None else None
            losses_k = loss_fn(
                next_surface,
                next_upper,
                target_surface_k,
                target_upper_k,
                out_diagnostic=next_diag,
                target_diagnostic=target_diag_k,
            )
            accum = accum + losses_k["loss"]
            state_surface = next_surface
            state_upper = next_upper
        return accum / float(unroll_steps)

    return captured_step


def _flatten_optimizer_cfg(opt_cfg: DictConfig) -> DictConfig:
    """Map ``cfg.training.optimizer.{type, lr, weight_decay, fused}`` → the
    flat keys ``make_optimizer`` expects (``optimizer_type``, ``lr``,
    ``weight_decay``, ``fused``). Keeps :func:`make_optimizer`'s API stable
    so the unit tests covering it stay untouched.
    """
    return OmegaConf.create(
        {
            "optimizer_type": str(opt_cfg.type),
            "lr": float(opt_cfg.lr),
            "weight_decay": float(opt_cfg.get("weight_decay", 0.0)),
            "fused": opt_cfg.get("fused", None),
        }
    )


def _flatten_scheduler_cfg(
    sched_cfg: DictConfig,
    *,
    lr: float,
    steps_per_epoch: int,
    num_epochs: int,
) -> DictConfig:
    """Map ``stage.scheduler.{type, ...}`` → the flat keys
    ``make_scheduler`` expects.

    Computes the LinearWarmupCosineAnnealingLR warmup-step count from
    ``num_warmup_epochs`` (epoch-count is the natural unit at config time).
    Provides a CosineAnnealingLR ``T_max`` default of ``steps_per_epoch *
    num_epochs`` when the user didn't override.
    """
    flat = OmegaConf.to_container(sched_cfg, resolve=True) or {}
    flat["scheduler"] = flat.pop("type", "OneCycleLR")
    flat["lr"] = lr
    if flat["scheduler"] == "LinearWarmupCosineAnnealingLR":
        flat["num_warmup_steps"] = int(
            flat.get("num_warmup_epochs", 0) * steps_per_epoch
        )
    if flat["scheduler"] == "CosineAnnealingLR" and "T_max" not in flat:
        flat["T_max"] = max(1, steps_per_epoch * num_epochs)
    return OmegaConf.create(flat)


def _build_perturber(cfg_val: DictConfig) -> Perturber:
    """Translate ``cfg.validation.rollout.perturber`` into a Perturber."""
    kind = str(cfg_val.get("perturber", "deterministic")).lower()
    if kind in ("deterministic", "off", "none"):
        return Deterministic()
    if kind in ("replicate", "replicate_only", "stochastic_model"):
        return ReplicateOnly()
    if kind in ("gaussian_ic", "ic_gaussian", "gaussian"):
        scales = OmegaConf.to_container(
            cfg_val.get("perturber_scales", {}), resolve=True
        )
        if not scales:
            raise ValueError(
                "validation.rollout.perturber=gaussian_ic requires "
                "validation.rollout.perturber_scales={var: std, ...}"
            )
        return GaussianIC(scales=scales)
    raise ValueError(f"unknown validation.rollout.perturber={kind!r}")


_MODEL_CONFIG_ONLY_KEYS = frozenset(
    {
        # Identification fields that are not model constructor args.
        "name",
        "module",
        "target",
        "model_type",
    }
)


def build_model(cfg_model: DictConfig):
    """Instantiate the model via :meth:`physicsnemo.Module.instantiate`.

    Reads ``cfg.model.name`` (class name, e.g. ``"SfnoPlasim"``) and
    ``cfg.model.module`` (import path, e.g.
    ``"physicsnemo.experimental.models.sfno_plasim"``). Every other key in
    ``cfg.model`` is forwarded as a keyword argument to the model's
    constructor — except a small set of config-only keys
    (:data:`_MODEL_CONFIG_ONLY_KEYS`) that identify the class.

    This mirrors the unified-recipe convention and lets the registry/
    import path handle model lookup so new model classes drop in by
    adding their config without touching this file.

    Legacy notes for downstream calls:

    Each ``model_type`` reads only the subset of ``cfg_model`` keys it needs;
    Pangu-specific (patch_size, depths, num_heads, …) and SFNO-specific
    (filter_type, spectral_layers, …) kwargs coexist in their respective YAML
    groups.
    """
    flat = OmegaConf.to_container(cfg_model, resolve=True) or {}
    if "name" not in flat or "module" not in flat:
        raise KeyError(
            "cfg.model must declare both `name` (model class name) and "
            "`module` (Python import path) so Module.instantiate can "
            "resolve the class. Example: name: SfnoPlasim, "
            "module: physicsnemo.experimental.models.sfno_plasim"
        )
    name = str(flat["name"])
    module_path = str(flat["module"])
    args = {k: v for k, v in flat.items() if k not in _MODEL_CONFIG_ONLY_KEYS}
    return Module.instantiate(
        {
            "__name__": name,
            "__module__": module_path,
            "__args__": args,
        }
    )


def build_datapipe(
    cfg: DictConfig,
    *,
    zarr_path: str,
    distributed: bool,
    device: torch.device,
    shuffle: bool,
    seed: int,
    batch_size_override: Optional[int] = None,
    unroll_steps: int = 1,
) -> PlasimClimateDatapipe:
    """Construct a PlasimClimateDatapipe wired with normalizer + NaN fill."""
    data = cfg.dataset
    model = cfg.model

    # A non-``.zarr`` directory is a multi-year archive (per-year sub-stores);
    # a single ``.zarr`` store uses the single-year dataset. Mirrors the routing
    # in ClimateDatapipe so ``raw_dataset`` (used for the normalizer layout)
    # matches what the datapipe builds internally.
    _ds_kwargs = dict(
        boundary_zarr_path=_resolve_path(data.boundary_zarr_path),
        yearly_repeating_boundary=bool(data.yearly_repeating_boundary),
        leap_boundary_zarr_path=_resolve_path(data.leap_boundary_zarr_path),
        non_leap_boundary_zarr_path=_resolve_path(data.non_leap_boundary_zarr_path),
    )
    _p = Path(zarr_path)
    if _p.is_dir() and not str(zarr_path).endswith(".zarr"):
        raw_dataset = ClimateZarrMultiYearDataset(zarr_path, **_ds_kwargs)
    else:
        raw_dataset = PlasimClimateDataset(zarr_path, **_ds_kwargs)

    # Pressure-level subset: when the model uses a strict subset of the data
    # store's PRESSURE levels (e.g. a 17-level model on the 18-level ERA5
    # archive), compute the subset indices, tell the normalizer which levels to
    # align to, and chain a subset transform (below) so the upper-air tensors
    # are trimmed to the model's levels before normalization + forward.
    #
    # Guarded to fire ONLY for that case: it is skipped when the data has no
    # pressure levels (sigma/hybrid datasets — E3SM, PLASIM-sigma), when the
    # level sets already match, or when the model's levels are not all present
    # in the data's pressure levels (a different level system, not a subset).
    # This keeps sigma-level recipes unaffected. A genuinely inconsistent
    # request still surfaces downstream via the normalizer's by-value matching.
    pressure_level_indices: Optional[list[int]] = None
    model_levels = list(model.get("levels", []) or [])
    data_pressure_levels = list(raw_dataset.pressure_levels)
    if model_levels and data_pressure_levels:
        model_set = set(map(float, model_levels))
        data_set = set(map(float, data_pressure_levels))
        if model_set != data_set and model_set.issubset(data_set):
            level_to_idx = {float(lv): i for i, lv in enumerate(data_pressure_levels)}
            pressure_level_indices = [level_to_idx[float(lv)] for lv in model_levels]

    normalizer_kwargs: dict = {}
    if pressure_level_indices is not None:
        normalizer_kwargs["pressure_levels"] = [float(lv) for lv in model_levels]
    if data.delta_std_path:
        normalizer_kwargs["predict_delta"] = True
        normalizer_kwargs["delta_std_path"] = _resolve_path(data.delta_std_path)
    # Opt-in normalization for constant boundary + diagnostic fields. Off by
    # default for back-compat (ERA5 stats often skip the const-boundary vars);
    # PLASIM stats include lsm/sg/z0, so PLASIM recipes should flip these on
    # to match PanguWeather's data loader (which always normalizes them).
    normalizer_kwargs["normalize_constant_boundary"] = bool(
        data.get("normalize_constant_boundary", False)
    )
    normalizer_kwargs["normalize_diagnostic"] = bool(
        data.get("normalize_diagnostic", False)
    )

    normalizer = PlasimNormalizer.from_dataset(
        raw_dataset,
        mean_path=_resolve_path(data.mean_path),
        std_path=_resolve_path(data.std_path),
        **normalizer_kwargs,
    )

    nan_fill = NanFillTransform(
        constant_boundary_variables=list(model.constant_boundary_variables),
        varying_boundary_variables=list(model.varying_boundary_variables),
        fill_values=dict(OmegaConf.to_container(data.nan_fill_values, resolve=True) or {}),
        default=float(data.nan_fill_default),
    )

    effective_batch = int(batch_size_override) if batch_size_override else int(data.batch_size)
    pipe = PlasimClimateDatapipe(
        zarr_path,
        forecast_lead_times=list(data.forecast_lead_times),
        normalizer=normalizer,
        nan_fill=None,  # per-variable NaN fill runs CPU-side via dataset.transform below.
        batch_size=effective_batch,
        num_samples_per_epoch=data.num_samples_per_epoch,
        shuffle=shuffle,
        num_workers=int(data.num_workers),
        prefetch_factor=int(data.prefetch_factor),
        persistent_workers=bool(data.persistent_workers),
        pin_memory=bool(data.pin_memory),
        device=device,
        seed=int(seed),
        distributed=distributed,
        boundary_zarr_path=_resolve_path(data.boundary_zarr_path),
        yearly_repeating_boundary=bool(data.yearly_repeating_boundary),
        leap_boundary_zarr_path=_resolve_path(data.leap_boundary_zarr_path),
        non_leap_boundary_zarr_path=_resolve_path(data.non_leap_boundary_zarr_path),
        unroll_steps=int(unroll_steps),
    )

    # Attach the per-variable NaN fill as the dataset's transform so it runs
    # in worker processes (CPU) before pin_memory + device transfer.
    # In sequence mode the datapipe wraps the dataset in SequenceDataset,
    # whose `.transform` setter forwards to the base dataset.
    #
    # When the model uses a subset of the data's pressure levels, trim the
    # upper-air tensors to those levels BEFORE the NaN-fill so every downstream
    # tensor (normalizer + model) is on the model's level set.
    if pressure_level_indices is not None:
        pipe.dataset.transform = ComposeTransform(
            _PressureLevelSubsetTransform(pressure_level_indices), nan_fill
        )
    else:
        pipe.dataset.transform = nan_fill
    return pipe


def build_loss(cfg: DictConfig) -> PanguPlasimLoss:
    """Build :class:`PanguPlasimLoss` aligned with the model's variable groups.

    ``cfg.model.upper_air_variables`` is assumed to be in sigma-then-pressure
    order so it matches the channel ordering produced by
    :class:`PlasimClimateDataset` (and the model's forward output).
    """
    cfg_loss = cfg.loss
    cfg_model = cfg.model

    def _maybe_dict(v):
        if v is None:
            return None
        return dict(OmegaConf.to_container(v, resolve=True) or {})

    return PanguPlasimLoss(
        surface_variables=list(cfg_model.surface_variables),
        upper_air_variable_names=list(cfg_model.upper_air_variables),
        diagnostic_variables=list(cfg_model.diagnostic_variables),
        num_lat=int(cfg_model.horizontal_resolution[0]),
        loss_type=str(cfg_loss.loss_type),
        surface_weight=float(cfg_loss.surface_weight),
        upper_air_weight=float(cfg_loss.upper_air_weight),
        diagnostic_weight=float(cfg_loss.diagnostic_weight),
        surface_var_weights=_maybe_dict(cfg_loss.surface_var_weights),
        upper_air_var_weights=_maybe_dict(cfg_loss.upper_air_var_weights),
        diagnostic_var_weights=_maybe_dict(cfg_loss.diagnostic_var_weights),
        latitude_weighted=bool(cfg_loss.get("latitude_weighted", True)),
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # A100 TF32 + cudnn autotune. PanguWeather's reference SFNO trainer flips
    # these on by default; without them ai-rossby was running true fp32 matmul
    # against PanguWeather's TF32 and losing ~15% throughput per benchmarks.
    # TF32 changes mantissa precision (10 bits vs 23) but on A100 stays within
    # ~3 decimals of fp32 — well below typical training noise.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("pangu_plasim_train")

    torch.manual_seed(int(cfg.seed) + dist.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed) + dist.rank)

    # --- Wandb + logger backend -------------------------------------------
    # Create the wandb run (rank 0) FIRST, then hand the result to
    # LaunchLogger.initialize so it binds the wandb backend. With the
    # backend on, every LaunchLogger log_minibatch / log_epoch dict —
    # training ('train/') AND validation ('valid/') — is routed to wandb
    # automatically. Ordering matters: LaunchLogger only binds wandb when a
    # run already exists, so this must precede initialize().
    wandb_active = _maybe_init_wandb(cfg, dist=dist)
    LaunchLogger.initialize(use_wandb=wandb_active)

    # --- Optional: stage data to node-local disk --------------------------
    # Must run before build_datapipe so the datapipe opens the local copies.
    # No-op unless dataset.stage_to_local=True.
    _maybe_stage_data(cfg, dist=dist, logger=logger)

    # --- Data -------------------------------------------------------------
    datapipe = build_datapipe(
        cfg,
        zarr_path=_resolve_path(cfg.dataset.zarr_path),
        distributed=(dist.world_size > 1),
        device=dist.device,
        shuffle=bool(cfg.dataset.shuffle),
        seed=int(cfg.seed),
    )

    has_val = cfg.dataset.val_zarr_path is not None
    val_datapipe = None
    if has_val:
        val_datapipe = build_datapipe(
            cfg,
            zarr_path=_resolve_path(cfg.dataset.val_zarr_path),
            distributed=(dist.world_size > 1),
            device=dist.device,
            shuffle=False,
            seed=int(cfg.seed) + 1,
        )

    # --- Rollout validator (optional) -------------------------------------
    # Multi-step autoregressive RMSE + ACC against the held-out year. Streams
    # metrics on the fly (no per-step history retained) and is ensemble-aware
    # via the Perturber API.
    rollout_validator = None
    cfg_rollout = cfg.get("validation", {}).get("rollout", None) if cfg.get("validation") else None
    if cfg_rollout is not None and bool(cfg_rollout.get("enabled", False)):
        if not has_val:
            logger.warning(
                "validation.rollout.enabled=True but dataset.val_zarr_path is None; "
                "skipping rollout validation."
            )
        else:
            rollout_validator = RolloutValidator(
                dataset=val_datapipe.dataset,
                log_steps=list(cfg_rollout.get("log_steps", [1])),
                device=dist.device,
                ensemble_size=int(cfg_rollout.get("ensemble_size", 1)),
                perturber=_build_perturber(cfg_rollout),
                has_diagnostic=False,  # populated after model is built
                batch_size=int(cfg_rollout.get("batch_size", 1)),
                max_initial_conditions=int(
                    cfg_rollout.get("max_initial_conditions", 4)
                ),
                ic_stride=int(cfg_rollout.get("ic_stride", 1)),
                normalizer=val_datapipe.normalizer,
                seed=int(cfg.seed) + 17,
            )

    cfg_train = cfg.training
    stages = list(cfg_train.stages)
    total_epochs = sum(int(s.num_epochs) for s in stages)
    steps_per_epoch = len(datapipe)
    logger.info(
        f"steps_per_epoch={steps_per_epoch}, stages={len(stages)}, "
        f"total_epochs={total_epochs}, world_size={dist.world_size}, "
        f"device={dist.device}"
    )

    # --- Model + DDP ------------------------------------------------------
    model = build_model(cfg.model).to(dist.device)
    if dist.world_size > 1:
        # bucket_cap_mb / static_graph are opt-in (None/False preserves the
        # prior PyTorch-default behavior). Motivated by profiling that showed
        # NCCL AllReduce at ~18.5% of GPU time across many small (25MB
        # default) buckets — both NVIDIA's makani (SFNO/FourCastNet training
        # library; defaults to ONE bucket sized to the whole local parameter
        # set) and the PanguWeather-e3sm reference trainer (bucket_cap_mb=250,
        # find_unused_parameters=False) consolidate buckets to cut per-call
        # NCCL overhead. static_graph=True is safe here because checkpointing
        # branches on a fixed hyperparameter (same set of params gets
        # gradients every iteration), not on data-dependent control flow.
        #
        # NOTE: torch is pinned <2.11 (see pyproject) because torch 2.11
        # regressed DDP for the SFNO models — DDP.__init__ raises an
        # int-overflow in _verify_params_across_processes, and the reducer
        # deadlocks mid-epoch. On 2.10.x this standard wrap matches the
        # working SFNO-PlaSim benchmark. Separately, wandb is initialized on
        # every rank (see _maybe_init_wandb) — running it on rank 0 alone
        # desyncs the ranks and its background threads stall NCCL, deadlocking
        # training; all-rank init keeps the GIL/CUDA jitter symmetric.
        bucket_cap_mb = cfg_train.get("ddp_bucket_cap_mb", None)
        ddp_kwargs = {}
        if bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = int(bucket_cap_mb)

        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            output_device=dist.device if dist.device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=bool(cfg_train.get("ddp_static_graph", False)),
            **ddp_kwargs,
        )
    inner_model = model.module if hasattr(model, "module") else model

    # Optional torch.compile — NVIDIA's makani applies this unconditionally
    # (`self.model = torch.compile(self.model)`) to its SFNO/FourCastNet
    # models. The SHT + complex-contraction island is carved out of the graph
    # via `@torch.compiler.disable` on SpectralFilterLayer.forward (torch-
    # inductor can't codegen the complex ops), so compile fuses the real-valued
    # encoder/decoder/norm/MLP/skip compute — the ~60% of per-step cost that is
    # GEMM/conv/pointwise. dynamic=False because our shapes are static (fixed
    # grid + fixed local batch), avoiding recompiles on the shape-dependent
    # scatter guards. Opt-in; validate loss stays within noise before adopting.
    if bool(cfg_train.get("jit_compile", False)):
        model = torch.compile(model, dynamic=False)

    # --- Loss + optim ------------------------------------------------------
    loss_fn = build_loss(cfg).to(dist.device)
    optimizer = make_optimizer(inner_model, _flatten_optimizer_cfg(cfg_train.optimizer))

    # --- Mixed precision --------------------------------------------------
    amp_dtype = _resolve_amp_dtype(cfg_train.get("amp", None))
    # fp16 needs a GradScaler; bf16 retains enough dynamic range to skip it.
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

    # --- Checkpoint resume ------------------------------------------------
    # We resume the model + optimizer state only — the scheduler is
    # reconstructed per-stage to keep the multi-stage semantics clean. To
    # resume the scheduler exactly, the per-stage iteration count is replayed
    # in-loop below.
    ckpt_dir = Path("./checkpoints")
    loaded_epoch = load_checkpoint(
        str(ckpt_dir),
        models=inner_model,
        optimizer=optimizer,
        device=dist.device,
    )
    start_global_epoch = max(int(cfg.start_epoch), loaded_epoch + 1)

    # --- Per-batch loss TSV (benchmarking) --------------------------------
    # When cfg.bench.per_batch_tsv is set, every minibatch's wall-clock time
    # + loss components are appended to a TSV for downstream comparison
    # against PanguWeather. Only rank 0 writes to avoid file contention.
    bench_tsv_path = None
    bench_tsv_file = None
    if cfg.get("bench") and cfg.bench.get("per_batch_tsv"):
        if dist.rank == 0:
            bench_tsv_path = Path(_resolve_path(cfg.bench.per_batch_tsv))
            bench_tsv_path.parent.mkdir(parents=True, exist_ok=True)
            bench_tsv_file = open(bench_tsv_path, "w", buffering=1)  # line-buffered
            bench_tsv_file.write(
                "epoch\tbatch_idx\twall_s\tloss\tsurface\tupper_air\tdiagnostic\tvae_kl\tlr\n"
            )
            logger.info(f"benchmark per-batch TSV → {bench_tsv_path}")

    import time as _time
    _bench_start_wall = _time.perf_counter() if bench_tsv_file is not None else None

    # --- Training loop: iterate stages -----------------------------------
    # Each stage has its own scheduler (built fresh at stage start), its own
    # num_epochs, optionally its own batch_size and unroll_steps. The global
    # epoch index advances continuously across stages so checkpoint resume
    # and EMA warmup keep working unchanged.
    has_diagnostic = inner_model.has_diagnostic
    if rollout_validator is not None:
        rollout_validator.has_diagnostic = has_diagnostic
    grad_clip_norm = float(cfg_train.get("grad_clip_norm", 0.0))
    vae_kl_weight = float(cfg.loss.get("vae_kl_weight", 0.0))
    use_static_capture = bool(cfg_train.get("use_static_capture", False))
    if use_static_capture and vae_kl_weight > 0.0:
        logger.warning(
            "training.use_static_capture=True is incompatible with VAE-KL "
            "training (loss.vae_kl_weight>0); falling back to eager path."
        )
        use_static_capture = False

    global_epoch = 1
    for stage_idx, stage in enumerate(stages):
        stage_num_epochs = int(stage.num_epochs)
        stage_end = global_epoch + stage_num_epochs
        if start_global_epoch >= stage_end:
            global_epoch = stage_end
            continue
        unroll_steps = int(stage.get("unroll_steps", 1))
        max_iterations = stage.get("max_iterations", float("inf"))
        max_iterations = (
            int(max_iterations) if max_iterations != float("inf") else None
        )
        stage_batch_override = stage.get("batch_size", None)

        # Rebuild the datapipe when the stage overrides batch_size OR
        # unroll_steps > 1 (the latter switches the underlying dataset to
        # sequence emission via SequenceDataset).
        needs_new_datapipe = (
            unroll_steps > 1
            or (stage_batch_override and int(stage_batch_override) != int(cfg.dataset.batch_size))
        )
        if needs_new_datapipe:
            stage_datapipe = build_datapipe(
                cfg,
                zarr_path=_resolve_path(cfg.dataset.zarr_path),
                distributed=(dist.world_size > 1),
                device=dist.device,
                shuffle=bool(cfg.dataset.shuffle),
                seed=int(cfg.seed) + stage_idx,
                batch_size_override=(
                    int(stage_batch_override) if stage_batch_override else None
                ),
                unroll_steps=unroll_steps,
            )
        else:
            stage_datapipe = datapipe

        steps_per_epoch_stage = len(stage_datapipe)
        scheduler = make_scheduler(
            optimizer,
            _flatten_scheduler_cfg(
                stage.scheduler,
                lr=float(cfg_train.optimizer.lr),
                steps_per_epoch=steps_per_epoch_stage,
                num_epochs=stage_num_epochs,
            ),
            total_steps=steps_per_epoch_stage * stage_num_epochs,
        )

        # If resuming partway through this stage, advance the scheduler.
        epochs_already_run_in_stage = max(0, start_global_epoch - global_epoch)
        for _ in range(epochs_already_run_in_stage * steps_per_epoch_stage):
            scheduler.step()
        global_epoch = max(global_epoch, start_global_epoch)
        epochs_remaining = stage_end - global_epoch

        if dist.rank == 0:
            logger.info(
                f"stage {stage_idx} '{stage.get('name', '?')}' starting at "
                f"global_epoch={global_epoch}, num_epochs_remaining={epochs_remaining}, "
                f"unroll_steps={unroll_steps}, batch_size={int(stage_batch_override) if stage_batch_override else int(cfg.dataset.batch_size)}"
            )

        # StaticCapture closure for this stage (graph topology depends on
        # unroll_steps + batch shape so we re-bind per stage).
        captured_train_fn = None
        if use_static_capture:
            captured_train_fn = _build_captured_train_step(
                inner_model=inner_model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                has_diagnostic=has_diagnostic,
                unroll_steps=unroll_steps,
                amp_dtype=amp_dtype,
                grad_clip_norm=grad_clip_norm,
                label=f"stage{stage_idx}_unroll{unroll_steps}",
            )
            if dist.rank == 0:
                logger.info(
                    f"  static capture ENABLED (use_amp={amp_dtype is not None}, "
                    f"amp_type={amp_dtype})"
                )

        stage_iter = 0
        for _ in range(epochs_remaining):
            stage_datapipe.set_epoch(global_epoch)
            model.train()
            with LaunchLogger(
                "train",
                epoch=global_epoch,
                num_mini_batch=steps_per_epoch_stage,
                epoch_alert_freq=1,
            ) as log:
                for batch_idx, batch in enumerate(stage_datapipe):
                    if max_iterations is not None and stage_iter >= max_iterations:
                        break
                    if captured_train_fn is not None:
                        # StaticCaptureTraining-wrapped path: returns scalar
                        # loss only. The decorator handles backward + step +
                        # grad clip + AMP internally. We still drive the
                        # scheduler from out here.
                        scalar = captured_train_fn(inner_model, batch)
                        if scheduler is not None:
                            scheduler.step()
                        # Synthesize a losses dict that matches the eager
                        # path's shape so downstream logging stays uniform.
                        zero = torch.zeros((), device=scalar.device, dtype=scalar.dtype)
                        losses = {
                            "loss": scalar.detach(),
                            "surface": zero,
                            "upper_air": zero,
                            "diagnostic": zero,
                            "vae_kl": zero,
                        }
                    elif unroll_steps > 1:
                        losses = multistep_train_step(
                            model=model,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            batch=batch,
                            has_diagnostic=has_diagnostic,
                            unroll_steps=unroll_steps,
                            vae_kl_weight=vae_kl_weight,
                            amp_dtype=amp_dtype,
                            grad_scaler=grad_scaler,
                        )
                    else:
                        losses = train_step(
                            model=model,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            batch=batch,
                            has_diagnostic=has_diagnostic,
                            vae_kl_weight=vae_kl_weight,
                            amp_dtype=amp_dtype,
                            grad_scaler=grad_scaler,
                        )
                    # Gradient clipping: StaticCapture handles it inside the
                    # graph; eager path applies it here after backward.
                    if grad_clip_norm > 0 and captured_train_fn is None:
                        torch.nn.utils.clip_grad_norm_(
                            inner_model.parameters(), grad_clip_norm
                        )
                    if ema is not None:
                        ema.update(inner_model, epoch=global_epoch)
                    # Aggregate the per-GPU minibatch losses to a global mean
                    # before logging, so the single wandb run's per-step series
                    # reflects all ranks (not just rank 0's local batch). On-
                    # device, no per-step host sync (see _ddp_mean_scalars).
                    log.log_minibatch(
                        _ddp_mean_scalars(
                            {
                                "loss": losses["loss"].detach(),
                                "surface": losses["surface"],
                                "upper_air": losses["upper_air"],
                                "diagnostic": losses["diagnostic"],
                                "vae_kl": losses["vae_kl"],
                            },
                            dist=dist,
                        )
                    )
                    if bench_tsv_file is not None:
                        bench_tsv_file.write(
                            f"{global_epoch}\t{batch_idx}\t"
                            f"{_time.perf_counter() - _bench_start_wall:.4f}\t"
                            f"{float(losses['loss'].detach()):.6f}\t"
                            f"{float(losses['surface']):.6f}\t"
                            f"{float(losses['upper_air']):.6f}\t"
                            f"{float(losses['diagnostic']):.6f}\t"
                            f"{float(losses['vae_kl']):.6f}\t"
                            f"{optimizer.param_groups[0]['lr']:.6e}\n"
                        )
                    stage_iter += 1
                log.log_epoch(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "stage_idx": stage_idx,
                    }
                )

            # --- Validation (optional) ----------------------------------
            if val_datapipe is not None:
                val_datapipe.set_epoch(global_epoch)
                if ema is not None:
                    ema.apply_to(inner_model)
                model.eval()
                with LaunchLogger("valid", epoch=global_epoch) as log:
                    with torch.no_grad():
                        total = 0
                        accum = torch.zeros((), device=dist.device)
                        for batch in val_datapipe:
                            out = inner_model(
                                batch["surface_in"],
                                batch["constant_boundary"],
                                batch["varying_boundary"],
                                batch["upper_air_in"],
                            )
                            if has_diagnostic:
                                o_s, o_u, o_d = out[0], out[1], out[2]
                            else:
                                o_s, o_u = out[0], out[1]
                                o_d = None
                            l = loss_fn(
                                o_s,
                                o_u,
                                batch["target_surface"],
                                batch["target_upper_air"],
                                out_diagnostic=o_d,
                                target_diagnostic=batch.get("diagnostic")
                                if has_diagnostic
                                else None,
                            )["loss"]
                            accum += l.detach() * batch["surface_in"].shape[0]
                            total += batch["surface_in"].shape[0]
                        # Aggregate the running loss + sample count across all
                        # ranks so val_loss is the GLOBAL mean — each rank sees
                        # only its shard of the validation set. Mirrors the
                        # cross-rank reduction LaunchLogger applies to the train
                        # loss. One collective per validation pass.
                        if getattr(dist, "distributed", False) and dist.world_size > 1:
                            import torch.distributed as tdist

                            total_t = torch.tensor(float(total), device=dist.device)
                            tdist.all_reduce(accum, op=tdist.ReduceOp.SUM)
                            tdist.all_reduce(total_t, op=tdist.ReduceOp.SUM)
                            total = float(total_t.item())
                        val_loss = (accum / max(total, 1.0)).item()

                    rollout_metrics: dict[str, float] = {}
                    if rollout_validator is not None:
                        every = int(
                            (cfg.get("validation") or {}).get("every_n_epochs", 1)
                        )
                        if every > 0 and global_epoch % every == 0:
                            rollout_metrics = rollout_validator.run(
                                inner_model, epoch=global_epoch
                            )

                    log.log_epoch({"val_loss": val_loss, **rollout_metrics})
                if ema is not None:
                    ema.restore(inner_model)
                model.train()

            if dist.world_size > 1:
                torch.distributed.barrier()

            # --- Save checkpoint ----------------------------------------
            if (
                global_epoch % int(cfg.checkpoint_save_interval) == 0
                and dist.rank == 0
            ):
                save_checkpoint(
                    str(ckpt_dir),
                    models=inner_model,
                    optimizer=optimizer,
                    epoch=global_epoch,
                    metadata={"ema": ema.state_dict() if ema is not None else None},
                )

            global_epoch += 1
            if max_iterations is not None and stage_iter >= max_iterations:
                break

    if dist.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
