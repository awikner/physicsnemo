# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mid-training rollout validator for the AMIP diffusion recipes.

Sibling of :mod:`validate` — same memory discipline (streaming metrics,
no per-step state retention, DDP-safe finalize) but the model contract
is different:

* The model is a *wrapper* (``AmipDiTWrapper`` / ``RollingDiTWrapper`` /
  ``ERDMWrapper``) whose forward expects packed flat tensors, not the
  structured dict. The validator drives ``wrapper.pack_state`` /
  ``unpack_state`` between rollout steps.
* A prediction step is **a full diffusion sample**, not a single model
  forward — the validator calls ``scheduler.sample(...)`` /
  ``scheduler.sample_rollout(...)`` and pays ``num_steps`` model forwards
  per emitted frame. The sampler ``num_steps`` is decoupled from the
  training scheduler's ``num_steps`` so that long-horizon validation can
  run with a fast sampler (e.g. 4 steps) while training keeps the
  high-fidelity 10–20 step schedule.
* Three metrics are scored per (log_step, channel-group):
  lat-weighted RMSE, lat-weighted anomaly correlation (ACC), and
  ensemble spread (lat-weighted stddev across ensemble members). When
  ``ensemble_size=1`` spread is suppressed.

Dispatch is on the inference scheduler type — schedulers exposing
``sample_rollout`` (RFM / ERDM) take the window-rollout path,
single-step schedulers (DriftScheduler / DynamicInterpolant) take the
autoregressive single-step path. Horizon defaults to the training
window size for rolling schedulers and to ``max(log_steps)`` for
single-step schedulers; both can be overridden via the validator
config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist

# Reuse the metric + perturber building blocks from the deterministic
# validator — they are model-agnostic and DDP-safe by construction.
from validate import (  # noqa: E402
    Deterministic,
    GaussianIC,
    Perturber,
    ReplicateOnly,
    StreamingLatWeightedACC,
    StreamingLatWeightedRMSE,
    _all_reduce_sum,
    cos_lat_weights,
)


# ---------------------------------------------------------------------------
# Ensemble spread metric — new for diffusion validation.
# ---------------------------------------------------------------------------


class StreamingLatWeightedSpread:
    r"""Per-(step, channel) lat-weighted ensemble standard deviation.

    Streaming, DDP-safe analogue of
    :class:`StreamingLatWeightedRMSE`. Maintains two running sums per
    (step, channel) across all (IC × ensemble × spatial) entries:

    * ``sum_var_w[s, c] = Σ cos(lat) · Var_E(pred)`` summed over batch +
      spatial dims
    * ``weight_total[s, c] = Σ cos(lat)`` summed over the same dims

    ``finalize()`` returns ``sqrt(sum_var_w / weight_total)`` — the
    lat-weighted RMS of the per-IC per-pixel ensemble standard deviation.
    With ``ensemble_size=1`` the per-IC variance is undefined; the
    caller skips the metric entirely in that case.
    """

    def __init__(
        self,
        *,
        n_steps: int,
        n_channels: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (n_steps, n_channels)
        self.sum_var_w = torch.zeros(shape, device=device, dtype=dtype)
        self.weight_total = torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def update(
        self,
        step_index: int,
        pred_ensemble: torch.Tensor,
        lat_weights: torch.Tensor,
        ensemble_size: int,
    ) -> None:
        r"""Accumulate the (step, channel) variance contribution.

        ``pred_ensemble`` is the diffusion ensemble *before* the
        per-IC mean reduction — shape ``(B*E, C, [L,] H, W)`` with the
        ensemble axis interleaved per IC. Reshape to
        ``(B, E, C, [L,] H, W)``, take ``var(dim=1, unbiased=False)``,
        then accumulate lat-weighted sums over (B, [L,], H, W).
        """
        if ensemble_size <= 1:
            return
        rest = pred_ensemble.shape[1:]
        if pred_ensemble.shape[0] % ensemble_size != 0:
            raise ValueError(
                f"batch dim {pred_ensemble.shape[0]} not divisible by "
                f"ensemble_size {ensemble_size}"
            )
        n_ic = pred_ensemble.shape[0] // ensemble_size
        var = (
            pred_ensemble.float()
            .view(n_ic, ensemble_size, *rest)
            .var(dim=1, unbiased=False)
        )  # → (B, C, [L,] H, W)
        self.update_from_var(step_index, var, lat_weights)

    @torch.no_grad()
    def update_from_var(
        self,
        step_index: int,
        var: torch.Tensor,
        lat_weights: torch.Tensor,
    ) -> None:
        r"""Accumulate an already-computed per-pixel ensemble variance.

        The member-split (multi-GPU) path computes the variance over the
        member UNION across ranks (validator's ``_cross_rank_ensemble_var``)
        — a rank-local ``var`` over a subset would be biased low and average
        wrongly — and hands the finished ``(B, C, [L,] H, W)`` field here.
        The single-process :meth:`update` delegates to this after its local
        ``var``, so the non-split path is numerically unchanged.
        """
        weight_shape = [1] * var.ndim
        weight_shape[-2] = lat_weights.shape[0]
        w = lat_weights.view(weight_shape)
        # Reduce over batch + level + spatial → (C,)
        reduce_dims = [d for d in range(var.ndim) if d != 1]
        self.sum_var_w[step_index] += (var * w).sum(dim=reduce_dims).detach()
        self.weight_total[step_index] += (
            w.expand_as(var).sum(dim=reduce_dims).detach()
        )

    def finalize(self, *, local_only: bool = False) -> torch.Tensor:
        # Clones, not in-place: see StreamingLatWeightedRMSE.finalize.
        s = self.sum_var_w.clone()
        w = self.weight_total.clone()
        if not local_only:
            _all_reduce_sum(s)
            _all_reduce_sum(w)
        return torch.sqrt(s / w.clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Diffusion rollout validator.
# ---------------------------------------------------------------------------


def _interleave_ensemble(sample, ensemble_size):
    """``B → B * E`` along dim 0 for tensor entries with batch dim."""
    if ensemble_size <= 1:
        return sample
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            out[k] = v.repeat_interleave(ensemble_size, dim=0)
        else:
            out[k] = v
    return out


@dataclass
class StepContext:
    """Everything a scorer may need for one (frame, channel-group) — computed
    ONCE per (step, kind) by the drive, so N scorers cost zero extra
    collectives, denormalizations or ensemble reductions. ``pred_mean`` is the
    cross-rank ensemble mean; ``pred_var`` is the member-UNION per-pixel
    variance and is None iff ensemble_size == 1."""

    step: int                     # k, 1-indexed emitted-frame number
    m_idx: int                    # index into the validator's log_steps
    kind: str                     # "surface" | "upper_air" | "diagnostic"
    pred_ensemble: torch.Tensor   # (B*local_E, C, [L,] H, W), normalized
    pred_mean: torch.Tensor       # (B, ...), normalized
    pred_var: Optional[torch.Tensor]
    pred_phys: torch.Tensor       # denormalized pred_mean
    truth: torch.Tensor           # (B, ...), normalized
    truth_phys: torch.Tensor
    lat_weights: torch.Tensor
    ensemble_size: int
    local_ensemble_size: int


class DiffusionRolloutValidator:
    r"""Diffusion-aware rollout validator.

    Runs ``ensemble_size`` parallel autoregressive rollouts per initial
    condition, scoring RMSE / ACC / spread at the requested
    ``log_steps``. The cost per emitted frame is
    ``ensemble_size × sampler_num_steps`` model forwards.

    Parameters
    ----------
    dataset
        :class:`ClimateZarrDataset` opened on the validation Zarr, with
        ``emit_calendar=True`` and the training transform pipeline
        (normalizer + nan-fill) already wired.
    wrapper
        The model wrapper (``AmipDiTWrapper`` etc.) — used for
        pack/unpack between rollout steps.
    inference_scheduler
        The diffusion scheduler used at inference. Distinct from the
        training scheduler so the sampler step count can differ.
    log_steps
        Lead times (in dataset steps) at which to record metrics.
    horizon
        Number of frames to roll out per IC. Defaults to
        ``max(log_steps)`` for single-step schedulers and to the
        training window size for rolling schedulers.
    ensemble_size
        Number of ensemble members per IC.
    perturber
        :class:`Perturber` strategy. Defaults to :class:`Deterministic`
        when ``ensemble_size=1`` and :class:`ReplicateOnly` otherwise.
    has_diagnostic
        Whether the wrapper emits a diagnostic channel group.
    batch_size
        Number of ICs to roll out per validator iteration. Effective
        device batch is ``batch_size × ensemble_size``.
    max_initial_conditions
        Total ICs evaluated per ``run()`` call across all ranks.
        Default 4 per Phase 8c follow-up Q3.
    ic_stride
        Spacing in dataset steps between consecutive ICs.
    climatology_*
        Optional climatologies for the three channel groups, used by
        ACC.
    normalizer
        Optional :class:`ClimateNormalizer` for denormalizing
        predictions and targets before scoring RMSE (so RMSE numbers
        are in physical units). ACC and spread are unit-invariant and
        skip denorm.
    sampler_num_steps
        Number of diffusion solver steps per emitted frame at
        inference. Accepts three forms (Phase 8f, F4):

        * ``None`` — falls back to the scheduler's own ``num_steps``
          attribute (training default).
        * ``int`` — applied uniformly to every emitted frame (previous
          behavior).
        * ``Sequence[int]`` of length ``horizon`` — a per-emitted-frame
          schedule, e.g. more solver steps for the first few (harder)
          frames and fewer for later ones, capping sampling cost at
          long horizons. Frame ``k`` (1-indexed, ``k=1..horizon``) uses
          ``sampler_num_steps[k - 1]``. For window-mode schedulers
          (RFM / ERDM), the schedule is forwarded verbatim to
          ``scheduler.sample_rollout(..., num_steps=...)``, which
          indexes it the same way internally.
    seed
        Per-epoch RNG seed for the perturber.
    """

    def __init__(
        self,
        dataset,
        *,
        wrapper,
        inference_scheduler,
        log_steps: Sequence[int],
        device: torch.device,
        horizon: Optional[int] = None,
        ensemble_size: int = 1,
        perturber: Optional[Perturber] = None,
        has_diagnostic: bool = False,
        batch_size: int = 1,
        max_initial_conditions: int = 4,
        ic_stride: int = 1,
        step_size: int = 1,
        climatology_surface: Optional[torch.Tensor] = None,
        climatology_upper_air: Optional[torch.Tensor] = None,
        climatology_diagnostic: Optional[torch.Tensor] = None,
        normalizer=None,
        sampler_num_steps: "Optional[int | Sequence[int]]" = None,
        seed: int = 0,
        split_ensemble_across_ranks: bool = False,
        scorers: Sequence = (),
        on_frame_scored: Optional[Callable[[int], None]] = None,
    ):
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be ≥ 1")
        log_steps = sorted({int(s) for s in log_steps})
        if not log_steps or log_steps[0] < 1:
            raise ValueError("log_steps must be a non-empty list of positive ints")

        self.dataset = dataset
        self.wrapper = wrapper
        self.scheduler = inference_scheduler
        self.log_steps = log_steps
        self.device = device
        # ``ensemble_size`` is always the TOTAL ensemble — the config-facing
        # number that gates spread metrics, the envelope's E>1 check and the
        # default perturber. Under ``split_ensemble_across_ranks`` the members
        # are divided EVENLY across the distributed ranks: every rank rolls
        # the SAME initial conditions with ``local_ensemble_size`` members,
        # and the per-step ensemble statistics are completed by cross-rank
        # reductions (see _cross_rank_ensemble_mean/_var). All ranks must run
        # in lockstep — same ICs, same horizon — or those collectives desync.
        self.ensemble_size = ensemble_size
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank, self._world_size = 0, 1
        self.member_split = bool(split_ensemble_across_ranks) and self._world_size > 1
        if self.member_split:
            if ensemble_size % self._world_size != 0:
                raise ValueError(
                    f"split_ensemble_across_ranks requires the total "
                    f"ensemble_size ({ensemble_size}) to divide evenly over "
                    f"the {self._world_size} ranks — members must be spread "
                    f"evenly or the cross-rank mean is wrong."
                )
            self.local_ensemble_size = ensemble_size // self._world_size
        else:
            self.local_ensemble_size = ensemble_size
        self.perturber = perturber or (
            Deterministic() if ensemble_size == 1 else ReplicateOnly()
        )
        # An explicit Deterministic perturber with a real ensemble would
        # normally raise at call time — but under member-split with
        # E == world_size each rank replicates by local_E == 1, so it would
        # NOT raise and every "member" would be identical: a silent
        # zero-spread ensemble. Refuse at construction instead.
        if ensemble_size > 1 and isinstance(self.perturber, Deterministic):
            raise ValueError(
                f"ensemble_size={ensemble_size} with a Deterministic "
                f"perturber would make every member identical; use "
                f"replicate_only (scheduler noise only) or gaussian_ic."
            )
        self.has_diagnostic = has_diagnostic
        self.batch_size = max(1, int(batch_size))
        self.max_initial_conditions = max(1, int(max_initial_conditions))
        self.ic_stride = max(1, int(ic_stride))
        # Store rows per MODEL step (see validate.py's RolloutValidator and
        # datapipes.climate.resolve_step_stride). Every index below — the IC
        # bound, the oracle window's past frames, the forcing trajectory and
        # the scoring targets — is in model steps, not rows: the AMIP archives
        # are 6-hourly under a 24-hour step, so a stride of 1 would score a
        # 1-step forecast against truth 6 hours out and roll the model through
        # forcings 4x too fast.
        self.step_size = max(1, int(step_size))
        self.normalizer = normalizer
        self.sampler_num_steps = sampler_num_steps
        self.seed = int(seed)
        # Scorer plug-ins (the fused eval suite): each receives the shared
        # StepContext per (frame, kind) — the drive performs the rollout, the
        # ensemble reductions and the denormalization exactly once regardless
        # of how many scorers ride it. Empty by default: the mid-training
        # validation path is unchanged.
        self.scorers = list(scorers)
        self.on_frame_scored = on_frame_scored

        # Dispatch on scheduler: rolling = has sample_rollout.
        self.window_mode = hasattr(self.scheduler, "sample_rollout")
        self.window_size = (
            int(getattr(self.scheduler, "window_size", 0))
            if self.window_mode
            else 0
        )
        # Frames the scheduler wants for its first window. ERDM/RFM noise the
        # true W-frame window onto their t=0 staircase; a data-coupled
        # scheduler (RSI) additionally needs the frame BEFORE the window, as
        # slot 1's interpolant anchor. Defaulting to window_size leaves every
        # existing scheduler on exactly its old path.
        self.init_frames = (
            int(getattr(self.scheduler, "init_frames", self.window_size))
            if self.window_mode
            else 0
        )

        # Horizon default: training W for rolling, last log_step for
        # single-step. Either way, log_steps[-1] must fit.
        if horizon is None:
            horizon = self.window_size if self.window_mode else log_steps[-1]
        self.horizon = int(horizon)
        if log_steps[-1] > self.horizon:
            raise ValueError(
                f"max(log_steps)={log_steps[-1]} exceeds horizon={self.horizon}"
            )
        if isinstance(sampler_num_steps, (list, tuple)):
            if len(sampler_num_steps) != self.horizon:
                raise ValueError(
                    f"sampler_num_steps schedule has length "
                    f"{len(sampler_num_steps)}, expected horizon={self.horizon}"
                )
            self.sampler_num_steps = [int(s) for s in sampler_num_steps]

        # Derive grid + channel layout from a probe sample.
        sample = dataset[0]
        self.n_surface = sample["surface_in"].shape[0]
        self.n_lat = sample["surface_in"].shape[-2]
        self.has_upper_air = "upper_air_in" in sample
        self.n_upper_var = (
            sample["upper_air_in"].shape[0] if self.has_upper_air else 0
        )

        lat_w = cos_lat_weights(self.n_lat, device, torch.float32)
        self.register_lat = lat_w

        # Streaming metrics.
        n_log = len(log_steps)
        self.rmse_surface = StreamingLatWeightedRMSE(
            n_steps=n_log, n_channels=self.n_surface, device=device
        )
        self.rmse_upper_air = (
            StreamingLatWeightedRMSE(
                n_steps=n_log, n_channels=self.n_upper_var, device=device
            )
            if self.has_upper_air
            else None
        )
        self.rmse_diagnostic = None
        if has_diagnostic and "diagnostic" in sample:
            self.rmse_diagnostic = StreamingLatWeightedRMSE(
                n_steps=n_log,
                n_channels=sample["diagnostic"].shape[0],
                device=device,
            )

        # Spread metrics (only when ensemble_size > 1).
        self.spread_surface = None
        self.spread_upper_air = None
        self.spread_diagnostic = None
        if ensemble_size > 1:
            self.spread_surface = StreamingLatWeightedSpread(
                n_steps=n_log, n_channels=self.n_surface, device=device
            )
            if self.has_upper_air:
                self.spread_upper_air = StreamingLatWeightedSpread(
                    n_steps=n_log, n_channels=self.n_upper_var, device=device
                )
            if self.rmse_diagnostic is not None:
                self.spread_diagnostic = StreamingLatWeightedSpread(
                    n_steps=n_log,
                    n_channels=self.rmse_diagnostic.sum_sq_w.shape[1],
                    device=device,
                )

        # Optional ACC metrics.
        self.acc_surface = None
        self.acc_upper_air = None
        self.acc_diagnostic = None
        if climatology_surface is not None:
            self.acc_surface = StreamingLatWeightedACC(
                n_steps=n_log,
                n_channels=self.n_surface,
                climatology=climatology_surface,
                device=device,
            )
        if climatology_upper_air is not None and self.has_upper_air:
            self.acc_upper_air = StreamingLatWeightedACC(
                n_steps=n_log,
                n_channels=self.n_upper_var,
                climatology=climatology_upper_air,
                device=device,
            )
        if climatology_diagnostic is not None and self.rmse_diagnostic is not None:
            self.acc_diagnostic = StreamingLatWeightedACC(
                n_steps=n_log,
                n_channels=self.rmse_diagnostic.sum_sq_w.shape[1],
                climatology=climatology_diagnostic,
                device=device,
            )

        for scorer in self.scorers:
            bind = getattr(scorer, "bind", None)
            if bind is not None:
                bind(self)

    # ------------------------------------------------------------------ #
    # IC selection — identical contract to deterministic RolloutValidator.
    # ------------------------------------------------------------------ #

    def _select_ic_indices(self, rank: int, world_size: int) -> list[int]:
        # The maximum admissible IC index depends on the dispatch path.
        # Single-step needs ``horizon`` future frames after the IC. Window
        # mode reads NOTHING before the IC (the oracle window is the FUTURE
        # window y_{1:W}; RSI's extra anchor frame is the IC itself) but
        # reaches further forward: the init window ends at t + W, the forcing
        # trajectory at t + (W + horizon - 2 + nocean), the scored truth at
        # t + horizon.
        if self.window_mode:
            nocean = int(bool(getattr(self.scheduler, "nocean", 0)))
            last_future = self.step_size * max(
                self.horizon,
                self.window_size,
                self.window_size + self.horizon - 2 + nocean,
            )
            first_past = 0
        else:
            last_future = self.horizon * self.step_size
            first_past = 0
        max_idx = self.dataset.n_time - last_future - 1
        candidates = list(range(first_past, max_idx + 1, self.ic_stride))
        if not candidates:
            # Refuse rather than score nothing (2026-08-14). An empty IC list
            # ran zero samples and reported RMSE 0.0 — an eval suite claiming a
            # perfect model because it never evaluated one. The shipped
            # eval_suite horizon (1460, a 6-hourly year) does exactly this on a
            # one-year store at the AMIP 24-hour step.
            reach_past_horizon = last_future // max(1, self.step_size) - self.horizon
            max_horizon = (
                self.dataset.n_time - 1
            ) // max(1, self.step_size) - reach_past_horizon
            raise ValueError(
                f"no admissible initial condition: horizon={self.horizon} x "
                f"step_size={self.step_size} needs {last_future} future rows "
                f"but the store has {self.dataset.n_time}. Largest horizon "
                f"this store supports is {max(0, max_horizon)}."
            )
        candidates = candidates[: self.max_initial_conditions]
        if self.member_split:
            # Member-parallel mode: every rank rolls the SAME ICs (with its
            # own slice of the ensemble); the per-step cross-rank reductions
            # require lockstep, so the rank-modulo IC split must NOT apply —
            # it would average members of different ICs into one "mean".
            return candidates
        return [c for i, c in enumerate(candidates) if i % world_size == rank]

    # ------------------------------------------------------------------ #
    # Stacking + normalization plumbing.
    # ------------------------------------------------------------------ #

    def _fetch(self, t: int) -> dict[str, torch.Tensor]:
        # ClimateZarrDataset indexes by either ``t`` or ``(t, lead)``.
        # Lead is irrelevant for the diffusion validator — we only consume
        # ``surface_in / upper_air_in / diagnostic / constant_boundary /
        # varying_boundary / calendar`` at time ``t``.
        try:
            return self.dataset[(int(t), 1)]
        except (TypeError, KeyError):
            return self.dataset[int(t)]

    def _stack(self, t_list: list[int]) -> dict[str, torch.Tensor]:
        samples = [self._fetch(t) for t in t_list]
        out: dict[str, torch.Tensor] = {}
        for k, v0 in samples[0].items():
            if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
                out[k] = torch.stack([s[k] for s in samples], dim=0)
            elif isinstance(v0, torch.Tensor):
                out[k] = torch.stack([s[k] for s in samples], dim=0)
            else:
                out[k] = v0
        return out

    def _to_device(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

    def _num_steps_for_frame(self, k: int) -> Optional[int]:
        """Resolve ``sampler_num_steps`` for emitted frame ``k`` (1-indexed)."""
        if isinstance(self.sampler_num_steps, list):
            return self.sampler_num_steps[k - 1]
        return self.sampler_num_steps

    def _denorm_pred_truth(
        self, kind: str, pred: torch.Tensor, truth: torch.Tensor
    ):
        if self.normalizer is None:
            return pred, truth
        # ClimateNormalizer.denormalize_state expects kwargs by channel
        # group; mirror the deterministic validator's contract.
        pred_phys = self.normalizer.denormalize_state(**{kind: pred})[kind]
        truth_phys = self.normalizer.denormalize_state(**{kind: truth})[kind]
        return pred_phys, truth_phys

    # ------------------------------------------------------------------ #
    # Public entry.
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def run(self, model, *, epoch: int = 0) -> dict[str, float]:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1

        ic_indices = self._select_ic_indices(rank, world_size)
        try:
            gen = torch.Generator(device=self.device).manual_seed(
                self._generator_seed(epoch)
            )
        except (RuntimeError, TypeError):
            gen = torch.Generator(device="cpu").manual_seed(
                self._generator_seed(epoch)
            )

        log_step_to_idx = {s: i for i, s in enumerate(self.log_steps)}

        for batch_start in range(0, len(ic_indices), self.batch_size):
            batch_ics = ic_indices[batch_start : batch_start + self.batch_size]
            if not batch_ics:
                continue
            if self.window_mode:
                self._rollout_window(model, batch_ics, gen, log_step_to_idx)
            else:
                self._rollout_single_step(model, batch_ics, gen, log_step_to_idx)

        return self._finalize()

    # ------------------------------------------------------------------ #
    # Single-step diffusion rollout (DriftScheduler / DynamicInterpolant).
    # ------------------------------------------------------------------ #

    def _generator_seed(self, epoch: int) -> int:
        """Seed for the run()'s Generator (GaussianIC perturbations).

        Rank-offset ONLY in member-split mode: there the ranks hold different
        members of the same ICs, so their perturbations must differ. In
        IC-split mode ranks hold different ICs and keep the shared seed, as
        they always have.
        """
        return (
            self.seed
            + epoch * 100003
            + (self._rank * 7919 if self.member_split else 0)
        )

    def _cross_rank_ensemble_mean(self, pred_ensemble: torch.Tensor) -> torch.Tensor:
        """``(n_ic * local_E, ...)`` -> ``(n_ic, ...)`` mean over ALL members.

        Local mean over ``local_ensemble_size``, then — member-split only —
        ``all_reduce(SUM)`` divided by ``world_size``, which is exact because
        every rank holds the same number of members (the divisibility guard).
        SUM+divide rather than ``ReduceOp.AVG`` because gloo has no AVG.
        Collective-free (and byte-identical to the old inline reshape-mean)
        when member_split is off.
        """
        if self.local_ensemble_size > 1:
            rest = pred_ensemble.shape[1:]
            n_ic = pred_ensemble.shape[0] // self.local_ensemble_size
            pred_mean = pred_ensemble.view(
                n_ic, self.local_ensemble_size, *rest
            ).mean(dim=1)
        elif self.member_split:
            # all_reduce is in-place and the E==1 "mean" is the input itself,
            # which is a live VIEW of the marching rollout state — reducing it
            # in place would corrupt the trajectory on every rank.
            pred_mean = pred_ensemble.clone()
        else:
            return pred_ensemble
        if self.member_split:
            dist.all_reduce(pred_mean, op=dist.ReduceOp.SUM)
            pred_mean /= self._world_size
        return pred_mean

    def _cross_rank_ensemble_var(
        self, pred_ensemble: torch.Tensor, global_mean: torch.Tensor
    ) -> torch.Tensor:
        """Per-pixel ensemble variance over the member UNION across ranks.

        Squared deviations from the (already cross-rank) ensemble mean,
        summed over the local members, all-reduced, divided by the TOTAL
        ensemble size — exactly ``var(unbiased=False)`` over all members.
        """
        rest = pred_ensemble.shape[1:]
        n_ic = pred_ensemble.shape[0] // self.local_ensemble_size
        dev = (
            pred_ensemble.float().view(n_ic, self.local_ensemble_size, *rest)
            - global_mean.float().unsqueeze(1)
        )
        sq = dev.pow(2).sum(dim=1)
        if self.member_split:
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        return sq / float(self.ensemble_size)

    def _build_ctx(
        self,
        m_idx: int,
        pred_ensemble: torch.Tensor,
        truth: torch.Tensor,
        kind: str,
    ) -> StepContext:
        """Assemble the shared per-(step, kind) context — reductions ONCE.

        The cross-rank ensemble mean (a collective under member-split), the
        member-union variance and the denormalization each happen exactly one
        time here, however many scorers consume the result. Collective
        symmetry across ranks follows from every rank building the same
        contexts in the same order.
        """
        pred_mean = self._cross_rank_ensemble_mean(pred_ensemble)
        # Union variance: exact in BOTH split modes (sq-dev sum / total E ==
        # var(unbiased=False) when world_size == 1 too). Only computed when an
        # ensemble exists — it is what the spread metrics consume.
        pred_var = (
            self._cross_rank_ensemble_var(pred_ensemble, pred_mean)
            if self.ensemble_size > 1
            else None
        )
        pred_phys, truth_phys = self._denorm_pred_truth(kind, pred_mean, truth)
        return StepContext(
            step=self.log_steps[m_idx],
            m_idx=m_idx,
            kind=kind,
            pred_ensemble=pred_ensemble,
            pred_mean=pred_mean,
            pred_var=pred_var,
            pred_phys=pred_phys,
            truth=truth,
            truth_phys=truth_phys,
            lat_weights=self.register_lat,
            ensemble_size=self.ensemble_size,
            local_ensemble_size=self.local_ensemble_size,
        )

    def _update_base_metrics(self, ctx: StepContext) -> None:
        """RMSE / ACC / spread from the shared context."""
        rmse = getattr(self, f"rmse_{ctx.kind}", None)
        if rmse is not None:
            rmse.update(ctx.m_idx, ctx.pred_phys, ctx.truth_phys, ctx.lat_weights)
        acc = getattr(self, f"acc_{ctx.kind}", None)
        if acc is not None:
            acc.update(ctx.m_idx, ctx.pred_mean, ctx.truth, ctx.lat_weights)
        spread = getattr(self, f"spread_{ctx.kind}", None)
        if spread is not None and ctx.pred_var is not None:
            # Member-UNION variance in both split modes — a rank-local var
            # over a subset would be biased low and average wrongly.
            spread.update_from_var(ctx.m_idx, ctx.pred_var, ctx.lat_weights)

    def _score_step(
        self,
        m_idx: int,
        pred_ensemble: torch.Tensor,
        truth: torch.Tensor,
        kind: str,
    ) -> None:
        """One (log_step, channel group): base metrics + every scorer."""
        ctx = self._build_ctx(m_idx, pred_ensemble, truth, kind)
        self._update_base_metrics(ctx)
        for scorer in self.scorers:
            scorer.score_step(ctx)

    def _rollout_single_step(
        self,
        model,
        batch_ics: list[int],
        gen: torch.Generator,
        log_step_to_idx: dict[int, int],
    ) -> None:
        # Initial dataset sample at each IC, on device + normalized.
        init = self._to_device(self._stack(batch_ics))
        state = self.perturber(init, self.local_ensemble_size, generator=gen)
        n_ic = len(batch_ics)
        const_boundary = state.get("constant_boundary")
        # Stateful steppers (the deterministic adapter's prev-frame memory)
        # reset per IC batch; schedulers without the hook are untouched.
        if hasattr(self.scheduler, "on_rollout_start"):
            self.scheduler.on_rollout_start(state)

        wrapper = self.wrapper.module if hasattr(self.wrapper, "module") else self.wrapper

        x = wrapper.pack_state(state)

        for k in range(1, self.horizon + 1):
            # Build c_grid / c_scalar at the *input* time t + (k - 1).
            c_grid = wrapper.pack_c_grid(state)
            c_scalar = state["calendar"]

            # Diffusion sample → next-step prediction (still normalized).
            x_next = self.scheduler.sample(
                model, x, c_grid, c_scalar, num_steps=self._num_steps_for_frame(k)
            )
            # DynamicInterpolant (SI_X) returns ``(y, model_last_pred)`` under
            # its ``return_model_last=True`` default; DriftScheduler returns a
            # bare tensor. Take the sample either way — same unwrap as
            # ``CombinedModule.forward``. Without it every SI_X rollout dies on
            # ``'tuple' object has no attribute 'narrow'`` inside unpack_state.
            if isinstance(x_next, tuple):
                x_next = x_next[0]

            # ONE dataset fetch of the t+k frames serves both scoring and the
            # boundary/calendar advance (they used to be fetched twice).
            frame_k = None
            if k in log_step_to_idx or k < self.horizon:
                frame_times = [t + k * self.step_size for t in batch_ics]
                frame_k = self._to_device(self._stack(frame_times))

            # Score this step (if requested) against the dataset's frame at t+k.
            if k in log_step_to_idx:
                m_idx = log_step_to_idx[k]
                target = frame_k
                unpacked = wrapper.unpack_state(x_next)
                self._score_step(m_idx, unpacked["surface_in"], target["surface_in"], "surface")
                if self.has_upper_air and "upper_air_in" in unpacked:
                    self._score_step(
                        m_idx,
                        unpacked["upper_air_in"],
                        target["upper_air_in"],
                        "upper_air",
                    )
                if self.has_diagnostic and "diagnostic" in unpacked and "diagnostic" in target:
                    self._score_step(
                        m_idx, unpacked["diagnostic"], target["diagnostic"], "diagnostic"
                    )
                if self.on_frame_scored is not None:
                    self.on_frame_scored(k)

            # Advance: next state's surface/upper_air/diag come from the
            # diffusion sample. Boundary + calendar march to the next step
            # using the dataset sample at t+k.
            if k < self.horizon:
                next_step = frame_k
                next_var_boundary = next_step["varying_boundary"]
                next_calendar = next_step["calendar"]
                if self.local_ensemble_size > 1:
                    next_var_boundary = next_var_boundary.repeat_interleave(
                        self.local_ensemble_size, dim=0
                    )
                    next_calendar = next_calendar.repeat_interleave(
                        self.local_ensemble_size, dim=0
                    )
                unpacked = wrapper.unpack_state(x_next)
                state = {
                    "surface_in": unpacked["surface_in"],
                    "constant_boundary": const_boundary,
                    "varying_boundary": next_var_boundary,
                    "calendar": next_calendar,
                }
                if self.has_upper_air:
                    state["upper_air_in"] = unpacked["upper_air_in"]
                if self.has_diagnostic and "diagnostic" in unpacked:
                    state["diagnostic"] = unpacked["diagnostic"]
                x = x_next

    # ------------------------------------------------------------------ #
    # Window-rollout diffusion (RFM / ERDM).
    # ------------------------------------------------------------------ #

    def _stack_window(
        self, batch_ics: list[int], w_offset: int, n_frames: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Stack an (B, n, ...) window batch ending at ``t + w_offset`` steps.

        ``w_offset`` and the intra-window spacing are both in MODEL steps, so
        the frames land ``step_size`` store rows apart, the last one on
        ``t + w_offset``. ``n_frames`` defaults to the scheduler's
        ``init_frames`` (= W for ERDM/RFM, W+1 for RSI, whose leading frame
        is slot 1's anchor): at ``w_offset = W`` the ERDM window is the
        future oracle y_{1:W} and RSI's extra frame is the anchor y_0 = the
        IC itself.
        """
        n_frames = int(n_frames if n_frames is not None else self.init_frames)
        per_batch_windows = []
        for t in batch_ics:
            frames = [
                self._fetch(
                    t + (w_offset - n_frames + 1 + i) * self.step_size
                )
                for i in range(n_frames)
            ]
            # Stack frames into a (W, ...) per-batch dict.
            window = {}
            for k, v0 in frames[0].items():
                if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
                    window[k] = torch.stack([f[k] for f in frames], dim=0)
                elif isinstance(v0, torch.Tensor):
                    window[k] = torch.stack([f[k] for f in frames], dim=0)
                else:
                    window[k] = v0
            per_batch_windows.append(window)
        # Stack over batch axis → (B, W, ...).
        out: dict[str, torch.Tensor] = {}
        for k in per_batch_windows[0]:
            v0 = per_batch_windows[0][k]
            if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
                out[k] = torch.stack([w[k] for w in per_batch_windows], dim=0)
            else:
                out[k] = v0
        return out

    def _rollout_window(
        self,
        model,
        batch_ics: list[int],
        gen: torch.Generator,
        log_step_to_idx: dict[int, int],
    ) -> None:
        wrapper = self.wrapper.module if hasattr(self.wrapper, "module") else self.wrapper

        # Oracle init window: the schedulers' contract is the FUTURE window
        # y_{1:W} (erdm.py sample_rollout: "oracle true first window
        # y_{1:W}"; emit k is scored against t + k below, which only lines
        # up when the first window really holds t+1 .. t+W). RSI
        # (init_frames = W + 1) additionally wants the anchor y_0 — the IC
        # frame itself — so the stack ends at t + W and reaches back
        # init_frames, leaving the anchor at t unmoved.
        init_window = self._to_device(
            self._stack_window(batch_ics, w_offset=self.window_size)
        )
        init_window_ens = self.perturber(
            init_window, self.local_ensemble_size, generator=gen
        )
        init_y = wrapper.pack_window_state(init_window_ens)  # (B*E, W, C, H, W)

        # Build the trajectory of forcings + scalars over the horizon.
        # Trajectory slot i is the forcing at absolute step i (store row
        # t + i * step_size): the scheduler's roll k conditions window slot
        # w (holding state y_{k+w+1}) on traj[k+w], giving every state its
        # LAG-1 forcing — the training alignment (sequence.py forcing_lag=1,
        # forcing frame j conditions state frame j+1). Slots therefore span
        # [t, t + (W + horizon - 2) * step_size].
        # Phase 12f: predicted ocean channels are imposed from the boundary at
        # each window slot's OWN time, which reaches one step past the last
        # forcing the model is conditioned on. Without this extra frame
        # ``_gather_window`` would clamp and the final roll would be imposed
        # from a stale SST — silently, since the shapes still line up.
        traj_len = (
            self.window_size
            + self.horizon
            - 1
            + int(bool(getattr(self.scheduler, "nocean", 0)))
        )
        traj_frames = [
            self._to_device(
                self._stack([t + i * self.step_size for t in batch_ics])
            )
            for i in range(traj_len)
        ]

        def _stack_traj(key):
            xs = [f[key] for f in traj_frames]
            return torch.stack(xs, dim=1)  # (B, T, ...)

        const_boundary = traj_frames[0]["constant_boundary"]
        c_grid_traj = wrapper.pack_window_c_grid(
            {
                "surface_in": _stack_traj("surface_in"),
                "constant_boundary": const_boundary,
                "varying_boundary": _stack_traj("varying_boundary"),
            }
        )
        c_scalar_traj = _stack_traj("calendar")
        if self.local_ensemble_size > 1:
            c_grid_traj = c_grid_traj.repeat_interleave(
                self.local_ensemble_size, dim=0
            )
            c_scalar_traj = c_scalar_traj.repeat_interleave(
                self.local_ensemble_size, dim=0
            )

        if hasattr(self.scheduler, "on_rollout_start"):
            self.scheduler.on_rollout_start(init_window_ens)
        traj = self.scheduler.sample_rollout(
            model,
            init_y,
            c_grid_traj,
            c_scalar_traj,
            horizon=self.horizon,
            num_steps=self.sampler_num_steps,
        )
        # traj is (B*E, horizon, C, H, W) of packed flat channels.

        # Score each requested log_step against the dataset frame at t+k.
        for k in range(1, self.horizon + 1):
            if k not in log_step_to_idx:
                continue
            m_idx = log_step_to_idx[k]
            x_k = traj[:, k - 1]
            unpacked = wrapper.unpack_state(x_k)
            target = self._to_device(
                self._stack([t + k * self.step_size for t in batch_ics])
            )
            self._score_step(m_idx, unpacked["surface_in"], target["surface_in"], "surface")
            if self.has_upper_air and "upper_air_in" in unpacked:
                self._score_step(
                    m_idx, unpacked["upper_air_in"], target["upper_air_in"], "upper_air"
                )
            if self.has_diagnostic and "diagnostic" in unpacked and "diagnostic" in target:
                self._score_step(
                    m_idx, unpacked["diagnostic"], target["diagnostic"], "diagnostic"
                )
            if self.on_frame_scored is not None:
                self.on_frame_scored(k)

    # ------------------------------------------------------------------ #
    # Finalize.
    # ------------------------------------------------------------------ #

    def _finalize(self, *, local_only: bool = False) -> dict[str, float]:
        """Flat metric dict. ``local_only=True`` skips the cross-rank
        collectives (rank-local partial view for progress snapshots — the
        metric finalizes are clone-based and repeatable, so the eventual
        collective finalize is unaffected)."""
        results: dict[str, float] = {}

        def _emit(prefix: str, metric, group: str):
            if metric is None:
                return
            vals = metric.finalize(local_only=local_only)  # (n_steps, n_channels)
            per_step = vals.mean(dim=1)
            for i, step in enumerate(self.log_steps):
                results[f"{prefix}_step{step}_{group}"] = float(per_step[i].item())

        _emit("rmse", self.rmse_surface, "surface")
        _emit("rmse", self.rmse_upper_air, "upper_air")
        _emit("rmse", self.rmse_diagnostic, "diagnostic")
        _emit("acc", self.acc_surface, "surface")
        _emit("acc", self.acc_upper_air, "upper_air")
        _emit("acc", self.acc_diagnostic, "diagnostic")
        _emit("spread", self.spread_surface, "surface")
        _emit("spread", self.spread_upper_air, "upper_air")
        _emit("spread", self.spread_diagnostic, "diagnostic")
        return results


__all__ = [
    "DiffusionRolloutValidator",
    "StreamingLatWeightedSpread",
]
