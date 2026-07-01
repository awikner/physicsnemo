# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Optional, Sequence

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
        assert pred_ensemble.shape[0] % ensemble_size == 0, (
            f"batch dim {pred_ensemble.shape[0]} not divisible by "
            f"ensemble_size {ensemble_size}"
        )
        n_ic = pred_ensemble.shape[0] // ensemble_size
        var = (
            pred_ensemble.float()
            .view(n_ic, ensemble_size, *rest)
            .var(dim=1, unbiased=False)
        )  # → (B, C, [L,] H, W)
        weight_shape = [1] * var.ndim
        weight_shape[-2] = lat_weights.shape[0]
        w = lat_weights.view(weight_shape)
        # Reduce over batch + level + spatial → (C,)
        reduce_dims = [d for d in range(var.ndim) if d != 1]
        self.sum_var_w[step_index] += (var * w).sum(dim=reduce_dims).detach()
        self.weight_total[step_index] += (
            w.expand_as(var).sum(dim=reduce_dims).detach()
        )

    def finalize(self) -> torch.Tensor:
        _all_reduce_sum(self.sum_var_w)
        _all_reduce_sum(self.weight_total)
        return torch.sqrt(
            self.sum_var_w / self.weight_total.clamp(min=1e-12)
        )


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
        climatology_surface: Optional[torch.Tensor] = None,
        climatology_upper_air: Optional[torch.Tensor] = None,
        climatology_diagnostic: Optional[torch.Tensor] = None,
        normalizer=None,
        sampler_num_steps: "Optional[int | Sequence[int]]" = None,
        seed: int = 0,
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
        self.ensemble_size = ensemble_size
        self.perturber = perturber or (
            Deterministic() if ensemble_size == 1 else ReplicateOnly()
        )
        self.has_diagnostic = has_diagnostic
        self.batch_size = max(1, int(batch_size))
        self.max_initial_conditions = max(1, int(max_initial_conditions))
        self.ic_stride = max(1, int(ic_stride))
        self.normalizer = normalizer
        self.sampler_num_steps = sampler_num_steps
        self.seed = int(seed)

        # Dispatch on scheduler: rolling = has sample_rollout.
        self.window_mode = hasattr(self.scheduler, "sample_rollout")
        self.window_size = (
            int(getattr(self.scheduler, "window_size", 0))
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

    # ------------------------------------------------------------------ #
    # IC selection — identical contract to deterministic RolloutValidator.
    # ------------------------------------------------------------------ #

    def _select_ic_indices(self, rank: int, world_size: int) -> list[int]:
        # The maximum admissible IC index depends on the dispatch path:
        # single-step needs ``horizon`` future frames after the IC;
        # window mode needs ``W - 1`` past frames before *and* ``horizon``
        # future frames after.
        last_future = self.horizon
        first_past = self.window_size - 1 if self.window_mode else 0
        max_idx = self.dataset.n_time - last_future - 1
        candidates = list(range(first_past, max_idx + 1, self.ic_stride))
        candidates = candidates[: self.max_initial_conditions]
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
                self.seed + epoch * 100003
            )
        except (RuntimeError, TypeError):
            gen = torch.Generator(device="cpu").manual_seed(
                self.seed + epoch * 100003
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

    def _score_step(
        self,
        m_idx: int,
        pred_ensemble: torch.Tensor,
        truth: torch.Tensor,
        kind: str,
    ) -> None:
        """Update RMSE / ACC / spread for one (log_step, channel group)."""
        n_ic = pred_ensemble.shape[0] // max(self.ensemble_size, 1)
        if self.ensemble_size > 1:
            rest = pred_ensemble.shape[1:]
            pred_mean = (
                pred_ensemble.view(n_ic, self.ensemble_size, *rest).mean(dim=1)
            )
        else:
            pred_mean = pred_ensemble

        # RMSE in physical units.
        pred_phys, truth_phys = self._denorm_pred_truth(kind, pred_mean, truth)
        rmse = getattr(self, f"rmse_{kind}", None)
        if rmse is not None:
            rmse.update(m_idx, pred_phys, truth_phys, self.register_lat)
        acc = getattr(self, f"acc_{kind}", None)
        if acc is not None:
            acc.update(m_idx, pred_mean, truth, self.register_lat)
        spread = getattr(self, f"spread_{kind}", None)
        if spread is not None:
            spread.update(
                m_idx, pred_ensemble, self.register_lat, self.ensemble_size
            )

    def _rollout_single_step(
        self,
        model,
        batch_ics: list[int],
        gen: torch.Generator,
        log_step_to_idx: dict[int, int],
    ) -> None:
        # Initial dataset sample at each IC, on device + normalized.
        init = self._to_device(self._stack(batch_ics))
        state = self.perturber(init, self.ensemble_size, generator=gen)
        n_ic = len(batch_ics)
        const_boundary = state.get("constant_boundary")

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

            # Score this step (if requested) against the dataset's frame at t+k.
            if k in log_step_to_idx:
                m_idx = log_step_to_idx[k]
                target_times = [t + k for t in batch_ics]
                target = self._to_device(self._stack(target_times))
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

            # Advance: next state's surface/upper_air/diag come from the
            # diffusion sample. Boundary + calendar march to the next step
            # using the dataset sample at t+k.
            if k < self.horizon:
                next_times = [t + k for t in batch_ics]
                next_step = self._to_device(self._stack(next_times))
                next_var_boundary = next_step["varying_boundary"]
                next_calendar = next_step["calendar"]
                if self.ensemble_size > 1:
                    next_var_boundary = next_var_boundary.repeat_interleave(
                        self.ensemble_size, dim=0
                    )
                    next_calendar = next_calendar.repeat_interleave(
                        self.ensemble_size, dim=0
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
        self, batch_ics: list[int], w_offset: int
    ) -> dict[str, torch.Tensor]:
        """Stack a (B, W, ...) window batch ending at (t + w_offset)."""
        per_batch_windows = []
        for t in batch_ics:
            frames = [
                self._fetch(t + w_offset - self.window_size + 1 + i)
                for i in range(self.window_size)
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

        # Initial oracle window ending at t (so frames span t - W + 1 .. t).
        init_window = self._to_device(self._stack_window(batch_ics, w_offset=0))
        init_window_ens = self.perturber(
            init_window, self.ensemble_size, generator=gen
        )
        init_y = wrapper.pack_window_state(init_window_ens)  # (B*E, W, C, H, W)

        # Build the trajectory of forcings + scalars over the horizon
        # (the rolling sampler advances one frame at a time; it needs
        # forcings at each emitted frame's input slot, i.e. across
        # [t - W + 1, t + horizon - 1] inclusive — same W-frame window
        # shifted right by k for the k-th emit).
        traj_len = self.window_size + self.horizon - 1
        traj_frames = [
            self._to_device(
                self._stack(
                    [t - self.window_size + 1 + i for t in batch_ics]
                )
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
        if self.ensemble_size > 1:
            c_grid_traj = c_grid_traj.repeat_interleave(self.ensemble_size, dim=0)
            c_scalar_traj = c_scalar_traj.repeat_interleave(
                self.ensemble_size, dim=0
            )

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
                self._stack([t + k for t in batch_ics])
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

    # ------------------------------------------------------------------ #
    # Finalize.
    # ------------------------------------------------------------------ #

    def _finalize(self) -> dict[str, float]:
        results: dict[str, float] = {}

        def _emit(prefix: str, metric, group: str):
            if metric is None:
                return
            vals = metric.finalize()  # (n_steps, n_channels)
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
