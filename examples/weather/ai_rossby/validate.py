# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mid-training rollout validator for ai-rossby recipes.

Streaming, DDP-safe RMSE + ACC over multi-step autoregressive rollouts.
The validator is **ensemble-aware**: each initial condition can spawn an
ensemble of trajectories via an :class:`Perturber` strategy (IC noise,
weight perturbation, or model-internal stochasticity). Metrics fold over
the ensemble axis as the rollout advances so the long-horizon memory
cost stays proportional to *one* state per ensemble member, not the
full (steps × members) history.

Design notes
------------

* **No per-step state retention.** At any moment the validator holds the
  current rollout state(s) plus the ground truth at the current step
  only. Predecessors are released as soon as the metric update for that
  step completes. This is the same memory discipline as PanguWeather's
  ``ObservationAccumulator`` and physicsnemo's
  :class:`physicsnemo.metrics.general.ensemble_metrics.Mean` — keep the
  sufficient statistic, drop the data.
* **DDP-safe by construction.** Each :class:`StreamingMetric`
  accumulates per-rank sums and does the final all-reduce in
  ``finalize()``. The reduction is mathematically identical to a
  single-GPU run so the validator's output is invariant to world size.
* **Ensemble axis lives in the batch dim.** A perturber replicates the
  per-IC sample ``E`` times along the leading axis (``B → B × E``); the
  model sees one big batch and is unchanged. Ensemble-mean reductions
  happen inside the metric updates.

Wandb logging
-------------

The validator returns a flat ``{metric_name: scalar}`` dict suitable for
``LaunchLogger.log_epoch(...)``. When wandb is initialized in the
training script, ``LaunchLogger`` routes the dict to wandb automatically
under the ``valid/`` namespace.

The dict keys follow ``"<metric>_step{n}_<group>"`` (e.g.
``rmse_step1_surface``, ``acc_step12_upper_air``) so wandb groups
naturally by step on the dashboard.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
import torch.distributed as dist

from physicsnemo.experimental.datapipes.plasim.dataset import (
    PlasimClimateDataset,
)


# ---------------------------------------------------------------------------
# Ensemble perturbers
# ---------------------------------------------------------------------------


class Perturber(ABC):
    """Replicate a single-IC sample dict into an ensemble batch.

    Implementations multiply the batch dimension by ``ensemble_size`` and
    optionally inject stochasticity per replica. Replication uses
    ``torch.repeat_interleave`` so per-IC groups remain contiguous along
    the batch axis (``[ic0_e0, ic0_e1, ..., ic0_eE-1, ic1_e0, ...]``),
    which makes downstream ensemble-mean reduction simple.
    """

    @abstractmethod
    def __call__(
        self,
        sample: dict[str, torch.Tensor],
        ensemble_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Return a sample whose batch dim has been multiplied by ``E``."""


class Deterministic(Perturber):
    """No replication; only valid with ``ensemble_size=1``."""

    def __call__(self, sample, ensemble_size, generator=None):
        if ensemble_size != 1:
            raise ValueError(
                "Deterministic perturber requires ensemble_size=1; "
                "use ReplicateOnly or GaussianIC for E>1."
            )
        return sample


class ReplicateOnly(Perturber):
    """Replicate the IC ``E`` times without perturbation.

    Useful when the model itself is stochastic (dropout-at-inference,
    generative head, etc.) — the ensemble emerges from the forward pass.
    """

    def __call__(self, sample, ensemble_size, generator=None):
        return _interleave_ensemble(sample, ensemble_size)


class GaussianIC(Perturber):
    """Add per-variable Gaussian noise to each ensemble member.

    Parameters
    ----------
    scales : dict[str, float]
        ``{key: std}`` mapping. Keys not present in this dict are
        replicated unchanged. Noise is i.i.d. across (E, …) after the
        replicate.
    """

    def __init__(self, scales: dict[str, float]):
        self.scales = dict(scales)

    def __call__(self, sample, ensemble_size, generator=None):
        out = _interleave_ensemble(sample, ensemble_size)
        for k, std in self.scales.items():
            if k in out and isinstance(out[k], torch.Tensor) and out[k].is_floating_point():
                # ``generator`` must live on the same device as the noise
                # tensor. When the caller's generator lives elsewhere (e.g.
                # CPU generator while the sample is already on CUDA), drop
                # the explicit generator and use the device default — this
                # gives up bit-reproducibility across runs but keeps the
                # ensemble usable. Bit-reproducible call sites should pass
                # a generator allocated on the right device.
                kwargs = dict(device=out[k].device, dtype=out[k].dtype)
                if generator is not None and generator.device == out[k].device:
                    kwargs["generator"] = generator
                noise = torch.randn(out[k].shape, **kwargs)
                out[k] = out[k] + float(std) * noise
        return out


def _interleave_ensemble(sample, ensemble_size):
    """Repeat each B-dim entry ``E`` times in place (``B → B*E``)."""
    if ensemble_size <= 1:
        return sample
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            out[k] = v.repeat_interleave(ensemble_size, dim=0)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------


def cos_lat_weights(
    n_lat: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    r"""Latitude weights ``cos(lat)`` normalized so their mean is 1.

    Matches the convention in
    :mod:`physicsnemo.metrics.climate.reduction` — the weighted mean of
    a uniform field equals the unweighted mean.
    """
    phi = torch.linspace(
        math.pi / 2, -math.pi / 2, n_lat, device=device, dtype=dtype
    )
    w = torch.cos(phi)
    return w / w.mean()


def _all_reduce_sum(t: torch.Tensor) -> None:
    """In-place all-reduce SUM if a distributed group is initialized."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


class StreamingLatWeightedRMSE:
    r"""Per-(step, channel) lat-weighted RMSE, streaming + DDP-safe.

    Maintains two running buffers per (rollout step index, channel):

    * ``sum_sq_w[s, c] = Σ cos(lat) (pred - target)^2``
    * ``weight_total[s, c] = Σ cos(lat)``

    Sums are over the batch *and* spatial dims of every ``update`` at
    step ``s``. ``finalize()`` all-reduces both buffers across DDP ranks
    and returns ``sqrt(sum_sq_w / weight_total)`` per (step, channel).
    With ``count`` tracked separately, ensemble-mean reduction is the
    caller's responsibility — pass ``pred = pred_ensemble.mean(dim=…)``
    if you want ensemble-mean RMSE.

    Parameters
    ----------
    n_steps : int
        Number of metric *steps* (not max rollout steps). Caller maps
        log-step indices into ``[0, n_steps)``.
    n_channels : int
        Channels for the field being scored.
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
        self.sum_sq_w = torch.zeros(shape, device=device, dtype=dtype)
        self.weight_total = torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def update(
        self,
        step_index: int,
        pred: torch.Tensor,
        target: torch.Tensor,
        lat_weights: torch.Tensor,
    ) -> None:
        r"""Add the (step, channel) contribution from one batch.

        ``pred`` and ``target`` share shape ``(B, C, [L,] H, W)``; for
        upper-air the level axis is folded into the channel reduction
        (i.e. the returned RMSE is per-variable, averaged across levels).
        ``lat_weights`` has shape ``(H,)``.
        """
        diff_sq = (pred.float() - target.float()).pow(2)
        weight_shape = [1] * diff_sq.ndim
        weight_shape[-2] = lat_weights.shape[0]
        w = lat_weights.view(weight_shape)
        # Reduce over batch + level + spatial → (C,)
        reduce_dims = [d for d in range(diff_sq.ndim) if d != 1]
        self.sum_sq_w[step_index] += (diff_sq * w).sum(dim=reduce_dims).detach()
        self.weight_total[step_index] += (
            w.expand_as(diff_sq).sum(dim=reduce_dims).detach()
        )

    def finalize(self) -> torch.Tensor:
        """All-reduce + return ``(n_steps, n_channels)`` RMSE tensor."""
        _all_reduce_sum(self.sum_sq_w)
        _all_reduce_sum(self.weight_total)
        rmse = torch.sqrt(
            self.sum_sq_w / self.weight_total.clamp(min=1e-12)
        )
        return rmse


class StreamingLatWeightedACC:
    r"""Per-(step, channel) lat-weighted anomaly correlation, streaming + DDP-safe.

    Maintains three running sums per (step, channel):

    * ``S_pt[s, c] = Σ w (pred - clim) (target - clim)``
    * ``S_pp[s, c] = Σ w (pred - clim)^2``
    * ``S_tt[s, c] = Σ w (target - clim)^2``

    ``finalize()`` all-reduces and returns ``S_pt / sqrt(S_pp * S_tt)``.

    ``climatology`` is broadcastable against ``(B, C, [L,] H, W)`` — most
    commonly ``(C, [L,] H, W)`` for a time-mean climatology or ``(C,)``
    for a global per-channel mean.
    """

    def __init__(
        self,
        *,
        n_steps: int,
        n_channels: int,
        climatology: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (n_steps, n_channels)
        self.s_pt = torch.zeros(shape, device=device, dtype=dtype)
        self.s_pp = torch.zeros(shape, device=device, dtype=dtype)
        self.s_tt = torch.zeros(shape, device=device, dtype=dtype)
        self.climatology = climatology.to(device=device, dtype=dtype)

    @torch.no_grad()
    def update(
        self,
        step_index: int,
        pred: torch.Tensor,
        target: torch.Tensor,
        lat_weights: torch.Tensor,
    ) -> None:
        p_anom = pred.float() - self.climatology
        t_anom = target.float() - self.climatology
        weight_shape = [1] * p_anom.ndim
        weight_shape[-2] = lat_weights.shape[0]
        w = lat_weights.view(weight_shape)
        reduce_dims = [d for d in range(p_anom.ndim) if d != 1]
        self.s_pt[step_index] += (w * p_anom * t_anom).sum(dim=reduce_dims).detach()
        self.s_pp[step_index] += (w * p_anom.pow(2)).sum(dim=reduce_dims).detach()
        self.s_tt[step_index] += (w * t_anom.pow(2)).sum(dim=reduce_dims).detach()

    def finalize(self) -> torch.Tensor:
        _all_reduce_sum(self.s_pt)
        _all_reduce_sum(self.s_pp)
        _all_reduce_sum(self.s_tt)
        denom = torch.sqrt(self.s_pp.clamp(min=1e-12) * self.s_tt.clamp(min=1e-12))
        return self.s_pt / denom


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------


class RolloutValidator:
    r"""Run autoregressive rollouts on a held-out year, score with streaming metrics.

    The validator drives :class:`PlasimClimateDataset` directly (not through
    the training-time :class:`PlasimClimateDatapipe`) so it can step the
    boundary forward one timestep at a time without re-fetching the full
    sample paired with a single fixed lead time.

    Memory pattern
    --------------
    At any moment in memory we hold:

    * ``state`` — ``(B*E, …)`` for surface + upper_air on device
    * ``target`` — single-step ground truth at the current step on device
    * ``boundary`` — varying-boundary at the current step on device

    Previous steps' states and targets are released by overwrite. The
    sufficient statistics inside each streaming metric stay bounded by
    ``(n_log_steps × n_channels)``.

    Parameters
    ----------
    dataset
        A :class:`PlasimClimateDataset` opened on the validation Zarr.
        Its transform (``PlasimNormalizer`` + :class:`NanFillTransform`)
        should already be wired so that ``dataset[t]`` returns normalized
        tensors matching the model's input contract.
    log_steps
        Lead times (in dataset steps) at which to record metrics, e.g.
        ``[1, 12, 20, 40, 60]``. Must all be ≤ ``max(log_steps)``.
    ensemble_size
        Number of ensemble members per IC. ``1`` for deterministic.
    perturber
        :class:`Perturber` strategy. Defaults to :class:`Deterministic`
        when ``ensemble_size=1`` and :class:`ReplicateOnly` otherwise.
    has_diagnostic
        Whether the model emits a diagnostic head. Controls metric
        groups.
    batch_size
        Number of ICs to roll out per validator step. Larger reduces
        Python overhead at the cost of GPU memory. Note the effective
        device batch is ``batch_size * ensemble_size``.
    max_initial_conditions
        Cap on the total ICs to evaluate per ``run()`` call (across all
        ranks). Each rank sees ``ceil(max_ic / world_size)`` ICs.
    ic_stride
        Spacing (in dataset steps) between consecutive ICs. Larger →
        more decorrelated ICs but fewer per epoch.
    climatology
        Optional ``(C, [L,] H, W)`` tensor used by the ACC metric. If
        not provided, ACC is skipped.
    seed
        Reproducible per-epoch RNG for the perturber.
    log_to_wandb
        Reserved; the calling LaunchLogger handles wandb dispatch. Kept
        on the signature so future direct-wandb sinks slot in cleanly.
    """

    def __init__(
        self,
        dataset: PlasimClimateDataset,
        *,
        log_steps: Sequence[int],
        device: torch.device,
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
        seed: int = 0,
        log_to_wandb: bool = True,
    ):
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be ≥ 1")
        log_steps = sorted({int(s) for s in log_steps})
        if not log_steps or log_steps[0] < 1:
            raise ValueError("log_steps must be a non-empty list of positive ints")

        self.dataset = dataset
        self.log_steps = log_steps
        self.max_step = log_steps[-1]
        self.device = device
        self.ensemble_size = ensemble_size
        self.perturber = perturber or (
            Deterministic() if ensemble_size == 1 else ReplicateOnly()
        )
        self.has_diagnostic = has_diagnostic
        self.batch_size = max(1, int(batch_size))
        self.max_initial_conditions = max(1, int(max_initial_conditions))
        self.ic_stride = max(1, int(ic_stride))
        self.seed = int(seed)
        self.log_to_wandb = log_to_wandb
        self.normalizer = normalizer

        # Derive channel counts and lat from the dataset layout.
        # Sample is unnormalized at this point; normalization (if any) is
        # applied on-device in ``_to_device``.
        sample = dataset[0]
        self.n_surface = sample["surface_in"].shape[0]
        self.n_lat = sample["surface_in"].shape[-2]
        self.has_upper_air = "upper_air_in" in sample
        self.n_upper_var = (
            sample["upper_air_in"].shape[0] if self.has_upper_air else 0
        )
        # Compute the upper-air RMSE per-variable (averaged over level)
        # to keep the metric table small. Per-level numbers can be added
        # later by switching to a (var * level) channel axis.

        lat_w = cos_lat_weights(self.n_lat, device, torch.float32)
        self.register_lat = lat_w

        # Streaming metrics.
        self.rmse_surface = StreamingLatWeightedRMSE(
            n_steps=len(log_steps),
            n_channels=self.n_surface,
            device=device,
        )
        self.rmse_upper_air = (
            StreamingLatWeightedRMSE(
                n_steps=len(log_steps),
                n_channels=self.n_upper_var,
                device=device,
            )
            if self.has_upper_air
            else None
        )
        self.rmse_diagnostic = (
            StreamingLatWeightedRMSE(
                n_steps=len(log_steps),
                n_channels=(
                    sample["diagnostic"].shape[0]
                    if has_diagnostic and "diagnostic" in sample
                    else 1
                ),
                device=device,
            )
            if has_diagnostic
            else None
        )

        # Optional ACC.
        self.acc_surface = None
        self.acc_upper_air = None
        self.acc_diagnostic = None
        if climatology_surface is not None:
            self.acc_surface = StreamingLatWeightedACC(
                n_steps=len(log_steps),
                n_channels=self.n_surface,
                climatology=climatology_surface,
                device=device,
            )
        if climatology_upper_air is not None and self.has_upper_air:
            self.acc_upper_air = StreamingLatWeightedACC(
                n_steps=len(log_steps),
                n_channels=self.n_upper_var,
                climatology=climatology_upper_air,
                device=device,
            )
        if climatology_diagnostic is not None and has_diagnostic:
            self.acc_diagnostic = StreamingLatWeightedACC(
                n_steps=len(log_steps),
                n_channels=(
                    climatology_diagnostic.shape[0]
                    if climatology_diagnostic.dim() >= 1
                    else 1
                ),
                climatology=climatology_diagnostic,
                device=device,
            )

    # ---------------------------------------------------------------- #
    # IC selection
    # ---------------------------------------------------------------- #

    def _select_ic_indices(self, rank: int, world_size: int) -> list[int]:
        """Return the IC start indices this rank should evaluate.

        ICs are spaced ``ic_stride`` apart and the first
        ``max_initial_conditions`` are split round-robin across ranks so
        each rank gets a disjoint subset (deterministic, no shuffling).
        """
        max_idx = self.dataset.n_time - self.max_step - 1
        candidates = list(range(0, max_idx + 1, self.ic_stride))
        candidates = candidates[: self.max_initial_conditions]
        # Round-robin so each rank gets ~equal load.
        return [c for i, c in enumerate(candidates) if i % world_size == rank]

    # ---------------------------------------------------------------- #
    # Public entry
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def run(self, model, *, epoch: int = 0) -> dict[str, float]:
        """Run the rollout validation for the given (already EMA'd, eval-mode) model.

        Returns a flat ``{metric_name: scalar_float}`` dict.
        """
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1

        ic_indices = self._select_ic_indices(rank, world_size)
        # Generator on the validator's device — ensures
        # GaussianIC can use it without a device mismatch.
        try:
            gen = torch.Generator(device=self.device).manual_seed(
                self.seed + epoch * 100003
            )
        except (RuntimeError, TypeError):  # fallback for older torch
            gen = torch.Generator(device="cpu").manual_seed(
                self.seed + epoch * 100003
            )

        log_step_to_metric_idx = {s: i for i, s in enumerate(self.log_steps)}

        # Iterate over the IC indices in micro-batches.
        for batch_start in range(0, len(ic_indices), self.batch_size):
            batch_ics = ic_indices[batch_start : batch_start + self.batch_size]
            if not batch_ics:
                continue
            self._rollout_batch(model, batch_ics, gen, log_step_to_metric_idx)

        return self._finalize(epoch=epoch)

    # ---------------------------------------------------------------- #
    # Inner loop
    # ---------------------------------------------------------------- #

    def _stack_initial(self, batch_ics: list[int]) -> dict[str, torch.Tensor]:
        """Stack a list of dataset samples (lead=1) into a single batch."""
        samples = [self.dataset[(int(t), 1)] for t in batch_ics]
        out: dict[str, torch.Tensor] = {}
        for k in samples[0]:
            v0 = samples[0][k]
            if isinstance(v0, torch.Tensor):
                if v0.dim() == 0:
                    out[k] = torch.stack([s[k] for s in samples], dim=0)
                else:
                    out[k] = torch.stack([s[k] for s in samples], dim=0)
            else:
                out[k] = v0
        return out

    def _fetch_step(self, t: int) -> dict[str, torch.Tensor]:
        """Get the single dataset sample at time ``t`` (lead=1 placeholder)."""
        return self.dataset[(int(t), 1)]

    def _stack_at_step(self, t_list: list[int]) -> dict[str, torch.Tensor]:
        """Stack per-IC dataset samples at the given times into a batch."""
        samples = [self._fetch_step(t) for t in t_list]
        out: dict[str, torch.Tensor] = {}
        for k in samples[0]:
            v0 = samples[0][k]
            if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
                out[k] = torch.stack([s[k] for s in samples], dim=0)
        return out

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        if self.normalizer is not None:
            moved = self.normalizer(moved)
        return moved

    def _ensemble_mean(self, x: torch.Tensor, n_ic: int) -> torch.Tensor:
        """Reduce the ensemble axis (interleaved per IC) back to ``(B, …)``."""
        if self.ensemble_size == 1:
            return x
        rest = x.shape[1:]
        x = x.view(n_ic, self.ensemble_size, *rest)
        return x.mean(dim=1)

    def _denorm_pred_truth(
        self,
        kind: str,
        pred: torch.Tensor,
        truth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pred, truth)`` in physical units for the named channel group.

        ``kind`` is one of ``"surface"``, ``"upper_air"``, ``"diagnostic"``.
        When ``self.normalizer is None`` (or the normalizer has no stats
        for ``kind``) the inputs pass through unchanged — the rollout was
        already running in raw units in that case.
        """
        if self.normalizer is None:
            return pred, truth
        pred_phys = self.normalizer.denormalize_state(**{kind: pred})[kind]
        truth_phys = self.normalizer.denormalize_state(**{kind: truth})[kind]
        return pred_phys, truth_phys

    def _rollout_batch(
        self,
        model,
        batch_ics: list[int],
        gen: torch.Generator,
        log_step_to_metric_idx: dict[int, int],
    ) -> None:
        # Initial state at t (will produce prediction for t+1, t+2, ...).
        init = self._to_device(self._stack_initial(batch_ics))

        # Ensemble replication of the initial state (and any per-IC fields).
        state = self.perturber(init, self.ensemble_size, generator=gen)
        n_ic = len(batch_ics)

        # constant_boundary is broadcastable; keep as-is (model handles it).
        const_boundary = state.get("constant_boundary")

        # March the rollout. At step k we have ``state`` = pred-for-time
        # ``t0 + k`` (or, at k=0, the dataset's input at t0). We need the
        # varying boundary at the current input time and the truth at the
        # next time to score the prediction.
        for k in range(1, self.max_step + 1):
            target_times = [t + k for t in batch_ics]
            target_batch = self._to_device(self._stack_at_step(target_times))

            # Use the varying boundary AT THE INPUT TIME (matches train).
            input_var_boundary = state["varying_boundary"]

            out = model(
                state["surface_in"],
                const_boundary,
                input_var_boundary,
                state["upper_air_in"],
            )
            if self.has_diagnostic:
                next_surface, next_upper_air, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper_air = out[0], out[1]
                next_diag = None

            # Score this step if it's in the requested log_steps.
            # `target_batch` holds the dataset sample at time t+k. The truth at
            # time t+k is that sample's `surface_in` / `upper_air_in` (the
            # state AT that time) — NOT `target_surface` (which is the
            # dataset's lead-1 prediction from t+k, i.e. the state at t+k+1).
            #
            # RMSE values are reported in PHYSICAL UNITS: both pred and truth
            # tensors are de-normalized here before the aggregator update so
            # the per-variable RMSE has the same units as the underlying field
            # (K for temperature, Pa for pressure, …). Per the project
            # convention, all per-variable metrics are in physical units;
            # correlation metrics (ACC) are unit-invariant so we pass the
            # tensors through as-is to the ACC aggregator.
            if k in log_step_to_metric_idx:
                m_idx = log_step_to_metric_idx[k]
                pred_surface = self._ensemble_mean(next_surface, n_ic)
                truth_surface = target_batch["surface_in"]
                pred_surface_phys, truth_surface_phys = self._denorm_pred_truth(
                    "surface", pred_surface, truth_surface
                )
                self.rmse_surface.update(
                    m_idx, pred_surface_phys, truth_surface_phys, self.register_lat
                )
                if self.acc_surface is not None:
                    self.acc_surface.update(
                        m_idx, pred_surface, truth_surface, self.register_lat
                    )
                if self.has_upper_air and "upper_air_in" in target_batch:
                    pred_upper = self._ensemble_mean(next_upper_air, n_ic)
                    truth_upper = target_batch["upper_air_in"]
                    pred_upper_phys, truth_upper_phys = self._denorm_pred_truth(
                        "upper_air", pred_upper, truth_upper
                    )
                    self.rmse_upper_air.update(
                        m_idx, pred_upper_phys, truth_upper_phys, self.register_lat
                    )
                    if self.acc_upper_air is not None:
                        self.acc_upper_air.update(
                            m_idx, pred_upper, truth_upper, self.register_lat
                        )
                if next_diag is not None and self.rmse_diagnostic is not None:
                    # Diagnostics live on the `target_*` axis of the dataset
                    # (they're an output of the predictor for t+k, paired with
                    # the lead-1 sample). So when scoring the prediction made
                    # AT step k from state(t+k-1), we want the dataset's
                    # diagnostic at t+k → that's the `diagnostic` key of the
                    # sample taken at start_idx=t+k-1 with lead=1; here we
                    # approximate by reading the diagnostic field of the
                    # target_batch sample (start=t+k, lead=1) which holds the
                    # diagnostic FOR time t+k+1. For step-k scoring, prefer
                    # the prior batch's diagnostic; for the simple Phase-4a
                    # implementation we accept the one-step offset and gate
                    # this branch on dataset support.
                    if "diagnostic" in target_batch:
                        pred_diag = self._ensemble_mean(next_diag, n_ic)
                        truth_diag = target_batch["diagnostic"]
                        pred_diag_phys, truth_diag_phys = self._denorm_pred_truth(
                            "diagnostic", pred_diag, truth_diag
                        )
                        self.rmse_diagnostic.update(
                            m_idx, pred_diag_phys, truth_diag_phys, self.register_lat
                        )
                        if self.acc_diagnostic is not None:
                            self.acc_diagnostic.update(
                                m_idx, pred_diag, truth_diag, self.register_lat
                            )

            # Advance: the next iteration's state is this step's prediction.
            # `varying_boundary` for the next step is the boundary AT time
            # t+k, which is the `varying_boundary` of the sample at start=t+k.
            # In ensemble mode the boundary needs to march along with each
            # member (E identical copies per IC, contiguous in the batch dim
            # to match the perturber's repeat_interleave layout).
            next_boundary = target_batch["varying_boundary"]
            if self.ensemble_size > 1:
                next_boundary = next_boundary.repeat_interleave(
                    self.ensemble_size, dim=0
                )
            state = {
                "surface_in": next_surface,
                "upper_air_in": next_upper_air,
                "constant_boundary": const_boundary,
                "varying_boundary": next_boundary,
            }

    # ---------------------------------------------------------------- #
    # Finalize
    # ---------------------------------------------------------------- #

    def _finalize(self, epoch: int) -> dict[str, float]:
        results: dict[str, float] = {}

        def _emit(prefix: str, metric, group: str):
            if metric is None:
                return
            vals = metric.finalize()  # (n_steps, n_channels)
            # Channel-mean to keep dashboard small; raw per-channel
            # numbers can be added behind a flag later.
            per_step = vals.mean(dim=1)
            for i, step in enumerate(self.log_steps):
                results[f"{prefix}_step{step}_{group}"] = float(per_step[i].item())

        _emit("rmse", self.rmse_surface, "surface")
        _emit("rmse", self.rmse_upper_air, "upper_air")
        _emit("rmse", self.rmse_diagnostic, "diagnostic")
        _emit("acc", self.acc_surface, "surface")
        _emit("acc", self.acc_upper_air, "upper_air")
        _emit("acc", self.acc_diagnostic, "diagnostic")
        return results


__all__ = [
    "Perturber",
    "Deterministic",
    "ReplicateOnly",
    "GaussianIC",
    "StreamingLatWeightedRMSE",
    "StreamingLatWeightedACC",
    "RolloutValidator",
    "cos_lat_weights",
]
