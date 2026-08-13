# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-emitting wrapper around :class:`ClimateZarrDataset`.

The training loop's multi-step rollout curriculum needs windowed access
to the underlying time series: for an ``unroll_steps=K`` stage, each
training sample should expose state(t), boundary(t), boundary(t+1), …,
boundary(t+K-1), and the target states at t+1 … t+K so the model can be
unrolled with per-step loss accumulation.

:class:`SequenceDataset` produces dicts whose tensors carry an extra
leading time dim of length ``K+1`` (one initial state + ``K`` target
frames). Composed via :class:`ClimateDatapipe`, the loader stacks
across batch and emits ``(B, K+1, C, [L,] H, W)`` tensors on device.

Normalization continues to apply per-channel through PyTorch broadcast
rules — see :func:`physicsnemo.experimental.datapipes.plasim.transforms.ClimateNormalizer`
for the routing that recognizes the ``_seq`` keys.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset


# State (predicted) fields. Under ``forcing_lag > 0`` these are read one step
# LATER than the forcing keys below.
_STATE_SEQ_KEYS = (
    "surface_in",
    "upper_air_in",
    "upper_air_sigma_in",
    "upper_air_pressure_in",
    "diagnostic",
)

# Exogenous (conditioning) fields — the gridded boundary and the per-frame
# calendar vector (present when the base dataset runs with
# ``emit_calendar=True``; rolling-window diffusion needs the
# ``(W, scalar_dim)`` stack as ``c_scalar`` — Phase 12b).
_FORCING_SEQ_KEYS = (
    "varying_boundary",
    "calendar",
)

_SEQ_KEYS = _STATE_SEQ_KEYS + _FORCING_SEQ_KEYS


class SequenceDataset(Dataset):
    r"""Stack a base dataset's ``__getitem__`` outputs across a rollout window.

    Wraps a :class:`ClimateZarrDataset`-shaped object and emits, for each
    integer index ``t``, a dict with the following keys (``T = unroll_steps``):

    * ``surface_in_seq``:        ``(T+1, C_s, H, W)`` — frames at t, t+1, …, t+T
    * ``upper_air_in_seq``:      ``(T+1, C_u, L, H, W)`` (when single-coord layout)
    * ``upper_air_sigma_in_seq`` / ``upper_air_pressure_in_seq``: same when the
                                   dataset emits separated sigma + pressure keys
    Frames are ``step_stride`` store rows apart (default 1); see the parameter
notes — a 6-hourly archive under a 24-hour model step needs ``step_stride=4``.

* ``varying_boundary_seq``:  ``(T+1, C_b, H, W)`` — the forcing window
      (one step behind the state under ``forcing_lag=1``)
    * ``varying_boundary_next_seq``: ``(T+1, C_b, H, W)`` — same channels at
      the *state* frames' own times, when ``emit_boundary_next=True``
    * ``diagnostic_seq``:        ``(T+1, C_d, H, W)`` (when the layout has diag)
    * ``constant_boundary``:     unchanged (constant in time)
    * ``start_idx``, ``unroll_steps``: scalar tensors for debug/replay

    For ``T = 0`` the leading dim is 1 and the emitted dict carries the
    same data as a single-step ``(start, 1)`` lookup, just under the
    ``_seq`` keys.

    .. note:: **Forcing alignment (Phase 12f).** ``forcing_lag`` chooses
        which forcing frame conditions a state frame:

        * ``0`` (default) — own-time: slot ``j`` is state(t+j) with
          boundary(t+j). The historical fork behavior; pinned by the frozen
          ``channel_layout="fork"`` configs.
        * ``1`` — upstream amip's convention: slot ``j`` is state(t+1+j)
          conditioned on boundary(t+j), i.e. denoising the frame at time T
          uses the forcing from T-1. This is what real v1/v2 checkpoints
          were trained with, so it is what the ``"v1"`` / ``"v2"`` layouts
          use.

        The two differ by a whole-window shift, so nothing changes shape and
        a mismatch is silent — which is why
        :class:`~physicsnemo.experimental.models.amip_si.RollingDiTWrapper`
        derives it from the channel layout rather than a config knob.

    Parameters
    ----------
    base
        Underlying dataset (must have integer-indexed ``__getitem__`` returning
        dicts with the keys above plus ``constant_boundary``, plus the
        ``n_time`` attribute and a compatible ``layout``).
    unroll_steps
        Number of rollout steps the model will be trained on. Sequence
        length is ``unroll_steps + 1``.
    forcing_lag
        Frames by which the exogenous keys (``varying_boundary``,
        ``calendar``) lag the state keys — see the note above. Each sample
        reads ``unroll_steps + 1 + forcing_lag`` base frames.
    step_stride
        Store rows advanced per model step (upstream's
        ``timedelta_hours // data_timedelta_hours``). ``1`` — one row per step —
        is right for PLASIM / E3SM, whose archives are stored at the model's own
        timestep; the AMIP and ERA5 archives are 6-hourly under a 24-hour model
        step, so they need ``4``. Derive it with
        :func:`~physicsnemo.experimental.datapipes.climate.resolve_step_stride`
        rather than hardcoding, so it is cross-checked against the store's own
        ``data_timedelta_hours``. Both the window frames *and* the
        ``forcing_lag`` offset stride: "one step back" is one model step, not
        one row.
    emit_boundary_next
        Also emit ``varying_boundary_next_seq``: the boundary at each state
        frame's *own* time, the ocean-channel training target (upstream's
        ``[1:]`` view of its ``W+1`` boundary stack). Requires
        ``forcing_lag >= 1``, because at lag 0 it would be bit-identical to
        ``varying_boundary_seq`` and the ocean task would be an identity
        copy of an input channel.
    """

    def __init__(
        self,
        base,
        unroll_steps: int,
        *,
        forcing_lag: int = 0,
        emit_boundary_next: bool = False,
        step_stride: int = 1,
    ):
        if unroll_steps < 0:
            raise ValueError(f"unroll_steps must be ≥ 0, got {unroll_steps}")
        if forcing_lag < 0:
            raise ValueError(f"forcing_lag must be ≥ 0, got {forcing_lag}")
        if step_stride < 1:
            raise ValueError(f"step_stride must be ≥ 1, got {step_stride}")
        if emit_boundary_next and forcing_lag == 0:
            raise ValueError(
                "emit_boundary_next=True with forcing_lag=0 would emit "
                "varying_boundary_next_seq identical to varying_boundary_seq "
                "— the ocean target would be a copy of an input channel in the "
                "same token (a silent identity task). Set forcing_lag=1."
            )
        self.base = base
        self.unroll_steps = int(unroll_steps)
        self.forcing_lag = int(forcing_lag)
        self.emit_boundary_next = bool(emit_boundary_next)
        self.step_stride = int(step_stride)
        self.layout = getattr(base, "layout", None)

    @property
    def n_time(self) -> int:
        return int(self.base.n_time)

    @property
    def frames_per_sample(self) -> int:
        """Base-dataset frames read per sample: ``W + forcing_lag``."""
        return self.unroll_steps + 1 + self.forcing_lag

    @property
    def row_span(self) -> int:
        """Store rows between the first and last frame of a sample.

        ``(frames - 1) * step_stride`` — the frames are ``step_stride`` rows
        apart, so a W-frame window covers more of the archive than W rows.
        """
        return (self.frames_per_sample - 1) * self.step_stride

    def __len__(self) -> int:
        # A sample needs ``row_span`` rows of headroom past its start index.
        return max(0, self.n_time - self.row_span)

    @property
    def transform(self):
        return getattr(self.base, "transform", None)

    @transform.setter
    def transform(self, value):
        # Forward to base so per-variable NaN-fill etc. apply per frame.
        self.base.transform = value

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        # The sampler is the basic IntSampler so index is always an int.
        # Tuples ``(int, int)`` are accepted for parity with the base
        # dataset's API but only the start_idx is read.
        if isinstance(index, tuple):
            start_idx = int(index[0])
        else:
            start_idx = int(index)
        span = self.row_span
        if start_idx < 0 or start_idx + span >= self.n_time:
            raise IndexError(
                f"sequence index {start_idx} (+{span}) out of "
                f"range [0, {self.n_time})"
            )

        # Fetch ``W + forcing_lag`` frames, ``step_stride`` rows apart.
        # Each lookup with lead=1 gives surface_in/upper_air_in at start
        # (and a target at start+1 we ignore). We just want the input
        # frame at each successive start; reading lead=1 also pulls the
        # next frame which we discard.
        frames = []
        for k in range(self.frames_per_sample):
            t = start_idx + k * self.step_stride
            # Use lead=1 for k<T (so the dataset can build target_*) and
            # lead=1 also at the last frame — the target is never used in
            # sequence mode but the base requires lead>=1 + (start+lead) in
            # range. Fall back to reading just the input frame via a direct
            # _sample_at if that path is exposed; otherwise the lead=1 form
            # is fine when t+1 <= n_time-1.
            if t + 1 < self.n_time:
                sample = self.base[(t, 1)]
            else:
                # Edge case: last frame in dataset; index out-of-range guard
                # above prevents this, but stay defensive.
                sample = self.base[(t - 1, 1)]
            frames.append(sample)

        # Slice the read window into the two views. With ``forcing_lag=0``
        # both are ``frames[:W]`` and this is the historical behavior; with
        # ``forcing_lag=1`` the state frames sit one step later than the
        # forcings they are conditioned on (upstream amip's convention).
        W = self.unroll_steps + 1
        lag = self.forcing_lag
        state_frames = frames[lag : lag + W]
        forcing_frames = frames[:W]

        def _stack(key, src):
            return torch.stack([f[key] for f in src], dim=0)

        out: dict[str, torch.Tensor] = {}
        for key in _SEQ_KEYS:
            if key in frames[0] and isinstance(frames[0][key], torch.Tensor):
                src = forcing_frames if key in _FORCING_SEQ_KEYS else state_frames
                out[f"{key}_seq"] = _stack(key, src)
        if self.emit_boundary_next and "varying_boundary" in frames[0]:
            # The boundary at each STATE frame's own time — the ocean-channel
            # training target (upstream's ``[1:]`` view of its W+1 stack). Not
            # a third read: it is the same frames the state came from.
            out["varying_boundary_next_seq"] = _stack(
                "varying_boundary", state_frames
            )
        if "constant_boundary" in frames[0]:
            out["constant_boundary"] = frames[0]["constant_boundary"]
        out["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
        out["unroll_steps"] = torch.tensor(self.unroll_steps, dtype=torch.long)
        # Propagate non-tensor info (e.g. forecast time) from the first frame
        # if the caller wants it; conservative subset.
        return out


class IntSampler(torch.utils.data.Sampler):
    """Plain integer sampler over ``[0, dataset_length)``.

    The default :class:`LeadTimePairSampler` emits ``(start, lead)`` tuples
    suitable for the single-step dataset path; sequence mode wants plain
    ints so :class:`SequenceDataset` can compute its window.
    """

    def __init__(
        self,
        dataset_length: int,
        *,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_length = int(dataset_length)
        self.num_samples = num_samples
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 100003 * self._epoch)
        if self.shuffle:
            order = torch.randperm(self.dataset_length, generator=g).tolist()
        else:
            order = list(range(self.dataset_length))
        # Round-robin across ranks for a deterministic, disjoint shard.
        order = [order[i] for i in range(self.rank, len(order), self.world_size)]
        if self.num_samples is not None:
            order = order[: int(self.num_samples)]
        return iter(order)

    def __len__(self) -> int:
        n = self.dataset_length
        if self.num_samples is not None:
            n = min(n, int(self.num_samples))
        return (n + self.world_size - 1) // self.world_size
