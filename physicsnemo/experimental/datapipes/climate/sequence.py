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


_SEQ_KEYS = (
    "surface_in",
    "upper_air_in",
    "upper_air_sigma_in",
    "upper_air_pressure_in",
    "varying_boundary",
    "diagnostic",
)


class SequenceDataset(Dataset):
    r"""Stack a base dataset's ``__getitem__`` outputs across a rollout window.

    Wraps a :class:`ClimateZarrDataset`-shaped object and emits, for each
    integer index ``t``, a dict with the following keys (``T = unroll_steps``):

    * ``surface_in_seq``:        ``(T+1, C_s, H, W)`` — frames at t, t+1, …, t+T
    * ``upper_air_in_seq``:      ``(T+1, C_u, L, H, W)`` (when single-coord layout)
    * ``upper_air_sigma_in_seq`` / ``upper_air_pressure_in_seq``: same when the
                                   dataset emits separated sigma + pressure keys
    * ``varying_boundary_seq``:  ``(T+1, C_b, H, W)``
    * ``diagnostic_seq``:        ``(T+1, C_d, H, W)`` (when the layout has diag)
    * ``constant_boundary``:     unchanged (constant in time)
    * ``start_idx``, ``unroll_steps``: scalar tensors for debug/replay

    For ``T = 0`` the leading dim is 1 and the emitted dict carries the
    same data as a single-step ``(start, 1)`` lookup, just under the
    ``_seq`` keys.

    Parameters
    ----------
    base
        Underlying dataset (must have integer-indexed ``__getitem__`` returning
        dicts with the keys above plus ``constant_boundary``, plus the
        ``n_time`` attribute and a compatible ``layout``).
    unroll_steps
        Number of rollout steps the model will be trained on. Sequence
        length is ``unroll_steps + 1``.
    """

    def __init__(self, base, unroll_steps: int):
        if unroll_steps < 0:
            raise ValueError(f"unroll_steps must be ≥ 0, got {unroll_steps}")
        self.base = base
        self.unroll_steps = int(unroll_steps)
        self.layout = getattr(base, "layout", None)

    @property
    def n_time(self) -> int:
        return int(self.base.n_time)

    def __len__(self) -> int:
        # We need ``unroll_steps + 1`` consecutive frames starting at idx.
        return max(0, self.n_time - self.unroll_steps)

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
        if start_idx < 0 or start_idx + self.unroll_steps >= self.n_time:
            raise IndexError(
                f"sequence index {start_idx} (+{self.unroll_steps}) out of "
                f"range [0, {self.n_time})"
            )

        # Fetch ``unroll_steps + 1`` consecutive frames.
        # Each lookup with lead=1 gives surface_in/upper_air_in at start
        # (and a target at start+1 we ignore). We just want the input
        # frame at each successive start; reading lead=1 also pulls the
        # next frame which we discard.
        frames = []
        for k in range(self.unroll_steps + 1):
            t = start_idx + k
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

        out: dict[str, torch.Tensor] = {}
        for key in _SEQ_KEYS:
            if key in frames[0] and isinstance(frames[0][key], torch.Tensor):
                out[f"{key}_seq"] = torch.stack(
                    [f[key] for f in frames], dim=0
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
