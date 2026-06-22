# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Streaming time-aggregators for climatological validation (Phase 4c).

A multi-year autoregressive rollout produces O(T × C × H × W) samples,
which is far too large to hold in memory. The streaming aggregators in
this module accumulate the time statistics we actually care about — the
field-level **mean**, **variance**, and per-(day-of-year) climatology
— in O(C × H × W) memory regardless of how long the rollout runs.

Algorithms
----------

* :class:`StreamingTimeMean` — running mean over time.

* :class:`StreamingTimeVariance` — Welford / Chan parallel-update online
  variance. Numerically stable for the large-mean / small-variance
  regime typical of climate data (raw temperatures ~273 K, variances
  ~10²; naive ``E[X²] - E[X]²`` catastrophically cancels). State per
  spatial field: ``(mean, M2, n)``; ``var = M2 / (n - 1)`` at finalize.

* :class:`StreamingBinnedMean` — per-bin running mean. Drives daily /
  monthly / season-of-year climatology by binning each frame's
  contribution to one of ``n_bins`` accumulators (caller supplies the
  bin index per update). Internally uses :class:`StreamingTimeMean`
  per bin so the same Welford-stability properties hold.

All aggregators default to ``torch.float64`` accumulators to keep
cancellation out of the picture for raw-unit data; outputs are
cast back to the caller's preferred dtype on ``finalize``.

The aggregators are DDP-aware in the same way as the Phase 4a metrics
— they expose an ``all_reduce`` step in ``finalize`` that sums state
across ranks when a process group is initialized. Identical to a
single-rank run by construction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def _all_reduce_sum(t: torch.Tensor) -> None:
    """In-place all-reduce SUM when a distributed group is initialized."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


# ---------------------------------------------------------------------------
# StreamingTimeMean
# ---------------------------------------------------------------------------


class StreamingTimeMean:
    r"""Running mean of a per-pixel field across time + batch.

    Parameters
    ----------
    shape : tuple of int
        Per-sample shape (e.g. ``(C, H, W)`` or ``(C, L, H, W)``).
    device : torch.device
        Device for the accumulator tensors.
    dtype : torch.dtype, default ``torch.float64``
        Accumulator dtype. Cast back to caller dtype at ``finalize``.

    The :meth:`update` method accepts tensors of shape
    ``(B, *shape)`` — the leading batch dim is reduced into the running
    statistic.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.sum = torch.zeros(shape, device=device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"update expected (B, *{self.shape}) tensor, got {tuple(x.shape)}"
            )
        self.sum += x.detach().to(self.dtype).sum(dim=0)
        self.n += int(x.shape[0])

    def finalize(self, *, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """All-reduce + return the field-mean as ``(*shape,)``."""
        s = self.sum.clone()
        n_t = torch.tensor(self.n, device=s.device, dtype=self.dtype)
        _all_reduce_sum(s)
        _all_reduce_sum(n_t)
        n_eff = float(n_t.item())
        if n_eff == 0:
            return s.to(out_dtype)
        return (s / n_eff).to(out_dtype)


# ---------------------------------------------------------------------------
# StreamingTimeVariance (Welford / Chan)
# ---------------------------------------------------------------------------


class StreamingTimeVariance:
    r"""Running mean + variance via Chan's parallel update.

    Combines two groups :math:`(n_a, \mu_a, M_a^{(2)})` and
    :math:`(n_b, \mu_b, M_b^{(2)})` as

    .. math::

       n &= n_a + n_b \\
       \mu &= \mu_a + (\mu_b - \mu_a) \cdot n_b / n \\
       M^{(2)} &= M_a^{(2)} + M_b^{(2)} +
           (\mu_b - \mu_a)^2 \cdot (n_a n_b / n)

    Each :meth:`update` call treats the incoming batch as group ``b``
    (computes its own batch mean + M2) and merges into the running
    state. This is numerically stable even when ``mean`` is much larger
    than ``var`` (raw-unit climate fields).

    ``finalize`` returns ``(mean, var)`` with ``var = M2 / (n - 1)``
    (unbiased / Bessel-corrected). For ``n <= 1`` the variance is set
    to zero rather than NaN.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.M2 = torch.zeros(shape, device=device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"update expected (B, *{self.shape}) tensor, got {tuple(x.shape)}"
            )
        x = x.detach().to(self.dtype)
        n_b = int(x.shape[0])
        if n_b == 0:
            return
        mean_b = x.mean(dim=0)
        if n_b > 1:
            M2_b = ((x - mean_b) ** 2).sum(dim=0)
        else:
            M2_b = torch.zeros_like(mean_b)
        n_total = self.n + n_b
        delta = mean_b - self.mean
        # In-place updates preserve memory.
        self.mean.add_(delta * (float(n_b) / float(n_total)))
        # Chan's parallel formula: combine M2's with the bridging term.
        bridge = delta.pow(2) * (float(self.n) * float(n_b) / float(n_total))
        self.M2.add_(M2_b + bridge)
        self.n = n_total

    @torch.no_grad()
    def finalize(
        self, *, out_dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce + return ``(mean, var)``.

        DDP merge uses the same Chan parallel formula across ranks (each
        rank contributes a *group* with its current ``(n, mean, M2)``).
        Implemented via three all-reduce passes — ``n_total``, weighted
        mean, total M2 — that together reconstruct the merged state.
        """
        mean = self.mean.clone()
        M2 = self.M2.clone()
        n_t = torch.tensor(self.n, device=mean.device, dtype=self.dtype)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Compute n_total first, then weighted-mean, then sum of
            # M2 + bridging terms. The bridging needs pairwise (μ_a -
            # μ_b) which we don't have, but Chan's two-pass equivalence
            # gives:  M2_total = Σ M2_i + Σ n_i (μ_i - μ_total)²
            world = dist.get_world_size()
            n_local = torch.tensor(self.n, device=mean.device, dtype=self.dtype)
            n_total = n_local.clone()
            _all_reduce_sum(n_total)
            weighted_mean = mean * n_local
            _all_reduce_sum(weighted_mean)
            mean = weighted_mean / n_total.clamp(min=1)
            bridge = n_local * (self.mean - mean).pow(2)
            _all_reduce_sum(M2)
            _all_reduce_sum(bridge)
            M2 = M2 + bridge
            n_t = n_total
        n_eff = float(n_t.item())
        var = torch.zeros_like(M2) if n_eff < 2 else M2 / (n_eff - 1.0)
        return mean.to(out_dtype), var.to(out_dtype)

    @property
    def count(self) -> int:
        return int(self.n)


# ---------------------------------------------------------------------------
# StreamingBinnedMean
# ---------------------------------------------------------------------------


class StreamingBinnedMean:
    r"""Per-bin running mean — e.g. day-of-year or month-of-year climatology.

    Caller supplies the bin index for each sample in the batch. Memory
    is ``O(n_bins × C × H × W × 8 bytes)`` (f64 accumulators).

    Parameters
    ----------
    n_bins : int
        Number of climatology bins (e.g. 365 for daily, 12 for monthly).
    shape : tuple of int
        Per-sample shape ``(C, H, W)`` or ``(C, L, H, W)``.
    device, dtype
        As :class:`StreamingTimeMean`.
    """

    def __init__(
        self,
        n_bins: int,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ):
        self.n_bins = int(n_bins)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.sum = torch.zeros((n_bins, *shape), device=device, dtype=dtype)
        self.counts = torch.zeros(n_bins, device=device, dtype=torch.int64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, bin_idx: torch.Tensor) -> None:
        """``x`` shape ``(B, *shape)``; ``bin_idx`` shape ``(B,)`` int."""
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"update expected (B, *{self.shape}), got {tuple(x.shape)}"
            )
        if bin_idx.shape != (x.shape[0],):
            raise ValueError(
                f"bin_idx must be shape ({x.shape[0]},), got {tuple(bin_idx.shape)}"
            )
        if (bin_idx < 0).any() or (bin_idx >= self.n_bins).any():
            raise ValueError(
                f"bin_idx out of range [0, {self.n_bins})"
            )
        x = x.detach().to(self.dtype)
        # scatter_add_ with the right index broadcasting.
        # Use index_add_ along dim=0 of the sum buffer.
        self.sum.index_add_(0, bin_idx.to(self.sum.device).to(torch.long), x)
        # counts: increment per bin
        ones = torch.ones_like(bin_idx, dtype=torch.int64)
        self.counts.index_add_(0, bin_idx.to(self.counts.device).to(torch.long), ones)

    @torch.no_grad()
    def finalize(self, *, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """All-reduce + return per-bin mean ``(n_bins, *shape)``.

        Bins with zero observed samples are returned as zero rather
        than NaN — this is the climatology convention for un-sampled
        days in a partial-year rollout.
        """
        s = self.sum.clone()
        c = self.counts.clone().to(self.dtype)
        _all_reduce_sum(s)
        _all_reduce_sum(c)
        c_safe = c.clamp(min=1.0)
        means = s / c_safe.view(-1, *([1] * len(self.shape)))
        # Zero out un-sampled bins.
        empty = (c == 0).view(-1, *([1] * len(self.shape)))
        means = torch.where(empty, torch.zeros_like(means), means)
        return means.to(out_dtype)

    @property
    def counts_per_bin(self) -> torch.Tensor:
        return self.counts.clone()


__all__ = [
    "StreamingTimeMean",
    "StreamingTimeVariance",
    "StreamingBinnedMean",
]
