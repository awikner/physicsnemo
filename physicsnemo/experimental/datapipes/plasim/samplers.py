# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Multi-lead-time pair samplers for :class:`PlasimClimateDataset`.

The companion sampler yields ``(start_time_idx, lead_time_steps)`` tuples that
:meth:`PlasimClimateDataset.__getitem__` indexes by. Lead times are integers in
units of the store's ``data_timedelta_hours`` (typically 6 h on PLASIM PanguWeather
configs), matching PanguWeather v2.0's ``forecast_lead_times`` convention.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler


class LeadTimePairSampler(Sampler):
    r"""Sampler that yields ``(start_t, lead_t)`` pairs.

    Per epoch, generates ``num_samples`` random pairs where ``start_t`` is
    drawn uniformly from valid start indices (such that ``start_t + lead_t``
    is in range) and ``lead_t`` is drawn uniformly from
    ``forecast_lead_times``. The optional ``shuffle=False`` mode iterates
    deterministically through start indices and cycles lead times.

    DDP-friendly: when ``rank`` and ``world_size`` are supplied (e.g., from
    :class:`physicsnemo.distributed.DistributedManager`), each rank produces a
    disjoint subset of pairs and the per-rank length is the per-epoch count
    divided by ``world_size`` (truncated, to keep all ranks at equal length
    — matches :class:`torch.utils.data.distributed.DistributedSampler`).

    Parameters
    ----------
    dataset_length : int
        Length of the underlying :class:`PlasimClimateDataset` along ``time``.
    forecast_lead_times : sequence of int
        Allowed lead times (in units of ``data_timedelta_hours``). PanguWeather
        v2.0 PLASIM configs use, e.g., ``[1, 12, 20, 40, 60]``.
    num_samples : int, optional, default=None
        Number of pairs per epoch. ``None`` defaults to ``dataset_length``.
    shuffle : bool, optional, default=True
        Random sampling each epoch when ``True``; deterministic walk when ``False``.
    seed : int, optional, default=0
        RNG seed for shuffled epochs.
    rank : int, optional, default=0
        DDP rank. When ``world_size > 1``, this rank's per-epoch slice.
    world_size : int, optional, default=1
        Number of DDP ranks.
    drop_last : bool, optional, default=True
        Drop trailing samples so ``len(self)`` is identical across ranks.

    Forward
    -------
    epoch : int, optional
        Set via :meth:`set_epoch` before iterating to advance the RNG state.

    Outputs
    -------
    Iterator of tuple of int
        ``(start_t, lead_t)`` pairs.
    """

    def __init__(
        self,
        dataset_length: int,
        forecast_lead_times: Sequence[int],
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ) -> None:
        if dataset_length <= 0:
            raise ValueError(f"dataset_length must be > 0, got {dataset_length}")
        leads = sorted(int(lt) for lt in forecast_lead_times)
        if not leads or leads[0] < 1:
            raise ValueError(
                f"forecast_lead_times must be a non-empty sequence of positive "
                f"ints, got {forecast_lead_times!r}"
            )
        if max(leads) >= dataset_length:
            raise ValueError(
                f"max(forecast_lead_times)={max(leads)} >= dataset_length="
                f"{dataset_length}; no valid pairs exist."
            )

        self.dataset_length = dataset_length
        self.forecast_lead_times = leads
        self.num_samples = int(num_samples) if num_samples is not None else dataset_length
        self.shuffle = shuffle
        self.seed = seed
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = drop_last
        self.epoch = 0

        self._per_rank_len = (
            self.num_samples // self.world_size
            if drop_last
            else math.ceil(self.num_samples / self.world_size)
        )

    def set_epoch(self, epoch: int) -> None:
        """DDP-style epoch setter to keep cross-rank shuffles aligned."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._per_rank_len

    def __iter__(self) -> Iterator[tuple[int, int]]:
        max_lead = max(self.forecast_lead_times)
        valid_starts = self.dataset_length - max_lead
        # Note: a lead_t < max_lead allows start_t up to (dataset_length - lead_t - 1),
        # but for simplicity we cap all starts at the strictest bound so every drawn
        # (start, lead) is in range without per-sample bookkeeping.

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            full_starts = torch.randint(
                low=0, high=valid_starts, size=(self.num_samples,), generator=g
            ).tolist()
            full_lead_idx = torch.randint(
                low=0,
                high=len(self.forecast_lead_times),
                size=(self.num_samples,),
                generator=g,
            ).tolist()
        else:
            # Deterministic walk: starts cycle through [0, valid_starts), leads cycle
            # through forecast_lead_times in order.
            full_starts = [i % valid_starts for i in range(self.num_samples)]
            full_lead_idx = [
                i % len(self.forecast_lead_times) for i in range(self.num_samples)
            ]

        # DDP slice: rank r gets indices [r, r + world_size, r + 2*world_size, ...].
        per_rank_starts = full_starts[self.rank :: self.world_size][: self._per_rank_len]
        per_rank_lead_idx = full_lead_idx[self.rank :: self.world_size][: self._per_rank_len]
        for s, li in zip(per_rank_starts, per_rank_lead_idx):
            yield (s, self.forecast_lead_times[li])
