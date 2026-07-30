# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``HindcastMixtureDataset``: multi-expert hindcast + IMERG truth samples.

One sample = one (init_time, tau) pair from the precomputed index
(:mod:`.index`): every live expert contributes a normalized
``(1 + C, H, W)`` block (channel 0 = daily precip for the day ending at
``init + tau*24h``, mm/day before normalization; other channels =
instantaneous dynamical predictors at lead ``tau``), stacked to
``(E, 1 + C, H, W)`` with a per-sample expert availability mask.

Conventions
-----------
* Missing data is exactly **0 in normalized space** (== the climatological
  mean): expert blocks are normalized first, then NaNs are zero-filled and
  masked channels / dead experts re-zeroed.
* A live expert whose block is mostly non-finite (fraction below
  ``nan_expert_threshold``) is demoted to masked-off for that sample.
* The IMERG target keeps its NaNs (both normalized and mm/day copies) —
  losses/metrics mask them.

Reads go through one ``asyncio.gather`` per sample against a lazy
per-worker cache of zarr ``AsyncArray`` handles (pattern from
``physicsnemo/experimental/datapipes/climate/dataset.py``); the cache is
guarded by an owner-PID check so both fork and spawn DataLoader workers
are safe.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Sequence

import cftime
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from zarr.api.asynchronous import open_group as _zarr_open_group_async
from zarr.core.sync import sync as _zarr_sync

from .adapters import ExpertAdapter, ReadRequest
from .index import SampleIndex, build_sample_index
from .stats import ChannelStats
from .truth import TRUTH_OWNER, ImergTruth
from .variables import ChannelLayout

logger = logging.getLogger(__name__)

_EPOCH = cftime.DatetimeGregorian(1900, 1, 1, 0)


def _hours_since_1900(dt: cftime.datetime) -> int:
    return int((dt - _EPOCH).total_seconds() // 3600)


class HindcastMixtureDataset(Dataset):
    """See the module docstring.

    Parameters
    ----------
    experts : adapters in config order — this order defines the ``E`` axis
        and the expert-mask bit positions everywhere downstream.
    truth : the IMERG archive; its grid defines the common ``(H, W)`` grid
        every expert store must match.
    layout / stats : master channel layout and its normalization stats.
    exclude_years : whole init years to drop from inside ``years`` (k-fold CV).
    years, init_months, lead_days : init-time filter (inclusive year range,
        set of months) and the inclusive tau range in whole days.
    min_experts : int or ``"all"`` — minimum live experts per pair.
    zarr_concurrency : per-worker ``zarr.config`` async concurrency (bounds
        both parallel chunk decodes and peak transient memory).
    nan_expert_threshold : demote a live expert whose finite fraction over
        its supplied channels falls below this.
    """

    def __init__(
        self,
        experts: Sequence[ExpertAdapter],
        truth: ImergTruth,
        layout: ChannelLayout,
        stats: ChannelStats,
        *,
        years: tuple[int, int],
        init_months: Sequence[int],
        lead_days: tuple[int, int],
        min_experts: int | str = 1,
        exclude_years: Sequence[int] = (),
        zarr_concurrency: int = 6,
        nan_expert_threshold: float = 0.5,
    ) -> None:
        if not experts:
            raise ValueError("need at least one expert")
        names = [e.name for e in experts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate expert names: {names}")
        self.experts = list(experts)
        self.truth = truth
        self.layout = layout
        self.stats = stats
        self.zarr_concurrency = int(zarr_concurrency)
        self.nan_expert_threshold = float(nan_expert_threshold)

        truth.discover()
        self.lat = np.asarray(truth.lat, dtype=np.float64)
        self.lon = np.asarray(truth.lon, dtype=np.float64)
        for e in self.experts:
            e.discover(self.lat, self.lon)

        # (E, 1 + C) static channel coverage; every master channel must be
        # supplied by at least one expert.
        masks = np.stack([e.channel_mask for e in self.experts])
        uncovered = [
            name
            for i, name in enumerate(layout.channel_names)
            if not masks[:, i].any()
        ]
        if uncovered:
            raise ValueError(
                f"master channels supplied by no expert: {uncovered} — "
                f"remove them from the layout or add an expert that has them"
            )
        self._channel_masks_np = masks
        self.channel_masks = torch.from_numpy(masks.copy())

        self.index: SampleIndex = build_sample_index(
            self.experts,
            truth,
            years=years,
            exclude_years=exclude_years,
            init_months=init_months,
            lead_days=lead_days,
            min_experts=min_experts,
        )

        self._store_roots: dict[str, Path] = {
            e.name: Path(e.root) for e in self.experts
        }
        self._store_roots[TRUTH_OWNER] = Path(truth.root)

        # Lazy per-worker zarr handle caches (never pickled with content).
        self._owner_pid: int | None = None
        self._groups: dict = {}
        self._handles: dict = {}
        self._nan_demotions = 0
        self._empty_samples = 0

    # ------------------------------------------------------------------ #
    # exposure for model / loss / metrics code
    # ------------------------------------------------------------------ #
    @property
    def expert_names(self) -> list[str]:
        return [e.name for e in self.experts]

    @property
    def channel_names(self) -> list[str]:
        return self.layout.channel_names

    @property
    def precip_mean(self) -> float:
        return self.stats.precip_mean

    @property
    def precip_std(self) -> float:
        return self.stats.precip_std

    @property
    def precip_transform(self):
        """Optional log(eps + P) transform (model v1); None = linear.

        When present, the precip channel and ``target`` are
        ``(transform.forward(mm) - mean) / std`` with stats computed in the
        transformed space; ``target_mm`` stays physical mm/day.
        """
        return getattr(self.stats, "precip_transform", None)

    @property
    def pairs(self) -> np.ndarray:
        return self.index.pairs

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------ #
    # per-worker zarr handles + batched async reads
    # ------------------------------------------------------------------ #
    def _reset_handles_if_forked(self) -> None:
        pid = os.getpid()
        if self._owner_pid != pid:
            self._groups = {}
            self._handles = {}
            self._owner_pid = pid
            # Bound parallel chunk decodes (and transient memory) per worker.
            zarr.config.set({"async.concurrency": self.zarr_concurrency})

    def _array(self, key: tuple[str, int, str]):
        if key not in self._handles:
            owner, year, var = key
            gkey = (owner, year)
            if gkey not in self._groups:
                path = str(self._store_roots[owner] / f"{year}.zarr")
                self._groups[gkey] = _zarr_sync(
                    _zarr_open_group_async(path, mode="r")
                )
            self._handles[key] = _zarr_sync(self._groups[gkey].get(var))
        return self._handles[key]

    def _gather(self, reqs: list[ReadRequest]) -> list[np.ndarray]:
        """Resolve all reads of one sample in a single asyncio.gather."""
        self._reset_handles_if_forked()
        arrays = [self._array(r.array_key) for r in reqs]

        async def _one(arr, index: tuple):
            sel = tuple(
                np.asarray(i) if isinstance(i, (list, np.ndarray)) else i
                for i in index
            )
            if any(isinstance(s, np.ndarray) for s in sel):
                return await arr.get_orthogonal_selection(sel)
            return await arr.getitem(sel)

        async def _batch():
            return await asyncio.gather(
                *(_one(a, r.index) for a, r in zip(arrays, reqs))
            )

        return [np.asarray(a, dtype=np.float32) for a in _zarr_sync(_batch())]

    # ------------------------------------------------------------------ #
    # sample assembly
    # ------------------------------------------------------------------ #
    def __getitem__(self, pair_idx: int) -> dict[str, torch.Tensor]:
        row = self.index.pairs[int(pair_idx)]
        tau = int(row["tau"])
        init_row = int(row["init_row"])
        live = self.index.expert_mask(row)

        reqs: list[ReadRequest] = []
        spans: list[tuple[int, int, int]] = []
        for ei, expert in enumerate(self.experts):
            if not live[ei]:
                continue
            year, local_idx = self.index.init_locs[init_row, ei]
            r = expert.plan(int(year), int(local_idx), tau)
            spans.append((ei, len(reqs), len(r)))
            reqs.extend(r)
        reqs.append(self.truth.plan(int(row["imerg_year"]), int(row["imerg_idx"])))
        arrays = self._gather(reqs)

        n_exp, n_chan = len(self.experts), self.layout.num_channels
        h, w = self.lat.size, self.lon.size
        x = np.zeros((n_exp, n_chan, h, w), dtype=np.float32)
        finite_fracs = np.zeros(n_exp, dtype=np.float64)
        for ei, off, n in spans:
            block = self.experts[ei].assemble(arrays[off : off + n], tau)
            supplied = block[self._channel_masks_np[ei]]
            finite_frac = float(np.isfinite(supplied).mean())
            finite_fracs[ei] = finite_frac
            # An expert whose PRECIP channel is entirely non-finite is
            # useless in the mixture regardless of its other channels
            # (e.g. merged-graphcast inits whose wb2-sourced precip lacks
            # the full 24h window at the shortest lead).
            precip_dead = not np.isfinite(block[0]).any()
            if finite_frac < self.nan_expert_threshold or precip_dead:
                live[ei] = False
                self._nan_demotions += 1
                if self._nan_demotions <= 20 or self._nan_demotions % 100 == 0:
                    logger.warning(
                        "expert '%s' demoted for pair %d (finite fraction "
                        "%.2f < %.2f; %d demotions so far)",
                        self.experts[ei].name,
                        pair_idx,
                        finite_frac,
                        self.nan_expert_threshold,
                        self._nan_demotions,
                    )
                continue
            x[ei] = block

        # The model contract requires at least one live expert. The index
        # guarantees that from coordinates, but read-time NaN demotion can
        # still empty a sample (e.g. a graphcast-only init whose day-7 precip
        # window is incomplete). Reinstate the least-bad expert rather than
        # letting the model raise mid-epoch; configure min_lead_day to avoid
        # the situation in the first place.
        if not live.any():
            best = int(np.argmax(finite_fracs))
            live[best] = True
            self._empty_samples += 1
            if self._empty_samples <= 10 or self._empty_samples % 100 == 0:
                logger.error(
                    "pair %d had NO live expert after NaN demotion; "
                    "reinstating '%s' (finite fraction %.2f). %d such samples "
                    "so far — check the expert lead ranges.",
                    pair_idx,
                    self.experts[best].name,
                    finite_fracs[best],
                    self._empty_samples,
                )

        # Optional precip log-transform (model v1) BEFORE standardizing —
        # the stats were computed in the transformed space.
        transform = self.precip_transform
        if transform is not None:
            x[:, 0] = transform.forward(x[:, 0])
        # Normalize -> zero-fill NaN -> re-zero unsupplied channels and dead
        # experts, so "missing" is exactly 0 in z-space (the channel mean).
        x = (x - self.stats.mean[None]) / self.stats.std[None]
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        for ei in range(n_exp):
            x[ei, ~self._channel_masks_np[ei]] = 0.0
            if not live[ei]:
                x[ei] = 0.0

        target_mm = arrays[-1]
        if target_mm.ndim != 2:
            target_mm = target_mm.reshape(h, w)
        target_t = (
            transform.forward(target_mm) if transform is not None else target_mm
        )
        target = (target_t - self.stats.precip_mean) / self.stats.precip_std

        init_dt = cftime.DatetimeGregorian(*self.index.init_keys[init_row])
        init_hours = _hours_since_1900(init_dt)
        return {
            "expert_inputs": torch.from_numpy(x),
            "expert_mask": torch.from_numpy(live.astype(np.float32)),
            "target": torch.from_numpy(target[np.newaxis]),
            "target_mm": torch.from_numpy(target_mm[np.newaxis]),
            "lead_days": torch.tensor(tau, dtype=torch.int64),
            "init_time": torch.tensor(init_hours, dtype=torch.int64),
            "valid_time": torch.tensor(init_hours + tau * 24, dtype=torch.int64),
            "pair_idx": torch.tensor(int(pair_idx), dtype=torch.int64),
        }
