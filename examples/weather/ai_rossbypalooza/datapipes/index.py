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

"""Global (init, tau) sample-index construction.

Deterministic and coord-only: runs identically on every DDP rank at dataset
init. The init universe is the *union* over experts (missing experts are
masked per pair, not dropped), filtered by config year/month ranges; each
pair resolves its IMERG record at build time so the hot path can never miss.

Day-alignment convention (see the recipe plan): the sample's precip channel
and the IMERG target both describe ``[init + (tau-1)*24h, init + tau*24h)``
— "the day ending at init + tau*24h". IMERG's record stamped 00Z on day X
covers ``[X, X+1)``, so the target record is stamped ``date(init) + (tau-1)
days``.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Sequence

import cftime
import numpy as np

from .adapters import ExpertAdapter, InitKey
from .truth import ImergTruth

logger = logging.getLogger(__name__)

PAIR_DTYPE = np.dtype(
    [
        ("init_row", np.int32),
        ("tau", np.int16),
        ("expert_bits", np.uint32),
        ("imerg_year", np.int16),
        ("imerg_idx", np.int32),
    ]
)


@dataclass
class SampleIndex:
    """The precomputed pair table plus per-init expert store locations."""

    init_keys: list[InitKey]
    #: (n_inits, n_experts, 2) int32 (store_year, local_idx), -1 if absent
    init_locs: np.ndarray
    #: structured array with PAIR_DTYPE fields, one row per (init, tau) pair
    pairs: np.ndarray

    def __len__(self) -> int:
        return len(self.pairs)

    def expert_mask(self, pair_row: np.void) -> np.ndarray:
        bits = int(pair_row["expert_bits"])
        n = self.init_locs.shape[1]
        return np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)


def build_sample_index(
    experts: Sequence[ExpertAdapter],
    truth: ImergTruth,
    *,
    years: tuple[int, int],
    init_months: Sequence[int],
    lead_days: tuple[int, int],
    min_experts: int | str = 1,
) -> SampleIndex:
    """Enumerate all valid (init, tau) pairs. See the module docstring.

    ``min_experts`` is an int, or the literal ``"all"`` to require every
    configured expert (the intersection / all-experts training mode).
    """
    if len(experts) > 32:
        raise ValueError("expert_bits is uint32; at most 32 experts supported")
    if isinstance(min_experts, str):
        if min_experts != "all":
            raise ValueError(f"min_experts must be an int or 'all', got {min_experts!r}")
        min_live = len(experts)
    else:
        min_live = int(min_experts)
    if not 1 <= min_live <= len(experts):
        raise ValueError(
            f"min_experts={min_live} out of range for {len(experts)} experts"
        )
    months = set(int(m) for m in init_months)
    lo_tau, hi_tau = int(lead_days[0]), int(lead_days[1])
    if lo_tau < 1 or hi_tau < lo_tau:
        raise ValueError(f"bad lead_days range {lead_days}")

    lut = {e.name: e.init_lookup() for e in experts}
    all_keys = sorted({k for table in lut.values() for k in table})
    init_keys = [
        k for k in all_keys if years[0] <= k[0] <= years[1] and k[1] in months
    ]
    imerg = truth.day_lookup()

    # Per-tau expert support is init-independent; precompute it.
    tau_support = {
        tau: [e.lead_supported(tau) for e in experts]
        for tau in range(lo_tau, hi_tau + 1)
    }

    init_locs = np.full((len(init_keys), len(experts), 2), -1, dtype=np.int32)
    rows: list[tuple] = []
    n_no_imerg = 0
    n_few_experts = 0
    for i, key in enumerate(init_keys):
        for ei, e in enumerate(experts):
            loc = lut[e.name].get(key)
            if loc is not None:
                init_locs[i, ei, 0] = loc[0]
                init_locs[i, ei, 1] = loc[1]
        init_dt = cftime.DatetimeGregorian(*key)
        for tau in range(lo_tau, hi_tau + 1):
            # Record stamped date(init) + (tau - 1): covers the 24h ending
            # at init + tau*24h (inits are 00Z).
            day_dt = init_dt + datetime.timedelta(days=tau - 1)
            tkey = (int(day_dt.year), int(day_dt.month), int(day_dt.day))
            loc = imerg.get(tkey)
            if loc is None:
                n_no_imerg += 1
                continue
            bits = 0
            live = 0
            for ei in range(len(experts)):
                if init_locs[i, ei, 0] >= 0 and tau_support[tau][ei]:
                    bits |= 1 << ei
                    live += 1
            if live < min_live:
                n_few_experts += 1
                continue
            rows.append((i, tau, bits, loc[0], loc[1]))

    pairs = np.array(rows, dtype=PAIR_DTYPE)
    logger.info(
        "sample index: %d inits (union), %d pairs "
        "(%d dropped: no IMERG day, %d dropped: < %d live experts)",
        len(init_keys),
        len(pairs),
        n_no_imerg,
        n_few_experts,
        min_live,
    )
    if len(pairs) == 0:
        raise ValueError(
            f"empty sample index: {len(all_keys)} inits across experts, "
            f"{len(init_keys)} after years={years} months={sorted(months)}, "
            f"{n_no_imerg} pairs dropped for missing IMERG days, "
            f"{n_few_experts} for < {min_live} live experts"
        )
    return SampleIndex(init_keys=init_keys, init_locs=init_locs, pairs=pairs)
