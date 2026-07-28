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

"""Per-expert precipitation metadata and conversion to mm/day.

Every expert stores precipitation differently (6-hourly vs daily lead axis;
accumulation vs mean rate vs cumulative-since-init; m vs mm vs kg m-2 s-1).
``PrecipSpec`` is the config surface describing one expert's convention and
``to_mm_per_day`` converts the raw slabs the adapter read into the daily
accumulation (expressed as mm/day) for the day ending at ``init + tau*24h``.

The day-alignment escape hatch ``day_offset`` shifts which stored leads are
read (in whole days) — set per expert from the results of
``tools/verify_precip_alignment.py``, never in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Multiplicative factor to mm for stored *depth* units.
_DEPTH_TO_MM = {"m": 1000.0, "mm": 1.0}
#: Rate units: kg m-2 s-1 is identically mm/s (1 kg water over 1 m^2 = 1 mm).
_RATE_UNITS = ("kg m-2 s-1", "mm/s")

PrecipAxis = Literal["6h", "daily"]
PrecipKind = Literal["accum", "rate", "cumulative"]


@dataclass(frozen=True)
class PrecipSpec:
    """How one expert stores precipitation.

    Parameters
    ----------
    var : native variable name in the store ("tp", "total_precipitation_24hr").
    axis : which lead axis the variable lives on ("6h" or "daily").
    kind : "accum" (per-step accumulation), "rate" (mean rate over the step),
        or "cumulative" (running total since init; daily axis only).
    units : "m" / "mm" for depths, "kg m-2 s-1" (== "mm/s") for rates.
    step_hours : accumulation/averaging window of one 6h-axis record.
    day_offset : whole-day shift applied to the stored leads that are read,
        pinned empirically per expert (see module docstring).
    """

    var: str
    axis: PrecipAxis
    kind: PrecipKind
    units: str
    step_hours: int = 6
    day_offset: int = 0

    def __post_init__(self) -> None:
        if self.axis not in ("6h", "daily"):
            raise ValueError(f"precip axis must be '6h' or 'daily', got {self.axis!r}")
        if self.kind not in ("accum", "rate", "cumulative"):
            raise ValueError(
                f"precip kind must be accum|rate|cumulative, got {self.kind!r}"
            )
        if self.kind == "cumulative" and self.axis != "daily":
            raise ValueError("cumulative precip is only supported on the daily axis")
        if self.kind == "rate":
            if self.units not in _RATE_UNITS:
                raise ValueError(
                    f"rate precip units must be one of {_RATE_UNITS}, got {self.units!r}"
                )
        elif self.units not in _DEPTH_TO_MM:
            raise ValueError(
                f"depth precip units must be one of {tuple(_DEPTH_TO_MM)}, "
                f"got {self.units!r}"
            )
        if self.axis == "6h" and 24 % self.step_hours != 0:
            raise ValueError(f"step_hours must divide 24, got {self.step_hours}")

    @property
    def steps_per_day(self) -> int:
        return 24 // self.step_hours if self.axis == "6h" else 1

    def lead_values(self, tau_days: int) -> list[int]:
        """The stored lead values to read for target day ``tau`` (after offset).

        Returns lead *hours* for the 6h axis and lead *days* for the daily
        axis, matching each axis's native coordinate units.
        """
        day = tau_days + self.day_offset
        if self.axis == "6h":
            end = day * 24
            return [end - self.step_hours * i for i in range(self.steps_per_day - 1, -1, -1)]
        if self.kind == "cumulative":
            return [day - 1, day]
        return [day]

    def to_mm_per_day(self, fields: np.ndarray) -> np.ndarray:
        """Convert the raw slabs read at ``lead_values`` into mm/day.

        ``fields`` has shape ``(k, H, W)`` with ``k = len(lead_values(tau))``.
        Returns ``(H, W)`` float32; negative results (float noise on
        cumulative differences) are clipped at 0.
        """
        fields = np.asarray(fields, dtype=np.float32)
        if fields.ndim != 3:
            raise ValueError(f"expected (k, H, W) slabs, got shape {fields.shape}")
        expected = self.steps_per_day if self.kind != "cumulative" else 2
        if fields.shape[0] != expected:
            raise ValueError(
                f"{self.kind} precip on the {self.axis} axis needs {expected} "
                f"slabs, got {fields.shape[0]}"
            )
        if self.kind == "rate":
            if self.axis == "6h":
                # Mean rate over each step: depth = sum(rate * step seconds).
                out = fields.sum(axis=0) * (self.step_hours * 3600.0)
            else:
                out = fields[0] * 86400.0
        elif self.kind == "cumulative":
            out = (fields[1] - fields[0]) * _DEPTH_TO_MM[self.units]
        else:  # accum
            out = fields.sum(axis=0) * _DEPTH_TO_MM[self.units]
        return np.clip(out, 0.0, None, out=out)
