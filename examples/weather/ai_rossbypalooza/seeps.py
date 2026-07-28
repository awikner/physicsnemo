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

"""SEEPS (Stable Equitable Error in Probability Space) for daily precip.

Rodwell et al. (2010, QJRMS) formulation, WeatherBench-2-compatible so the
hackathon scores are directly comparable:

* three categories from the local monthly climatology — "dry"
  (precip <= ``dry_threshold_mm``, default 0.25 mm/day), "light", and
  "heavy", with the light/heavy boundary ``t2`` at the climatological 2/3
  quantile of wet days (light is climatologically twice as likely as heavy);
* penalty matrix from the climatological dry probability ``p1``
  (``rows = forecast category, cols = observed category``)::

      1/2 * [[0,                1/(1-p1),   4/(1-p1)],
             [1/p1,             0,          3/(1-p1)],
             [1/p1 + 3/(2+p1),  3/(2+p1),   0       ]]

* gridpoints with ``p1`` outside ``[0.1, 0.85]`` are excluded (standard
  practice — degenerate climates make the score unstable).

Lower is better; 0 is perfect. Report ``1 - SEEPS`` as skill.

The per-(month, gridpoint) ``p1`` / ``t2`` fields come from the small zarr
written by ``tools/compute_seeps_climatology.py``. The streaming accumulator
follows ``examples/weather/ai_rossby/validate.py``'s DDP-safe
update/finalize pattern.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import cftime
import numpy as np
import torch
import torch.distributed as dist
import xarray as xr

#: p1 validity window (Rodwell et al. 2010 / WeatherBench 2).
P1_MIN, P1_MAX = 0.1, 0.85
DEFAULT_DRY_THRESHOLD_MM = 0.25


def seeps_matrix(p1: torch.Tensor) -> torch.Tensor:
    """Penalty matrix ``(..., 3, 3)`` from dry probability ``p1 (...)``.

    Rows = forecast category, columns = observed category (dry/light/heavy).
    """
    z = torch.zeros_like(p1)
    row0 = torch.stack([z, 1.0 / (1.0 - p1), 4.0 / (1.0 - p1)], dim=-1)
    row1 = torch.stack([1.0 / p1, z, 3.0 / (1.0 - p1)], dim=-1)
    row2 = torch.stack(
        [1.0 / p1 + 3.0 / (2.0 + p1), 3.0 / (2.0 + p1), z], dim=-1
    )
    return 0.5 * torch.stack([row0, row1, row2], dim=-2)


def categorize(
    precip_mm: torch.Tensor, t2: torch.Tensor, dry_threshold_mm: float
) -> torch.Tensor:
    """0 = dry, 1 = light, 2 = heavy (int64, same shape as input)."""
    cat = torch.zeros_like(precip_mm, dtype=torch.int64)
    cat[precip_mm > dry_threshold_mm] = 1
    cat[precip_mm > t2] = 2
    return cat


def seeps_penalty(
    pred_mm: torch.Tensor,
    target_mm: torch.Tensor,
    p1: torch.Tensor,
    t2: torch.Tensor,
    *,
    dry_threshold_mm: float = DEFAULT_DRY_THRESHOLD_MM,
) -> torch.Tensor:
    """Pointwise SEEPS penalty; all args broadcastable to a common shape.

    Computed by masked accumulation of the six off-diagonal matrix entries
    (cheaper than materializing the 3x3 matrix per gridpoint).
    """
    pred_mm, target_mm, p1, t2 = torch.broadcast_tensors(
        pred_mm, target_mm, p1, t2
    )
    f = categorize(pred_mm, t2, dry_threshold_mm)
    o = categorize(target_mm, t2, dry_threshold_mm)
    wet = 1.0 / (1.0 - p1)
    dry = 1.0 / p1
    heavy_row = 3.0 / (2.0 + p1)
    s = torch.zeros_like(p1)
    s = torch.where((f == 0) & (o == 1), wet, s)
    s = torch.where((f == 0) & (o == 2), 4.0 * wet, s)
    s = torch.where((f == 1) & (o == 0), dry, s)
    s = torch.where((f == 1) & (o == 2), 3.0 * wet, s)
    s = torch.where((f == 2) & (o == 0), dry + heavy_row, s)
    s = torch.where((f == 2) & (o == 1), heavy_row, s)
    return 0.5 * s


class SeepsClimatology:
    """Loads the ``(month, lat, lon)`` ``p1`` / ``t2`` fields from zarr."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        try:
            ds = xr.open_zarr(path, consolidated=True)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise FileNotFoundError(
                f"SEEPS climatology store {path} is missing or unreadable — "
                f"generate it with tools/compute_seeps_climatology.py ({exc})"
            ) from exc
        with ds:
            for var in ("p1", "t2"):
                if var not in ds:
                    raise KeyError(f"{path} lacks variable '{var}'")
            if ds["p1"].dims != ("month", "lat", "lon"):
                raise ValueError(
                    f"expected p1 dims (month, lat, lon), got {ds['p1'].dims}"
                )
            self.p1 = torch.from_numpy(
                ds["p1"].values.astype(np.float32)
            )  # (12, H, W)
            self.t2 = torch.from_numpy(ds["t2"].values.astype(np.float32))
            self.lat = ds["lat"].values.astype(np.float64)
            self.lon = ds["lon"].values.astype(np.float64)
            self.dry_threshold_mm = float(
                ds.attrs.get("dry_threshold_mm", DEFAULT_DRY_THRESHOLD_MM)
            )

    def to(self, device: torch.device) -> "SeepsClimatology":
        self.p1 = self.p1.to(device)
        self.t2 = self.t2.to(device)
        return self


def _all_reduce_sum(t: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


class StreamingRegionalSEEPS:
    """Per-lead-day regional SEEPS, streaming + DDP-safe.

    ``region_weights (H, W)`` (from :func:`losses.region_weights`) restricts
    and cos-lat-weights the average; gridpoints whose climatological ``p1``
    is outside ``[P1_MIN, P1_MAX]`` for the sample's month are excluded, as
    are non-finite target cells. ``finalize()`` all-reduces and returns the
    ``(n_leads,)`` mean penalty (lower = better; 1 - value = skill).
    """

    def __init__(
        self,
        *,
        n_leads: int,
        climatology: SeepsClimatology,
        region_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.clim = climatology.to(device)
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.score_sum = torch.zeros(n_leads, device=device)
        self.weight_sum = torch.zeros(n_leads, device=device)
        self._valid = (self.clim.p1 >= P1_MIN) & (self.clim.p1 <= P1_MAX)

    @torch.no_grad()
    def update(
        self,
        lead_index: int,
        pred_mm: torch.Tensor,
        target_mm: torch.Tensor,
        months: torch.Tensor,
    ) -> None:
        """Add one batch: ``pred_mm``/``target_mm (B, H, W)``, ``months (B,)``
        the calendar month (1..12) of each sample's *valid* day."""
        m = months.long() - 1
        p1 = self.clim.p1[m]  # (B, H, W)
        t2 = self.clim.t2[m]
        penalty = seeps_penalty(
            pred_mm,
            target_mm,
            p1.clamp(1e-6, 1.0 - 1e-6),
            t2,
            dry_threshold_mm=self.clim.dry_threshold_mm,
        )
        w = (
            self.weights.unsqueeze(0)
            * self._valid[m].float()
            * torch.isfinite(target_mm).float()
        )
        self.score_sum[lead_index] += (penalty * w).sum()
        self.weight_sum[lead_index] += w.sum()

    def finalize(self) -> torch.Tensor:
        _all_reduce_sum(self.score_sum)
        _all_reduce_sum(self.weight_sum)
        return self.score_sum / self.weight_sum.clamp(min=1e-12)


def months_from_hours_since_1900(hours: torch.Tensor) -> torch.Tensor:
    """Calendar months (1..12) for int hours since 1900-01-01 00Z (standard
    calendar) — matches the dataset's ``valid_time`` encoding."""
    epoch = cftime.DatetimeGregorian(1900, 1, 1, 0)
    out = [
        (epoch + datetime.timedelta(hours=int(h))).month for h in hours.tolist()
    ]
    return torch.tensor(out, dtype=torch.int64, device=hours.device)
