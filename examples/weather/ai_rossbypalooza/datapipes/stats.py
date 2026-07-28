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

"""Per-channel normalization stats for the master channel layout.

Dynamical predictors use the existing ERA5 normalization store (combined
zarr with a ``stat`` coord {mean, std} — the ``ClimateNormalizer``
convention); the precip channel (experts *and* target) uses shared
IMERG-derived stats from the small store written by
``tools/compute_precip_norm.py``. Levels are matched **by value** with the
``ClimateNormalizer._nearest_indices`` tolerance rule and a hard raise on
any miss; a Pa-vs-hPa level coord is auto-detected (coord max > 2000 => Pa).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from .variables import ChannelLayout, levels_match

logger = logging.getLogger(__name__)

#: Accepted level-dimension names in stats stores, in probe order —
#: mirrors ClimateNormalizer's alias table (transforms.py).
LEVEL_DIMS = ("pressure_level", "level", "Z")


def _open_stats(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if path.suffix == ".zarr" or (path / "zarr.json").exists():
        return xr.open_zarr(path, consolidated=True)
    return xr.open_dataset(path)


def _stat_slices(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Split a combined store on its ``stat`` coord (or pass through)."""
    if "stat" in ds.coords:
        return ds.sel(stat="mean", drop=True), ds.sel(stat="std", drop=True)
    return ds, ds


def _level_values(da: xr.DataArray) -> tuple[str, np.ndarray]:
    for dim in LEVEL_DIMS:
        if dim in da.dims:
            vals = np.asarray(da[dim].values, dtype="float64")
            if np.nanmax(vals) > 2000.0:  # Pa -> hPa
                vals = vals / 100.0
            return dim, vals
    raise ValueError(
        f"variable '{da.name}' has no level dimension among {LEVEL_DIMS} "
        f"(dims: {da.dims})"
    )


class ChannelStats:
    """(mean, std) vectors aligned to a :class:`ChannelLayout`.

    ``mean`` / ``std`` are float32 arrays of shape ``(1 + C, 1, 1)`` ready to
    broadcast over ``(1 + C, H, W)`` expert blocks; ``precip_mean`` /
    ``precip_std`` expose channel 0's scalars for target (de)normalization.
    """

    def __init__(
        self,
        dyn_stats_path: str | Path,
        precip_stats_path: str | Path,
        layout: ChannelLayout,
        *,
        precip_var: str = "total_precipitation_24hr",
    ) -> None:
        self.layout = layout
        n = layout.num_channels
        mean = np.zeros(n, dtype=np.float64)
        std = np.ones(n, dtype=np.float64)

        # ---- precip channel (index 0) from the IMERG stats store
        try:
            pstats = _open_stats(precip_stats_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise FileNotFoundError(
                f"precip stats store {precip_stats_path} is missing or "
                f"unreadable — generate it with "
                f"tools/compute_precip_norm.py ({exc})"
            ) from exc
        with pstats:
            p_mean, p_std = _stat_slices(pstats)
            if precip_var not in p_mean:
                raise KeyError(
                    f"precip stats store lacks '{precip_var}' "
                    f"(has: {sorted(p_mean.data_vars)})"
                )
            mean[0] = float(p_mean[precip_var].values)
            std[0] = float(p_std[precip_var].values)

        # ---- dynamical channels from the ERA5 stats store
        with _open_stats(dyn_stats_path) as dstats:
            d_mean, d_std = _stat_slices(dstats)
            for i, ch in enumerate(layout.channels, start=1):
                if ch.canonical not in d_mean:
                    raise KeyError(
                        f"ERA5 stats store lacks '{ch.canonical}' "
                        f"(has: {sorted(d_mean.data_vars)})"
                    )
                if ch.level_hpa is None:
                    mean[i] = float(d_mean[ch.canonical].values)
                    std[i] = float(d_std[ch.canonical].values)
                else:
                    dim, levels = _level_values(d_mean[ch.canonical])
                    matches = [
                        j
                        for j, lv in enumerate(levels)
                        if levels_match(lv, ch.level_hpa)
                    ]
                    if not matches:
                        raise ValueError(
                            f"no level near {ch.level_hpa} hPa for "
                            f"'{ch.canonical}' in the stats store "
                            f"(available: {levels.tolist()})"
                        )
                    j = matches[0]
                    mean[i] = float(d_mean[ch.canonical].isel({dim: j}).values)
                    std[i] = float(d_std[ch.canonical].isel({dim: j}).values)

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            bad = [
                layout.channel_names[i]
                for i in range(n)
                if not (np.isfinite(mean[i]) and np.isfinite(std[i]))
            ]
            raise ValueError(f"non-finite normalization stats for {bad}")
        if np.any(std <= 0):
            bad = [layout.channel_names[i] for i in range(n) if std[i] <= 0]
            raise ValueError(f"non-positive normalization std for {bad}")

        self.mean = mean.astype(np.float32).reshape(n, 1, 1)
        self.std = std.astype(np.float32).reshape(n, 1, 1)

    @property
    def precip_mean(self) -> float:
        return float(self.mean[0, 0, 0])

    @property
    def precip_std(self) -> float:
        return float(self.std[0, 0, 0])
