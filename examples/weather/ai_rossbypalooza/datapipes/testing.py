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

"""Synthetic tiny-grid zarr stores mimicking the real hindcast/IMERG schemas.

Fixture style follows ``tools/data/hindcast/test_dsi_hindcast_to_formats.py``:
small grids, real coord encodings, value-coded fields so reads are
self-verifying. Used by ``test/recipes/ai_rossbypalooza/`` and the
end-to-end smoke test.

Value coding: every (variable, init, lead) slab is spatially constant with
``coded_value(var_code, init_idx, lead)`` so a test can recompute exactly
what any read should return.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Sequence

import cftime
import numpy as np
import xarray as xr

#: Tiny common "1-degree-like" grid shared by pre-regridded expert stores,
#: IMERG truth, and the dataset tests. lat is N->S like the real stores.
GRID_LAT = np.linspace(3.5, -3.5, 8).astype("float32")
GRID_LON = np.arange(0.0, 360.0, 45.0).astype("float32")

#: "0.25-degree-like" source grid (4x finer) for regrid-tool tests.
FINE_LAT = np.linspace(3.875, -3.875, 32).astype("float32")
FINE_LON = np.arange(0.0, 360.0, 11.25).astype("float32")

CALENDAR = "standard"


def coded_value(var_code: int, init_idx: int, lead: float) -> float:
    """Deterministic, distinct scalar for one (variable, init, lead) slab."""
    return float(var_code) * 10000.0 + float(init_idx) * 100.0 + float(lead)


def resolve(reqs, roots) -> list[np.ndarray]:
    """Resolve adapter ReadRequests against stores on disk (sync zarr).

    Test-side stand-in for the dataset's async gather: ``roots`` maps
    owner name -> archive root holding ``{year}.zarr``.
    """
    import zarr

    out = []
    for r in reqs:
        owner, year, var = r.array_key
        grp = zarr.open_group(str(Path(roots[owner]) / f"{year}.zarr"), mode="r")
        arr = grp[var]
        sel = tuple(np.asarray(i) if isinstance(i, list) else i for i in r.index)
        if any(isinstance(i, np.ndarray) for i in sel):
            out.append(np.asarray(arr.oindex[sel], dtype=np.float32))
        else:
            out.append(np.asarray(arr[sel], dtype=np.float32))
    return out


def _time_encoding(year: int) -> dict:
    return {
        "units": f"hours since {year}-01-01 00:00:00",
        "calendar": CALENDAR,
        "dtype": "int64",
    }


def write_schema_a_store(
    path: str | Path,
    *,
    year: int,
    init_dates: Sequence[tuple[int, int]],
    vars_6h: Sequence[str] = ("2t", "z_500"),
    vars_daily: Sequence[str] = ("tp",),
    lead_hours: Sequence[int] = tuple(range(168, 361, 6)),
    lead_days: Sequence[int] = tuple(range(7, 16)),
    lat: np.ndarray = GRID_LAT,
    lon: np.ndarray = GRID_LON,
    var_codes: Optional[dict[str, int]] = None,
) -> Path:
    """Write a tiny DSI-schema store (two lead axes, flat channel names).

    ``init_dates`` are (month, day) at 00Z in ``year``. All variables are
    value-coded via :func:`coded_value`; ``var_codes`` overrides the default
    enumeration order (6h vars first, then daily).
    """
    path = Path(path)
    if var_codes is None:
        var_codes = {v: i for i, v in enumerate([*vars_6h, *vars_daily])}
    inits = [
        cftime.DatetimeGregorian(year, m, d, 0) for (m, d) in init_dates
    ]
    lead_hours = np.asarray(lead_hours, dtype="int64")
    lead_days = np.asarray(lead_days, dtype="int64")
    n_init, n6, nd = len(inits), lead_hours.size, lead_days.size
    n_lat, n_lon = lat.size, lon.size

    coords = {
        "init_time": ("init_time", np.asarray(inits)),
        "lat": ("lat", np.asarray(lat, dtype="float32")),
        "lon": ("lon", np.asarray(lon, dtype="float32")),
    }
    if vars_6h:
        coords["prediction_timedelta"] = ("prediction_timedelta", lead_hours)
    if vars_daily:
        coords["prediction_timedelta_daily"] = (
            "prediction_timedelta_daily",
            lead_days,
        )
    data_vars = {}
    encoding: dict = {"init_time": _time_encoding(year)}
    for v in vars_6h:
        arr = np.empty((n_init, n6, n_lat, n_lon), dtype="float32")
        for i in range(n_init):
            for j, lh in enumerate(lead_hours):
                arr[i, j] = coded_value(var_codes[v], i, int(lh))
        data_vars[v] = (("init_time", "prediction_timedelta", "lat", "lon"), arr)
        encoding[v] = {"chunks": (1, n6, n_lat, n_lon), "dtype": "float32"}
    for v in vars_daily:
        arr = np.empty((n_init, nd, n_lat, n_lon), dtype="float32")
        for i in range(n_init):
            for j, ld in enumerate(lead_days):
                arr[i, j] = coded_value(var_codes[v], i, int(ld))
        data_vars[v] = (("init_time", "prediction_timedelta_daily", "lat", "lon"), arr)
        encoding[v] = {"chunks": (1, nd, n_lat, n_lon), "dtype": "float32"}

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "hindcast_schema_version": "1.0",
            "model": path.parent.name or "test_model",
            "source_dataset": "test",
            "calendar": CALENDAR,
            "n_init": n_init,
            "lead_window_hours": (
                [int(lead_hours[0]), int(lead_hours[-1])] if n6 else []
            ),
            "lead_window_days": (
                [int(lead_days[0]), int(lead_days[-1])] if nd else []
            ),
            "channel_variables_6h": list(vars_6h),
            "channel_variables_daily": list(vars_daily),
            "diagnostic_variables": [v for v in [*vars_6h, *vars_daily] if v == "tp"],
            "note": "synthetic test store",
            "generator": "datapipes/testing.py",
        },
    )
    if "prediction_timedelta" in ds.coords:
        ds["prediction_timedelta"].attrs.update(
            {"units": "hours", "long_name": "forecast lead time"}
        )
    if "prediction_timedelta_daily" in ds.coords:
        ds["prediction_timedelta_daily"].attrs.update(
            {"units": "days", "long_name": "forecast lead time (daily)"}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    return path


def write_schema_b_store(
    path: str | Path,
    *,
    year: int,
    init_dates: Sequence[tuple[int, int]] = ((6, 1), (6, 5), (6, 9)),
    surface_vars: Sequence[str] = ("2m_temperature",),
    diagnostic_vars: Sequence[str] = ("total_precipitation_24hr",),
    upper_vars: Sequence[str] = ("geopotential",),
    pressure_levels: Sequence[float] = (850.0, 500.0),
    n_lead: int = 17,
    lat: np.ndarray = GRID_LAT,
    lon: np.ndarray = GRID_LON,
    var_codes: Optional[dict[str, int]] = None,
) -> Path:
    """Write a tiny consolidated-schema store (lead_time day index, 3-D upper).

    Upper-air slabs are additionally offset by ``level_index`` so tests can
    verify by-value level selection:
    ``value = coded_value(code, i, lead) + level_index``.
    """
    path = Path(path)
    all_vars = [*surface_vars, *diagnostic_vars, *upper_vars]
    if var_codes is None:
        var_codes = {v: 50 + i for i, v in enumerate(all_vars)}
    inits = [cftime.DatetimeGregorian(year, m, d, 0) for (m, d) in init_dates]
    levels = np.asarray(pressure_levels, dtype="float32")
    n_init, n_lev = len(inits), levels.size
    n_lat, n_lon = lat.size, lon.size

    coords = {
        "init_time": ("init_time", np.asarray(inits)),
        "lead_time": ("lead_time", np.arange(n_lead, dtype="int32")),
        "pressure_level": ("pressure_level", levels),
        "lat": ("lat", np.asarray(lat, dtype="float32")),
        "lon": ("lon", np.asarray(lon, dtype="float32")),
    }
    data_vars = {}
    encoding: dict = {"init_time": _time_encoding(year)}
    for v in [*surface_vars, *diagnostic_vars]:
        arr = np.empty((n_init, n_lead, n_lat, n_lon), dtype="float32")
        for i in range(n_init):
            for d in range(n_lead):
                arr[i, d] = coded_value(var_codes[v], i, d)
        data_vars[v] = (("init_time", "lead_time", "lat", "lon"), arr)
        encoding[v] = {"chunks": (1, n_lead, n_lat, n_lon), "dtype": "float32"}
    for v in upper_vars:
        arr = np.empty((n_init, n_lead, n_lev, n_lat, n_lon), dtype="float32")
        for i in range(n_init):
            for d in range(n_lead):
                for li in range(n_lev):
                    arr[i, d, li] = coded_value(var_codes[v], i, d) + li
        data_vars[v] = (
            ("init_time", "lead_time", "pressure_level", "lat", "lon"),
            arr,
        )
        encoding[v] = {"chunks": (1, n_lead, n_lev, n_lat, n_lon), "dtype": "float32"}

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "hindcast_schema_version": "1.0",
            "model": path.parent.name or "test_model",
            "source_dataset": "test",
            "calendar": CALENDAR,
            "lead_time_hours": 24,
            "n_lead": n_lead,
            "surface_variables": list(surface_vars),
            "diagnostic_variables": list(diagnostic_vars),
            "upper_air_variables": list(upper_vars),
            "pressure_upper_air_variables": list(upper_vars),
            "sigma_upper_air_variables": [],
            "constant_boundary_variables": [],
            "varying_boundary_variables": [],
            "generator": "datapipes/testing.py",
        },
    )
    ds["lead_time"].attrs.update({"units": "days"})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    return path


def write_imerg_store(
    path: str | Path,
    *,
    year: int,
    months: Sequence[int] = (5, 6, 7, 8, 9, 10),
    gap_days: Sequence[tuple[int, int]] = (),
    lat: np.ndarray = GRID_LAT,
    lon: np.ndarray = GRID_LON,
    value_fn=None,
) -> Path:
    """Write a tiny IMERG-schema truth store (daily precip, possible gaps).

    Default value coding: ``day_of_year + 0.5`` mm/day, spatially constant.
    """
    path = Path(path)
    days = []
    for m in months:
        d = cftime.DatetimeGregorian(year, m, 1, 0)
        while d.month == m:
            if (d.month, d.day) not in tuple(gap_days):
                days.append(d)
            d = d + datetime.timedelta(days=1)
    n_time, n_lat, n_lon = len(days), lat.size, lon.size
    arr = np.empty((n_time, n_lat, n_lon), dtype="float32")
    for i, d in enumerate(days):
        arr[i] = (
            value_fn(d) if value_fn is not None else float(d.dayofyr) + 0.5
        )
    ds = xr.Dataset(
        {"total_precipitation_24hr": (("time", "lat", "lon"), arr)},
        coords={
            "time": ("time", np.asarray(days)),
            "lat": ("lat", np.asarray(lat, dtype="float32")),
            "lon": ("lon", np.asarray(lon, dtype="float32")),
        },
        attrs={
            "dataset_name": "imerg",
            "calendar": CALENDAR,
            "data_timedelta_hours": 24,
            "surface_variables": [],
            "constant_boundary_variables": [],
            "varying_boundary_variables": [],
            "diagnostic_variables": ["total_precipitation_24hr"],
            "pressure_upper_air_variables": [],
            "sigma_upper_air_variables": [],
            "era5_zarr_schema_version": "1.0",
        },
    )
    ds["total_precipitation_24hr"].attrs["units"] = "mm/day"
    encoding = {
        "time": _time_encoding(year),
        "total_precipitation_24hr": {
            "chunks": (1, n_lat, n_lon),
            "dtype": "float32",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    return path


def write_stats_store(
    path: str | Path,
    *,
    surface: dict[str, tuple[float, float]],
    upper: dict[str, dict[float, tuple[float, float]]] | None = None,
    level_units: str = "Pa",
    log_epsilon: float | None = None,
    log_units: str = "m",
) -> Path:
    """Write a combined normalization-stats zarr (``stat`` coord {mean,std}).

    ``upper`` maps canonical name -> {level_hPa: (mean, std)}; the coord is
    written in ``level_units`` ("Pa" exercises the auto-detect divide-by-100).
    ``log_epsilon`` marks the store as model-v1 log-space precip stats.
    """
    path = Path(path)
    data_vars: dict = {}
    coords: dict = {"stat": ("stat", np.array(["mean", "std"]))}
    for name, (mu, sd) in surface.items():
        data_vars[name] = (("stat",), np.array([mu, sd], dtype="float64"))
    if upper:
        levels_hpa = sorted({lv for spec in upper.values() for lv in spec})
        scale = 100.0 if level_units == "Pa" else 1.0
        coords["pressure_level"] = (
            "pressure_level",
            np.array([lv * scale for lv in levels_hpa], dtype="float64"),
        )
        for name, spec in upper.items():
            arr = np.full((2, len(levels_hpa)), np.nan, dtype="float64")
            for j, lv in enumerate(levels_hpa):
                if lv in spec:
                    arr[0, j], arr[1, j] = spec[lv]
            data_vars[name] = (("stat", "pressure_level"), arr)
    attrs: dict = {"schema_version": "1.0"}
    if log_epsilon is not None:
        attrs.update(
            {"transform": "log", "log_epsilon": float(log_epsilon),
             "log_units": log_units}
        )
    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True)
    return path
