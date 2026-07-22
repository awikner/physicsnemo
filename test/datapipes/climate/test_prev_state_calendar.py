# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ArchesWeather datapipe additions (all off by default):

* ``prev_state_steps`` — emit ``surface_prev_in`` / ``upper_air_prev_in`` read
  at ``start - k`` (single-store and multi-year, incl. cross-year reads).
* ``calendar_encoding='month_hour'`` — emit ``(month_1_12, hour_0_23)`` instead
  of the default ``(second_of_day, day_of_year)``.
* ``LeadTimePairSampler(min_start=...)`` — reserve room for the previous frame.

These features are additive: with the defaults the dataset behaves exactly as
before (verified by the ``defaults`` test), so the SFNO / Pangu / diffusion
recipes are unaffected.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.datapipes.climate import (
        ClimateZarrDataset,
        ClimateZarrMultiYearDataset,
    )
    from physicsnemo.experimental.datapipes.climate.samplers import LeadTimePairSampler

H, W, NLEV = 4, 8, 3
LEVELS = [500.0, 850.0, 1000.0]
SURFACE = ["2m_temperature", "10m_u_component_of_wind"]
UPPER = ["temperature", "geopotential"]
CONST = ["land_sea_mask", "geopotential_at_surface"]
VARYING = ["sea_surface_temperature"]


def _write_store(path: Path, *, year: int, n_time: int = 40) -> None:
    import cftime

    base = cftime.DatetimeGregorian(year, 1, 1, 0, 0, 0)
    times = [base + timedelta(hours=6 * i) for i in range(n_time)]
    rng = np.random.default_rng(year)
    data = {}
    for v in SURFACE + VARYING:
        # value == year*10000 + time index, so prev reads are exactly checkable.
        arr = np.stack(
            [np.full((H, W), float(year * 10000 + i), dtype="float32") for i in range(n_time)]
        )
        data[v] = (("time", "lat", "lon"), arr)
    for v in UPPER:
        arr = rng.standard_normal((n_time, NLEV, H, W)).astype("float32")
        for i in range(n_time):
            arr[i, 0] = float(year * 10000 + i)
        data[v] = (("time", "pressure_level", "lat", "lon"), arr)
    for v in CONST:
        data[v] = (("lat", "lon"), rng.standard_normal((H, W)).astype("float32"))
    ds = xr.Dataset(
        data,
        coords=dict(
            time=times,
            lat=np.linspace(89.5, -89.5, H).astype("float32"),
            lon=np.linspace(0, 360, W, endpoint=False).astype("float32"),
            pressure_level=np.array(LEVELS, dtype="float32"),
        ),
    )
    ds.attrs.update(
        surface_variables=SURFACE,
        constant_boundary_variables=CONST,
        varying_boundary_variables=VARYING,
        diagnostic_variables=[],
        pressure_upper_air_variables=UPPER,
        sigma_upper_air_variables=[],
        calendar="standard",
        data_timedelta_hours=6,
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=3)


def test_single_store_prev_and_month_hour(tmp_path):
    store = tmp_path / "2000.zarr"
    _write_store(store, year=2000)
    ds = ClimateZarrDataset(
        store, prev_state_steps=4, emit_calendar=True, calendar_encoding="month_hour"
    )
    sample = ds[(8, 4)]
    assert "surface_prev_in" in sample and "upper_air_prev_in" in sample
    assert float(sample["surface_prev_in"][0, 0, 0]) == 2000 * 10000 + 4
    assert float(sample["surface_in"][0, 0, 0]) == 2000 * 10000 + 8
    assert float(sample["upper_air_prev_in"][0, 0, 0, 0]) == 2000 * 10000 + 4
    cal = sample["calendar"]
    assert tuple(cal.shape) == (2,)
    # idx 8 -> 48h from Jan 1 -> Jan 3 00Z -> month 1, hour 0
    assert int(cal[0]) == 1 and int(cal[1]) == 0
    # idx 9 -> 54h -> Jan 3 06Z -> hour 6
    assert int(ds[(9, 4)]["calendar"][1]) == 6


def test_multiyear_cross_year_prev(tmp_path):
    _write_store(tmp_path / "2000.zarr", year=2000)
    _write_store(tmp_path / "2001.zarr", year=2001)
    mds = ClimateZarrMultiYearDataset(
        tmp_path, prev_state_steps=4, emit_calendar=True, calendar_encoding="month_hour"
    )
    # global 40 == start of 2001 (local 0); prev = 36 -> 2000 idx 36.
    sample = mds[(40, 4)]
    assert float(sample["surface_in"][0, 0, 0]) == 2001 * 10000 + 0
    assert float(sample["surface_prev_in"][0, 0, 0]) == 2000 * 10000 + 36


def test_sampler_min_start(tmp_path):
    _write_store(tmp_path / "2000.zarr", year=2000)
    ds = ClimateZarrDataset(tmp_path / "2000.zarr")
    sampler = LeadTimePairSampler(
        dataset_length=len(ds),
        forecast_lead_times=[4],
        num_samples=1000,
        shuffle=True,
        min_start=4,
    )
    starts = [s for s, _ in sampler]
    assert min(starts) >= 4
    assert max(starts) < len(ds) - 4


def test_defaults_unchanged(tmp_path):
    store = tmp_path / "2000.zarr"
    _write_store(store, year=2000)
    ds = ClimateZarrDataset(store)
    s = ds[(2, 1)]
    assert "surface_prev_in" not in s and "calendar" not in s
    # default calendar encoding is still (second_of_day, day_of_year)
    ds_cal = ClimateZarrDataset(store, emit_calendar=True)
    c = ds_cal[(4, 1)]["calendar"]
    assert int(c[0]) == 0 and int(c[1]) == 1  # 24h -> Jan 2 00Z -> sec 0, doy 1
