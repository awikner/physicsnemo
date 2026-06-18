# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the three boundary-substitution modes of ClimateZarrDataset.

The dataset class supports three ways of routing varying-boundary reads:

1. **Inline** (default): no boundary store; varying boundaries come from the
   prognostic store at the same time index.
2. **Single-year static**: ``boundary_zarr_path=<one_store>`` — every
   prognostic time index reads from that one store at the same time index.
3. **Yearly-repeating cycle**: ``yearly_repeating_boundary=True`` plus
   ``leap_boundary_zarr_path`` + ``non_leap_boundary_zarr_path``. The dataset
   picks the leap or non-leap store based on
   ``cftime.is_leap_year(year, calendar)``, then maps the prognostic time
   into the boundary store via ``(dayofyear-1) * steps_per_day + hour /
   timedelta_hours``. This is critical for multi-year training across the
   leap-year boundary — PLASIM's repeating SST/SIC cycle must align with the
   model time step exactly, and the same mechanism applies to ERA5 / E3SM
   when run with a fixed-climate boundary.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.datapipes.climate import ClimateZarrDataset


def _write_prognostic_store(path: Path, *, year: int, n_time: int = 8):
    """Write a tiny prognostic Zarr store at year ``year`` with 6-h cadence."""
    import cftime

    base = cftime.DatetimeProlepticGregorian(year, 1, 1, 0, 0, 0)
    times = [base + timedelta(hours=6 * i) for i in range(n_time)]
    rng = np.random.default_rng(year)
    H, W = 4, 8

    ds = xr.Dataset(
        {
            # prognostic surface
            "pl": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "tas": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32") + 280),
            # constants
            "lsm": (("lat", "lon"), rng.standard_normal((H, W), dtype="float32")),
            # sigma upper-air
            "ta": (
                ("time", "sigma_level", "lat", "lon"),
                rng.standard_normal((n_time, 2, H, W), dtype="float32"),
            ),
            # boundary-source vars (filled with the year as a marker — that way
            # we can assert which store a read came from)
            "sst": (("time", "lat", "lon"), np.full((n_time, H, W), float(year), dtype="float32")),
            "rsdt": (("time", "lat", "lon"), np.full((n_time, H, W), float(year), dtype="float32")),
            "sic": (("time", "lat", "lon"), np.full((n_time, H, W), float(year), dtype="float32")),
            # diagnostic
            "pr_6h": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
        },
        coords={
            "time": ("time", times),
            "lat": ("lat", np.linspace(-87.5, 87.5, H, dtype="float32")),
            "lon": ("lon", np.linspace(0, 360 * (W - 1) / W, W, dtype="float32")),
            "sigma_level": ("sigma_level", np.asarray([0.1, 0.9], dtype="float32")),
        },
        attrs={
            "calendar": "proleptic_gregorian",
            "data_timedelta_hours": 6,
            "surface_variables": ["pl", "tas"],
            "constant_boundary_variables": ["lsm"],
            "varying_boundary_variables": ["sst", "rsdt", "sic"],
            "diagnostic_variables": ["pr_6h"],
            "pressure_upper_air_variables": [],
            "sigma_upper_air_variables": ["ta"],
        },
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=3)


def _write_boundary_store(
    path: Path,
    *,
    n_time: int,
    sentinel: float,
):
    """Write a single-year boundary Zarr with `sst`, `rsdt`, `sic` filled with `sentinel`.

    No `time` dim coord is needed for boundary stores — the dataset routes by
    integer time index (computed from day-of-year + hour-of-day) into the
    array directly. So this fixture just needs the per-var arrays at the
    expected (n_time, lat, lon) shape.
    """
    H, W = 4, 8
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), np.full((n_time, H, W), sentinel, dtype="float32")),
            "rsdt": (("time", "lat", "lon"), np.full((n_time, H, W), sentinel, dtype="float32")),
            "sic": (("time", "lat", "lon"), np.full((n_time, H, W), sentinel, dtype="float32")),
        },
        coords={
            "lat": ("lat", np.linspace(-87.5, 87.5, H, dtype="float32")),
            "lon": ("lon", np.linspace(0, 360 * (W - 1) / W, W, dtype="float32")),
        },
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=3)


@pytest.fixture
def prognostic_leap_year(tmp_path):
    """Prognostic store for leap year 2024 (8 samples, Jan 1 00:00..Jan 2 18:00)."""
    path = tmp_path / "prog_leap.zarr"
    _write_prognostic_store(path, year=2024, n_time=8)
    return path


@pytest.fixture
def prognostic_non_leap_year(tmp_path):
    """Prognostic store for non-leap year 2023 (8 samples)."""
    path = tmp_path / "prog_nonleap.zarr"
    _write_prognostic_store(path, year=2023, n_time=8)
    return path


@pytest.fixture
def leap_boundary(tmp_path):
    path = tmp_path / "boundary_leap.zarr"
    _write_boundary_store(path, n_time=366 * 4, sentinel=999.0)
    return path


@pytest.fixture
def non_leap_boundary(tmp_path):
    path = tmp_path / "boundary_nonleap.zarr"
    _write_boundary_store(path, n_time=365 * 4, sentinel=888.0)
    return path


@pytest.fixture
def single_year_boundary(tmp_path):
    """A non-cycling, single-year boundary store sized to the prognostic record."""
    path = tmp_path / "boundary_single.zarr"
    _write_boundary_store(path, n_time=8, sentinel=777.0)
    return path


# ---------------------------------------------------------------------------
# Mode 1: inline — no boundary store kwargs.
# ---------------------------------------------------------------------------
def test_inline_boundary_reads_from_prognostic_store(prognostic_leap_year):
    """No boundary kwargs → varying boundaries come from the prognostic store
    at the same time index. The prognostic fixture fills sst/rsdt/sic with the
    year value (2024.0); inline reads should see that."""
    ds = ClimateZarrDataset(prognostic_leap_year)
    sample = ds[0]
    assert torch.all(sample["varying_boundary"] == 2024.0)


# ---------------------------------------------------------------------------
# Mode 2: single-year static boundary store.
# ---------------------------------------------------------------------------
def test_single_year_boundary_store_routes_reads_through_it(
    prognostic_leap_year, single_year_boundary
):
    ds = ClimateZarrDataset(
        prognostic_leap_year, boundary_zarr_path=single_year_boundary
    )
    sample = ds[0]
    # The boundary store's sentinel (777) replaces the prognostic store's year.
    assert torch.all(sample["varying_boundary"] == 777.0)
    assert not torch.all(sample["varying_boundary"] == 2024.0)


# ---------------------------------------------------------------------------
# Mode 3: yearly-repeating cycle — leap vs non-leap routing.
# ---------------------------------------------------------------------------
def test_yearly_repeating_routes_leap_to_leap_store(
    prognostic_leap_year, leap_boundary, non_leap_boundary
):
    """A leap year's prognostic reads should hit the leap (366*4) store."""
    ds = ClimateZarrDataset(
        prognostic_leap_year,
        yearly_repeating_boundary=True,
        leap_boundary_zarr_path=leap_boundary,
        non_leap_boundary_zarr_path=non_leap_boundary,
    )
    sample = ds[0]
    # Sentinel 999.0 == leap boundary store.
    assert torch.all(sample["varying_boundary"] == 999.0)
    # Confirm via the helper method that the routing is correct.
    assert ds._boundary_store_key(0) == "leap"


def test_yearly_repeating_routes_non_leap_to_non_leap_store(
    prognostic_non_leap_year, leap_boundary, non_leap_boundary
):
    """A non-leap year's prognostic reads should hit the non-leap (365*4) store."""
    ds = ClimateZarrDataset(
        prognostic_non_leap_year,
        yearly_repeating_boundary=True,
        leap_boundary_zarr_path=leap_boundary,
        non_leap_boundary_zarr_path=non_leap_boundary,
    )
    sample = ds[0]
    assert torch.all(sample["varying_boundary"] == 888.0)
    assert ds._boundary_store_key(0) == "non_leap"


def test_yearly_repeating_boundary_time_index_mapping(
    prognostic_leap_year, leap_boundary, non_leap_boundary
):
    """Index translation: dayofyear-1 * steps_per_day + hour/timedelta_hours."""
    ds = ClimateZarrDataset(
        prognostic_leap_year,
        yearly_repeating_boundary=True,
        leap_boundary_zarr_path=leap_boundary,
        non_leap_boundary_zarr_path=non_leap_boundary,
    )
    # First prognostic time index in 2024 is Jan 1 00:00 → dayofyear=1, hour=0
    # → boundary index = 0 * 4 + 0 = 0.
    assert ds._boundary_time_index(0) == 0
    # Index 5 in the fixture is Jan 2 06:00 → dayofyear=2, hour=6 → 1*4 + 1 = 5.
    assert ds._boundary_time_index(5) == 5


def test_yearly_repeating_requires_both_leap_and_non_leap_paths(prognostic_leap_year):
    """The dataset rejects yearly_repeating=True without BOTH boundary paths."""
    with pytest.raises(ValueError, match="leap_boundary_zarr_path"):
        ClimateZarrDataset(
            prognostic_leap_year,
            yearly_repeating_boundary=True,
            leap_boundary_zarr_path=None,
            non_leap_boundary_zarr_path=None,
        )
