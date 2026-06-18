# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`PlasimMultiYearDataset`.

The multi-year composite glues a directory of per-year Zarr sub-stores into a
single ``torch.utils.data.Dataset`` with a contiguous global time index.
Tests use a synthetic in-tmp_path fixture so they run on the login node.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.datapipes.plasim.multiyear import PlasimMultiYearDataset


def _write_synth_year(path: Path, *, year: int, n_time: int = 16):
    """Write a small synthetic PLASIM Zarr store matching the production schema."""
    import cftime
    from datetime import timedelta

    base = cftime.DatetimeProlepticGregorian(year, 1, 1, 0, 0, 0)
    times = [base + timedelta(hours=6 * i) for i in range(n_time)]
    rng = np.random.default_rng(year)
    n_lat, n_lon = 8, 16
    sigma = np.asarray([0.1, 0.5, 0.9], dtype="float32")
    pressure = np.asarray([5000.0, 50000.0, 100000.0], dtype="float32")

    ds = xr.Dataset(
        {
            "tas": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_time, n_lat, n_lon), dtype="float32") + 280,
            ),
            "pl": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_time, n_lat, n_lon), dtype="float32") + 10,
            ),
            "ta": (
                ("time", "sigma_level", "lat", "lon"),
                rng.standard_normal((n_time, len(sigma), n_lat, n_lon), dtype="float32"),
            ),
            "zg": (
                ("time", "pressure_level", "lat", "lon"),
                rng.standard_normal((n_time, len(pressure), n_lat, n_lon), dtype="float32"),
            ),
            "lsm": (
                ("lat", "lon"),
                rng.standard_normal((n_lat, n_lon), dtype="float32"),
            ),
            "sg": (("lat", "lon"), rng.standard_normal((n_lat, n_lon), dtype="float32")),
            "z0": (("lat", "lon"), rng.standard_normal((n_lat, n_lon), dtype="float32")),
            "sst": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_time, n_lat, n_lon), dtype="float32") + 290,
            ),
            "rsdt": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_time, n_lat, n_lon), dtype="float32") + 400,
            ),
            "sic": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_time, n_lat, n_lon), dtype="float32"),
            ),
            "pr_6h": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_time, n_lat, n_lon), dtype="float32"),
            ),
        },
        coords={
            "time": ("time", times),
            "lat": ("lat", np.linspace(-87.5, 87.5, n_lat, dtype="float32")),
            "lon": ("lon", np.linspace(0, 360 * (n_lon - 1) / n_lon, n_lon, dtype="float32")),
            "sigma_level": ("sigma_level", sigma),
            "pressure_level": ("pressure_level", pressure),
        },
        attrs={
            "plasim_zarr_schema_version": "1.0",
            "calendar": "proleptic_gregorian",
            "data_timedelta_hours": 6,
            "surface_variables": ["pl", "tas"],
            "constant_boundary_variables": ["lsm", "sg", "z0"],
            "varying_boundary_variables": ["sst", "rsdt", "sic"],
            "diagnostic_variables": ["pr_6h"],
            "pressure_upper_air_variables": ["zg"],
            "sigma_upper_air_variables": ["ta"],
        },
    )
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=3)


@pytest.fixture
def multi_year_root(tmp_path):
    """Build a 3-year synthetic Zarr archive at tmp_path/multi/{year}.zarr."""
    root = tmp_path / "multi"
    root.mkdir()
    for year in (100, 101, 102):
        _write_synth_year(root / f"{year}.zarr", year=year, n_time=16)
    return root


def test_directory_open_and_concat(multi_year_root):
    ds = PlasimMultiYearDataset(multi_year_root)
    assert len(ds) == 16 * 3
    assert ds.num_levels == 3
    assert ds.upper_air_variable_names == ["ta", "zg"]
    assert ds.horizontal_resolution == (8, 16)
    assert len(ds.sub_datasets) == 3


def test_global_index_maps_correctly(multi_year_root):
    ds = PlasimMultiYearDataset(multi_year_root)
    # First sub-store: indices 0..15
    sub_idx, local = ds._global_to_local(0)
    assert (sub_idx, local) == (0, 0)
    sub_idx, local = ds._global_to_local(15)
    assert (sub_idx, local) == (0, 15)
    # Second sub-store: indices 16..31
    sub_idx, local = ds._global_to_local(16)
    assert (sub_idx, local) == (1, 0)
    sub_idx, local = ds._global_to_local(31)
    assert (sub_idx, local) == (1, 15)
    # Third sub-store
    sub_idx, local = ds._global_to_local(32)
    assert (sub_idx, local) == (2, 0)


def test_same_year_sample_dispatch(multi_year_root):
    ds = PlasimMultiYearDataset(multi_year_root)
    # Read a within-first-year pair: start=2, lead=4 stays in year 0.
    sample = ds[(2, 4)]
    assert sample["surface_in"].shape == (2, 8, 16)
    assert sample["upper_air_in"].shape == (2, 3, 8, 16)
    assert int(sample["time_idx"]) == 2
    assert int(sample["lead_time"]) == 4


def test_cross_year_sample_dispatch(multi_year_root):
    ds = PlasimMultiYearDataset(multi_year_root)
    # start=14 in year 0 (length 16), lead=4 → target is at global 18 == year 1, local 2.
    sample = ds[(14, 4)]
    assert sample["surface_in"].shape == (2, 8, 16)
    assert sample["target_surface"].shape == (2, 8, 16)
    assert int(sample["time_idx"]) == 14
    assert int(sample["lead_time"]) == 4
    # Cross-year: start surface and target surface differ (different rng seeds).
    assert not torch.equal(sample["surface_in"], sample["target_surface"])


def test_out_of_range_raises(multi_year_root):
    ds = PlasimMultiYearDataset(multi_year_root)
    with pytest.raises(IndexError):
        ds._global_to_local(48)


def test_layout_mismatch_raises(tmp_path):
    """If two sub-stores have different channel lists, the composite refuses."""
    root = tmp_path / "multi"
    root.mkdir()
    _write_synth_year(root / "100.zarr", year=100, n_time=8)
    # Build a second store with a different surface var list to trigger the check.
    other_path = root / "101.zarr"
    _write_synth_year(other_path, year=101, n_time=8)
    other = xr.open_zarr(other_path)
    other.attrs["surface_variables"] = ["pl"]  # mismatch: missing "tas"
    other.to_zarr(other_path, mode="w", consolidated=True, zarr_format=3)
    with pytest.raises(ValueError, match="layout field"):
        PlasimMultiYearDataset(root)


def test_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no \\*\\.zarr sub-stores"):
        PlasimMultiYearDataset(empty)


def test_not_a_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        PlasimMultiYearDataset(tmp_path / "does-not-exist")
