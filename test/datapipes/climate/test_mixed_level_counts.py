# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mixed sigma + pressure level counts in ClimateZarrDataset.

The original PLASIM datapipe enforced equal sigma_level and pressure_level
counts because PanguPlasim concatenates them along the variable axis.
Phase 7 (SFNO support) relaxes that — SFNO doesn't care about level
geometry, so the dataset must handle unequal counts gracefully.

Contract under test:

* When sigma + pressure level counts match (the common PLASIM case), the
  sample dict has a single ``upper_air_in`` key (concat along var axis), plus
  the separate ``upper_air_sigma_in`` + ``upper_air_pressure_in`` keys.
* When counts differ, ``upper_air_in`` is OMITTED; consumers must read the
  separate keys.
* The dataset does NOT raise at ``__init__`` time — that level-count opinion
  is now the model's to make.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.datapipes.climate import ClimateZarrDataset


def _write_zarr(path: Path, *, n_sigma: int, n_pressure: int, n_time: int = 8):
    """Synthetic PLASIM-shape Zarr with configurable sigma + pressure counts."""
    base = cftime.DatetimeProlepticGregorian(100, 1, 1, 0, 0, 0)
    times = [base + timedelta(hours=6 * i) for i in range(n_time)]
    rng = np.random.default_rng(0)
    H, W = 4, 8
    sigma_lv = np.linspace(0.1, 0.9, n_sigma, dtype="float32")
    pressure_lv = np.linspace(5000, 100000, n_pressure, dtype="float32")
    coords = {
        "time": ("time", times),
        "lat": ("lat", np.linspace(-87.5, 87.5, H, dtype="float32")),
        "lon": ("lon", np.linspace(0, 360 * (W - 1) / W, W, dtype="float32")),
        "sigma_level": ("sigma_level", sigma_lv),
        "pressure_level": ("pressure_level", pressure_lv),
    }
    ds = xr.Dataset(
        {
            "pl": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "tas": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "lsm": (("lat", "lon"), rng.standard_normal((H, W), dtype="float32")),
            "sg": (("lat", "lon"), rng.standard_normal((H, W), dtype="float32")),
            "z0": (("lat", "lon"), rng.standard_normal((H, W), dtype="float32")),
            "sst": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "rsdt": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "sic": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "pr_6h": (("time", "lat", "lon"), rng.standard_normal((n_time, H, W), dtype="float32")),
            "ta": (
                ("time", "sigma_level", "lat", "lon"),
                rng.standard_normal((n_time, n_sigma, H, W), dtype="float32"),
            ),
            "zg": (
                ("time", "pressure_level", "lat", "lon"),
                rng.standard_normal((n_time, n_pressure, H, W), dtype="float32"),
            ),
        },
        coords=coords,
        attrs={
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


def test_equal_counts_emits_concat_key(tmp_path):
    """Common case: sigma_levels == pressure_levels → `upper_air_in` is concat."""
    path = tmp_path / "equal.zarr"
    _write_zarr(path, n_sigma=3, n_pressure=3)
    ds = ClimateZarrDataset(path)
    sample = ds[0]
    assert "upper_air_in" in sample
    assert sample["upper_air_in"].shape == (2, 3, 4, 8)  # (n_vars=2, n_levels=3, H, W)
    assert "upper_air_sigma_in" in sample
    assert "upper_air_pressure_in" in sample


def test_mixed_counts_omits_concat_key(tmp_path):
    """SFNO-style: sigma=4, pressure=10 → no `upper_air_in`, separate keys only."""
    path = tmp_path / "mixed.zarr"
    _write_zarr(path, n_sigma=4, n_pressure=10)
    ds = ClimateZarrDataset(path)
    sample = ds[0]
    assert "upper_air_in" not in sample
    assert sample["upper_air_sigma_in"].shape == (1, 4, 4, 8)
    assert sample["upper_air_pressure_in"].shape == (1, 10, 4, 8)


def test_mixed_counts_does_not_raise_at_init(tmp_path):
    """Constructor stays silent on mixed counts (Phase 7 relaxation)."""
    path = tmp_path / "mixed_init.zarr"
    _write_zarr(path, n_sigma=2, n_pressure=5)
    # No exception expected — model-level concerns are now model-side.
    ds = ClimateZarrDataset(path)
    assert ds._upper_air_concat_supported is False


def test_target_keys_mirror_input_keys(tmp_path):
    """The (start, lead) tuple path also propagates mixed-count keys to targets."""
    path = tmp_path / "mixed_target.zarr"
    _write_zarr(path, n_sigma=3, n_pressure=7)
    ds = ClimateZarrDataset(path)
    sample = ds[(0, 1)]
    assert "target_upper_air" not in sample
    assert "target_upper_air_sigma" in sample
    assert "target_upper_air_pressure" in sample
    assert sample["target_upper_air_sigma"].shape == sample["upper_air_sigma_in"].shape
    assert sample["target_upper_air_pressure"].shape == sample["upper_air_pressure_in"].shape
