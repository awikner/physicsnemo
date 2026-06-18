# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PLASIM normalization Zarr converter.

The converter takes a PanguWeather-style mean+std NetCDF pair and emits one
Zarr store with the unified ai-rossby schema (``stat`` coord ∈ {mean, std},
``sigma_level`` + ``pressure_level`` coords, one var per source channel).

Tests build small synthetic NetCDFs in tmp_path so they can run on the login
node without any Delta fixture; one further test exercises the real PLASIM
fixture if it's been staged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_TOOLS_DIR = Path(__file__).resolve().parents[4] / "tools" / "data"
sys.path.insert(0, str(_TOOLS_DIR))

from _common.normalization import (  # noqa: E402
    NORMALIZATION_SCHEMA_VERSION,
    build_normalization_dataset,
    write_normalization_zarr,
)


def _synthetic_mean_std(
    sigma_levels=(0.0383, 0.1191, 0.21085, 0.31685),
    pressure_levels=(5000.0, 10000.0, 50000.0, 100000.0),
):
    """Build a tiny PanguWeather-style mean+std .nc-like pair in memory.

    Mirrors the PLASIM convention: ``Z_2`` for sigma, ``Z`` for pressure,
    upper-air vars on one of the two level systems, surface vars scalar.
    """
    coords = {
        "Z_2": ("Z_2", np.asarray(sigma_levels, dtype="float64")),
        "Z": ("Z", np.asarray(pressure_levels, dtype="float64")),
    }
    mean_vars = {
        "ta": xr.DataArray(np.arange(len(sigma_levels), dtype="float32"), dims=("Z_2",)),
        "zg": xr.DataArray(np.linspace(0, 1, len(pressure_levels), dtype="float32"), dims=("Z",)),
        "tas": xr.DataArray(np.float32(277.0)),
        "lsm": xr.DataArray(np.float32(0.5)),
    }
    std_vars = {
        "ta": xr.DataArray(
            np.linspace(1.0, 5.0, len(sigma_levels), dtype="float32"), dims=("Z_2",)
        ),
        "zg": xr.DataArray(
            np.linspace(100.0, 500.0, len(pressure_levels), dtype="float32"), dims=("Z",)
        ),
        "tas": xr.DataArray(np.float32(22.0)),
        "lsm": xr.DataArray(np.float32(0.4)),
    }
    return xr.Dataset(mean_vars, coords=coords), xr.Dataset(std_vars, coords=coords)


def test_synthetic_build_layout(tmp_path):
    mean_ds, std_ds = _synthetic_mean_std()
    out = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
    )

    # Coord renames.
    assert list(out.coords["stat"].values) == ["mean", "std"]
    assert out.sizes["sigma_level"] == 4
    assert out.sizes["pressure_level"] == 4

    # Sigma-only var: (stat, sigma_level).
    assert out["ta"].dims == ("stat", "sigma_level")
    assert out["ta"].sel(stat="mean").shape == (4,)
    np.testing.assert_allclose(
        out["ta"].sel(stat="mean").values, mean_ds["ta"].values, atol=1e-6
    )
    np.testing.assert_allclose(
        out["ta"].sel(stat="std").values, std_ds["ta"].values, atol=1e-6
    )

    # Pressure-only var.
    assert out["zg"].dims == ("stat", "pressure_level")
    np.testing.assert_allclose(
        out["zg"].sel(stat="std").values, std_ds["zg"].values, atol=1e-6
    )

    # Scalar vars: (stat,) only.
    assert out["tas"].dims == ("stat",)
    assert float(out["tas"].sel(stat="mean").values) == pytest.approx(277.0)
    assert float(out["lsm"].sel(stat="std").values) == pytest.approx(0.4)


def test_round_trip_zarr_write_read(tmp_path):
    mean_ds, std_ds = _synthetic_mean_std()
    out = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
    )
    out_path = tmp_path / "norm.zarr"
    write_normalization_zarr(
        out,
        out_path,
        source_mean=Path("synthetic_mean.nc"),
        source_std=Path("synthetic_std.nc"),
    )
    # Read it back.
    ds = xr.open_zarr(out_path)
    assert set(ds.data_vars) == set(out.data_vars)
    assert ds.attrs["normalization_schema_version"] == NORMALIZATION_SCHEMA_VERSION
    assert ds.attrs["source_mean"] == "synthetic_mean.nc"
    assert ds.attrs["source_std"] == "synthetic_std.nc"
    assert ds.attrs["coord_convention"] == "ai_rossby_v1"
    np.testing.assert_allclose(
        ds["ta"].sel(stat="std").values, std_ds["ta"].values, atol=1e-6
    )


def test_mismatched_variable_sets_raise(tmp_path):
    mean_ds, std_ds = _synthetic_mean_std()
    std_ds = std_ds.drop_vars("lsm")
    with pytest.raises(ValueError, match="variable sets differ"):
        build_normalization_dataset(
            mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
        )


def test_mismatched_per_var_dims_raise(tmp_path):
    mean_ds, std_ds = _synthetic_mean_std()
    # Mean has ta on sigma, but std now has ta on pressure.
    std_ds["ta"] = xr.DataArray(
        np.zeros(std_ds.sizes["Z"], dtype="float32"), dims=("Z",)
    )
    with pytest.raises(ValueError, match="mean dims .* != std dims"):
        build_normalization_dataset(
            mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
        )


def test_year_range_parsed_from_filename(tmp_path):
    mean_ds, std_ds = _synthetic_mean_std()
    out = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
    )
    out_path = tmp_path / "norm.zarr"
    write_normalization_zarr(
        out,
        out_path,
        source_mean=Path("data_12-132_mean_sigma.nc"),
        source_std=Path("data_12-132_std_sigma.nc"),
    )
    ds = xr.open_zarr(out_path)
    assert ds.attrs["source_year_range"] == "12-132"


# ---------------------------------------------------------------------------
# Real PLASIM fixture — exercises the actual sim52 mean/std .nc files.
# ---------------------------------------------------------------------------
_STATS_DIR = Path(
    "/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data"
)
_MEAN = _STATS_DIR / "data_12-132_mean_sigma.nc"
_STD = _STATS_DIR / "data_12-132_std_sigma.nc"
_HAS_REAL = _MEAN.exists() and _STD.exists()


@pytest.mark.skipif(not _HAS_REAL, reason="real PLASIM stats fixture missing")
def test_real_plasim_fixture_round_trip(tmp_path):
    mean_ds = xr.open_dataset(_MEAN)
    std_ds = xr.open_dataset(_STD)
    out = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
    )
    out_path = tmp_path / "real_norm.zarr"
    write_normalization_zarr(
        out, out_path, source_mean=_MEAN, source_std=_STD, overwrite=True
    )
    ds = xr.open_zarr(out_path)
    # 39 source vars covering both level systems + scalars.
    assert len(ds.data_vars) == 39
    assert ds.sizes["sigma_level"] == 10
    assert ds.sizes["pressure_level"] == 13
    # Spot-check the surface temperature mean.
    assert float(ds["tas"].sel(stat="mean").values) > 200.0
    assert float(ds["tas"].sel(stat="std").values) > 0.0
