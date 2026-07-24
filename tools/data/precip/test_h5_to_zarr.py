#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/data/precip/h5_to_zarr.py.

Synthesizes tiny per-day HDF5 files (a global IMERG-like case using an ERA5
reference-store grid, and an IMD-like regional case with a coordinates.nc),
including a GAP (missing day) and a NaN field, runs the converter, and asserts
the resulting Zarr store's dims / coords / time coord / attrs / NaN handling.

Runnable directly (``python test_h5_to_zarr.py``) or via pytest. Uses only
h5py / xarray / zarr / numpy / cftime -- no physicsnemo import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5_to_zarr  # noqa: E402


def _write_day(dirpath: Path, date_token: str, var: str, field: np.ndarray) -> None:
    """Write one per-day H5 file named ``<date_token>.h5`` (YYYYMMDDHH)."""
    iso = (
        f"{date_token[0:4]}-{date_token[4:6]}-{date_token[6:8]}"
        f"T{date_token[8:10]}:00:00.000000000"
    )
    with h5py.File(dirpath / f"{date_token}.h5", "w") as f:
        g = f.create_group("input")
        g.create_dataset(var, data=field.astype("float32"))
        g.create_dataset("time", data=np.bytes_(iso))


def _make_ref_store(path: Path, n_lat: int, n_lon: int) -> None:
    """Minimal ERA5-like reference store carrying just lat/lon (N->S, 0..359)."""
    lat = np.linspace(89.5, -89.5, n_lat, dtype="float32")
    lon = np.linspace(0.0, 360.0 * (n_lon - 1) / n_lon, n_lon, dtype="float32")
    ds = xr.Dataset(
        {"dummy": (("time", "lat", "lon"), np.zeros((1, n_lat, n_lon), "float32"))},
        coords={"time": ("time", [0]), "lat": ("lat", lat), "lon": ("lon", lon)},
    )
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True)


def _run(argv: list[str]) -> None:
    assert h5_to_zarr.main(argv) == 0


def test_imerg_global_grid(tmp_path: Path) -> None:
    """Global IMERG-like case: ERA5-grid coords from --ref-store, gap day, no NaN.

    Field is built with a tropical maximum so the orientation check passes.
    """
    n_lat, n_lon = 18, 36  # tiny global grid, N->S order
    lat = np.linspace(89.5, -89.5, n_lat, dtype="float32")
    inp = tmp_path / "imerg_in"
    inp.mkdir()

    # Three days in 2000 with a GAP (2000-06-01, -02, then jump to -05).
    tokens = ["2000060100", "2000060200", "2000060500"]
    for k, tok in enumerate(tokens):
        # Tropical-peaked field: high near |lat|=0, low at poles + a per-day offset.
        band = np.cos(np.deg2rad(lat)) ** 2 * 10.0  # peak at equator
        field = np.tile(band[:, None], (1, n_lon)) + float(k)
        _write_day(inp, tok, "total_precipitation_24hr", field)

    ref = tmp_path / "ref_2000.zarr"
    _make_ref_store(ref, n_lat, n_lon)

    out = tmp_path / "imerg_2000.zarr"
    _run([
        "--input-dir", str(inp),
        "--source-var", "total_precipitation_24hr",
        "--out-var", "total_precipitation_24hr",
        "--dataset-name", "imerg",
        "--year", "2000",
        "--units", "mm/day",
        "--ref-store", str(ref),
        "--out-store", str(out),
    ])

    ds = xr.open_zarr(
        out, consolidated=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    try:
        # dims / coords
        assert ds.sizes["time"] == 3
        assert ds.sizes["lat"] == n_lat and ds.sizes["lon"] == n_lon
        np.testing.assert_allclose(ds["lat"].values, lat)
        assert float(ds["lon"].values[0]) == 0.0

        # time coord derived from the real (gappy) timestamps
        tvals = ds["time"].values
        assert isinstance(tvals[0], cftime.datetime)
        assert (tvals[0].year, tvals[0].month, tvals[0].day) == (2000, 6, 1)
        assert (tvals[1].month, tvals[1].day) == (6, 2)
        assert (tvals[2].month, tvals[2].day) == (6, 5)  # gap preserved

        # attrs -- diagnostic-only + daily cadence + grid assumption
        assert ds.attrs["diagnostic_variables"] == ["total_precipitation_24hr"]
        assert ds.attrs["surface_variables"] == []
        assert ds.attrs["pressure_upper_air_variables"] == []
        assert ds.attrs["sigma_upper_air_variables"] == []
        assert int(ds.attrs["data_timedelta_hours"]) == 24
        assert ds.attrs["calendar"] == "standard"
        assert "grid_assumption" in ds.attrs
        assert "pressure_level" not in ds.coords and "sigma_level" not in ds.coords

        var = ds["total_precipitation_24hr"]
        assert var.dims == ("time", "lat", "lon")
        assert var.dtype == np.float32
        assert bool(np.isfinite(var.values).all())  # no NaN in IMERG

        # orientation sanity check embedded in attrs, and it should read "looks_ok=True"
        assert "grid_orientation_check" in ds.attrs
        assert "looks_ok=True" in ds.attrs["grid_orientation_check"]
    finally:
        ds.close()

    # Chunking = one day (full field) per chunk.
    zds = xr.open_zarr(out, consolidated=True, decode_times=False)
    enc_chunks = zds["total_precipitation_24hr"].encoding.get("chunks")
    zds.close()
    assert tuple(enc_chunks) == (1, n_lat, n_lon)
    print("test_imerg_global_grid PASSED")


def test_imd_regional_grid_with_nan(tmp_path: Path) -> None:
    """IMD-like regional case: coords.nc grid, RAINFALL rename, NaN preserved, gap."""
    lat = np.arange(6.5, 39.0, 1.0, dtype="float64")  # 33 pts, ascending (native)
    lon = np.arange(66.5, 101.0, 1.0, dtype="float64")  # 35 pts
    n_lat, n_lon = lat.size, lon.size

    coords_nc = tmp_path / "coordinates.nc"
    xr.Dataset(coords={"lat": ("lat", lat), "lon": ("lon", lon)}).to_netcdf(coords_nc)

    inp = tmp_path / "imd_in"
    inp.mkdir()
    # 1901: two days then a gap to a later month.
    tokens = ["1901010100", "1901010200", "1901030100"]
    nan_mask = np.zeros((n_lat, n_lon), dtype=bool)
    nan_mask[:5, :] = True  # a block of NaN (ocean-like)
    for k, tok in enumerate(tokens):
        field = np.full((n_lat, n_lon), float(k) + 1.0, dtype="float32")
        field[nan_mask] = np.nan
        _write_day(inp, tok, "RAINFALL", field)

    out = tmp_path / "imd_1901.zarr"
    _run([
        "--input-dir", str(inp),
        "--source-var", "RAINFALL",
        "--out-var", "total_precipitation_24hr",
        "--dataset-name", "imd",
        "--year", "1901",
        "--units", "mm/day",
        "--coords-nc", str(coords_nc),
        "--out-store", str(out),
    ])

    ds = xr.open_zarr(
        out, consolidated=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    try:
        assert ds.sizes["time"] == 3
        assert ds.sizes["lat"] == n_lat and ds.sizes["lon"] == n_lon
        # native (ascending) lat preserved verbatim
        np.testing.assert_allclose(ds["lat"].values, lat.astype("float32"))
        np.testing.assert_allclose(ds["lon"].values, lon.astype("float32"))

        # RAINFALL renamed to total_precipitation_24hr
        assert "total_precipitation_24hr" in ds.data_vars
        assert "RAINFALL" not in ds.data_vars
        assert ds.attrs["diagnostic_variables"] == ["total_precipitation_24hr"]
        assert int(ds.attrs["data_timedelta_hours"]) == 24

        # time coord + gap
        tvals = ds["time"].values
        assert (tvals[0].month, tvals[0].day) == (1, 1)
        assert (tvals[2].month, tvals[2].day) == (3, 1)  # gap preserved

        # NaN preserved exactly where written
        var = ds["total_precipitation_24hr"].values
        assert np.isnan(var[:, :5, :]).all()
        assert np.isfinite(var[:, 5:, :]).all()

        # IMD has no orientation attr (regional, native grid)
        assert "grid_orientation_check" not in ds.attrs
    finally:
        ds.close()
    print("test_imd_regional_grid_with_nan PASSED")


if __name__ == "__main__":
    import tempfile

    for fn in (test_imerg_global_grid, test_imd_regional_grid_with_nan):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL TESTS PASSED")
