# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ERA5 converters.

Covers:

* The variant table in ``tools/data/era5/build_normalization_zarr.py`` — each
  variant resolves to a known set of source files.
* The shared
  :func:`tools.data._common.normalization.build_normalization_dataset` over
  ERA5-shaped synthetic input (no sigma coord, ``Z`` pressure coord, surface
  scalars merged from two files).
* The shared
  :func:`tools.data._common.climatology_bias.build_climatology_bias_dataset`
  with ERA5 settings (no bias dir, mean + std both supplied → stat axis
  populated).
* The pangu_h5_to_zarr converter on a synthetic ERA5 HDF5 fixture in
  tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

_TOOLS_DIR = Path(__file__).resolve().parents[4] / "tools" / "data"
sys.path.insert(0, str(_TOOLS_DIR))

from _common.climatology_bias import build_climatology_bias_dataset  # noqa: E402
from _common.normalization import build_normalization_dataset  # noqa: E402
from era5.build_normalization_zarr import ERA5_VARIANTS  # noqa: E402
from era5.pangu_h5_to_zarr import (  # noqa: E402
    PANGU_ERA5_CHANNELS,
    _decode_time,
    _level_key,
    convert,
)


# ---------------------------------------------------------------------------
# Variant table is well-formed.
# ---------------------------------------------------------------------------
def test_variant_table_lists_pangu_s2s():
    assert "pangu_s2s" in ERA5_VARIANTS
    # Each tuple has exactly four entries.
    for variant, files in ERA5_VARIANTS.items():
        assert len(files) == 4, f"variant {variant} has {len(files)} files"
        for f in files:
            assert f.endswith(".nc"), f"variant {variant} file {f} not a .nc"


def test_variant_table_has_pangu_s2s_withnino_and_log_precip():
    for variant in ("pangu_s2s_withnino", "pangu_s2s_log_precip"):
        assert variant in ERA5_VARIANTS


# ---------------------------------------------------------------------------
# Normalization assembly — ERA5-shaped synthetic input.
# ---------------------------------------------------------------------------
def _era5_synthetic_mean_std(pressure_levels=(50, 100, 500, 1000)):
    """Synthetic ERA5-style mean/std pair: 5 upper-air vars × N pressure levels +
    scalar surface vars. Mimics ``pangu_s2s_*_mean.nc`` (upper-air) +
    ``..._surface_mean.nc`` (surface) AFTER xr.merge."""
    coords = {"Z": ("Z", np.asarray(pressure_levels, dtype="float64"))}
    upper_vars = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
    surface_vars = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure"]
    mean_data: dict = {}
    std_data: dict = {}
    rng = np.random.default_rng(0)
    for v in upper_vars:
        mean_data[v] = xr.DataArray(rng.standard_normal(len(pressure_levels), dtype="float32"), dims=("Z",))
        std_data[v] = xr.DataArray(rng.standard_normal(len(pressure_levels), dtype="float32").__abs__() + 1, dims=("Z",))
    for v in surface_vars:
        mean_data[v] = xr.DataArray(np.float32(rng.standard_normal()))
        std_data[v] = xr.DataArray(np.float32(abs(rng.standard_normal()) + 1))
    return xr.Dataset(mean_data, coords=coords), xr.Dataset(std_data, coords=coords)


def test_era5_normalization_layout():
    mean_ds, std_ds = _era5_synthetic_mean_std()
    out = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name=None, pressure_coord_name="Z"
    )
    assert "sigma_level" not in out.coords  # ERA5 has no sigma
    assert out.sizes["pressure_level"] == 4
    assert out["temperature"].dims == ("stat", "pressure_level")
    assert out["2m_temperature"].dims == ("stat",)
    np.testing.assert_allclose(
        out["temperature"].sel(stat="mean").values, mean_ds["temperature"].values
    )


# ---------------------------------------------------------------------------
# Climatology+bias assembly with both mean+std supplied (ERA5 pattern).
# ---------------------------------------------------------------------------
def _era5_synthetic_climatology(n_dayofyear=5, n_pressure=4, n_lat=8, n_lon=16):
    pressure = np.linspace(50, 1000, n_pressure, dtype="float32")
    coords = {
        "time": ("time", np.arange(n_dayofyear, dtype="int32")),
        "plev": ("plev", pressure),
        "lat": ("lat", np.linspace(89.5, -89.5, n_lat, dtype="float32")),
        "lon": ("lon", np.linspace(0, 360 * (n_lon - 1) / n_lon, n_lon, dtype="float32")),
    }
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "temperature": (
                ("time", "plev", "lat", "lon"),
                rng.standard_normal((n_dayofyear, n_pressure, n_lat, n_lon), dtype="float32") + 250,
            ),
            "2m_temperature": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_dayofyear, n_lat, n_lon), dtype="float32") + 280,
            ),
        },
        coords=coords,
    )


def test_era5_climatology_mean_plus_std_stat_axis():
    mean_clim = _era5_synthetic_climatology()
    rng = np.random.default_rng(1)
    std_clim = mean_clim.copy()
    std_clim["temperature"].values[:] = np.abs(
        rng.standard_normal(mean_clim["temperature"].shape, dtype="float32")
    ) + 1
    std_clim["2m_temperature"].values[:] = np.abs(
        rng.standard_normal(mean_clim["2m_temperature"].shape, dtype="float32")
    ) + 1
    out = build_climatology_bias_dataset(
        mean_clim,
        {},
        std_climatology_ds=std_clim,
        sigma_dim="lev",
        pressure_dim="plev",
        n_workers=1,
    )
    # ERA5 has no bias dir → bias vars absent.
    assert "temperature_bias_annual_sigma" not in out.data_vars
    assert "temperature_bias_annual_pressure" not in out.data_vars
    # Stat axis populated for both mean + std slots.
    assert out["temperature"].dims == ("stat", "dayofyear", "pressure_level", "lat", "lon")
    np.testing.assert_allclose(
        out["temperature"].sel(stat="mean").values, mean_clim["temperature"].values
    )
    np.testing.assert_allclose(
        out["temperature"].sel(stat="std").values, std_clim["temperature"].values
    )


# ---------------------------------------------------------------------------
# pangu_h5_to_zarr: filename parsing + level-key resolution.
# ---------------------------------------------------------------------------
def test_decode_time_parses_iso_timestamp():
    t = _decode_time(b"1979-01-01T00:00:00.000000000")
    assert t.year == 1979 and t.month == 1 and t.day == 1 and t.hour == 0
    t = _decode_time("2018-12-31T18:00:00.000000000")
    assert t.year == 2018 and t.month == 12 and t.day == 31 and t.hour == 18


def test_level_key_matches_with_tolerance():
    """``_level_key`` resolves `prefix_500.0` from a dict of H5 keys."""
    class FakeGroup:
        def __init__(self, keys):
            self._keys = list(keys)
        def keys(self):
            return self._keys

    g = FakeGroup(
        ["temperature_50.0", "temperature_500.0", "temperature_1000.0", "2m_temperature"]
    )
    assert _level_key(g, "temperature", 500.0) == "temperature_500.0"
    assert _level_key(g, "temperature", 50.0) == "temperature_50.0"


def test_pangu_era5_channels_defaults_have_full_source_levels():
    levels = PANGU_ERA5_CHANNELS["pressure_levels"]
    # Full ERA5 source coverage: 18 pressure levels in hPa.
    assert len(levels) == 18
    # Spot-check both ends of the axis (the stratospheric levels were missing
    # in the old 13-level default).
    assert 5.0 in levels and 10.0 in levels and 1000.0 in levels
    # No sigma vars for ERA5.
    assert PANGU_ERA5_CHANNELS["sigma_upper_air_variables"] == []


# ---------------------------------------------------------------------------
# End-to-end synthetic round-trip of pangu_h5_to_zarr.
# ---------------------------------------------------------------------------
def _write_synth_era5_h5(path: Path, *, time_iso: str, n_lat: int = 8, n_lon: int = 16,
                          levels=(50.0, 500.0, 1000.0)):
    """Write one synthetic ERA5 .h5 sample at ``path``."""
    rng = np.random.default_rng(int(time_iso.replace("-", "").replace("T", "")[:8]))
    with h5py.File(path, "w") as f:
        g = f.create_group("input")
        # Surface vars (singletons).
        for v in ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
                  "mean_sea_level_pressure"):
            g.create_dataset(v, data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        # Constant boundary.
        for v in ("land_sea_mask", "geopotential_at_surface"):
            g.create_dataset(v, data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        # Varying boundary.
        for v in ("sea_surface_temperature", "sea_ice_cover", "toa_incident_solar_radiation"):
            g.create_dataset(v, data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        # Diagnostic.
        g.create_dataset(
            "total_precipitation_24hr",
            data=rng.standard_normal((n_lat, n_lon)).astype("float32"),
        )
        # Upper-air pressure-level vars.
        for v in ("temperature", "u_component_of_wind", "v_component_of_wind",
                  "specific_humidity", "geopotential"):
            for lev in levels:
                g.create_dataset(
                    f"{v}_{lev}",
                    data=rng.standard_normal((n_lat, n_lon)).astype("float32"),
                )
        # Time scalar.
        g.create_dataset("time", data=np.bytes_(time_iso))


def test_pangu_h5_to_zarr_round_trip_one_year(tmp_path):
    """End-to-end: 3 synthetic ERA5 .h5 files → one Zarr → read back the layout."""
    input_dir = tmp_path / "era5_h5"
    input_dir.mkdir()
    for i in range(3):
        _write_synth_era5_h5(
            input_dir / f"1979_{i:04d}.h5",
            time_iso=f"1979-01-01T{i*6:02d}:00:00.000000000",
        )

    # Use a smaller channel config so the synthetic fixture covers it.
    config = {
        "surface_variables": ["2m_temperature", "10m_u_component_of_wind"],
        "constant_boundary_variables": ["land_sea_mask"],
        "varying_boundary_variables": ["sea_surface_temperature"],
        "diagnostic_variables": ["total_precipitation_24hr"],
        "pressure_upper_air_variables": ["temperature"],
        "sigma_upper_air_variables": [],
        "pressure_levels": [50.0, 500.0, 1000.0],
    }
    config_path = tmp_path / "channels.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh)

    out_path = tmp_path / "1979.zarr"

    class _Args:
        pass
    args = _Args()
    args.input_dir = input_dir
    args.year = 1979
    args.sample_range = None
    args.output = out_path
    args.channel_config = config_path
    args.data_timedelta_hours = 6
    args.n_workers = 1
    args.overwrite = False
    args.verbose = False

    convert(args)

    ds = xr.open_zarr(
        out_path,
        consolidated=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    # Coords + shapes.
    assert ds.sizes["time"] == 3
    assert ds.sizes["pressure_level"] == 3
    assert ds.sizes["lat"] == 8 and ds.sizes["lon"] == 16
    # Channel-group attrs.
    assert list(ds.attrs["surface_variables"]) == ["2m_temperature", "10m_u_component_of_wind"]
    assert list(ds.attrs["pressure_upper_air_variables"]) == ["temperature"]
    assert list(ds.attrs["sigma_upper_air_variables"]) == []
    assert ds.attrs["calendar"] == "standard"
    # Per-variable shapes.
    assert ds["2m_temperature"].dims == ("time", "lat", "lon")
    assert ds["land_sea_mask"].dims == ("lat", "lon")
    assert ds["temperature"].dims == ("time", "pressure_level", "lat", "lon")
