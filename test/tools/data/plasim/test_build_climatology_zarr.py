# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PLASIM climatology+bias Zarr converter.

The converter takes a CDO-style climatology NetCDF (``time, lev, plev, lat, lon``)
plus a flat directory of ``{var}[_{level}]_bias[_{H}z].npy`` files and emits one
Zarr store with the unified ai-rossby schema:

* ``{var}`` — daily climatology, dims ``(dayofyear, [level,] lat, lon)``.
* ``{var}_bias_annual[_sigma|_pressure]`` — annual bias.
* ``{var}_bias_diurnal[_sigma|_pressure]`` — diurnal cycle.

These tests build small synthetic fixtures in ``tmp_path`` and round-trip the
converter end-to-end; one further test exercises the real PLASIM fixture if
the smoke fixture has been staged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_TOOLS_DIR = Path(__file__).resolve().parents[4] / "tools" / "data"
sys.path.insert(0, str(_TOOLS_DIR))

from _common.bias import (  # noqa: E402
    parse_bias_filename,
    scan_bias_dir,
)
from _common.climatology_bias import (  # noqa: E402
    CLIMATOLOGY_BIAS_SCHEMA_VERSION,
    DIURNAL_HOURS,
    _tolerance_union,
    build_climatology_bias_dataset,
    write_climatology_bias_zarr,
)


def _synth_climatology(n_dayofyear=5, n_sigma=3, n_pressure=4, n_lat=8, n_lon=16):
    """Tiny synthetic climatology with both level systems and a couple of vars."""
    sigma = np.linspace(0.1, 0.9, n_sigma, dtype="float32")
    pressure = np.linspace(5000.0, 100000.0, n_pressure, dtype="float32")
    lat = np.linspace(-87.5, 87.5, n_lat, dtype="float32")
    lon = np.linspace(0.0, 360.0 * (n_lon - 1) / n_lon, n_lon, dtype="float32")
    coords = {
        "time": ("time", np.arange(n_dayofyear, dtype="int32")),
        "lev": ("lev", sigma),
        "plev": ("plev", pressure),
        "lat": ("lat", lat),
        "lon": ("lon", lon),
    }
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "ta": (
                ("time", "lev", "lat", "lon"),
                rng.standard_normal((n_dayofyear, n_sigma, n_lat, n_lon), dtype="float32") + 250,
            ),
            "zg": (
                ("time", "plev", "lat", "lon"),
                rng.standard_normal((n_dayofyear, n_pressure, n_lat, n_lon), dtype="float32") * 100,
            ),
            "tas": (("time", "lat", "lon"), rng.standard_normal((n_dayofyear, n_lat, n_lon), dtype="float32") + 280),
        },
        coords=coords,
    ), sigma, pressure, n_lat, n_lon


def _write_synth_bias_dir(
    bias_dir: Path,
    *,
    surface_vars: list[str],
    sigma_vars: list[str],
    pressure_vars: list[str],
    sigma_levels: np.ndarray,
    pressure_levels: np.ndarray,
    n_lat: int,
    n_lon: int,
) -> None:
    """Write tiny synthetic bias .npy files for the requested var/level combos."""
    rng = np.random.default_rng(1)
    bias_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str) -> None:
        arr = rng.standard_normal((n_lat, n_lon), dtype="float32")
        np.save(bias_dir / name, arr)

    for v in surface_vars:
        _write(f"{v}_bias.npy")
        for h in DIURNAL_HOURS:
            _write(f"{v}_bias_{h}z.npy")
    for v in sigma_vars:
        for lev in sigma_levels:
            _write(f"{v}_{float(lev)}_bias.npy")
            for h in DIURNAL_HOURS:
                _write(f"{v}_{float(lev)}_bias_{h}z.npy")
    for v in pressure_vars:
        for lev in pressure_levels:
            _write(f"{v}_{float(lev)}_bias.npy")
            for h in DIURNAL_HOURS:
                _write(f"{v}_{float(lev)}_bias_{h}z.npy")


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
def test_parse_bias_filename_surface_annual():
    spec = parse_bias_filename(Path("evap_bias.npy"))
    assert spec.var == "evap"
    assert spec.level is None
    assert spec.hour is None


def test_parse_bias_filename_surface_diurnal():
    spec = parse_bias_filename(Path("evap_bias_12z.npy"))
    assert spec.var == "evap"
    assert spec.level is None
    assert spec.hour == 12


def test_parse_bias_filename_pressure_level():
    spec = parse_bias_filename(Path("zg_5000.0_bias.npy"))
    assert spec.var == "zg"
    assert spec.level == 5000.0
    assert spec.hour is None


def test_parse_bias_filename_sigma_level_full_precision():
    spec = parse_bias_filename(Path("hus_0.03830000013113022_bias_6z.npy"))
    assert spec.var == "hus"
    assert spec.level == pytest.approx(0.03830000013113022)
    assert spec.hour == 6


def test_parse_bias_filename_multitoken_varname():
    """The var name can contain underscores — `pr_24h` is a single var."""
    spec = parse_bias_filename(Path("pr_24h_bias_18z.npy"))
    assert spec.var == "pr_24h"
    assert spec.level is None
    assert spec.hour == 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_tolerance_union_collapses_near_duplicates():
    out = _tolerance_union([0.21085, 0.21085001, 0.21085000783205032], tol=1e-4)
    assert len(out) == 1


def test_tolerance_union_keeps_distinct_values():
    out = _tolerance_union([0.1, 0.2, 0.3], tol=1e-4)
    assert out == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# End-to-end synthetic round-trip
# ---------------------------------------------------------------------------
def test_synthetic_round_trip_climatology_only(tmp_path):
    """No bias dir — climatology-only store, all bias vars absent."""
    clim, sigma, pressure, n_lat, n_lon = _synth_climatology()
    out = build_climatology_bias_dataset(
        clim, {}, sigma_dim="lev", pressure_dim="plev", n_workers=1
    )
    # Only the climatology vars survive.
    assert "ta" in out.data_vars
    assert "ta_bias_annual_sigma" not in out.data_vars
    assert out["ta"].dims == ("dayofyear", "sigma_level", "lat", "lon")
    assert out.coords["dayofyear"].values.tolist() == [1, 2, 3, 4, 5]


def test_synthetic_round_trip_with_bias(tmp_path):
    clim, sigma, pressure, n_lat, n_lon = _synth_climatology()
    bias_dir = tmp_path / "bias"
    _write_synth_bias_dir(
        bias_dir,
        surface_vars=["tas"],
        sigma_vars=["ta"],
        pressure_vars=["zg", "ta"],  # `ta` exists on BOTH systems in PLASIM
        sigma_levels=sigma,
        pressure_levels=pressure,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    groups = scan_bias_dir(bias_dir)
    out = build_climatology_bias_dataset(
        clim, groups, sigma_dim="lev", pressure_dim="plev", n_workers=2
    )
    # `ta` got both sigma + pressure bias arrays.
    assert out["ta"].dims == ("dayofyear", "sigma_level", "lat", "lon")
    assert out["ta_bias_annual_sigma"].dims == ("sigma_level", "lat", "lon")
    assert out["ta_bias_annual_pressure"].dims == ("pressure_level", "lat", "lon")
    assert out["ta_bias_diurnal_sigma"].dims == ("hour_of_day", "sigma_level", "lat", "lon")
    assert out["ta_bias_diurnal_pressure"].dims == (
        "hour_of_day",
        "pressure_level",
        "lat",
        "lon",
    )
    assert out["tas_bias_annual"].dims == ("lat", "lon")
    assert out["tas_bias_diurnal"].dims == ("hour_of_day", "lat", "lon")
    assert out["zg"].dims == ("dayofyear", "pressure_level", "lat", "lon")


def test_zarr_write_round_trip(tmp_path):
    clim, sigma, pressure, n_lat, n_lon = _synth_climatology()
    bias_dir = tmp_path / "bias"
    _write_synth_bias_dir(
        bias_dir,
        surface_vars=["tas"],
        sigma_vars=["ta"],
        pressure_vars=["zg"],
        sigma_levels=sigma,
        pressure_levels=pressure,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    groups = scan_bias_dir(bias_dir)
    out = build_climatology_bias_dataset(
        clim, groups, sigma_dim="lev", pressure_dim="plev", n_workers=1
    )
    out_path = tmp_path / "out.zarr"
    write_climatology_bias_zarr(
        out,
        out_path,
        source_climatology=Path("synth_clim.nc"),
        source_bias_dir=bias_dir,
    )
    ds = xr.open_zarr(out_path)
    assert ds.attrs["climatology_bias_schema_version"] == CLIMATOLOGY_BIAS_SCHEMA_VERSION
    assert ds.attrs["source_climatology"] == "synth_clim.nc"
    assert ds.attrs["source_bias_dir"] == str(bias_dir)
    assert "ta_bias_annual_sigma" in ds.data_vars
    # dayofyear coord is the integer 1..N convention.
    assert ds["dayofyear"].values.tolist() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Real PLASIM fixture — exercises the actual sim52 climatology + bias dir.
# ---------------------------------------------------------------------------
_PLASIM_CLIM = Path(
    "/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/sigma_data/climatology.nc"
)
_PLASIM_BIAS = Path("/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/bias")
_HAS_REAL = _PLASIM_CLIM.exists() and _PLASIM_BIAS.is_dir()


@pytest.mark.skipif(not _HAS_REAL, reason="real PLASIM climatology+bias fixture missing")
def test_real_plasim_fixture_parses(tmp_path):
    """Quick smoke that the real bias dir parses without errors.

    We don't run the full converter here — that's a Delta CPU job, not a
    login-node test. Just verify the filename parser and dir scan don't blow
    up on the 635 real files.
    """
    groups = scan_bias_dir(_PLASIM_BIAS)
    # Expect at least the canonical surface vars + the upper-air vars.
    for canonical in ("evap", "lsm", "ta", "ua", "va", "hus", "zg", "tas", "pr_6h"):
        assert canonical in groups, f"missing {canonical} in scanned bias dir"
    # `ta` has both sigma and pressure files.
    assert any(lv < 2.0 for lv in groups["ta"].levels)
    assert any(lv >= 2.0 for lv in groups["ta"].levels)
