# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the E3SM converters.

Covers:

* Soil-level decomposition: ``H2OSOI(time, levgrnd, lat, lon)`` →
  per-depth 2D channels (``H2OSOI_0.007101``, etc.) per user answer #7.
* Per-year H5→Zarr converter on a synthetic E3SM HDF5 fixture (uppercase var
  names, hPa pressure suffixes, ``noleap`` calendar).
* Normalization helper with E3SM's ``Z_2`` source coord → unified
  ``pressure_level``.
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

from _common.normalization import build_normalization_dataset  # noqa: E402
from e3sm.build_climatology_zarr import _decompose_soil_vars  # noqa: E402
from e3sm.pangu_h5_to_zarr import (  # noqa: E402
    PANGU_E3SM_CHANNELS,
    _decode_time,
    _level_key,
    convert,
)


# ---------------------------------------------------------------------------
# Soil-level decomposition.
# ---------------------------------------------------------------------------
def test_decompose_soil_vars_splits_levgrnd():
    """H2OSOI / TSOI lose their levgrnd dim and become per-depth flat channels."""
    depths = np.asarray([0.007101, 0.027925, 0.062259], dtype="float32")
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "H2OSOI": (
                ("time", "levgrnd", "lat", "lon"),
                rng.standard_normal((3, 3, 4, 8), dtype="float32"),
            ),
            "TSOI": (
                ("time", "levgrnd", "lat", "lon"),
                rng.standard_normal((3, 3, 4, 8), dtype="float32"),
            ),
            "T": (("time", "lat", "lon"), rng.standard_normal((3, 4, 8), dtype="float32")),
        },
        coords={
            "time": ("time", np.arange(3, dtype="int32")),
            "levgrnd": ("levgrnd", depths),
            "lat": ("lat", np.linspace(-89.5, 89.5, 4, dtype="float32")),
            "lon": ("lon", np.linspace(0.5, 359.5, 8, dtype="float32")),
        },
    )
    out = _decompose_soil_vars(ds, soil_dim="levgrnd")
    # Soil vars are gone, replaced by per-depth flat channels.
    assert "H2OSOI" not in out.data_vars
    assert "TSOI" not in out.data_vars
    assert "T" in out.data_vars  # non-soil vars untouched
    # Per-depth channels exist with the depth value baked in.
    for d in depths:
        assert f"H2OSOI_{float(d)}" in out.data_vars
        assert f"TSOI_{float(d)}" in out.data_vars
        # Shape: (time, lat, lon), no levgrnd.
        assert out[f"H2OSOI_{float(d)}"].dims == ("time", "lat", "lon")
    assert "levgrnd" not in out.dims
    assert "levgrnd" not in out.coords


def test_decompose_soil_vars_noop_when_no_levgrnd():
    ds = xr.Dataset({"T": (("time", "lat", "lon"), np.zeros((2, 4, 8), dtype="float32"))})
    out = _decompose_soil_vars(ds, soil_dim="levgrnd")
    assert "T" in out.data_vars
    assert "levgrnd" not in out.dims


# ---------------------------------------------------------------------------
# Normalization with E3SM's Z_2 source coord.
# ---------------------------------------------------------------------------
def test_e3sm_normalization_uses_pressure_axis():
    pressure = [50.0, 100.0, 500.0, 1000.0]
    mean_ds = xr.Dataset(
        {
            "T": (("Z_2",), np.arange(4, dtype="float32")),
            "U": (("Z_2",), np.arange(4, dtype="float32")),
            "SST": ((), np.float32(290.0)),
        },
        coords={"Z_2": ("Z_2", np.asarray(pressure, dtype="float64"))},
    )
    std_ds = xr.Dataset(
        {
            "T": (("Z_2",), np.ones(4, dtype="float32")),
            "U": (("Z_2",), np.ones(4, dtype="float32") * 2),
            "SST": ((), np.float32(2.0)),
        },
        coords={"Z_2": ("Z_2", np.asarray(pressure, dtype="float64"))},
    )
    out = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name=None, pressure_coord_name="Z_2"
    )
    assert out.sizes["pressure_level"] == 4
    np.testing.assert_allclose(
        out.coords["pressure_level"].values, np.asarray(pressure, dtype="float32")
    )
    # Per-var shapes.
    assert out["T"].dims == ("stat", "pressure_level")
    assert out["SST"].dims == ("stat",)


# ---------------------------------------------------------------------------
# pangu_h5_to_zarr helpers.
# ---------------------------------------------------------------------------
def test_decode_time_synthesizes_when_h5_has_no_iso():
    """E3SM noleap calendar: idx=0 → Jan 1 00:00 of the given year."""
    t = _decode_time(b"", year=2015, idx=0, data_timedelta_hours=6)
    assert t.year == 2015 and t.month == 1 and t.day == 1 and t.hour == 0
    # idx=4 with 6-hour cadence → 24 hours after Jan 1 → Jan 2 00:00.
    t = _decode_time(b"", year=2015, idx=4, data_timedelta_hours=6)
    assert t.year == 2015 and t.month == 1 and t.day == 2 and t.hour == 0


def test_decode_time_parses_iso_when_present():
    t = _decode_time(
        b"2015-06-15T12:00:00.000000000",
        year=2015,
        idx=0,
        data_timedelta_hours=6,
    )
    assert t.year == 2015 and t.month == 6 and t.day == 15 and t.hour == 12


def test_level_key_tolerates_full_float64_precision():
    class FakeGroup:
        def __init__(self, keys):
            self._keys = list(keys)
        def keys(self):
            return self._keys

    g = FakeGroup(
        ["T_998.4964394917621", "T_925.5197481473349", "U_998.4964394917621"]
    )
    # Looking for 1000.0 should match 998.5 within tolerance.
    assert _level_key(g, "T", 1000.0) == "T_998.4964394917621"
    assert _level_key(g, "U", 1000.0) == "U_998.4964394917621"


def test_pangu_e3sm_channels_has_uppercase_vars():
    assert "T" in PANGU_E3SM_CHANNELS["pressure_upper_air_variables"]
    assert "TREFHT" in PANGU_E3SM_CHANNELS["surface_variables"]
    # No sigma — pressure only.
    assert PANGU_E3SM_CHANNELS["sigma_upper_air_variables"] == []


def test_pangu_e3sm_channels_full_source_levels():
    levels = PANGU_E3SM_CHANNELS["pressure_levels"]
    # Full E3SM source coverage: 18 hybrid-pressure levels in hPa.
    assert len(levels) == 18
    # The top stratospheric levels (4.7, 10.7 hPa) were missing in the old
    # 13-level default — make sure they're present now.
    assert min(levels) < 5.0
    assert max(levels) > 990.0


# ---------------------------------------------------------------------------
# End-to-end synthetic round-trip.
# ---------------------------------------------------------------------------
def _write_synth_e3sm_h5(
    path: Path, *, time_iso: str, n_lat: int = 8, n_lon: int = 16,
    pressure_levels=(50.0, 500.0, 1000.0),
):
    rng = np.random.default_rng(int(time_iso.replace("-", "").replace("T", "")[:8]))
    with h5py.File(path, "w") as f:
        g = f.create_group("input")
        for v in ("TREFHT", "U10", "PSL"):
            g.create_dataset(v, data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        for v in ("TOPO", "PCT_GLACIER"):
            g.create_dataset(v, data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        for v in ("SST", "ICE", "sol_in"):
            g.create_dataset(v, data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        g.create_dataset("PRECT", data=rng.standard_normal((n_lat, n_lon)).astype("float32"))
        for v in ("T", "U", "V", "RELHUM", "Z3"):
            for lev in pressure_levels:
                # Encode level with full E3SM-style float64 precision tail.
                g.create_dataset(
                    f"{v}_{lev}",
                    data=rng.standard_normal((n_lat, n_lon)).astype("float32"),
                )
        g.create_dataset("time", data=np.bytes_(time_iso))


def test_pangu_h5_to_zarr_round_trip_one_year(tmp_path):
    input_dir = tmp_path / "e3sm_h5"
    input_dir.mkdir()
    for i in range(3):
        _write_synth_e3sm_h5(
            input_dir / f"2015_{i:04d}.h5",
            time_iso=f"2015-01-01T{i*6:02d}:00:00.000000000",
        )

    config = {
        "surface_variables": ["TREFHT"],
        "constant_boundary_variables": ["TOPO"],
        "varying_boundary_variables": ["SST"],
        "diagnostic_variables": ["PRECT"],
        "pressure_upper_air_variables": ["T"],
        "sigma_upper_air_variables": [],
        "pressure_levels": [50.0, 500.0, 1000.0],
    }
    config_path = tmp_path / "channels.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh)

    out_path = tmp_path / "2015.zarr"

    class _Args:
        pass
    args = _Args()
    args.input_dir = input_dir
    args.year = 2015
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
    assert ds.sizes["time"] == 3
    assert ds.sizes["pressure_level"] == 3
    assert ds.attrs["calendar"] == "noleap"
    assert list(ds.attrs["surface_variables"]) == ["TREFHT"]
    assert list(ds.attrs["pressure_upper_air_variables"]) == ["T"]
    assert ds["T"].dims == ("time", "pressure_level", "lat", "lon")
    assert ds["TOPO"].dims == ("lat", "lon")
