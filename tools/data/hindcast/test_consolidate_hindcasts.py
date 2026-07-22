# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the hindcast consolidator (plain xarray/zarr/numpy).

Synthesizes tiny fake Pangu NetCDF per-init files and a minimal ai_rossby
per-IC store (matching ``_build_per_ic_dataset`` in
``examples/weather/ai_rossby/inference.py``), runs each adapter through the
consolidator, and asserts on the resulting zarr v3 store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cftime
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import consolidate_hindcasts as ch  # noqa: E402

# Tiny grid for fast tests.
NLAT, NLON, NLEV, NTIME = 4, 8, 3, 16
LEVELS = np.array([300.0, 500.0, 850.0], dtype="float32")
SURF_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
             "mean_sea_level_pressure", "surface_pressure"]
DIAG_VARS = ["total_precipitation_24hr", "mean_top_net_long_wave_radiation_flux"]
UPPER_VARS = ["temperature", "u_component_of_wind", "v_component_of_wind",
              "specific_humidity", "geopotential"]

# Short NetCDF names to exercise alias detection in the Pangu adapter.
PANGU_SURF_NAMES = {"2m_temperature": "t2m", "10m_u_component_of_wind": "u10",
                    "10m_v_component_of_wind": "v10", "mean_sea_level_pressure": "msl",
                    "surface_pressure": "sp"}
PANGU_DIAG_NAMES = {"total_precipitation_24hr": "tp",
                    "mean_top_net_long_wave_radiation_flux": "mtnlwrf"}
PANGU_UPPER_NAMES = {"temperature": "T", "u_component_of_wind": "U",
                     "v_component_of_wind": "V", "specific_humidity": "Q",
                     "geopotential": "Z"}


def _field_value(base: float, canon: str, t: int) -> float:
    """A deterministic per-(var, frame) value so we can check frame mapping."""
    return base + 100.0 * (hash(canon) % 7) + t


def _make_pangu_nc(path: Path, init_dt: cftime.DatetimeGregorian) -> None:
    lat = np.linspace(89.5, -89.5, NLAT, dtype="float32")
    lon = np.linspace(0.0, 360.0 * (NLON - 1) / NLON, NLON, dtype="float32")
    data_vars = {}
    for canon, short in {**PANGU_SURF_NAMES, **PANGU_DIAG_NAMES}.items():
        arr = np.stack(
            [np.full((NLAT, NLON), _field_value(0.0, canon, t), dtype="float32")
             for t in range(NTIME)], axis=0)
        data_vars[short] = (("time", "lat", "lon"), arr)
    for canon, short in PANGU_UPPER_NAMES.items():
        arr = np.stack(
            [np.stack([np.full((NLAT, NLON), _field_value(lev, canon, t), dtype="float32")
                       for lev in range(NLEV)], axis=0)
             for t in range(NTIME)], axis=0)
        data_vars[short] = (("time", "plev", "lat", "lon"), arr)
    ds = xr.Dataset(
        data_vars,
        coords={"time": np.arange(NTIME, dtype="float64"),
                "plev": LEVELS, "lat": lat, "lon": lon},
    )
    # scipy engine -> NetCDF3 (the only backend available in the test env).
    ds.to_netcdf(path, engine="scipy")


def _make_ai_rossby_store(path: Path, ic_dt: cftime.DatetimeGregorian) -> None:
    """Mirror _build_per_ic_dataset "members" mode with ensemble=1."""
    lat = np.linspace(89.5, -89.5, NLAT, dtype="float32")
    lon = np.linspace(0.0, 360.0 * (NLON - 1) / NLON, NLON, dtype="float32")
    ens = 1
    surf = np.zeros((ens, NTIME, len(SURF_VARS), NLAT, NLON), dtype="float32")
    for vi, canon in enumerate(SURF_VARS):
        for t in range(NTIME):
            surf[0, t, vi] = _field_value(0.0, canon, t)
    up = np.zeros((ens, NTIME, len(UPPER_VARS), NLEV, NLAT, NLON), dtype="float32")
    for vi, canon in enumerate(UPPER_VARS):
        for t in range(NTIME):
            for lev in range(NLEV):
                up[0, t, vi, lev] = _field_value(lev, canon, t)
    diag = np.zeros((ens, NTIME, len(DIAG_VARS), NLAT, NLON), dtype="float32")
    for vi, canon in enumerate(DIAG_VARS):
        for t in range(NTIME):
            diag[0, t, vi] = _field_value(0.0, canon, t)

    ds = xr.Dataset(
        {
            "pred_surface": (("ensemble", "frame", "surface_var", "lat", "lon"), surf),
            "pred_upper_air": (("ensemble", "frame", "upper_air_var", "level", "lat", "lon"), up),
            "pred_diagnostic": (("ensemble", "frame", "diag_var", "lat", "lon"), diag),
        },
        coords={
            "ensemble": np.arange(ens, dtype="int64"),
            "frame": np.arange(NTIME, dtype="int64"),
            "surface_var": np.asarray(SURF_VARS),
            "upper_air_var": np.asarray(UPPER_VARS),
            "diag_var": np.asarray(DIAG_VARS),
            "level": LEVELS,
            "lat": lat, "lon": lon,
        },
    )
    ds.attrs["frame_zero_is_ic"] = 1
    ds.attrs["ensemble_size"] = 1
    ds.attrs["max_step"] = NTIME - 1
    ds.attrs["ic_time"] = str(ic_dt)
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True)


def _args(fmt, input_dir, out_store, ref_store=None, prepend_ic=False):
    return SimpleNamespace(
        format=fmt, input_dir=input_dir, out_store=out_store, model="pangu_s2s",
        year=2000, ref_store=ref_store, checkpoint="ckpt-xyz",
        source_dataset="pangu_s2s", boundary_clamped_inits=[], created="2026-07-21T00:00Z",
        commit="deadbeef", generator="", prepend_ic=prepend_ic, overwrite=True, verbose=False,
    )


# --------------------------------------------------------------------------- #
# Pangu adapter
# --------------------------------------------------------------------------- #
def test_pangu_consolidation(tmp_path):
    in_dir = tmp_path / "predictions"
    in_dir.mkdir()
    # Two valid inits (2000-01-01, 2000-01-05) + a Feb-29 that must be dropped.
    kept = [cftime.DatetimeGregorian(2000, 1, 5, 0), cftime.DatetimeGregorian(2000, 1, 1, 0)]
    feb29 = cftime.DatetimeGregorian(2000, 2, 29, 0)
    for dt in kept + [feb29]:
        stamp = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}"
        _make_pangu_nc(in_dir / f"pangu_plasim_2000_24h_15step_{stamp}.nc", dt)

    out = tmp_path / "2000.zarr"
    summary = ch.run(_args("pangu", in_dir, out))

    assert summary["n_dropped_feb29"] == 1
    assert summary["n_init"] == 2

    r = xr.open_zarr(out, consolidated=True, decode_timedelta=False)
    try:
        assert r.sizes["init_time"] == 2
        assert r.sizes["lead_time"] == NTIME
        assert r.sizes["pressure_level"] == NLEV
        assert r.sizes["lat"] == NLAT and r.sizes["lon"] == NLON

        # Feb-29 dropped, inits sorted chronologically.
        it = r["init_time"].values
        assert str(it[0])[:10] == "2000-01-01"
        assert str(it[1])[:10] == "2000-01-05"
        assert not any(str(t)[5:10] == "02-29" for t in it)

        # Variable groups + canonical names via alias detection.
        assert set(r.attrs["surface_variables"]) == set(SURF_VARS)
        assert set(r.attrs["diagnostic_variables"]) == set(DIAG_VARS)
        assert set(r.attrs["upper_air_variables"]) == set(UPPER_VARS)

        # lead_time coord + units attr.
        assert list(r["lead_time"].values) == list(range(NTIME))
        assert r["lead_time"].attrs["units"] == "days"

        # lead 0 matches the fake IC (frame 0) for init 2000-01-01.
        exp0 = _field_value(0.0, "2m_temperature", 0)
        np.testing.assert_allclose(
            r["2m_temperature"].isel(init_time=0, lead_time=0).values, exp0)
        # lead 3 == frame 3.
        exp3 = _field_value(0.0, "2m_temperature", 3)
        np.testing.assert_allclose(
            r["2m_temperature"].isel(init_time=0, lead_time=3).values, exp3)
        # upper-air level mapping (level index 1 -> 500 hPa).
        expT = _field_value(1.0, "temperature", 2)
        np.testing.assert_allclose(
            r["temperature"].isel(init_time=0, lead_time=2, pressure_level=1).values, expT)

        # attrs present + everything finite.
        for k in ["hindcast_schema_version", "model", "checkpoint", "source_dataset",
                  "init_schedule", "lead_time_hours", "n_lead", "calendar",
                  "boundary_clamped_inits", "created", "generator"]:
            assert k in r.attrs, f"missing attr {k}"
        assert r.attrs["hindcast_schema_version"] == "1.0"
        assert r.attrs["n_lead"] == NTIME
        assert r.attrs["generator"].endswith("@deadbeef")
        for v in SURF_VARS + DIAG_VARS + UPPER_VARS:
            assert bool(np.isfinite(r[v].values).all()), f"{v} has non-finite values"
    finally:
        r.close()


def test_pangu_filename_datetime_parse():
    dt = ch._parse_pangu_init_dt("pangu_plasim_2000_24h_15step_2007121300.nc")
    assert (dt.year, dt.month, dt.day, dt.hour) == (2007, 12, 13, 0)


# --------------------------------------------------------------------------- #
# ai_rossby adapter
# --------------------------------------------------------------------------- #
def test_ai_rossby_consolidation(tmp_path):
    in_dir = tmp_path / "per_ic"
    in_dir.mkdir()
    kept = [cftime.DatetimeGregorian(2000, 1, 9, 0), cftime.DatetimeGregorian(2000, 1, 1, 0)]
    feb29 = cftime.DatetimeGregorian(2000, 2, 29, 0)
    for dt in kept + [feb29]:
        stamp = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}00"
        _make_ai_rossby_store(in_dir / f"sfno__run0__{stamp}_{stamp}.zarr", dt)

    out = tmp_path / "2000_air.zarr"
    summary = ch.run(_args("ai_rossby", in_dir, out))

    assert summary["n_dropped_feb29"] == 1
    assert summary["n_init"] == 2

    r = xr.open_zarr(out, consolidated=True, decode_timedelta=False)
    try:
        assert r.sizes["init_time"] == 2
        assert r.sizes["lead_time"] == NTIME
        assert r.sizes["pressure_level"] == NLEV
        # ensemble axis squeezed away.
        assert "ensemble" not in r.dims

        it = r["init_time"].values
        assert str(it[0])[:10] == "2000-01-01"
        assert str(it[1])[:10] == "2000-01-09"

        assert set(r.attrs["surface_variables"]) == set(SURF_VARS)
        assert set(r.attrs["upper_air_variables"]) == set(UPPER_VARS)
        assert set(r.attrs["diagnostic_variables"]) == set(DIAG_VARS)

        # frame 0 (IC) -> lead 0 (async_writer includes the IC; no prepend).
        exp0 = _field_value(0.0, "geopotential", 0)  # unused surface baseline sanity
        expZ = _field_value(2.0, "geopotential", 0)
        np.testing.assert_allclose(
            r["geopotential"].isel(init_time=0, lead_time=0, pressure_level=2).values, expZ)
        exp_t2m0 = _field_value(0.0, "2m_temperature", 0)
        np.testing.assert_allclose(
            r["2m_temperature"].isel(init_time=0, lead_time=0).values, exp_t2m0)
        exp_t2m5 = _field_value(0.0, "2m_temperature", 5)
        np.testing.assert_allclose(
            r["2m_temperature"].isel(init_time=0, lead_time=5).values, exp_t2m5)

        for v in SURF_VARS + DIAG_VARS + UPPER_VARS:
            assert bool(np.isfinite(r[v].values).all()), f"{v} not finite"
        assert r.attrs["n_lead"] == NTIME
    finally:
        r.close()


def test_ai_rossby_frame_zero_is_ic_detected(tmp_path):
    """The async_writer output has frame_zero_is_ic=1 -> no prepend, no NaN lead 0."""
    p = tmp_path / "one.zarr"
    _make_ai_rossby_store(p, cftime.DatetimeGregorian(2000, 3, 1, 0))
    it = ch.load_ai_rossby_init(p, prepend_ic=False)
    assert it["surface"]["2m_temperature"].shape[0] == NTIME
    # lead 0 is the IC frame (finite), not a prepended NaN.
    assert np.isfinite(it["surface"]["2m_temperature"][0]).all()
    np.testing.assert_allclose(
        it["surface"]["2m_temperature"][0], _field_value(0.0, "2m_temperature", 0))


def test_expected_schedule_counts():
    # Leap year: Feb 29 excluded -> February has 7 slots, total 95.
    sched = ch.expected_schedule(2000)
    assert len(sched) == 95
    assert (2, 29) not in sched
    feb = [d for (m, d) in sched if m == 2]
    assert len(feb) == 7
    # Non-leap year: also 95 (Feb 29 doesn't exist).
    assert len(ch.expected_schedule(2001)) == 95
