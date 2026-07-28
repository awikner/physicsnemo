# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for dsi_hindcast_to_formats.py.

Builds a tiny synthetic per-init source (two inits, two lead axes, flat
channels, lat N->S), runs the converter, and asserts both output formats:
lead-window bounds, ragged 6h h5 series, lat orientation (S->N in h5, N->S in
zarr), and value round-trip.  Run in the hindcast_conv env (xarray/zarr>=3.1.3/
numcodecs/h5py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsi_hindcast_to_formats as conv  # noqa: E402

LAT = np.array([3.0, 1.0, -1.0, -3.0], dtype="float32")   # N->S
LON = np.array([0.0, 120.0, 240.0], dtype="float32")
PT_HOURS = np.arange(6, 193, 6, dtype="int64")            # 6..192
PT_DAYS = np.arange(1, 9, dtype="int64")                  # 1..8


def _make_source(dir_: Path, init_iso: str):
    """One synthetic init store: 2t on the 6h axis, tp on the daily axis."""
    nlat, nlon = LAT.size, LON.size
    t2 = (LAT[None, :, None] + PT_HOURS[:, None, None].astype("float32")) \
        * np.ones((1, PT_HOURS.size, nlat, nlon), dtype="float32")
    tp = (LAT[None, :, None] + PT_DAYS[:, None, None].astype("float32") * 100.0) \
        * np.ones((1, PT_DAYS.size, nlat, nlon), dtype="float32")
    ds = xr.Dataset(
        {
            "2t": (("time", "prediction_timedelta", "lat", "lon"), t2),
            "tp": (("time", "prediction_timedelta_daily", "lat", "lon"), tp),
        },
        coords={
            "time": ("time", [np.datetime64(init_iso)]),
            "prediction_timedelta": ("prediction_timedelta", PT_HOURS),
            "prediction_timedelta_daily": ("prediction_timedelta_daily", PT_DAYS),
            "lat": ("lat", LAT), "lon": ("lon", LON),
        },
    )
    stamp = init_iso.replace("-", "").replace(":", "")[:8]
    store = dir_ / f"init_{stamp}T00.zarr"
    ds.to_zarr(store, mode="w", zarr_format=3, consolidated=True)


def test_convert_both_formats(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_source(src, "2000-06-01")
    _make_source(src, "2000-06-05")
    out = tmp_path / "out"

    conv.main([
        "--model", "testm", "--source-kind", "zarr", "--source-dir", str(src),
        "--out-root", str(out), "--years", "2000",
        "--lead-hours", "168", "192", "--lead-days", "7", "8",
        "--n-workers", "1", "--overwrite",
    ])

    # ---- Format 1: ragged per-6h h5 series (lat S->N) ----
    idir = out / "h5" / "testm" / "2000060100"
    got = sorted(p.name for p in idir.glob("*.h5"))
    assert got == ["0168.h5", "0174.h5", "0180.h5", "0186.h5", "0192.h5"]

    with h5py.File(idir / "0174.h5", "r") as f:      # non-00Z: 6h only
        assert "2t" in f["input"] and "tp" not in f["input"]
        arr = f["input/2t"][:]
        assert arr.shape == (4, 3)
        # S->N: row 0 is southmost lat=-3 -> -3 + 174
        assert np.isclose(arr[0, 0], -3 + 174)
        assert np.isclose(arr[3, 0], 3 + 174)
        # 2000-06-01 00Z + 174 h = 2000-06-08 06:00
        assert f["input/time"][()].decode().startswith("2000-06-08T06:00")

    with h5py.File(idir / "0168.h5", "r") as f:      # 00Z day7: both
        assert "2t" in f["input"] and "tp" in f["input"]
        assert np.isclose(f["input/2t"][0, 0], -3 + 168)
        assert np.isclose(f["input/tp"][0, 0], -3 + 7 * 100)   # day 7, southmost

    # ---- Format 2: per-year zarr, two lead axes, lat N->S ----
    ds2 = xr.open_zarr(out / "zarr" / "testm" / "2000.zarr", consolidated=True,
                       decode_timedelta=False)
    assert ds2.sizes["init_time"] == 2
    assert list(ds2["prediction_timedelta"].values) == [168, 174, 180, 186, 192]
    assert list(ds2["prediction_timedelta_daily"].values) == [7, 8]
    assert float(ds2["lat"].values[0]) == 3.0 and float(ds2["lat"].values[-1]) == -3.0
    # round-trip: 2t at init0, lead 168, lat=3 (N, index 0)
    v = ds2["2t"].isel(init_time=0).sel(prediction_timedelta=168).isel(lat=0, lon=0)
    assert np.isclose(float(v), 3 + 168)
    v = ds2["tp"].isel(init_time=0).sel(prediction_timedelta_daily=7).isel(lat=3, lon=0)
    assert np.isclose(float(v), -3 + 7 * 100)
    assert ds2.attrs["channel_variables_6h"] == ["2t"]
    assert ds2.attrs["channel_variables_daily"] == ["tp"]
    assert ds2.attrs["lead_window_days"] == [7, 8]
    ds2.close()

    # ---- reverse path: expand Format-2 zarr -> h5, must match the forward h5 ----
    exp = tmp_path / "expanded"
    conv.main(["--model", "testm", "--out-root", str(exp),
               "--expand-zarr", str(out / "zarr" / "testm" / "2000.zarr")])
    eidir = exp / "h5" / "testm" / "2000060100"
    assert sorted(p.name for p in eidir.glob("*.h5")) == \
        ["0168.h5", "0174.h5", "0180.h5", "0186.h5", "0192.h5"]
    for fname in ("0168.h5", "0174.h5"):
        with h5py.File(eidir / fname) as fe, h5py.File(idir / fname) as ff:
            assert set(fe["input"].keys()) == set(ff["input"].keys())
            for k in fe["input"].keys():
                if k == "time":
                    assert fe["input/time"][()] == ff["input/time"][()]
                else:
                    assert np.array_equal(fe["input"][k][:], ff["input"][k][:])
