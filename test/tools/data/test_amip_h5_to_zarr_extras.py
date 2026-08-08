# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12c: the AMIP converter drops absent extras instead of failing.

The daily-avg archive's 1986 files carry no ``*_climatology`` datasets
(every neighboring year does) — job 3383725 hard-failed the whole year on
an EXTRA variable. Extras are best-effort preservation: absent ones are
dropped with a warning; role-listed variables keep the hard error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "data" / "amip"
sys.path.insert(0, str(_TOOLS_DIR))

from amip_h5_to_zarr import convert  # noqa: E402

_H, _W = 8, 16
_LEVELS = [500.0, 850.0]


def _write_year(
    dirpath: Path,
    year: int,
    n_files: int,
    *,
    with_extras: bool,
    extras_until: int | None = None,
):
    """``extras_until=k`` writes extras only for the first ``k`` files —
    the 1986 mid-year-vanishing pattern."""
    for idx in range(n_files):
        extras_here = with_extras and (extras_until is None or idx < extras_until)
        with h5py.File(dirpath / f"{year}_{idx:04d}.h5", "w") as f:
            g = f.create_group("input")
            g["time"] = np.bytes_(f"{year}-01-{idx + 1:02d}T00:00:00.000000000")
            for v in ("t2m", "lsm", "sst"):
                g[v] = np.random.rand(_H, _W).astype("float32")
            for lev in _LEVELS:
                g[f"temperature_{lev}"] = np.random.rand(_H, _W).astype("float32")
            if extras_here:
                g["snow_depth"] = np.random.rand(_H, _W).astype("float32")
                for lev in _LEVELS:
                    g[f"vertical_velocity_{lev}"] = np.random.rand(_H, _W).astype(
                        "float32"
                    )


def _config():
    return {
        "surface_variables": ["t2m"],
        "pressure_upper_air_variables": ["temperature"],
        "constant_boundary_variables": ["lsm"],
        "varying_boundary_variables": ["sst"],
        "diagnostic_variables": [],
        "extra_variables": {
            "surface_variables": ["snow_depth"],
            "pressure_upper_air_variables": ["vertical_velocity"],
        },
        "levels": _LEVELS,
        "calendar": "standard",
        "horizontal_resolution": [_H, _W],
        "lat": list(np.linspace(89.5, -89.5, _H)),
        "lon": list((np.arange(_W) + 0.5).astype(float)),
        "data_timedelta_hours": 6,
    }


def test_missing_extras_dropped_with_warning(tmp_path, caplog):
    import logging

    src = tmp_path / "h5"
    src.mkdir()
    _write_year(src, 1986, 3, with_extras=False)  # archive gap year
    out = tmp_path / "1986.zarr"
    with caplog.at_level(logging.WARNING):
        convert(
            _config(),
            input_dir=src,
            year=1986,
            sample_range=None,
            output=out,
            time_chunk=1,
        )
    assert any("dropping extra_variables" in r.message for r in caplog.records)
    ds = xr.open_zarr(out)
    assert "snow_depth" not in ds.data_vars
    assert "vertical_velocity" not in ds.data_vars
    assert "t2m" in ds.data_vars  # role-listed vars unaffected
    assert ds.attrs["extra_surface_variables"] == []


def test_present_extras_still_written(tmp_path):
    src = tmp_path / "h5"
    src.mkdir()
    _write_year(src, 1987, 3, with_extras=True)
    out = tmp_path / "1987.zarr"
    convert(
        _config(),
        input_dir=src,
        year=1987,
        sample_range=None,
        output=out,
        time_chunk=1,
    )
    ds = xr.open_zarr(out)
    assert "snow_depth" in ds.data_vars
    assert "vertical_velocity" in ds.data_vars
    assert ds.attrs["extra_surface_variables"] == ["snow_depth"]


def test_extras_vanishing_mid_year_are_dropped(tmp_path, caplog):
    # The real 1986 failure mode: extras present in files 0000-0003, gone
    # from 0004 on. First-file probing missed it (job 3384825) — the scan
    # must cover every file.
    import logging

    src = tmp_path / "h5"
    src.mkdir()
    _write_year(src, 1986, 6, with_extras=True, extras_until=4)
    out = tmp_path / "1986.zarr"
    with caplog.at_level(logging.WARNING):
        convert(
            _config(),
            input_dir=src,
            year=1986,
            sample_range=None,
            output=out,
            time_chunk=1,
        )
    assert any("not present in every" in r.message for r in caplog.records)
    ds = xr.open_zarr(out)
    assert "snow_depth" not in ds.data_vars
    assert "vertical_velocity" not in ds.data_vars
    assert ds.sizes["time"] == 6  # the year itself converts fully


def test_missing_role_listed_var_still_hard_fails(tmp_path):
    src = tmp_path / "h5"
    src.mkdir()
    _write_year(src, 1990, 2, with_extras=True)
    cfg = _config()
    cfg["surface_variables"] = ["t2m", "not_in_archive"]
    with pytest.raises(KeyError, match="not_in_archive"):
        convert(
            cfg,
            input_dir=src,
            year=1990,
            sample_range=None,
            output=tmp_path / "1990.zarr",
            time_chunk=1,
        )