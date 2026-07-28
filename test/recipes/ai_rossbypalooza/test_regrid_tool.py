# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the offline DSI regrid tool (tools/regrid_dsi_to_1deg.py)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from datapipes.regrid import Regridder
from datapipes.testing import (
    FINE_LAT,
    FINE_LON,
    GRID_LAT,
    GRID_LON,
    write_imerg_store,
    write_schema_a_store,
)
from tools.regrid_dsi_to_1deg import main, regrid_store


@pytest.fixture()
def fine_store(tmp_path):
    """A DSI-schema store on the fine (0.25-deg-like) grid, with one
    non-constant variable so the pooling math is actually exercised."""
    path = write_schema_a_store(
        tmp_path / "src" / "test_model" / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 4)],
        vars_6h=("2t", "z_500"),
        vars_daily=("tp",),
        lead_hours=range(168, 199, 6),
        lead_days=(7, 8),
        lat=FINE_LAT,
        lon=FINE_LON,
    )
    # Overwrite one variable with a lat/lon-dependent field.
    ds = xr.open_zarr(path, consolidated=True, decode_times=False)
    shape = ds["2t"].shape
    lat2d = np.deg2rad(FINE_LAT)[:, None]
    lon2d = np.deg2rad(FINE_LON)[None, :]
    field = 280.0 + 10.0 * np.sin(lat2d) + 2.0 * np.cos(lon2d)
    data = np.broadcast_to(field, shape).astype("float32")
    ds.close()
    import zarr

    grp = zarr.open_group(str(path), mode="r+")
    grp["2t"][:] = data
    zarr.consolidate_metadata(str(path))
    return path


def test_regrid_store_values_and_schema(tmp_path, fine_store):
    dst = tmp_path / "out" / "test_model" / "2001.zarr"
    summary = regrid_store(
        fine_store, dst, GRID_LAT, GRID_LON, commit="testcommit"
    )
    assert summary["variables"] == 3
    assert summary["nan_vars"] == []

    src = xr.open_zarr(fine_store, consolidated=True)
    out = xr.open_zarr(dst, consolidated=True)

    # Grid replaced, everything else preserved.
    np.testing.assert_allclose(out["lat"].values, GRID_LAT)
    np.testing.assert_allclose(out["lon"].values, GRID_LON)
    np.testing.assert_array_equal(
        out["prediction_timedelta"].values, src["prediction_timedelta"].values
    )
    np.testing.assert_array_equal(
        out["prediction_timedelta_daily"].values,
        src["prediction_timedelta_daily"].values,
    )
    assert list(out["init_time"].values) == list(src["init_time"].values)
    assert out.attrs["regrid_method"] == "1d-conservative"
    assert out.attrs["regridded_from"] == str(fine_store)
    assert "testcommit" in out.attrs["generator"]
    assert out.attrs["channel_variables_6h"] == ["2t", "z_500"]
    assert out.attrs["channel_variables_daily"] == ["tp"]

    # Values match a direct Regridder application, per variable and init.
    r = Regridder(FINE_LAT, FINE_LON, GRID_LAT, GRID_LON)
    for v in ("2t", "z_500", "tp"):
        expected = np.stack(
            [r(src[v].isel(init_time=i).values) for i in range(2)]
        ).astype("float32")
        np.testing.assert_allclose(out[v].values, expected, rtol=1e-6)

    # Constant-in-space variables stay exactly constant.
    tp0 = out["tp"].isel(init_time=0, prediction_timedelta_daily=0).values
    assert np.allclose(tp0, tp0.flat[0])


def test_cli_sentinel_skip_and_ref_store(tmp_path, fine_store, caplog):
    src_root = fine_store.parents[1]
    out_root = tmp_path / "out1deg"
    ref = write_imerg_store(
        tmp_path / "imerg" / "2001.zarr", year=2001, months=(6,)
    )
    argv = [
        "--src-root", str(src_root),
        "--out-root", str(out_root),
        "--model", "all",
        "--ref-store", str(ref),
        "--n-workers", "1",
    ]
    assert main(argv) == 0
    assert (out_root / "test_model" / "2001.zarr").exists()
    assert (out_root / ".regrid_done" / "test_model_2001.done").exists()
    out = xr.open_zarr(out_root / "test_model" / "2001.zarr", consolidated=True)
    np.testing.assert_allclose(out["lat"].values, GRID_LAT)

    # Second run: sentinel makes it a no-op (would fail with mode="w-" otherwise).
    assert main(argv) == 0


def test_regrid_store_rejects_already_regridded(tmp_path):
    path = write_schema_a_store(
        tmp_path / "src" / "m" / "2001.zarr",
        year=2001,
        init_dates=[(6, 1)],
        lat=GRID_LAT,
        lon=GRID_LON,
    )
    with pytest.raises(ValueError, match="already on the target grid"):
        regrid_store(path, tmp_path / "o" / "2001.zarr", GRID_LAT, GRID_LON)
