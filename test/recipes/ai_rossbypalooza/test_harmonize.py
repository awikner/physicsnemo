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

"""Tests for the harmonization tool (tools/harmonize_hindcasts.py) and the
HarmonizedAdapter reading its output."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from datapipes.adapters import HarmonizedAdapter, build_adapter
from datapipes.precip import PrecipSpec
from datapipes.testing import (
    FINE_LAT,
    FINE_LON,
    GRID_LAT,
    GRID_LON,
    coded_value,
    resolve,
    write_imerg_store,
    write_schema_a_store,
    write_schema_b_store,
)
from datapipes.variables import ChannelLayout
from tools.harmonize_hindcasts import main as harmonize_main

HARMONIZED_PRECIP = PrecipSpec(
    "total_precipitation_24hr", axis="daily", kind="accum", units="m"
)


@pytest.fixture()
def ref_store(tmp_path):
    return write_imerg_store(tmp_path / "imerg" / "2001.zarr", year=2001, months=(6,))


@pytest.fixture()
def dsi_pair(tmp_path):
    """e2s-like (fine grid, short names, daily tp in m) + wb2-like (fine grid,
    long names, 6h precip) with partially overlapping inits."""
    e2s = tmp_path / "src" / "graphcast_e2s"
    wb2 = tmp_path / "src" / "graphcast_wb2"
    write_schema_a_store(
        e2s / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 4)],
        vars_6h=("2t", "q_850"),
        vars_daily=("tp", "z_500"),
        lead_hours=range(168, 361, 6),
        lead_days=range(7, 16),
        lat=FINE_LAT,
        lon=FINE_LON,
        var_codes={"2t": 0, "q_850": 1, "tp": 2, "z_500": 3},
    )
    write_schema_a_store(
        wb2 / "2001.zarr",
        year=2001,
        init_dates=[(6, 4), (6, 8)],  # 6/4 overlaps e2s
        vars_6h=(
            "2m_temperature",
            "geopotential_500",
            "u_component_of_wind_250",
            "total_precipitation_6hr",
        ),
        vars_daily=(),
        lead_hours=range(168, 361, 6),
        lead_days=(),
        lat=FINE_LAT,
        lon=FINE_LON,
        var_codes={
            "2m_temperature": 10,
            "geopotential_500": 11,
            "u_component_of_wind_250": 12,
            "total_precipitation_6hr": 13,
        },
    )
    return e2s, wb2


def _run_dsi(tmp_path, ref_store, e2s, wb2):
    out_root = tmp_path / "mowe"
    rc = harmonize_main([
        "--source", "dsi", "--model", "graphcast",
        "--src-root", str(e2s), "--src-root", str(wb2),
        "--src-label", "e2s", "--src-label", "wb2",
        "--out-root", str(out_root), "--ref-store", str(ref_store),
        "--n-workers", "1", "--commit", "test",
    ])
    assert rc == 0
    return out_root


def test_dsi_merge_naming_and_values(tmp_path, ref_store, dsi_pair):
    e2s, wb2 = dsi_pair
    out_root = _run_dsi(tmp_path, ref_store, e2s, wb2)
    ds = xr.open_zarr(out_root / "graphcast" / "2001.zarr", consolidated=True, decode_timedelta=False)

    # Canonical flat names, union of both sources.
    assert sorted(ds.data_vars) == [
        "2m_temperature",
        "geopotential_500",
        "specific_humidity_850",
        "total_precipitation_24hr",
        "u_component_of_wind_250",
    ]
    # Unified daily lead axis (values, not indices), 1-degree grid.
    assert list(ds["lead_time"].values) == list(range(7, 16))
    np.testing.assert_allclose(ds["lat"].values, GRID_LAT)
    assert ds.attrs["mowe_hindcast_schema_version"] == "1.0"
    assert ds["total_precipitation_24hr"].attrs["units"] == "m"
    assert ds["geopotential_500"].attrs["units"] == "m**2 s**-2"

    # Union of inits with e2s-priority provenance.
    assert list(ds["init_source"].values) == ["e2s", "e2s", "wb2"]

    # Fields are spatially constant -> regridding preserves them exactly.
    # init 6/1 (e2s only, local idx 0), day 8:
    v = ds["specific_humidity_850"].sel(init_time="2001-06-01", lead_time=8).values
    np.testing.assert_allclose(v, coded_value(1, 0, 192), rtol=1e-6)
    # e2s daily tp (m) passes through unchanged (rename only):
    v = ds["total_precipitation_24hr"].sel(init_time="2001-06-01", lead_time=8).values
    np.testing.assert_allclose(v, coded_value(2, 0, 8), rtol=1e-6)
    # Overlap init 6/4: e2s wins for shared vars (2m_temperature) ...
    v = ds["2m_temperature"].sel(init_time="2001-06-04", lead_time=8).values
    np.testing.assert_allclose(v, coded_value(0, 1, 192), rtol=1e-6)
    # ... but wb2-only u_250 is filled from wb2 (its local idx for 6/4 is 0).
    v = ds["u_component_of_wind_250"].sel(init_time="2001-06-04", lead_time=8).values
    np.testing.assert_allclose(v, coded_value(12, 0, 192), rtol=1e-6)
    # u_250 is NaN for the e2s-only init.
    v = ds["u_component_of_wind_250"].sel(init_time="2001-06-01", lead_time=8).values
    assert np.isnan(v).all()


def test_dsi_wb2_precip_accumulation(tmp_path, ref_store, dsi_pair):
    e2s, wb2 = dsi_pair
    out_root = _run_dsi(tmp_path, ref_store, e2s, wb2)
    ds = xr.open_zarr(out_root / "graphcast" / "2001.zarr", consolidated=True, decode_timedelta=False)
    # wb2-only init 6/8 (its local idx 1): tp = sum of four 6h steps, m.
    v = ds["total_precipitation_24hr"].sel(init_time="2001-06-08", lead_time=8).values
    expected = sum(coded_value(13, 1, h) for h in (174, 180, 186, 192))
    np.testing.assert_allclose(v, expected, rtol=1e-6)
    # Day 7 needs hours 150..168; only 168 exists -> NaN.
    v = ds["total_precipitation_24hr"].sel(init_time="2001-06-08", lead_time=7).values
    assert np.isnan(v).all()
    # e2s-sourced init: daily tp exists at day 7.
    v = ds["total_precipitation_24hr"].sel(init_time="2001-06-01", lead_time=7).values
    assert np.isfinite(v).all()


def test_consolidated_subset_flatten(tmp_path, ref_store):
    src = tmp_path / "src" / "pangu_s2s"
    write_schema_b_store(
        src / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 5)],
        surface_vars=("2m_temperature", "mean_sea_level_pressure"),
        diagnostic_vars=("total_precipitation_24hr",),
        upper_vars=("geopotential", "u_component_of_wind"),
        pressure_levels=(850.0, 500.0),
        n_lead=17,
        var_codes={
            "2m_temperature": 50,
            "mean_sea_level_pressure": 51,
            "total_precipitation_24hr": 52,
            "geopotential": 53,
            "u_component_of_wind": 54,
        },
    )
    out_root = tmp_path / "mowe"
    rc = harmonize_main([
        "--source", "consolidated", "--model", "pangu_s2s",
        "--src-root", str(src),
        "--variables", "mean_sea_level_pressure", "total_precipitation_24hr",
        "geopotential_500", "geopotential_850", "u_component_of_wind_500",
        "--out-root", str(out_root), "--ref-store", str(ref_store),
        "--n-workers", "1", "--commit", "test",
    ])
    assert rc == 0
    ds = xr.open_zarr(out_root / "pangu_s2s" / "2001.zarr", consolidated=True, decode_timedelta=False)
    # Subset only; 3-D flattened with integer level suffix; 2t dropped.
    assert sorted(ds.data_vars) == [
        "geopotential_500",
        "geopotential_850",
        "mean_sea_level_pressure",
        "total_precipitation_24hr",
        "u_component_of_wind_500",
    ]
    assert list(ds["lead_time"].values) == list(range(17))
    # Level pick by value: 500 hPa is store level index 1 -> value + 1.
    v = ds["geopotential_500"].sel(init_time="2001-06-05", lead_time=8).values
    np.testing.assert_allclose(v, coded_value(53, 1, 8) + 1, rtol=1e-6)
    v = ds["geopotential_850"].sel(init_time="2001-06-05", lead_time=8).values
    np.testing.assert_allclose(v, coded_value(53, 1, 8) + 0, rtol=1e-6)
    # Lead-0 trailing-24h diagnostics are NaN; instantaneous vars are not.
    assert np.isnan(
        ds["total_precipitation_24hr"].isel(init_time=0, lead_time=0).values
    ).all()
    assert np.isfinite(
        ds["mean_sea_level_pressure"].isel(init_time=0, lead_time=0).values
    ).all()
    # Missing requested variable raises.
    rc = harmonize_main([
        "--source", "consolidated", "--model", "pangu_bad",
        "--src-root", str(src), "--variables", "specific_humidity_700",
        "--out-root", str(out_root), "--ref-store", str(ref_store),
        "--n-workers", "1",
    ])
    assert rc == 1


def test_dsi_variable_list_filter(tmp_path, ref_store, dsi_pair):
    """--variables in dsi mode drops everything outside the master list."""
    e2s, wb2 = dsi_pair
    out_root = tmp_path / "mowe_filtered"
    rc = harmonize_main([
        "--source", "dsi", "--model", "graphcast",
        "--src-root", str(e2s), "--src-root", str(wb2),
        "--src-label", "e2s", "--src-label", "wb2",
        "--variables", "total_precipitation_24hr", "geopotential_500",
        "u_component_of_wind_250", "sea_surface_temperature",  # sst absent
        "--out-root", str(out_root), "--ref-store", str(ref_store),
        "--n-workers", "1", "--commit", "test",
    ])
    assert rc == 0
    ds = xr.open_zarr(
        out_root / "graphcast" / "2001.zarr", consolidated=True,
        decode_timedelta=False,
    )
    # Intersection only: 2m_temperature / q_850 dropped; absent sst simply
    # missing (available per-model coverage varies).
    assert sorted(ds.data_vars) == [
        "geopotential_500",
        "total_precipitation_24hr",
        "u_component_of_wind_250",
    ]


def test_sentinel_skip(tmp_path, ref_store, dsi_pair):
    e2s, wb2 = dsi_pair
    out_root = _run_dsi(tmp_path, ref_store, e2s, wb2)
    assert (out_root / ".harmonize_done" / "graphcast_2001.done").exists()
    # Second run: no-op (mode="w-" would fail otherwise).
    _run_dsi(tmp_path, ref_store, e2s, wb2)


def test_harmonized_adapter_reads_tool_output(tmp_path, ref_store, dsi_pair):
    e2s, wb2 = dsi_pair
    out_root = _run_dsi(tmp_path, ref_store, e2s, wb2)
    layout = ChannelLayout(["z/500", "q/850", "2t", "u_component_of_wind/250"])
    adapter = build_adapter(
        "graphcast", "harmonized", out_root / "graphcast", layout,
        HARMONIZED_PRECIP,
    )
    assert isinstance(adapter, HarmonizedAdapter)
    adapter.discover(GRID_LAT, GRID_LON)
    assert set(adapter.init_lookup()) == {
        (2001, 6, 1, 0), (2001, 6, 4, 0), (2001, 6, 8, 0),
    }
    # All master channels supplied (u_250 exists as a variable).
    np.testing.assert_array_equal(
        adapter.channel_mask, [True, True, True, True, True]
    )
    assert adapter.lead_supported(8)
    assert not adapter.lead_supported(16)
    assert not adapter.lead_supported(0)

    # Assemble init 6/1 (e2s-sourced), tau 8; resolve with sync zarr.
    reqs = adapter.plan(2001, 0, 8)
    block = adapter.assemble(resolve(reqs, {"graphcast": out_root / "graphcast"}), 8)
    assert block.shape == (5, 8, 8)
    # ch 0: tp in m * 1000 -> mm/day.
    np.testing.assert_allclose(block[0], coded_value(2, 0, 8) * 1000, rtol=1e-6)
    np.testing.assert_allclose(block[1], coded_value(3, 0, 8), rtol=1e-6)  # z500 daily
    np.testing.assert_allclose(block[2], coded_value(1, 0, 192), rtol=1e-6)  # q850 6h
    # u_250: NaN for this e2s-only init (handled downstream by the dataset).
    assert np.isnan(block[4]).all()
