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

"""Tests for the schema adapters (datapipes/adapters.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from datapipes.adapters import SchemaAAdapter, SchemaBAdapter, build_adapter
from datapipes.precip import PrecipSpec
from datapipes.testing import (
    FINE_LAT,
    FINE_LON,
    GRID_LAT,
    GRID_LON,
    coded_value,
    write_schema_a_store,
    write_schema_b_store,
)
from datapipes.variables import ChannelLayout


def resolve(reqs, roots: dict[str, Path]) -> list[np.ndarray]:
    """Resolve ReadRequests against stores on disk (sync zarr)."""
    out = []
    for r in reqs:
        owner, year, var = r.array_key
        grp = zarr.open_group(str(Path(roots[owner]) / f"{year}.zarr"), mode="r")
        arr = grp[var]
        sel = tuple(
            np.asarray(i) if isinstance(i, list) else i for i in r.index
        )
        if any(isinstance(i, np.ndarray) for i in sel):
            out.append(arr.oindex[sel])
        else:
            out.append(arr[sel])
    return out


VAR_CODES_A = {"2t": 0, "z_500": 1, "tp": 2, "swvl1": 3}


@pytest.fixture()
def schema_a_root(tmp_path):
    root = tmp_path / "model_a"
    write_schema_a_store(
        root / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 4)],
        vars_6h=("2t", "z_500", "swvl1"),
        vars_daily=("tp",),
        lead_hours=range(168, 361, 6),
        lead_days=range(7, 16),
        var_codes=VAR_CODES_A,
    )
    return root


def _layout():
    return ChannelLayout(["z/500", "2t"])


def _adapter_a(root, **precip_kw):
    spec = dict(var="tp", axis="daily", kind="accum", units="mm")
    spec.update(precip_kw)
    return SchemaAAdapter(
        "model_a", root, _layout(), PrecipSpec(**spec)
    )


class TestSchemaA:
    def test_discover_and_masks(self, schema_a_root):
        a = _adapter_a(schema_a_root)
        a.discover(GRID_LAT, GRID_LON)
        assert set(a.init_lookup()) == {(2001, 6, 1, 0), (2001, 6, 4, 0)}
        assert a.init_lookup()[(2001, 6, 4, 0)] == (2001, 1)
        # precip + z/500 + 2t all supplied; swvl1 unmapped and ignored.
        np.testing.assert_array_equal(a.channel_mask, [True, True, True])

    def test_lead_supported(self, schema_a_root):
        a = _adapter_a(schema_a_root)
        a.discover(GRID_LAT, GRID_LON)
        assert a.lead_supported(8)
        assert a.lead_supported(15)
        assert not a.lead_supported(16)  # beyond both windows
        assert not a.lead_supported(6)  # below both windows

    def test_plan_assemble_daily_precip(self, schema_a_root):
        a = _adapter_a(schema_a_root)
        a.discover(GRID_LAT, GRID_LON)
        reqs = a.plan(2001, 1, 8)
        arrays = resolve(reqs, {"model_a": schema_a_root})
        block = a.assemble(arrays, 8)
        assert block.shape == (3, 8, 8)
        # ch 0: tp daily accum in mm at day 8, init 1.
        np.testing.assert_allclose(block[0], coded_value(2, 1, 8), rtol=1e-6)
        # ch 1 (z/500) and ch 2 (2t) at lead hour 192.
        np.testing.assert_allclose(block[1], coded_value(1, 1, 192), rtol=1e-6)
        np.testing.assert_allclose(block[2], coded_value(0, 1, 192), rtol=1e-6)

    def test_assemble_6h_precip_sum(self, tmp_path):
        root = tmp_path / "model_6h"
        write_schema_a_store(
            root / "2001.zarr",
            year=2001,
            init_dates=[(6, 1)],
            vars_6h=("2t", "z_500", "tp"),
            vars_daily=(),
            lead_hours=range(168, 361, 6),
            lead_days=(),
            var_codes=VAR_CODES_A,
        )
        a = _adapter_a(root, axis="6h", units="m")
        a.discover(GRID_LAT, GRID_LON)
        reqs = a.plan(2001, 0, 8)
        arrays = resolve(reqs, {"model_a": root})
        block = a.assemble(arrays, 8)
        # Sum of the four 6h accumulations ending at 192h, m -> mm.
        expected = sum(coded_value(2, 0, h) for h in (174, 180, 186, 192)) * 1000
        np.testing.assert_allclose(block[0], expected, rtol=1e-6)

    def test_native_grid_raises_with_regrid_hint(self, tmp_path):
        root = tmp_path / "native"
        write_schema_a_store(
            root / "2001.zarr",
            year=2001,
            init_dates=[(6, 1)],
            lat=FINE_LAT,
            lon=FINE_LON,
        )
        a = _adapter_a(root)
        with pytest.raises(ValueError, match="regrid_dsi_to_1deg"):
            a.discover(GRID_LAT, GRID_LON)

    def test_heterogeneous_years_raise(self, schema_a_root):
        write_schema_a_store(
            schema_a_root / "2002.zarr",
            year=2002,
            init_dates=[(6, 1)],
            lead_hours=range(168, 337, 6),  # different lead window
            lead_days=range(7, 15),
            var_codes=VAR_CODES_A,
        )
        a = _adapter_a(schema_a_root)
        with pytest.raises(ValueError, match="differ from the first year"):
            a.discover(GRID_LAT, GRID_LON)

    def test_missing_precip_var_raises(self, tmp_path):
        root = tmp_path / "noprecip"
        write_schema_a_store(
            root / "2001.zarr",
            year=2001,
            init_dates=[(6, 1)],
            vars_6h=("2t",),
            vars_daily=(),
        )
        a = _adapter_a(root)
        with pytest.raises(ValueError, match="precip var 'tp' not in"):
            a.discover(GRID_LAT, GRID_LON)


VAR_CODES_B = {
    "2m_temperature": 50,
    "total_precipitation_24hr": 51,
    "geopotential": 52,
}


@pytest.fixture()
def schema_b_root(tmp_path):
    root = tmp_path / "model_b"
    write_schema_b_store(
        root / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 5), (6, 9)],
        pressure_levels=(850.0, 500.0),
        n_lead=17,
        var_codes=VAR_CODES_B,
    )
    return root


def _adapter_b(root, layout=None):
    return SchemaBAdapter(
        "model_b",
        root,
        layout or _layout(),
        PrecipSpec("total_precipitation_24hr", axis="daily", kind="accum", units="mm"),
    )


class TestSchemaB:
    def test_discover_and_level_pick(self, schema_b_root):
        b = _adapter_b(schema_b_root)
        b.discover(GRID_LAT, GRID_LON)
        assert len(b.init_lookup()) == 3
        np.testing.assert_array_equal(b.channel_mask, [True, True, True])
        reqs = b.plan(2001, 2, 8)
        arrays = resolve(reqs, {"model_b": schema_b_root})
        block = b.assemble(arrays, 8)
        # ch 0: precip (mm) at lead day 8, init 2.
        np.testing.assert_allclose(block[0], coded_value(51, 2, 8), rtol=1e-6)
        # ch 1: geopotential @ 500 hPa = store level index 1 -> value + 1.
        np.testing.assert_allclose(
            block[1], coded_value(52, 2, 8) + 1, rtol=1e-6
        )
        # ch 2: 2m_temperature.
        np.testing.assert_allclose(block[2], coded_value(50, 2, 8), rtol=1e-6)

    def test_lead_supported_bounds(self, schema_b_root):
        b = _adapter_b(schema_b_root)
        b.discover(GRID_LAT, GRID_LON)
        assert not b.lead_supported(0)  # lead 0 is the IC
        assert b.lead_supported(1)
        assert b.lead_supported(16)
        assert not b.lead_supported(17)

    def test_missing_master_level_raises(self, schema_b_root):
        layout = ChannelLayout(["z/200", "2t"])
        b = _adapter_b(schema_b_root, layout=layout)
        with pytest.raises(ValueError, match="master level 200.0"):
            b.discover(GRID_LAT, GRID_LON)

    def test_absent_variable_masks_channel(self, schema_b_root):
        layout = ChannelLayout(["z/500", "q/700", "2t"])
        b = _adapter_b(schema_b_root, layout=layout)
        b.discover(GRID_LAT, GRID_LON)
        # q/700: store has no specific_humidity at all -> channel masked off.
        np.testing.assert_array_equal(
            b.channel_mask, [True, True, False, True]
        )


def test_cross_schema_identity(tmp_path, schema_a_root, schema_b_root):
    """z_500 (schema A) and geopotential@500 (schema B) share a channel."""
    layout = ChannelLayout(["geopotential/500"])
    a = SchemaAAdapter(
        "model_a", schema_a_root, layout,
        PrecipSpec("tp", axis="daily", kind="accum", units="mm"),
    )
    b = SchemaBAdapter(
        "model_b", schema_b_root, layout,
        PrecipSpec("total_precipitation_24hr", axis="daily", kind="accum", units="mm"),
    )
    a.discover(GRID_LAT, GRID_LON)
    b.discover(GRID_LAT, GRID_LON)
    block_a = a.assemble(resolve(a.plan(2001, 0, 8), {"model_a": schema_a_root}), 8)
    block_b = b.assemble(resolve(b.plan(2001, 0, 8), {"model_b": schema_b_root}), 8)
    assert block_a.shape == block_b.shape == (2, 8, 8)
    # Both experts populate index 1 (the shared master channel), and both
    # masks agree that it is supplied.
    np.testing.assert_allclose(block_a[1], coded_value(1, 0, 192), rtol=1e-6)
    np.testing.assert_allclose(block_b[1], coded_value(52, 0, 8) + 1, rtol=1e-6)
    np.testing.assert_array_equal(a.channel_mask, b.channel_mask)


def test_build_adapter_factory(schema_a_root):
    a = build_adapter(
        "m", "dsi", schema_a_root, _layout(),
        PrecipSpec("tp", axis="daily", kind="accum", units="mm"),
    )
    assert isinstance(a, SchemaAAdapter)
    with pytest.raises(ValueError, match="unknown schema"):
        build_adapter("m", "netcdf", schema_a_root, _layout(),
                      PrecipSpec("tp", axis="daily", kind="accum", units="mm"))
