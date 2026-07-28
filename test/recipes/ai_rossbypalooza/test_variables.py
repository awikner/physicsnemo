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

"""Tests for canonical variable identity (datapipes/variables.py)."""

from __future__ import annotations

import pytest

from datapipes.variables import (
    PRECIP_INDEX,
    Channel,
    ChannelLayout,
    canonicalize_scalar,
    canonicalize_upper,
    levels_match,
    parse_flat_name,
)


@pytest.mark.parametrize(
    ("flat", "expected"),
    [
        ("z_500", ("geopotential", 500.0)),
        ("q_850", ("specific_humidity", 850.0)),
        ("t_850", ("temperature", 850.0)),
        ("u_200", ("u_component_of_wind", 200.0)),
        ("2t", ("2m_temperature", None)),
        ("10u", ("10m_u_component_of_wind", None)),
        ("10v", ("10m_v_component_of_wind", None)),
        ("msl", ("mean_sea_level_pressure", None)),
        ("sp", ("surface_pressure", None)),
        ("tp", ("total_precipitation_24hr", None)),
        ("T_850", ("temperature", 850.0)),  # case-insensitive
        ("geopotential", ("geopotential", None)),  # 3-D name, level from coord
        ("z_50000", ("geopotential", 500.0)),  # Pa level auto-converted
        ("swvl1", ("volumetric_soil_water_layer_1", None)),
        ("stl1", ("soil_temperature_level_1", None)),
        ("soil_temperature_level_1", ("soil_temperature_level_1", None)),
        ("u_component_of_wind_250", ("u_component_of_wind", 250.0)),
        ("2d", ("2m_dewpoint_temperature", None)),
        ("tcw", ("total_column_water", None)),
        ("unknown_var", None),
        ("z_abc", None),  # non-numeric suffix, not upper-air
    ],
)
def test_parse_flat_name(flat, expected):
    assert parse_flat_name(flat) == expected


def test_canonicalize_lookups():
    assert canonicalize_scalar("T2M") == "2m_temperature"
    assert canonicalize_scalar("geopotential") is None
    assert canonicalize_upper("Z") == "geopotential"
    assert canonicalize_upper("tp") is None


def test_levels_match_tolerance():
    assert levels_match(500.0, 500.0)
    assert levels_match(500.0001, 500.0)
    assert not levels_match(500.0, 850.0)
    assert not levels_match(501.0, 500.0)


def test_layout_order_and_indexing():
    layout = ChannelLayout(["z/500", "temperature/850", "2t", "msl"])
    # Aliases normalize to canonical channels, in config order.
    assert layout.channels == (
        Channel("geopotential", 500.0),
        Channel("temperature", 850.0),
        Channel("2m_temperature", None),
        Channel("mean_sea_level_pressure", None),
    )
    assert layout.num_channels == 5  # precip + 4
    assert layout.channel_names[PRECIP_INDEX] == "precip_mm_day"
    # index_of returns positions offset past the precip slot.
    assert layout.index_of("geopotential", 500.0) == 1
    assert layout.index_of("geopotential", 500.0002) == 1  # by-value tolerance
    assert layout.index_of("temperature", 850.0) == 2
    assert layout.index_of("2m_temperature", None) == 3
    assert layout.index_of("geopotential", 850.0) is None
    assert layout.index_of("specific_humidity", 700.0) is None


def test_layout_cross_schema_identity():
    """The same physical variable from both schemas maps to one slot."""
    layout = ChannelLayout(["geopotential/500"])
    # Schema A: flat name.
    canon, level = parse_flat_name("z_500")
    idx_a = layout.index_of(canon, level)
    # Schema B: canonical 3-D name + level from the pressure_level coord.
    idx_b = layout.index_of(canonicalize_upper("geopotential"), 500.0)
    assert idx_a == idx_b == 1


def test_layout_upper_air_levels():
    layout = ChannelLayout(["z/500", "z/850", "q/700"])
    assert layout.upper_air_levels("geopotential") == [500.0, 850.0]
    assert layout.upper_air_levels("specific_humidity") == [700.0]
    assert layout.upper_air_levels("temperature") == []


def test_layout_rejects_unknown_and_duplicates():
    with pytest.raises(ValueError, match="not a known"):
        ChannelLayout(["not_a_variable"])
    with pytest.raises(ValueError, match="not a known upper-air"):
        ChannelLayout(["tp/500"])
    with pytest.raises(ValueError, match="duplicate"):
        ChannelLayout(["z/500", "geopotential/500"])
