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

"""Tests for normalization-stat assembly (datapipes/stats.py)."""

from __future__ import annotations

import numpy as np
import pytest

from datapipes.stats import ChannelStats
from datapipes.testing import write_stats_store
from datapipes.variables import ChannelLayout


@pytest.fixture()
def stats_paths(tmp_path):
    era5 = write_stats_store(
        tmp_path / "era5_stats.zarr",
        surface={"2m_temperature": (280.0, 15.0)},
        upper={
            "geopotential": {500.0: (54000.0, 3000.0), 850.0: (14000.0, 1500.0)},
            "temperature": {500.0: (250.0, 10.0), 850.0: (280.0, 12.0)},
        },
        level_units="Pa",  # exercises the Pa -> hPa auto-detect
    )
    precip = write_stats_store(
        tmp_path / "imerg_stats.zarr",
        surface={"total_precipitation_24hr": (3.0, 8.0)},
    )
    return era5, precip


def test_assembly_order_and_values(stats_paths):
    era5, precip = stats_paths
    layout = ChannelLayout(["z/500", "t/850", "2t"])
    s = ChannelStats(era5, precip, layout)
    assert s.mean.shape == s.std.shape == (4, 1, 1)
    np.testing.assert_allclose(
        s.mean.ravel(), [3.0, 54000.0, 280.0, 280.0]
    )
    np.testing.assert_allclose(s.std.ravel(), [8.0, 3000.0, 12.0, 15.0])
    assert s.precip_mean == 3.0
    assert s.precip_std == 8.0


def test_missing_level_raises(stats_paths):
    era5, precip = stats_paths
    layout = ChannelLayout(["z/200"])
    with pytest.raises(ValueError, match="no level near 200"):
        ChannelStats(era5, precip, layout)


def test_missing_variable_raises(stats_paths):
    era5, precip = stats_paths
    layout = ChannelLayout(["q/500"])
    with pytest.raises(KeyError, match="specific_humidity"):
        ChannelStats(era5, precip, layout)


def test_missing_precip_store_names_generator(stats_paths, tmp_path):
    era5, _ = stats_paths
    layout = ChannelLayout(["2t"])
    with pytest.raises(FileNotFoundError, match="compute_precip_norm"):
        ChannelStats(era5, tmp_path / "nope.zarr", layout)


def test_bad_std_raises(tmp_path, stats_paths):
    _, precip = stats_paths
    era5 = write_stats_store(
        tmp_path / "bad_stats.zarr",
        surface={"2m_temperature": (280.0, 0.0)},
    )
    with pytest.raises(ValueError, match="non-positive normalization std"):
        ChannelStats(era5, precip, ChannelLayout(["2t"]))
