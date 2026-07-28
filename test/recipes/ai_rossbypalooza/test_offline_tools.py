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

"""Tests for the offline stats tools (precip norm + SEEPS climatology)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from datapipes.testing import write_imerg_store
from tools.compute_precip_norm import main as norm_main
from tools.compute_seeps_climatology import main as seeps_main


def test_precip_norm_end_to_end(tmp_path):
    root = tmp_path / "imerg"
    # Two years, constant value_fn -> exactly computable stats.
    write_imerg_store(root / "2001.zarr", year=2001, months=(6,),
                      value_fn=lambda d: 4.0)
    write_imerg_store(root / "2002.zarr", year=2002, months=(6,),
                      value_fn=lambda d: 8.0)
    out = tmp_path / "stats.zarr"
    assert norm_main([
        "--imerg-root", str(root), "--years", "2001-2002",
        "--out", str(out),
    ]) == 0
    ds = xr.open_zarr(out, consolidated=True)
    mean = float(ds["total_precipitation_24hr"].sel(stat="mean"))
    std = float(ds["total_precipitation_24hr"].sel(stat="std"))
    assert abs(mean - 6.0) < 1e-9
    assert abs(std - 2.0) < 1e-9
    assert ds.attrs["units"] == "mm/day"


def test_precip_norm_month_filter(tmp_path):
    root = tmp_path / "imerg"
    write_imerg_store(
        root / "2001.zarr", year=2001, months=(6, 7),
        value_fn=lambda d: 3.0 if d.month == 6 else 9.0,
    )
    out = tmp_path / "stats_jun.zarr"
    norm_main([
        "--imerg-root", str(root), "--years", "2001",
        "--months", "6", "--out", str(out),
    ])
    ds = xr.open_zarr(out, consolidated=True)
    assert abs(float(ds["total_precipitation_24hr"].sel(stat="mean")) - 3.0) < 1e-9


def test_seeps_climatology_p1_and_t2(tmp_path):
    root = tmp_path / "imerg"

    # June: alternate dry (0.1 mm) and wet days with amounts 1..30.
    def value_fn(d):
        return 0.1 if d.day % 2 == 0 else float(d.day)

    for year in (2001, 2002):
        write_imerg_store(
            root / f"{year}.zarr", year=year,
            months=tuple(range(1, 13)), value_fn=value_fn,
        )
    out = tmp_path / "clim.zarr"
    assert seeps_main([
        "--imerg-root", str(root), "--years", "2001-2002",
        "--out", str(out),
    ]) == 0
    ds = xr.open_zarr(out, consolidated=True)
    assert ds["p1"].shape == (12, 8, 8)
    # June: 15 even days (dry) of 30 -> p1 = 0.5 exactly (both years alike).
    june_p1 = float(ds["p1"].sel(month=6)[0, 0])
    assert abs(june_p1 - 0.5) < 1e-6
    # Wet-day amounts in June: odd days 1,3,...,29 -> 2/3 quantile ~ 19.67.
    june_t2 = float(ds["t2"].sel(month=6)[0, 0])
    wet = np.array([d for d in range(1, 31) if d % 2 == 1], dtype=float)
    expected = np.quantile(np.concatenate([wet, wet]), 2 / 3)
    assert abs(june_t2 - expected) < 1e-5
    assert ds.attrs["dry_threshold_mm"] == 0.25
