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

"""Tests for HindcastMixtureDataset (datapipes/dataset.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from datapipes.adapters import SchemaAAdapter, SchemaBAdapter
from datapipes.dataset import HindcastMixtureDataset
from datapipes.precip import PrecipSpec
from datapipes.stats import ChannelStats
from datapipes.testing import (
    GRID_LAT,
    GRID_LON,
    coded_value,
    write_imerg_store,
    write_schema_a_store,
    write_schema_b_store,
    write_stats_store,
)
from datapipes.truth import ImergTruth
from datapipes.variables import ChannelLayout

VAR_CODES_A = {"2t": 0, "z_500": 1, "tp": 2}
VAR_CODES_B = {
    "2m_temperature": 50,
    "total_precipitation_24hr": 51,
    "geopotential": 52,
}

# Stats chosen so normalized values are easy to verify by hand.
PRECIP_STATS = (5.0, 10.0)
Z500_STATS = (54000.0, 3000.0)
T2M_STATS = (280.0, 15.0)


@pytest.fixture()
def env(tmp_path):
    """Two experts (one per schema, partially overlapping inits) + truth."""
    a_root = tmp_path / "experts" / "model_a"
    b_root = tmp_path / "experts" / "model_b"
    # model_a (DSI schema): inits 6/1, 6/4; tp on the daily axis in mm.
    write_schema_a_store(
        a_root / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 4)],
        vars_6h=("2t", "z_500"),
        vars_daily=("tp",),
        lead_hours=range(168, 361, 6),
        lead_days=range(7, 16),
        var_codes=VAR_CODES_A,
    )
    # model_b (consolidated): inits 6/1, 6/5, 6/9.
    write_schema_b_store(
        b_root / "2001.zarr",
        year=2001,
        init_dates=[(6, 1), (6, 5), (6, 9)],
        pressure_levels=(850.0, 500.0),
        n_lead=17,
        var_codes=VAR_CODES_B,
    )
    truth_root = tmp_path / "imerg"
    write_imerg_store(truth_root / "2001.zarr", year=2001, months=(6,))

    era5_stats = write_stats_store(
        tmp_path / "era5_stats.zarr",
        surface={"2m_temperature": T2M_STATS},
        upper={"geopotential": {500.0: Z500_STATS, 850.0: (14000.0, 1500.0)}},
    )
    precip_stats = write_stats_store(
        tmp_path / "imerg_stats.zarr",
        surface={"total_precipitation_24hr": PRECIP_STATS},
    )
    layout = ChannelLayout(["z/500", "2t"])
    stats = ChannelStats(era5_stats, precip_stats, layout)
    experts = [
        SchemaAAdapter(
            "model_a", a_root, layout,
            PrecipSpec("tp", axis="daily", kind="accum", units="mm"),
        ),
        SchemaBAdapter(
            "model_b", b_root, layout,
            PrecipSpec(
                "total_precipitation_24hr", axis="daily", kind="accum", units="mm"
            ),
        ),
    ]
    truth = ImergTruth(truth_root)
    return experts, truth, layout, stats


def _dataset(env, **kwargs):
    experts, truth, layout, stats = env
    kw = dict(years=(2001, 2001), init_months=(6,), lead_days=(8, 9))
    kw.update(kwargs)
    return HindcastMixtureDataset(experts, truth, layout, stats, **kw)


def _find_pair(ds, init_key, tau):
    for i, row in enumerate(ds.pairs):
        if ds.index.init_keys[int(row["init_row"])] == init_key and int(
            row["tau"]
        ) == tau:
            return i
    raise AssertionError(f"no pair for {init_key} tau={tau}")


def test_shapes_dtypes_and_metadata(env):
    ds = _dataset(env)
    assert ds.expert_names == ["model_a", "model_b"]
    assert ds.channel_names == ["precip_mm_day", "geopotential/500", "2m_temperature"]
    assert ds.channel_masks.shape == (2, 3)
    assert ds.channel_masks.all()
    sample = ds[0]
    assert sample["expert_inputs"].shape == (2, 3, 8, 8)
    assert sample["expert_inputs"].dtype == torch.float32
    assert sample["expert_mask"].shape == (2,)
    assert sample["target"].shape == (1, 8, 8)
    assert sample["target_mm"].shape == (1, 8, 8)
    for key in ("lead_days", "init_time", "valid_time", "pair_idx"):
        assert sample[key].dtype == torch.int64


def test_values_and_normalization(env):
    ds = _dataset(env)
    # Shared init 6/1, tau 8: both experts live.
    i = _find_pair(ds, (2001, 6, 1, 0), 8)
    s = ds[i]
    assert s["expert_mask"].tolist() == [1.0, 1.0]
    x = s["expert_inputs"].numpy()
    # model_a precip: coded tp (mm) day 8, init 0, normalized by IMERG stats.
    raw = coded_value(2, 0, 8)
    np.testing.assert_allclose(
        x[0, 0], (raw - PRECIP_STATS[0]) / PRECIP_STATS[1], rtol=1e-5
    )
    # model_a z/500 at hour 192, ERA5 stats.
    raw = coded_value(1, 0, 192)
    np.testing.assert_allclose(
        x[0, 1], (raw - Z500_STATS[0]) / Z500_STATS[1], rtol=1e-5
    )
    # model_b geopotential@500 = coded + level_index(1).
    raw = coded_value(52, 0, 8) + 1
    np.testing.assert_allclose(
        x[1, 1], (raw - Z500_STATS[0]) / Z500_STATS[1], rtol=1e-5
    )
    # Target: IMERG day = init(6/1) + tau-1 = June 8 -> dayofyr 159 + 0.5.
    expected_mm = 159 + 0.5
    np.testing.assert_allclose(s["target_mm"].numpy(), expected_mm, rtol=1e-6)
    np.testing.assert_allclose(
        s["target"].numpy(),
        (expected_mm - PRECIP_STATS[0]) / PRECIP_STATS[1],
        rtol=1e-5,
    )
    # Round-trip: target == (target_mm - mean) / std with dataset scalars.
    np.testing.assert_allclose(
        s["target"].numpy(),
        (s["target_mm"].numpy() - ds.precip_mean) / ds.precip_std,
        rtol=1e-6,
    )
    # Times: init 2001-06-01 00Z in hours since 1900-01-01.
    assert (s["valid_time"] - s["init_time"]).item() == 8 * 24
    assert s["lead_days"].item() == 8


def test_masked_expert_slab_is_zero(env):
    ds = _dataset(env)
    # Init 6/4 exists only in model_a; 6/5 and 6/9 only in model_b.
    i = _find_pair(ds, (2001, 6, 4, 0), 8)
    s = ds[i]
    assert s["expert_mask"].tolist() == [1.0, 0.0]
    assert s["expert_inputs"][1].abs().max().item() == 0.0
    assert s["expert_inputs"][0].abs().max().item() > 0.0


def test_determinism(env):
    ds = _dataset(env)
    a = ds[3]
    b = ds[3]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_collate_shapes(env):
    ds = _dataset(env)
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch["expert_inputs"].shape == (2, 2, 3, 8, 8)
    assert batch["expert_mask"].shape == (2, 2)
    assert batch["target"].shape == (2, 1, 8, 8)
    assert batch["lead_days"].shape == (2,)


def test_nan_demotion(env, tmp_path):
    experts, truth, layout, stats = env
    # Poison model_a's z_500 AND tp for init 0 with NaN everywhere.
    store = tmp_path / "experts" / "model_a" / "2001.zarr"
    grp = zarr.open_group(str(store), mode="r+")
    z = grp["z_500"]
    z[0] = np.full(z.shape[1:], np.nan, dtype="float32")
    t2 = grp["2t"]
    t2[0] = np.full(t2.shape[1:], np.nan, dtype="float32")
    zarr.consolidate_metadata(str(store))
    ds = _dataset(env)
    i = _find_pair(ds, (2001, 6, 1, 0), 8)
    s = ds[i]
    # 2/3 of supplied values are NaN -> finite fraction ~1/3 < 0.5 -> demoted.
    assert s["expert_mask"].tolist() == [0.0, 1.0]
    assert s["expert_inputs"][0].abs().max().item() == 0.0
    assert torch.isfinite(s["expert_inputs"]).all()


def test_unsupplied_channel_stays_zero_after_normalize(env, tmp_path):
    experts, truth, layout, stats = env
    # model_b lacking 2m_temperature: rebuild with an excluded variable.
    b = SchemaBAdapter(
        "model_b",
        tmp_path / "experts" / "model_b",
        layout,
        PrecipSpec(
            "total_precipitation_24hr", axis="daily", kind="accum", units="mm"
        ),
        exclude_variables=("2m_temperature",),
    )
    ds = HindcastMixtureDataset(
        [experts[0], b], truth, layout, stats,
        years=(2001, 2001), init_months=(6,), lead_days=(8, 9),
    )
    i = _find_pair(ds, (2001, 6, 1, 0), 8)
    s = ds[i]
    # Channel 2 (2m_temperature) unsupplied by model_b: exactly 0 in z-space
    # even though (0 - mean)/std != 0.
    assert s["expert_inputs"][1, 2].abs().max().item() == 0.0
    assert s["expert_inputs"][1, 0].abs().max().item() > 0.0
    assert not ds.channel_masks[1, 2]


def test_min_experts_all(env):
    ds = _dataset(env, min_experts="all")
    # Only the shared init 6/1 survives.
    keys = {ds.index.init_keys[int(r["init_row"])] for r in ds.pairs}
    assert keys == {(2001, 6, 1, 0)}
    for i in range(len(ds)):
        assert ds[i]["expert_mask"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("num_workers,ctx", [(2, "fork"), (2, "spawn")])
def test_dataloader_workers(env, num_workers, ctx):
    import sys

    if sys.platform == "darwin" and ctx == "fork":
        pytest.skip("fork start method is unreliable on macOS")
    ds = _dataset(env)
    loader = DataLoader(
        ds,
        batch_size=2,
        num_workers=num_workers,
        multiprocessing_context=ctx,
        persistent_workers=True,
    )
    batches = list(loader)
    assert sum(b["expert_inputs"].shape[0] for b in batches) == len(ds)
    # Same data as single-process reads (order is sequential without sampler).
    s0 = ds[0]
    torch.testing.assert_close(batches[0]["expert_inputs"][0], s0["expert_inputs"])
