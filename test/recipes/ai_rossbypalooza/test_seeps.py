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

"""Tests for SEEPS (seeps.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr

from seeps import (
    P1_MAX,
    P1_MIN,
    SeepsClimatology,
    StreamingRegionalSEEPS,
    categorize,
    months_from_hours_since_1900,
    seeps_matrix,
    seeps_penalty,
)


def test_seeps_matrix_hand_values():
    p1 = torch.tensor(0.5)
    m = seeps_matrix(p1)
    # 1/2 * [[0, 2, 8], [2, 0, 6], [2 + 1.2, 1.2, 0]]
    expected = 0.5 * torch.tensor(
        [[0.0, 2.0, 8.0], [2.0, 0.0, 6.0], [2.0 + 3.0 / 2.5, 3.0 / 2.5, 0.0]]
    )
    torch.testing.assert_close(m, expected)
    assert m.diag().abs().max() == 0  # perfect forecast scores 0


def test_penalty_matches_matrix_lookup():
    torch.manual_seed(0)
    p1 = torch.rand(4, 6) * 0.7 + 0.15
    t2 = torch.full((4, 6), 5.0)
    pred = torch.rand(4, 6) * 12.0
    target = torch.rand(4, 6) * 12.0
    pen = seeps_penalty(pred, target, p1, t2, dry_threshold_mm=0.25)
    mat = seeps_matrix(p1)  # (4, 6, 3, 3)
    f = categorize(pred, t2, 0.25)
    o = categorize(target, t2, 0.25)
    expected = torch.gather(
        torch.gather(mat, -1, o[..., None, None].expand(4, 6, 3, 1)).squeeze(-1),
        -1,
        f[..., None],
    ).squeeze(-1)
    torch.testing.assert_close(pen, expected)


def test_categorize_boundaries():
    t2 = torch.tensor([5.0, 5.0, 5.0, 5.0])
    x = torch.tensor([0.0, 0.25, 3.0, 7.0])
    assert categorize(x, t2, 0.25).tolist() == [0, 0, 1, 2]


def test_perfect_forecast_scores_zero():
    p1 = torch.full((3, 4), 0.4)
    t2 = torch.full((3, 4), 4.0)
    x = torch.rand(3, 4) * 10
    assert seeps_penalty(x, x, p1, t2).abs().max() == 0


def _write_clim(path, h=4, w=6, p1_value=0.5, t2_value=5.0):
    ds = xr.Dataset(
        {
            "p1": (("month", "lat", "lon"), np.full((12, h, w), p1_value, "f4")),
            "t2": (("month", "lat", "lon"), np.full((12, h, w), t2_value, "f4")),
        },
        coords={
            "month": np.arange(1, 13),
            "lat": np.linspace(3.0, -3.0, h),
            "lon": np.arange(0.0, 360.0, 360.0 / w),
        },
        attrs={"dry_threshold_mm": 0.25},
    )
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True)
    return path


def test_streaming_seeps_end_to_end(tmp_path):
    clim = SeepsClimatology(_write_clim(tmp_path / "clim.zarr"))
    assert clim.p1.shape == (12, 4, 6)
    weights = torch.ones(4, 6)
    acc = StreamingRegionalSEEPS(
        n_leads=2,
        climatology=clim,
        region_weights=weights,
        device=torch.device("cpu"),
    )
    # Lead 0: perfect forecasts -> 0.
    target = torch.rand(3, 4, 6) * 10
    acc.update(0, target, target, months=torch.tensor([6, 7, 8]))
    # Lead 1: forecast dry (0), observe heavy (10 > t2=5):
    # penalty = 0.5 * 4/(1-p1) = 0.5*8 = 4 with p1=0.5.
    acc.update(
        1,
        torch.zeros(2, 4, 6),
        torch.full((2, 4, 6), 10.0),
        months=torch.tensor([6, 6]),
    )
    out = acc.finalize()
    torch.testing.assert_close(out[0], torch.tensor(0.0))
    torch.testing.assert_close(out[1], torch.tensor(4.0))


def test_streaming_seeps_excludes_invalid_p1_and_nan(tmp_path):
    # p1 = 0.95 > P1_MAX everywhere -> no valid gridpoints -> weight 0.
    clim = SeepsClimatology(
        _write_clim(tmp_path / "clim_bad.zarr", p1_value=0.95)
    )
    assert P1_MAX < 0.95
    acc = StreamingRegionalSEEPS(
        n_leads=1,
        climatology=clim,
        region_weights=torch.ones(4, 6),
        device=torch.device("cpu"),
    )
    acc.update(0, torch.zeros(1, 4, 6), torch.full((1, 4, 6), 10.0),
               months=torch.tensor([6]))
    assert acc.weight_sum[0] == 0
    # NaN target cells excluded.
    clim_ok = SeepsClimatology(_write_clim(tmp_path / "clim_ok.zarr"))
    acc2 = StreamingRegionalSEEPS(
        n_leads=1,
        climatology=clim_ok,
        region_weights=torch.ones(4, 6),
        device=torch.device("cpu"),
    )
    target = torch.full((1, 4, 6), 10.0)
    target[0, 0, 0] = torch.nan
    acc2.update(0, torch.zeros(1, 4, 6), target, months=torch.tensor([6]))
    assert acc2.weight_sum[0] == 23  # 24 cells - 1 NaN
    assert P1_MIN == 0.1


def test_months_from_hours():
    import cftime

    epoch = cftime.DatetimeGregorian(1900, 1, 1)
    hs = []
    for (y, m, d) in [(2001, 6, 9), (2001, 12, 31), (2004, 2, 29)]:
        hs.append(int((cftime.DatetimeGregorian(y, m, d) - epoch).total_seconds() // 3600))
    months = months_from_hours_since_1900(torch.tensor(hs))
    assert months.tolist() == [6, 12, 2]


def test_missing_climatology_names_generator(tmp_path):
    with pytest.raises(FileNotFoundError, match="compute_seeps_climatology"):
        SeepsClimatology(tmp_path / "nope.zarr")


def test_streaming_monthly_rmse_acc():
    from validation import StreamingMonthlyRmseAcc

    h, w = 4, 6
    bins = {(2021, 6): 0, (2021, 7): 1}
    clim = torch.zeros(12, h, w)
    clim[5] = 2.0  # June climatology = 2 mm/day
    acc_w = torch.ones(h, w)
    m = StreamingMonthlyRmseAcc(
        bins=bins, clim_mean=clim, region_weights=acc_w,
        device=torch.device("cpu"),
    )
    # June: pred anomalies exactly equal target anomalies -> ACC 1, RMSE 0.
    target = torch.rand(3, h, w) * 5
    months = torch.tensor([6, 6, 6])
    m.update(0, target.clone(), target, months)
    # July: pred = -target anomalies (clim 0) -> ACC -1; RMSE = 2*|target|.
    t2 = torch.rand(2, h, w) + 1.0
    m.update(1, -t2, t2, torch.tensor([7, 7]))
    rmse, acc, wt = m.finalize()
    torch.testing.assert_close(rmse[0], torch.tensor(0.0))
    torch.testing.assert_close(acc[0], torch.tensor(1.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(acc[1], torch.tensor(-1.0), atol=1e-5, rtol=0)
    expected_rmse = (4 * t2**2).mean().sqrt()
    torch.testing.assert_close(rmse[1], expected_rmse, rtol=1e-5, atol=1e-5)
    assert wt[0] > 0 and wt[1] > 0


def test_years_from_hours():
    import cftime

    from seeps import years_from_hours_since_1900

    epoch = cftime.DatetimeGregorian(1900, 1, 1)
    hs = [
        int((cftime.DatetimeGregorian(y, m, d) - epoch).total_seconds() // 3600)
        for (y, m, d) in [(2020, 3, 9), (2024, 12, 31), (2025, 1, 3)]
    ]
    assert years_from_hours_since_1900(torch.tensor(hs)).tolist() == [
        2020, 2024, 2025,
    ]
