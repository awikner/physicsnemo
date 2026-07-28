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

"""Tests for the regional losses (losses.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from losses import (
    RegionalPrecipLogMSE,
    RegionalPrecipMSE,
    build_loss,
    region_mask,
    region_weights,
)

LAT = np.linspace(35.0, -35.0, 15)  # 5-deg grid, N->S
LON = np.arange(0.0, 360.0, 5.0)
BOX = (5.0, 35.0, 60.0, 100.0)  # the monsoon domain


def test_region_mask_matches_hand_computed_box():
    m = region_mask(LAT, LON, BOX)
    assert m.shape == (15, 72)
    la_idx = np.nonzero((LAT >= 5) & (LAT <= 35))[0]
    lo_idx = np.nonzero((LON >= 60) & (LON <= 100))[0]
    expected = torch.zeros(15, 72, dtype=torch.bool)
    expected[np.ix_(la_idx, lo_idx)] = True
    assert torch.equal(m, expected)


def test_region_mask_wraparound_and_empty():
    m = region_mask(LAT, LON, (0, 35, 350, 10))
    assert m[3, 0] and m[3, 70] and not m[3, 30]
    with pytest.raises(ValueError, match="selects no gridpoints"):
        region_mask(LAT, LON, (36, 37, 0, 360))


def test_region_weights_cos_lat():
    w = region_weights(LAT, LON, BOX)
    m = region_mask(LAT, LON, BOX)
    assert (w[~m] == 0).all()
    # Rows closer to the equator get larger weights.
    inbox = w[m].reshape(-1)
    assert inbox.min() > 0
    row5 = np.argmin(np.abs(LAT - 5.0))
    row35 = np.argmin(np.abs(LAT - 35.0))
    col = np.argmin(np.abs(LON - 80.0))
    assert w[row5, col] > w[row35, col]
    flat = region_weights(LAT, LON, BOX, lat_weighted=False)
    assert (flat[m] == 1.0).all()


def test_regional_mse_normalized_exact_value():
    loss = RegionalPrecipMSE(LAT, LON, BOX, space="normalized")
    target = torch.zeros(2, 1, 15, 72)
    pred = torch.zeros(2, 15, 72)
    pred[:, :, :] = 2.0  # error 2 everywhere -> weighted mean of 4 = 4
    out = loss(pred, target, target)
    torch.testing.assert_close(out, torch.tensor(4.0))
    # Error outside the box must not contribute.
    pred2 = torch.zeros(2, 15, 72)
    pred2[:, 0, 0] = 100.0  # (35N... actually row 0 = 35N, lon 0: outside box)
    m = region_mask(LAT, LON, BOX)
    assert not m[0, 0]
    torch.testing.assert_close(
        loss(pred2, target, target), torch.tensor(0.0)
    )


def test_regional_mse_physical_space_scaling():
    mean, std = 5.0, 10.0
    loss_n = RegionalPrecipMSE(LAT, LON, BOX, space="normalized")
    loss_p = RegionalPrecipMSE(
        LAT, LON, BOX, space="physical", precip_mean=mean, precip_std=std
    )
    torch.manual_seed(0)
    target_mm = torch.rand(2, 15, 72) * 20
    target_norm = (target_mm - mean) / std
    pred_norm = target_norm + torch.randn(2, 15, 72) * 0.1
    a = loss_n(pred_norm, target_norm, target_mm)
    b = loss_p(pred_norm, target_norm, target_mm)
    torch.testing.assert_close(b, a * std**2)  # physical = normalized * std^2


def test_nan_target_cells_excluded():
    loss = RegionalPrecipMSE(LAT, LON, BOX)
    target = torch.zeros(1, 15, 72)
    m = region_mask(LAT, LON, BOX)
    rows, cols = torch.nonzero(m, as_tuple=True)
    target[0, rows[0], cols[0]] = torch.nan
    pred = torch.full((1, 15, 72), 3.0)
    out = loss(pred, target, target)
    assert torch.isfinite(out)
    torch.testing.assert_close(out, torch.tensor(9.0))


def test_log_mse_clamps_negative_pred_and_matches_hand_value():
    mean, std, eps = 5.0, 10.0, 0.1
    loss = RegionalPrecipLogMSE(
        LAT, LON, BOX, precip_mean=mean, precip_std=std, epsilon_mm=eps
    )
    target_mm = torch.full((1, 15, 72), 10.0)
    target_norm = (target_mm - mean) / std
    # Prediction of -20 mm (normalized (-20-5)/10) clamps to 0 mm.
    pred_norm = torch.full((1, 15, 72), (-20.0 - mean) / std)
    out = loss(pred_norm, target_norm, target_mm)
    expected = float(np.log1p(10.0 / eps)) ** 2
    torch.testing.assert_close(out, torch.tensor(expected), rtol=1e-5, atol=0)
    # Perfect forecast -> 0.
    perfect = (target_mm - mean) / std
    torch.testing.assert_close(
        loss(perfect, target_norm, target_mm), torch.tensor(0.0)
    )


def test_build_loss_dispatcher():
    kw = dict(lat=LAT, lon=LON, box=BOX, precip_mean=5.0, precip_std=10.0)
    assert isinstance(build_loss({"name": "regional_mse"}, **kw), RegionalPrecipMSE)
    log = build_loss({"name": "regional_log_mse", "epsilon_mm": 0.5}, **kw)
    assert isinstance(log, RegionalPrecipLogMSE)
    assert log.epsilon_mm == 0.5
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss({"name": "huber"}, **kw)


def test_physical_space_with_log_transform():
    from datapipes.precip import LogPrecipTransform

    t = LogPrecipTransform(epsilon=1e-3, units="m")
    mean, std = -6.0, 1.5
    loss = RegionalPrecipMSE(
        LAT, LON, BOX, space="physical",
        precip_mean=mean, precip_std=std, precip_transform=t,
    )
    target_mm = torch.full((1, 15, 72), 10.0)
    # Normalized prediction that decodes exactly to 4 mm/day.
    pred_norm = torch.full(
        (1, 15, 72), (float(np.log(1e-3 + 0.004)) - mean) / std
    )
    out = loss(pred_norm, torch.zeros_like(target_mm), target_mm)
    torch.testing.assert_close(out, torch.tensor(36.0), rtol=1e-4, atol=1e-3)
