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

"""Tests for the validation driver (validation.py)."""

from __future__ import annotations

import cftime
import numpy as np
import torch
import xarray as xr

from datapipes.testing import GRID_LAT, GRID_LON
from losses import RegionalPrecipMSE, denormalize_precip, normalize_precip, region_weights
from seeps import SeepsClimatology
from validation import MixtureValidator

BOX = (-90.0, 90.0, 0.0, 360.0)
H, W = GRID_LAT.size, GRID_LON.size


def _clim(path):
    xr.Dataset(
        {
            "p1": (("month", "lat", "lon"), np.full((12, H, W), 0.5, "f4")),
            "t2": (("month", "lat", "lon"), np.full((12, H, W), 5.0, "f4")),
            "clim_mean": (("month", "lat", "lon"), np.full((12, H, W), 3.0, "f4")),
        },
        coords={"month": np.arange(1, 13), "lat": GRID_LAT, "lon": GRID_LON},
        attrs={"dry_threshold_mm": 0.25},
    ).to_zarr(path, mode="w", zarr_format=3, consolidated=True)
    return path


class _PickExpertZero(torch.nn.Module):
    """Stub gate: all weight on expert 0, zero bias."""

    def forward(self, x, mask, t):
        b, e = x.shape[0], x.shape[1]
        w = torch.zeros(b, e, x.shape[-2], x.shape[-1])
        w[:, 0] = 1.0
        return w, torch.zeros_like(w)


def _batch(target, offsets, tau=8, month=7):
    """One batch: expert i's precip = target + offsets[i] (mm/day)."""
    n = target.shape[0]
    e = len(offsets)
    x = torch.stack([target + o for o in offsets], dim=1).unsqueeze(2)
    hours = int(
        (
            cftime.DatetimeGregorian(2021, month, 15)
            - cftime.DatetimeGregorian(1900, 1, 1)
        ).total_seconds()
        // 3600
    )
    return {
        "expert_inputs": x,
        "expert_mask": torch.ones(n, e),
        "target": target.unsqueeze(1),
        "target_mm": target.unsqueeze(1),
        "lead_days": torch.full((n,), tau, dtype=torch.long),
        "valid_time": torch.full((n,), hours, dtype=torch.long),
    }


def _validator(tmp_path, loss_fn=None):
    return MixtureValidator(
        expert_names=["e0", "e1"],
        lead_days=(8, 9),
        region_weights=region_weights(GRID_LAT, GRID_LON, BOX),
        seeps_climatology=SeepsClimatology(_clim(tmp_path / "clim.zarr")),
        precip_mean=0.0,
        precip_std=1.0,
        precip_transform=None,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
    )


def test_validation_loss_matches_training_criterion(tmp_path):
    """`loss` is the gate's training criterion on the val split; a perfect
    gate scores 0 and each baseline gets its own comparable number."""
    loss_fn = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, space="normalized")
    v = _validator(tmp_path, loss_fn=loss_fn)
    target = torch.rand(3, H, W) * 8.0
    # expert 0 is perfect; expert 1 is +4 mm/day everywhere.
    metrics, _ = v.run(_PickExpertZero(), [_batch(target, [0.0, 4.0])])

    assert metrics["loss"] == metrics["gate/loss"]
    assert metrics["gate/loss"] < 1e-10          # gate copies the perfect expert
    assert metrics["e0/loss"] < 1e-10
    np.testing.assert_allclose(metrics["e1/loss"], 16.0, rtol=1e-5)
    # equal-weight is the mm/day mean of (target, target+4) => +2 everywhere.
    np.testing.assert_allclose(metrics["equal_weight/loss"], 4.0, rtol=1e-5)
    # The loss agrees with the RMSE of the same forecast (MSE = RMSE^2 here).
    np.testing.assert_allclose(
        metrics["e1/loss"], metrics["e1/rmse_lead8"] ** 2, rtol=1e-4
    )


def test_no_loss_keys_without_loss_fn(tmp_path):
    v = _validator(tmp_path, loss_fn=None)
    target = torch.rand(2, H, W) * 5.0
    metrics, _ = v.run(_PickExpertZero(), [_batch(target, [0.0, 1.0])])
    assert not [k for k in metrics if k.endswith("loss")]
    assert "gate/rmse_lead8" in metrics


def test_monthly_keys_cover_all_four_scores(tmp_path):
    v = _validator(tmp_path)
    target = torch.rand(2, H, W) * 5.0
    metrics, _ = v.run(_PickExpertZero(), [_batch(target, [0.0, 1.0], month=7)])
    for name in ("rmse", "bias", "acc", "seeps"):
        assert f"gate/imd_{name}_07" in metrics, name
        assert f"gate/imd_{name}_mean" in metrics, name
    # Months with no samples are not emitted at all.
    assert "gate/imd_rmse_01" not in metrics


def test_normalize_precip_round_trips():
    from datapipes.precip import LogPrecipTransform

    tr = LogPrecipTransform(epsilon=1e-3, units="m")
    mm = torch.tensor([0.0, 0.5, 7.0, 120.0])
    norm = normalize_precip(mm, mean=-6.379, std=0.858, transform=tr)
    back = denormalize_precip(norm, mean=-6.379, std=0.858, transform=tr)
    torch.testing.assert_close(back, mm, rtol=1e-4, atol=1e-4)
