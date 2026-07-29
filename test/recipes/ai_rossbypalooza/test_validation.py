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


def test_physical_mixing_is_arithmetic_log_mixing_is_geometric():
    """The point of model.mix_space: combining in mm/day gives the arithmetic
    expert mean, combining the log channels gives the (drier) geometric one."""
    from datapipes.precip import LogPrecipTransform
    from mowe_precip import mix

    tr = LogPrecipTransform(epsilon=1e-3, units="m")
    mu, sd = -6.379, 0.858
    # Two experts that disagree strongly, as they do for heavy monsoon rain.
    p_mm = torch.tensor([[[[2.0]], [[50.0]]]]).squeeze(-1)  # (1, 2, 1)
    z = normalize_precip(p_mm, mean=mu, std=sd, transform=tr)
    w = torch.full_like(p_mm, 0.5)
    b = torch.zeros_like(p_mm)

    phys = mix(w, b, p_mm)
    logmix = denormalize_precip(mix(w, b, z), mean=mu, std=sd, transform=tr)

    arithmetic = 26.0
    geometric = ((2.0 + 1.0) * (50.0 + 1.0)) ** 0.5 - 1.0  # eps = 1e-3 m = 1 mm
    torch.testing.assert_close(phys.squeeze(), torch.tensor(arithmetic))
    torch.testing.assert_close(
        logmix.squeeze(), torch.tensor(geometric), rtol=1e-3, atol=1e-2
    )
    assert float(logmix) < float(phys)          # the structural dry bias
    assert float(phys) / float(logmix) > 2.0    # and it is large


def test_loss_pred_space_physical_transforms_before_mse():
    """With pred_space=physical the loss log-transforms the mm/day mixture,
    so a perfect physical forecast scores 0 and the error is log-space."""
    from datapipes.precip import LogPrecipTransform

    tr = LogPrecipTransform(epsilon=1e-3, units="m")
    mu, sd = -6.379, 0.858
    loss = RegionalPrecipMSE(
        GRID_LAT, GRID_LON, BOX, space="normalized", pred_space="physical",
        precip_mean=mu, precip_std=sd, precip_transform=tr,
    )
    t_mm = torch.rand(2, H, W) * 20.0
    t_norm = normalize_precip(t_mm, mean=mu, std=sd, transform=tr)
    assert float(loss(t_mm, t_norm, t_mm)) < 1e-10
    # A 2x-too-wet forecast: error is the log ratio, not the mm/day gap.
    got = float(loss(2.0 * t_mm, t_norm, t_mm))
    expect = float(
        (
            (
                normalize_precip(2.0 * t_mm, mean=mu, std=sd, transform=tr)
                - t_norm
            )
            ** 2
        ).mean()
    )
    np.testing.assert_allclose(got, expect, rtol=1e-4)
    # Negative rain is clipped rather than producing NaN.
    assert torch.isfinite(loss(-1.0 * t_mm, t_norm, t_mm))


def test_composite_loss_adds_physical_bias_penalty():
    """bias_weight adds lambda * (regional mean error in mm/day)^2 on top of
    the log-space MSE, and is inert at bias_weight=0."""
    from datapipes.precip import LogPrecipTransform

    tr = LogPrecipTransform(epsilon=1e-3, units="m")
    mu, sd, lam = -6.379, 0.858, 0.02
    kw = dict(
        space="normalized", pred_space="physical",
        precip_mean=mu, precip_std=sd, precip_transform=tr,
    )
    plain = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, bias_weight=0.0, **kw)
    comp = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, bias_weight=lam, **kw)

    t_mm = torch.rand(3, H, W) * 15.0 + 6.0   # stays >4 so the >=0 clamp is inert
    t_norm = normalize_precip(t_mm, mean=mu, std=sd, transform=tr)
    pred_mm = t_mm - 4.0                      # uniformly 4 mm/day too dry

    base = float(plain(pred_mm, t_norm, t_mm))
    total = float(comp(pred_mm, t_norm, t_mm))
    np.testing.assert_allclose(total, base + lam * 16.0, rtol=1e-4)
    np.testing.assert_allclose(comp.last_bias_mm, -4.0, rtol=1e-4)
    np.testing.assert_allclose(comp.last_mse, base, rtol=1e-6)

    # A perfect forecast incurs no penalty; the penalty is bias, not spread.
    assert float(comp(t_mm, t_norm, t_mm)) < 1e-9
    # Equal-and-opposite errors cancel in the bias term but not in the MSE.
    offset = torch.zeros_like(t_mm)
    offset[:, : H // 2] = 3.0
    offset[:, H // 2 :] = -3.0
    unbiased = comp(t_mm + offset, t_norm, t_mm)
    assert abs(comp.last_bias_mm) < 0.5           # cos-lat weights, not exact 0
    assert float(unbiased) > 1e-3                  # MSE still sees the error


def test_composite_loss_rejects_negative_weight():
    import pytest

    with pytest.raises(ValueError, match="bias_weight"):
        RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, bias_weight=-1.0)
