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
import pytest
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


def test_acc_uses_daily_climatology_not_monthly(tmp_path):
    """ACC anomalies must reference the day-of-year climatology. A monthly
    12-step reference leaves the within-month seasonal signal in both the
    forecast and observed anomalies, which inflates the correlation."""
    from validation import StreamingMonthlyScores

    # Monthly clim = 3 everywhere; daily clim ramps 1..366 so the two differ.
    path = _clim(tmp_path / "clim_d.zarr")
    ds = xr.open_zarr(path).load()
    daily = np.tile(
        np.linspace(1.0, 10.0, 366, dtype="f4")[:, None, None], (1, H, W)
    )
    ds["clim_mean_daily"] = (("dayofyear", "lat", "lon"), daily)
    ds = ds.assign_coords(dayofyear=np.arange(1, 367, dtype="int32"))
    ds.to_zarr(tmp_path / "clim_d2.zarr", mode="w", zarr_format=3, consolidated=True)
    clim = SeepsClimatology(tmp_path / "clim_d2.zarr")
    assert clim.clim_mean_daily is not None

    doy = 196  # mid-July
    ref_daily = float(np.linspace(1.0, 10.0, 366)[doy - 1])
    obs = torch.full((2, H, W), ref_daily + 4.0)   # +4 anomaly vs daily clim
    pred = torch.full((2, H, W), ref_daily + 2.0)  # +2 anomaly, same sign

    m = StreamingMonthlyScores(
        bins={7: 0}, climatology=clim,
        region_weights=torch.ones(H, W), device=torch.device("cpu"),
    )
    m.update(0, pred, obs, torch.tensor([7, 7]), torch.tensor([doy, doy]))
    with_daily = float(m.finalize()["acc"][0])

    # Same fields scored against the monthly reference (doys omitted).
    m2 = StreamingMonthlyScores(
        bins={7: 0}, climatology=clim,
        region_weights=torch.ones(H, W), device=torch.device("cpu"),
    )
    m2.clim_daily = None
    m2.update(0, pred, obs, torch.tensor([7, 7]))
    with_monthly = float(m2.finalize()["acc"][0])

    # Both are +1 here (anomalies are co-signed), but the references differ,
    # so the two paths are genuinely distinct code.
    assert with_daily == pytest.approx(1.0, abs=1e-5)
    assert with_monthly == pytest.approx(1.0, abs=1e-5)
    # Now make the daily reference matter: an obs anomaly that is POSITIVE
    # against the monthly clim but NEGATIVE against the daily one.
    obs2 = torch.full((2, H, W), 4.0)      # monthly clim 3 -> +1 ; daily 5.8 -> -1.8
    pred2 = torch.full((2, H, W), 8.0)     # monthly +5     ; daily +2.2
    m3 = StreamingMonthlyScores(
        bins={7: 0}, climatology=clim,
        region_weights=torch.ones(H, W), device=torch.device("cpu"),
    )
    m3.update(0, pred2, obs2, torch.tensor([7, 7]), torch.tensor([doy, doy]))
    acc_daily = float(m3.finalize()["acc"][0])
    m4 = StreamingMonthlyScores(
        bins={7: 0}, climatology=clim,
        region_weights=torch.ones(H, W), device=torch.device("cpu"),
    )
    m4.clim_daily = None
    m4.update(0, pred2, obs2, torch.tensor([7, 7]))
    acc_monthly = float(m4.finalize()["acc"][0])
    assert acc_daily < 0 < acc_monthly, (acc_daily, acc_monthly)


def test_doy_from_hours_matches_cftime():
    import cftime as _cf

    from seeps import doy_from_hours_since_1900

    epoch = _cf.DatetimeGregorian(1900, 1, 1)
    cases = [(2021, 1, 1, 1), (2021, 7, 15, 196), (2020, 12, 31, 366)]
    hs = [
        int((_cf.DatetimeGregorian(y, m, d) - epoch).total_seconds() // 3600)
        for (y, m, d, _) in cases
    ]
    got = doy_from_hours_since_1900(torch.tensor(hs)).tolist()
    assert got == [c[3] for c in cases], got


def test_physical_space_mse_is_unbiased_and_scaled():
    """space=physical takes the error in mm/day (so squared error elicits the
    arithmetic mean, no geometric shortfall) and scale_mm only rescales."""
    from datapipes.precip import LogPrecipTransform

    tr = LogPrecipTransform(epsilon=1e-3, units="m")
    mu, sd = -6.379, 0.858
    kw = dict(
        space="physical", pred_space="physical",
        precip_mean=mu, precip_std=sd, precip_transform=tr,
    )
    unscaled = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, **kw)
    scaled = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, scale_mm=9.3, **kw)

    t_mm = torch.rand(2, H, W) * 20.0
    t_norm = normalize_precip(t_mm, mean=mu, std=sd, transform=tr)
    pred = t_mm + 3.0                      # uniformly 3 mm/day too wet

    # Error is in mm/day, not log space: MSE is exactly 9.
    np.testing.assert_allclose(float(unscaled(pred, t_norm, t_mm)), 9.0, rtol=1e-5)
    np.testing.assert_allclose(
        float(scaled(pred, t_norm, t_mm)), 9.0 / 9.3**2, rtol=1e-5
    )
    # A perfect forecast scores zero either way.
    assert float(scaled(t_mm, t_norm, t_mm)) < 1e-12
    # Unlike the log-space loss, the minimiser is the arithmetic mean: for a
    # two-outcome target the optimal constant prediction is their mean.
    obs = torch.cat([torch.full((1, H, W), 2.0), torch.full((1, H, W), 50.0)])
    tn = normalize_precip(obs, mean=mu, std=sd, transform=tr)
    losses = {
        c: float(unscaled(torch.full_like(obs, c), tn, obs))
        for c in (11.4, 26.0, 40.0)     # geometric mean, arithmetic mean, high
    }
    assert min(losses, key=losses.get) == 26.0, losses


def test_scale_mm_rejects_non_positive():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="scale_mm"):
        RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, space="physical", scale_mm=0.0)


def test_amplitude_ratio_diagnostic(tmp_path):
    """`amp` is sigma_pred/sigma_obs on anomalies -- the shrinkage diagnostic.
    MSE is minimised at amp = ACC, so amp << 1 means the loss is hedging."""
    from validation import StreamingMonthlyScores

    clim = SeepsClimatology(_clim(tmp_path / "clim_amp.zarr"))
    clim.clim_mean = torch.full((12, H, W), 3.0)
    clim.clim_mean_daily = None      # exercise the monthly fallback path

    torch.manual_seed(0)
    obs_anom = torch.randn(4, H, W) * 5.0
    obs = 3.0 + obs_anom

    for factor in (1.0, 0.5, 0.29):
        m = StreamingMonthlyScores(
            bins={7: 0}, climatology=clim,
            region_weights=torch.ones(H, W), device=torch.device("cpu"),
        )
        m.update(0, 3.0 + factor * obs_anom, obs, torch.tensor([7] * 4))
        out = m.finalize()
        # Perfectly correlated but shrunk: ACC stays 1, amp reports the shrink.
        np.testing.assert_allclose(float(out["amp"][0]), factor, rtol=1e-4)
        np.testing.assert_allclose(float(out["acc"][0]), 1.0, rtol=1e-4)
    # Emitted for every month and as a mean, alongside the other scores.
    v = _validator(tmp_path)
    target = torch.rand(2, H, W) * 5.0
    metrics, _ = v.run(_PickExpertZero(), [_batch(target, [0.0, 1.0], month=7)])
    assert "gate/imd_amp_07" in metrics and "gate/imd_amp_mean" in metrics
    assert "equal_weight/imd_amp_mean" in metrics


def test_variance_matching_term_penalises_shrinkage():
    """var_weight adds (sigma_pred/sigma_obs - 1)^2 on spatial std, so an
    over-smoothed forecast is penalised even when its MSE is lower."""
    from datapipes.precip import LogPrecipTransform

    tr = LogPrecipTransform(epsilon=1e-3, units="m")
    mu, sd = -6.379, 0.858
    kw = dict(
        space="physical", pred_space="physical", scale_mm=9.3,
        precip_mean=mu, precip_std=sd, precip_transform=tr,
    )
    plain = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, **kw)
    matched = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, var_weight=1.0, **kw)

    torch.manual_seed(0)
    obs = (torch.rand(3, H, W) * 30.0).clamp(min=0.0)
    t_norm = normalize_precip(obs, mean=mu, std=sd, transform=tr)
    mean_obs = obs.mean(dim=(-2, -1), keepdim=True)

    # Correct amplitude costs nothing extra; shrunk amplitude does.
    exact = matched(obs, t_norm, obs)
    np.testing.assert_allclose(float(matched.last_amp), 1.0, rtol=1e-3)
    assert float(exact) < 1e-6

    shrunk = mean_obs + 0.3 * (obs - mean_obs)      # 30% of observed contrast
    np.testing.assert_allclose(
        float(matched(shrunk, t_norm, obs)) - float(plain(shrunk, t_norm, obs)),
        (0.3 - 1.0) ** 2,
        rtol=1e-3,
    )
    np.testing.assert_allclose(float(matched.last_amp), 0.3, rtol=1e-3)

    # The decisive case. Build a rival with FULL amplitude but correlation
    # r = 0.6: pred = mean + 0.6*a + 0.8*n with std(n) = std(a), so
    # std(pred - mean) = std(a) exactly and corr(pred - mean, a) = 0.6.
    # Plain MSE prefers the shrunk forecast (hedging wins whenever r < 0.5 in
    # the decomposition, and 2*var*(1-r) here exceeds (1-c)^2*var); adding the
    # amplitude term flips the ranking, which is the whole point.
    a = obs - mean_obs
    n = torch.randn_like(a)
    n = n * (a.std() / n.std())
    full_amp = mean_obs + 0.6 * a + 0.8 * n

    assert float(plain(shrunk, t_norm, obs)) < float(plain(full_amp, t_norm, obs))
    assert float(matched(shrunk, t_norm, obs)) > float(matched(full_amp, t_norm, obs))
    # And the rival is near-full-amplitude, so it pays little penalty. (Not
    # exactly 1.0: the noise is scaled by unweighted std while the loss uses
    # cos-lat weights, and cov(a, n) is only approximately zero at this size.)
    matched(full_amp, t_norm, obs)
    assert 0.85 < float(matched.last_amp) < 1.10, float(matched.last_amp)


def test_var_weight_rejects_negative():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="var_weight"):
        RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX, var_weight=-0.5)


def test_threshold_scores_detect_intensity_compression():
    """exc_bias at a threshold is P(pred>T)/P(obs>T): the direct read on
    whether the forecast produces enough heavy rain."""
    from validation import StreamingThresholdScores

    thr = [1.0, 10.0, 20.0]
    acc = StreamingThresholdScores(
        thresholds=thr, region_weights=torch.ones(H, W),
        device=torch.device("cpu"),
    )
    # Observed: half the domain at 30 mm/day, half at 0.
    obs = torch.zeros(1, H, W)
    obs[0, : H // 2] = 30.0
    # Forecast: same footprint but smoothed to 15 mm/day -- clears 10, not 20.
    pred = torch.zeros(1, H, W)
    pred[0, : H // 2] = 15.0
    acc.update(pred, obs)
    out = acc.finalize()

    # At 1 and 10 mm/day the footprint is right, so bias is 1 and CSI is 1.
    np.testing.assert_allclose(float(out["exc_bias"][0]), 1.0, rtol=1e-5)
    np.testing.assert_allclose(float(out["csi"][0]), 1.0, rtol=1e-5)
    np.testing.assert_allclose(float(out["exc_bias"][1]), 1.0, rtol=1e-5)
    # At 20 mm/day the smoothing has removed the events entirely.
    np.testing.assert_allclose(float(out["exc_bias"][2]), 0.0, atol=1e-9)
    np.testing.assert_allclose(float(out["csi"][2]), 0.0, atol=1e-9)


def test_threshold_scores_nan_when_threshold_unreached():
    from validation import StreamingThresholdScores

    acc = StreamingThresholdScores(
        thresholds=[5.0, 500.0], region_weights=torch.ones(H, W),
        device=torch.device("cpu"),
    )
    acc.update(torch.full((1, H, W), 8.0), torch.full((1, H, W), 9.0))
    out = acc.finalize()
    assert not torch.isnan(out["exc_bias"][0])
    assert torch.isnan(out["exc_bias"][1])      # nothing ever reaches 500


def test_validator_emits_threshold_keys(tmp_path):
    v = _validator(tmp_path)
    target = torch.rand(2, H, W) * 30.0
    metrics, _ = v.run(_PickExpertZero(), [_batch(target, [0.0, 2.0], month=7)])
    for key in ("gate/exc_bias_1mm", "gate/csi_10mm", "equal_weight/exc_bias_20mm"):
        assert key in metrics, key


# --------------------------------------------------------------------------- #
# Ensemble mode + new additive metrics
# --------------------------------------------------------------------------- #


class _NoisyGate(torch.nn.Module):
    """Stub noise-conditioned gate: all weight on expert 0, bias = 0.5 * z0
    per member — members differ deterministically given the noise."""

    def forward(self, x, mask, t, noise=None):
        assert noise is not None and noise.ndim == 3
        b, e = x.shape[0], x.shape[1]
        n = noise.shape[1]
        h, w = x.shape[-2], x.shape[-1]
        weights = torch.zeros(b, n, e, h, w)
        weights[:, :, 0] = 1.0
        biases = (
            noise[..., 0].view(b, n, 1, 1, 1).expand(b, n, e, h, w) * 0.5
        ).contiguous()
        return weights, biases


def _ens_validator(tmp_path, *, loss_fn=None, fss=False):
    from losses import RegionalAlmostFairCRPS

    if loss_fn is None:
        loss_fn = RegionalAlmostFairCRPS(
            GRID_LAT, GRID_LON, BOX, pred_space="physical"
        )
    kw = {}
    if fss:
        kw["fss_windows"] = [3]
        kw["fss_threshold_maps"] = torch.full((1, H, W), 1.0)
        kw["fss_threshold_labels"] = ["p90"]
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
        ens_size=3,
        noise_dim=4,
        noise_seed=7,
        **kw,
    )


def test_ensemble_validator_keys_and_determinism(tmp_path):
    torch.manual_seed(0)
    target = torch.rand(4, H, W) * 5 + 1
    batch = _batch(target, offsets=[0.4, -0.6])
    validator = _ens_validator(tmp_path, fss=True)
    model = _NoisyGate()
    m1, _ = validator.run(model, [batch])
    m2, _ = validator.run(model, [batch])
    # Seeded noise: probabilistic scores identical across passes/epochs.
    assert m1["gate/crps_mean"] == m2["gate/crps_mean"]
    assert m1["gate/crps_mean"] > 0
    for key in (
        "gate/crps_lead8",
        "gate/spread_skill_mean",
        "gate/rank_hist_dev",
        "expert_ensemble/crps_mean",
        "gate/fss_w3_p90",
        "gate/amp_band0",
        "equal_weight/amp_band0",
    ):
        assert key in m1, key
    # Rank histogram normalizes to 1 over ens_size + 1 bins.
    hist = [m1[f"gate/rank_hist_{i}"] for i in range(4)]
    assert sum(hist) == pytest.approx(1.0, abs=1e-6)
    # Pre-existing deterministic keys still emitted, fed by the ens mean.
    for key in ("gate/rmse_lead8", "equal_weight/rmse_lead8", "e0/rmse_lead8",
                "gate/csi_1mm", "loss"):
        assert key in m1, key


def test_ensemble_gate_loss_uses_full_ensemble(tmp_path):
    """The monitored `loss` equals the CRPS loss on the member stack, not on
    the ensemble mean (a collapsed read would hide all spread)."""
    from losses import RegionalAlmostFairCRPS

    torch.manual_seed(1)
    target = torch.rand(2, H, W) * 5 + 1
    batch = _batch(target, offsets=[0.0, 0.0])
    loss_fn = RegionalAlmostFairCRPS(GRID_LAT, GRID_LON, BOX, pred_space="physical")
    validator = _ens_validator(tmp_path, loss_fn=loss_fn)
    model = _NoisyGate()
    metrics, _ = validator.run(model, [batch])

    # Reproduce the members by hand from the seeded noise recipe.
    g = torch.Generator()
    g.manual_seed(7 * 1_000_003 + 0)
    noise = torch.randn(2, 3, 4, generator=g)
    members = (target.unsqueeze(1) + 0.5 * noise[..., 0:1, None]).clamp(min=0.0)
    expected = float(
        loss_fn(members, target.unsqueeze(1), target.unsqueeze(1))
    )
    assert metrics["loss"] == pytest.approx(expected, rel=1e-5)
    mean_read = float(
        loss_fn(members.mean(dim=1), target.unsqueeze(1), target.unsqueeze(1))
    )
    assert metrics["loss"] != pytest.approx(mean_read, rel=1e-3)


def test_deterministic_validator_unchanged_with_new_kwargs(tmp_path):
    """fss/amp kwargs are additive: the deterministic path still works and
    the FSS metric appears for every source."""
    loss_fn = RegionalPrecipMSE(GRID_LAT, GRID_LON, BOX)
    validator = MixtureValidator(
        expert_names=["e0", "e1"],
        lead_days=(8, 9),
        region_weights=region_weights(GRID_LAT, GRID_LON, BOX),
        seeps_climatology=SeepsClimatology(_clim(tmp_path / "clim.zarr")),
        precip_mean=0.0,
        precip_std=1.0,
        precip_transform=None,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
        fss_windows=[3],
        fss_threshold_maps=torch.full((1, H, W), 1.0),
        fss_threshold_labels=["p90"],
    )
    torch.manual_seed(2)
    target = torch.rand(2, H, W) * 5
    batch = _batch(target, offsets=[0.3, -0.3])
    metrics, _ = validator.run(_PickExpertZero(), [batch])
    for key in ("gate/fss_w3_p90", "e0/fss_w3_p90", "gate/amp_band0", "loss"):
        assert key in metrics, key
    assert "gate/crps_mean" not in metrics  # ensemble keys stay ensemble-only
