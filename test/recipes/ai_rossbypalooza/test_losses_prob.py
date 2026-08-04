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

"""Tests for the probabilistic / spatial losses (losses.py):
almost-fair CRPS, neighborhood FSS, AMSE, gate-map TV, threshold loading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from losses import (
    RegionalAlmostFairCRPS,
    RegionalPrecipAMSE,
    RegionalPrecipFSS,
    RegionalPrecipMSE,
    almost_fair_crps,
    build_loss,
    gate_smoothness_penalty,
    load_precip_quantile_thresholds,
    region_mask,
    region_weights,
    resolve_fss_thresholds,
)

LAT = np.linspace(35.0, -35.0, 15)  # 5-deg grid, N->S
LON = np.arange(0.0, 360.0, 5.0)
BOX = (5.0, 35.0, 60.0, 100.0)  # the monsoon domain
BOX_ALL = (-90.0, 90.0, 0.0, 360.0)


def _fixed_maps(values, h=15, w=72):
    return torch.stack([torch.full((h, w), float(v)) for v in values])


# --------------------------------------------------------------------------- #
# almost-fair CRPS
# --------------------------------------------------------------------------- #


def test_afcrps_matches_kcrps_fair_and_biased():
    """physicsnemo's kcrps is the oracle: alpha=1 == fair, alpha=0 == biased."""
    pytest.importorskip("physicsnemo", reason="oracle needs physicsnemo")
    from physicsnemo.metrics.general.crps import kcrps

    torch.manual_seed(0)
    members = torch.randn(3, 4, 8, 8) * 2.0
    obs = torch.randn(3, 8, 8)
    fair = almost_fair_crps(members, obs, alpha=1.0)
    biased = almost_fair_crps(members, obs, alpha=0.0)
    torch.testing.assert_close(
        fair, kcrps(members, obs, dim=1, biased=False), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        biased, kcrps(members, obs, dim=1, biased=True), atol=1e-5, rtol=1e-5
    )
    # The 0.95 blend sits between the two.
    blend = almost_fair_crps(members, obs, alpha=0.95)
    assert ((blend >= fair - 1e-6) & (blend <= biased + 1e-6)).all()


def test_afcrps_fair_leq_biased_and_perfect_ensemble_zero():
    torch.manual_seed(1)
    members = torch.randn(2, 3, 6, 6)
    obs = torch.randn(2, 6, 6)
    fair = almost_fair_crps(members, obs, alpha=1.0)
    biased = almost_fair_crps(members, obs, alpha=0.0)
    assert (fair <= biased + 1e-6).all()
    # All members equal to the observation -> exactly 0.
    perfect = obs.unsqueeze(1).expand(2, 3, 6, 6).contiguous()
    torch.testing.assert_close(
        almost_fair_crps(perfect, obs, alpha=1.0), torch.zeros(2, 6, 6)
    )


def test_afcrps_validation_errors():
    with pytest.raises(ValueError, match="alpha"):
        almost_fair_crps(torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4), alpha=1.5)
    with pytest.raises(ValueError, match="ensemble dim"):
        almost_fair_crps(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))


def test_regional_crps_n1_reduces_to_regional_mae():
    loss = RegionalAlmostFairCRPS(LAT, LON, BOX, pred_space="physical")
    target_mm = torch.zeros(2, 15, 72)
    pred = torch.full((2, 15, 72), 3.0)
    out3 = loss(pred, target_mm, target_mm)
    torch.testing.assert_close(out3, torch.tensor(3.0))  # weighted MAE
    out4 = loss(pred.unsqueeze(1), target_mm, target_mm)  # (B, 1, H, W)
    torch.testing.assert_close(out4, out3)
    assert np.isnan(loss.last_spread)  # no spread at N=1


def test_regional_crps_nan_cells_excluded():
    loss = RegionalAlmostFairCRPS(LAT, LON, BOX, pred_space="physical")
    target = torch.zeros(1, 15, 72)
    m = region_mask(LAT, LON, BOX)
    rows, cols = torch.nonzero(m, as_tuple=True)
    target[0, rows[0], cols[0]] = torch.nan
    pred = torch.full((1, 2, 15, 72), 2.0)
    base = loss(pred, target, target)
    pred2 = pred.clone()
    pred2[0, :, rows[0], cols[0]] = 500.0  # only touches the NaN cell
    torch.testing.assert_close(loss(pred2, target, target), base)


def test_regional_crps_rewards_calibrated_spread():
    """Fair CRPS prefers a dispersed calibrated ensemble over a collapsed one."""
    torch.manual_seed(2)
    # Observation is +-1 per cell; a {-1, +1} 2-member ensemble is perfectly
    # calibrated, the collapsed ensemble at 0 is the MSE-style hedge.
    obs = torch.where(torch.rand(4, 15, 72) > 0.5, 1.0, -1.0)
    dispersed = torch.stack(
        [torch.full_like(obs, -1.0), torch.full_like(obs, 1.0)], dim=1
    )
    collapsed = torch.zeros_like(dispersed)
    loss = RegionalAlmostFairCRPS(LAT, LON, BOX_ALL, alpha=1.0, pred_space="physical")
    # clamp(min=0) would destroy the +-1 toy fields; shift into positive range.
    assert loss(dispersed + 5.0, obs + 5.0, obs + 5.0) < loss(
        collapsed + 5.0, obs + 5.0, obs + 5.0
    )


def test_regional_crps_scale_mm_is_linear():
    loss = RegionalAlmostFairCRPS(LAT, LON, BOX, pred_space="physical", scale_mm=4.0)
    ref = RegionalAlmostFairCRPS(LAT, LON, BOX, pred_space="physical")
    target = torch.zeros(1, 15, 72)
    pred = torch.full((1, 15, 72), 2.0)
    torch.testing.assert_close(loss(pred, target, target) * 4.0, ref(pred, target, target))


# --------------------------------------------------------------------------- #
# neighborhood FSS
# --------------------------------------------------------------------------- #


def _fss(windows=(3,), thresholds=(5.0,), anchor=None, fss_weight=1.0, **kw):
    anchor = anchor or RegionalPrecipMSE(
        LAT, LON, BOX, space="physical", pred_space="physical"
    )
    labels = [f"{t:g}mm" for t in thresholds]
    return RegionalPrecipFSS(
        LAT,
        LON,
        BOX,
        anchor=anchor,
        threshold_maps=_fixed_maps(thresholds),
        threshold_labels=labels,
        windows=windows,
        fss_weight=fss_weight,
        pred_space="physical",
        **kw,
    )


def test_fss_zero_at_perfect_alignment():
    loss = _fss()
    torch.manual_seed(3)
    target = torch.rand(2, 15, 72) * 20
    out = loss(target.clone(), target, target)
    torch.testing.assert_close(out, torch.tensor(0.0), atol=1e-10, rtol=0)
    assert loss.last_fss_term == pytest.approx(0.0, abs=1e-12)


def test_fss_neighborhood_forgives_displacement():
    """A 1-cell displaced rain blob costs less at larger windows."""
    m = region_mask(LAT, LON, BOX)
    rows, cols = torch.nonzero(m, as_tuple=True)
    # A cell well inside the box, so the shifted blob stays inside too.
    r, c = int(rows[len(rows) // 2]), int(cols[len(cols) // 2])
    target = torch.zeros(1, 15, 72)
    target[0, r, c] = 50.0
    pred = torch.zeros(1, 15, 72)
    pred[0, r, c + 1] = 50.0  # displaced by one cell
    terms = {}
    for k in (1, 3, 5):
        loss = _fss(windows=(k,))
        loss(pred, target, target)
        terms[k] = loss.last_fss_term
    assert terms[1] > terms[3] > terms[5]
    assert terms[1] == pytest.approx(1.0, abs=1e-2)  # total miss at grid scale


def test_fss_degenerate_dry_batch_is_zero_not_nan():
    loss = _fss(thresholds=(20.0,))
    target = torch.zeros(1, 15, 72)
    out = loss(torch.zeros(1, 15, 72), target, target)
    assert torch.isfinite(out)
    assert loss.last_fss_term == pytest.approx(0.0, abs=1e-12)


def test_fss_mask_isolation():
    """Values outside the region (or on NaN target cells) cannot leak into
    the fractions through the pooling window."""
    torch.manual_seed(4)
    target = torch.rand(1, 15, 72) * 20
    pred = torch.rand(1, 15, 72) * 20
    loss = _fss(windows=(5,))
    loss(pred, target, target)
    base = loss.last_fss_term
    outside = ~region_mask(LAT, LON, BOX)
    pred2 = pred.clone()
    pred2[0, outside] = 999.0
    loss(pred2, target, target)
    assert loss.last_fss_term == pytest.approx(base, rel=1e-6)


def test_fss_composite_total_and_ramp():
    loss = _fss(fss_weight=0.4, ramp_epochs=4)
    torch.manual_seed(5)
    target = torch.rand(2, 15, 72) * 20
    pred = torch.rand(2, 15, 72) * 20
    loss.train()
    loss.set_epoch(1)  # ramp fraction (1+1)/4 = 0.5
    out = loss(pred, target, target)
    expected = loss.last_anchor + 0.4 * 0.5 * loss.last_fss_term
    assert float(out) == pytest.approx(expected, rel=1e-6)
    loss.eval()  # full weight in eval so early stopping compares one objective
    out_eval = loss(pred, target, target)
    expected_eval = loss.last_anchor + 0.4 * loss.last_fss_term
    assert float(out_eval) == pytest.approx(expected_eval, rel=1e-6)


def test_pfss_ensemble_reduces_to_deterministic_when_members_equal():
    anchor = RegionalAlmostFairCRPS(LAT, LON, BOX, pred_space="physical")
    loss = _fss(anchor=anchor)
    torch.manual_seed(6)
    target = torch.rand(2, 15, 72) * 20
    pred = torch.rand(2, 15, 72) * 20
    loss(pred, target, target)
    det_term = loss.last_fss_term
    ens = pred.unsqueeze(1).expand(2, 3, 15, 72).contiguous()
    loss(ens, target, target)
    assert loss.last_fss_term == pytest.approx(det_term, rel=1e-6)


# --------------------------------------------------------------------------- #
# AMSE
# --------------------------------------------------------------------------- #


def test_amse_zero_for_perfect_forecast_and_dry_guard():
    loss = RegionalPrecipAMSE(LAT, LON, BOX, pred_space="physical")
    torch.manual_seed(7)
    target = torch.rand(2, 15, 72) * 20
    out = loss(target.clone(), target, target)
    # float32 sigma/rho arithmetic leaves ~1e-7 residue on a ~100-scale loss.
    assert float(out) == pytest.approx(0.0, abs=1e-5)
    # All-dry sample: informationless, not NaN.
    dry = torch.zeros(1, 15, 72)
    out2 = loss(dry.clone(), dry, dry)
    assert torch.isfinite(out2) and float(out2) == pytest.approx(0.0, abs=1e-10)


def test_amse_penalizes_blur_via_amplitude_term():
    """A smoothed forecast loses fine-band amplitude: the diagnostic must see
    it (amp < 1) and the loss must be positive -- the anti-blur property."""
    import torch.nn.functional as F

    torch.manual_seed(8)
    target = torch.rand(1, 15, 72) * 20
    blurred = F.avg_pool2d(target.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    loss = RegionalPrecipAMSE(LAT, LON, BOX_ALL, pred_space="physical", lat_weighted=False)
    out = loss(blurred, target, target)
    assert float(out) > 0
    assert loss.last_amp_bands[0] < 0.6  # fine band lost most of its variance


def test_amse_displacement_keeps_amplitude():
    """A displaced-but-sharp forecast pays only decorrelation: the banded
    amplitude ratios stay ~1 (the double-penalty fix in one assert)."""
    torch.manual_seed(9)
    target = torch.rand(1, 15, 72) * 20
    displaced = torch.roll(target, shifts=3, dims=-1)
    loss = RegionalPrecipAMSE(LAT, LON, BOX_ALL, pred_space="physical", lat_weighted=False)
    out = loss(displaced, target, target)
    assert float(out) > 0  # decorrelation is still penalized...
    for amp in loss.last_amp_bands:  # ...but amplitude is recognized as kept
        assert amp == pytest.approx(1.0, abs=0.05)


def test_amse_amp_diagnostic_tracks_sigma_ratio():
    torch.manual_seed(10)
    target = torch.rand(1, 15, 72) * 10 + 5
    loss = RegionalPrecipAMSE(
        LAT, LON, BOX_ALL, windows=(), pred_space="physical", lat_weighted=False
    )
    # Single whole-field band: doubling the anomaly doubles sigma.
    mean = target.mean()
    pred = (mean + 2.0 * (target - mean)).clamp(min=0.0)
    loss(pred, target, target)
    assert loss.last_amp_bands[0] == pytest.approx(2.0, abs=0.05)


def test_amse_mask_isolation_and_ensemble_fold():
    torch.manual_seed(11)
    target = torch.rand(1, 15, 72) * 20
    pred = torch.rand(1, 15, 72) * 20
    loss = RegionalPrecipAMSE(LAT, LON, BOX, pred_space="physical")
    base = float(loss(pred, target, target))
    outside = ~region_mask(LAT, LON, BOX)
    pred2 = pred.clone()
    pred2[0, outside] = 999.0
    assert float(loss(pred2, target, target)) == pytest.approx(base, rel=1e-5)
    # Ensemble with identical members == deterministic value.
    ens = pred.unsqueeze(1).expand(1, 3, 15, 72).contiguous()
    assert float(loss(ens, target, target)) == pytest.approx(base, rel=1e-5)


# --------------------------------------------------------------------------- #
# gate-map TV
# --------------------------------------------------------------------------- #


def test_gate_tv_zero_for_constant_maps_and_prefers_smooth():
    w = region_weights(LAT, LON, BOX)
    const_w = torch.full((2, 4, 15, 72), 0.25)
    const_b = torch.zeros(2, 4, 15, 72)
    torch.testing.assert_close(
        gate_smoothness_penalty(const_w, const_b, w), torch.tensor(0.0)
    )
    torch.manual_seed(12)
    rough = torch.rand(2, 4, 15, 72)
    import torch.nn.functional as F

    smooth = F.avg_pool2d(rough.reshape(8, 1, 15, 72), 3, 1, 1).reshape(2, 4, 15, 72)
    assert gate_smoothness_penalty(rough, const_b, w) > gate_smoothness_penalty(
        smooth, const_b, w
    )
    # Outside-region structure is inert.
    outside = ~region_mask(LAT, LON, BOX)
    rough2 = rough.clone()
    rough2[..., outside] = 7.0
    base = gate_smoothness_penalty(rough, const_b, w)
    # Edges into the region boundary carry min(w_in, w_out=0) = 0 weight.
    torch.testing.assert_close(
        gate_smoothness_penalty(rough2, const_b, w), base, rtol=1e-5, atol=1e-7
    )
    # 5-D (ensemble) input works.
    assert torch.isfinite(
        gate_smoothness_penalty(rough.unsqueeze(1), const_b.unsqueeze(1), w)
    )


# --------------------------------------------------------------------------- #
# dispatcher + threshold loading
# --------------------------------------------------------------------------- #


def test_build_loss_dispatcher_new_names():
    kw = dict(lat=LAT, lon=LON, box=BOX, precip_mean=5.0, precip_std=10.0)
    crps = build_loss({"name": "regional_crps", "alpha": 0.9}, **kw)
    assert isinstance(crps, RegionalAlmostFairCRPS) and crps.alpha == 0.9
    amse = build_loss({"name": "regional_amse", "windows": [3]}, **kw)
    assert isinstance(amse, RegionalPrecipAMSE) and amse.windows == [3]
    fss = build_loss(
        {
            "name": "regional_fss",
            "anchor": {"name": "regional_mse", "space": "physical"},
            "thresholds": {"kind": "fixed", "values_mm": [5.0, 20.0]},
            "windows": [3, 5],
        },
        **kw,
    )
    assert isinstance(fss, RegionalPrecipFSS)
    assert isinstance(fss.anchor, RegionalPrecipMSE)
    assert fss.threshold_labels == ["5mm", "20mm"]
    with pytest.raises(ValueError, match="anchor"):
        build_loss({"name": "regional_fss", "thresholds": {"kind": "fixed"}}, **kw)
    with pytest.raises(ValueError, match="alpha"):
        build_loss({"name": "regional_crps", "alpha": 2.0}, **kw)


def test_resolve_fss_thresholds_fixed_and_errors():
    maps, labels = resolve_fss_thresholds(
        {"kind": "fixed", "values_mm": [5.0, 20.0]}, LAT, LON
    )
    assert maps.shape == (2, 15, 72) and labels == ["5mm", "20mm"]
    assert (maps[1] == 20.0).all()
    with pytest.raises(ValueError, match="values_mm"):
        resolve_fss_thresholds({"kind": "fixed"}, LAT, LON)
    with pytest.raises(ValueError, match="percentile|fixed"):
        resolve_fss_thresholds({"kind": "banana"}, LAT, LON)


def test_quantile_tool_and_loader_roundtrip(tmp_path):
    """compute_precip_quantiles writes a store the loss loader accepts."""
    from datapipes.testing import GRID_LAT, GRID_LON, write_imerg_store
    from tools.compute_precip_quantiles import main as quant_main

    write_imerg_store(tmp_path / "imerg" / "2001.zarr", year=2001, months=(6, 7))
    out = tmp_path / "quants.zarr"
    rc = quant_main(
        [
            "--imerg-root", str(tmp_path / "imerg"),
            "--years", "2001",
            "--months", "6,7",
            "--quantiles", "50,90,95",
            "--out", str(out),
        ]
    )
    assert rc == 0
    maps = load_precip_quantile_thresholds(
        str(out), [50.0, 90.0, 95.0], GRID_LAT, GRID_LON, floor_mm=1.0
    )
    assert maps.shape == (3, 8, 8)
    assert (maps >= 1.0).all()  # floored
    assert (maps[0] <= maps[1]).all() and (maps[1] <= maps[2]).all()  # monotone
    # Missing quantile and grid mismatch raise, never silently regrid.
    with pytest.raises(ValueError, match="quantile 75"):
        load_precip_quantile_thresholds(str(out), [75.0], GRID_LAT, GRID_LON)
    with pytest.raises(ValueError, match="grid"):
        load_precip_quantile_thresholds(str(out), [50.0], LAT, LON)
    # And the percentile branch of the resolver round-trips.
    maps2, labels2 = resolve_fss_thresholds(
        {"kind": "percentile", "values": [90], "store": str(out)},
        GRID_LAT,
        GRID_LON,
    )
    assert labels2 == ["p90"] and maps2.shape == (1, 8, 8)
