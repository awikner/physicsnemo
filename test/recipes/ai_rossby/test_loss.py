# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`PanguPlasimLoss`.

The loss imports live under ``examples/weather/ai_rossby/`` so we add that
directory to ``sys.path`` (examples don't ship as an installable package).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from loss import (  # noqa: E402
    PanguPlasimLoss,
    cos_lat_weights,
    lat_weighted_residual,
    per_var_lat_weighted_residual,
)


def _zeros_dict(B=2, Cs=2, Cu=3, L=4, H=8, W=16, Cd=1):
    return {
        "out_surface": torch.zeros(B, Cs, H, W),
        "out_upper_air": torch.zeros(B, Cu, L, H, W),
        "target_surface": torch.zeros(B, Cs, H, W),
        "target_upper_air": torch.zeros(B, Cu, L, H, W),
        "out_diagnostic": torch.zeros(B, Cd, H, W),
        "target_diagnostic": torch.zeros(B, Cd, H, W),
    }


def test_cos_lat_weights_mean_is_one():
    w = cos_lat_weights(64, torch.device("cpu"), torch.float32)
    assert w.shape == (64,)
    assert torch.allclose(w.mean(), torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize("loss_type", ["l1", "l2"])
def test_lat_weighted_residual_zero_when_pred_eq_target(loss_type):
    H, W = 8, 16
    lat = cos_lat_weights(H, torch.device("cpu"), torch.float32)
    a = torch.randn(2, 3, H, W)
    out = lat_weighted_residual(a, a, lat, loss_type=loss_type)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


def test_per_var_weight_amplifies_targeted_channel():
    H, W = 4, 8
    C = 3
    lat = cos_lat_weights(H, torch.device("cpu"), torch.float32)
    pred = torch.zeros(1, C, H, W)
    target = torch.ones(1, C, H, W)
    uniform = per_var_lat_weighted_residual(
        pred, target, lat, torch.ones(C), loss_type="l1"
    )
    biased = per_var_lat_weighted_residual(
        pred, target, lat, torch.tensor([10.0, 1.0, 1.0]), loss_type="l1"
    )
    assert biased > uniform


def test_loss_zero_on_identity():
    d = _zeros_dict()
    loss = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=["q"],
        num_lat=8,
        loss_type="l1",
    )
    out = loss(
        d["out_surface"],
        d["out_upper_air"],
        d["target_surface"],
        d["target_upper_air"],
        out_diagnostic=d["out_diagnostic"],
        target_diagnostic=d["target_diagnostic"],
    )
    assert torch.allclose(out["loss"], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(out["surface"], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(out["upper_air"], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(out["diagnostic"], torch.tensor(0.0), atol=1e-7)


def test_loss_components_have_expected_sign_and_weight():
    """A unit-magnitude residual on each branch should yield positive components
    that scale with the per-branch weight."""
    B, Cs, Cu, L, H, W = 1, 2, 3, 2, 8, 16
    pred_s = torch.ones(B, Cs, H, W)
    pred_u = torch.ones(B, Cu, L, H, W)
    pred_d = torch.ones(B, 1, H, W)
    tgt_s = torch.zeros_like(pred_s)
    tgt_u = torch.zeros_like(pred_u)
    tgt_d = torch.zeros_like(pred_d)

    base = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=["q"],
        num_lat=H,
        loss_type="l1",
    )
    weighted = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=["q"],
        num_lat=H,
        loss_type="l1",
        surface_weight=2.0,
        upper_air_weight=2.0,
        diagnostic_weight=2.0,
    )
    b_out = base(pred_s, pred_u, tgt_s, tgt_u, pred_d, tgt_d)
    w_out = weighted(pred_s, pred_u, tgt_s, tgt_u, pred_d, tgt_d)
    assert b_out["surface"] > 0
    assert b_out["upper_air"] > 0
    assert b_out["diagnostic"] > 0
    # total weighted == 2× base (since each component doubled).
    assert torch.allclose(w_out["loss"], 2.0 * b_out["loss"], atol=1e-6)


def test_loss_rejects_unknown_type():
    with pytest.raises(ValueError, match="loss_type"):
        PanguPlasimLoss(
            surface_variables=["a"],
            upper_air_variable_names=["x"],
            diagnostic_variables=[],
            num_lat=4,
            loss_type="huber",
        )


def test_loss_backward_produces_gradients():
    """Gradients flow to all predictions when each branch contributes."""
    B, Cs, Cu, L, H, W = 1, 2, 3, 2, 8, 16
    pred_s = torch.zeros(B, Cs, H, W, requires_grad=True)
    pred_u = torch.zeros(B, Cu, L, H, W, requires_grad=True)
    pred_d = torch.zeros(B, 1, H, W, requires_grad=True)
    tgt_s = torch.ones_like(pred_s)
    tgt_u = torch.ones_like(pred_u)
    tgt_d = torch.ones_like(pred_d)

    loss = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=["q"],
        num_lat=H,
        loss_type="l2",
    )
    out = loss(pred_s, pred_u, tgt_s, tgt_u, pred_d, tgt_d)
    out["loss"].backward()
    for g in (pred_s.grad, pred_u.grad, pred_d.grad):
        assert g is not None
        assert torch.isfinite(g).all()
        assert (g != 0).any()


# ---------------------------------------------------------------------------
# latitude_weighted=False (PanguWeather raw_l1 / raw_l2 semantics)
# ---------------------------------------------------------------------------
def test_lat_weighted_residual_raw_l2_drops_cos_weighting():
    """`latitude_weighted=False` reduces to plain `(pred - target)**2 .mean()`."""
    # Use H >= 4 so cos(lat) varies meaningfully across the axis — at H=2 the
    # symmetric Gauss-ish grid produces uniform weights and the two paths
    # would (correctly) collapse to the same number.
    torch.manual_seed(0)
    pred = torch.randn(1, 1, 8, 16)
    target = torch.zeros_like(pred)
    lat = cos_lat_weights(8, pred.device, pred.dtype)
    plain = (pred - target).pow(2).mean()
    raw = lat_weighted_residual(
        pred, target, lat, loss_type="l2", latitude_weighted=False
    )
    assert torch.allclose(raw, plain)
    weighted = lat_weighted_residual(
        pred, target, lat, loss_type="l2", latitude_weighted=True
    )
    assert not torch.allclose(raw, weighted)


def test_per_var_lat_weighted_residual_raw_keeps_per_var_weights():
    """raw_l2 drops cos(lat) but per_var weights still apply."""
    pred = torch.ones(1, 2, 2, 2)  # (B=1, C=2, H=2, W=2) — all ones
    target = torch.zeros_like(pred)
    lat = cos_lat_weights(2, pred.device, pred.dtype)
    per_var = torch.tensor([1.0, 4.0])
    raw = per_var_lat_weighted_residual(
        pred, target, lat, per_var, loss_type="l2", latitude_weighted=False
    )
    # mean over (1,2,2,2) of resid * (1 or 4) → (1*4 + 4*4) / (2*4) = 20 / 8 = 2.5
    assert raw.item() == pytest.approx(2.5)


def test_pangu_plasim_loss_latitude_weighted_flag_propagates():
    """PanguPlasimLoss(latitude_weighted=False) matches the raw_l2 residual semantics."""
    torch.manual_seed(0)
    B, Cs, Cu, L, H, W = 2, 2, 3, 4, 8, 16
    pred_s = torch.randn(B, Cs, H, W)
    pred_u = torch.randn(B, Cu, L, H, W)
    tgt_s = torch.randn(B, Cs, H, W)
    tgt_u = torch.randn(B, Cu, L, H, W)
    weighted = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=[],
        num_lat=H,
        loss_type="l2",
        latitude_weighted=True,
    )
    raw = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=[],
        num_lat=H,
        loss_type="l2",
        latitude_weighted=False,
    )
    out_w = weighted(pred_s, pred_u, tgt_s, tgt_u)
    out_r = raw(pred_s, pred_u, tgt_s, tgt_u)
    # The raw variant differs from the cos-lat-weighted variant on the same inputs.
    assert not torch.allclose(out_w["loss"], out_r["loss"])
    # The raw variant matches per_var_lat_weighted_residual(latitude_weighted=False).
    # Surface: per_var ones[Cs] → just the channel-weighted residual mean.
    expected_surface = (pred_s - tgt_s).pow(2).mean()
    assert torch.allclose(out_r["surface"], expected_surface)
