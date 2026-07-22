# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ArchesWeatherLoss (run from the recipe dir: pytest test_archesweather_loss.py)."""

import json

import torch

from loss import ArchesWeatherLoss

SURFACE = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
UPPER = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
LEVELS = [5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
GW = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
}


def _loss(**kw):
    return ArchesWeatherLoss(
        surface_variables=SURFACE,
        upper_air_variable_names=UPPER,
        levels=LEVELS,
        num_lat=180,
        surface_graphcast_weights=GW,
        **kw,
    )


def test_coefficients_and_forward():
    loss = _loss()
    # total_coeff = n_level(5) + sum(surface weights)(1.3) = 6.3
    # surface scale = weight * n_surf(4) / 6.3
    assert abs(float(loss.surface_scale[0]) - 1.0 * 4 / 6.3) < 1e-5  # t2m
    assert abs(float(loss.surface_scale[1]) - 0.1 * 4 / 6.3) < 1e-5  # u10
    # level scale = (n_level/6.3) * p/mean(p); mean of 17 levels = 350.588...
    mean_p = sum(LEVELS) / len(LEVELS)
    assert abs(float(loss.level_scale[0, 0]) - (5 / 6.3) * (LEVELS[0] / mean_p)) < 1e-4
    assert abs(float(loss.level_scale[0, -1]) - (5 / 6.3) * (LEVELS[-1] / mean_p)) < 1e-4

    B, H, W = 2, 180, 360
    ps = torch.randn(B, 4, H, W, requires_grad=True)
    pu = torch.randn(B, 5, 17, H, W, requires_grad=True)
    out = loss(ps, pu, torch.randn(B, 4, H, W), torch.randn(B, 5, 17, H, W))
    assert set(out) == {"loss", "surface", "upper_air", "diagnostic"}
    assert out["loss"].requires_grad and float(out["loss"]) > 0
    out["loss"].backward()
    assert torch.isfinite(ps.grad).all() and torch.isfinite(pu.grad).all()


def test_delta_scaler(tmp_path):
    base = _loss()
    spec = {"surface": {v: 2.0 for v in SURFACE}, "level": {v: [1.5] * 17 for v in UPPER}}
    p = tmp_path / "scaler.json"
    p.write_text(json.dumps(spec))
    scaled = _loss(delta_scaler_path=str(p))
    # scaler is squared: surface x4, level x2.25
    assert torch.allclose(scaled.surface_scale / base.surface_scale, torch.full((4,), 4.0), atol=1e-5)
    assert torch.allclose(
        scaled.level_scale / base.level_scale, torch.full_like(base.level_scale, 2.25), atol=1e-4
    )
