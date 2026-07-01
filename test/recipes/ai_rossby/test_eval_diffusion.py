# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8f (F5) unit tests for the long-horizon climate eval suite.

All tests use synthetic stubs — no real backbones, no Hydra compose,
no real data — so they finish in milliseconds on CPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from eval_diffusion import (  # noqa: E402
    BiasValidator,
    ClimatologyValidator,
    EnsembleEnvelopeValidator,
    GlobalMeanTimeseriesValidator,
    QBOValidator,
    _estimate_period_months,
    _tropical_band_mask_and_weights,
)
from validate import ReplicateOnly  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs: dataset (surface + upper_air), wrapper, single-step scheduler.
# ---------------------------------------------------------------------------

_SURFACE_VARS = ["t2m", "DSWRFtoa"]
_UPPER_VARS = ["ua"]
_LEVELS = [10.0, 30.0, 50.0]
_H, _W = 8, 8


class _StubDataset:
    def __init__(self, n_time=60):
        self.n_time = n_time
        torch.manual_seed(0)
        self._surface = torch.randn(n_time, len(_SURFACE_VARS), _H, _W)
        self._upper = torch.randn(n_time, len(_UPPER_VARS), len(_LEVELS), _H, _W)
        self._const = torch.randn(1, _H, _W)
        self._varying = torch.randn(n_time, 1, _H, _W)
        self._calendar = torch.randn(n_time, 2)

    def __len__(self):
        return self.n_time

    def __getitem__(self, idx):
        t = idx[0] if isinstance(idx, tuple) else int(idx)
        return {
            "surface_in": self._surface[t],
            "upper_air_in": self._upper[t],
            "constant_boundary": self._const,
            "varying_boundary": self._varying[t],
            "calendar": self._calendar[t],
        }


class _StubWrapper(nn.Module):
    surface_variables = list(_SURFACE_VARS)
    upper_air_variables = list(_UPPER_VARS)
    diagnostic_variables: list = []
    levels = list(_LEVELS)

    def pack_state(self, sample):
        s = sample["surface_in"]
        ua = sample["upper_air_in"]
        b_shape = ua.shape[:-4]
        ua_flat = ua.reshape(*b_shape, len(_UPPER_VARS) * len(_LEVELS), *ua.shape[-2:])
        return torch.cat([s, ua_flat], dim=-3)

    def unpack_state(self, x):
        n_s = len(_SURFACE_VARS)
        n_ul = len(_UPPER_VARS) * len(_LEVELS)
        surface = x.narrow(-3, 0, n_s)
        ua_flat = x.narrow(-3, n_s, n_ul)
        b_shape = ua_flat.shape[:-3]
        upper = ua_flat.reshape(*b_shape, len(_UPPER_VARS), len(_LEVELS), *ua_flat.shape[-2:])
        return {"surface_in": surface, "upper_air_in": upper}

    def pack_c_grid(self, sample):
        const = sample["constant_boundary"]
        surface = sample["surface_in"]
        while const.dim() < surface.dim():
            const = const.unsqueeze(0)
        const = const.expand(*surface.shape[:-3], -1, -1, -1)
        return torch.cat([const, sample["varying_boundary"]], dim=-3)


class _RecordingSingleStepScheduler:
    def __init__(self):
        self.num_steps = 4

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        return x + 0.1


def _make_kwargs(horizon=6, **overrides):
    kwargs = dict(
        wrapper=_StubWrapper(),
        inference_scheduler=_RecordingSingleStepScheduler(),
        horizon=horizon,
        device=torch.device("cpu"),
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# ClimatologyValidator / BiasValidator
# ---------------------------------------------------------------------------


def test_climatology_validator_returns_expected_keys_and_shapes():
    v = ClimatologyValidator(_StubDataset(), n_bins=3, steps_per_bin=2, **_make_kwargs())
    result = v.run(nn.Identity(), epoch=0)
    clim = result["climatology"]
    surf_shape = (len(_SURFACE_VARS), _H, _W)
    assert clim["surface_pred_mean"].shape == surf_shape
    assert clim["surface_truth_mean"].shape == surf_shape
    assert torch.allclose(
        clim["surface_bias"], clim["surface_pred_mean"] - clim["surface_truth_mean"]
    )
    assert clim["surface_pred_binned"].shape == (3, *surf_shape)
    upper_shape = (len(_UPPER_VARS), len(_LEVELS), _H, _W)
    assert clim["upper_air_pred_mean"].shape == upper_shape
    assert "rmse_acc" in result


def test_bias_validator_matches_lat_weighted_reduction_of_bias_field():
    from climatology import lat_weighted_global_scalars

    v = BiasValidator(_StubDataset(), n_bins=3, steps_per_bin=2, **_make_kwargs())
    result = v.run(nn.Identity(), epoch=0)
    expected = lat_weighted_global_scalars(result["climatology"]["surface_bias"])
    assert torch.allclose(result["global_bias"]["surface"], expected)
    assert result["global_bias"]["surface"].shape == (len(_SURFACE_VARS),)


# ---------------------------------------------------------------------------
# QBOValidator
# ---------------------------------------------------------------------------


def test_tropical_band_mask_and_weights_sum_to_one():
    mask, weights = _tropical_band_mask_and_weights(
        _H, 30.0, torch.device("cpu"), torch.float32
    )
    assert mask.dtype == torch.bool
    assert mask.sum().item() > 0
    assert weights.numel() == mask.sum().item()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_estimate_period_months_pure_sine():
    # 12 bins spanning one full period (matches how a clean periodic
    # signal composites into a climatological bin timeseries). A small
    # phase offset keeps the zero crossings off the array boundary.
    n_bins = 12
    ts = torch.tensor(
        [math.sin(2 * math.pi * i / n_bins + 0.3) for i in range(n_bins)]
    )
    period = _estimate_period_months(ts, months_per_bin=1.0)
    assert period == pytest.approx(12.0, abs=2.0)


def test_estimate_period_months_returns_nan_for_constant_series():
    # No oscillation at all (not even under the circular wrap-around
    # treatment) -> no crossings -> nan.
    ts = torch.full((10,), 3.0)
    assert math.isnan(_estimate_period_months(ts, months_per_bin=1.0))


def test_qbo_validator_smoke_returns_expected_keys_and_shapes():
    v = QBOValidator(
        _StubDataset(),
        qbo_levels=(10.0, 30.0, 50.0),
        steps_per_bin=2,
        months_per_bin=1.0,
        **_make_kwargs(horizon=12),
    )
    result = v.run(nn.Identity(), epoch=0)
    assert result["qbo_pred_timeseries"].shape == (v.n_bins, 3)
    assert result["qbo_truth_timeseries"].shape == (v.n_bins, 3)
    for lvl in (10, 30, 50):
        assert f"qbo_period_months_pred_hPa{lvl}" in result
        assert f"qbo_period_months_truth_hPa{lvl}" in result


def test_qbo_validator_rejects_unknown_level():
    with pytest.raises(ValueError, match="qbo_levels"):
        QBOValidator(
            _StubDataset(),
            qbo_levels=(999.0,),
            **_make_kwargs(horizon=6),
        )


def test_qbo_validator_rejects_unknown_u_variable():
    with pytest.raises(ValueError, match="u_variable_name"):
        QBOValidator(
            _StubDataset(),
            u_variable_name="nonexistent",
            **_make_kwargs(horizon=6),
        )


# ---------------------------------------------------------------------------
# GlobalMeanTimeseriesValidator
# ---------------------------------------------------------------------------


def test_global_mean_timeseries_validator_tracks_requested_flux_variables():
    v = GlobalMeanTimeseriesValidator(
        _StubDataset(),
        flux_variables=["DSWRFtoa"],
        **_make_kwargs(horizon=6),
    )
    result = v.run(nn.Identity(), epoch=0)
    assert result["flux_pred_series"]["DSWRFtoa"].shape == (6,)
    assert result["flux_truth_series"]["DSWRFtoa"].shape == (6,)


def test_global_mean_timeseries_validator_rejects_unknown_flux_variable():
    with pytest.raises(ValueError, match="flux variable"):
        GlobalMeanTimeseriesValidator(
            _StubDataset(),
            flux_variables=["not_a_real_channel"],
            **_make_kwargs(horizon=6),
        )


# ---------------------------------------------------------------------------
# EnsembleEnvelopeValidator
# ---------------------------------------------------------------------------


def test_ensemble_envelope_validator_requires_ensemble_size_gt_1():
    with pytest.raises(ValueError, match="ensemble_size"):
        EnsembleEnvelopeValidator(_StubDataset(), ensemble_size=1, **_make_kwargs(horizon=3))


def test_ensemble_envelope_validator_reports_spread_skill_ratio():
    v = EnsembleEnvelopeValidator(
        _StubDataset(),
        ensemble_size=3,
        perturber=ReplicateOnly(),
        **_make_kwargs(horizon=3),
    )
    result = v.run(nn.Identity(), epoch=0)
    ratio_keys = [k for k in result if k.startswith("spread_skill_ratio_")]
    assert len(ratio_keys) > 0
    for k in ratio_keys:
        assert isinstance(result[k], float)
