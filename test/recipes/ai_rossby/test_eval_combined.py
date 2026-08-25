# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Upstream-parity Combined-cascade eval machinery: the drive's
frame_transform / unpack_wrapper / init_downsample_factor / explicit-IC
hooks, and the headline bias reduction. Synthetic stubs only — CPU,
milliseconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_diffusion import (  # noqa: E402
    DiffusionRolloutValidator,
    _downsample_state_window,
)
from climate_eval_suite import (  # noqa: E402
    DEFAULT_HEADLINE_VARIABLES,
    VariableCatalog,
    _is_combined_model_cfg,
    compute_headline_bias,
)
from test_validate_diffusion import _StubDataset, _StubWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Cascade-shaped stubs
# ---------------------------------------------------------------------------
class _ConstFieldDataset(_StubDataset):
    """Surface fields are SPATIALLY CONSTANT (value = frame index).

    Bilinear down- then up-sampling is exact on constants, so a correct
    cascade — downsample the oracle window, roll, strip the tail, upsample,
    score at full res — must reproduce the truth bit-for-bit. Any indexing,
    stripping or resampling mistake breaks the zero-RMSE assertion.
    """

    def __init__(self, n_time=20, C=2, H=8, W=8):
        super().__init__(n_time=n_time, C=C, H=H, W=W)
        for t in range(n_time):
            self._surface[t] = float(t)


class _EchoWithOceanTailScheduler(nn.Module):
    """Oracle-echo rolling scheduler that appends a 1-channel 'ocean' tail.

    The emitted frames therefore have C+1 channels, like a real ocean-carrying
    forecaster; the drive's frame_transform must see (and strip) that tail.
    """

    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.nocean = 1
        self.num_steps = 2
        self.transform_input_channels: list[int] = []

    def strip_ocean(self, x):
        return x[:, : x.shape[1] - self.nocean]

    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        n_anchor = init_window.shape[1] - self.window_size
        frames = init_window[:, n_anchor:n_anchor + horizon]
        tail = torch.zeros(frames.shape[0], horizon, self.nocean,
                           *frames.shape[-2:])
        return torch.cat([frames, tail], dim=2)


class _UpsamplingDownscalerWrapper(nn.Module):
    """The 'downscaler' side of the cascade: unpack at the high-res grid."""

    def unpack_state(self, x):
        return {"surface_in": x}


def _upsample_transform(sched, factor, record):
    def transform(x):
        record.append(int(x.shape[1]))
        bare = sched.strip_ocean(x)
        return F.interpolate(bare, scale_factor=factor, mode="bilinear",
                             align_corners=False)
    return transform


# ---------------------------------------------------------------------------
# End-to-end cascade through the drive
# ---------------------------------------------------------------------------
def test_cascade_scores_zero_rmse_at_the_highres_grid():
    ds = _ConstFieldDataset(H=8, W=8)
    sched = _EchoWithOceanTailScheduler(window_size=3)
    seen_channels: list[int] = []
    v = DiffusionRolloutValidator(
        ds,
        wrapper=_StubWrapper(),
        inference_scheduler=sched,
        log_steps=[1, 2, 3],
        device=torch.device("cpu"),
        horizon=3,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=2,
        frame_transform=_upsample_transform(sched, 2, seen_channels),
        unpack_wrapper=_UpsamplingDownscalerWrapper(),
        init_downsample_factor=2,
    )
    results = v.run(nn.Identity(), epoch=0)
    # scored at full res against full-res truth: exact on constant fields
    for step in (1, 2, 3):
        assert results[f"rmse_step{step}_surface"] == pytest.approx(0.0, abs=1e-6)
    # the transform received the PRE-strip frame (state + ocean tail)
    assert seen_channels and all(c == 2 + 1 for c in seen_channels)


def test_transform_hooks_require_window_mode():
    class _SingleStep:
        num_steps = 2
        def sample(self, model, x, c_grid, c_scalar, num_steps=None):
            return x
    with pytest.raises(ValueError, match="window-mode"):
        DiffusionRolloutValidator(
            _StubDataset(),
            wrapper=_StubWrapper(),
            inference_scheduler=_SingleStep(),
            log_steps=[1],
            device=torch.device("cpu"),
            horizon=1,
            frame_transform=lambda x: x,
        )


# ---------------------------------------------------------------------------
# Init-window downsampling
# ---------------------------------------------------------------------------
def test_downsample_state_window_matches_interpolate_and_skips_forcings():
    torch.manual_seed(3)
    window = {
        "surface_in": torch.randn(2, 3, 4, 8, 8),
        "upper_air_in": torch.randn(2, 3, 5, 2, 8, 8),     # 6-D
        "diagnostic": torch.randn(2, 3, 6, 8, 8),
        "varying_boundary": torch.randn(2, 3, 1, 8, 8),
        "calendar": torch.randn(2, 3, 2),
    }
    out = _downsample_state_window(window, 2)
    for key in ("surface_in", "upper_air_in", "diagnostic"):
        v = window[key]
        ref = F.interpolate(
            v.reshape(-1, 1, 8, 8), scale_factor=0.5, mode="bilinear",
            align_corners=False,
        ).reshape(*v.shape[:-2], 4, 4)
        torch.testing.assert_close(out[key], ref)
    # forcing/calendar side untouched — byte identity, not just shape
    assert out["varying_boundary"] is window["varying_boundary"]
    assert out["calendar"] is window["calendar"]


# ---------------------------------------------------------------------------
# Explicit initial conditions
# ---------------------------------------------------------------------------
def test_explicit_ic_indices_used_verbatim():
    ds = _StubDataset(n_time=20)
    from test_validate_diffusion import _OracleEchoScheduler
    sched = _OracleEchoScheduler(window_size=3)
    v = DiffusionRolloutValidator(
        ds, wrapper=_StubWrapper(), inference_scheduler=sched,
        log_steps=[1], device=torch.device("cpu"), horizon=3,
        max_initial_conditions=4, ic_stride=1, sampler_num_steps=2,
        ic_indices=[7],
    )
    assert v._select_ic_indices(rank=0, world_size=1) == [7]


def test_explicit_ic_out_of_bounds_raises_with_the_bound():
    ds = _StubDataset(n_time=20)
    from test_validate_diffusion import _OracleEchoScheduler
    sched = _OracleEchoScheduler(window_size=3)
    v = DiffusionRolloutValidator(
        ds, wrapper=_StubWrapper(), inference_scheduler=sched,
        log_steps=[1], device=torch.device("cpu"), horizon=3,
        sampler_num_steps=2, ic_indices=[19],
    )
    with pytest.raises(ValueError, match=r"admissible range \[0, \d+\]"):
        v._select_ic_indices(rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Headline bias reduction
# ---------------------------------------------------------------------------
def _toy_catalog():
    return VariableCatalog(
        surface=["2m_temperature", "10m_u_component_of_wind"],
        upper_air=["temperature", "geopotential"],
        diagnostic=["PRATEsfc_24h"],
        levels=[850.0, 500.0],       # deliberately NOT ascending
    )


def _lat_weights(H):
    import math
    phi = torch.linspace(math.pi / 2, -math.pi / 2, H, dtype=torch.float64)
    w = torch.cos(phi)
    return w / w.mean()


def test_headline_bias_matches_hand_computed_reduction():
    torch.manual_seed(5)
    H, W = 6, 12
    maps = {
        "surface_bias": torch.randn(2, H, W),
        "upper_air_bias": torch.randn(2, 2, H, W),
        "diagnostic_bias": torch.randn(1, H, W),
    }
    spec = [
        ["z500", "geopotential", 500],       # by VALUE: level index 1
        ["t2m", "2m_temperature", None],
        ["prate", "PRATEsfc_24h", None],     # diagnostic fallback
    ]
    out = compute_headline_bias(maps, _toy_catalog(), spec)
    w = _lat_weights(H)[:, None]
    for label, bias in (
        ("z500", maps["upper_air_bias"][1, 1]),
        ("t2m", maps["surface_bias"][0]),
        ("prate", maps["diagnostic_bias"][0]),
    ):
        b = bias.double()
        assert out[label]["mean_bias"] == pytest.approx(
            float((b * w).mean()), rel=1e-6)
        assert out[label]["rmse_bias"] == pytest.approx(
            float((b.pow(2) * w).mean().sqrt()), rel=1e-6)


def test_headline_bias_unknown_name_and_level_raise():
    maps = {"surface_bias": torch.randn(2, 4, 8),
            "upper_air_bias": torch.randn(2, 2, 4, 8)}
    with pytest.raises(ValueError, match="not in surface"):
        compute_headline_bias(maps, _toy_catalog(), [["x", "no_such_var", None]])
    with pytest.raises(ValueError, match="not in the model's levels"):
        compute_headline_bias(maps, _toy_catalog(), [["z250", "geopotential", 250]])


def test_default_headline_matches_upstream_field_set():
    labels = [e[0] for e in DEFAULT_HEADLINE_VARIABLES]
    assert labels == ["z500", "u250", "t850", "q850",
                      "t2m", "prate", "q2m", "u10m", "v10m"]


# ---------------------------------------------------------------------------
# Combined-config detection
# ---------------------------------------------------------------------------
def test_is_combined_model_cfg():
    from omegaconf import OmegaConf
    combined = OmegaConf.create(
        {"forecaster": {"model": "a"}, "downscaler": {"model": "b"}}
    )
    plain = OmegaConf.create({"name": "RollingDiTWrapper"})
    assert _is_combined_model_cfg(combined)
    assert not _is_combined_model_cfg(plain)
    assert not _is_combined_model_cfg(None)
