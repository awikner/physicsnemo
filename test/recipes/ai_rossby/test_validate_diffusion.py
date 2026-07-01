# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8f (F4) unit tests for the DiffusionRolloutValidator per-emitted-frame
``sampler_num_steps`` schedule.

All tests use synthetic stubs — no real backbones, no Hydra compose, no
real data — so they finish in milliseconds on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from validate_diffusion import DiffusionRolloutValidator  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal stubs: dataset, wrapper, single-step + window-mode schedulers.
# ---------------------------------------------------------------------------


class _StubDataset:
    """Single-channel-group synthetic dataset — no upper_air/diagnostic."""

    def __init__(self, n_time=20, C=2, H=8, W=8):
        self.n_time = n_time
        torch.manual_seed(0)
        self._surface = torch.randn(n_time, C, H, W)
        self._const = torch.randn(1, H, W)
        self._varying = torch.randn(n_time, 1, H, W)
        self._calendar = torch.randn(n_time, 2)

    def __len__(self):
        return self.n_time

    def __getitem__(self, idx):
        t = idx[0] if isinstance(idx, tuple) else int(idx)
        return {
            "surface_in": self._surface[t],
            "constant_boundary": self._const,
            "varying_boundary": self._varying[t],
            "calendar": self._calendar[t],
        }


class _StubWrapper(nn.Module):
    """Identity pack/unpack — single channel group (surface only)."""

    def pack_state(self, sample):
        return sample["surface_in"]

    def unpack_state(self, x):
        return {"surface_in": x}

    def pack_c_grid(self, sample):
        const = sample["constant_boundary"]
        surface = sample["surface_in"]
        while const.dim() < surface.dim():
            const = const.unsqueeze(0)
        const = const.expand(*surface.shape[:-3], -1, -1, -1)
        return torch.cat([const, sample["varying_boundary"]], dim=-3)

    def pack_window_state(self, window):
        return window["surface_in"]

    def pack_window_c_grid(self, window):
        const = window["constant_boundary"]
        var_b = window["varying_boundary"]
        while const.dim() < var_b.dim():
            const = const.unsqueeze(0)
        const = const.expand(*var_b.shape[:-3], -1, -1, -1)
        return torch.cat([const, var_b], dim=-3)


class _RecordingSingleStepScheduler:
    """Records every ``num_steps`` passed to ``sample()``, in call order."""

    def __init__(self):
        self.num_steps = 4
        self.calls: list = []

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        self.calls.append(num_steps)
        return x + 0.1


class _RecordingRollingScheduler(nn.Module):
    """Records the single ``num_steps`` passed to ``sample_rollout()``."""

    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.num_steps = 2
        self.calls: list = []

    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj, horizon, num_steps=None):
        self.calls.append(num_steps)
        B, W, C, H, Wd = init_window.shape
        out = torch.zeros(B, horizon, C, H, Wd)
        for k in range(horizon):
            out[:, k] = init_window[:, -1] + 0.1 * (k + 1)
        return out


def _make_validator(scheduler, *, horizon, sampler_num_steps):
    return DiffusionRolloutValidator(
        _StubDataset(),
        wrapper=_StubWrapper(),
        inference_scheduler=scheduler,
        log_steps=[horizon],
        device=torch.device("cpu"),
        horizon=horizon,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=sampler_num_steps,
    )


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


def test_sampler_num_steps_schedule_length_must_match_horizon():
    scheduler = _RecordingSingleStepScheduler()
    with pytest.raises(ValueError, match="horizon"):
        _make_validator(scheduler, horizon=4, sampler_num_steps=[5, 4])


def test_sampler_num_steps_accepts_none_int_or_list():
    for value in (None, 3, [5, 4, 3, 2]):
        scheduler = _RecordingSingleStepScheduler()
        v = _make_validator(scheduler, horizon=4, sampler_num_steps=value)
        assert v.sampler_num_steps == value


# ---------------------------------------------------------------------------
# Single-step dispatch: per-frame resolution.
# ---------------------------------------------------------------------------


def test_single_step_uniform_int_applies_to_every_frame():
    scheduler = _RecordingSingleStepScheduler()
    v = _make_validator(scheduler, horizon=4, sampler_num_steps=3)
    v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == [3, 3, 3, 3]


def test_single_step_none_falls_back_to_scheduler_default():
    scheduler = _RecordingSingleStepScheduler()
    v = _make_validator(scheduler, horizon=4, sampler_num_steps=None)
    v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == [None, None, None, None]


def test_single_step_schedule_resolves_per_emitted_frame():
    scheduler = _RecordingSingleStepScheduler()
    schedule = [20, 20, 10, 4]
    v = _make_validator(scheduler, horizon=4, sampler_num_steps=schedule)
    v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == schedule


def test_num_steps_for_frame_helper_is_1_indexed():
    scheduler = _RecordingSingleStepScheduler()
    schedule = [20, 10, 5]
    v = _make_validator(scheduler, horizon=3, sampler_num_steps=schedule)
    assert v._num_steps_for_frame(1) == 20
    assert v._num_steps_for_frame(2) == 10
    assert v._num_steps_for_frame(3) == 5


# ---------------------------------------------------------------------------
# Window-mode dispatch: schedule forwarded verbatim to sample_rollout.
# ---------------------------------------------------------------------------


def test_window_mode_schedule_forwarded_verbatim():
    scheduler = _RecordingRollingScheduler(window_size=3)
    schedule = [8, 4, 2]
    v = _make_validator(scheduler, horizon=3, sampler_num_steps=schedule)
    v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == [schedule]


def test_window_mode_uniform_int_forwarded_verbatim():
    scheduler = _RecordingRollingScheduler(window_size=3)
    v = _make_validator(scheduler, horizon=3, sampler_num_steps=6)
    v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == [6]
