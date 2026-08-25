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
        # Delegates to the REAL broadcast helper on purpose. This stub used to
        # reimplement it — and did so incorrectly, inserting the missing axis at
        # dim 0 rather than after the batch axis, which happens to work only when
        # B == 1. That is precisely why every test here passed while the rolling
        # validator could not run against a v1/v2 model on real data
        # (`Tensors must have same number of dimensions: got 5 and 4`). A stub
        # that re-derives the contract it is standing in for cannot catch the
        # contract being wrong.
        from physicsnemo.experimental.models.amip_si.wrappers import (
            _broadcast_constant,
        )

        var_b = window["varying_boundary"]
        const = _broadcast_constant(window["constant_boundary"], var_b.shape[:-3])
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


# ---------------------------------------------------------------------------
# Tuple-returning schedulers (SI_X). Fixed 2026-08-14.
# ---------------------------------------------------------------------------


class _TupleReturningScheduler(_RecordingSingleStepScheduler):
    """``sample`` returns ``(y, model_last_pred)``, like SI_X.

    :class:`~physicsnemo.experimental.diffusion.DynamicInterpolant.sample`
    defaults to ``return_model_last=True``, so it hands back a 2-tuple where
    DriftScheduler hands back a tensor. The validator used to pass that
    straight into ``unpack_state`` and die on ``'tuple' object has no attribute
    'narrow'`` — i.e. SI_X could be trained but never validated or rolled out.
    """

    def __init__(self):
        super().__init__()
        self.seen: list = []

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        self.calls.append(num_steps)
        self.seen.append(x.clone())
        y = x + 0.1
        return y, y * 2.0  # second element is the endpoint prediction


def test_single_step_unwraps_a_tuple_returning_sampler():
    scheduler = _TupleReturningScheduler()
    v = _make_validator(scheduler, horizon=3, sampler_num_steps=2)
    metrics = v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == [2, 2, 2]
    # Scored, not crashed: some RMSE came back for the logged step.
    assert any("rmse" in k for k in metrics), sorted(metrics)


def test_the_first_tuple_element_is_the_one_that_marches():
    """The sample, not the endpoint prediction, must feed the next step.

    Taking ``[1]`` is also shape-legal and would silently walk a different
    trajectory, so pin it by what the sampler is *handed*: chaining the first
    element advances by +0.1 per step, chaining the second would double.
    """
    scheduler = _TupleReturningScheduler()
    v = _make_validator(scheduler, horizon=3, sampler_num_steps=1)
    v.run(nn.Identity(), epoch=0)
    assert len(scheduler.seen) == 3
    for k in (1, 2):
        torch.testing.assert_close(
            scheduler.seen[k], scheduler.seen[k - 1] + 0.1, rtol=0, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Over-long horizons (2026-08-14). An empty IC list used to run zero samples
# and report RMSE 0.0 — a perfect score for having evaluated nothing, which is
# what the shipped eval_suite horizon (1460, a 6-hourly year) produced on a
# one-year store at the AMIP 24-hour step.
# ---------------------------------------------------------------------------


def test_a_horizon_the_store_cannot_serve_raises():
    scheduler = _RecordingSingleStepScheduler()
    v = _make_validator(scheduler, horizon=40, sampler_num_steps=1)  # store: 20
    with pytest.raises(ValueError, match="no admissible initial condition"):
        v.run(nn.Identity(), epoch=0)
    assert scheduler.calls == [], "sampled despite having no IC"


def test_the_error_reports_the_largest_horizon_that_fits():
    """A bare "won't fit" is a puzzle; the number is the fix."""
    v = _make_validator(
        _RecordingSingleStepScheduler(), horizon=40, sampler_num_steps=1
    )
    with pytest.raises(ValueError) as excinfo:
        v.run(nn.Identity(), epoch=0)
    msg = str(excinfo.value)
    assert "Largest horizon this store supports is 19" in msg, msg


def test_a_horizon_that_just_fits_is_still_accepted():
    """Guard the boundary, so it cannot drift into rejecting valid runs."""
    scheduler = _RecordingSingleStepScheduler()
    v = _make_validator(scheduler, horizon=19, sampler_num_steps=1)
    v.run(nn.Identity(), epoch=0)
    assert len(scheduler.calls) == 19


# ---------------------------------------------------------------------------
# init_frames — a data-coupled scheduler (RSI) wants W+1 oracle frames
# ---------------------------------------------------------------------------


class _RecordingInitFrames(_RecordingRollingScheduler):
    """Records the shape of the oracle window it is handed."""

    def __init__(self, window_size=3, init_frames=None):
        super().__init__(window_size=window_size)
        if init_frames is not None:
            self.init_frames = init_frames
        self.init_shapes: list = []

    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        self.init_shapes.append(tuple(init_window.shape))
        return super().sample_rollout(
            model, init_window, c_grid_traj, c_scalar_traj, horizon, num_steps)


def test_validator_defaults_init_frames_to_window_size():
    """ERDM/RFM expose no init_frames and must keep their W-frame window."""
    s = _RecordingRollingScheduler(window_size=3)
    v = _make_validator(s, horizon=3, sampler_num_steps=2)
    assert v.init_frames == 3
    v.run(nn.Identity(), epoch=0)
    assert s.init_shapes[0][1] == 3 if hasattr(s, "init_shapes") else True


def test_validator_stacks_the_extra_anchor_frame():
    s = _RecordingInitFrames(window_size=3, init_frames=4)
    v = _make_validator(s, horizon=3, sampler_num_steps=2)
    assert v.init_frames == 4
    v.run(nn.Identity(), epoch=0)
    assert s.init_shapes and s.init_shapes[0][1] == 4


def _validator_all_ics(scheduler, *, horizon):
    """Like _make_validator but without the 1-IC cap, so the full range shows."""
    return DiffusionRolloutValidator(
        _StubDataset(),
        wrapper=_StubWrapper(),
        inference_scheduler=scheduler,
        log_steps=[horizon],
        device=torch.device("cpu"),
        horizon=horizon,
        max_initial_conditions=1000,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=2,
    )


def test_extra_anchor_frame_costs_no_ic_headroom():
    """The oracle window is the FUTURE y_{1:W}; RSI's extra anchor frame is
    the IC itself, so asking for it must not change which ICs are admissible
    (nothing before the IC is ever read)."""
    a = _validator_all_ics(_RecordingRollingScheduler(window_size=3),
                           horizon=3)._select_ic_indices(rank=0, world_size=1)
    b = _validator_all_ics(_RecordingInitFrames(window_size=3, init_frames=4),
                           horizon=3)._select_ic_indices(rank=0, world_size=1)
    assert b == a
    assert min(a) == 0                            # no past rows reserved


def test_window_rollout_with_a_multi_ic_batch():
    """B > 1 in window mode — the shape the constant-boundary bug needed.

    Every other test here runs ``batch_size=1``, which is exactly why the
    ``(B, C, H, W)`` vs ``(B, T, C, H, W)`` rank mismatch in
    ``_rollout_window`` stayed invisible: at B == 1 a wrongly-inserted axis
    still broadcasts. With B == 2 and a trajectory length != 2 it does not, and
    ``pack_window_c_grid`` raises *Tensors must have same number of dimensions*.
    """
    scheduler = _RecordingRollingScheduler(window_size=3)
    v = DiffusionRolloutValidator(
        _StubDataset(),
        wrapper=_StubWrapper(),
        inference_scheduler=scheduler,
        log_steps=[3],
        device=torch.device("cpu"),
        horizon=3,
        max_initial_conditions=2,
        batch_size=2,
        ic_stride=1,
        sampler_num_steps=2,
    )
    out = v.run(nn.Identity(), epoch=0)
    assert scheduler.calls, "the rolling sampler was never reached"
    assert out is None or isinstance(out, dict)


# ---------------------------------------------------------------------------
# Emit-time alignment regression (2026-08). The driver used to hand the
# schedulers a PAST window ending at the IC where their contract is the
# FUTURE oracle window y_{1:W} — every emit then trailed its scored truth by
# W steps and the forcings ran one step off the lag-1 training alignment.
# ---------------------------------------------------------------------------


class _OracleEchoScheduler(nn.Module):
    """Records its inputs and echoes the oracle window back as the forecast.

    When the driver honors the contract, emit k IS the truth at t + k, so the
    validator's RMSE must be exactly zero at every step.
    """

    def __init__(self, window_size=3, init_frames=None, nocean=0):
        super().__init__()
        self.window_size = window_size
        if init_frames is not None:
            self.init_frames = init_frames
        self.nocean = nocean
        self.num_steps = 2
        self.seen: dict = {}

    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        self.seen = {
            "init": init_window.detach().clone(),
            "c_grid": c_grid_traj.detach().clone(),
            "c_scalar": c_scalar_traj.detach().clone(),
        }
        n_anchor = init_window.shape[1] - self.window_size
        return init_window[:, n_anchor:n_anchor + horizon]


def _emit_time_validator(scheduler, *, horizon):
    return DiffusionRolloutValidator(
        _StubDataset(),
        wrapper=_StubWrapper(),
        inference_scheduler=scheduler,
        log_steps=list(range(1, horizon + 1)),
        device=torch.device("cpu"),
        horizon=horizon,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=2,
    )


def test_window_rollout_oracle_echo_scores_zero_rmse():
    s = _OracleEchoScheduler(window_size=3)
    v = _emit_time_validator(s, horizon=3)
    results = v.run(nn.Identity(), epoch=0)
    for step in (1, 2, 3):
        assert results[f"rmse_step{step}_surface"] == 0.0


def test_window_rollout_hands_future_oracle_and_lag1_forcings():
    s = _OracleEchoScheduler(window_size=3)
    v = _emit_time_validator(s, horizon=3)
    v.run(nn.Identity(), epoch=0)
    ds = v.dataset
    t0 = v._select_ic_indices(rank=0, world_size=1)[0]

    # Oracle init window = y_{1:W} — the FUTURE frames t+1 .. t+W.
    init = s.seen["init"]
    assert init.shape[1] == 3
    for j in range(3):
        assert torch.equal(init[0, j], ds._surface[t0 + j + 1])

    # Forcing slot i is the LAG-1 conditioning at absolute step i (row
    # t + i), spanning [t, t + W + horizon - 2]; the varying-boundary
    # channel is the last one of the packed c_grid.
    c_grid = s.seen["c_grid"]
    assert c_grid.shape[1] == 3 + 3 - 1
    for i in range(c_grid.shape[1]):
        assert torch.equal(c_grid[0, i, -1:], ds._varying[t0 + i])
        assert torch.equal(s.seen["c_scalar"][0, i], ds._calendar[t0 + i])


def test_window_rollout_anchor_is_the_ic_frame():
    """RSI-style init_frames = W + 1: the extra leading frame is y_0 = the
    IC itself, and the window above it is still the future y_{1:W}."""
    s = _OracleEchoScheduler(window_size=3, init_frames=4)
    v = _emit_time_validator(s, horizon=3)
    results = v.run(nn.Identity(), epoch=0)
    ds = v.dataset
    t0 = v._select_ic_indices(rank=0, world_size=1)[0]
    init = s.seen["init"]
    assert init.shape[1] == 4
    for j in range(4):
        assert torch.equal(init[0, j], ds._surface[t0 + j])
    for step in (1, 2, 3):
        assert results[f"rmse_step{step}_surface"] == 0.0


def test_window_rollout_nocean_lookahead_frame_is_own_time():
    """With predicted ocean channels the trajectory carries ONE extra frame
    so the final roll's imposition (forcing window shifted +1) reaches the
    last state's own time, t + W + horizon - 1."""
    s = _OracleEchoScheduler(window_size=3, nocean=1)
    v = _emit_time_validator(s, horizon=3)
    v.run(nn.Identity(), epoch=0)
    ds = v.dataset
    t0 = v._select_ic_indices(rank=0, world_size=1)[0]
    c_grid = s.seen["c_grid"]
    assert c_grid.shape[1] == 3 + 3 - 1 + 1
    assert torch.equal(c_grid[0, -1, -1:], ds._varying[t0 + 3 + 3 - 1])
