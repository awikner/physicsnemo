# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8a scheduler unit tests — `compute_loss` + `sample` contract.

CPU-runnable tiny-tensor tests for the five vendored diffusion
schedulers. Each test calls the scheduler against a *stub* denoiser
network that matches the scheduler's documented model signature; this
isolates the scheduler math from any specific backbone.

The schedulers are:

* :class:`DriftScheduler` — stochastic-interpolant baseline (SI).
* :class:`DynamicInterpolant` — x-prediction stochastic interpolant (SI_X).
* :class:`ERDMScheduler` — Elucidated Rolling Diffusion Model.
* :class:`RFMScheduler` — Rolling Flow Matching.
* :class:`EDMSchedulerModule` — Karras EDM (nn.Module shim over the plain class).

Each scheduler is verified to:

* instantiate without error.
* compute a finite scalar loss with a stub model.
* run its sampler (or rollout) end-to-end and return a finite tensor of
  the expected shape.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.diffusion import (
        DriftScheduler,
        DynamicInterpolant,
        EDMSchedulerModule,
        ERDMScheduler,
        RFMScheduler,
    )


# ---------------------------------------------------------------------------
# Stub denoisers — minimal nn.Modules matching each scheduler's contract.
# Each returns a tensor with the same shape as the input "x_noised" so
# scheduler arithmetic + loss reductions can flow.
# ---------------------------------------------------------------------------


class _SingleStepStub(nn.Module):
    """Stub for single-step schedulers (Drift, DynamicInterpolant).

    Signature: ``model(x_noised, x_cond, t, c_grid, c_scalar) -> tensor``
    where ``x_noised`` is ``(b, c, h, w)``. Returns a learnable
    1x1 conv applied to ``x_noised`` so gradients flow.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_noised, x_cond, t, c_grid=None, c_scalar=None):
        return self.conv(x_noised)


class _RollingStub(nn.Module):
    """Stub for rolling schedulers (ERDM, RFM).

    Signature: ``model(x_noised, c_noise, c_grid, c_scalar) -> tensor``
    where ``x_noised`` is ``(b, W, C, H, W)``. Returns a learnable
    1x1 conv applied per-frame.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_noised, c_noise, c_grid=None, c_scalar=None):
        b, W, C, H, Wd = x_noised.shape
        x = x_noised.reshape(b * W, C, H, Wd)
        out = self.conv(x)
        return out.reshape(b, W, C, H, Wd)


class _EDMStub(nn.Module):
    """Stub for the EDM scheduler.

    EDM's ``model_forward_wrapper`` concats ``[initial_cond,
    preconditioned_x]`` along the last channel dim before calling the
    model, then expects ``model(model_in, sigma_t=c_noise, **kwargs)``
    to return a tensor with the same shape as the un-concatenated input.

    For ``ndim=2`` the data layout is ``(b, nx, ny, d)`` (channel-last).
    """

    def __init__(self, channels: int):
        super().__init__()
        # Input is concat([cond, x]) → 2C channels in; output is C channels.
        self.proj = nn.Linear(2 * channels, channels)

    def forward(self, model_in, sigma_t=None, **kwargs):
        return self.proj(model_in)


# ---------------------------------------------------------------------------
# DriftScheduler (SI)
# ---------------------------------------------------------------------------


def test_drift_scheduler_compute_loss_finite():
    torch.manual_seed(0)
    sched = DriftScheduler(num_steps=4, noise="gaussian")
    model = _SingleStepStub(channels=3)
    b, c, h, w = 2, 3, 8, 16
    x = torch.randn(b, c, h, w)
    y = torch.randn(b, c, h, w)
    loss = sched.compute_loss(model, x, c_grid=None, c_scalar=None, y=y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_drift_scheduler_sample_shape_finite():
    torch.manual_seed(0)
    sched = DriftScheduler(num_steps=4, noise="gaussian")
    model = _SingleStepStub(channels=3).eval()
    b, c, h, w = 2, 3, 8, 16
    x = torch.randn(b, c, h, w)
    with torch.no_grad():
        y_pred = sched.sample(model, x, c_grid=None, c_scalar=None)
    assert y_pred.shape == (b, c, h, w)
    assert torch.isfinite(y_pred).all()


# ---------------------------------------------------------------------------
# DynamicInterpolant (SI_X)
# ---------------------------------------------------------------------------


def test_dynamic_interpolant_compute_loss_finite():
    torch.manual_seed(0)
    sched = DynamicInterpolant(num_steps=4, noise="gaussian")
    model = _SingleStepStub(channels=3)
    b, c, h, w = 2, 3, 8, 16
    x = torch.randn(b, c, h, w)
    y = torch.randn(b, c, h, w)
    loss = sched.compute_loss(model, x, c_grid=None, c_scalar=None, y=y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_dynamic_interpolant_sample_shape_finite():
    torch.manual_seed(0)
    sched = DynamicInterpolant(num_steps=4, noise="gaussian")
    model = _SingleStepStub(channels=3).eval()
    b, c, h, w = 2, 3, 8, 16
    x = torch.randn(b, c, h, w)
    with torch.no_grad():
        result = sched.sample(model, x, c_grid=None, c_scalar=None)
    # The default ``return_model_last=True`` returns (tensor, tensor_pred).
    assert isinstance(result, tuple) and len(result) == 2
    y_int, y_model = result
    assert y_int.shape == (b, c, h, w)
    assert torch.isfinite(y_int).all()


# ---------------------------------------------------------------------------
# ERDMScheduler (rolling diffusion)
# ---------------------------------------------------------------------------


def test_erdm_scheduler_compute_loss_finite():
    torch.manual_seed(0)
    sched = ERDMScheduler(window_size=3, num_steps=2, noise="gaussian")
    model = _RollingStub(channels=3)
    b, W, C, H, Ww = 1, 3, 3, 8, 16
    y = torch.randn(b, W, C, H, Ww)
    loss = sched.compute_loss(model, c_grid=None, c_scalar=None, y=y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_erdm_scheduler_sample_rollout_shape_finite():
    torch.manual_seed(0)
    sched = ERDMScheduler(window_size=3, num_steps=2, noise="gaussian")
    model = _RollingStub(channels=3).eval()
    b, W, C, H, Ww = 1, 3, 3, 8, 16
    init_window = torch.randn(b, W, C, H, Ww)
    horizon = 5
    with torch.no_grad():
        out = sched.sample_rollout(
            model,
            init_window=init_window,
            c_grid_traj=None,
            c_scalar_traj=None,
            horizon=horizon,
        )
    assert out.shape[0] == b and out.shape[1] == horizon
    assert out.shape[2:] == (C, H, Ww), out.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# RFMScheduler (rolling flow matching)
# ---------------------------------------------------------------------------


def test_rfm_scheduler_compute_loss_finite():
    torch.manual_seed(0)
    sched = RFMScheduler(window_size=3, num_steps=2, noise="gaussian")
    model = _RollingStub(channels=3)
    b, W, C, H, Ww = 1, 3, 3, 8, 16
    y = torch.randn(b, W, C, H, Ww)
    loss = sched.compute_loss(model, c_grid=None, c_scalar=None, y=y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_rfm_scheduler_sample_rollout_shape_finite():
    torch.manual_seed(0)
    sched = RFMScheduler(window_size=3, num_steps=2, noise="gaussian")
    model = _RollingStub(channels=3).eval()
    b, W, C, H, Ww = 1, 3, 3, 8, 16
    init_window = torch.randn(b, W, C, H, Ww)
    horizon = 5
    with torch.no_grad():
        out = sched.sample_rollout(
            model,
            init_window=init_window,
            c_grid_traj=None,
            c_scalar_traj=None,
            horizon=horizon,
        )
    assert out.shape[0] == b and out.shape[1] == horizon
    assert out.shape[2:] == (C, H, Ww), out.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# EDMSchedulerModule (single-step EDM, channel-last)
# ---------------------------------------------------------------------------


def test_edm_scheduler_compute_loss_finite():
    torch.manual_seed(0)
    sched = EDMSchedulerModule(num_steps=2, ndim=2)
    model = _EDMStub(channels=3)
    b, nx, ny, d = 2, 8, 8, 3
    x = torch.randn(b, nx, ny, d)
    y = torch.randn(b, nx, ny, d)
    loss = sched.compute_loss(x=x, y=y, model=model)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_edm_scheduler_sample_shape_finite():
    torch.manual_seed(0)
    sched = EDMSchedulerModule(num_steps=2, ndim=2)
    model = _EDMStub(channels=3).eval()
    b, nx, ny, d = 2, 8, 8, 3
    initial_cond = torch.randn(b, nx, ny, d)
    with torch.no_grad():
        out = sched.sample(initial_cond=initial_cond, model=model, edm_solver="euler")
    assert out.shape == (b, nx, ny, d)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# State-dict round-trip (smoke check for to(device) consistency)
# ---------------------------------------------------------------------------


def test_all_schedulers_state_dict_roundtrip():
    """Every scheduler in the public API should round-trip its buffers."""
    for ctor, kwargs in [
        (DriftScheduler, dict(num_steps=4, noise="gaussian")),
        (DynamicInterpolant, dict(num_steps=4, noise="gaussian")),
        (ERDMScheduler, dict(window_size=3, num_steps=2, noise="gaussian")),
        (RFMScheduler, dict(window_size=3, num_steps=2, noise="gaussian")),
        (EDMSchedulerModule, dict(num_steps=2)),
    ]:
        torch.manual_seed(0)
        a = ctor(**kwargs)
        sd = a.state_dict()
        b = ctor(**kwargs)
        b.load_state_dict(sd)
        for k, v in sd.items():
            assert torch.equal(b.state_dict()[k], v), f"{type(a).__name__}: {k} mismatch"
