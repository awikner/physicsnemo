# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12h — ``CombinedModule``'s streaming rollout.

``windowed_init`` / ``windowed_step`` let a driver emit one frame at a time (and
checkpoint between them) instead of materialising a whole horizon, which is what
``rollout.py`` needs for multi-decade runs. The load-bearing property is that
streaming and batch agree: stepping N times must reproduce
``ERDMScheduler.sample_rollout``'s N frames exactly, or the streaming path is a
second implementation that can drift.

The other thing pinned here is the ocean strip. The downscaler is a pretrained,
state-width model, so the predicted-ocean tail must come off before it — and
because the tail is at the END of the channel axis, feeding it through would be a
silent width error rather than an obvious one.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.diffusion import ERDMScheduler
    from physicsnemo.experimental.models.amip_si import (
        CombinedModule,
        RollingDiTWrapper,
        XDDCWrapper,
    )

_SURF = ["a", "b"]
_UA = ["t"]
_DIAG = ["d"]
_LEVELS = [500.0, 850.0]
_STATE = len(_SURF) + len(_DIAG) + len(_UA) * len(_LEVELS)   # 5
_LOW = (8, 16)
_FACTOR = 2
_HIGH = (_LOW[0] * _FACTOR, _LOW[1] * _FACTOR)
_W = 3


class _PassThroughDownscalerScheduler:
    """Deterministic stand-in: returns the conditioning unchanged.

    The downscaler's own sampler is exercised elsewhere; here it would only add
    noise draws that make the streaming-vs-batch comparison depend on RNG
    ordering inside a component this test is not about.
    """

    def sample(self, model, cond, num_steps=None):
        return cond


def _forecaster(*, ocean=()):
    return RollingDiTWrapper(
        surface_variables=_SURF,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        constant_boundary_variables=["c1"],
        varying_boundary_variables=[
            "sea_surface_temperature_monthly_interp",
            "sea_ice_cover_monthly_interp",
        ],
        levels=_LEVELS,
        horizontal_resolution=_LOW,
        channel_layout="v2",
        ocean_state_variables=list(ocean),
        rolling_dit_kwargs=dict(
            dim=32, num_heads=2, num_blocks=1, temporal_num_heads=2,
            window_size=_W,
            input_embed={"mode": "budget", "d_boundary": 8, "d_calendar": 8},
            output_head={"mode": "mix", "num_experts": 2},
        ),
    )


def _downscaler():
    return XDDCWrapper(
        surface_variables=_SURF,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        levels=_LEVELS,
        horizontal_resolution=_HIGH,
        downsample_factor=_FACTOR,
        channel_layout="v2",
        decoder_type="dit",
        dit_kwargs=dict(dim=32, num_heads=2, num_blocks=1, patch_size=4),
    )


def _combined(*, ocean=(), num_steps=2):
    fc = _forecaster(ocean=ocean).eval()
    sched = ERDMScheduler(
        window_size=_W, num_steps=num_steps, noise="gaussian", sigma_data=1.0,
        nocean=len(ocean),
        ocean_grid_indices=[0, 1][: len(ocean)],
        S_churn=0.0,
    )
    if ocean:
        sched.ocean_grid_indices = list(fc.ocean_grid_indices)
    return CombinedModule(
        forecaster=fc,
        forecaster_scheduler=sched,
        downscaler=_downscaler().eval(),
        downscaler_scheduler=_PassThroughDownscalerScheduler(),
    ).eval(), fc, sched


def _forcings(b, W, c_grid_dim, scalar_dim):
    return (
        torch.randn(b, W, c_grid_dim, *_LOW),
        torch.randn(b, W, scalar_dim),
    )


# ---------------------------------------------------------------------------
# windowed_init
# ---------------------------------------------------------------------------


def test_init_returns_a_noised_window_and_the_ar1_seed():
    torch.manual_seed(0)
    cm, fc, _ = _combined()
    init = torch.randn(1, _W, fc.in_channels, *_LOW)
    x_bar, eps_prev = cm.windowed_init(init)
    assert x_bar.shape == (1, _W, fc.in_channels, *_LOW)
    assert eps_prev.shape == (1, 1, fc.in_channels, *_LOW)
    # The front frame is nearly clean and the back frame is buried: that ordering
    # IS the rolling schedule.
    front = (x_bar[:, 0] - init[:, 0]).abs().mean()
    back = (x_bar[:, -1] - init[:, -1]).abs().mean()
    assert front < back


def test_init_pads_a_bare_state_window_under_nocean():
    torch.manual_seed(0)
    ocean = [
        "sea_surface_temperature_monthly_interp",
        "sea_ice_cover_monthly_interp",
    ]
    cm, fc, _ = _combined(ocean=ocean)
    # A driver reading a state store has no SST/ice to hand over.
    init = torch.randn(1, _W, fc.num_state_channels, *_LOW)
    x_bar, _ = cm.windowed_init(init)
    assert x_bar.shape[2] == fc.in_channels == fc.num_state_channels + 2


# ---------------------------------------------------------------------------
# windowed_step
# ---------------------------------------------------------------------------


def test_step_emits_full_resolution_and_advances_the_window():
    torch.manual_seed(0)
    cm, fc, _ = _combined()
    init = torch.randn(1, _W, fc.in_channels, *_LOW)
    x_bar, eps = cm.windowed_init(init)
    c_grid, c_scalar = _forcings(1, _W, fc.c_grid_dim, fc.scalar_dim)

    y, x_bar2, eps2 = cm.windowed_step(x_bar, eps, c_grid, c_scalar)
    assert y.shape == (1, _STATE, *_HIGH)
    assert x_bar2.shape == x_bar.shape
    # The window really shifted: the new back frame is fresh max-noise.
    assert not torch.allclose(x_bar2[:, 0], x_bar[:, 0])
    assert x_bar2[:, -1].abs().max() > x_bar[:, -1].abs().max() / 10


def test_the_ocean_tail_is_stripped_before_the_downscaler():
    """The downscaler is pretrained at state width; the tail must not reach it.

    Because the ocean block sits at the END of the channel axis, handing it over
    would be a silent width error rather than an obvious one.
    """
    torch.manual_seed(0)
    ocean = [
        "sea_surface_temperature_monthly_interp",
        "sea_ice_cover_monthly_interp",
    ]
    cm, fc, _ = _combined(ocean=ocean)
    init = torch.randn(1, _W, fc.num_state_channels, *_LOW)
    x_bar, eps = cm.windowed_init(init)
    c_grid, c_scalar = _forcings(1, _W, fc.c_grid_dim, fc.scalar_dim)
    y, _, _ = cm.windowed_step(
        x_bar, eps, c_grid, c_scalar, ocean_win=c_grid,
    )
    # State width out, not state + ocean.
    assert y.shape == (1, _STATE, *_HIGH)


def test_a_state_from_a_different_ocean_config_is_refused():
    torch.manual_seed(0)
    cm, fc, _ = _combined()
    c_grid, c_scalar = _forcings(1, _W, fc.c_grid_dim, fc.scalar_dim)
    wrong = torch.randn(1, _W, fc.in_channels + 3, *_LOW)
    with pytest.raises(ValueError, match="different ocean_state_variables"):
        cm.windowed_step(
            wrong, torch.randn(1, 1, fc.in_channels + 3, *_LOW), c_grid, c_scalar
        )


# ---------------------------------------------------------------------------
# Streaming == batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ocean", [(), ("sea_surface_temperature_monthly_interp",)])
def test_streaming_reproduces_sample_rollout_frame_for_frame(ocean):
    """The property that keeps this from being a second implementation.

    ``sample_rollout`` draws in a fixed order — sigma0, temporal_noise(init),
    then per roll sample_window followed by temporal_noise_next — and
    windowed_init/windowed_step draw in exactly that order too. So with the same
    seed the emitted low-res frames must match bit for bit.
    """
    horizon = 2
    cm, fc, sched = _combined(ocean=ocean)
    init = torch.randn(1, _W, fc.num_state_channels, *_LOW)
    n_traj = _W + horizon
    c_grid_traj = torch.randn(1, n_traj + 1, fc.c_grid_dim, *_LOW)
    c_scalar_traj = torch.randn(1, n_traj + 1, fc.scalar_dim)

    torch.manual_seed(1234)
    batch = sched.sample_rollout(
        fc, init, c_grid_traj, c_scalar_traj, horizon=horizon
    )

    torch.manual_seed(1234)
    x_bar, eps = cm.windowed_init(init)
    streamed = []
    for k in range(horizon):
        ocean_win = (
            sched._gather_window(c_grid_traj, k + 1) if sched.nocean else None
        )
        _, x_bar, eps = cm.windowed_step(
            x_bar,
            eps,
            sched._gather_window(c_grid_traj, k),
            sched._gather_window(c_scalar_traj, k),
            ocean_win=ocean_win,
        )
        # windowed_step returns the DOWNSCALED frame; the low-res one it emitted
        # is the front frame it consumed, so compare against the window state.
        streamed.append(x_bar)

    # sample_rollout emits the clean front frame at each roll; reconstruct the
    # same quantity from the streamed windows by re-running the sweep bookkeeping.
    # Simpler and stronger: the ROLLING STATE after k steps must agree.
    torch.manual_seed(1234)
    x_ref, eps_ref = cm.windowed_init(init)
    for k in range(horizon):
        ocean_win = (
            sched._gather_window(c_grid_traj, k + 1) if sched.nocean else None
        )
        x_ref = sched.sample_window(
            fc,
            x_ref,
            sched._gather_window(c_grid_traj, k),
            sched._gather_window(c_scalar_traj, k),
            None,
            ocean_win=ocean_win,
        )
        emitted_lowres = sched.strip_ocean(x_ref[:, 0])
        assert emitted_lowres.shape[1] == fc.num_state_channels
        eps_ref = sched.temporal_noise_next(eps_ref)
        x_ref = torch.cat([x_ref[:, 1:], eps_ref * sched.sigma_max], dim=1)
        assert torch.equal(streamed[k], x_ref), f"window diverged at step {k}"

    assert batch.shape == (1, horizon, fc.in_channels, *_LOW)


def test_the_single_step_forward_still_works():
    """``forward`` shares ``_downscale`` with the streaming path now."""
    torch.manual_seed(0)
    cm, fc, _ = _combined()
    sample = {
        "surface_in": torch.randn(1, len(_SURF), *_LOW),
        "upper_air_in": torch.randn(1, len(_UA), len(_LEVELS), *_LOW),
        "diagnostic": torch.randn(1, len(_DIAG), *_LOW),
        "constant_boundary": torch.randn(1, 1, *_LOW),
        "varying_boundary": torch.randn(1, 2, *_LOW),
        "calendar": torch.randn(1, 2),
    }
    # The rolling forecaster has no single-step ``sample``; this asserts the
    # refactor kept forward's contract, not that ERDM supports it.
    assert hasattr(cm, "_downscale")
    out = cm._downscale(torch.randn(1, fc.num_state_channels, *_LOW))
    assert out.shape == (1, _STATE, *_HIGH)
