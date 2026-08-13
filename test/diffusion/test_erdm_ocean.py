# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12f — predicted ocean channels in :class:`ERDMScheduler`.

The contract under test (upstream amip_v2 ``modules/diffusion/erdm.py``):

* ``ocean_truth`` is the *single* definition of "the true ocean field", so
  the training target and the value imposed at inference cannot drift apart.
* ``append_ocean_target`` / ``strip_ocean`` / ``pad_state`` round-trip.
* ``impose_ocean`` is total on every roll, so a predicted ocean channel never
  survives into the next window — the forecast is always forced by truth.
* ``compute_loss(return_parts=True)`` splits state from ocean.
* At ``nocean=0`` every helper is the identity and the loss is bit-identical
  to the pre-12f code path.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.diffusion import ERDMScheduler


class _RollingStub(nn.Module):
    """``model(x_noised, c_noise, c_grid, c_scalar) -> (b, W, C, H, W)``."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_noised, c_noise, c_grid=None, c_scalar=None):
        b, W, C, H, Wd = x_noised.shape
        out = self.conv(x_noised.reshape(b * W, C, H, Wd))
        return out.reshape(b, W, C, H, Wd)


def _sched(**kw):
    base = dict(window_size=3, num_steps=2, noise="gaussian", sigma_data=1.0)
    base.update(kw)
    return ERDMScheduler(**base)


def _ocean_sched(**kw):
    return _sched(nocean=2, ocean_grid_indices=[1, 3], **kw)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_nocean_requires_matching_index_count():
    with pytest.raises(ValueError, match="ocean_grid_indices"):
        _sched(nocean=2, ocean_grid_indices=[0])


def test_index_count_unchecked_when_nocean_is_zero():
    # Indices without nocean are inert, not an error: a config can carry them
    # while the ocean block itself is switched off.
    sched = _sched(nocean=0, ocean_grid_indices=[1, 3])
    assert sched.nocean == 0


# ---------------------------------------------------------------------------
# ocean_truth
# ---------------------------------------------------------------------------


def test_ocean_truth_selects_the_named_channels():
    sched = _ocean_sched()
    b, W, Cb, H, Wd = 2, 3, 5, 8, 16
    bnd = torch.randn(b, W, Cb, H, Wd)
    truth = sched.ocean_truth(bnd, (H, Wd))
    assert truth.shape == (b, W, 2, H, Wd)
    assert torch.equal(truth[:, :, 0], bnd[:, :, 1])
    assert torch.equal(truth[:, :, 1], bnd[:, :, 3])


def test_ocean_truth_resamples_a_finer_boundary_grid():
    # The real pairing: 1-degree forcings (180x360) with a 45x90 state grid.
    sched = _ocean_sched()
    bnd = torch.randn(1, 3, 5, 16, 32)
    truth = sched.ocean_truth(bnd, (8, 16))
    assert truth.shape == (1, 3, 2, 8, 16)
    assert torch.isfinite(truth).all()


def test_ocean_truth_resample_is_bilinear_not_pooling():
    # Registration matters: the predicted SST has to be co-located with the
    # state's skin_temperature, which was coarsened bilinearly. At the real 4x
    # ratio (180x360 boundary -> 45x90 state) bilinear and average pooling are
    # measurably different fields, so the choice is not cosmetic. (At exactly
    # 2x they coincide, which is why this uses 4x.)
    sched = _ocean_sched()
    bnd = torch.randn(1, 3, 5, 32, 64)
    got = sched.ocean_truth(bnd, (8, 16))
    want = torch.nn.functional.interpolate(
        bnd[:, :, [1, 3]].flatten(0, 1), size=(8, 16),
        mode="bilinear", align_corners=False,
    ).unflatten(0, (1, 3))
    assert torch.allclose(got, want)
    pooled = torch.nn.functional.avg_pool2d(
        bnd[:, :, [1, 3]].flatten(0, 1), 4
    ).unflatten(0, (1, 3))
    assert not torch.allclose(got, pooled)


def test_ocean_truth_rejects_a_wrong_length_window():
    # The one guard against the silent time-alignment bug: a caller holding
    # W+1 boundary frames must pass the shifted slice, and both shapes would
    # otherwise broadcast fine.
    sched = _ocean_sched()
    bnd = torch.randn(1, 4, 5, 8, 16)      # W+1 frames, unshifted
    with pytest.raises(ValueError, match="3-frame window"):
        sched.ocean_truth(bnd, (8, 16))


def test_ocean_truth_is_none_without_ocean_channels():
    assert _sched().ocean_truth(torch.randn(1, 3, 5, 8, 16), (8, 16)) is None
    assert _ocean_sched().ocean_truth(None, (8, 16)) is None


# ---------------------------------------------------------------------------
# append / pad / strip
# ---------------------------------------------------------------------------


def test_append_ocean_target_widens_and_carries_truth():
    sched = _ocean_sched()
    y = torch.randn(2, 3, 6, 8, 16)
    bnd = torch.randn(2, 3, 5, 8, 16)
    out = sched.append_ocean_target(y, bnd)
    assert out.shape == (2, 3, 8, 8, 16)
    assert torch.equal(out[:, :, :6], y)
    assert torch.equal(out[:, :, 6], bnd[:, :, 1])
    assert torch.equal(out[:, :, 7], bnd[:, :, 3])


def test_append_ocean_target_needs_a_boundary():
    with pytest.raises(ValueError, match="append_ocean_target needs"):
        _ocean_sched().append_ocean_target(torch.randn(1, 3, 6, 8, 16), None)


def test_strip_ocean_inverts_append():
    sched = _ocean_sched()
    y = torch.randn(2, 3, 6, 8, 16)
    bnd = torch.randn(2, 3, 5, 8, 16)
    assert torch.equal(sched.strip_ocean(sched.append_ocean_target(y, bnd)), y)


def test_strip_ocean_handles_a_single_frame():
    # Rollout drivers strip the emitted frame, which has no window axis.
    sched = _ocean_sched()
    frame = torch.randn(2, 8, 8, 16)
    assert sched.strip_ocean(frame).shape == (2, 6, 8, 16)


def test_pad_state_zero_fills_then_strip_recovers():
    sched = _ocean_sched()
    x = torch.randn(2, 3, 6, 8, 16)
    padded = sched.pad_state(x)
    assert padded.shape == (2, 3, 8, 8, 16)
    assert torch.all(padded[:, :, 6:] == 0)
    assert torch.equal(sched.strip_ocean(padded), x)


def test_pad_strip_append_are_identities_without_ocean():
    sched = _sched()
    x = torch.randn(2, 3, 6, 8, 16)
    bnd = torch.randn(2, 3, 5, 8, 16)
    assert torch.equal(sched.pad_state(x), x)
    assert torch.equal(sched.strip_ocean(x), x)
    assert torch.equal(sched.append_ocean_target(x, bnd), x)


# ---------------------------------------------------------------------------
# impose_ocean
# ---------------------------------------------------------------------------


def test_impose_ocean_leaves_the_state_block_untouched():
    torch.manual_seed(0)
    sched = _ocean_sched()
    x_bar = torch.randn(2, 3, 8, 8, 16)
    bnd = torch.randn(2, 3, 5, 8, 16)
    out = sched.impose_ocean(x_bar, bnd)
    assert torch.equal(out[:, :, :6], x_bar[:, :, :6])
    assert not torch.equal(out[:, :, 6:], x_bar[:, :, 6:])


def test_impose_ocean_writes_truth_plus_schedule_matched_noise():
    torch.manual_seed(0)
    sched = _ocean_sched()
    x_bar = torch.zeros(1, 3, 8, 8, 16)
    bnd = torch.randn(1, 3, 5, 8, 16)
    truth = sched.ocean_truth(bnd, (8, 16))
    # The residual over the imposed block is w5(sigma_schedule(0)) * eps, so
    # its per-frame scale must follow the rolling schedule: back slots are
    # noisier than the front slot, by construction.
    torch.manual_seed(1234)
    out = sched.impose_ocean(x_bar, bnd)
    resid = (out[:, :, 6:] - truth).flatten(2).std(dim=-1)[0]   # (W,)
    assert resid[0] < resid[1] < resid[2]
    sigma0 = sched.sigma_schedule(torch.zeros(1))[0]
    assert torch.all(sigma0[:-1] < sigma0[1:])


def test_impose_ocean_is_a_noop_without_ocean_or_boundary():
    sched = _sched()
    x = torch.randn(1, 3, 6, 8, 16)
    assert torch.equal(sched.impose_ocean(x, torch.randn(1, 3, 5, 8, 16)), x)
    assert torch.equal(_ocean_sched().impose_ocean(x, None), x)


def test_imposition_is_total_across_a_roll():
    """A predicted ocean channel never survives into the next window.

    The point of imposition: whatever the network wrote into the ocean block
    is overwritten from truth at the top of every roll, so an emitted frame's
    ocean channels trace back to the boundary data, not to the model. Here the
    model is a constant garbage generator; if imposition ever leaked, the
    emitted ocean channels would correlate with the garbage instead of truth.
    """
    torch.manual_seed(0)
    sched = _ocean_sched(num_steps=2, S_churn=0.0)

    class _Garbage(nn.Module):
        def forward(self, x_noised, c_noise, c_grid=None, c_scalar=None):
            out = torch.zeros_like(x_noised)
            out[:, :, 6:] = 1e4          # absurd ocean prediction
            return out

    b, W, C, H, Wd = 1, 3, 6, 8, 16
    horizon = 4
    traj = torch.randn(b, horizon + W + 1, 5, H, Wd)
    with torch.no_grad():
        out = sched.sample_rollout(
            _Garbage(), init_window=torch.randn(b, W, C, H, Wd),
            c_grid_traj=traj, c_scalar_traj=None, horizon=horizon,
        )
    assert out.shape == (b, horizon, C + 2, H, Wd)
    # Emitted frame k is the front slot after roll k, imposed from the
    # boundary at absolute step k+1 (the front slot's own time).
    for k in range(horizon):
        emitted = out[:, k, 6:]
        truth = traj[:, k + 1][:, [1, 3]]
        garbage = torch.full_like(emitted, 1e4)
        assert (emitted - truth).abs().mean() < (emitted - garbage).abs().mean()


def test_rollout_pads_a_bare_state_init_window():
    torch.manual_seed(0)
    sched = _ocean_sched()
    b, W, C, H, Wd = 1, 3, 6, 8, 16
    with torch.no_grad():
        out = sched.sample_rollout(
            _RollingStub(C + 2).eval(),
            init_window=torch.randn(b, W, C, H, Wd),   # state width only
            c_grid_traj=torch.randn(b, 8, 5, H, Wd),
            c_scalar_traj=None,
            horizon=3,
        )
    assert out.shape == (b, 3, C + 2, H, Wd)
    assert torch.isfinite(out).all()


def test_rollout_generator_accepts_a_provider_ocean_window():
    torch.manual_seed(0)
    sched = _ocean_sched()
    b, W, C, H, Wd = 1, 3, 6, 8, 16
    traj = torch.randn(b, 10, 5, H, Wd)
    seen = []

    def provider(k):
        seen.append(k)
        return traj[:, k : k + W], None, traj[:, k + 1 : k + 1 + W]

    with torch.no_grad():
        frames = list(
            sched.sample_rollout_generator(
                _RollingStub(C + 2).eval(),
                init_window=torch.randn(b, W, C, H, Wd),
                c_grid_traj=None, c_scalar_traj=None, horizon=3,
                forcing_provider=provider,
            )
        )
    assert seen == [0, 1, 2]
    assert [k for k, _ in frames] == [0, 1, 2]
    assert all(f.shape == (b, C + 2, H, Wd) for _, f in frames)


def test_rollout_generator_tolerates_a_two_element_provider():
    # Pre-12f providers return (c_grid, c_scalar); with no ocean channels
    # nothing needs the third element.
    torch.manual_seed(0)
    sched = _sched()
    b, W, C, H, Wd = 1, 3, 6, 8, 16
    traj = torch.randn(b, 10, 5, H, Wd)
    with torch.no_grad():
        frames = list(
            sched.sample_rollout_generator(
                _RollingStub(C).eval(),
                init_window=torch.randn(b, W, C, H, Wd),
                c_grid_traj=None, c_scalar_traj=None, horizon=2,
                forcing_provider=lambda k: (traj[:, k : k + W], None),
            )
        )
    assert len(frames) == 2


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


def test_compute_loss_return_parts_splits_the_ocean_block():
    torch.manual_seed(0)
    sched = _ocean_sched()
    model = _RollingStub(8)
    y = torch.randn(2, 3, 8, 8, 16)
    torch.manual_seed(7)
    loss, ocean = sched.compute_loss(model, None, None, y, return_parts=True)
    assert loss.dim() == 0 and ocean.dim() == 0
    assert torch.isfinite(loss) and torch.isfinite(ocean)
    # 2 of 8 channels, and at unit weight the ocean part is a strict subset.
    assert 0.0 < float(ocean.detach()) < float(loss.detach())


def test_ocean_loss_weight_scales_only_the_ocean_part():
    torch.manual_seed(0)
    model = _RollingStub(8)
    y = torch.randn(2, 3, 8, 8, 16)

    def run(weight):
        sched = _ocean_sched(ocean_loss_weight=weight)
        torch.manual_seed(7)
        return sched.compute_loss(model, None, None, y, return_parts=True)

    l1, o1 = run(1.0)
    l3, o3 = run(3.0)
    # The reported ocean part is the UNWEIGHTED contribution (so the log is
    # comparable across runs with different weights) while the total moves.
    assert torch.allclose(o1, o3)
    assert torch.allclose(l3 - l1, 2.0 * o1, rtol=1e-5, atol=1e-6)


def test_return_parts_gives_zero_ocean_without_ocean_channels():
    torch.manual_seed(0)
    sched = _sched()
    loss, ocean = sched.compute_loss(
        _RollingStub(6), None, None, torch.randn(2, 3, 6, 8, 16),
        return_parts=True,
    )
    assert float(ocean) == 0.0
    assert torch.isfinite(loss)


def test_nocean_zero_loss_is_bit_identical_to_the_default_path():
    """Phase 12f must be inert for every existing config."""
    torch.manual_seed(0)
    model = _RollingStub(6)
    y = torch.randn(2, 3, 6, 8, 16)

    torch.manual_seed(11)
    plain = _sched().compute_loss(model, None, None, y)
    torch.manual_seed(11)
    parts, _ = _sched().compute_loss(model, None, None, y, return_parts=True)
    torch.manual_seed(11)
    with_indices = _sched(ocean_grid_indices=[1, 3]).compute_loss(
        model, None, None, y
    )
    assert torch.equal(plain, parts)
    assert torch.equal(plain, with_indices)


def test_compute_loss_gradients_reach_the_model_through_the_ocean_block():
    torch.manual_seed(0)
    sched = _ocean_sched()
    model = _RollingStub(8)
    y = torch.randn(2, 3, 8, 8, 16)
    _, ocean = sched.compute_loss(model, None, None, y, return_parts=True)
    ocean.backward()
    assert model.conv.weight.grad is not None
    assert torch.isfinite(model.conv.weight.grad).all()
    assert model.conv.weight.grad.abs().sum() > 0
