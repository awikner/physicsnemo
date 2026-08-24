# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""RSI's predicted-ocean block (the Phase 12f contract under a W+1 stack).

Mirrors ``test_erdm_ocean.py``. Two things differ from ERDM and both are the
kind of mistake that keeps every tensor the right shape:

* the training target stack is ``W+1`` frames, because ``y`` carries slot 1's
  anchor — so ``ocean_truth``'s frame-count check is parameterized rather than
  hard-coded to W, and it still has to be a check.
* the imposed value is the INTERPOLANT between the anchor-time and own-time
  truths, not truth-plus-noise. Writing ERDM's form would put the ocean block
  on a different forward process from every other channel while looking right.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.diffusion import RSIScheduler

_B, _W, _C, _H, _WD = 2, 3, 4, 8, 16
_NOCEAN = 2
_OCEAN_IDX = (1, 3)


def _assert_bounded(out, limit=50.0, what="rollout"):
    """Assert a trajectory stayed on-scale, not merely that it is a number.

    ``torch.isfinite`` passes at 1e30. It passed throughout the preconditioning
    bug, where the frame emitted at lead W reached 4.65e5 against unit-variance
    data. See test_rsi_scheduler.py for the full note.
    """
    m = float(out.abs().max())
    assert torch.isfinite(out).all(), f"{what} produced non-finite values"
    assert m < limit, (
        f"{what} reached |{m:.3e}| against unit-variance data (limit {limit})"
    )


def _sched(**kw):
    kw.setdefault("window_size", _W)
    kw.setdefault("num_steps", 2)
    kw.setdefault("nocean", _NOCEAN)
    kw.setdefault("ocean_grid_indices", _OCEAN_IDX)
    return RSIScheduler(**kw)


def _bnd(n_frames, base=0.0, c=5):
    """Boundary window whose every channel encodes (frame, channel)."""
    out = torch.zeros(_B, n_frames, c, _H, _WD)
    for f in range(n_frames):
        for ch in range(c):
            out[:, f, ch] = base + 100.0 * f + ch
    return out


def _bnd_unit(n_frames, c=5):
    """A boundary window on a REALISTIC scale (z-scored, O(1)).

    ``_bnd`` encodes ``100 * frame + channel`` so a mis-selected channel or a
    mis-shifted window shows up as a wrong NUMBER — indispensable for the
    indexing tests, and wrong for anything that measures magnitude. The imposed
    ocean channels are part of the state tensor the backbone consumes, so with
    the index-encoded fixture a stub's 1x1 conv mixes values in the hundreds
    into the state-channel predictions and the rollout grows steadily
    (11.6 -> 23.8 -> 35.7 -> 51.1 over four leads, measured). Real SST and sea
    ice arrive z-scored.
    """
    g = torch.Generator().manual_seed(11)
    return torch.randn(_B, n_frames, c, _H, _WD, generator=g)


class _TwoHeadStub(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h1 = nn.Conv2d(channels, channels, 1)
        self.hz = nn.Conv2d(channels, channels, 1)

    def forward(self, x, label, c_grid, c_scalar):
        b, W = x.shape[0], x.shape[1]
        flat = x.flatten(0, 1)
        return torch.cat([self.h1(flat), self.hz(flat)], dim=1).unflatten(0, (b, W))


# ---------------------------------------------------------------------------
# Contract plumbing
# ---------------------------------------------------------------------------


def test_nocean_zero_leaves_every_hook_a_no_op():
    s = RSIScheduler(window_size=_W)
    x = torch.randn(_B, _W, _C, _H, _WD)
    assert s.ocean_truth(_bnd(_W), (_H, _WD)) is None
    assert torch.equal(s.pad_state(x), x)
    assert torch.equal(s.strip_ocean(x), x)
    assert torch.equal(s.append_ocean_target(x, _bnd(_W)), x)
    assert torch.equal(s.impose_ocean(x, _bnd(_W), _bnd(_W)), x)


def test_index_count_must_match_nocean():
    with pytest.raises(ValueError, match="ocean_grid_indices"):
        RSIScheduler(window_size=_W, nocean=2, ocean_grid_indices=(1,))


def test_pad_and_strip_round_trip():
    s = _sched()
    bare = torch.randn(_B, _W, _C, _H, _WD)
    padded = s.pad_state(bare)
    assert padded.shape[2] == _C + _NOCEAN
    assert torch.equal(padded[:, :, -_NOCEAN:], torch.zeros_like(padded[:, :, -_NOCEAN:]))
    torch.testing.assert_close(s.strip_ocean(padded), bare)


def test_strip_ocean_accepts_a_single_frame():
    s = _sched()
    frame = torch.randn(_B, _C + _NOCEAN, _H, _WD)
    assert s.strip_ocean(frame).shape == (_B, _C, _H, _WD)


# ---------------------------------------------------------------------------
# ocean_truth
# ---------------------------------------------------------------------------


def test_ocean_truth_selects_the_configured_channels():
    s = _sched()
    bnd = _bnd(_W)
    truth = s.ocean_truth(bnd, (_H, _WD))
    assert truth.shape == (_B, _W, _NOCEAN, _H, _WD)
    for j, ch in enumerate(_OCEAN_IDX):
        torch.testing.assert_close(truth[:, :, j], bnd[:, :, ch])


def test_ocean_truth_resamples_to_the_state_grid():
    """Forcings arrive at native resolution; the state block is coarse."""
    s = _sched()
    truth = s.ocean_truth(_bnd(_W), (_H // 2, _WD // 2))
    assert truth.shape == (_B, _W, _NOCEAN, _H // 2, _WD // 2)


@pytest.mark.parametrize("expect,n", [(_W, _W + 1), (_W + 1, _W)])
def test_ocean_truth_rejects_the_wrong_frame_count(expect, n):
    """The shapes line up either way — this check is the only thing that doesn't."""
    with pytest.raises(ValueError, match="frame window"):
        _sched().ocean_truth(_bnd(n), (_H, _WD), expect=expect)


def test_ocean_truth_defaults_to_the_in_window_count():
    s = _sched()
    assert s.ocean_truth(_bnd(_W), (_H, _WD)).shape[1] == _W
    with pytest.raises(ValueError):
        s.ocean_truth(_bnd(_W + 1), (_H, _WD))


# ---------------------------------------------------------------------------
# Training target (W+1)
# ---------------------------------------------------------------------------


def test_append_ocean_target_widens_the_w_plus_one_stack():
    s = _sched()
    y = torch.randn(_B, _W + 1, _C, _H, _WD)
    bnd_ext = _bnd(_W + 1)
    out = s.append_ocean_target(y, bnd_ext)
    assert out.shape == (_B, _W + 1, _C + _NOCEAN, _H, _WD)
    torch.testing.assert_close(out[:, :, :_C], y)
    for j, ch in enumerate(_OCEAN_IDX):
        torch.testing.assert_close(out[:, :, _C + j], bnd_ext[:, :, ch])


def test_append_ocean_target_rejects_a_w_frame_boundary_stack():
    s = _sched()
    y = torch.randn(_B, _W + 1, _C, _H, _WD)
    with pytest.raises(ValueError, match="frame window"):
        s.append_ocean_target(y, _bnd(_W))


def test_append_ocean_target_needs_a_boundary():
    with pytest.raises(ValueError, match="boundary"):
        _sched().append_ocean_target(torch.randn(_B, _W + 1, _C, _H, _WD), None)


# ---------------------------------------------------------------------------
# Imposition
# ---------------------------------------------------------------------------


def test_impose_ocean_writes_the_interpolant_not_truth_plus_noise():
    """a + beta(tau_w(0)) (y - a) + Gamma(tau_w(0)) z, with Gamma -> 0 checked."""
    s = _sched(gamma_0=1e-8, gamma_1=1e-8)      # latent effectively off
    x = torch.zeros(_B, _W, _C + _NOCEAN, _H, _WD)
    bnd_curr, bnd_next = _bnd(_W, base=0.0), _bnd(_W, base=1000.0)
    out = s.impose_ocean(x, bnd_next, bnd_curr)

    tau0 = s.local_time(torch.zeros(_B))
    beta = s.beta(tau0)
    a = s.ocean_truth(bnd_curr, (_H, _WD))
    y = s.ocean_truth(bnd_next, (_H, _WD))
    expected = a + beta[:, :, None, None, None] * (y - a)
    torch.testing.assert_close(out[:, :, -_NOCEAN:], expected, atol=1e-4, rtol=1e-4)
    # The state channels must be untouched.
    torch.testing.assert_close(out[:, :, :_C], x[:, :, :_C])


def test_impose_ocean_front_slot_is_nearly_the_own_time_truth():
    """The front slot exits at tau = 1 - 1/W, so it is dominated by its target."""
    s = _sched(gamma_0=1e-8, gamma_1=1e-8, window_size=6)
    W = 6
    x = torch.zeros(_B, W, _C + _NOCEAN, _H, _WD)
    bnd_curr, bnd_next = _bnd(W, base=0.0), _bnd(W, base=1000.0)
    out = s.impose_ocean(x, bnd_next, bnd_curr)
    front, back = out[:, 0, -_NOCEAN:], out[:, -1, -_NOCEAN:]
    y_front = s.ocean_truth(bnd_next, (_H, _WD))[:, 0]
    a_back = s.ocean_truth(bnd_curr, (_H, _WD))[:, -1]
    # tau_1(0) = (W-1)/W -> 5/6 of the way to the own-time truth.
    assert (front - y_front).abs().mean() < (front - a_back).abs().mean()
    # tau_W(0) = 0 -> exactly the anchor-time truth.
    torch.testing.assert_close(back, a_back, atol=1e-4, rtol=1e-4)


def test_impose_ocean_falls_back_to_own_time_truth_without_an_anchor_window():
    s = _sched(gamma_0=1e-8, gamma_1=1e-8)
    x = torch.zeros(_B, _W, _C + _NOCEAN, _H, _WD)
    bnd_next = _bnd(_W, base=1000.0)
    out = s.impose_ocean(x, bnd_next, None)
    y = s.ocean_truth(bnd_next, (_H, _WD))
    torch.testing.assert_close(out[:, :, -_NOCEAN:], y, atol=1e-4, rtol=1e-4)


def test_impose_ocean_is_a_no_op_without_a_boundary():
    s = _sched()
    x = torch.randn(_B, _W, _C + _NOCEAN, _H, _WD)
    assert torch.equal(s.impose_ocean(x, None, None), x)


def test_impose_ocean_does_not_mutate_its_input():
    s = _sched()
    x = torch.randn(_B, _W, _C + _NOCEAN, _H, _WD)
    before = x.clone()
    s.impose_ocean(x, _bnd(_W, base=1000.0), _bnd(_W))
    torch.testing.assert_close(x, before)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_loss_splits_out_the_ocean_term_and_reaches_both_heads():
    torch.manual_seed(0)
    s = _sched()
    model = _TwoHeadStub(_C + _NOCEAN)
    y = s.append_ocean_target(torch.randn(_B, _W + 1, _C, _H, _WD), _bnd(_W + 1))
    loss, ocean = s.compute_loss(model, None, None, y, return_parts=True)
    assert torch.isfinite(loss) and torch.isfinite(ocean)
    assert ocean.item() > 0
    assert ocean.item() < loss.item()          # a fraction of the total
    loss.backward()
    assert model.h1.weight.grad.abs().sum() > 0
    assert model.hz.weight.grad.abs().sum() > 0


def test_ocean_loss_weight_scales_only_the_ocean_term():
    torch.manual_seed(0)
    model = _TwoHeadStub(_C + _NOCEAN)
    y = torch.randn(_B, _W + 1, _C, _H, _WD)
    losses = {}
    for w in (1.0, 3.0):
        s = _sched(ocean_loss_weight=w)
        yo = s.append_ocean_target(y, _bnd(_W + 1))
        torch.manual_seed(4)
        total, ocean = s.compute_loss(model, None, None, yo, return_parts=True)
        losses[w] = (total.item(), ocean.item())
    # The ocean part enters the total with the weight; reported on its own it
    # is the unweighted term, so it must NOT move.
    assert losses[1.0][1] == pytest.approx(losses[3.0][1], rel=1e-5)
    assert losses[3.0][0] > losses[1.0][0]


def test_rollout_imposes_and_strips_the_ocean_block():
    torch.manual_seed(0)
    s = _sched(gamma_0=0.3)
    model = _TwoHeadStub(_C + _NOCEAN).eval()
    init = torch.randn(_B, _W + 1, _C, _H, _WD)          # BARE state stack
    traj = _bnd_unit(_W + 8)          # magnitude test => realistic scale
    with torch.no_grad():
        out = s.sample_rollout(model, init, traj, None, horizon=4)
    assert out.shape == (_B, 4, _C + _NOCEAN, _H, _WD)   # ocean block included
    assert s.strip_ocean(out).shape == (_B, 4, _C, _H, _WD)
    # Boundedness applies to the PREDICTED state. The ocean block is prescribed
    # truth imposed from the boundary, and this fixture encodes frame/channel
    # indices as values in the hundreds so mis-indexing is visible — so it is
    # legitimately off unit scale and must not be included here.
    _assert_bounded(s.strip_ocean(out), what="ocean-run state rollout")
    assert torch.isfinite(out).all()


def test_impose_ocean_with_per_channel_scales(tmp_path):
    """Ocean imposition must work when noise_scale_path is set.

    The imposition builds its interpolant on the nocean-channel TAIL block
    alone, so the per-channel scale S must be tail-sliced for that path.
    Untested, this combination shipped broken: training never calls
    impose_ocean (sampling-only), no incr-config run reached its first
    validation, and the ocean suite never set noise_scale_path — the first
    eval of a real checkpoint died on a (3) vs (154) shape mismatch
    (Midway job 54491307, 2026-08-23).
    """
    full_c = _C + _NOCEAN
    scales = torch.linspace(0.05, 0.3, full_c)[:, None, None]
    path = tmp_path / "scales.pt"
    torch.save(scales, path)
    s = _sched(gamma_0=1.0, gamma_1=0.04, noise_scale_path=str(path))
    x = torch.zeros(_B, _W, full_c, _H, _WD)
    bnd_curr, bnd_next = _bnd(_W, base=0.0), _bnd(_W, base=1000.0)
    out = s.impose_ocean(x, bnd_next, bnd_curr)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # The tail slice must be the OCEAN channels' scales: at the back slot
    # (tau=0, beta=0) the imposed block is a + gamma_0 * S_ocean * z, so its
    # deviation from the anchor-time truth is bounded by ~5 sigma of the
    # LARGEST ocean-channel scale — feasible only if the tail slice (not the
    # head) was used.
    a_back = s.ocean_truth(bnd_curr, (_H, _WD))[:, -1]
    dev = (out[:, -1, -_NOCEAN:] - a_back).abs().max()
    s_ocean_max = float(scales[-_NOCEAN:].max())
    assert dev < 5.0 * s_ocean_max * s.gamma_0

    # A sub-width tensor that is NOT the ocean tail must raise, not
    # silently tail-slice.
    import pytest as _pytest
    bad = torch.randn(_B, _W, _NOCEAN + 1, _H, _WD)
    with _pytest.raises(ValueError, match="delta_scale"):
        s.gamma_apply(bad, s.local_time(torch.zeros(_B)))


def test_rollout_with_per_channel_scales_and_ocean(tmp_path):
    """End-to-end sample_rollout under scales+ocean — the eval-crash path."""
    full_c = _C + _NOCEAN
    scales = torch.full((full_c, 1, 1), 0.1)
    path = tmp_path / "scales.pt"
    torch.save(scales, path)
    s = _sched(gamma_0=1.0, gamma_1=0.04, noise_scale_path=str(path),
               h1_precond="edm")

    class _Zero(torch.nn.Module):
        def forward(self, x, label, c_grid, c_scalar):
            return torch.cat([torch.zeros_like(x)] * 2, dim=2)

    init = torch.randn(_B, _W + 1, _C, _H, _WD)
    c_grid = _bnd_unit(_W + 8)
    with torch.no_grad():
        out = s.sample_rollout(_Zero(), init, c_grid, None, horizon=4)
    _assert_bounded(out, what="scales+ocean rollout")
