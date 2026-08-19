# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Spectrally shaped latent amplitude (RSI ablation A4).

The load-bearing facts pinned here:

* ``lmax = nlat // 2``, not ``nlat``. Equiangular quadrature is exact only to
  half the latitude count, and at ``lmax = nlat`` the analysis-synthesis pair is
  aliased badly enough that it is not a projection at all. Since ``Gamma`` and
  ``Gamma^{-1}`` are built from that same pair, an aliased transform breaks the
  score identity silently -- nothing raises, the numbers are just wrong.
* ``sharpness == 0`` with a unit envelope reproduces the scalar geometric
  profile EXACTLY, so turning spectral shaping on cannot move the overall noise
  level; only its distribution across scales.
* the envelope is normalized to unit band-mean, so swapping envelopes
  redistributes amplitude rather than rescaling it.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_harmonics")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.diffusion import RSIScheduler
    from physicsnemo.experimental.diffusion._spectral import SphericalSpectralFilter

_NLAT, _NLON = 16, 32
_B, _W, _C = 2, 3, 4


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


def _filter(**kw):
    kw.setdefault("gamma_0", 0.5)
    kw.setdefault("gamma_1", 0.02)
    return SphericalSpectralFilter(_NLAT, _NLON, **kw)


def _project(f, v):
    """The band-limited subspace the operator lives on."""
    b, W, C, H, Wd = v.shape
    return f.isht(f.sht(v.reshape(b * W, C, H, Wd))).reshape(b, W, C, H, Wd)


def _field():
    torch.manual_seed(0)
    return torch.randn(_B, _W, _C, _NLAT, _NLON)


# ---------------------------------------------------------------------------
# The transform pair
# ---------------------------------------------------------------------------


def test_default_bandwidth_is_half_the_latitude_count():
    f = _filter()
    assert f.l_max == _NLAT // 2


def test_the_transform_pair_is_a_projection():
    f = _filter()
    v = _project(f, _field())
    torch.testing.assert_close(_project(f, v), v, atol=1e-5, rtol=1e-4)


def test_full_bandwidth_would_not_be_a_projection():
    """Guards the default: lmax = nlat is aliased, and silently so."""
    f = _filter(lmax=_NLAT)
    v = _project(f, _field())
    assert (_project(f, v) - v).abs().max() > 1e-2


# ---------------------------------------------------------------------------
# Reduction to the scalar profile
# ---------------------------------------------------------------------------


def test_flat_envelope_and_zero_sharpness_reproduce_the_scalar_profile():
    f = _filter(sharpness=0.0)
    v = _project(f, _field())
    tau = torch.rand(_B, _W)
    scalar = 0.5 * (0.02 / 0.5) ** tau           # the geometric gamma(tau)
    torch.testing.assert_close(
        f.apply(v, tau), scalar[:, :, None, None, None] * v,
        atol=1e-5, rtol=1e-4,
    )


def test_scheduler_with_flat_spectral_matches_the_scalar_scheduler():
    """The whole corruption, not just the operator."""
    kw = dict(window_size=_W, gamma_0=0.5, gamma_1=0.02, gamma_profile="geometric")
    scal = RSIScheduler(**kw)
    spec = RSIScheduler(**kw, spectral_sharpness=0.0, grid=(_NLAT, _NLON))
    a = torch.zeros(_B, _W, _C, _NLAT, _NLON)
    y = torch.ones(_B, _W, _C, _NLAT, _NLON)
    tau = torch.rand(_B, _W)
    z = _project(spec.spectral, torch.randn(_B, _W, _C, _NLAT, _NLON))
    torch.testing.assert_close(
        spec.interpolant(a, y, tau, z), scal.interpolant(a, y, tau, z),
        atol=1e-4, rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# Shaping behaviour
# ---------------------------------------------------------------------------


def test_endpoints_hold_for_every_band():
    """h(0, l) = 1 and h(1, l) = h_1 regardless of sharpness."""
    f = _filter(sharpness=3.0)
    lo = f._coeff(torch.zeros(1))
    hi = f._coeff(torch.ones(1))
    torch.testing.assert_close(lo, torch.full_like(lo, 0.5), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(hi, torch.full_like(hi, 0.02), atol=1e-6, rtol=1e-5)


def test_sharpness_releases_high_wavenumbers_later():
    """Mid-traverse, small scales must still carry more amplitude."""
    f = _filter(sharpness=3.0)
    mid = f._coeff(torch.tensor([0.5]))[0, :, 0]      # (L,)
    assert mid[-1] > mid[0], "high-l should resolve later than low-l"
    assert torch.all(mid.diff() > 0)
    # …and with no sharpness every band is released together.
    flat = _filter(sharpness=0.0)._coeff(torch.tensor([0.5]))[0, :, 0]
    torch.testing.assert_close(flat, torch.full_like(flat, flat[0]),
                               atol=1e-6, rtol=1e-5)


def test_envelope_is_normalized_to_unit_band_mean():
    """An envelope redistributes amplitude; it must not rescale it."""
    L = _NLAT // 2
    env = torch.linspace(0.1, 5.0, L)
    f = _filter(envelope=env, sharpness=0.0)
    torch.testing.assert_close(f.g.squeeze(-1).mean(), torch.tensor(1.0),
                               atol=1e-6, rtol=1e-5)
    # With a release profile that is flat in l, the band-mean amplitude is
    # therefore identical to the unit-envelope filter at every tau. (With
    # staggered release it is NOT, and should not be: g then reweights bands
    # that are at different points in their traverse.)
    tau = torch.rand(8)
    torch.testing.assert_close(
        f.band_amplitude(tau), _filter(sharpness=0.0).band_amplitude(tau),
        atol=1e-5, rtol=1e-4,
    )


def test_envelope_must_match_the_bandwidth():
    with pytest.raises(ValueError, match="degrees"):
        _filter(envelope=torch.ones(3))


def test_grid_mismatch_is_refused():
    f = _filter()
    with pytest.raises(ValueError, match="grid"):
        f.apply(torch.randn(1, 1, 1, _NLAT + 2, _NLON), torch.rand(1, 1))


# ---------------------------------------------------------------------------
# Operator algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sharpness", [0.0, 2.0])
def test_inverse_undoes_apply_on_the_band_limited_subspace(sharpness):
    f = _filter(sharpness=sharpness)
    v = _project(f, _field())
    tau = torch.rand(_B, _W)
    torch.testing.assert_close(f.apply_inv(f.apply(v, tau), tau), v,
                               atol=1e-4, rtol=1e-3)


def test_delta_equals_the_difference_of_two_applications():
    f = _filter(sharpness=2.0)
    v = _project(f, _field())
    a, b = torch.rand(_B, _W) * 0.4, torch.rand(_B, _W) * 0.4 + 0.6
    torch.testing.assert_close(
        f.apply_delta(v, a, b), f.apply(v, b) - f.apply(v, a),
        atol=1e-5, rtol=1e-4,
    )


def test_apply_dot_matches_a_finite_difference():
    f = _filter(sharpness=2.0)
    v = _project(f, _field())
    tau = torch.full((_B, _W), 0.4)
    h = 1e-3
    fd = (f.apply(v, tau + h) - f.apply(v, tau - h)) / (2 * h)
    torch.testing.assert_close(f.apply_dot(v, tau), fd, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# End to end through the scheduler
# ---------------------------------------------------------------------------


class _TwoHeadStub(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h1 = nn.Conv2d(channels, channels, 1)
        self.hz = nn.Conv2d(channels, channels, 1)

    def forward(self, x, label, c_grid, c_scalar):
        b, W = x.shape[0], x.shape[1]
        flat = x.flatten(0, 1)
        return torch.cat([self.h1(flat), self.hz(flat)], dim=1).unflatten(0, (b, W))


def test_spectral_scheduler_trains_and_samples():
    torch.manual_seed(0)
    s = RSIScheduler(window_size=_W, num_steps=2, spectral_sharpness=2.0,
                     grid=(_NLAT, _NLON))
    model = _TwoHeadStub(_C)
    y = torch.randn(_B, _W + 1, _C, _NLAT, _NLON)
    loss = s.compute_loss(model, None, None, y)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.hz.weight.grad).all()
    with torch.no_grad():
        out = s.sample_rollout(model, y, None, None, horizon=3)
    assert out.shape == (_B, 3, _C, _NLAT, _NLON)
    _assert_bounded(out, what="spectral rollout")


def test_loss_weight_follows_the_spectral_band_amplitude():
    """omega(tau) must be driven by the actual Gamma, not a stale scalar."""
    s = RSIScheduler(window_size=_W, spectral_sharpness=2.0, grid=(_NLAT, _NLON),
                     weighting="snr_bump")
    tau = s.local_time(torch.rand(4))
    torch.testing.assert_close(s.sigma_eff(tau) * s.beta(tau).clamp(min=s.beta_floor),
                               s.spectral.band_amplitude(tau), atol=1e-5, rtol=1e-4)
