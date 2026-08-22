# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""RSIScheduler unit tests — Rolling Stochastic Interpolants (Phase A).

CPU-runnable tiny-tensor tests against stub backbones, in the style of
``test_schedulers.py``. The load-bearing test here is
:func:`test_a1_reduces_to_erdm_exactly`: RSI claims ERDM as its uncoupled,
white-noise, ``beta == 1`` special case, and ablation A1 rests on that
reduction being an *implementation* identity rather than a story. It is checked
by driving one fixed linear denoiser through both schedulers' contracts and
comparing whole sampler trajectories.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.diffusion import ERDMScheduler, RSIScheduler


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _TwoHeadStub(nn.Module):
    """Backbone matching RSI's contract: ``(b,W,C,H,W) -> (b,W,2C,H,W)``.

    A 1x1 conv per head so gradients reach both — and, deliberately, the
    conditioning ``label`` is CONSUMED rather than ignored.

    The original version took ``label`` and dropped it. That made every test
    using this stub blind to the label being wrong: a NaN, a constant, or the
    wrong ``tau`` would all have produced identical, passing results. It is the
    same blindness that let a missing ``c_in`` through — an idealized stand-in
    that does not depend on what the real backbone depends on. Folding the label
    into the output makes it causally load-bearing, and ``labels_seen`` lets a
    test assert on it directly.
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        self.h1 = nn.Conv2d(channels, channels, 1)
        self.hz = nn.Conv2d(channels, channels, 1)
        self.labels_seen: list[torch.Tensor] = []
        self.input_stats: list[tuple[float, float]] = []

    def forward(self, x, label, c_grid, c_scalar):
        self.labels_seen.append(label.detach().clone())
        self.input_stats.append((float(x.detach().float().std()),
                                 float(x.detach().float().abs().max())))
        b, W = x.shape[0], x.shape[1]
        flat = x.flatten(0, 1)
        out = torch.cat([self.h1(flat), self.hz(flat)], dim=1)
        out = out.unflatten(0, (b, W))
        # A bounded, monotone function of the label, so a broken label changes
        # the output instead of vanishing.
        return out * (1.0 + 0.1 * label)[:, :, None, None, None]


class _OneHeadStub(nn.Module):
    """Emits C channels — the width mistake RSI must reject loudly."""

    def forward(self, x, label, c_grid, c_scalar):
        return torch.zeros_like(x)


def _assert_bounded(out, limit=50.0, what="rollout"):
    """Assert a trajectory stayed on-scale, not merely that it is a number.

    ``torch.isfinite`` is not a quality gate: it passes at 1e30, and it passed
    throughout the preconditioning bug, where the emitted frame at lead W
    reached 4.65e5 on real data (3.7e2 with a zero-init net) against
    unit-variance inputs. Every rollout assertion in this suite was
    finite-only, which is exactly why none of them noticed.

    Inputs here are unit-variance, and the measured maximum across every shipped
    configuration is ~9.4 with a random stub, so 50x leaves wide headroom while
    still catching a divergence by orders of magnitude.
    """
    m = float(out.abs().max())
    assert torch.isfinite(out).all(), f"{what} produced non-finite values"
    assert m < limit, (
        f"{what} reached |{m:.3e}| against unit-variance data (limit {limit}). "
        f"A rollout that grows by orders of magnitude is the preconditioning "
        f"failure mode; finiteness alone would not have caught it."
    )


def _linear_denoiser(channels, seed=0):
    """A fixed linear map standing in for a trained denoiser D(x) = A x."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(channels, channels, generator=g) * 0.3

    def D(x):
        return torch.einsum("ij,bwjhk->bwihk", A, x)

    return D


class _ERDMLinearStub(nn.Module):
    """Presents ``D`` behind ERDM's preconditioned contract."""

    def __init__(self, sched, D):
        super().__init__()
        self.s, self.D = sched, D

    def forward(self, x_in, c_noise, c_grid, c_scalar):
        sigma = (c_noise * 4.0).exp()
        c_in, c_skip, c_out, _ = self.s.precondition(sigma)
        x = x_in / self.s.w5(c_in)
        return (self.D(x) - self.s.w5(c_skip) * x) / self.s.w5(c_out)


class _RSILinearStub(nn.Module):
    """Presents the same ``D`` behind RSI's two-head contract.

    With ``label_mode='log_sigma_eff'`` and ``reduce_to_erdm`` the label is
    ERDM's own ``log(sigma)/4``, so both stubs see identical conditioning and
    ``zhat = (x - D)/sigma`` is EDM's eps-prediction.
    """

    def __init__(self, D):
        super().__init__()
        self.D = D
        self.last_x = None

    def forward(self, x_in, label, c_grid, c_scalar):
        # Mirror both of the scheduler's preconditionings, exactly as
        # _ERDMLinearStub mirrors ERDM's — under reduce_to_erdm they are the same
        # expressions, which is what makes A1 an equivalence rather than a
        # coincidence.
        #   in : undo c_in to recover x
        #   out: emit the RAW head output F_z, since the scheduler reads the
        #        latent out as zhat = skip * x + out * F_z
        sigma = (label * 4.0).exp()[:, :, None, None, None]
        c_in = 1.0 / (sigma ** 2 + 1.0).sqrt()
        x = x_in / c_in
        self.last_x = x
        zhat = (x - self.D(x)) / sigma               # the latent this D implies
        skip = sigma / (sigma ** 2 + 1.0)
        out = 1.0 / (sigma ** 2 + 1.0).sqrt()
        f_z = (zhat - skip * x) / out
        return torch.cat([torch.zeros_like(x), f_z], dim=2)


# ---------------------------------------------------------------------------
# Driver-surface contract
# ---------------------------------------------------------------------------


def test_driver_surface_attributes():
    """The attributes validate_diffusion / inference / rollout read directly."""
    sched = RSIScheduler(window_size=6, num_steps=2)
    assert sched.window_size == 6 == sched.W
    assert sched.init_frames == 7          # W+1: slot 1 needs its anchor
    assert sched.anchor_frames == 1
    assert sched.nocean == 0
    assert sched.num_steps == 2
    for name in ("compute_loss", "sample_window", "sample_rollout",
                 "sample_rollout_generator", "_gather_window", "ocean_truth",
                 "append_ocean_target", "pad_state", "strip_ocean",
                 "impose_ocean", "stream_init", "stream_step"):
        assert hasattr(sched, name), f"missing driver entry point {name}"


@pytest.mark.parametrize("ctor,kwargs", [
    (ERDMScheduler, dict(window_size=3, num_steps=2)),
    (RSIScheduler, dict(window_size=3, num_steps=2)),
])
def test_rolling_schedulers_expose_window_size(ctor, kwargs):
    """Regression guard: rollout.py reads ``scheduler.window_size`` directly."""
    assert ctor(**kwargs).window_size == 3


# ---------------------------------------------------------------------------
# Schedule + profiles
# ---------------------------------------------------------------------------


def test_local_time_shift_identity():
    """tau_w(1) == tau_{w-1}(0): what makes the window shift exact."""
    sched = RSIScheduler(window_size=5)
    t0 = sched.local_time(torch.zeros(1))[0]
    t1 = sched.local_time(torch.ones(1))[0]
    torch.testing.assert_close(t1[1:], t0[:-1])
    assert t1[0].item() == pytest.approx(1.0)      # front slot exits clean
    assert t0[-1].item() == pytest.approx(0.0)     # back slot enters at the base


@pytest.mark.parametrize("beta_mode", ["linear", "power", "smoothstep"])
def test_beta_endpoints_and_derivative(beta_mode):
    sched = RSIScheduler(window_size=4, beta=beta_mode)
    tau = torch.tensor([[0.0, 0.25, 0.5, 1.0]])
    beta = sched.beta(tau)
    assert beta[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert beta[0, -1].item() == pytest.approx(1.0, abs=1e-6)
    assert torch.all(beta.diff() > 0)
    # beta_dot must agree with a finite difference away from the endpoints.
    h = 1e-4
    mid = torch.tensor([[0.3, 0.6]])
    fd = (sched.beta(mid + h) - sched.beta(mid - h)) / (2 * h)
    torch.testing.assert_close(sched.beta_dot(mid), fd, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("gamma_mode", ["geometric", "linear", "cosine"])
def test_gamma_endpoints_and_derivative(gamma_mode):
    sched = RSIScheduler(window_size=4, gamma_profile=gamma_mode,
                         gamma_0=0.5, gamma_1=0.02)
    tau = torch.tensor([[0.0, 0.5, 1.0]])
    g = sched.gamma(tau)
    assert g[0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert g[0, -1].item() == pytest.approx(0.02, abs=1e-6)
    # Gamma(0) > 0 is mandatory (degenerate couplings) and Gamma(1) > 0 is the
    # emission noise floor -- the sigma_min analogue.
    assert torch.all(g > 0)
    assert torch.all(g.diff() < 0)
    h = 1e-4
    mid = torch.tensor([[0.3, 0.6]])
    fd = (sched.gamma(mid + h) - sched.gamma(mid - h)) / (2 * h)
    torch.testing.assert_close(sched.gamma_dot(mid), fd, atol=1e-3, rtol=1e-3)


def test_interpolant_hits_its_endpoints():
    """x(0) = anchor + Gamma_0 z and x(1) = target + Gamma_1 z."""
    sched = RSIScheduler(window_size=2, gamma_0=0.5, gamma_1=0.02)
    a = torch.zeros(1, 2, 1, 4, 4)
    y = torch.ones(1, 2, 1, 4, 4)
    z = torch.ones(1, 2, 1, 4, 4)
    tau = torch.tensor([[0.0, 1.0]])
    x = sched.interpolant(a, y, tau, z)
    assert x[0, 0].mean().item() == pytest.approx(0.5, abs=1e-6)   # anchor + g0
    assert x[0, 1].mean().item() == pytest.approx(1.02, abs=1e-6)  # target + g1


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", ["state", "residual"])
def test_compute_loss_finite_and_scalar(param):
    torch.manual_seed(0)
    sched = RSIScheduler(window_size=3, num_steps=2, parameterization=param)
    model = _TwoHeadStub(channels=3)
    y = torch.randn(1, 4, 3, 8, 16)          # W+1 frames
    loss = sched.compute_loss(model, c_grid=None, c_scalar=None, y=y)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.h1.weight.grad is not None and model.hz.weight.grad is not None
    assert torch.isfinite(model.h1.weight.grad).all()


def test_compute_loss_rejects_w_frames():
    """W frames is ERDM's contract; RSI needs W+1 and must say so."""
    sched = RSIScheduler(window_size=3)
    y = torch.randn(1, 3, 3, 8, 16)
    with pytest.raises(ValueError, match=r"W\+1"):
        sched.compute_loss(_TwoHeadStub(3), None, None, y)


def test_heads_reject_single_head_backbone():
    sched = RSIScheduler(window_size=3)
    y = torch.randn(1, 4, 3, 8, 16)
    with pytest.raises(ValueError, match="num_output_heads"):
        sched.compute_loss(_OneHeadStub(), None, None, y)


def test_return_parts_matches_plain_loss():
    torch.manual_seed(0)
    sched = RSIScheduler(window_size=3)
    model = _TwoHeadStub(channels=3)
    y = torch.randn(1, 4, 3, 8, 16)
    torch.manual_seed(7)
    plain = sched.compute_loss(model, None, None, y)
    torch.manual_seed(7)
    loss, ocean = sched.compute_loss(model, None, None, y, return_parts=True)
    torch.testing.assert_close(plain, loss)
    assert ocean.item() == 0.0               # inert without an ocean block


@pytest.mark.parametrize("weighting",
                         ["uniform", "midrange", "lognormal_logit", "snr_bump"])
def test_loss_weight_modes_finite_positive(weighting):
    sched = RSIScheduler(window_size=6, weighting=weighting)
    tau = sched.local_time(torch.rand(4))
    w = sched.loss_weight(tau)
    assert torch.isfinite(w).all() and (w >= 0).all()


def test_snr_bump_reduces_to_erdm_loss_weight():
    """Under the ERDM reduction omega(tau) IS lambda(sigma) f(sigma)."""
    kw = dict(window_size=6, sigma_min=0.002, sigma_max=500.0, rho=-10.0,
              P_mean=2.0, P_std=1.2)
    rsi = RSIScheduler(reduce_to_erdm=True, weighting="snr_bump", delta_std=1.0, **kw)
    erdm = ERDMScheduler(sigma_data=1.0, **kw)
    t = torch.rand(8)
    tau = rsi.local_time(t)
    torch.testing.assert_close(
        rsi.loss_weight(tau), erdm.loss_weight(erdm.sigma_schedule(t)),
        rtol=1e-5, atol=1e-8,
    )


def test_anchor_perturbation_is_off_by_default_and_on_when_asked():
    a = torch.zeros(1, 3, 2, 4, 4)
    assert torch.equal(RSIScheduler(window_size=3).perturb_anchor(a), a)
    torch.manual_seed(0)
    out = RSIScheduler(window_size=3, anchor_noise=0.5).perturb_anchor(a)
    assert out.abs().mean() > 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", ["state", "residual"])
def test_sample_rollout_shape_finite(param):
    torch.manual_seed(0)
    sched = RSIScheduler(window_size=3, num_steps=2, parameterization=param)
    model = _TwoHeadStub(channels=3).eval()
    init = torch.randn(1, 4, 3, 8, 16)       # W+1 oracle frames
    with torch.no_grad():
        out = sched.sample_rollout(model, init, None, None, horizon=5)
    assert out.shape == (1, 5, 3, 8, 16)
    _assert_bounded(out, what=f"{param} rollout")


def test_sample_rollout_rejects_w_frame_init():
    sched = RSIScheduler(window_size=3, num_steps=1)
    with pytest.raises(ValueError, match="init_frames"):
        sched.sample_rollout(_TwoHeadStub(3), torch.randn(1, 3, 3, 8, 16),
                             None, None, horizon=2)


def test_generator_matches_batch_rollout():
    """The streaming path must emit exactly what sample_rollout does."""
    model = _TwoHeadStub(channels=3).eval()
    init = torch.randn(1, 4, 3, 8, 16)
    mk = lambda: RSIScheduler(window_size=3, num_steps=2, gamma_0=0.4)
    torch.manual_seed(3)
    batch = mk().sample_rollout(model, init, None, None, horizon=4)
    torch.manual_seed(3)
    frames = [f for _, f in mk().sample_rollout_generator(
        model, init, None, None, horizon=4)]
    torch.testing.assert_close(torch.stack(frames, dim=1), batch)


def test_stream_hooks_match_batch_rollout():
    """CombinedModule/rollout.py drive the same math through stream_*."""
    model = _TwoHeadStub(channels=3).eval()
    init = torch.randn(1, 4, 3, 8, 16)
    mk = lambda: RSIScheduler(window_size=3, num_steps=2, gamma_0=0.4)
    torch.manual_seed(5)
    batch = mk().sample_rollout(model, init, None, None, horizon=4)
    torch.manual_seed(5)
    sched = mk()
    state = sched.stream_init(init)
    out = []
    with torch.no_grad():
        for _ in range(4):
            frame, state = sched.stream_step(model, state, None, None, 2)
            out.append(frame)
    torch.testing.assert_close(torch.stack(out, dim=1), batch)


def test_fresh_slot_is_anchored_not_pure_noise():
    """RSI's headline change: no slot ever starts from structureless noise."""
    sched = RSIScheduler(window_size=3, gamma_0=1e-6)   # noise ~ off
    y_hat = torch.full((1, 3, 2, 4, 4), 7.0)
    fresh = sched._fresh_slot(y_hat)
    assert fresh.shape == (1, 1, 2, 4, 4)
    assert fresh.mean().item() == pytest.approx(7.0, abs=1e-3)


def test_eps_family_zero_is_bit_identical_to_pf_ode():
    """eps_scale = 0 must leave the deterministic path untouched."""
    model = _TwoHeadStub(channels=3).eval()
    init = torch.randn(1, 4, 3, 8, 16)
    torch.manual_seed(11)
    a = RSIScheduler(window_size=3, num_steps=2, eps_scale=0.0).sample_rollout(
        model, init, None, None, horizon=3)
    torch.manual_seed(11)
    b = RSIScheduler(window_size=3, num_steps=2).sample_rollout(
        model, init, None, None, horizon=3)
    assert torch.equal(a, b)


def test_eps_family_injects_spread():
    """eps > 0 must actually disperse an otherwise deterministic rollout."""
    model = _TwoHeadStub(channels=3).eval()
    init = torch.randn(1, 4, 3, 8, 16)
    sched = RSIScheduler(window_size=3, num_steps=2, eps_scale=0.05)
    torch.manual_seed(1)
    a = sched.sample_rollout(model, init, None, None, horizon=3)
    torch.manual_seed(2)
    b = sched.sample_rollout(model, init, None, None, horizon=3)
    _assert_bounded(a, what="eps-family rollout")
    _assert_bounded(b, what="eps-family rollout")
    assert (a - b).abs().max() > 0


# ---------------------------------------------------------------------------
# A1 — the ERDM reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps,solver", [(1, "euler"), (2, "heun"),
                                              (4, "heun"), (8, "heun")])
def test_a1_reduces_to_erdm_exactly(num_steps, solver):
    """``reduce_to_erdm`` must reproduce ERDM's sweep to floating point.

    Both schedulers are driven by the SAME fixed linear denoiser, each behind
    its own contract. The default ``integrator='coeff'`` advances with exact
    coefficient increments, which for beta == 1 is precisely ERDM's
    ``dx = (x - D)/sigma dsigma`` Euler/Heun step -- so this is an equality, not
    an O(dt) agreement, and a failure means a real implementation divergence
    rather than solver error.
    """
    b, W, C, H, Wd = 2, 4, 3, 8, 16
    kw = dict(window_size=W, sigma_min=0.002, sigma_max=500.0, rho=-10.0,
              num_steps=num_steps, solver=solver)
    D = _linear_denoiser(C)
    erdm = ERDMScheduler(sigma_data=1.0, S_churn=0.0, alpha=0.0, **kw)
    rsi = RSIScheduler(reduce_to_erdm=True, parameterization="residual",
                       label_mode="log_sigma_eff", integrator="coeff",
                       eps_scale=0.0, delta_std=1.0, **kw)

    torch.manual_seed(1)
    x0 = torch.randn(b, W, C, H, Wd) * 5.0
    with torch.no_grad():
        x_erdm = erdm.sample_window(_ERDMLinearStub(erdm, D), x0.clone(),
                                    None, None, num_steps)
        x_rsi, _ = rsi.sample_window(_RSILinearStub(D), x0.clone(),
                                     None, None, num_steps)
    torch.testing.assert_close(x_rsi, x_erdm, rtol=1e-5, atol=1e-4)


def test_a1_denoised_readout_matches_the_denoiser():
    """y_hat must be D evaluated at the point the heads were last given.

    Pairing a lagged ``zhat`` with the endpoint ``Gamma`` instead is not a small
    error -- Gamma spans orders of magnitude across a step, so it rescales the
    correction outright. This pins the pairing.
    """
    b, W, C, H, Wd = 2, 4, 3, 8, 16
    D = _linear_denoiser(C)
    for final_denoise in (False, True):
        rsi = RSIScheduler(window_size=W, num_steps=2, solver="heun",
                           reduce_to_erdm=True, parameterization="residual",
                           label_mode="log_sigma_eff", delta_std=1.0,
                           final_denoise=final_denoise)
        model = _RSILinearStub(D)
        torch.manual_seed(1)
        x0 = torch.randn(b, W, C, H, Wd) * 5.0
        with torch.no_grad():
            _, y_hat = rsi.sample_window(model, x0.clone(), None, None, 2)
        torch.testing.assert_close(y_hat, D(model.last_x), rtol=1e-4, atol=1e-4)


def test_a1_loss_reduces_to_erdm_exactly():
    """``reduce_to_erdm`` must reproduce ERDM's *objective*, not just its sampler.

    This is the test that was missing. ``test_snr_bump_reduces_to_erdm_loss_weight``
    checks that omega(tau) equals lambda(sigma) f(sigma) — but matching the WEIGHT
    while the weighted QUANTITY differs is not equivalence. EDM's loss is on the
    denoiser, ``lambda f ||D - y||^2``, and since ``D - y = -Gamma (zhat - z)`` an
    eps-parameterized term carries a ``Gamma^2`` Jacobian. Regressing the raw
    ``||zhat - z||^2`` instead silently under-weights every high-Gamma slot by up
    to ``sigma_max^2`` (2.5e5 here) — the back of the window, which is precisely
    what a rolling forecast must reconstruct. Measured before the fix: identical
    skill at leads 1-3 and ~180x worse than ERDM at lead W.
    """
    b, W, C, H, Wd = 2, 4, 3, 8, 16
    kw = dict(window_size=W, sigma_min=0.002, sigma_max=500.0, rho=-10.0)
    D = _linear_denoiser(C)
    erdm = ERDMScheduler(sigma_data=1.0, alpha=0.0, **kw)
    rsi = RSIScheduler(reduce_to_erdm=True, parameterization="residual",
                       label_mode="log_sigma_eff", delta_std=1.0, **kw)

    y = torch.randn(b, W, C, H, Wd)
    # The leading frame is slot 1's anchor, which the reduction ignores entirely.
    y_ext = torch.cat([torch.randn(b, 1, C, H, Wd), y], dim=1)

    torch.manual_seed(5)
    loss_erdm = erdm.compute_loss(_ERDMLinearStub(erdm, D), None, None, y)
    torch.manual_seed(5)
    loss_rsi = rsi.compute_loss(_RSILinearStub(D), None, None, y_ext)
    torch.testing.assert_close(loss_rsi, loss_erdm, rtol=1e-4, atol=0.0)


def test_loss_terms_are_in_state_units():
    """Each head's term must carry its Jacobian to the reconstructed state."""
    b, W, C, H, Wd = 1, 3, 2, 4, 8
    y = torch.randn(b, W + 1, C, H, Wd)

    # Residual mode: the Delta term scales as (1 - beta), so at tau -> 1 (the
    # window front, beta -> 1) an error in Delta_hat barely moves the state.
    res = RSIScheduler(window_size=W, parameterization="residual", gamma_1=1e-6)
    tau_front = torch.ones(b, W)
    assert float(res.w5(1.0 - res.beta(tau_front)).abs().max()) == pytest.approx(0.0)

    # The latent term scales as Gamma, which spans orders of magnitude — so it
    # must not be weighted flat.
    tau = res.local_time(torch.zeros(b))
    g = res.gamma(tau)
    assert float(g.max() / g.min()) > 5.0, "Gamma should vary strongly across slots"
    d = torch.ones(b, W, C, H, Wd)
    scaled = res.gamma_apply(d, tau)
    per_slot = scaled.flatten(2).abs().mean(dim=2)[0]
    torch.testing.assert_close(per_slot, g[0], rtol=1e-5, atol=1e-7)


def test_backbone_sees_unit_variance_inputs():
    """What the network is actually handed — the check the stubs cannot make.

    Every other test here drives an analytic stub, which is indifferent to input
    SCALE. That is exactly how a missing ``c_in`` survived: the math was
    equivalent to ERDM's while the real backbone was being handed
    ``x = y + sigma z`` with sigma up to sigma_max = 500, hundreds of times
    outside anything an RMSNorm'd DiT is scaled for. Assert on the magnitude the
    model receives, not just on the algebra around it.
    """
    b, W, C, H, Wd = 2, 6, 3, 8, 16
    seen = {}

    class _Recorder(nn.Module):
        def forward(self, x_in, label, c_grid, c_scalar):
            seen["std"] = float(x_in.float().std())
            seen["max"] = float(x_in.float().abs().max())
            return torch.cat([torch.zeros_like(x_in)] * 2, dim=2)

    y = torch.randn(b, W + 1, C, H, Wd)          # unit-variance (z-scored) data

    # The ERDM-reduction limit is the hostile case: Gamma runs to sigma_max.
    a1 = RSIScheduler(window_size=W, reduce_to_erdm=True, sigma_data=1.0,
                      sigma_min=0.002, sigma_max=500.0, rho=-10.0,
                      parameterization="residual")
    a1.compute_loss(_Recorder(), None, None, y)
    assert seen["std"] < 3.0, f"backbone input std {seen['std']:.1f} — c_in missing?"
    assert seen["max"] < 60.0, f"backbone input max {seen['max']:.1f} — c_in missing?"

    # …and the coupled default must stay O(1) too.
    a2 = RSIScheduler(window_size=W, gamma_0=0.5, gamma_1=0.02, sigma_data=1.0)
    a2.compute_loss(_Recorder(), None, None, y)
    assert seen["std"] < 3.0


# ---------------------------------------------------------------------------
# Guards derived from the preconditioning bug — see _assert_bounded
# ---------------------------------------------------------------------------


_ROLLOUT_CONFIGS = {
    "state": dict(),
    "residual": dict(parameterization="residual"),
    "sde": dict(eps_scale=0.05),
    "spectral": dict(spectral_sharpness=2.0, grid=(8, 16)),
    # Residual + a wide Gamma is the ONLY combination in which the zero-init
    # pathology is visible, so it has to be in this table or the guard is
    # decorative. Under the state parameterization zero-init gives
    # ``y_hat = h1 = 0`` — inherently safe — while under residual it gives
    # ``y_hat = x``, the raw noisy state, whose magnitude then tracks Gamma.
    # Measured at horizon 8 with a zero-output network: WITH the output skip the
    # rollout is bounded at ~3.5-4.5 for every gamma_0 from 0.5 to 500 (that
    # scale-invariance IS the preconditioning); without it, 5.9 / 36.7 / 370.8 /
    # 3712 respectively. gamma_0=50 therefore separates pass from fail by ~10x
    # in both directions.
    "residual_wide_gamma": dict(parameterization="residual", gamma_0=50.0,
                                gamma_1=0.01),
}


@pytest.mark.parametrize("name", sorted(_ROLLOUT_CONFIGS))
def test_zero_init_rollout_stays_bounded(name):
    """A zero-init network must not produce a diverging rollout — every rung.

    Both backbones initialise to zero output by design, and what that MEANS is
    parameterization-specific: a bare latent regression turns it into "do not
    transport", so a slot injected at high Gamma is emitted still carrying it.
    That was caught for the ERDM reduction only; the coupled and spectral rungs
    had no such guard, and they are the ones actually being trained.
    """
    b, W, C, H, Wd = 1, 3, 3, 8, 16

    class _Zero(nn.Module):
        def forward(self, x, label, c_grid, c_scalar):
            return torch.cat([torch.zeros_like(x)] * 2, dim=2)

    sched = RSIScheduler(window_size=W, num_steps=2, **_ROLLOUT_CONFIGS[name])
    init = torch.randn(b, W + 1, C, H, Wd)
    with torch.no_grad():
        out = sched.sample_rollout(_Zero(), init, None, None, horizon=8)
    _assert_bounded(out, what=f"zero-init {name} rollout")


@pytest.mark.parametrize("name", sorted(_ROLLOUT_CONFIGS))
def test_backbone_input_scale_across_configs(name):
    """The backbone must see ~unit-variance inputs under every configuration.

    The original check hard-coded two configs. Input scale is a property of
    ``Gamma``, so it has to hold wherever ``Gamma`` is configurable — including
    a deliberately wide ``gamma_0``, which is what a spread-calibration sweep
    would reach for.
    """
    b, W, C, H, Wd = 1, 3, 3, 8, 16
    sched = RSIScheduler(window_size=W, num_steps=2, **_ROLLOUT_CONFIGS[name])
    model = _TwoHeadStub(C)
    sched.compute_loss(model, None, None, torch.randn(b, W + 1, C, H, Wd))
    std, mx = model.input_stats[-1]
    assert std < 3.0, f"{name}: backbone input std {std:.2f} — c_in not applied?"
    assert mx < 60.0, f"{name}: backbone input max {mx:.2f} — c_in not applied?"


@pytest.mark.parametrize("label_mode", ["tau", "log_sigma_eff"])
def test_conditioning_label_is_well_formed(label_mode):
    """The label the backbone is conditioned on must be sane and slot-distinct.

    Nothing checked this before: the stub took ``label`` and dropped it, so a
    NaN, a constant, or the wrong ``tau`` would have left every test green.
    """
    W = 6
    sched = RSIScheduler(window_size=W, label_mode=label_mode, num_steps=2)
    for t in (0.0, 0.5, 1.0):
        tau = sched.local_time(torch.full((2,), t))
        lab = sched.label(tau)
        assert lab.shape == (2, W)
        assert torch.isfinite(lab).all(), f"{label_mode}: non-finite label"
        # Slots sit at distinct points of the staircase, so their labels must
        # differ — a constant label would silently un-condition the model.
        assert lab[0].unique().numel() == W, f"{label_mode}: label not slot-distinct"
        # …and must be ordered the same way the schedule is (tau increases
        # toward the clean front; sigma_eff decreases).
        diffs = lab[0].diff()
        assert (diffs > 0).all() or (diffs < 0).all(), (
            f"{label_mode}: label is not monotone across the window"
        )
    if label_mode == "tau":
        tau = sched.local_time(torch.zeros(1))
        assert float(sched.label(tau).min()) >= 0.0
        assert float(sched.label(tau).max()) <= 1.0


def test_the_stub_actually_depends_on_its_label():
    """Meta-test: pin that the stub cannot go back to ignoring the label.

    If this fails, the ~25 tests driving ``_TwoHeadStub`` have quietly stopped
    being able to detect a broken conditioning label.
    """
    torch.manual_seed(0)
    model = _TwoHeadStub(3).eval()
    x = torch.randn(1, 3, 3, 8, 16)
    a = model(x, torch.zeros(1, 3), None, None)
    b = model(x, torch.ones(1, 3), None, None)
    assert not torch.allclose(a, b), "the stub ignores its conditioning label"


def test_a1_matches_erdm_at_zero_init():
    """The reduction must hold for an UNTRAINED network, not just algebraically.

    Both backbones are zero-init by design, and what a zero output means differs
    entirely between parameterizations: ERDM's ``F = 0`` gives ``D = c_skip x``
    (contracting — "predict zero"), while a bare latent regression makes
    ``zhat = 0`` mean *zero transport*, so a slot injected at ``sigma_max`` is
    emitted still carrying it. Measured before the fix, emitted magnitude at
    lead W: ERDM 7.0e-1 vs RSI 3.7e2. Every other A1 test drove an analytic stub
    that reproduced a real denoiser, so none of them could see it — this one
    drives the degenerate network that training actually starts from.
    """
    b, W, C, H, Wd = 1, 6, 3, 8, 16
    kw = dict(window_size=W, sigma_min=0.002, sigma_max=500.0, rho=-10.0,
              num_steps=2, solver="heun")

    class _Zero(nn.Module):
        def __init__(self, wide):
            super().__init__()
            self.wide = wide

        def forward(self, x, label, c_grid, c_scalar):
            return torch.cat([torch.zeros_like(x)] * (2 if self.wide else 1), dim=2)

    erdm = ERDMScheduler(sigma_data=1.0, S_churn=0.0, alpha=0.0, **kw)
    rsi = RSIScheduler(reduce_to_erdm=True, parameterization="residual",
                       label_mode="log_sigma_eff", delta_std=1.0,
                       sigma_data=1.0, **kw)
    truth = torch.randn(b, W + 1, C, H, Wd)
    horizon = 8
    with torch.no_grad():
        torch.manual_seed(0)
        out_e = erdm.sample_rollout(_Zero(False), truth[:, 1:], None, None, horizon)
        torch.manual_seed(0)
        out_r = rsi.sample_rollout(_Zero(True), truth, None, None, horizon)
    torch.testing.assert_close(out_r, out_e, rtol=1e-3, atol=1e-3)
    # And it must stay bounded rather than inheriting the injected sigma_max.
    assert out_r.abs().max() < 20.0, out_r.abs().max()


def test_a1_c_in_is_edms_c_in():
    """The reduction must reproduce ERDM's input scaling, not merely its math."""
    kw = dict(window_size=6, sigma_min=0.002, sigma_max=500.0, rho=-10.0)
    rsi = RSIScheduler(reduce_to_erdm=True, sigma_data=1.0, **kw)
    erdm = ERDMScheduler(sigma_data=1.0, **kw)
    t = torch.rand(8)
    torch.testing.assert_close(
        rsi.c_in(rsi.local_time(t)),
        erdm.precondition(erdm.sigma_schedule(t))[0],
        rtol=1e-6, atol=1e-8,
    )


def test_a1_forward_corruption_matches_erdm():
    """x = a + beta(y-a) + Gamma z collapses to ERDM's y + sigma eps."""
    b, W, C, H, Wd = 2, 4, 3, 8, 16
    kw = dict(window_size=W, sigma_min=0.002, sigma_max=500.0, rho=-10.0)
    rsi = RSIScheduler(reduce_to_erdm=True, delta_std=1.0, **kw)
    erdm = ERDMScheduler(sigma_data=1.0, alpha=0.0, **kw)
    t = torch.rand(b)
    tau = rsi.local_time(t)
    y = torch.randn(b, W, C, H, Wd)
    anchors = torch.randn(b, W, C, H, Wd)      # must be ignored entirely
    z = torch.randn(b, W, C, H, Wd)
    torch.testing.assert_close(
        rsi.interpolant(anchors, y, tau, z),
        y + erdm.w5(erdm.sigma_schedule(t)) * z,
        rtol=1e-5, atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    kwargs = dict(window_size=3, num_steps=2, noise="gaussian")
    a = RSIScheduler(**kwargs)
    b = RSIScheduler(**kwargs)
    b.load_state_dict(a.state_dict())
    for k, v in a.state_dict().items():
        assert torch.equal(b.state_dict()[k], v), f"{k} mismatch"


def test_spectral_gamma_needs_the_grid():
    """Gamma = g(l) h(tau,l) is grid-specific; guessing it would be wrong."""
    with pytest.raises(ValueError, match="grid"):
        RSIScheduler(window_size=3, spectral_sharpness=2.0)


def test_spectral_and_erdm_reduction_are_mutually_exclusive():
    with pytest.raises(ValueError, match="WHITE-noise limit"):
        RSIScheduler(window_size=3, reduce_to_erdm=True,
                     spectral_sharpness=2.0, grid=(16, 32))


@pytest.mark.parametrize("bad,match", [
    (dict(parameterization="delta"), "parameterization"),
    (dict(integrator="rk4"), "integrator"),
    (dict(label_mode="sigma"), "label_mode"),
    (dict(nocean=2, ocean_grid_indices=(1,)), "ocean_grid_indices"),
])
def test_constructor_rejects_bad_config(bad, match):
    with pytest.raises(ValueError, match=match):
        RSIScheduler(window_size=3, **bad)


# ---------------------------------------------------------------------------
# h1_precond="edm": the EDM skip/out readout for the H1 head (state mode)
# ---------------------------------------------------------------------------
def test_h1_precond_rejects_residual_mode():
    with pytest.raises(ValueError, match="h1_precond"):
        RSIScheduler(window_size=3, parameterization="residual",
                     h1_precond="edm")
    with pytest.raises(ValueError, match="h1_precond"):
        RSIScheduler(window_size=3, h1_precond="banana")


def test_h1_precond_zero_init_is_the_contract_map():
    """At F=0 the H1 readout must be c_skip(tau) * x — the safe EDM default.

    Raw-state readout gives h1 = 0 at zero init ("predict the zero field");
    the edm readout gives the Gaussian-optimal shrinkage of x instead, exactly
    as ERDM's D does. Under reduce_to_erdm the two coefficient sets coincide
    with EDM's, so this also pins the A1 story for the H1 head.
    """
    W, C, H, Wd = 3, 3, 8, 16

    class _Zero(nn.Module):
        def forward(self, x, label, c_grid, c_scalar):
            return torch.cat([torch.zeros_like(x)] * 2, dim=2)

    for kwargs in (dict(), dict(reduce_to_erdm=True)):
        sched = RSIScheduler(window_size=W, num_steps=2, h1_precond="edm",
                             **kwargs)
        x = torch.randn(1, W, C, H, Wd)
        tau = sched.local_time(torch.rand(1))
        h1, _ = sched.heads(_Zero(), x, tau, None, None)
        g = sched.gamma(tau)
        c_skip = sched.sigma_data ** 2 / (g ** 2 + sched.sigma_data ** 2)
        torch.testing.assert_close(h1, sched.w5(c_skip) * x)


def test_h1_precond_loss_and_grads():
    """Loss stays finite and gradients reach both heads through the readout."""
    torch.manual_seed(0)
    W, C, H, Wd = 3, 3, 8, 16
    sched = RSIScheduler(window_size=W, num_steps=2, h1_precond="edm")
    model = _TwoHeadStub(C)
    loss = sched.compute_loss(model, None, None, torch.randn(2, W + 1, C, H, Wd))
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, f"no gradient reached {name}"


def test_h1_precond_zero_init_rollout_stays_bounded():
    """The edm readout keeps the zero-init rollout on-scale (state mode)."""
    W, C, H, Wd = 3, 3, 8, 16

    class _Zero(nn.Module):
        def forward(self, x, label, c_grid, c_scalar):
            return torch.cat([torch.zeros_like(x)] * 2, dim=2)

    sched = RSIScheduler(window_size=W, num_steps=2, h1_precond="edm")
    init = torch.randn(1, W + 1, C, H, Wd)
    with torch.no_grad():
        out = sched.sample_rollout(_Zero(), init, None, None, horizon=8)
    _assert_bounded(out, what="zero-init h1_precond=edm rollout")


def test_h1_precond_denoised_is_the_readout():
    """State-mode denoised() must return the preconditioned readout itself."""
    torch.manual_seed(1)
    W, C, H, Wd = 3, 3, 8, 16
    sched = RSIScheduler(window_size=W, num_steps=2, h1_precond="edm")
    model = _TwoHeadStub(C)
    x = torch.randn(1, W, C, H, Wd)
    tau = sched.local_time(torch.rand(1))
    h1, zhat = sched.heads(model, x, tau, None, None)
    torch.testing.assert_close(sched.denoised(x, tau, h1, zhat), h1)
