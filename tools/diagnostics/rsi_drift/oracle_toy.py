# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Oracle-head Gaussian toy for the RSI long-rollout drift diagnosis.

Question
--------
Does the RSI *formulation as implemented* (RSIScheduler + the shipped
``rsi_sstpred_e1`` recipe) collapse / under-disperse a forced climate even when
the network is PERFECT?  ERDM through the same harness is the control.

Design
------
Toy "climate": ``C`` independent channels, each a Gaussian AR(1) in anomaly
space with a prescribed seasonal mean,

    y_k[c] = mu_k[c] + a_k[c],    a_{k+1} = rho_c a_k + sqrt(1-rho_c^2) eps,
    mu_k[c] = A_c sin(2 pi k / P)                    (stationary Var(a) = 1)

so the one-step increment std is exactly S_c = sqrt(2 (1-rho_c)) up to the
O(2 pi A/P) seasonal drift -- i.e. the ``noise_scale_path`` the shipped recipe
loads is, in this toy, *exactly self-consistent* with the process.  The
seasonal mean is handed to the scheduler as ``c_grid`` = [sin, cos] of the
slot's anchor-time phase (spatially constant) and, redundantly, as
``c_scalar``; the oracle reads mu off ``c_grid`` alone by phase rotation --
exactly the information a real network gets from insolation/calendar.

Oracle heads.  Under the RSI *training* joint (true anchors a_w = y_{w-1},
latents z_w ~ N(0,I) independent) the window

    x_w = (1-beta_w) y_{w-1} + beta_w y_w + gamma_w S_c z_w,   w = 1..W

is a linear Gaussian observation of v = (y_0..y_W).  The Bayes-optimal heads
are therefore closed form (information form, per channel):

    Prec = Sigma_y^{-1} + A^T N^{-1} A,    A = [(1-beta) | beta] bidiagonal,
    N    = diag(nv_w),  nv_w = ((1-beta_w)^2 sigma_a^2 + gamma_w^2) S_c^2
    E[v|x]   = Prec^{-1} (Sigma_y^{-1} mu + A^T N^{-1} x)
    E[z_w|x] = (gamma_w S_c / nv_w) (x_w - (A E[v|x])_w)

(``sigma_a`` = train-time ``anchor_noise``; the sigma_a > 0 oracle is the
perturbed-joint oracle, which is what intervention (f) needs.)  The oracle
module then *inverts the scheduler's preconditioning* -- undo ``c_in``, and
emit the RAW heads F_1 = (E[y|x] - c_skip x)/c_out, F_z = (E[z|x] - skip x)/out
-- so that ``RSIScheduler.heads()`` returns the conditional means bit-exactly.
This mirrors ``test/diffusion/test_rsi_scheduler.py::_RSILinearStub``.

The oracle recovers the global time from ``label[:, 0]`` (slot 1's tau is never
clamped by ``time_eps``), then rebuilds every slot's tau with the scheduler's
own formula -- so gamma/beta/c_in agree to ~1e-16 and the c_in undo is exact.

ERDM oracle: xbar_w = y_w + sigma_w eps_w, D = E[y_w | xbar_{1:W}] by the same
Gaussian conditioning, presented through ``ERDMScheduler.precondition`` as
``_ERDMLinearStub`` does.

Nothing in the sampler is re-implemented: both rollouts go through the real
``sample_rollout`` / ``sample_window`` / ``_fresh_slot`` code paths.

CLI
---
    python tools/diagnostics/rsi_drift/oracle_toy.py \
        --horizon 3000 --members 64 --experiments all --out <dir>

Run from the repo root with ``.venv/bin/python``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Run the scheduler code from THIS repo, not a site-packages/editable copy.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

import torch

from physicsnemo.experimental.diffusion import ERDMScheduler, RSIScheduler

DT = torch.float64

# ── the toy process ──────────────────────────────────────────────────────────
RHOS = (0.5, 0.8, 0.95, 0.99)      # lag-1 autocorrelation per channel
AMPS = (1.0, 1.0, 1.0, 1.0)        # seasonal amplitude, in stationary-std units
PERIOD = 120                       # steps per "year"

# ── the shipped RSI recipe (conf/sampler/rsi_sstpred_e1.yaml) ────────────────
RSI_KW = dict(
    window_size=6, num_steps=2, solver="heun", integrator="coeff",
    final_denoise=False, parameterization="state", h1_precond="edm",
    beta="linear", beta_floor=1.0e-3, gamma_0=1.0, gamma_1=0.04,
    gamma_profile="geometric", delta_std=1.0, time_eps=1.0e-3,
    label_mode="tau", anchor_noise=0.0, weighting="snr_bump",
    P_mean=2.0, P_std=1.2, w_1=1.0, w_z=1.0, eps_scale=0.0,
    eps_tmin=0.0, eps_tmax=1.0, noise="gaussian", l_max=90,
    ocean_loss_weight=1.0,
)
# ── the shipped ERDM recipe (conf/sampler/erdm_v2_nochurn.yaml) ──────────────
ERDM_KW = dict(
    window_size=6, num_steps=2, sigma_min=0.002, sigma_max=500.0, rho=-10.0,
    sigma_data=1.0, P_mean=2.0, P_std=1.2, solver="heun", S_churn=0.0,
    S_tmin=10.0, S_tmax=10.0, S_noise=1.0, noise="gaussian", l_max=90,
    noise_scale_path=None, alpha=0.0, ocean_loss_weight=1.0,
)


class ToyProcess:
    """Independent AR(1) channels with a seasonal mean."""

    def __init__(self, rhos=RHOS, amps=AMPS, period=PERIOD):
        self.rho = torch.tensor(rhos, dtype=DT)
        self.amp = torch.tensor(amps, dtype=DT)
        self.P = int(period)
        self.C = len(rhos)
        # one-step increment std of the stationary AR(1) (the sigma_c artifact)
        self.S = torch.sqrt(2.0 * (1.0 - self.rho))

    # mu at absolute frame indices ------------------------------------------
    def mu(self, idx):
        idx = torch.as_tensor(idx, dtype=DT)
        return self.amp * torch.sin(2.0 * math.pi * idx[..., None] / self.P)

    def phase(self, idx):
        idx = torch.as_tensor(idx, dtype=DT)
        th = 2.0 * math.pi * idx / self.P
        return torch.stack([torch.sin(th), torch.cos(th)], dim=-1)

    def cov_y(self, n):
        """(C, n, n) AR(1) covariance of n consecutive anomalies."""
        i = torch.arange(n, dtype=DT)
        lag = (i[:, None] - i[None, :]).abs()
        return self.rho[:, None, None] ** lag[None]

    def simulate(self, n, k0, gen, grid=1):
        """Truth: returns y (n, C, grid, grid) and mu (n, C)."""
        a = torch.empty(n, self.C, grid, grid, dtype=DT)
        a[0] = torch.randn(self.C, grid, grid, generator=gen, dtype=DT)
        sd = torch.sqrt(1.0 - self.rho ** 2)[:, None, None]
        for k in range(1, n):
            a[k] = self.rho[:, None, None] * a[k - 1] + sd * torch.randn(
                self.C, grid, grid, generator=gen, dtype=DT)
        mu = self.mu(torch.arange(k0, k0 + n))
        return mu[:, :, None, None] + a, mu

    # analytic reference numbers -------------------------------------------
    def anchor_variance_deficit(self, W):
        """Var(y_{k+W} | y_{k+1}) vs the Gamma(0)=S variance actually injected.

        The fresh slot needs frame k+W as its anchor; the window's only
        near-resolved sample is slot 1 (frame k+1), a lag of W-1.  L3/L4.
        """
        lag = W - 1
        missing = 1.0 - self.rho ** (2 * lag)
        return dict(
            missing_std=missing.sqrt().tolist(),
            injected_std=self.S.tolist(),
            std_ratio=(missing.sqrt() / self.S).tolist(),
            c_to_match=((missing / self.S ** 2 - 1.0).clamp(min=0.0)).sqrt().tolist(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Oracle heads
# ──────────────────────────────────────────────────────────────────────────────
def _phase_from_cgrid(c_grid):
    """theta0 of slot 1's anchor time from the [sin, cos] c_grid, (b,)."""
    return torch.atan2(c_grid[:, 0, 0, 0, 0], c_grid[:, 0, 1, 0, 0])


def _fold(x):
    """(b, n, C, H, Ws) -> ((b*H*Ws), C, n) plus the shape needed to unfold."""
    b, n, C, H, Ws = x.shape
    return x.permute(0, 3, 4, 2, 1).reshape(b * H * Ws, C, n), (b, n, C, H, Ws)


def _unfold(v, shape):
    b, n, C, H, Ws = shape
    return v.reshape(b, H, Ws, C, n).permute(0, 4, 3, 1, 2)


class RSIOracle(torch.nn.Module):
    """Bayes-optimal RSI two-head network under the (possibly perturbed) joint.

    ``anchor_sigma`` is the TRAIN-time ``anchor_noise``: the oracle is derived
    under the perturbed joint a'_w = y_{w-1} + anchor_sigma * S z'_w, whose
    only effect on the algebra is the per-slot observation variance
    ``nv_w = ((1-beta_w)^2 anchor_sigma^2 + gamma_w^2) S_c^2`` (z and z' enter
    x only through their sum, so E[z_w|x] = (gamma_w S_c / nv_w) * residual).
    ``anchor_sigma = 0`` is the shipped recipe.
    """

    def __init__(self, sched: RSIScheduler, proc: ToyProcess, anchor_sigma=0.0,
                 head_scale=1.0, record_terms=False):
        super().__init__()
        self.s, self.p = sched, proc
        self.anchor_sigma = float(anchor_sigma)
        # head_scale multiplies the RAW H1 head F_1 only -- the toy's proxy for
        # a small relative error in the learned map (L2 / H4).  1.0 = exact.
        self.head_scale = float(head_scale)
        self.record_terms = bool(record_terms)
        self.W = sched.W
        self.Sy_inv = torch.linalg.inv(proc.cov_y(sched.W + 1))   # (C, W+1, W+1)
        self._cache = {}
        self.calls = 0
        self.terms = {}

    # ---- prepped linear algebra for a (B, W) block of local times ----------
    def _prep_tau(self, tau):
        B, W = tau.shape
        C = self.p.C
        beta = self.s.beta(tau)
        g = self.s.gamma(tau)
        S = self.p.S                                            # (C,)
        nv = (((1.0 - beta) ** 2 * self.anchor_sigma ** 2 + g ** 2)[:, None, :]
              * (S ** 2)[None, :, None])                        # (B, C, W)
        ar = torch.arange(W)
        A = torch.zeros(B, W, W + 1, dtype=DT)
        A[:, ar, ar] = 1.0 - beta
        A[:, ar, ar + 1] = beta
        Ninv = torch.diag_embed(1.0 / nv)                       # (B, C, W, W)
        prec = self.Sy_inv[None] + A.transpose(-1, -2)[:, None] @ Ninv @ A[:, None]
        return dict(tau=tau, beta=beta, g=g, nv=nv, A=A,
                    prec_inv=torch.linalg.inv(prec),
                    zc=g[:, None, :] * S[None, :, None] / nv)

    def prep_scalar(self, t):
        """Cached prep for a global time shared by the whole batch."""
        key = float(t)
        if key not in self._cache:
            w = torch.arange(1, self.W + 1, dtype=DT)
            tau = (1.0 - (w - key) / self.W)[None]              # scheduler formula
            self._cache[key] = self._prep_tau(tau)
        return self._cache[key]

    def posterior(self, x, pr, theta0):
        """E[y_{0:W}|x], E[z_{1:W}|x].  x: (N, C, W); theta0: (N,)."""
        i = torch.arange(self.W + 1, dtype=DT)
        th = theta0[:, None] + i[None, :] * (2.0 * math.pi / self.p.P)
        mu = self.p.amp[None, :, None] * torch.sin(th)[:, None, :]    # (N,C,W+1)
        rhs = (self.Sy_inv[None] @ mu[..., None])[..., 0] \
            + (pr["A"].transpose(-1, -2)[:, None]
               @ (x / pr["nv"])[..., None])[..., 0]
        ybar = (pr["prec_inv"] @ rhs[..., None])[..., 0]              # (N,C,W+1)
        resid = x - (pr["A"][:, None] @ ybar[..., None])[..., 0]
        return ybar, pr["zc"] * resid

    def forward(self, x_in, label, c_grid, c_scalar):
        self.calls += 1
        W = self.W
        lab = label.to(DT)
        t = W * lab[:, 0] - (W - 1)                       # slot 1 is never clamped
        const = bool(torch.all(t == t[0]))
        if const:
            pr = self.prep_scalar(float(t[0]))
        else:
            w = torch.arange(1, W + 1, dtype=DT)
            pr = self._prep_tau(1.0 - (w[None] - t[:, None]) / W)
        # harness self-check: the reconstructed tau must reproduce the labels
        # the scheduler actually handed us (modulo the time_eps clamp).
        lab_ref = pr["tau"].clamp(self.s.time_eps, 1.0)
        assert torch.allclose(lab, lab_ref.expand_as(lab), atol=1e-10), \
            f"tau reconstruction failed: {lab[0]} vs {lab_ref[0]}"
        g, sd = pr["g"], self.s.sigma_data                # g: (B, W)
        c_in = 1.0 / (g ** 2 + sd ** 2).sqrt()
        B = g.shape[0]
        b, Wn, C, H, Ws = x_in.shape
        rep = H * Ws
        gx = g if B == b else g.expand(b, W)
        x = x_in.to(DT) / (1.0 / (gx ** 2 + sd ** 2).sqrt())[:, :, None, None, None]
        xf, shape = _fold(x)
        theta0 = _phase_from_cgrid(c_grid.to(DT)).repeat_interleave(rep)
        if B != 1:
            pr = {k: (v.repeat_interleave(rep, dim=0) if torch.is_tensor(v) else v)
                  for k, v in pr.items()}
        ybar, zbar = self.posterior(xf, pr, theta0)
        Ey = _unfold(ybar[:, :, 1:], shape)
        Ez = _unfold(zbar, shape)
        denom = gx ** 2 + sd ** 2
        c_skip = (sd ** 2 / denom)[:, :, None, None, None]
        c_out = (gx * sd / denom.sqrt())[:, :, None, None, None]
        z_skip = (gx / denom)[:, :, None, None, None]
        z_out = (sd / denom.sqrt())[:, :, None, None, None]
        F1 = (Ey - c_skip * x) / c_out
        Fz = (Ez - z_skip * x) / z_out
        if self.record_terms:
            d = lambda v: v[:, :, :, 0, 0].std(0).tolist()
            self.terms = dict(
                t=float(t[0]), tau=pr["tau"][0].tolist(), gamma=g[0].tolist(),
                c_skip=c_skip[0, :, 0, 0, 0].tolist(),
                c_out=c_out[0, :, 0, 0, 0].tolist(),
                std_skip_term=d(c_skip * x), std_out_term=d(c_out * F1),
                std_yhat=d(Ey), std_x=d(x), std_F1=d(F1),
            )
        if self.head_scale != 1.0:
            F1 = F1 * self.head_scale
        return torch.cat([F1, Fz], dim=2).to(x_in.dtype)


class ERDMOracle(torch.nn.Module):
    """Bayes-optimal EDM denoiser D = E[y_{1:W} | xbar_{1:W}], xbar = y + sigma eps.

    Presented through ``ERDMScheduler.precondition`` exactly as
    ``test/diffusion/test_rsi_scheduler.py::_ERDMLinearStub`` does.
    """

    def __init__(self, sched: ERDMScheduler, proc: ToyProcess, head_scale=1.0):
        super().__init__()
        self.s, self.p = sched, proc
        self.head_scale = float(head_scale)
        self.W = sched.W
        self.Sy_inv = torch.linalg.inv(proc.cov_y(sched.W))     # (C, W, W)
        self._cache = {}
        self.calls = 0

    def _prep(self, sigma):
        """sigma: (B, W) -> prec_inv (B, C, W, W)."""
        if sigma.ndim == 1:
            sigma = sigma[None]
        prec = self.Sy_inv[None] + torch.diag_embed(
            (1.0 / sigma.to(DT) ** 2)[:, None, :].expand(
                sigma.shape[0], self.p.C, self.W))
        return dict(sig=sigma.to(DT), prec_inv=torch.linalg.inv(prec))

    def prep_cached(self, sigma):
        key = tuple(float(v) for v in sigma)
        if key not in self._cache:
            self._cache[key] = self._prep(sigma)
        return self._cache[key]

    def denoiser(self, x, pr, theta0):
        """E[y|xbar].  x: (N, C, W); theta0: (N,) slot-1 ANCHOR-time phase."""
        i = torch.arange(1, self.W + 1, dtype=DT)               # slot own times
        th = theta0[:, None] + i[None, :] * (2.0 * math.pi / self.p.P)
        mu = self.p.amp[None, :, None] * torch.sin(th)[:, None, :]
        rhs = (self.Sy_inv[None] @ mu[..., None])[..., 0] \
            + x / pr["sig"][:, None, :] ** 2
        return (pr["prec_inv"] @ rhs[..., None])[..., 0]

    def forward(self, x_in, c_noise, c_grid, c_scalar):
        self.calls += 1
        sigma = (c_noise.to(DT) * 4.0).exp()                     # (b, W)
        const = bool(torch.all(sigma == sigma[0]))
        pr = self.prep_cached(sigma[0]) if const else self._prep(sigma)
        c_in, c_skip, c_out, _ = self.s.precondition(sigma)
        x = x_in.to(DT) / self.s.w5(c_in.to(DT))
        xf, shape = _fold(x)
        b, Wn, C, H, Ws = x.shape
        rep = H * Ws
        theta0 = _phase_from_cgrid(c_grid.to(DT)).repeat_interleave(rep)
        prx = pr if const else {
            k: v.repeat_interleave(rep, dim=0) for k, v in pr.items()}
        D = _unfold(self.denoiser(xf, prx, theta0), shape)
        F = (D - self.s.w5(c_skip.to(DT)) * x) / self.s.w5(c_out.to(DT))
        return (F * self.head_scale).to(x_in.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented / intervened RSI schedulers
# ──────────────────────────────────────────────────────────────────────────────
class InstrumentedRSI(RSIScheduler):
    """Records the anchor chain and per-slot ODE window states; allows the
    fresh-slot interventions of E2(c)/(d)."""

    def configure_probe(self, fresh_mode="yhat", extra_c=0.0, record=True,
                        proc: ToyProcess | None = None):
        self.fresh_mode = fresh_mode
        self.extra_c = extra_c            # float or (C,) tensor
        self.record = record
        self.proc = proc
        self.rec_anchor, self.rec_slot_mean, self.rec_slot_std = [], [], []
        self._last_x = None
        return self

    def sample_window(self, model, x, c_grid_win, c_scalar_win, num_steps=None,
                      ocean_win=None):
        x_out, y_hat = super().sample_window(
            model, x, c_grid_win, c_scalar_win, num_steps, ocean_win=ocean_win)
        self._last_x = x_out
        if getattr(self, "record", False):
            self.rec_anchor.append(y_hat[:, -1].mean(dim=(-1, -2)).clone())
            sl = x_out.mean(dim=(-1, -2))            # (b, W, C)
            self.rec_slot_mean.append(sl.mean(0).clone())
            self.rec_slot_std.append(sl.std(0).clone())
        return x_out, y_hat

    def _fresh_slot(self, y_hat):
        mode = getattr(self, "fresh_mode", "yhat")
        if mode == "xstate":
            anchor = self._last_x[:, -1:]
        else:
            anchor = y_hat[:, -1:]
        tau0 = torch.zeros(anchor.shape[0], 1, device=anchor.device, dtype=DT)
        out = anchor + self.gamma_apply(self.get_noise(anchor), tau0)
        c = getattr(self, "extra_c", 0.0)
        if torch.is_tensor(c) or c:
            cc = torch.as_tensor(c, dtype=out.dtype)
            if cc.ndim == 1:
                cc = cc[None, None, :, None, None]
            s = self._scale(out)
            out = out + cc * (s if s is not None else 1.0) * self.get_noise(out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def metrics(Y, idx, proc: ToyProcess, burn=None):
    """Y: (b, T, C) member trajectories (spatial mean); idx: absolute indices."""
    b, T, C = Y.shape
    burn = (T // 5) if burn is None else burn
    Y = Y[:, burn:]
    idx = idx[burn:]
    th = 2.0 * math.pi * idx.to(DT) / proc.P
    B = torch.stack([torch.sin(th), torch.cos(th), torch.ones_like(th)], dim=1)  # (T,3)
    G = torch.linalg.pinv(B)                                     # (3,T)
    mu = proc.mu(idx)                                            # (T, C)
    Ybar = Y.mean(0)                                             # (T, C)

    # seasonal regression of the ensemble mean on [sin, cos, 1]
    coef = G @ Ybar                                              # (3, C)
    amp_fit = coef[:2].pow(2).sum(0).sqrt()                      # recovered amplitude
    # projection coefficient onto the TRUE seasonal pattern (the toy analogue of
    # the brief's spatial shrinkage fit): slope of Ybar on mu; alpha = 1 - slope
    mm = (mu - mu.mean(0))
    slope = ((Ybar - Ybar.mean(0)) * mm).sum(0) / mm.pow(2).sum(0)
    alpha_season = 1.0 - slope

    # deseasonalized member anomalies
    coefm = torch.einsum("kt,btc->bkc", G, Y)
    fit = torch.einsum("tk,bkc->btc", B, coefm)
    res = Y - fit                                                # (b, T, C)
    var_int = res.var(dim=1).mean(0)                             # vs 1.0
    var_tot = Y.var(dim=1).mean(0)                               # vs 1 + A^2/2
    var_tot_ref = 1.0 + proc.amp ** 2 / 2.0
    spread = Y.std(0).mean(0)                                    # vs 1.0
    r = res - res.mean(dim=1, keepdim=True)
    lag1 = (r[:, 1:] * r[:, :-1]).sum(1) / r.pow(2).sum(1).clamp(min=1e-300)
    nb = min(30, max(1, Y.shape[1] // 20))
    chunk = Y.shape[1] // nb
    curve = torch.stack([Y[:, i * chunk:(i + 1) * chunk].std(0).mean(0)
                         for i in range(nb)])                 # (nb, C)
    return dict(
        spread_curve=curve.tolist(), spread_curve_chunk=chunk,
        time_mean=Ybar.mean(0).tolist(),
        time_mean_err=(Ybar.mean(0) - mu.mean(0)).tolist(),
        seasonal_amp_fit=amp_fit.tolist(),
        seasonal_amp_true=proc.amp.tolist(),
        seasonal_amp_ratio=(amp_fit / proc.amp).tolist(),
        alpha_season=alpha_season.tolist(),
        var_internal_ratio=var_int.tolist(),
        var_total_ratio=(var_tot / var_tot_ref).tolist(),
        member_spread=spread.tolist(),
        member_spread_ratio=spread.tolist(),   # sigma_clim(anomaly) = 1
        lag1_autocorr=lag1.mean(0).tolist(),
        lag1_target=proc.rho.tolist(),
        burn=burn,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rollout drivers
# ──────────────────────────────────────────────────────────────────────────────
def first_nonfinite(Y):
    bad = ~torch.isfinite(Y)
    if not bool(bad.any()):
        return None
    return int(bad.any(dim=2).any(dim=0).nonzero()[0])


def onset(Y, n=240):
    """Member spread at every step over the first ``n`` steps (per channel) --
    the collapse-onset curve, which the burn-in of ``metrics`` hides."""
    return Y[:, :n].std(0).tolist()


def safe_metrics(Y, idx, proc):
    nf = first_nonfinite(Y)
    if nf is not None and nf < 200:
        return dict(nonfinite_at=nf, blew_up=True)
    m = metrics(Y if nf is None else Y[:, :nf], idx if nf is None else idx[:nf],
                proc)
    m["nonfinite_at"] = nf
    m["blew_up"] = nf is not None
    return m


def truth_control(proc, k0, T, members, seed=99):
    """metrics() applied to ``members`` INDEPENDENT truth trajectories over the
    same index range -- the finite-sample reference for every ratio (an AR(1)
    with rho=0.99 has a 100-step decorrelation time, so the raw sample variance
    of a 3000-step window is biased low by a few percent and the naive ratios
    would blame the sampler for it)."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(members, proc.C, generator=gen, dtype=DT)
    sd = torch.sqrt(1.0 - proc.rho ** 2)
    idx = torch.arange(k0 + 1, k0 + 1 + T)
    mu = proc.mu(idx)
    Y = torch.empty(members, T, proc.C, dtype=DT)
    for j in range(T):
        if j:
            a = proc.rho * a + sd * torch.randn(members, proc.C, generator=gen,
                                                dtype=DT)
        Y[:, j] = mu[j] + a
    return metrics(Y, idx, proc)


def make_forcings(proc, k0, T, W, b, grid):
    """c_grid[:, j] = [sin, cos] of ANCHOR-time phase of absolute frame k0+j.

    Mirrors the real pipeline: slot w's forcing is at its own time minus one,
    so ``c_grid_traj[:, k+w-1]`` conditions slot w at roll k.
    """
    n = T + W + 2
    idx = torch.arange(k0, k0 + n)
    ph = proc.phase(idx)                                     # (n, 2)
    c_grid = ph[None, :, :, None, None].expand(b, n, 2, grid, grid).contiguous()
    c_scalar = ((idx % proc.P).to(DT) / proc.P)[None, :, None].expand(b, n, 1).contiguous()
    return c_grid, c_scalar


def run_rsi(proc, k0, T, members, grid, seed, sched_kw=None, oracle_sigma_a=0.0,
            fresh_mode="yhat", extra_c=0.0, record=False, scale_path=None,
            head_scale=1.0):
    kw = dict(RSI_KW)
    kw.update(sched_kw or {})
    kw["noise_scale_path"] = scale_path
    sched = InstrumentedRSI(**kw).to(DT)
    sched.configure_probe(fresh_mode=fresh_mode, extra_c=extra_c, record=record,
                          proc=proc)
    W = sched.W
    oracle = RSIOracle(sched, proc, anchor_sigma=oracle_sigma_a,
                       head_scale=head_scale, record_terms=True)
    gen = torch.Generator().manual_seed(seed)
    y, _ = proc.simulate(T + W + 4, k0, gen, grid=grid)
    init = y[: W + 1][None].expand(members, -1, -1, -1, -1).contiguous()
    c_grid, c_scalar = make_forcings(proc, k0, T, W, members, grid)
    torch.manual_seed(seed + 1)
    t_start = time.time()
    Y = sched.sample_rollout(oracle, init, c_grid, c_scalar, T)
    dt = time.time() - t_start
    idx = torch.arange(k0 + 1, k0 + 1 + T)
    out = safe_metrics(Y.mean(dim=(-1, -2)), idx, proc)
    out["seconds"] = round(dt, 1)
    out["head_calls"] = oracle.calls
    out["head_terms_last"] = oracle.terms
    out["onset_spread"] = onset(Y.mean(dim=(-1, -2)))
    if record:
        out["onset_anchor_spread"] = onset(torch.stack(sched.rec_anchor, dim=1))
    if record:
        anch = torch.stack(sched.rec_anchor, dim=1)          # (b, T, C)
        # the anchor at roll k is the estimate of absolute frame k0+k+W
        out["anchor"] = safe_metrics(anch, torch.arange(k0 + W, k0 + W + T), proc)
        sm = torch.stack(sched.rec_slot_mean, dim=0)         # (T, W, C)
        ss = torch.stack(sched.rec_slot_std, dim=0)
        burn = T // 5
        out["slot_member_std"] = ss[burn:].mean(0).tolist()
        out["slot_time_std"] = sm[burn:].std(0).tolist()
    return out, Y


def run_erdm(proc, k0, T, members, grid, seed, head_scale=1.0):
    sched = ERDMScheduler(**ERDM_KW).to(DT)
    W = sched.W
    oracle = ERDMOracle(sched, proc, head_scale=head_scale)
    gen = torch.Generator().manual_seed(seed)
    y, _ = proc.simulate(T + W + 4, k0, gen, grid=grid)
    init = y[1: W + 1][None].expand(members, -1, -1, -1, -1).contiguous()
    c_grid, c_scalar = make_forcings(proc, k0, T, W, members, grid)
    torch.manual_seed(seed + 1)
    t0 = time.time()
    Y = sched.sample_rollout(oracle, init, c_grid, c_scalar, T)
    dt = time.time() - t0
    idx = torch.arange(k0 + 1, k0 + 1 + T)
    out = safe_metrics(Y.mean(dim=(-1, -2)), idx, proc)
    out["seconds"] = round(dt, 1)
    out["head_calls"] = oracle.calls
    out["onset_spread"] = onset(Y.mean(dim=(-1, -2)))
    return out, Y


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────
def _ar_window(proc, n, k0, nframes, gen):
    """(n, nframes, C, 1, 1) truth frames at absolute indices k0..k0+nframes-1."""
    ys = torch.empty(n, nframes, proc.C, 1, 1, dtype=DT)
    a = torch.randn(n, proc.C, generator=gen, dtype=DT)
    sd = torch.sqrt(1.0 - proc.rho ** 2)
    mu = proc.mu(torch.arange(k0, k0 + nframes))
    for j in range(nframes):
        if j:
            a = proc.rho * a + sd * torch.randn(n, proc.C, generator=gen, dtype=DT)
        ys[:, j, :, 0, 0] = mu[j] + a
    return ys


def validate(proc, n=4096, n_loss=1024, W=6, seed=0, scale_path=None):
    """Oracle correctness:
      (V1) sched.heads(oracle, ...) == independently computed posterior mean
      (V2) Monte-Carlo MSE of the head == analytic Bayes MSE (per slot/channel)
      (V3) compute_loss(oracle) < compute_loss(any perturbed oracle)
    for RSI (anchor_noise 0 and 1) and ERDM.
    """
    rep = {}
    gen = torch.Generator().manual_seed(seed)
    k0 = 17

    for sig_a in (0.0, 1.0):
        sched = InstrumentedRSI(**{**RSI_KW, "noise_scale_path": scale_path,
                                   "anchor_noise": sig_a}).to(DT)
        sched.configure_probe(record=False, proc=proc)
        oracle = RSIOracle(sched, proc, anchor_sigma=sig_a)
        res = {}
        ys = _ar_window(proc, n, k0, W + 1, gen)
        c_grid, c_scalar = make_forcings(proc, k0, 1, W, n, 1)
        for t_glob in (0.0, 0.25, 0.5, 11.0 / 12.0):
            tau = sched.local_time(torch.full((n,), t_glob, dtype=DT))
            anchors = sched.perturb_anchor(ys[:, :-1])
            z = torch.randn(n, W, proc.C, 1, 1, generator=gen, dtype=DT)
            x = sched.interpolant(anchors, ys[:, 1:], tau, z)
            h1, zhat = sched.heads(oracle, x, tau, c_grid[:, :W], c_scalar[:, :W])
            xf, shape = _fold(x)
            pr = oracle.prep_scalar(t_glob)
            th0 = torch.full((n,), 2 * math.pi * k0 / proc.P, dtype=DT)
            ybar, zbar = oracle.posterior(xf, pr, th0)
            mc = (h1 - ys[:, 1:]).pow(2).mean(dim=(0, 3, 4))          # (W, C)
            an = torch.diagonal(pr["prec_inv"], dim1=-2, dim2=-1)[0, :, 1:].T
            res[f"t={t_glob:.4f}"] = dict(
                max_abs_err_h1=(h1 - _unfold(ybar[:, :, 1:], shape)).abs().max().item(),
                max_abs_err_zhat=(zhat - _unfold(zbar, shape)).abs().max().item(),
                mc_mse=_r(mc), bayes_mse=_r(an), mse_ratio=_r(mc / an),
            )
        ysl = ys[:n_loss]
        cg, cs = c_grid[:n_loss, :W], c_scalar[:n_loss, :W]
        torch.manual_seed(seed)
        base = float(sched.compute_loss(oracle, cg, cs, ysl))
        pert = {}
        for name, wrap in _perturbations(oracle):
            torch.manual_seed(seed)
            pert[name] = round(float(sched.compute_loss(wrap, cg, cs, ysl)) / base, 6)
        res["loss_optimality"] = dict(oracle_loss=base, perturbed_over_oracle=pert)
        rep[f"rsi_anchor_noise={sig_a}"] = res

    # ---------- ERDM ----------
    sched = ERDMScheduler(**ERDM_KW).to(DT)
    oracle = ERDMOracle(sched, proc)
    ys = _ar_window(proc, n, k0 + 1, W, gen)
    c_grid, c_scalar = make_forcings(proc, k0, 1, W, n, 1)
    res = {}
    for t_glob in (0.0, 0.5):
        sigma = sched.sigma_schedule(torch.full((n,), t_glob, dtype=DT))
        eps = torch.randn(n, W, proc.C, 1, 1, generator=gen, dtype=DT)
        xb = ys + sched.w5(sigma) * eps
        D = sched.denoise(oracle, xb, sigma, c_grid[:, :W], c_scalar[:, :W])
        xf, shape = _fold(xb)
        pr = oracle.prep_cached(sigma[0])
        th0 = torch.full((n,), 2 * math.pi * k0 / proc.P, dtype=DT)
        Dref = _unfold(oracle.denoiser(xf, pr, th0), shape)
        mc = (D - ys).pow(2).mean(dim=(0, 3, 4))
        an = torch.diagonal(pr["prec_inv"], dim1=-2, dim2=-1)[0].T
        res[f"t={t_glob:.2f}"] = dict(
            max_abs_err_D=(D - Dref).abs().max().item(),
            sigma=_r(sigma[0]), mc_mse=_r(mc), bayes_mse=_r(an),
            mse_ratio=_r(mc / an),
        )
    ysl, cg, cs = ys[:n_loss], c_grid[:n_loss, :W], c_scalar[:n_loss, :W]
    torch.manual_seed(seed)
    base = float(sched.compute_loss(oracle, cg, cs, ysl))
    pert = {}
    for name, wrap in _perturbations(oracle, erdm=True):
        torch.manual_seed(seed)
        pert[name] = round(float(sched.compute_loss(wrap, cg, cs, ysl)) / base, 6)
    res["loss_optimality"] = dict(oracle_loss=base, perturbed_over_oracle=pert)
    rep["erdm"] = res
    return rep


def _r(t, nd=5):
    t = t.detach()
    if t.ndim == 0:
        return round(float(t), nd)
    return [_r(v, nd) for v in t]


def _perturbations(oracle, erdm=False):
    """Wrappers that degrade the oracle's raw head output; any Bayes-optimal
    oracle must have strictly lower compute_loss than all of these."""
    class _W(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.o, self.f = oracle, f

        def forward(self, *a):
            return self.f(self.o(*a))
    return [
        ("scale_head_0.5", _W(lambda o: 0.5 * o)),
        ("scale_head_0.95", _W(lambda o: 0.95 * o)),
        ("scale_head_1.05", _W(lambda o: 1.05 * o)),
        ("scale_head_2.0", _W(lambda o: 2.0 * o)),
        ("shift_head_+0.02", _W(lambda o: o + 0.02)),
    ]


def teacher_forced(proc, T=1, members=4096, seed=3, scale_path=None):
    """One roll from a TRUE W+1 window: per-slot readout ratio std(yhat_w)/
    std(y_w) and the emitted-frame error, vs the analytic posterior std."""
    sched = InstrumentedRSI(**{**RSI_KW, "noise_scale_path": scale_path}).to(DT)
    sched.configure_probe(record=False, proc=proc)
    W = sched.W
    oracle = RSIOracle(sched, proc)
    k0 = 41
    gen = torch.Generator().manual_seed(seed)
    ys = _ar_window(proc, members, k0, W + 1, gen)
    c_grid, c_scalar = make_forcings(proc, k0, T, W, members, 1)
    torch.manual_seed(seed)
    x0 = sched.warmup_window(ys)
    x1, y_hat = sched.sample_window(oracle, x0, c_grid[:, :W], c_scalar[:, :W])
    yh = y_hat[:, :, :, 0, 0]                      # (b, W, C)
    yt = ys[:, 1:, :, 0, 0]
    amp_ratio = (yh - yh.mean(0)).std(0) / (yt - yt.mean(0)).std(0)
    rmse = (yh - yt).pow(2).mean(0).sqrt()
    # analytic Bayes std at the readout point, for reference
    pr = oracle.prep_scalar(0.5)
    bayes = torch.diagonal(pr["prec_inv"], dim1=-2, dim2=-1)[0, :, 1:].sqrt().T
    return dict(
        readout_amp_ratio=amp_ratio.mean(0).tolist(),
        readout_amp_ratio_per_slot=amp_ratio.tolist(),
        emitted_rmse=rmse[0].tolist(),
        anchor_slot_rmse=rmse[-1].tolist(),
        bayes_std_at_t0p5=bayes.tolist(),
        emitted_bias=(yh - yt).mean(0)[0].tolist(),
        anchor_slot_bias=(yh - yt).mean(0)[-1].tolist(),
    )


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--ics", type=int, default=3)
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--experiments", default="all",
                    help="comma list of validate,E0,E1,E2 or 'all'")
    ap.add_argument("--e2-ics", type=int, default=1)
    ap.add_argument("--e2-only", default=None,
                    help="comma list of substrings; only matching E2 variants run")
    ap.add_argument("--period", type=int, default=PERIOD)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    out_dir = Path(args.out or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = ToyProcess(period=args.period)
    scale_path = out_dir / "sigma_c_toy.pt"
    torch.save(proc.S[:, None, None].contiguous(), scale_path)          # the noise_scale_path artifact

    want = ("validate", "E0", "E1", "E2") if args.experiments == "all" \
        else tuple(s.strip() for s in args.experiments.split(","))
    T, M, G = args.horizon, args.members, args.grid
    K0 = [0, 37, 73][: max(1, args.ics)]
    rep = dict(
        config=dict(rhos=list(RHOS), amps=list(AMPS), period=args.period,
                    S=proc.S.tolist(), horizon=T, members=M, grid=G, ics=K0,
                    rsi=RSI_KW, erdm=ERDM_KW),
        analytics=dict(anchor_variance_deficit=proc.anchor_variance_deficit(6)),
    )

    if "validate" in want:
        t0 = time.time()
        rep["validation"] = validate(proc, scale_path=str(scale_path))
        rep["validation"]["teacher_forced_rsi"] = teacher_forced(
            proc, scale_path=str(scale_path))
        rep["validation"]["seconds"] = round(time.time() - t0, 1)
        print("[validate] done", rep["validation"]["seconds"], "s")

    ctrl = {k0: truth_control(proc, k0, T, M) for k0 in K0}
    rep["truth_control"] = {f"ic{k}": v for k, v in ctrl.items()}

    def norm(m, k0):
        """Attach ratios of the finite-sample-biased statistics to the truth
        control run over the identical index range."""
        c = ctrl[k0]
        for tag in (None, "anchor"):
            d = m if tag is None else m.get(tag)
            if d is None:
                continue
            if d.get("blew_up") and "member_spread" not in d:
                continue
            for key in ("var_internal_ratio", "var_total_ratio",
                        "member_spread", "seasonal_amp_ratio"):
                d[key + "_vs_control"] = [
                    a / b for a, b in zip(d[key], c[key])]
        return m

    if "E0" in want:
        rep["E0_erdm_oracle"] = {}
        for k0 in K0:
            m, _ = run_erdm(proc, k0, T, M, G, seed=1000 + k0)
            rep["E0_erdm_oracle"][f"ic{k0}"] = norm(m, k0)
            print(f"[E0 ic{k0}] alpha={fmt(m['alpha_season'])} "
                  f"varint={fmt(m['var_internal_ratio_vs_control'])} "
                  f"spread={fmt(m['member_spread_vs_control'])} ({m['seconds']}s)")

    if "E0b" in want or "E2" in want:
        rep["E0b_erdm_head_scale"] = {}
        for hs in (0.8, 0.9, 0.95, 0.99, 0.999):
            m, _ = run_erdm(proc, K0[0], T, M, G, seed=1500, head_scale=hs)
            rep["E0b_erdm_head_scale"][f"scale{hs}"] = norm(m, K0[0])
            print(f"[E0b erdm head_scale={hs}] alpha={fmt(m['alpha_season'])} "
                  f"varint={fmt(m['var_internal_ratio_vs_control'])} "
                  f"spread={fmt(m['member_spread_vs_control'])} ({m['seconds']}s)")

    if "E1" in want:
        rep["E1_rsi_shipped"] = {}
        for k0 in K0:
            m, _ = run_rsi(proc, k0, T, M, G, seed=2000 + k0, record=True,
                           scale_path=str(scale_path))
            rep["E1_rsi_shipped"][f"ic{k0}"] = norm(m, k0)
            print(f"[E1 ic{k0}] alpha={fmt(m['alpha_season'])} "
                  f"varint={fmt(m['var_internal_ratio_vs_control'])} "
                  f"spread={fmt(m['member_spread_vs_control'])} "
                  f"anchspread={fmt(m['anchor']['member_spread_vs_control'])} "
                  f"({m['seconds']}s)")

    if "E2" in want:
        cmatch = proc.anchor_variance_deficit(6)["c_to_match"]
        variants = {
            "a_final_denoise": dict(sched_kw=dict(final_denoise=True)),
            "b_num_steps6": dict(sched_kw=dict(num_steps=6)),
            "c_anchor_xstate": dict(fresh_mode="xstate"),
            "d_anchor_plus_1S": dict(extra_c=1.0),
            "d_anchor_plus_2S": dict(extra_c=2.0),
            "d_anchor_plus_3S": dict(extra_c=3.0),
            "d_anchor_matched": dict(extra_c=torch.tensor(cmatch, dtype=DT)),
            "e_eps_scale0.003": dict(sched_kw=dict(eps_scale=0.003)),
            "e_eps_scale0.03": dict(sched_kw=dict(eps_scale=0.03)),
            "e_eps_scale0.5": dict(sched_kw=dict(eps_scale=0.5)),
            "e_eps_scale2.0": dict(sched_kw=dict(eps_scale=2.0)),
            "f_anchor_noise1": dict(sched_kw=dict(anchor_noise=1.0),
                                    oracle_sigma_a=1.0),
            "f_anchor_noise2": dict(sched_kw=dict(anchor_noise=2.0),
                                    oracle_sigma_a=2.0),
            "h_gamma0_2": dict(sched_kw=dict(gamma_0=2.0)),
            "h_gamma0_0.5": dict(sched_kw=dict(gamma_0=0.5)),
            # (i) L2/H4: a small RELATIVE error in the raw H1 head.  At the
            # anchor-producing readout c_skip = 0.63, so the head carries 37%
            # of the state and a 1% head error is a 0.37% per-roll contraction.
            "i_head_scale_0.8": dict(head_scale=0.8),
            "i_head_scale_0.9": dict(head_scale=0.9),
            "i_head_scale_0.95": dict(head_scale=0.95),
            "i_head_scale_0.99": dict(head_scale=0.99),
            "i_head_scale_0.999": dict(head_scale=0.999),
            "i_head_scale_1.01": dict(head_scale=1.01),
            # train-time anchor_noise WITH a matched inference injection
            "f2_anchor_noise1_matched": dict(sched_kw=dict(anchor_noise=1.0),
                                             oracle_sigma_a=1.0, extra_c=1.0),
            "f2_anchor_noise2_matched": dict(sched_kw=dict(anchor_noise=2.0),
                                             oracle_sigma_a=2.0, extra_c=2.0),
        }
        if args.e2_only:
            pats = [q.strip() for q in args.e2_only.split(",")]
            variants = {k: v for k, v in variants.items()
                        if any(q in k for q in pats)}
        rep["E2_interventions"] = {}
        for name, kw in variants.items():
            rep["E2_interventions"][name] = {}
            for k0 in K0[: max(1, args.e2_ics)]:
                m, _ = run_rsi(proc, k0, T, M, G, seed=3000 + k0, record=True,
                               scale_path=str(scale_path), **kw)
                rep["E2_interventions"][name][f"ic{k0}"] = norm(m, k0)
                # A blown-up rollout (eps-family at small S_c) returns only
                # {nonfinite_at, blew_up, seconds}; print nan rather than die.
                print(f"[E2 {name} ic{k0}] alpha={fmt(m.get('alpha_season'))} "
                      f"varint={fmt(m.get('var_internal_ratio_vs_control'))} "
                      f"spread={fmt(m.get('member_spread_vs_control'))} "
                      f"nf={m.get('nonfinite_at')} ({m.get('seconds')}s)")
            # Checkpoint after every variant so a late failure keeps the rest.
            (out_dir / "oracle_toy_results.json").write_text(json.dumps(rep, indent=1))

    p = out_dir / "oracle_toy_results.json"
    p.write_text(json.dumps(rep, indent=1))
    print("wrote", p)


def fmt(v):
    if v is None:
        return "[nan]"
    return "[" + " ".join(f"{x:+.3f}" for x in v) + "]"


if __name__ == "__main__":
    main()
