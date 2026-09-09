# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small TRAINED nonlinear toy that tests whether RSI drifts in its time-mean
climatology while the ERDM control does not, and whether cheap interventions
fix it. Uses the REAL schedulers (``RSIScheduler`` / ``ERDMScheduler``) and
their real ``compute_loss``; nothing here re-implements the interpolant math.

Why this exists
---------------
The RSI forecaster collapses every field's 5-year time-mean pattern ~70% toward
its global mean in free rollouts while ERDM through the identical pipeline does
not (docs/dev/context/rsi-drift-diagnosis-brief.md). The real experiment costs
tens of GPU-hours. This file builds the smallest system that has the
ingredients the drift hypotheses need --

  * a spatially structured climatological mean pattern MAINTAINED BY THE
    DYNAMICS (not an additive constant), whose amplitude dominates the
    day-to-day variability, as in the AMIP state;
  * a seasonal forcing that modulates that pattern and is handed to the network
    every step as ``c_grid`` / ``c_scalar``;
  * nonlinear, non-Gaussian, mixing dynamics whose one-step increment std S is
    a small fraction of the state std -- the regime where RSI's Gamma_0 = 1*S
    latent is far smaller than the multi-step forecast-error spread (lead L3);

-- trains RSI and ERDM on it with the shipped recipes (same backbone, same
optimizer, same seed), free-runs both for >= 5000 steps x 32 members, and
measures the brief's shrinkage diagnostics.

Inference-time interventions are implemented by (a) constructor kwargs the
shipped configs already expose (``final_denoise``, ``eps_scale``) and (b) a
pluggable fresh-slot function in a rollout loop that is validated BIT-EXACTLY
against ``RSIScheduler.sample_rollout``. Train-time anchor interventions use
the scheduler's own ``anchor_noise``, or a subclass that overrides the
``perturb_anchor`` hook only -- ``compute_loss`` itself is never re-written.

Toy system (1-D periodic ring, N sites, C = 1, tensors (b, T, 1, 1, N))
-----------------------------------------------------------------------
    m_k(x) = M p(x) (1 + 0.5 sin(2 pi k / P))            forced mean pattern
    du/dt  = -a (u - m) - b (u - m)^3 + q (u - m)^2
             + kappa Lap(u) - lam u du/dx  + sig xi      (nsub Euler substeps)

``p`` is a fixed three-harmonic ring pattern with unit spatial std. The cubic
term bounds the state, the quadratic term makes the marginals skewed and
heavy-tailed, the Burgers advection makes the dynamics nonlinear/propagating,
the Laplacian couples neighbours. The state is z-scored with a SCALAR mean/std
over space and time, exactly as the real pipeline normalizes each channel, so
normalized 0 IS the channel's global-and-time mean and the brief's shrinkage
diagnostic transfers verbatim.

Forcing handed to the network (same alignment as the real recipe: states
t+1..t+W, forcings t..t+W-1):
    c_grid[j]   = normalized m_j(x)                    (b, W, 1, 1, N)
    c_scalar[j] = [sin, cos](2 pi j / P)               (b, W, 2)
i.e. the network is shown the *complete* forced climatology at every slot --
strictly more forcing information than the real model has. If it still fails to
use it, that is informative.

Usage
-----
    .venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py --out DIR
    .venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py --out DIR --quick
Outputs ``results.json``, ``run.log`` and CSV traces (matplotlib is not
installed in this venv, so no figures are drawn).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from physicsnemo.experimental.diffusion.erdm import ERDMScheduler  # noqa: E402
from physicsnemo.experimental.diffusion.rsi import RSIScheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Shipped recipes, transcribed from the configs the evaluated run used.
# conf/sampler/rsi_sstpred_e1.yaml == the training scheduler init line (brief 1.5)
RSI_RECIPE = dict(
    window_size=6, num_steps=2, solver="heun", integrator="coeff",
    final_denoise=False, parameterization="state", h1_precond="edm",
    beta="linear", beta_floor=1.0e-3, gamma_0=1.0, gamma_1=0.04,
    gamma_profile="geometric", delta_std=1.0, sigma_data=1.0, time_eps=1.0e-3,
    label_mode="tau", anchor_noise=0.0, weighting="snr_bump", P_mean=2.0,
    P_std=1.2, w_1=1.0, w_z=1.0, eps_scale=0.0, eps_tmin=0.0, eps_tmax=1.0,
    noise="gaussian", ocean_loss_weight=1.0,
)
# conf/loss/erdm_v2.yaml for training; conf/sampler/erdm_v2_nochurn.yaml at eval
ERDM_TRAIN_RECIPE = dict(
    window_size=6, num_steps=2, sigma_min=0.002, sigma_max=500.0, rho=-10.0,
    sigma_data=1.0, P_mean=2.0, P_std=1.2, solver="heun", S_churn=1.0,
    S_tmin=0.0, S_tmax=1000.0, S_noise=1.5, noise="gaussian", alpha=0.0,
    ocean_loss_weight=1.0,
)
ERDM_EVAL_RECIPE = dict(ERDM_TRAIN_RECIPE, S_churn=0.0, S_tmin=10.0,
                        S_tmax=10.0, S_noise=1.0)


# ===========================================================================
# 1. Toy dynamics
# ===========================================================================
def ring_pattern(n):
    x = np.arange(n) / n
    p = (1.0 * np.sin(2 * np.pi * x)
         + 0.7 * np.cos(4 * np.pi * x + 0.7)
         + 0.45 * np.sin(6 * np.pi * x + 1.9))
    return (p - p.mean()) / p.std()


class ToySystem:
    """Nonlinear stochastic ring with a forced, seasonally modulated mean."""

    def __init__(self, n=32, period=180, M=3.0, a=0.12, b=0.06, q=0.2,
                 kappa=0.3, lam=0.6, sig=0.9, k0=2.5, nsub=2, dt=0.125):
        self.n, self.P = int(n), int(period)
        self.M, self.a, self.b, self.q = M, a, b, q
        self.kappa, self.lam, self.sig = kappa, lam, sig
        self.k0 = float(k0)
        self.nsub, self.dt = int(nsub), float(dt)
        self.p = ring_pattern(self.n)
        # Large-scale (low-wavenumber) stochastic forcing: weather noise is not
        # white in space, and white forcing on a diffusive ring is annihilated
        # within one model step, which would make the day-to-day variance both
        # tiny and unpredictable. env is normalized so xi has unit variance.
        kk = np.fft.rfftfreq(self.n, d=1.0 / self.n)
        self.env = np.exp(-(kk / self.k0) ** 2)
        e2 = np.r_[self.env[0] ** 2, 2 * self.env[1:-1] ** 2,
                   self.env[-1] ** 2]
        self.env_scale = math.sqrt(e2.sum() / self.n)

    def mean_field(self, k):
        """m_k(x) for integer step(s) k -> (len(k), n)."""
        f = np.sin(2.0 * np.pi * np.asarray(k, dtype=np.float64) / self.P)
        return self.M * self.p[None, :] * (1.0 + 0.5 * f.reshape(-1, 1))

    def _rhs(self, u, m):
        d = u - m
        lap = np.roll(u, 1, -1) + np.roll(u, -1, -1) - 2.0 * u
        dx = 0.5 * (np.roll(u, -1, -1) - np.roll(u, 1, -1))
        return (-self.a * d - self.b * d ** 3 + self.q * d ** 2
                + self.kappa * lap - self.lam * u * dx)

    def run(self, nsteps, seed=0, u0=None, k0=0, nens=1):
        """Free run. Returns (nens, nsteps, n) states at integer steps."""
        rng = np.random.default_rng(seed)
        u = (np.tile(self.M * self.p, (nens, 1)) if u0 is None
             else np.array(u0, dtype=np.float64).reshape(nens, self.n).copy())
        out = np.empty((nens, nsteps, self.n))
        amp = self.sig * math.sqrt(self.dt)
        for i in range(nsteps):
            m = self.mean_field([k0 + i])
            for _ in range(self.nsub):
                w = rng.normal(0, 1, u.shape)
                xi = np.fft.irfft(np.fft.rfft(w, axis=-1) * self.env,
                                  n=self.n, axis=-1) / self.env_scale
                u = u + self.dt * self._rhs(u, m) + amp * xi
            out[:, i] = u
        if not np.isfinite(out).all():
            raise RuntimeError("toy dynamics blew up -- retune")
        return out


def climatology_stats(z, phase, P):
    """Time-mean pattern, seasonal composite, deseasonalized variance.

    z: (T, N) normalized states; phase: (T,) int in [0, P)."""
    z = np.asarray(z)
    clim = z.mean(0)
    comp = np.zeros((P, z.shape[1]))
    for p in range(P):
        sel = phase == p
        comp[p] = z[sel].mean(0) if sel.any() else clim   # empty-bin guard
    seas = comp - clim[None, :]
    resid = z - comp[phase]
    return dict(
        clim=clim,
        clim_pattern_std=float(clim.std()),
        seas_amp=seas.std(0),
        seas_amp_rms=float(np.sqrt((seas.std(0) ** 2).mean())),
        resid_std=resid.std(0),
        resid_std_rms=float(np.sqrt((resid.std(0) ** 2).mean())),
        total_var=float(z.var(0).mean()),
    )


class ToyData:
    """One long training trajectory + independent reference climatology."""

    def __init__(self, sys_, n_train=200_000, n_stat=100_000, burn=2000, seed=0,
                 forcing_mode="direct"):
        t0 = time.time()
        raw = sys_.run(burn + n_train, seed=seed)[0][burn:]
        self.mu, self.sd = float(raw.mean()), float(raw.std())
        z = (raw - self.mu) / self.sd
        self.sys, self.n, self.P = sys_, sys_.n, sys_.P
        self.forcing_mode = forcing_mode
        self.z = torch.from_numpy(z).float()
        self.S = float(np.diff(z, axis=0).std())      # noise_scale_path artifact
        fc = (sys_.mean_field(np.arange(n_train)) - self.mu) / self.sd
        if forcing_mode == "static":
            # The TIME MEAN is still handed over (c_grid = the annual-mean
            # pattern, which for this system IS the climatology), but the
            # SEASONAL MODULATION is not: it has to be reconstructed by
            # combining the spatially uniform c_scalar calendar with the
            # static pattern. That is the real model's actual situation for
            # the seasonal cycle -- insolation/SST vary seasonally, but the
            # map from them to the 153-channel seasonal response is learned.
            fc = np.repeat(fc.mean(axis=0, keepdims=True), n_train, axis=0)
        elif forcing_mode == "uniform":
            # The SPATIAL PATTERN is removed from the forcing: c_grid carries
            # only the (spatially constant) seasonal level. The network is a
            # circular conv net, so it cannot store m(x) in its weights either
            # -- the pattern now has to be recovered from the WINDOW CONTENT.
            # This is the toy analogue of the real model's situation, where
            # c_grid (insolation, SST, sea-ice, z_sfc, LSM) does not determine
            # the 153-channel climatology; "direct" hands m(x) over for free.
            fc = np.repeat(fc.mean(axis=1, keepdims=True), self.n, axis=1)
        elif forcing_mode != "direct":
            raise ValueError(f"unknown forcing_mode '{forcing_mode}'")
        self.fc = torch.from_numpy(fc).float()
        ph = 2.0 * np.pi * np.arange(n_train) / self.P
        self.cs = torch.from_numpy(np.stack([np.sin(ph), np.cos(ph)], 1)).float()
        rs = sys_.run(burn + n_stat, seed=seed + 991)[0][burn:]
        self.zs = (rs - self.mu) / self.sd
        # whole number of seasonal cycles: an incomplete cycle aliases the
        # seasonal signal into the time-mean pattern the drift is measured against
        nwhole = max(self.P, (n_stat // self.P) * self.P)
        self.stats = climatology_stats(self.zs[:nwhole],
                                       np.arange(nwhole) % self.P, self.P)
        self.gen_seconds = time.time() - t0

    def to(self, device):
        self.z = self.z.to(device)
        self.fc = self.fc.to(device)
        self.cs = self.cs.to(device)
        return self


# ===========================================================================
# 2. Network: identical architecture for RSI and ERDM
# ===========================================================================
class Block(nn.Module):
    def __init__(self, d, W, k=5):
        super().__init__()
        self.n1 = nn.GroupNorm(8, d)
        self.c1 = nn.Conv1d(d, d, k, padding=k // 2, padding_mode="circular")
        self.n2 = nn.GroupNorm(8, d)
        self.c2 = nn.Conv1d(d, d, k, padding=k // 2, padding_mode="circular")
        self.emb = nn.Linear(d, d)
        self.mix = nn.Parameter(torch.zeros(W, W))     # temporal mixing over slots
        self.mix_out = nn.Conv1d(d, d, 1)
        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)
        self.W = W

    def forward(self, h, cond, b):
        r = self.c1(nn.functional.silu(self.n1(h)))
        r = r + self.emb(cond)[:, :, None]
        r = self.c2(nn.functional.silu(self.n2(r)))
        h = h + r
        d, N = h.shape[1], h.shape[2]
        m = torch.softmax(self.mix + 3.0 * torch.eye(self.W, device=h.device),
                          dim=1)
        hw = torch.einsum("wv,bvdn->bwdn", m, h.view(b, self.W, d, N))
        return h + self.mix_out(hw.reshape(b * self.W, d, N))


class ToyNet(nn.Module):
    """(b,W,C,1,N) + label (b,W) + c_grid + c_scalar -> (b,W,C*out_mult,1,N)."""

    def __init__(self, c_state=1, c_grid=1, s_dim=2, out_mult=2, dim=96,
                 nblocks=4, W=6, nfreq=16):
        super().__init__()
        self.W, self.C, self.out_mult = W, c_state, out_mult
        self.inp = nn.Conv1d(c_state + c_grid, dim, 5, padding=2,
                             padding_mode="circular")
        self.lab = nn.Sequential(nn.Linear(2 * nfreq, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        self.scal = nn.Linear(s_dim, dim)
        self.blocks = nn.ModuleList([Block(dim, W) for _ in range(nblocks)])
        self.norm = nn.GroupNorm(8, dim)
        self.out = nn.Conv1d(dim, c_state * out_mult, 1)
        nn.init.zeros_(self.out.weight)                 # zero-init last layer
        nn.init.zeros_(self.out.bias)
        self.register_buffer("freqs", torch.exp(
            torch.linspace(math.log(1.0), math.log(60.0), nfreq)))

    def forward(self, x, label, c_grid, c_scalar):
        b, W = x.shape[0], x.shape[1]
        xs = x[..., 0, :]
        if c_grid is not None:
            xs = torch.cat([xs, c_grid[..., 0, :]], dim=2)
        h = self.inp(xs.reshape(b * W, xs.shape[2], xs.shape[3]))
        ang = label.reshape(-1, 1) * self.freqs[None, :]
        cond = self.lab(torch.cat([torch.sin(ang), torch.cos(ang)], dim=1))
        if c_scalar is not None:
            cond = cond + self.scal(c_scalar.reshape(b * W, -1))
        for blk in self.blocks:
            h = blk(h, cond, b)
        o = self.out(nn.functional.silu(self.norm(h)))
        return o.view(b, W, self.C * self.out_mult, 1, o.shape[-1])


# ===========================================================================
# 3. Train-time anchor interventions: the ONLY hook overridden
# ===========================================================================
class AnchorHookRSI(RSIScheduler):
    """RSIScheduler whose ``perturb_anchor`` hook can apply, on top of the
    scheduler's own white ``anchor_noise``:

    * ``shrink > 0``  -- brief H1 Test B: a <- <a> + s (a - <a>), s ~ U(1-shrink, 1);
      in normalized units <a> == 0, so a <- s a.
    * ``self_anchor_last`` -- slot W's anchor replaced by the tensor stashed
      here (the model's own ``y_hat[:, -1]`` from a preceding roll, which is
      literally what ``_fresh_slot`` copies forward at inference).

    ``compute_loss`` is the shipped one; it calls ``perturb_anchor(y[:, :-1])``
    exactly once, on the anchor stack ALONE, which is why this hook can perturb
    an anchor without also perturbing the target that shares the frame.
    """

    def __init__(self, *a, shrink=0.0, shrink_last_only=False, **kw):
        super().__init__(*a, **kw)
        self.shrink = float(shrink)
        self.shrink_last_only = bool(shrink_last_only)
        self.self_anchor_last = None

    def perturb_anchor(self, anchor):
        a = super().perturb_anchor(anchor)
        if self.self_anchor_last is not None:
            a = torch.cat([a[:, :-1], self.self_anchor_last], dim=1)
        if self.shrink > 0.0:
            s = 1.0 - self.shrink * torch.rand(
                a.shape[0], 1, 1, 1, 1, device=a.device, dtype=a.dtype)
            a = (a * s if not self.shrink_last_only
                 else torch.cat([a[:, :-1], a[:, -1:] * s], dim=1))
        return a


# ===========================================================================
# 4. Batching + training with the REAL compute_loss
# ===========================================================================
def sample_indices(T, W, b, gen, device):
    return torch.randint(0, T - (W + 2), (b,), generator=gen, device=device)


def make_batch(data, idx, W, kind):
    """RSI: y = z[j..j+W] (W+1 frames); ERDM: y = z[j+1..j+W] (W frames).
    Both: cg = fc[j..j+W-1], cs = cs[j..j+W-1] (states t+1..t+W, forcings
    t..t+W-1 -- the alignment verified bit-exactly for the real recipe)."""
    js = idx[:, None] + torch.arange(W + 1, device=idx.device)[None, :]
    y = data.z[js][:, :, None, None, :]
    cgi = idx[:, None] + torch.arange(W, device=idx.device)[None, :]
    cg = data.fc[cgi][:, :, None, None, :]
    cs = data.cs[cgi]
    return (y if kind == "rsi" else y[:, 1:]), cg, cs


def train(kind, sched, net, data, steps, batch, lr, wd, seed, device, log, tag,
          self_anchor=False, warm_net=None):
    if warm_net is not None:
        net.load_state_dict(warm_net.state_dict())
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    W, hist, t0 = sched.W, [], time.time()
    for it in range(steps):
        idx = sample_indices(data.z.shape[0], W, batch, gen, device)
        y, cg, cs = make_batch(data, idx, W, kind)
        if self_anchor:
            with torch.no_grad():
                prev, cgp, csp = make_batch(data, (idx - 1).clamp(min=0), W,
                                            "rsi")
                _, yh = sched.sample_window(net, sched.warmup_window(prev),
                                            cgp, csp)
            sched.self_anchor_last = yh[:, -1:]
        loss = sched.compute_loss(net, cg, cs, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lrs.step()
        hist.append(float(loss.detach()))
        if (it + 1) % max(1, steps // 8) == 0:
            log(f"  [{tag}] step {it+1}/{steps} "
                f"loss(med last 200)={np.median(hist[-200:]):.5g} "
                f"({time.time()-t0:.0f}s)")
    if hasattr(sched, "self_anchor_last"):
        sched.self_anchor_last = None
    net.eval()
    stride = max(1, steps // 40)
    return dict(tag=tag, steps=steps, seconds=time.time() - t0,
                loss_first200=float(np.median(hist[:200])),
                loss_last200=float(np.median(hist[-200:])),
                loss_curve=[float(np.median(hist[i:i + stride]))
                            for i in range(0, steps, stride)])


# ===========================================================================
# 5. Rollouts
# ===========================================================================
@torch.no_grad()
def rsi_rollout(sched, net, init, cg_traj, cs_traj, horizon, fresh="default",
                keep_diag=True):
    """Mirror of RSIScheduler.sample_rollout with a pluggable fresh slot.

    fresh: "default"    -> y_hat[:, -1] + Gamma(0) z         (shipped)
           ("scale", c) -> y_hat[:, -1] + c * Gamma(0) z
           "xlast"      -> x[:, -1] (the integrated interpolant state itself)
    Validated bit-exact against sample_rollout for fresh="default"."""
    x = sched.warmup_window(sched.pad_state(init))
    out, de, da, dw = [], [], [], []
    for k in range(horizon):
        cgw = sched._gather_window(cg_traj, k)
        csw = sched._gather_window(cs_traj, k)
        x, y_hat = sched.sample_window(net, x, cgw, csw)
        out.append(y_hat[:, 0])
        if fresh == "xlast":
            new = x[:, -1:]
        else:
            anchor = y_hat[:, -1:]
            z = sched.get_noise(anchor)
            tau0 = torch.zeros(anchor.shape[0], 1, device=anchor.device)
            c = 1.0 if fresh == "default" else float(fresh[1])
            new = anchor + c * sched.gamma_apply(z, tau0)
        if keep_diag:
            # spatial pattern amplitude (std over sites), member-mean
            de.append(y_hat[:, 0, 0, 0, :].std(-1).mean())
            da.append(y_hat[:, -1, 0, 0, :].std(-1).mean())
            dw.append(x[:, -1, 0, 0, :].std(-1).mean())
        x = torch.cat([x[:, 1:], new], dim=1)
    diag = {}
    if keep_diag:
        diag = dict(emit_patamp=torch.stack(de).cpu().tolist(),
                    anchor_patamp=torch.stack(da).cpu().tolist(),
                    xlast_patamp=torch.stack(dw).cpu().tolist())
    return torch.stack(out, dim=1), diag


@torch.no_grad()
def erdm_rollout(sched, net, init, cg_traj, cs_traj, horizon, keep_diag=True):
    """Mirror of ERDMScheduler.sample_rollout (validated bit-exact)."""
    x = sched.pad_state(init)
    sigma0 = sched.sigma_schedule(torch.zeros(x.shape[0], device=x.device))
    eps = sched.temporal_noise(x)
    x = x + sched.w5(sigma0) * eps
    eps_prev = eps[:, -1:]
    out, de = [], []
    for k in range(horizon):
        x = sched.sample_window(net, x, sched._gather_window(cg_traj, k),
                                sched._gather_window(cs_traj, k))
        out.append(x[:, 0])
        if keep_diag:
            de.append(x[:, 0, 0, 0, :].std(-1).mean())
        eps_prev = sched.temporal_noise_next(eps_prev)
        x = torch.cat([x[:, 1:], eps_prev * sched.sigma_max], dim=1)
    diag = dict(emit_patamp=torch.stack(de).cpu().tolist()) if keep_diag else {}
    return torch.stack(out, dim=1), diag


def rollout_traj(data, j0, horizon, W, nmem, device):
    """RSI init = z[j0..j0+W] (W+1 frames); ERDM init = init[:, 1:].
    Emitted frame k estimates z[j0+1+k]; forcing traj index 0 == fc[j0]."""
    fi = torch.arange(j0, j0 + horizon + W + 2, device=device)
    cg = data.fc[fi][None, :, None, None, :].expand(nmem, -1, -1, -1, -1)
    cs = data.cs[fi][None].expand(nmem, -1, -1)
    zi = torch.arange(j0, j0 + W + 1, device=device)
    init = data.z[zi][None, :, None, None, :].expand(nmem, -1, -1, -1, -1)
    return init.contiguous(), cg.contiguous(), cs.contiguous()


# ===========================================================================
# 6. Metrics
# ===========================================================================
def shrink_fit(bias, clim):
    """Brief 3.5: bias(x) = -alpha (clim(x) - <clim>) + c -> (alpha, R2)."""
    a = clim - clim.mean()
    y = bias - bias.mean()
    alpha = -float((a * y).sum() / (a * a).sum())
    ss = float(((y + alpha * a) ** 2).sum())
    return alpha, 1.0 - ss / max(float((y ** 2).sum()), 1e-30)


def rollout_metrics(emit, data, j0, burn):
    e = emit[:, :, 0, 0, :].detach().float().cpu().numpy()      # (M, H, N)
    H = e.shape[1]
    phase = np.arange(j0 + 1, j0 + 1 + H) % data.P
    st = data.stats
    ez, ph = e[:, burn:, :], phase[burn:]
    ps = climatology_stats(ez.reshape(-1, ez.shape[-1]),
                           np.tile(ph, ez.shape[0]), data.P)
    clim_t = st["clim"]
    bias = ps["clim"] - clim_t
    alpha, r2 = shrink_fit(bias, clim_t)
    ct, cp = clim_t - clim_t.mean(), ps["clim"] - ps["clim"].mean()
    spread = float(np.sqrt((ez.var(0, ddof=1)).mean())) if ez.shape[0] > 1 \
        else float("nan")
    mm = e.mean(0)
    trace = np.sqrt(((mm - clim_t[None, :]) ** 2).mean(1))
    return dict(
        alpha=alpha, r2=r2,
        pattern_amp_ratio=float(cp.std() / ct.std()),
        pattern_corr=float(np.corrcoef(cp, ct)[0, 1]),
        global_mean_bias=float(bias.mean()),
        bias_rms=float(np.sqrt((bias ** 2).mean())),
        resid_var_ratio=float((ps["resid_std"] ** 2).mean()
                              / (st["resid_std"] ** 2).mean()),
        total_var_ratio=float(ps["total_var"] / st["total_var"]),
        seas_amp_ratio=float(ps["seas_amp_rms"] / st["seas_amp_rms"]),
        spread=spread,
        spread_saturation=float(spread / (math.sqrt(2.0)
                                          * st["resid_std_rms"])),
        trace200=[float(v) for v in trace[:200]],
        trace_decim=[float(v) for v in trace[::max(1, H // 60)]],
        pred_clim=[float(v) for v in ps["clim"]],
    )


@torch.no_grad()
def free_run_skill(kind, sched, net, data, nic, horizon, device, seed=0):
    """Free-run RMSE from true ICs for the first ``horizon`` steps (brief 2.2)."""
    W = sched.W
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = sample_indices(data.z.shape[0] - horizon - W - 4, W, nic, gen, device)
    errs = np.zeros((nic, horizon))
    for i in range(nic):
        j0 = int(idx[i])
        torch.manual_seed(seed + i)
        init, cg, cs = rollout_traj(data, j0, horizon, W, 1, device)
        if kind == "rsi":
            em, _ = rsi_rollout(sched, net, init, cg, cs, horizon,
                                keep_diag=False)
        else:
            em, _ = erdm_rollout(sched, net, init[:, 1:], cg, cs, horizon,
                                 keep_diag=False)
        tr = data.z[torch.arange(j0 + 1, j0 + 1 + horizon, device=device)]
        errs[i] = ((em[0, :, 0, 0] - tr) ** 2).mean(-1).sqrt().cpu().numpy()
    return [float(v) for v in errs.mean(0)]


@torch.no_grad()
def per_slot_readout_ratio(sched, net, data, nb, device, t_global=0.5, seed=0):
    """H2/H4 teacher-forced probe: std(y_hat_w)/std(y_w) per slot at the
    sampler's actual final evaluation time (global t = 0.5, see L1)."""
    torch.manual_seed(seed)
    W = sched.W
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = sample_indices(data.z.shape[0], W, nb, gen, device)
    y, cg, cs = make_batch(data, idx, W, "rsi")
    anchors, targets = y[:, :-1], y[:, 1:]
    tau = sched.local_time(torch.full((y.shape[0],), t_global, device=device))
    z = sched.get_noise(targets)
    x = sched.interpolant(anchors, targets, tau, z)
    with torch.no_grad():
        h1, zhat = sched.heads(net, x, tau, cg, cs)
    out = {}
    for w in range(W):
        yh, yt = h1[:, w].flatten().float(), targets[:, w].flatten().float()
        xw = x[:, w].flatten().float()
        g = float(sched.gamma(tau)[0, w])
        out[f"slot{w+1}"] = dict(
            tau=float(tau[0, w]), gamma=g, c_skip=1.0 / (1.0 + g ** 2),
            std_ratio=float(yh.std() / yt.std()),
            slope_on_truth=float(((yh - yh.mean()) * (yt - yt.mean())).sum()
                                 / ((yt - yt.mean()) ** 2).sum()),
            slope_on_x=float(((yh - yh.mean()) * (xw - xw.mean())).sum()
                             / ((xw - xw.mean()) ** 2).sum()),
            rmse=float(((yh - yt) ** 2).mean().sqrt()),
            mean=float(yh.mean()),
        )
    return out


@torch.no_grad()
def restoring_probe(kind, sched, net, data, nb, device, seed=0):
    """Brief H1 Test A. Perturb TRUE windows, run ONE roll, measure how much of
    the perturbation survives in (i) the emitted frame y_hat[:, 0] and (ii) the
    quantity carried forward (RSI: fresh-slot anchor y_hat[:, -1]; ERDM: the
    back slot x_bar[:, -1]).

    r_resp = 1 - rms(perturbed - unperturbed response) / rms(perturbation):
      1 = fully restored, 0 = neutral pass-through, < 0 = amplifying.
    r_err  = same but against unperturbed truth (includes the model's own bias).
    """
    W = sched.W
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = sample_indices(data.z.shape[0], W, nb, gen, device)
    y, cg, cs = make_batch(data, idx, W, "rsi")
    rms = lambda v: float((v.float() ** 2).mean().sqrt())    # noqa: E731

    def one(yy, s):
        torch.manual_seed(s)
        if kind == "rsi":
            with torch.no_grad():
                _, yh = sched.sample_window(net, sched.warmup_window(yy), cg, cs)
            return yh[:, 0], yh[:, -1]
        init = yy[:, 1:]
        sig0 = sched.sigma_schedule(torch.zeros(init.shape[0], device=device))
        xb = init + sched.w5(sig0) * sched.temporal_noise(init)
        with torch.no_grad():
            xb = sched.sample_window(net, xb, cg, cs)
        return xb[:, 0], xb[:, -1]

    ref_e, ref_a = one(y, 1234)
    res = {}
    for name, fn in (("anom_x0.7", lambda v: 0.7 * v),
                     ("offset_-1", lambda v: v - 1.0)):
        yp = fn(y.clone())
        pe, pa = one(yp, 1234)
        d_e, d_a = yp[:, 1] - y[:, 1], yp[:, W] - y[:, W]
        res[name] = dict(
            pert_rms_emit=rms(d_e), pert_rms_anchor=rms(d_a),
            r_resp_emit=1.0 - rms(pe - ref_e) / rms(d_e),
            r_resp_anchor=1.0 - rms(pa - ref_a) / rms(d_a),
            r_err_emit=1.0 - rms(pe - y[:, 1]) / rms(d_e),
            r_err_anchor=1.0 - rms(pa - y[:, W]) / rms(d_a),
            baseline_err_emit=rms(ref_e - y[:, 1]),
            baseline_err_anchor=rms(ref_a - y[:, W]),
        )
    return res




@torch.no_grad()
def flush_probe(kind, sched, net, data, nic, device, rolls=48, seed=0):
    """Multi-roll version of the restoring probe -- the sharpest structural
    discriminator between the two formulations.

    Perturb the TRUE initial window, keep the forcing at truth, and follow the
    difference between the perturbed and unperturbed rollouts for ``rolls``
    steps. ERDM replaces one window slot with PURE NOISE per roll, so an
    initial-condition perturbation must be completely flushed after ~W rolls
    unless the model's own dynamics regenerate it; RSI anchors every new slot
    on ``y_hat[:, -1]``, so its memory of the perturbation has no finite
    horizon. Returns rms(pert - ref) per roll, normalized by the rms of the
    perturbation put in.
    """
    W = sched.W
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = sample_indices(data.z.shape[0] - rolls - W - 4, W, nic, gen, device)
    ar = torch.arange(W + 1, device=device)
    fr = torch.arange(rolls + W + 2, device=device)
    init = data.z[idx[:, None] + ar[None, :]][:, :, None, None, :]
    cg = data.fc[idx[:, None] + fr[None, :]][:, :, None, None, :]
    cs = data.cs[idx[:, None] + fr[None, :]]
    rms = lambda v: float((v.float() ** 2).mean().sqrt())        # noqa: E731

    def run(yy):
        torch.manual_seed(seed + 4242)
        if kind == "rsi":
            em, _ = rsi_rollout(sched, net, yy, cg, cs, rolls, keep_diag=False)
        else:
            em, _ = erdm_rollout(sched, net, yy[:, 1:], cg, cs, rolls,
                                 keep_diag=False)
        return em

    ref = run(init)
    out = {}
    for name, fn in (("anom_x0.7", lambda v: 0.7 * v),
                     ("offset_-1", lambda v: v - 1.0)):
        yp = fn(init.clone())
        d0 = rms(yp - init)
        pe = run(yp)
        curve = [float(((pe[:, k] - ref[:, k]).float() ** 2).mean().sqrt()
                       / d0) for k in range(rolls)]
        out[name] = dict(pert_rms=d0, residual_by_roll=curve)
    return out


# ===========================================================================
# 7. Driver, split into phases so the long rollouts can be sharded across
#    parallel processes (the sampler is kernel-launch bound, so 3 concurrent
#    processes on one GPU run ~2.5x faster than one).
#      --phase train  : data + all trainings + probes -> state.pt, train.json
#      --phase roll   : one or more long rollouts     -> roll_<name>.json
#      --phase collect: merge everything              -> results.json + CSVs
#      --phase all    : train, then every rollout in-process, then collect
# ===========================================================================
VARIANTS = {
    # name              -> (net key, spec)
    "erdm":              ("erdm", {}),
    "rsi_shipped":       ("rsi", {}),
    "rsi_final_denoise": ("rsi", dict(sched=dict(final_denoise=True))),
    "rsi_anchor_xlast":  ("rsi", dict(fresh="xlast")),
    "rsi_anchor_c1":     ("rsi", dict(fresh=("scale", 1.0))),
    "rsi_anchor_c2":     ("rsi", dict(fresh=("scale", 2.0))),
    "rsi_anchor_c3":     ("rsi", dict(fresh=("scale", 3.0))),
    "rsi_eps1.0":        ("rsi", dict(sched=dict(eps_scale=1.0))),
    "rsi_train_an1.0":   ("rsi_an1.0", {}),
    "rsi_train_an2.0":   ("rsi_an2.0", {}),
    "rsi_ft_shrink":     ("rsi_ft_shrink", {}),
    "rsi_ft_selfanchor": ("rsi_ft_selfanchor", {}),
    "rsi_train_h1none":  ("rsi_h1none", {}),
    "rsi_train_S1":      ("rsi_S1", {}),
}
DEFAULT_VARIANTS = [k for k in VARIANTS if k not in
                    ("rsi_anchor_c1", "rsi_train_h1none", "rsi_train_S1")]


def build_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Trained nonlinear toy for the RSI long-rollout drift.")
    ap.add_argument("--out", default="/tmp/learned_toy")
    ap.add_argument("--phase", default="all",
                    choices=["all", "train", "roll", "probe", "collect"])
    ap.add_argument("--variants", default="",
                    help="comma list for --phase roll/all (default: all but "
                         "the redundant rsi_anchor_c1 and the --extras nets)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-sites", type=int, default=32)
    ap.add_argument("--period", type=int, default=180)
    ap.add_argument("--train-len", type=int, default=200_000)
    ap.add_argument("--stat-len", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--ft-steps", type=int, default=4_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=5000)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skill-horizon", type=int, default=10)
    ap.add_argument("--skill-ics", type=int, default=32)
    ap.add_argument("--probe-batch", type=int, default=256)
    ap.add_argument("--extras", action="store_true",
                    help="also train h1_precond=none and S=1 RSI variants")
    ap.add_argument("--forcing-mode", default="direct",
                    choices=["direct", "uniform", "static"],
                    help="direct: c_grid = m_j(x) (the forced mean pattern "
                         "itself). uniform: c_grid = <m_j> (season only, no "
                         "spatial pattern) -- the mean pattern must then be "
                         "carried by the window. static: c_grid = the "
                         "annual-mean pattern (time mean given, seasonal "
                         "modulation must be learned from c_scalar).")
    ap.add_argument("--nets", default="",
                    help="comma list restricting which nets --phase train "
                         "trains (default: all; 'rsi' is always included)")
    ap.add_argument("--quick", action="store_true", help="smoke settings")
    args = ap.parse_args(argv)
    if args.quick:
        args.train_len, args.stat_len = 20_000, 20_000
        args.steps, args.ft_steps = 300, 150
        args.horizon, args.members, args.burn = 200, 8, 20
        args.period = 30
        args.probe_batch, args.skill_ics = 64, 3
    return args


def _logger(out, name):
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, name), "a")

    def log(m):
        print(m, flush=True)
        f.write(m + "\n")
        f.flush()
    return log


def _mk_net(args, W, out_mult, seed, dev):
    torch.manual_seed(seed)
    return ToyNet(c_state=1, c_grid=1, s_dim=2, out_mult=out_mult,
                  dim=args.dim, nblocks=args.blocks, W=W).to(dev)


def _rsi_kw(spath, **over):
    kw = dict(RSI_RECIPE)
    kw["noise_scale_path"] = spath
    kw.update(over)
    return kw


# --- net-key registry: how each trained net was produced -------------------
NET_SPECS = {
    "rsi":               dict(kind="rsi", over={}),
    "erdm":              dict(kind="erdm", over={}),
    "rsi_an1.0":         dict(kind="rsi", over=dict(anchor_noise=1.0)),
    "rsi_an2.0":         dict(kind="rsi", over=dict(anchor_noise=2.0)),
    "rsi_h1none":        dict(kind="rsi", over=dict(h1_precond="none")),
    "rsi_S1":            dict(kind="rsi", over=dict(S_one=True)),
    "rsi_ft_shrink":     dict(kind="rsi", over={}, ft=dict(shrink=0.3)),
    "rsi_ft_selfanchor": dict(kind="rsi", over={}, ft=dict(self_anchor=True)),
}


def sampler_for(netkey, spath, s1path, dev, **over):
    """The eval-time scheduler for a trained net (+ optional overrides)."""
    spec = NET_SPECS[netkey]
    if spec["kind"] == "erdm":
        return ERDMScheduler(**ERDM_EVAL_RECIPE).to(dev)
    kw = dict(spec["over"])
    kw.pop("anchor_noise", None)          # training-only knob
    if kw.pop("S_one", False):
        return RSIScheduler(**_rsi_kw(s1path, **kw, **over)).to(dev)
    return RSIScheduler(**_rsi_kw(spath, **kw, **over)).to(dev)


def phase_train(args, dev, log):
    W = RSI_RECIPE["window_size"]
    sysm = ToySystem(n=args.n_sites, period=args.period)
    data = ToyData(sysm, n_train=args.train_len, n_stat=args.stat_len,
                   seed=args.seed, forcing_mode=args.forcing_mode).to(dev)
    st = data.stats
    log(f"toy: mu={data.mu:.4f} sd={data.sd:.4f} S(1-step increment std)="
        f"{data.S:.4f} clim_pattern_std={st['clim_pattern_std']:.4f} "
        f"resid_std_rms={st['resid_std_rms']:.4f} "
        f"seas_amp_rms={st['seas_amp_rms']:.4f} "
        f"total_var={st['total_var']:.4f} ({data.gen_seconds:.0f}s)")

    half = args.stat_len // 2
    c1, c2 = data.zs[:half].mean(0), data.zs[half:].mean(0)
    zz = data.zs
    res_lag = zz[1:] - zz[:-1]
    stationarity = dict(
        clim_half_corr=float(np.corrcoef(c1, c2)[0, 1]),
        clim_half_amp_ratio=float(c2.std() / c1.std()),
        clim_half_rms_diff=float(np.sqrt(((c1 - c2) ** 2).mean())),
        skew=float(((zz - zz.mean()) ** 3).mean() / zz.std() ** 3),
        kurt=float(((zz - zz.mean()) ** 4).mean() / zz.std() ** 4),
        increment_skew=float((res_lag ** 3).mean() / res_lag.std() ** 3),
        increment_kurt=float((res_lag ** 4).mean() / res_lag.std() ** 4),
        finite=bool(np.isfinite(zz).all()),
        state_min=float(zz.min()), state_max=float(zz.max()))
    log(f"stationarity/non-gaussianity: {stationarity}")

    # L3: true multi-step forecast spread from a perfect IC vs Gamma(0) = 1*S
    u0 = data.mu + data.sd * data.zs[1000]
    ens = sysm.run(12, seed=args.seed + 7, nens=256,
                   u0=np.tile(u0, (256, 1)), k0=1000)
    ez = (ens - data.mu) / data.sd
    fc_err = [float(ez[:, k].std(0).mean()) for k in range(12)]
    l3 = dict(fc_err_std_by_lead=fc_err, S=data.S,
              gamma0_injected=data.S * RSI_RECIPE["gamma_0"],
              deficit_var_ratio_lead5=float((fc_err[4] / data.S) ** 2),
              deficit_var_ratio_lead6=float((fc_err[5] / data.S) ** 2),
              clim_spread_sqrt2sigma=float(math.sqrt(2)
                                           * st["resid_std_rms"]))
    log(f"L3: true fc-error std lead 1..6 = "
        f"{['%.4f' % v for v in fc_err[:6]]} vs Gamma(0)=1*S={data.S:.4f}; "
        f"variance deficit at lead 5 = {l3['deficit_var_ratio_lead5']:.2f}x")

    spath = os.path.join(args.out, "S_toy.pt")
    torch.save(torch.tensor([[[data.S]]], dtype=torch.float32), spath)
    s1path = os.path.join(args.out, "S_one.pt")
    torch.save(torch.tensor([[[1.0]]], dtype=torch.float32), s1path)

    rsi = RSIScheduler(**_rsi_kw(spath)).to(dev)
    erdm_tr = ERDMScheduler(**ERDM_TRAIN_RECIPE).to(dev)
    erdm_ev = ERDMScheduler(**ERDM_EVAL_RECIPE).to(dev)

    # -- L1: at which tau does the LAST head evaluation happen? ------------
    seen = []

    class Spy(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, label, cg, cs):
            seen.append(label.detach()[0].cpu().numpy().copy())
            return self.m(x, label, cg, cs)

    n_probe = _mk_net(args, W, 2, args.seed, dev)
    init, cg, cs = rollout_traj(data, 1000, 2, W, 2, dev)
    with torch.no_grad():
        rsi.sample_window(Spy(n_probe), rsi.warmup_window(init),
                          rsi._gather_window(cg, 0), rsi._gather_window(cs, 0))
    g_last = float(RSI_RECIPE["gamma_0"] * (
        RSI_RECIPE["gamma_1"] / RSI_RECIPE["gamma_0"]) ** seen[-1][-1])
    tauW = float(seen[-1][-1])
    beta_W = tauW
    l1 = dict(n_head_evals=len(seen),
              tau_per_eval=[[float(v) for v in s] for s in seen],
              last_eval_tau_slot1=float(seen[-1][0]),
              last_eval_tau_slotW=tauW,
              gamma_at_last_slotW=g_last,
              c_skip_at_last_slotW=1.0 / (1.0 + g_last ** 2),
              c_out_at_last_slotW=g_last / math.sqrt(1.0 + g_last ** 2),
              # L2: what F_1 must contain at the anchor-producing readout for
              # y_hat = y, given x = a + beta (y - a) + gamma S z:
              #   F_1 = [ (1 - c_skip beta) y - c_skip (1-beta) a
              #           - c_skip gamma S z ] / c_out
              L2_F1_coeff_y=(1.0 - beta_W / (1.0 + g_last ** 2))
              / (g_last / math.sqrt(1.0 + g_last ** 2)),
              L2_F1_coeff_anchor=-(1.0 - beta_W) / (1.0 + g_last ** 2)
              / (g_last / math.sqrt(1.0 + g_last ** 2)),
              L2_F1_coeff_Sz=-g_last / (1.0 + g_last ** 2)
              / (g_last / math.sqrt(1.0 + g_last ** 2)),
              L2_note="coeff_Sz multiplies S*z, so with S<<1 the latent term "
                      "is negligible and F_1's target variance is set by the "
                      "y/anchor terms, not by unit-variance noise")
    log(f"L1/L2: {json.dumps(l1)}")

    results = dict(
        args=vars(args), L1=l1, L3=l3, stationarity=stationarity,
        toy=dict(mu=data.mu, sd=data.sd, S=data.S,
                 clim_pattern_std=st["clim_pattern_std"],
                 resid_std_rms=st["resid_std_rms"],
                 seas_amp_rms=st["seas_amp_rms"], total_var=st["total_var"],
                 clim=[float(v) for v in st["clim"]],
                 params={k: getattr(sysm, k) for k in
                         ("n", "P", "M", "a", "b", "q", "kappa", "lam", "sig",
                          "k0", "nsub", "dt")}),
        training={}, variants={})

    # -- train -------------------------------------------------------------
    order = ["rsi", "erdm", "rsi_an1.0", "rsi_an2.0"]
    if args.extras:
        order += ["rsi_h1none", "rsi_S1"]
    order += ["rsi_ft_shrink", "rsi_ft_selfanchor"]
    if args.nets:
        keep = set(v for v in args.nets.split(",") if v) | {"rsi"}
        order = [t for t in order if t in keep]
    nets, scheds, kinds = {}, {}, {}
    for tag in order:
        spec = NET_SPECS[tag]
        kind = spec["kind"]
        ft = spec.get("ft")
        if kind == "erdm":
            sc, tr_sc, om = erdm_ev, erdm_tr, 1
        else:
            over = dict(spec["over"])
            s_one = over.pop("S_one", False)
            kw = _rsi_kw(s1path if s_one else spath, **over)
            if ft is not None:
                tr_sc = AnchorHookRSI(**kw,
                                      shrink=ft.get("shrink", 0.0)).to(dev)
            else:
                tr_sc = RSIScheduler(**kw).to(dev)
            sc = sampler_for(tag, spath, s1path, dev)
            om = 2
        net = _mk_net(args, W, om, args.seed, dev)
        if tag == "rsi":
            results["net_params"] = sum(p.numel() for p in net.parameters())
            log(f"net params: {results['net_params']}")
        steps = args.ft_steps if ft is not None else args.steps
        lr = args.lr * (0.3 if ft is not None else 1.0)
        seed = args.seed + (5 if ft is not None else 0)
        results["training"][tag] = train(
            kind, tr_sc, net, data, steps, args.batch, lr, args.wd, seed, dev,
            log, tag, self_anchor=bool(ft and ft.get("self_anchor")),
            warm_net=nets["rsi"] if ft is not None else None)
        nets[tag], scheds[tag], kinds[tag] = net, sc, kind

    # -- short-range skill (brief 2.2 analog) ------------------------------
    for tag in order:
        sk = free_run_skill(kinds[tag], scheds[tag], nets[tag], data,
                            args.skill_ics, args.skill_horizon, dev,
                            seed=args.seed + 3)
        results["training"][tag]["skill_rmse"] = sk
        log(f"  [{tag}] free-run RMSE steps 1..{args.skill_horizon}: "
            f"{['%.4f' % v for v in sk]}")

    # -- harness validation -----------------------------------------------
    init, cgt, cst = rollout_traj(data, min(5000, args.train_len // 3), 20, W,
                                  2, dev)
    torch.manual_seed(77)
    a1 = rsi.sample_rollout(nets["rsi"], init, cgt, cst, 20)
    torch.manual_seed(77)
    a2, _ = rsi_rollout(rsi, nets["rsi"], init, cgt, cst, 20, keep_diag=False)
    torch.manual_seed(77)
    a3, _ = rsi_rollout(rsi, nets["rsi"], init, cgt, cst, 20,
                        fresh=("scale", 1.0), keep_diag=False)
    torch.manual_seed(78)
    b1 = erdm_ev.sample_rollout(nets["erdm"], init[:, 1:], cgt, cst, 20)
    torch.manual_seed(78)
    b2, _ = erdm_rollout(erdm_ev, nets["erdm"], init[:, 1:], cgt, cst, 20,
                         keep_diag=False)
    results["harness"] = dict(
        rsi_max_abs_diff=float((a1 - a2).abs().max()),
        rsi_bitexact=bool(torch.equal(a1, a2)),
        rsi_freshc1_bitexact=bool(torch.equal(a1, a3)),
        erdm_max_abs_diff=float((b1 - b2).abs().max()),
        erdm_bitexact=bool(torch.equal(b1, b2)))
    log(f"harness: {results['harness']}")

    # -- readout ratios + restoring probes --------------------------------
    results["readout_ratio"], results["restoring"] = {}, {}
    for tag in order:
        if kinds[tag] == "rsi":
            results["readout_ratio"][tag] = per_slot_readout_ratio(
                scheds[tag], nets[tag], data, args.probe_batch, dev,
                seed=args.seed + 11)
        results["restoring"][tag] = restoring_probe(
            kinds[tag], scheds[tag], nets[tag], data, args.probe_batch, dev,
            seed=args.seed + 13)
    log("readout std_ratio by slot (RSI shipped): " + json.dumps(
        {k: round(v["std_ratio"], 4)
         for k, v in results["readout_ratio"]["rsi"].items()}))
    for k, v in results["restoring"].items():
        log(f"restoring[{k}]: " + json.dumps(
            {p: {kk: round(vv, 4) for kk, vv in d.items()}
             for p, d in v.items()}))

    torch.save(dict(
        z=data.z.cpu(), fc=data.fc.cpu(), cs=data.cs.cpu(), mu=data.mu,
        forcing_mode=args.forcing_mode,
        sd=data.sd, S=data.S, stats=data.stats, P=data.P, n=data.n,
        sys_params={k: getattr(sysm, k) for k in
                    ("n", "P", "M", "a", "b", "q", "kappa", "lam", "sig",
                     "k0", "nsub", "dt")},
        nets={k: {kk: vv.cpu() for kk, vv in v.state_dict().items()}
              for k, v in nets.items()},
    ), os.path.join(args.out, "state.pt"))
    with open(os.path.join(args.out, "train.json"), "w") as f:
        json.dump(results, f, indent=1)
    return results, data, nets, scheds, kinds, spath, s1path


class _LoadedData:
    """Minimal stand-in for ToyData rebuilt from state.pt."""

    def __init__(self, blob, dev):
        self.z = blob["z"].to(dev)
        self.fc = blob["fc"].to(dev)
        self.cs = blob["cs"].to(dev)
        self.mu, self.sd, self.S = blob["mu"], blob["sd"], blob["S"]
        self.stats, self.P, self.n = blob["stats"], blob["P"], blob["n"]
        self.forcing_mode = blob.get("forcing_mode", "direct")


def phase_roll(args, dev, log, names, state=None):
    W = RSI_RECIPE["window_size"]
    spath = os.path.join(args.out, "S_toy.pt")
    s1path = os.path.join(args.out, "S_one.pt")
    data, nets = _load_state(args, dev) if state is None else state
    j0 = args.train_len // 2
    init, cgt, cst = rollout_traj(data, j0, args.horizon, W, args.members, dev)
    out = {}
    for name in names:
        netkey, spec = VARIANTS[name]
        if netkey not in nets:
            log(f"[{name}] SKIP (net {netkey} not trained)")
            continue
        t0 = time.time()
        torch.manual_seed(args.seed + 101)
        if NET_SPECS[netkey]["kind"] == "erdm":
            sc = sampler_for(netkey, spath, s1path, dev)
            em, dg = erdm_rollout(sc, nets[netkey], init[:, 1:], cgt, cst,
                                  args.horizon)
        else:
            sc = sampler_for(netkey, spath, s1path, dev,
                             **spec.get("sched", {}))
            em, dg = rsi_rollout(sc, nets[netkey], init, cgt, cst,
                                 args.horizon,
                                 fresh=spec.get("fresh", "default"))
        m = rollout_metrics(em, data, j0, args.burn)
        m["diag_patamp"] = {k: v[::max(1, len(v) // 60)]
                            for k, v in dg.items()}
        m["seconds"] = time.time() - t0
        m["finite"] = bool(torch.isfinite(em).all())
        m["net"] = netkey
        out[name] = m
        with open(os.path.join(args.out, f"roll_{name}.json"), "w") as f:
            json.dump({name: m}, f, indent=1)
        log(f"[{name}] alpha={m['alpha']:.3f} R2={m['r2']:.3f} "
            f"amp={m['pattern_amp_ratio']:.3f} corr={m['pattern_corr']:.3f} "
            f"gm_bias={m['global_mean_bias']:+.3f} "
            f"varR={m['resid_var_ratio']:.3f} totvarR={m['total_var_ratio']:.3f} "
            f"seasR={m['seas_amp_ratio']:.3f} "
            f"spread_sat={m['spread_saturation']:.3f} ({m['seconds']:.0f}s)")
        del em
        torch.cuda.empty_cache()
    return out


def _load_state(args, dev):
    W = RSI_RECIPE["window_size"]
    blob = torch.load(os.path.join(args.out, "state.pt"), weights_only=False)
    data = _LoadedData(blob, dev)
    nets = {}
    for k, sd in blob["nets"].items():
        om = 1 if NET_SPECS[k]["kind"] == "erdm" else 2
        net = _mk_net(args, W, om, args.seed, dev)
        net.load_state_dict({kk: vv.to(dev) for kk, vv in sd.items()})
        net.eval()
        nets[k] = net
    return data, nets


def phase_probe(args, dev, log):
    """Multi-roll flush probe on the saved nets (see flush_probe)."""
    spath = os.path.join(args.out, "S_toy.pt")
    s1path = os.path.join(args.out, "S_one.pt")
    data, nets = _load_state(args, dev)
    out = {}
    for tag in nets:
        sc = sampler_for(tag, spath, s1path, dev)
        out[tag] = flush_probe(NET_SPECS[tag]["kind"], sc, nets[tag], data,
                               max(8, args.probe_batch // 16), dev, rolls=48,
                               seed=args.seed + 17)
        for pn, d in out[tag].items():
            c = d["residual_by_roll"]
            log(f"flush[{tag}][{pn}]: rolls 1,3,6,9,12,18,24,36,48 -> "
                + " ".join("%.3f" % c[i] for i in (0, 2, 5, 8, 11, 17, 23, 35,
                                                   47)))
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(dict(flush=out), f, indent=1)
    return out


def phase_collect(args, log):
    with open(os.path.join(args.out, "train.json")) as f:
        results = json.load(f)
    for fn in sorted(os.listdir(args.out)):
        if fn.startswith("roll_") and fn.endswith(".json"):
            with open(os.path.join(args.out, fn)) as f:
                results["variants"].update(json.load(f))
    pj = os.path.join(args.out, "probe.json")
    if os.path.exists(pj):
        with open(pj) as f:
            results.update(json.load(f))
    order = [k for k in VARIANTS if k in results["variants"]]
    results["variants"] = {k: results["variants"][k] for k in order}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    names = order
    clim = results["toy"]["clim"]
    with open(os.path.join(args.out, "trace200.csv"), "w") as f:
        f.write("step," + ",".join(names) + "\n")
        L = min(len(results["variants"][n]["trace200"]) for n in names)
        for i in range(L):
            f.write(f"{i+1}," + ",".join(
                f"{results['variants'][n]['trace200'][i]:.5f}"
                for n in names) + "\n")
    with open(os.path.join(args.out, "clim_profiles.csv"), "w") as f:
        f.write("site,truth," + ",".join(names) + "\n")
        for i in range(len(clim)):
            f.write(f"{i},{clim[i]:.5f}," + ",".join(
                f"{results['variants'][n]['pred_clim'][i]:.5f}"
                for n in names) + "\n")
    rsn = [n for n in names
           if results["variants"][n]["diag_patamp"].get("anchor_patamp")]
    if rsn:
        with open(os.path.join(args.out, "patamp.csv"), "w") as f:
            f.write("bin," + ",".join(
                f"{n}_emit,{n}_anchor" for n in rsn) + "\n")
            L = min(len(results["variants"][n]["diag_patamp"]["emit_patamp"])
                    for n in rsn)
            for i in range(L):
                f.write(f"{i}," + ",".join(
                    f"{results['variants'][n]['diag_patamp']['emit_patamp'][i]:.5f},"
                    f"{results['variants'][n]['diag_patamp']['anchor_patamp'][i]:.5f}"
                    for n in rsn) + "\n")
    log("\n=== SUMMARY (truth: alpha 0, every ratio 1) ===")
    log("{:<22}{:>8}{:>7}{:>7}{:>7}{:>9}{:>8}{:>9}{:>8}{:>10}".format(
        "variant", "alpha", "R2", "amp", "corr", "gmbias", "varR", "totvarR",
        "seasR", "sprdsat"))
    for n in names:
        m = results["variants"][n]
        log("{:<22}{:>8.3f}{:>7.3f}{:>7.3f}{:>7.3f}{:>9.3f}{:>8.3f}{:>9.3f}"
            "{:>8.3f}{:>10.3f}".format(
                n, m["alpha"], m["r2"], m["pattern_amp_ratio"],
                m["pattern_corr"], m["global_mean_bias"],
                m["resid_var_ratio"], m["total_var_ratio"],
                m["seas_amp_ratio"], m["spread_saturation"]))
    log(f"wrote {args.out}/results.json")
    return results


def main(argv=None):
    args = build_args(argv)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = _logger(args.out, f"run_{args.phase}.log")
    log(f"device={dev} torch={torch.__version__} phase={args.phase}")
    log(f"args={vars(args)}")
    names = ([v for v in args.variants.split(",") if v] if args.variants
             else list(DEFAULT_VARIANTS)
             + (["rsi_train_h1none", "rsi_train_S1"] if args.extras else []))
    t0 = time.time()
    if args.phase == "train":
        phase_train(args, dev, log)
    elif args.phase == "roll":
        phase_roll(args, dev, log, names)
    elif args.phase == "probe":
        phase_probe(args, dev, log)
    elif args.phase == "collect":
        phase_collect(args, log)
    else:
        _, data, nets, _, _, _, _ = phase_train(args, dev, log)
        phase_roll(args, dev, log, names, state=(data, nets))
        phase_probe(args, dev, log)
        phase_collect(args, log)
    log(f"phase {args.phase} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
