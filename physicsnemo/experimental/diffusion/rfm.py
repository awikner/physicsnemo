# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/rfm.py) for Phase 8a.

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Rolling Flow Matching (RFM)
#
# Conventions: flow-time t in [0, 1] with t=1 = clean data, t=0 = pure noise.
# A window holds W consecutive frames in slots w = 0..W-1. Slot w=0 is the FRONT
# (oldest, cleanest, about to be emitted); slot w=W-1 is the BACK (newest,
# noisiest, just entered). The per-slot flow-time staircase advances with the
# roll progress g in [0, 1]:
#
#     t_w(g) = ((W-1-w) + g) / W,   dt_w/dg = 1/W.
#
# At g=1 the front frame reaches t=1 (clean) and is emitted; the staircase shifts
# by one slot (shift identity t_w(0) = t_{w+1}(1)) and a fresh t=0 noise frame
# enters the back. No explicit conditioning frame is fed: the near-clean front
# slots serve as the autoregressive context, attended across slots by the
# backbone's causal temporal attention.
#
# This module implements the scheduler only (schedule, windowed velocity loss,
# rolling sampler). It reuses the ERDM backbone, whose forward contract is:
#
#     u = model(z, t_label, c_grid, c_scalar)
#       z        : (b, W, C, H, W)  -- the interpolant window (fed directly)
#       t_label  : (b, W)           -- per-frame flow-time used as the cond label
#       c_grid   : (b, W, c_grid, H, W)  -- per-frame forcings (or None)
#       c_scalar : (b, W, scalar_dim)    -- per-frame calendar (or None)
#       returns u: (b, W, C, H, W)  -- the predicted per-frame velocity x1 - eps
#
# Velocity / denoiser / noise reparameterizations (per frame, from
# z = t*x1 + (1-t)*eps and u = x1 - eps):
#     D = E[x1   | z, t] = z + (1-t) u            (denoiser, bridge to ERDM)
#         E[eps  | z, t] = z - t u
# Generation integrates dz/dt = u from t=0 (noise) to t=1 (data); for the rolling
# window the inner integration variable is g, so dz/dg = (1/W) u.


class RFMScheduler(nn.Module):
    def __init__(self,
                 window_size=8,
                 num_steps=4,
                 solver="euler",
                 weighting="midrange",
                 time_eps=1e-3,
                 init_mode="oracle",
                 P_mean=0.0,
                 P_std=1.0,
                 noise="gaussian",
                 l_max=45):
        super(RFMScheduler, self).__init__()

        self.W = int(window_size)
        self.num_steps = int(num_steps)
        self.solver = solver
        self.weighting = weighting
        self.time_eps = float(time_eps)
        self.init_mode = init_mode

        # Logit-lognormal weighting params (only used when weighting="lognormal_logit").
        self.P_mean = P_mean
        self.P_std = P_std

        # Slot indices w = 0..W-1 (0-indexed; front = slot 0).
        self.register_buffer("slots", torch.arange(0, self.W, dtype=torch.float32))

        if noise == "spherical":
            from ._utils import SphereNoiseGenerator
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        logger.info(
            "RFMScheduler initialized: W=%s, num_steps=%s, solver=%s, "
            "weighting=%s, init_mode=%s, noise=%s",
            self.W, self.num_steps, self.solver, self.weighting,
            self.init_mode, noise,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def w5(a):
        """Broadcast a per-frame tensor (b, W) to (b, W, 1, 1, 1)."""
        return a[:, :, None, None, None]

    def get_noise(self, ref):
        """Draw independent per-frame noise shaped like ``ref`` (..., C, H, W).

        Per-frame independence is required by RFM: sharing one noise across slots
        collapses the staircase and breaks the staggered-uncertainty structure.
        """
        if self.generator is None:
            return torch.randn_like(ref)
        *lead, H, Wd = ref.shape
        c = lead[-1]
        b = 1
        for s in lead[:-1]:
            b *= s
        noise = self.generator(b, c, device=ref.device).reshape(*lead, H, Wd).to(ref.dtype)
        return noise

    # ------------------------------------------------------------------
    # Rolling flow-time schedule
    # ------------------------------------------------------------------
    def t_steady(self, g):
        """Steady staircase t_w(g) = ((W-1-w) + g) / W, shape (b, W).

        g : (b,) roll progress in [0, 1]. Front slot (w=0) reaches t=1 at g=1.
        """
        w = self.slots.to(g.device)  # (W,)
        return ((self.W - 1 - w)[None, :] + g[:, None]) / self.W

    # ------------------------------------------------------------------
    # Interpolant and velocity field
    # ------------------------------------------------------------------
    def interpolant(self, x1, t, eps):
        """Linear flow-matching interpolant z = t*x1 + (1-t)*eps."""
        return self.w5(t) * x1 + self.w5(1.0 - t) * eps

    def velocity(self, model, z, t, c_grid, c_scalar):
        """Predicted per-frame velocity u_theta(z, t) = E[x1 - eps | z, t].

        z is fed directly (no EDM preconditioning); var(z) = t^2 + (1-t)^2 in
        [0.5, 1] for z-scored data, so inputs stay ~unit scale. The flow-time t is
        passed as the per-frame conditioning label (clamped away from the exact
        endpoints for a stable sinusoidal embedding).
        """

        return model(z, t, c_grid, c_scalar)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss_weight(self, t):
        """Per-frame flow-matching loss weight omega(t), shape (b, W).

        "midrange"        : t(1-t), peaked at t=1/2 (ERDM's mid-horizon emphasis).
        "uniform"         : 1.
        "lognormal_logit" : lognormal in the logit of t (transplanted from EDM).
        """
        if self.weighting == "uniform":
            return torch.ones_like(t)
        if self.weighting == "midrange":
            return t * (1.0 - t)
        if self.weighting == "lognormal_logit":
            tc = t.clamp(self.time_eps, 1.0 - self.time_eps)
            logit = torch.log(tc / (1.0 - tc))
            return torch.exp(-(logit - self.P_mean) ** 2 / (2.0 * self.P_std ** 2)) \
                / (self.P_std * math.sqrt(2.0 * math.pi))
        raise ValueError(f"unknown weighting '{self.weighting}'")

    def sample_times(self, b, device):
        """Sample the per-frame flow-time vector t (b, W) for a training batch.
        """
        g = torch.rand(b, device=device)
        return self.t_steady(g)

    def compute_loss(self, model, c_grid, c_scalar, y):
        """RFM training loss.

        y  : (b, W, C, H, W)     clean window of W consecutive frames x1
        c_grid   : (b, W, c_grid, H, W) per-frame forcings (or None)
        c_scalar : (b, W, scalar_dim) or None
        """
        b = y.shape[0]
        device = y.device

        t = self.sample_times(b, device)            # (b, W)
        eps = self.get_noise(y)                      # (b, W, C, H, W), independent per frame
        z = self.interpolant(y, t, eps)

        u = self.velocity(model, z, t, c_grid, c_scalar)
        target = y - eps                             # straight-line velocity x1 - eps

        weight = self.loss_weight(t)                 # (b, W)
        per_frame_mse = ((u - target) ** 2).sum(dim=[2, 3, 4])  # (b, W)

        loss = (weight * per_frame_mse).mean()
        return loss

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _gather_window(self, traj, k):
        """Slice W consecutive frames starting at k, clamping past the end."""
        if traj is None:
            return None
        T = traj.shape[1]
        idx = torch.arange(k, k + self.W, device=traj.device).clamp(max=T - 1)
        return traj[:, idx]

    def roll_once(self, model, z, t, c_grid_win, c_scalar_win, num_steps=None):
        """Advance the window by one physical step: integrate g from 0 to 1.

        Every slot's flow-time increases by exactly 1/W over the roll (dt_w/dg =
        1/W), so a single forward pass updates the whole window. After the roll
        the front slot has reached t=1.
        """
        if num_steps is None:
            num_steps = self.num_steps

        dg = 1.0 / num_steps
        step = dg / self.W                            # per-slot flow-time increment

        for i in range(num_steps):
            u = self.velocity(model, z, t, c_grid_win, c_scalar_win)
            z_pred = z + step * u
            t_next = (t + step).clamp(max=1.0)

            if self.solver == "heun" and i < num_steps - 1:
                u2 = self.velocity(model, z_pred, t_next, c_grid_win, c_scalar_win)
                z = z + 0.5 * step * (u + u2)
            else:
                z = z_pred
            t = t_next

        return z

    def _warmup_window(self, init_window):
        """Build the initial steady staircase window from clean ICs.

        "oracle" (default): noise the true first W frames to the steady staircase
            t_w(0), z_w = t_w(0)*x1_w + (1-t_w(0))*eps_w.
        """
        b = init_window.shape[0]
        device = init_window.device
        t0 = self.t_steady(torch.zeros(b, device=device))   # (b, W) steady staircase

        if self.init_mode == "oracle":
            eps = self.get_noise(init_window)
            return self.interpolant(init_window, t0, eps)

        raise ValueError(f"unknown init_mode '{self.init_mode}'")

    @torch.no_grad()
    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        """Rolling-window autoregressive sampler.

        init_window  : (b, W, C, H, W)  true first window x1_{0:W} for warm-up
        c_grid_traj  : (b, T, c_grid, H, W)  forcings over absolute future frames
        c_scalar_traj: (b, T, scalar_dim) or None
        horizon      : number of frames to forecast (emit)

        Returns predicted trajectory (b, horizon, C, H, W).
        """
        b = init_window.shape[0]
        device = init_window.device

        # Steady staircase, held fixed at the start of every roll (the shift
        # identity t_w(0) = t_{w+1}(1) makes this exact after each emit-shift).
        t0 = self.t_steady(torch.zeros(b, device=device))   # (b, W)

        z = self._warmup_window(init_window)

        outputs = []
        for k in range(horizon):
            c_grid_win = self._gather_window(c_grid_traj, k)
            c_scalar_win = self._gather_window(c_scalar_traj, k)

            z = self.roll_once(model, z, t0, c_grid_win, c_scalar_win, num_steps)

            emitted = z[:, 0]                # (b, C, H, W) front frame, now at t=1
            outputs.append(emitted)

            # Shift the window forward and append a fresh t=0 noise frame.
            fresh = self.get_noise(z[:, -1:])
            z = torch.cat([z[:, 1:], fresh], dim=1)

        return torch.stack(outputs, dim=1)   # (b, horizon, C, H, W)

    def forward(self, model, init_window, c_grid_traj, c_scalar_traj, horizon, num_steps=None):
        return self.sample_rollout(model, init_window, c_grid_traj, c_scalar_traj,
                                   horizon, num_steps)
