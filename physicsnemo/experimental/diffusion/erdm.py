# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/erdm.py) for Phase 8a.

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Elucidated Rolling Diffusion Model (ERDM), https://arxiv.org/abs/2506.20024
#
# ERDM combines the EDM noise schedule / preconditioning / sampler with rolling
# diffusion. Instead of denoising a single future state, it operates on a temporal
# window of `window_size` (W) frames. A single global diffusion time t in [0, 1)
# controls all frames: frame w carries noise level sigma_bar_w(t), with the front
# frame (w=1) nearly clean and the back frame (w=W) at sigma_max. Advancing t from
# 0 -> 1 fully denoises the front frame (which is emitted) and rotates the schedule
# by exactly one frame, so the window can slide forward (the "shift identity"
# sigma_bar_w(1) = sigma_bar_{w-1}(0)).
#
# This module implements the scheduler only (noise schedule, windowed loss, and
# the rolling sampler). The backbone is assumed to satisfy the contract:
#
#     F = model(x_noised, c_noise, c_grid, c_scalar)
#       x_noised : (b, W, C, H, W)  -- preconditioned noised window (c_in * x_bar)
#       c_noise  : (b, W)           -- per-frame noise label ln(sigma)/4
#       c_grid   : (b, W, c_grid, H, W)  -- per-frame forcings
#       c_scalar : (b, W, scalar_dim)    -- per-frame calendar (or None)
#       returns F: (b, W, C, H, W)  -- raw network output (the F_theta of EDM)
#
# No clean conditioning frame is passed: at the first window the partially noised
# rolling window already carries enough information (the front frames are nearly
# clean) to forecast how to denoise the sequence.


class ERDMScheduler(nn.Module):
    def __init__(self,
                 window_size=6,
                 num_steps=2,
                 sigma_min=0.002,
                 sigma_max=500.0,
                 rho=-10.0,
                 sigma_data=0.5,
                 P_mean=2.0,
                 P_std=1.2,
                 solver="heun",
                 S_churn=0.0,
                 S_tmin=0.0,
                 S_tmax=float("inf"),
                 S_noise=1.0,
                 noise="gaussian",
                 l_max=45,
                 noise_scale_path=None,
                 alpha=1.0):
        super(ERDMScheduler, self).__init__()

        self.W = int(window_size)
        self.num_steps = int(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.solver = solver

        # Temporal noise prior (ERDM App. C.4, Eq. 24): AR(1) correlation of the
        # diffusion noise across the W frames. alpha controls the correlation
        # strength; alpha=0 recovers independent per-frame noise.
        self.alpha = alpha

        # Optional stochastic churn (Karras et al. EDM sampler, Alg. 2)
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # Frame indices w = 1..W (1-indexed as in the paper).
        self.register_buffer("frames", torch.arange(1, self.W + 1, dtype=torch.float32))

        if noise == "spherical":
            from ._utils import SphereNoiseGenerator
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        if noise_scale_path is not None:
            noise_scales = torch.load(noise_scale_path)
            self.register_buffer("noise_scales", noise_scales)
        else:
            self.noise_scales = None

        logger.info(
            "ERDMScheduler initialized: W=%s, num_steps=%s, rho=%s, sigma=[%s, %s], "
            "sigma_data=%s, solver=%s, noise=%s, alpha=%s",
            self.W, self.num_steps, self.rho, self.sigma_min, self.sigma_max,
            self.sigma_data, self.solver, noise, self.alpha,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def w5(a):
        """Broadcast a per-frame tensor (b, W) to (b, W, 1, 1, 1)."""
        return a[:, :, None, None, None]

    def get_noise(self, ref):
        """Draw noise shaped like ``ref`` (..., C, H, W)."""
        if self.generator is None:
            noise = torch.randn_like(ref)
        else:
            *lead, H, Wd = ref.shape
            c = lead[-1]
            b = 1
            for s in lead[:-1]:
                b *= s
            noise = self.generator(b, c, device=ref.device).reshape(*lead, H, Wd).to(ref.dtype)
        if self.noise_scales is not None:
            noise = noise * self.noise_scales.to(ref.device)
        return noise

    def _ar_coeffs(self):
        """AR(1) coefficients (c, s) for the temporal noise prior (Eq. 24).

        eps^k = c * eps^{k-1} + s * eta^k, with eta^k ~ N(0, I) i.i.d. Then
        c = alpha / sqrt(1+alpha^2), s = 1/sqrt(1+alpha^2), so the independent
        term s*eta^k ~ N(0, 1/(1+alpha^2) I) and each eps^k is marginally N(0, I).
        """
        denom = math.sqrt(1.0 + self.alpha ** 2)
        return self.alpha / denom, 1.0 / denom

    def temporal_noise(self, ref):
        """Window of temporally-correlated noise (ERDM App. C.4, Eq. 24).

        Draws AR(1) noise along the frame axis (dim=1) of ``ref`` (b, W, C, H, W):
        the first frame is N(0, I) and each subsequent frame is correlated with the
        previous one. Marginally N(0, I) per frame, so it is a drop-in for an i.i.d.
        ``get_noise`` draw. Returns the noise and the last frame's latent so a
        rolling sampler can continue the same AR chain.
        """
        eta = self.get_noise(ref)                    # (b, W, C, H, W) i.i.d.
        if self.alpha == 0.0:
            return eta
        c, s = self._ar_coeffs()
        frames = list(eta.unbind(dim=1))
        out = [frames[0]]                            # eps^1 = eta^1 ~ N(0, I)
        for k in range(1, len(frames)):
            out.append(c * out[-1] + s * frames[k])
        return torch.stack(out, dim=1)

    def temporal_noise_next(self, prev_eps):
        """Next frame of the AR(1) chain given the previous frame's noise latent.

        prev_eps : (b, 1, C, H, W) noise of the current back frame. Returns the
        next frame's noise (same shape), continuing the temporal prior so freshly
        appended rolling-window frames stay correlated with the sequence.
        """
        eta = self.get_noise(prev_eps)
        if self.alpha == 0.0:
            return eta
        c, s = self._ar_coeffs()
        return c * prev_eps + s * eta

    # ------------------------------------------------------------------
    # Rolling noise schedule
    # ------------------------------------------------------------------
    def local_time(self, t):
        """Per-frame local diffusion time tau_w(t) = 1 - (w - t)/W.

        t : (b,) global diffusion time. Returns (b, W). Larger tau -> less noise.
        """
        w = self.frames.to(t.device)  # (W,)
        return 1.0 - (w[None, :] - t[:, None]) / self.W

    def sigma_from_tau(self, tau):
        """EDM rho-schedule mapping local time tau in [0,1] to sigma."""
        tau = tau.clamp(0.0, 1.0)
        smin = self.sigma_min ** (1.0 / self.rho)
        smax = self.sigma_max ** (1.0 / self.rho)
        return (smax + tau * (smin - smax)) ** self.rho

    def sigma_schedule(self, t):
        """Per-frame noise level sigma_bar(t), shape (b, W)."""
        return self.sigma_from_tau(self.local_time(t))

    # ------------------------------------------------------------------
    # EDM preconditioning / denoiser
    # ------------------------------------------------------------------
    def precondition(self, sigma):
        """Return (c_in, c_skip, c_out, c_noise), each shape (b, W)."""
        sd2 = self.sigma_data ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd2).sqrt()
        c_noise = sigma.log() / 4.0
        return c_in, c_skip, c_out, c_noise

    def denoise(self, model, x_bar, sigma, c_grid, c_scalar):
        """EDM denoiser D_theta on the window. x_bar: (b, W, C, H, W)."""
        c_in, c_skip, c_out, c_noise = self.precondition(sigma)
        F = model(self.w5(c_in) * x_bar, c_noise, c_grid, c_scalar)
        return self.w5(c_skip) * x_bar + self.w5(c_out) * F

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss_weight(self, sigma):
        """EDM unit-variance weight lambda(sigma) * lognormal emphasis f(sigma)."""
        lam = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        f = torch.exp(-(sigma.log() - self.P_mean) ** 2 / (2.0 * self.P_std ** 2)) \
            / (sigma * self.P_std * math.sqrt(2.0 * math.pi))
        return lam * f

    def compute_loss(self, model, c_grid, c_scalar, y):
        """ERDM training loss.

        y  : (b, W, C, H, W)     clean window of W future frames
        c_grid   : (b, W, c_grid, H, W) per-frame forcings
        c_scalar : (b, W, scalar_dim) or None
        """
        b = y.shape[0]
        device = y.device

        t = torch.rand(b, device=device)            # global diffusion time, (b,)
        sigma = self.sigma_schedule(t)               # (b, W)

        noise = self.temporal_noise(y)               # (b, W, C, H, W), AR(1) across frames
        x_bar = y + self.w5(sigma) * noise

        D = self.denoise(model, x_bar, sigma, c_grid, c_scalar)

        weight = self.loss_weight(sigma)             # (b, W)
        per_frame_mse = ((D - y) ** 2).sum(dim=[2, 3, 4])  # (b, W)

        # (1/W) sum_w weight_w * ||.||^2, averaged over the batch.
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

    def sample_window(self, model, x_bar, c_grid_win, c_scalar_win, num_steps=None):
        """One inner sweep: integrate the probability-flow ODE from t=0 to t=1.

        After the sweep the front frame (w=1) is denoised to sigma_min. Uses the
        EDM ODE in the sigma-parameterization, vectorized over the W frames.
        """
        if num_steps is None:
            num_steps = self.num_steps

        b = x_bar.shape[0]
        device = x_bar.device
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        churn = self.S_churn > 0.0

        for i in range(num_steps):
            t_cur = timesteps[i].expand(b)
            t_next = timesteps[i + 1].expand(b)
            sigma_cur = self.sigma_schedule(t_cur)    # (b, W)
            sigma_next = self.sigma_schedule(t_next)  # (b, W)

            # Optional stochastic churn: bump each frame up to sigma_hat >= sigma_cur.
            if churn:
                gamma = torch.where(
                    (sigma_cur >= self.S_tmin) & (sigma_cur <= self.S_tmax),
                    torch.full_like(sigma_cur, min(self.S_churn / num_steps, math.sqrt(2.0) - 1.0)),
                    torch.zeros_like(sigma_cur),
                )
                sigma_hat = sigma_cur + gamma * sigma_cur
                eps = self.temporal_noise(x_bar) * self.S_noise
                x_bar = x_bar + self.w5((sigma_hat ** 2 - sigma_cur ** 2).clamp(min=0.0).sqrt()) * eps
                sigma_cur = sigma_hat

            # Euler predictor: dx = (x - D) / sigma * dsigma.
            D = self.denoise(model, x_bar, sigma_cur, c_grid_win, c_scalar_win)
            d_cur = (x_bar - D) / self.w5(sigma_cur)
            x_euler = x_bar + self.w5(sigma_next - sigma_cur) * d_cur

            if self.solver == "heun" and i < num_steps - 1:
                # Second-order correction (sigma_next > sigma_min > 0, so safe to divide).
                D_next = self.denoise(model, x_euler, sigma_next, c_grid_win, c_scalar_win)
                d_next = (x_euler - D_next) / self.w5(sigma_next)
                x_bar = x_bar + self.w5(sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)
            else:
                x_bar = x_euler

        return x_bar

    @torch.no_grad()
    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        """Rolling-window autoregressive sampler.

        init_window  : (b, W, C, H, W)  oracle true first window y_{1:W}
        c_grid_traj  : (b, T, c_grid, H, W)  forcings over absolute future frames
        c_scalar_traj: (b, T, scalar_dim) or None
        horizon      : number of frames to forecast (emit)

        Returns predicted trajectory (b, horizon, C, H, W).
        """
        b = init_window.shape[0]
        device = init_window.device

        # Schedule-matched noising of the oracle window at global time t=0, using
        # the temporal noise prior so the W frames are AR(1)-correlated.
        sigma0 = self.sigma_schedule(torch.zeros(b, device=device))  # (b, W)
        eps_win = self.temporal_noise(init_window)                   # (b, W, C, H, W)
        x_bar = init_window + self.w5(sigma0) * eps_win
        eps_prev = eps_win[:, -1:]          # (b, 1, C, H, W) seed to continue the chain

        outputs = []
        for k in range(horizon):
            c_grid_win = self._gather_window(c_grid_traj, k)
            c_scalar_win = self._gather_window(c_scalar_traj, k)

            x_bar = self.sample_window(model, x_bar, c_grid_win, c_scalar_win, num_steps)

            emitted = x_bar[:, 0]            # (b, C, H, W) clean front frame
            outputs.append(emitted)

            # Shift the window forward by one and append a fresh max-noise frame whose
            # noise continues the AR(1) chain from the previous back frame.
            eps_prev = self.temporal_noise_next(eps_prev)
            fresh = eps_prev * self.sigma_max
            x_bar = torch.cat([x_bar[:, 1:], fresh], dim=1)

        return torch.stack(outputs, dim=1)   # (b, horizon, C, H, W)

    def forward(self, model, init_window, c_grid_traj, c_scalar_traj, horizon, num_steps=None):
        return self.sample_rollout(model, init_window, c_grid_traj, c_scalar_traj,
                                   horizon, num_steps)

    @torch.no_grad()
    def sample_rollout_generator(self, model, init_window, c_grid_traj, c_scalar_traj,
                             horizon, num_steps=None, forcing_provider=None):
        
        b = init_window.shape[0]
        device = init_window.device

        # Schedule-matched noising of the oracle window at global time t=0.
        sigma0 = self.sigma_schedule(torch.zeros(b, device=device))  # (b, W)
        eps_win = self.temporal_noise(init_window)                   # (b, W, C, H, W)
        x_bar = init_window + self.w5(sigma0) * eps_win
        eps_prev = eps_win[:, -1:]          # (b, 1, C, H, W) seed to continue the chain

        for k in range(horizon):
            # The forcing window is either pulled lazily (so the GPU only ever holds
            # the W-frame window, never the full horizon) or sliced from a trajectory.
            if forcing_provider is not None:
                c_grid_win, c_scalar_win = forcing_provider(k)
            else:
                c_grid_win = self._gather_window(c_grid_traj, k)
                c_scalar_win = self._gather_window(c_scalar_traj, k)
            if c_grid_win is not None and c_grid_win.device != device:
                c_grid_win = c_grid_win.to(device, non_blocking=True)
            if c_scalar_win is not None and c_scalar_win.device != device:
                c_scalar_win = c_scalar_win.to(device, non_blocking=True)

            x_bar = self.sample_window(
                model, x_bar, c_grid_win, c_scalar_win, num_steps)

            emitted = x_bar[:, 0]            # (b, C, H, W) clean front frame
            yield k, emitted

            # Shift the window forward by one and append a fresh max-noise frame whose
            # noise continues the AR(1) chain from the previous back frame.
            eps_prev = self.temporal_noise_next(eps_prev)
            fresh = eps_prev * self.sigma_max
            x_bar = torch.cat([x_bar[:, 1:], fresh], dim=1)
