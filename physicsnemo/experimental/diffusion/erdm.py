# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from the amip repo @ commit 497827e
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
# C optionally ends in an `nocean`-wide block of predicted ocean channels -- forcing
# fields (SST / sea ice) the forecaster is asked to predict as well as be forced by.
# They are supervised against the boundary data, and prescribed from truth at the
# top of every roll so the emitted state is forced by the true ocean; see the
# "Ocean channels" section below. nocean == 0 leaves this module numerically
# identical to before (Phase 12f).
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
                 alpha=1.0,
                 nocean=0,
                 ocean_grid_indices=(),
                 ocean_loss_weight=1.0):
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

        # ── Ocean channels (Phase 12f) ────────────────────────────────────
        # ``data.ocean_state_variables`` asks the forecaster to *predict* some
        # of the fields it is forced by (SST, sea ice, optionally the SST
        # anomaly). They live in a block at the TAIL of the state axis, and
        # everything about them is confined to this scheduler, because this is
        # the one place where both the state window and the gridded forcings
        # are in scope:
        #
        #   training   append_ocean_target(y, bnd)  adds the target channels
        #   inference  impose_ocean(x_bar, bnd)     overwrites them with truth
        #   both       ocean_truth(bnd, size)       the ONE definition of truth
        #
        # so the training target and the inference-imposed value are the same
        # expression and cannot drift apart.
        self.nocean = int(nocean or 0)
        self.ocean_grid_indices = list(ocean_grid_indices)
        self.ocean_loss_weight = float(ocean_loss_weight)
        if self.nocean and len(self.ocean_grid_indices) != self.nocean:
            raise ValueError(
                f"nocean={self.nocean} needs exactly that many "
                f"ocean_grid_indices, got {self.ocean_grid_indices}"
            )

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
    # ------------------------------------------------------------------
    # Ocean channels (Phase 12f) — faithful port of amip_v2 erdm.py
    #
    # Time alignment (the subtle part): window slot ``w`` holds the state at
    # step ``w+1`` but is conditioned on the forcing at step ``w``. The ocean
    # target is the boundary at the frame's OWN time, so callers pass the
    # forcing window SHIFTED FORWARD by one — steps 1..W in training,
    # k+1..k+W at roll k. Passing the unshifted window is a silent
    # identity-copy bug: the shapes match either way, which is why
    # :meth:`ocean_truth` checks the frame count and says so.
    # ------------------------------------------------------------------

    def ocean_truth(self, bnd_win, size):
        """True ocean fields for a window: ``(b, W, nocean, *size)`` or None.

        ``bnd_win`` is ``(b, W, c, H, W)`` — the varying-boundary window or the
        assembled ``c_grid`` (the varying channels are a prefix of it, so
        ``ocean_grid_indices`` addresses both), already shifted forward one
        step. Resampled with the same bilinear kernel the state coarsening
        uses, so the predicted SST is co-registered with
        ``skin_temperature`` — NOT ``avg_pool2d``, which has a different
        coastal registration.
        """
        if not self.nocean or bnd_win is None:
            return None
        if bnd_win.shape[1] != self.W:
            raise ValueError(
                f"ocean_truth expects a {self.W}-frame window, got "
                f"{bnd_win.shape[1]}. Callers holding W+1 boundary frames must "
                f"pass the SHIFTED slice (bnd[:, 1:]), not the whole stack — "
                f"see the time-alignment note above."
            )
        idx = torch.as_tensor(self.ocean_grid_indices, device=bnd_win.device)
        o = bnd_win.index_select(2, idx)            # (b, W, nocean, H, W)
        b, W = o.shape[0], o.shape[1]
        if tuple(o.shape[-2:]) != tuple(size):
            o = torch.nn.functional.interpolate(
                o.flatten(0, 1), size=tuple(size), mode="bilinear",
                align_corners=False,
            ).unflatten(0, (b, W))
        return o

    def append_ocean_target(self, y, bnd_next):
        """Append the supervised ocean channels to a clean state window.

        ``y``: ``(b, W, C, h, w)`` -> ``(b, W, C + nocean, h, w)``.
        ``bnd_next`` is the forcing window shifted forward one step.
        """
        if not self.nocean:
            return y
        truth = self.ocean_truth(bnd_next, y.shape[-2:])
        if truth is None:
            raise ValueError(
                "append_ocean_target needs the boundary window to read the "
                "ocean target from, but got None."
            )
        return torch.cat([y, truth.to(y.dtype)], dim=2)

    def pad_state(self, x):
        """Widen a bare state window to the model's channel count with zeros.

        Lets a rollout driver oracle-initialize from the state store (which has
        no SST / sea ice in it) without inventing ocean values: the zeros are
        overwritten by :meth:`impose_ocean` on the first roll. Call on a
        state-width window ONLY — it cannot detect an already-padded one.
        """
        if not self.nocean:
            return x
        b, W = x.shape[0], x.shape[1]
        return torch.cat(
            [x, x.new_zeros(b, W, self.nocean, *x.shape[-2:])], dim=2
        )

    def strip_ocean(self, x):
        """Drop the ocean block, returning a plain state tensor.

        Accepts a window ``(b, W, C, h, w)`` or a single frame
        ``(b, C, h, w)`` — the channel axis is always ``-3``. Every downstream
        consumer (unpack/disassemble, the pretrained downscaler, the NetCDF
        writer) is sized for the state alone, so emitted frames come through
        here first.
        """
        if not self.nocean:
            return x
        return x[..., : x.shape[-3] - self.nocean, :, :]

    def impose_ocean(self, x_bar, bnd_next):
        """Overwrite the window's ocean channels with the true fields.

        Call at the TOP of a roll (global diffusion time ``t=0``) and nowhere
        else. The rolling shift identity is what makes one noise level per slot
        correct there: ``local_time`` gives ``tau_w(t) = 1 - (w - t)/W``, so
        ``sigma_bar_w(1) == sigma_bar_{w-1}(0)`` exactly, and the freshly
        appended back frame carries ``eps * sigma_max == sigma_schedule(0)[W-1]``.
        Hence after every shift slot ``j`` sits at ``sigma_schedule(0)[j]`` and
        truth can be re-noised to match, with no eps history. Imposing inside
        the sampler sweep instead would inject ``sigma_schedule(0)`` noise at a
        frame whose sigma has already been integrated downward, corrupting the
        trajectory.

        The high-sigma back slots are effectively unaffected (truth buried
        under its own noise); that needs no threshold, because imposition is
        total on *every* roll, so each frame is re-imposed at its new lower
        sigma as it advances and carries the truth by the time it is emitted.
        """
        if not self.nocean or bnd_next is None:
            return x_bar
        truth = self.ocean_truth(bnd_next, x_bar.shape[-2:])
        sigma0 = self.sigma_schedule(
            torch.zeros(x_bar.shape[0], device=x_bar.device)
        )                                                    # (b, W)
        # Draw at full width and slice: get_noise multiplies by the per-channel
        # ``noise_scales`` buffer, which would mis-broadcast on a narrow tensor.
        eps = self.temporal_noise(x_bar)[:, :, -self.nocean:]
        x_bar = x_bar.clone()
        x_bar[:, :, -self.nocean:] = truth.to(x_bar.dtype) + self.w5(sigma0) * eps
        return x_bar

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

    def compute_loss(self, model, c_grid, c_scalar, y, return_parts=False):
        """ERDM training loss.

        y  : (b, W, C, H, W)     clean window of W future frames, ocean block
                                 included (see :meth:`append_ocean_target`)
        c_grid   : (b, W, c_grid, H, W) per-frame forcings
        c_scalar : (b, W, scalar_dim) or None

        ``return_parts`` additionally returns the ocean block's contribution as
        a scalar so it can be logged separately. Worth doing (Phase 12f): the
        ocean target is nearly exactly recoverable from a forcing present in
        the same forward pass, so its term collapses fast — without it broken
        out you cannot tell "learned it" from "weighted too low to matter", and
        it is only ~1-2% of a loss that is a raw sum over channels.
        """
        b = y.shape[0]
        device = y.device

        t = torch.rand(b, device=device)            # global diffusion time, (b,)
        sigma = self.sigma_schedule(t)               # (b, W)

        noise = self.temporal_noise(y)               # (b, W, C, H, W), AR(1) across frames
        x_bar = y + self.w5(sigma) * noise

        D = self.denoise(model, x_bar, sigma, c_grid, c_scalar)

        weight = self.loss_weight(sigma)             # (b, W)
        err2 = (D - y) ** 2
        if self.nocean:
            n = err2.shape[2] - self.nocean
            state_mse = err2[:, :, :n].sum(dim=[2, 3, 4])   # (b, W)
            ocean_mse = err2[:, :, n:].sum(dim=[2, 3, 4])   # (b, W)
            per_frame_mse = state_mse + self.ocean_loss_weight * ocean_mse
        else:
            per_frame_mse = err2.sum(dim=[2, 3, 4])         # (b, W)
            ocean_mse = None

        # (1/W) sum_w weight_w * ||.||^2, averaged over the batch.
        loss = (weight * per_frame_mse).mean()
        if not return_parts:
            return loss
        ocean_loss = (
            (weight * ocean_mse).mean() if ocean_mse is not None
            else loss.new_zeros(())
        )
        return loss, ocean_loss

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

    def sample_window(self, model, x_bar, c_grid_win, c_scalar_win, num_steps=None,
                      ocean_win=None):
        """One inner sweep: integrate the probability-flow ODE from t=0 to t=1.

        After the sweep the front frame (w=1) is denoised to sigma_min. Uses the
        EDM ODE in the sigma-parameterization, vectorized over the W frames.

        ``ocean_win`` (Phase 12f) is the forcing window shifted forward one
        step, used to impose the true ocean fields. Every rolling driver funnels
        through this method exactly once per roll, so that single call site is
        what keeps training, validation and the rollout CLIs evaluating the same
        model.
        """
        if num_steps is None:
            num_steps = self.num_steps

        b = x_bar.shape[0]
        device = x_bar.device
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        churn = self.S_churn > 0.0

        # Prescribe the ocean channels BEFORE integrating: this is the top of a
        # roll (global t=0), the only point where sigma_schedule(0) is the
        # correct level for every slot (see impose_ocean — never move this into
        # the loop). Churn below re-noises the imposed block along with
        # everything else, so emitted ocean channels are not bit-exact truth;
        # harmless, since they are stripped and discarded and imposition is
        # total on every roll. It is why downstream reads ocean truth from the
        # dataset rather than from an emitted frame.
        if self.nocean:
            x_bar = self.impose_ocean(x_bar, ocean_win)

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

        init_window  : (b, W, C, H, W)  oracle true first window y_{1:W}. Under
            ``nocean > 0`` pass a BARE state window (no ocean block) — it is
            padded here and the zeros are overwritten by the first imposition.
        c_grid_traj  : (b, T, c_grid, H, W)  forcings over absolute future frames
        c_scalar_traj: (b, T, scalar_dim) or None
        horizon      : number of frames to forecast (emit)
        num_steps    : diffusion solver steps per emitted frame. Either a
            single int/None (applied uniformly, previous behavior) or a
            sequence of length ``horizon`` giving a per-emitted-frame
            step count (Phase 8f, F4 — caps sampling cost at long
            horizons).

        Returns predicted trajectory (b, horizon, C, H, W) — ocean block
        included; call :meth:`strip_ocean` on it before handing frames to a
        consumer sized for the state alone.
        """
        b = init_window.shape[0]
        device = init_window.device

        if isinstance(num_steps, (list, tuple)) and len(num_steps) != horizon:
            raise ValueError(
                f"num_steps schedule has length {len(num_steps)}, expected "
                f"horizon={horizon}"
            )

        # Schedule-matched noising of the oracle window at global time t=0, using
        # the temporal noise prior so the W frames are AR(1)-correlated.
        init_window = self.pad_state(init_window)
        sigma0 = self.sigma_schedule(torch.zeros(b, device=device))  # (b, W)
        eps_win = self.temporal_noise(init_window)                   # (b, W, C, H, W)
        x_bar = init_window + self.w5(sigma0) * eps_win
        eps_prev = eps_win[:, -1:]          # (b, 1, C, H, W) seed to continue the chain

        outputs = []
        for k in range(horizon):
            c_grid_win = self._gather_window(c_grid_traj, k)
            c_scalar_win = self._gather_window(c_scalar_traj, k)
            # Shifted by one: slot j holds the frame at step k+1+j, and its ocean
            # target is the boundary at that same time.
            ocean_win = (
                self._gather_window(c_grid_traj, k + 1) if self.nocean else None
            )

            step_k = num_steps[k] if isinstance(num_steps, (list, tuple)) else num_steps
            x_bar = self.sample_window(model, x_bar, c_grid_win, c_scalar_win, step_k,
                                       ocean_win=ocean_win)

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
        init_window = self.pad_state(init_window)
        sigma0 = self.sigma_schedule(torch.zeros(b, device=device))  # (b, W)
        eps_win = self.temporal_noise(init_window)                   # (b, W, C, H, W)
        x_bar = init_window + self.w5(sigma0) * eps_win
        eps_prev = eps_win[:, -1:]          # (b, 1, C, H, W) seed to continue the chain

        for k in range(horizon):
            # The forcing window is either pulled lazily (so the GPU only ever holds
            # the W-frame window, never the full horizon) or sliced from a trajectory.
            # A provider may return a third element, the one-step-shifted window used
            # to impose the true ocean fields; two-element returns stay valid.
            if forcing_provider is not None:
                provided = forcing_provider(k)
                c_grid_win, c_scalar_win = provided[0], provided[1]
                ocean_win = provided[2] if len(provided) > 2 else None
            else:
                c_grid_win = self._gather_window(c_grid_traj, k)
                c_scalar_win = self._gather_window(c_scalar_traj, k)
                ocean_win = (
                    self._gather_window(c_grid_traj, k + 1) if self.nocean else None
                )
            if c_grid_win is not None and c_grid_win.device != device:
                c_grid_win = c_grid_win.to(device, non_blocking=True)
            if c_scalar_win is not None and c_scalar_win.device != device:
                c_scalar_win = c_scalar_win.to(device, non_blocking=True)
            if ocean_win is not None and ocean_win.device != device:
                ocean_win = ocean_win.to(device, non_blocking=True)

            x_bar = self.sample_window(
                model, x_bar, c_grid_win, c_scalar_win, num_steps, ocean_win=ocean_win)

            emitted = x_bar[:, 0]            # (b, C, H, W) clean front frame
            yield k, emitted

            # Shift the window forward by one and append a fresh max-noise frame whose
            # noise continues the AR(1) chain from the previous back frame.
            eps_prev = self.temporal_noise_next(eps_prev)
            fresh = eps_prev * self.sigma_max
            x_bar = torch.cat([x_bar[:, 1:], fresh], dim=1)
