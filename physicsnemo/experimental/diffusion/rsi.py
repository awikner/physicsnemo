# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Per-slot loss decomposition cadence; 0 (default) disables. See compute_loss.
_LOSS_DIAG_EVERY = int(os.environ.get("RSI_LOSS_DIAG", "0") or 0)
_LOSS_DIAG_CALLS = 0

# Rolling Stochastic Interpolants (RSI)
#
# A stochastic-interpolant reformulation of ERDM (physicsnemo.experimental.
# diffusion.erdm). ERDM transports every window slot from isotropic Gaussian
# noise to data; RSI transports each slot from (a perturbed copy of) its
# temporal PREDECESSOR to its target as it traverses the window. The window
# bookkeeping -- local time, telescoping shift identity, one global diffusion
# time per pass -- is ERDM's, verbatim.
#
# Slots are 1-indexed w = 1..W as in the ERDM paper. Local time is ERDM's
#
#     tau_w(t) = 1 - (w - t)/W = (W - w + t)/W,     dtau_w/dt = 1/W,
#
# with tau = 0 the "base" end (pure anchor + full noise) and tau = 1 the clean
# target end -- the same orientation as ERDM's sigma schedule, where tau = 0 maps
# to sigma_max and tau = 1 to sigma_min. The shift identity tau_w(1) = tau_{w-1}(0)
# is therefore inherited unchanged, which is what makes the window shift exact.
#
# Forward interpolant, per slot, with anchor a_w = y_{w-1} (the true predecessor):
#
#     x_w(tau) = a_w + beta(tau) * (y_w - a_w) + Gamma(tau) z_w,   z_w ~ N(0, I)
#
# with beta(0) = 0, beta(1) = 1, Gamma(tau) = gamma(tau) * S (S = per-channel
# increment scale; spectral shaping is Phase C). Gamma(0) = gamma_0 > 0 is
# mandatory: with strongly correlated endpoints a vanishing latent makes the
# conditional law degenerate and the regression ill-conditioned. Gamma(1) =
# gamma_1 is a small nonzero noise floor -- ERDM's sigma_min analogue.
#
# The backbone emits TWO heads (2*C channels; see the RollingDiT
# ``output_head.num_output_heads: 2`` option):
#
#     H1  -- either the clean state y_w ("state" parameterization) or the
#            increment Delta_w = y_w - a_w ("residual" parameterization)
#     zhat -- the white latent E[z_w | x_window]
#
# from which everything else is closed-form:
#
#     yhat = x + (1 - beta) Delta_hat - Gamma zhat        (denoised state)
#     b    = beta_dot Delta_hat + Gamma_dot zhat           (velocity)
#     score = -Gamma^{-1} zhat                             (Anderson identity)
#
# Sampling integrates one window pass from t = 0 to t = 1 with N steps. The
# default "coeff" integrator advances with the EXACT coefficient increments
#
#     x <- x + (beta(tau_next) - beta(tau_cur)) Delta_hat
#            + (Gamma(tau_next) - Gamma(tau_cur)) zhat
#
# rather than first-order Euler on the velocity. This is exact whenever the
# conditional expectations are constant over the step, and -- the reason it is
# the default -- it makes the ERDM reduction below hold to floating point rather
# than to O(dt), so ablation A1 is a real consistency check.
#
# ERDM is recovered exactly by ``reduce_to_erdm=True``: anchors a_w = 0,
# beta == 1, Gamma(tau_w) = sigma_bar_w (the EDM rho-schedule). Then
# x = y + sigma z, yhat = x - sigma zhat (so zhat is EDM's eps-prediction),
# the coefficient-increment update becomes ERDM's Euler/Heun step on
# dx = (x - D)/sigma dsigma, and the loss weight reduces to lambda(sigma) f(sigma).
#
# Backbone contract. The interpolant state is INPUT-PRECONDITIONED -- the
# backbone receives c_in(tau) * x with c_in = 1/sqrt(Gamma^2 + sigma_data^2), so
# it always sees ~unit-variance inputs and, under reduce_to_erdm, exactly EDM's
# c_in. (Feeding x raw, as flow matching does, only holds while the state stays
# O(1); it breaks outright once Gamma reaches sigma_max -- see c_in's docstring.)
# The heads' targets are unaffected, so this is input scaling only:
#
#     out = model(c_in(tau) * x, label, c_grid, c_scalar)
#       x        : (b, W, C, H, W)   -- the interpolant window
#       label    : (b, W)            -- per-slot conditioning label (see label_mode)
#       c_grid   : (b, W, c_grid, H, W)  -- per-slot forcings
#       c_scalar : (b, W, scalar_dim)    -- per-slot calendar (or None)
#       returns  : (b, W, 2*C, H, W) -- [H1 | zhat] on the channel axis
#
# Ocean channels (Phase 12f) are carried through with the same three-call
# discipline as ERDM -- append_ocean_target (training), impose_ocean
# (inference), ocean_truth (the one definition of truth) -- but the imposed
# value is the INTERPOLANT between the anchor-time and own-time truths, not
# truth-plus-noise; see :meth:`impose_ocean`.


def _as_tensor_like(x, ref):
    """Broadcast a python float or tensor to ``ref``'s device/dtype."""
    if torch.is_tensor(x):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


class RSIScheduler(nn.Module):
    """Rolling Stochastic Interpolant scheduler.

    Drop-in alternative to :class:`ERDMScheduler` under the recipe's ``loss=`` /
    ``sampler=`` groups. Differs from ERDM's contract in exactly two places, both
    advertised through attributes the drivers read:

    ``anchor_frames = 1``
        :meth:`compute_loss` wants ``W + 1`` state frames, not ``W``: slot 1's
        anchor is the frame *before* the window.
    ``init_frames = W + 1``
        :meth:`sample_rollout` wants ``W + 1`` oracle frames for the same reason.
    """

    def __init__(self,
                 window_size=6,
                 num_steps=2,
                 solver="heun",
                 integrator="coeff",
                 final_denoise=False,
                 # ── interpolant profiles ──────────────────────────────────
                 parameterization="state",
                 h1_precond="none",
                 beta="linear",
                 beta_power=2.0,
                 beta_floor=1e-3,
                 gamma_0=0.5,
                 gamma_1=0.02,
                 gamma_profile="geometric",
                 delta_std=1.0,
                 sigma_data=1.0,
                 time_eps=1e-3,
                 label_mode="tau",
                 # ── training ──────────────────────────────────────────────
                 anchor_noise=0.0,
                 weighting="snr_bump",
                 P_mean=2.0,
                 P_std=1.2,
                 w_1=1.0,
                 w_z=1.0,
                 # ── sampling stochasticity (eps-family; 0 => PF-ODE) ──────
                 eps_scale=0.0,
                 eps_tmin=0.0,
                 eps_tmax=1.0,
                 # ── noise machinery ───────────────────────────────────────
                 noise="gaussian",
                 l_max=45,
                 noise_scale_path=None,
                 spectrum_path=None,
                 spectral_sharpness=None,
                 spectral_lmax=None,
                 grid=None,
                 # ── ocean channels (Phase 12f contract) ───────────────────
                 nocean=0,
                 ocean_grid_indices=(),
                 ocean_loss_weight=1.0,
                 # ── ERDM reduction (ablation A1 / consistency test) ───────
                 reduce_to_erdm=False,
                 sigma_min=0.002,
                 sigma_max=500.0,
                 rho=-10.0):
        super(RSIScheduler, self).__init__()

        self.W = int(window_size)
        # Every rolling driver reads ``window_size`` off the scheduler
        # (validate_diffusion, inference, rollout). ERDM/RFM only ever set
        # ``W``, which is a latent bug in those paths; RSI sets both.
        self.window_size = self.W
        # The two contract deltas vs ERDM, advertised for the drivers.
        self.init_frames = self.W + 1
        self.anchor_frames = 1

        self.num_steps = int(num_steps)
        self.solver = solver
        if integrator not in ("coeff", "euler"):
            raise ValueError(
                f"integrator must be 'coeff' or 'euler', got {integrator!r}"
            )
        self.integrator = integrator
        self.final_denoise = bool(final_denoise)

        if parameterization not in ("state", "residual"):
            raise ValueError(
                "parameterization must be 'state' or 'residual', got "
                f"{parameterization!r}"
            )
        self.parameterization = parameterization
        if h1_precond not in ("none", "edm"):
            raise ValueError(
                f"h1_precond must be 'none' or 'edm', got {h1_precond!r}"
            )
        if h1_precond == "edm" and parameterization != "state":
            raise ValueError(
                "h1_precond='edm' reads H1 out as an EDM denoiser of the clean "
                "STATE; under parameterization='residual' H1 is the increment "
                "and has its own scale. Use it with the state parameterization."
            )
        self.h1_precond = h1_precond
        self.beta_mode = beta
        self.beta_power = float(beta_power)
        self.beta_floor = float(beta_floor)
        self.gamma_0 = float(gamma_0)
        self.gamma_1 = float(gamma_1)
        self.gamma_mode = gamma_profile
        self.delta_std = float(delta_std)
        self.sigma_data = float(sigma_data)
        self.time_eps = float(time_eps)
        if label_mode not in ("tau", "log_sigma_eff"):
            raise ValueError(
                f"label_mode must be 'tau' or 'log_sigma_eff', got {label_mode!r}"
            )
        self.label_mode = label_mode

        self.anchor_noise = float(anchor_noise)
        self.weighting = weighting
        self.P_mean = P_mean
        self.P_std = P_std
        self.w_1 = float(w_1)
        self.w_z = float(w_z)

        self.eps_scale = float(eps_scale)
        self.eps_tmin = float(eps_tmin)
        self.eps_tmax = float(eps_tmax)

        # ── ERDM reduction ────────────────────────────────────────────────
        self.reduce_to_erdm = bool(reduce_to_erdm)
        if self.reduce_to_erdm and (spectrum_path is not None
                                    or spectral_sharpness is not None):
            raise ValueError(
                "reduce_to_erdm=True is the WHITE-noise limit by definition; "
                "spectral shaping contradicts it. Ablation A1 and A4 are "
                "different rungs."
            )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        # ── Ocean channels: identical contract to ERDMScheduler ───────────
        self.nocean = int(nocean or 0)
        self.ocean_grid_indices = list(ocean_grid_indices)
        self.ocean_loss_weight = float(ocean_loss_weight)
        if self.nocean and len(self.ocean_grid_indices) != self.nocean:
            raise ValueError(
                f"nocean={self.nocean} needs exactly that many "
                f"ocean_grid_indices, got {self.ocean_grid_indices}"
            )

        # Slot indices w = 1..W (1-indexed as in the ERDM paper).
        self.register_buffer("frames", torch.arange(1, self.W + 1, dtype=torch.float32))

        if noise == "spherical":
            from ._utils import SphereNoiseGenerator
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        # Per-channel latent amplitude S: the std of the model's own one-step
        # increment in normalized units, i.e. tools/data/amip/make_noise_scales.py's
        # artifact. Injecting the latent at the natural per-channel increment
        # scale is the whole point -- isotropic noise over-perturbs the slow
        # channels. Also the unit in which the residual target is measured.
        if noise_scale_path is not None:
            scales = torch.load(noise_scale_path)
            self.register_buffer("delta_scale", scales)
        else:
            self.delta_scale = None

        # Spectral Gamma = gamma_0 * g(l) * h(tau, l): the 2-D schedule over
        # (lead time, wavenumber). None => the scalar white profile above.
        # ``spectral_sharpness=None`` means "not requested" -- distinct from
        # ``0.0``, which explicitly asks for a flat (envelope-only) filter and
        # is the reduction case the tests pin. A spectrum with no sharpness
        # stated gets the default staggering.
        self.spectral = None
        if spectrum_path is not None and spectral_sharpness is None:
            spectral_sharpness = 2.0
        if spectrum_path is not None or spectral_sharpness is not None:
            if grid is None:
                raise ValueError(
                    "spectral Gamma needs the state grid: pass grid=(nlat, nlon) "
                    "(the model config's horizontal_resolution)."
                )
            from ._spectral import SphericalSpectralFilter
            envelope = None
            if spectrum_path is not None:
                blob = torch.load(spectrum_path)
                envelope = blob["envelope"] if isinstance(blob, dict) else blob
            self.spectral = SphericalSpectralFilter(
                int(grid[0]), int(grid[1]),
                gamma_0=self.gamma_0, gamma_1=self.gamma_1,
                envelope=envelope, sharpness=float(spectral_sharpness),
                lmax=spectral_lmax,
            )

        logger.info(
            "RSIScheduler initialized: W=%s, num_steps=%s, param=%s (h1_precond=%s), "
            "beta=%s, gamma=[%s -> %s] (%s), integrator=%s, solver=%s, weighting=%s, "
            "eps_scale=%s, noise=%s, reduce_to_erdm=%s",
            self.W, self.num_steps, self.parameterization, self.h1_precond,
            self.beta_mode,
            self.gamma_0, self.gamma_1, self.gamma_mode, self.integrator,
            self.solver, self.weighting, self.eps_scale, noise, self.reduce_to_erdm,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def w5(a):
        """Broadcast a per-slot tensor (b, W) to (b, W, 1, 1, 1)."""
        return a[:, :, None, None, None]

    def get_noise(self, ref):
        """Draw WHITE latent noise shaped like ``ref`` (..., C, H, W).

        Unlike ERDM's ``get_noise`` this does NOT fold in the per-channel
        scale: the network regresses the unit-variance latent ``z`` and the
        amplitude lives in ``Gamma``, which keeps the score identity
        ``-Gamma^{-1} E[z|x]`` correct and the head's target well-conditioned.
        """
        if self.generator is None:
            return torch.randn_like(ref)
        *lead, H, Wd = ref.shape
        c = lead[-1]
        b = 1
        for s in lead[:-1]:
            b *= s
        return self.generator(b, c, device=ref.device).reshape(*lead, H, Wd).to(ref.dtype)

    def _gather_window(self, traj, k):
        """Slice W consecutive frames starting at k, clamping past the end."""
        if traj is None:
            return None
        T = traj.shape[1]
        idx = torch.arange(k, k + self.W, device=traj.device).clamp(max=T - 1)
        return traj[:, idx]

    # ------------------------------------------------------------------
    # Rolling schedule
    # ------------------------------------------------------------------
    def local_time(self, t):
        """Per-slot local interpolant time tau_w(t) = 1 - (w - t)/W.

        t : (b,) global time of the window pass. Returns (b, W). tau = 0 is the
        base end (anchor + full noise), tau = 1 the clean end -- ERDM's
        orientation, so the telescoping property transfers verbatim.
        """
        w = self.frames.to(t.device)  # (W,)
        return 1.0 - (w[None, :] - t[:, None]) / self.W

    def sigma_from_tau(self, tau):
        """EDM rho-schedule (only used under ``reduce_to_erdm``)."""
        tau = tau.clamp(0.0, 1.0)
        smin = self.sigma_min ** (1.0 / self.rho)
        smax = self.sigma_max ** (1.0 / self.rho)
        return (smax + tau * (smin - smax)) ** self.rho

    # ── interpolant coefficient profiles ──────────────────────────────────
    def beta(self, tau):
        """Signal profile beta(tau), (b, W). beta(0) = 0, beta(1) = 1."""
        if self.reduce_to_erdm:
            return torch.ones_like(tau)
        tau = tau.clamp(0.0, 1.0)
        if self.beta_mode == "linear":
            return tau
        if self.beta_mode == "power":
            return tau ** self.beta_power
        if self.beta_mode == "smoothstep":
            return tau * tau * (3.0 - 2.0 * tau)
        raise ValueError(f"unknown beta profile '{self.beta_mode}'")

    def beta_dot(self, tau):
        """d beta / d tau, (b, W)."""
        if self.reduce_to_erdm:
            return torch.zeros_like(tau)
        tau = tau.clamp(0.0, 1.0)
        if self.beta_mode == "linear":
            return torch.ones_like(tau)
        if self.beta_mode == "power":
            return self.beta_power * tau.clamp(min=self.time_eps) ** (self.beta_power - 1.0)
        if self.beta_mode == "smoothstep":
            return 6.0 * tau * (1.0 - tau)
        raise ValueError(f"unknown beta profile '{self.beta_mode}'")

    def gamma(self, tau):
        """Scalar latent amplitude gamma(tau), (b, W). Decreasing in tau."""
        if self.reduce_to_erdm:
            return self.sigma_from_tau(tau)
        tau = tau.clamp(0.0, 1.0)
        if self.gamma_mode == "geometric":
            return self.gamma_0 * (self.gamma_1 / self.gamma_0) ** tau
        if self.gamma_mode == "linear":
            return self.gamma_0 + tau * (self.gamma_1 - self.gamma_0)
        if self.gamma_mode == "cosine":
            half = 0.5 * (1.0 + torch.cos(math.pi * tau))       # 1 -> 0
            return self.gamma_1 + (self.gamma_0 - self.gamma_1) * half
        raise ValueError(f"unknown gamma profile '{self.gamma_mode}'")

    def gamma_dot(self, tau):
        """d gamma / d tau, (b, W)."""
        if self.reduce_to_erdm:
            # d/dtau (smax + tau (smin - smax))^rho
            tau_c = tau.clamp(0.0, 1.0)
            smin = self.sigma_min ** (1.0 / self.rho)
            smax = self.sigma_max ** (1.0 / self.rho)
            base = smax + tau_c * (smin - smax)
            return self.rho * base ** (self.rho - 1.0) * (smin - smax)
        tau = tau.clamp(0.0, 1.0)
        if self.gamma_mode == "geometric":
            return self.gamma(tau) * math.log(self.gamma_1 / self.gamma_0)
        if self.gamma_mode == "linear":
            return torch.full_like(tau, self.gamma_1 - self.gamma_0)
        if self.gamma_mode == "cosine":
            return -0.5 * math.pi * (self.gamma_0 - self.gamma_1) * torch.sin(math.pi * tau)
        raise ValueError(f"unknown gamma profile '{self.gamma_mode}'")

    # ── Gamma = gamma(tau) * S, diagonal in channel space ─────────────────
    def _scale(self, ref):
        """Per-channel latent scale S broadcast against (b, W, C, H, W)."""
        if self.delta_scale is None:
            return None
        return self.delta_scale.to(device=ref.device, dtype=ref.dtype)

    def gamma_apply(self, v, tau):
        """Gamma(tau) v -- amplitude times per-channel scale."""
        if self.spectral is not None:
            out = self.spectral.apply(v, tau)
        else:
            out = self.w5(self.gamma(tau)) * v
        s = self._scale(v)
        return out if s is None else out * s

    def gamma_dot_apply(self, v, tau):
        """Gamma'(tau) v."""
        if self.spectral is not None:
            out = self.spectral.apply_dot(v, tau)
        else:
            out = self.w5(self.gamma_dot(tau)) * v
        s = self._scale(v)
        return out if s is None else out * s

    def gamma_delta_apply(self, v, tau_a, tau_b):
        """(Gamma(tau_b) - Gamma(tau_a)) v -- the exact coefficient increment."""
        if self.spectral is not None:
            out = self.spectral.apply_delta(v, tau_a, tau_b)
        else:
            out = self.w5(self.gamma(tau_b) - self.gamma(tau_a)) * v
        s = self._scale(v)
        return out if s is None else out * s

    def gamma_inv_apply(self, v, tau):
        """Gamma(tau)^{-1} v, in fp32 (gamma_1 is small; bf16 would overflow)."""
        if self.spectral is not None:
            out = self.spectral.apply_inv(v.float(), tau)
        else:
            out = v.float() / self.w5(self.gamma(tau)).float().clamp(min=1e-12)
        s = self._scale(v)
        return out if s is None else out / s.float().clamp(min=1e-12)

    # ------------------------------------------------------------------
    # Forward interpolant
    # ------------------------------------------------------------------
    def interpolant(self, anchor, target, tau, z):
        """x(tau) = a + beta(tau) (y - a) + Gamma(tau) z."""
        if self.reduce_to_erdm:
            # anchors are dropped entirely; beta == 1 => x = y + Gamma z
            return target + self.gamma_apply(z, tau)
        return (
            anchor
            + self.w5(self.beta(tau)) * (target - anchor)
            + self.gamma_apply(z, tau)
        )

    def perturb_anchor(self, anchor):
        """Train-time anchor perturbation a <- a + anchor_noise * S z'.

        Mimics the inference-time reality that the anchor is itself a sample
        (exposure-bias control). Still a valid data-dependent coupling: the
        perturbation is part of the joint, and z stays independent of it.
        """
        if self.anchor_noise <= 0.0:
            return anchor
        zp = self.get_noise(anchor)
        s = self._scale(anchor)
        if s is not None:
            zp = zp * s
        return anchor + self.anchor_noise * zp

    # ------------------------------------------------------------------
    # Network heads and the closed-form quantities
    # ------------------------------------------------------------------
    def c_in(self, tau):
        """Input scaling so the raw network sees ~unit-variance inputs, (b, W).

        ``c_in = 1 / sqrt(Gamma(tau)^2 + sigma_data^2)`` — the variance of the
        interpolant state, since ``x = a + beta (y - a) + Gamma z`` has an
        anchor/target part of scale ``sigma_data`` and an independent latent part
        of scale ``Gamma``. Under ``reduce_to_erdm`` this is *exactly* EDM's
        ``c_in``.

        This is not optional, and its absence is not a small effect. Feeding
        ``x`` in raw (flow-matching style, which is where the pattern came from)
        is fine only when the interpolant is a convex combination and the state
        stays O(1) — true for the coupled scalar-Gamma default, where
        ``gamma_0 = 0.5`` gives ``c_in ~ 0.9``. It is badly false in the
        ERDM-reduction limit, where ``Gamma`` runs to ``sigma_max = 500`` and the
        back-of-window slots would arrive at the backbone with magnitude ~500 —
        far outside anything an RMSNorm'd DiT is scaled for. Measured on DeltaAI
        before this existed: identical skill to ERDM at leads 1-3 and ~180x worse
        at lead W, because the slots that need reconstructing are precisely the
        high-Gamma ones.
        """
        g = self.gamma(tau)
        return 1.0 / (g ** 2 + self.sigma_data ** 2).sqrt()

    def z_precond(self, tau):
        """EDM skip/out coefficients for the latent head, each (b, W).

        The latent head is read out through EDM's preconditioned-denoiser form
        rather than as a bare regression:

            D    = c_skip x + c_out F_z ,      zhat = (x - D) / Gamma
            =>   zhat = x * gamma / (gamma^2 + sigma_data^2)
                      + F_z * sigma_data / sqrt(gamma^2 + sigma_data^2)

        with ``c_skip = sigma_data^2/(gamma^2+sigma_data^2)`` and
        ``c_out = gamma sigma_data / sqrt(gamma^2+sigma_data^2)``.

        **Why it is not optional.** The backbones are zero-init by design (the
        intended soft start), and what a zero output MEANS differs completely
        between the two parameterizations. In ERDM ``F = 0`` gives
        ``D = c_skip x``, and ``c_skip ~ 4e-6`` at ``sigma = 500``, so the sweep
        CONTRACTS the state — a safe "predict zero" default. Regressing ``zhat``
        bare instead makes ``zhat = 0`` mean *zero transport*: a slot injected at
        ``sigma_max`` is emitted still carrying ``sigma_max`` noise. Measured
        with a zero-output network, emitted magnitude at lead W: ERDM 7.0e-1,
        bare-zhat RSI 3.7e2 — a 525x divergence that survives into real runs as
        an exploding long-lead RMSE while leads 1-3 look perfectly healthy.

        With the skip in place, ``F_z = 0`` gives
        ``(gamma_next - gamma_cur) * x * gamma / (gamma^2 + sigma_data^2)``,
        which is *identically* ERDM's Euler step at zero-init — so the A1
        reduction now holds at initialization too, not just in the algebra.

        The loss is unaffected: it regresses ``zhat`` (in state units, scaled by
        Gamma), so the network simply learns ``F_z``.
        """
        g = self.gamma(tau)
        denom = g ** 2 + self.sigma_data ** 2
        skip = g / denom                                  # multiplies x
        out = self.sigma_data / denom.sqrt()              # multiplies F_z
        return skip, out

    def label(self, tau):
        """Per-slot conditioning label handed to the backbone, (b, W)."""
        if self.label_mode == "tau":
            return tau.clamp(self.time_eps, 1.0)
        # log_sigma_eff: reduces to ERDM's log(sigma)/4 under reduce_to_erdm.
        return self.sigma_eff(tau).log() / 4.0

    def heads(self, model, x, tau, c_grid, c_scalar):
        """One backbone pass -> (H1, zhat), each (b, W, C, H, W).

        The backbone is handed ``c_in(tau) * x`` so it always sees ~unit-variance
        inputs; the heads' targets are unchanged (H1 predicts the state or the
        increment, zhat the white latent), so this is input preconditioning only.
        """
        out = model(self.w5(self.c_in(tau)) * x, self.label(tau), c_grid, c_scalar)
        c2 = out.shape[2]
        if c2 != 2 * x.shape[2]:
            raise ValueError(
                f"RSI expects the backbone to emit 2*C = {2 * x.shape[2]} channels "
                f"(H1 and zhat), got {c2}. Set "
                f"`rolling_dit_kwargs.output_head.num_output_heads: 2` in the model "
                f"config (see conf/model/amip_rsi_v2.yaml)."
            )
        h1, f_z = out.split(x.shape[2], dim=2)
        # Read the latent head out through EDM's skip/out form so a zero-init
        # network means "contract", not "do not move" (see z_precond).
        skip, out_c = self.z_precond(tau)
        zhat = self.w5(skip) * x + self.w5(out_c) * f_z
        if self.h1_precond == "edm":
            # Read H1 out through the same EDM denoiser form,
            #
            #     y_hat = c_skip(tau) x + c_out(tau) F_1,
            #     c_skip = sigma_d^2/(gamma^2+sigma_d^2),
            #     c_out  = gamma sigma_d/sqrt(gamma^2+sigma_d^2),
            #
            # instead of regressing the raw state. The proposal (sec 3.4)
            # prescribes exactly this skip path for the learned heads; reading
            # H1 raw makes the network responsible for a full-precision copy of
            # the state at every tau — at the front slots that is an identity
            # map the backbone must realize to ~gamma_1 precision through its
            # own layers, a needlessly stiff objective (the raw head's
            # output-space curvature is lambda*f vs the c_out-scaled head's f;
            # the ratio is lambda(sigma_eff), ~600-2500 at the front slots).
            # With the skip, the raw network regresses the unit-variance
            # residual (y - c_skip x)/c_out, exactly as ERDM's F does — and
            # under reduce_to_erdm this IS ERDM's D readout, so the A1
            # reduction holds for the H1 head at zero-init too.
            g = self.gamma(tau)
            denom = g ** 2 + self.sigma_data ** 2
            h1 = (
                self.w5(self.sigma_data ** 2 / denom) * x
                + self.w5(g * self.sigma_data / denom.sqrt()) * h1
            )
        return h1, zhat

    def delta_from(self, x, tau, h1, zhat):
        """Increment estimate Delta_hat = E[y - a | x], (b, W, C, H, W).

        Under ``parameterization='residual'`` the head predicts it directly.
        Under ``'state'`` the head predicts y_hat and the increment is recovered
        from the interpolant relation y = x + (1-beta) Delta - Gamma z, i.e.

            Delta_hat = (y_hat - x + Gamma zhat) / (1 - beta),

        whose denominator vanishes as tau -> 1. It is clamped at ``beta_floor``;
        this is the known weakness of the state parameterization and precisely
        what the residual mode (ablation A3) removes. In practice the sampler
        never evaluates the drift at tau = 1 exactly -- the Heun corrector is
        skipped on the final step, as in ERDM -- so 1 - beta >= 1/(W*N) there.
        """
        if self.reduce_to_erdm:
            # beta == 1: the increment term is identically absent from both the
            # velocity and the transport step. Returning zeros keeps the
            # state-mode reconstruction from dividing by beta_floor.
            return torch.zeros_like(h1)
        if self.parameterization == "residual":
            return h1
        denom = (1.0 - self.beta(tau)).clamp(min=self.beta_floor)
        return (h1 - x + self.gamma_apply(zhat, tau)) / self.w5(denom)

    def denoised(self, x, tau, h1, zhat):
        """Clean-state estimate y_hat = x + (1-beta) Delta_hat - Gamma zhat."""
        if self.parameterization == "state":
            return h1
        return (
            x
            + self.w5(1.0 - self.beta(tau)) * h1
            - self.gamma_apply(zhat, tau)
        )

    def velocity(self, x, tau, h1, zhat):
        """Interpolant velocity b = beta' Delta_hat + Gamma' zhat."""
        delta = self.delta_from(x, tau, h1, zhat)
        return self.w5(self.beta_dot(tau)) * delta + self.gamma_dot_apply(zhat, tau)

    def score(self, tau, zhat):
        """grad_x log p_tau(x) = -Gamma(tau)^{-1} E[z | x]  (fp32)."""
        return -self.gamma_inv_apply(zhat, tau)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def sigma_eff(self, tau):
        """Effective noise level sigma_eff = gamma(tau) / (beta(tau) * delta_std).

        The scalar that plays sigma's role in ERDM's weighting: the latent
        amplitude measured in units of the signal actually present at tau.
        Under ``reduce_to_erdm`` (beta == 1, delta_std == 1) it IS sigma_bar_w,
        so the weight below reduces to ERDM's lambda(sigma) f(sigma) exactly.
        """
        num = (
            self.spectral.band_amplitude(tau) if self.spectral is not None
            else self.gamma(tau)
        )
        den = (self.beta(tau) * self.delta_std).clamp(min=self.beta_floor)
        return num / den

    def loss_weight(self, tau):
        """Per-slot loss weight omega(tau), (b, W).

        ``snr_bump`` is the ERDM analogue: the EDM unit-variance normalizer
        lambda times the log-normal emphasis f, both evaluated at
        :meth:`sigma_eff`. ERDM's insight -- a fixed full-range schedule per
        training instance needs an explicit reweighting concentrating on the
        informative "demixing" region -- transfers directly.
        """
        if self.weighting == "uniform":
            return torch.ones_like(tau)
        if self.weighting == "midrange":
            return tau * (1.0 - tau)
        if self.weighting == "lognormal_logit":
            tc = tau.clamp(self.time_eps, 1.0 - self.time_eps)
            logit = torch.log(tc / (1.0 - tc))
            return torch.exp(-(logit - self.P_mean) ** 2 / (2.0 * self.P_std ** 2)) \
                / (self.P_std * math.sqrt(2.0 * math.pi))
        if self.weighting == "snr_bump":
            sig = self.sigma_eff(tau).clamp(min=1e-12)
            lam = (sig ** 2 + 1.0) / (sig ** 2)
            f = torch.exp(-(sig.log() - self.P_mean) ** 2 / (2.0 * self.P_std ** 2)) \
                / (sig * self.P_std * math.sqrt(2.0 * math.pi))
            return lam * f
        raise ValueError(f"unknown weighting '{self.weighting}'")

    def compute_loss(self, model, c_grid, c_scalar, y, return_parts=False):
        """RSI training loss.

        y : (b, W+1, C, H, W)  -- W+1 clean frames y_0 .. y_W. Slot w = 1..W
            targets y_w and is anchored on y_{w-1}; the leading frame is the
            anchor for slot 1 and is never itself a target. Ocean block
            included on all W+1 frames (see :meth:`append_ocean_target`).
        c_grid   : (b, W, c_grid, H, W) per-slot forcings (slots 1..W)
        c_scalar : (b, W, scalar_dim) or None
        """
        if y.shape[1] != self.W + 1:
            raise ValueError(
                f"RSI compute_loss expects W+1 = {self.W + 1} state frames "
                f"(slot 1's anchor is the frame before the window), got "
                f"{y.shape[1]}. The recipe supplies it via "
                f"SequenceDataset(emit_anchor=True) + _pack_window(anchor_frames=1)."
            )
        b = y.shape[0]
        device = y.device

        anchors = self.perturb_anchor(y[:, :-1])     # (b, W, C, H, W)
        targets = y[:, 1:]                            # (b, W, C, H, W)

        t = torch.rand(b, device=device)              # global time, (b,)
        tau = self.local_time(t)                      # (b, W)

        z = self.get_noise(targets)
        x = self.interpolant(anchors, targets, tau, z)

        h1, zhat = self.heads(model, x, tau, c_grid, c_scalar)

        # Both head terms are measured in STATE units, i.e. each is scaled by its
        # own Jacobian to the reconstructed state
        #
        #     y_hat = x + (1 - beta) Delta_hat - Gamma zhat.
        #
        # This is not cosmetic. Regressing the latent directly and weighting the
        # raw ||zhat - z||^2 is NOT EDM's objective: since D - y =
        # -Gamma (zhat - z), an eps-parameterized term carries a Gamma^2 factor,
        # and omitting it under-weights every high-Gamma slot by up to Gamma^2
        # (sigma_max^2 = 2.5e5 at the shipped schedule). Those are exactly the
        # back-of-window slots a rolling forecast has to reconstruct, so the
        # symptom was a front-of-window forecast that looked fine while the
        # frame emitted at lead W was ~180x worse than ERDM's -- measured on
        # DeltaAI, 30 iterations, before this was fixed. With the Jacobians in
        # place ``reduce_to_erdm`` reproduces ERDM's loss exactly, which is what
        # ablation A1 is for (test_a1_loss_reduces_to_erdm).
        if self.parameterization == "residual":
            h1_target = targets - anchors
            j1 = self.w5(1.0 - self.beta(tau))      # d y_hat / d Delta_hat
        else:
            h1_target = targets
            j1 = 1.0                                 # h1 IS the state estimate
        err_1 = j1 * (h1 - h1_target)
        err_z = self.gamma_apply(zhat - z, tau)      # Gamma (zhat - z)

        err2 = self.w_1 * err_1 ** 2 + self.w_z * err_z ** 2

        weight = self.loss_weight(tau)                # (b, W)

        # Diagnostic decomposition, off unless RSI_LOSS_DIAG=<N> is exported:
        # every N calls, log the weighted per-slot h1-term and z-term sums.
        # Cheap (reuses tensors already in hand) and exactly what is needed to
        # see WHICH term leads when a run destabilizes.
        if _LOSS_DIAG_EVERY > 0:
            global _LOSS_DIAG_CALLS
            _LOSS_DIAG_CALLS += 1
            if _LOSS_DIAG_CALLS % _LOSS_DIAG_EVERY == 0:
                with torch.no_grad():
                    p1 = (weight * (self.w_1 * err_1 ** 2).sum(dim=[2, 3, 4])).mean(0)
                    pz = (weight * (self.w_z * err_z ** 2).sum(dim=[2, 3, 4])).mean(0)
                    logger.info(
                        "rsi loss diag: h1/slot=%s z/slot=%s",
                        ["%.3e" % v for v in p1.tolist()],
                        ["%.3e" % v for v in pz.tolist()],
                    )
        if self.nocean:
            n = err2.shape[2] - self.nocean
            state_mse = err2[:, :, :n].sum(dim=[2, 3, 4])
            ocean_mse = err2[:, :, n:].sum(dim=[2, 3, 4])
            per_frame_mse = state_mse + self.ocean_loss_weight * ocean_mse
        else:
            per_frame_mse = err2.sum(dim=[2, 3, 4])
            ocean_mse = None

        loss = (weight * per_frame_mse).mean()
        if not return_parts:
            return loss
        ocean_loss = (
            (weight * ocean_mse).mean() if ocean_mse is not None
            else loss.new_zeros(())
        )
        return loss, ocean_loss

    # ------------------------------------------------------------------
    # Ocean channels (Phase 12f contract)
    # ------------------------------------------------------------------
    def ocean_truth(self, bnd_win, size, expect=None):
        """True ocean fields for a window: ``(b, n, nocean, *size)`` or None.

        ``expect`` is the frame count the caller intends (W for the in-window
        views, W+1 for RSI's training target stack, which carries the anchor
        frame). ERDM hard-codes W here; RSI needs both, so the check is
        parameterized -- but it is still a check, because the shapes match
        either way and a wrong window is a silent identity-copy bug.
        """
        if not self.nocean or bnd_win is None:
            return None
        expect = self.W if expect is None else int(expect)
        if bnd_win.shape[1] != expect:
            raise ValueError(
                f"ocean_truth expected a {expect}-frame window, got "
                f"{bnd_win.shape[1]}. Callers must pass the correctly shifted "
                f"slice -- see the time-alignment note in erdm.py."
            )
        idx = torch.as_tensor(self.ocean_grid_indices, device=bnd_win.device)
        o = bnd_win.index_select(2, idx)
        b, n = o.shape[0], o.shape[1]
        if tuple(o.shape[-2:]) != tuple(size):
            o = torch.nn.functional.interpolate(
                o.flatten(0, 1), size=tuple(size), mode="bilinear",
                align_corners=False,
            ).unflatten(0, (b, n))
        return o

    def append_ocean_target(self, y, bnd_ext):
        """Append supervised ocean channels to a ``W+1``-frame clean stack.

        ``y``: ``(b, W+1, C, h, w)`` -> ``(b, W+1, C + nocean, h, w)``.
        ``bnd_ext`` is the boundary at each frame's OWN time for all W+1
        frames -- i.e. ``cat([varying_boundary_seq[:, :1],
        varying_boundary_next_seq], dim=1)``: the anchor frame's own-time
        forcing is the slot-1 conditioning frame, which the loader already
        emits.
        """
        if not self.nocean:
            return y
        truth = self.ocean_truth(bnd_ext, y.shape[-2:], expect=y.shape[1])
        if truth is None:
            raise ValueError(
                "append_ocean_target needs the boundary stack to read the ocean "
                "target from, but got None."
            )
        return torch.cat([y, truth.to(y.dtype)], dim=2)

    def pad_state(self, x):
        """Widen a bare state window to the model's channel count with zeros."""
        if not self.nocean:
            return x
        b, n = x.shape[0], x.shape[1]
        return torch.cat(
            [x, x.new_zeros(b, n, self.nocean, *x.shape[-2:])], dim=2
        )

    def strip_ocean(self, x):
        """Drop the ocean block, returning a plain state tensor."""
        if not self.nocean:
            return x
        return x[..., : x.shape[-3] - self.nocean, :, :]

    def impose_ocean(self, x, bnd_next, bnd_curr):
        """Overwrite the window's ocean channels with the true interpolant.

        Call at the TOP of a roll (global time t = 0) and nowhere else -- the
        shift identity is what makes one tau per slot correct there, exactly as
        argued in ERDM's ``impose_ocean``. The difference is the imposed value:
        ERDM writes ``truth + sigma * eps`` because its base is noise, RSI
        writes the interpolant between the two truths the slot actually lies
        between,

            a + beta(tau_w(0)) (y - a) + Gamma(tau_w(0)) z,

        with ``y`` the own-time truth (``bnd_next``, the one-step-shifted
        window) and ``a`` the anchor-time truth (``bnd_curr``, the slot's own
        conditioning forcing window). Writing ERDM's form here would put the
        ocean block on a different path from every other channel.
        """
        if not self.nocean or bnd_next is None:
            return x
        truth_y = self.ocean_truth(bnd_next, x.shape[-2:], expect=self.W)
        truth_a = (
            self.ocean_truth(bnd_curr, x.shape[-2:], expect=self.W)
            if bnd_curr is not None else truth_y
        )
        tau0 = self.local_time(torch.zeros(x.shape[0], device=x.device))
        z = self.get_noise(x)[:, :, -self.nocean:]
        block = self.interpolant(
            truth_a.to(x.dtype), truth_y.to(x.dtype), tau0, z
        ) if not self.reduce_to_erdm else (
            truth_y.to(x.dtype) + self.gamma_apply(z, tau0)
        )
        x = x.clone()
        x[:, :, -self.nocean:] = block
        return x

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _eps(self, tau):
        """Diffusion coefficient eps(tau) of the sampling SDE, (b, W)."""
        if self.eps_scale <= 0.0:
            return None
        active = (tau >= self.eps_tmin) & (tau <= self.eps_tmax)
        return torch.where(
            active,
            torch.full_like(tau, self.eps_scale),
            torch.zeros_like(tau),
        )

    def _step(self, x_eval, tau_eval, tau_cur, tau_next, h1, zhat):
        """Transport increment over [tau_cur, tau_next] from one head evaluation.

        ``x_eval``/``tau_eval`` are where the heads were evaluated -- the step
        start for the Euler predictor, the step END for Heun's corrector, which
        is what the state-mode increment reconstruction has to be told (ERDM
        does the same thing when it forms ``d_next`` at ``sigma_next``).
        """
        delta = self.delta_from(x_eval, tau_eval, h1, zhat)
        if self.integrator == "coeff":
            # Exact coefficient increments: exact whenever the conditional
            # expectations are constant over the step, and bit-equal to ERDM's
            # Euler step under reduce_to_erdm.
            dbeta = self.w5(self.beta(tau_next) - self.beta(tau_cur))
            return dbeta * delta + self.gamma_delta_apply(zhat, tau_cur, tau_next)
        dtau = self.w5(tau_next - tau_cur)
        return dtau * (
            self.w5(self.beta_dot(tau_eval)) * delta
            + self.gamma_dot_apply(zhat, tau_eval)
        )

    def sample_window(self, model, x, c_grid_win, c_scalar_win, num_steps=None,
                      ocean_win=None):
        """One inner sweep: transport the window from t = 0 to t = 1.

        Returns ``(x, y_hat)``. ERDM's sweep returns only the noisy window
        because its emitted frame IS ``x[:, 0]`` at sigma_min; RSI's emitted
        frame is the denoised readout of the final evaluation (the noise floor
        Gamma_1 subtracted), and the freshly appended back slot is anchored on
        ``y_hat[:, -1]``. Both come from the same last head evaluation, so it is
        returned rather than recomputed.

        ``ocean_win`` is the forcing window shifted forward one step;
        ``c_grid_win`` doubles as the anchor-time boundary (see
        :meth:`impose_ocean`).
        """
        if num_steps is None:
            num_steps = self.num_steps

        b = x.shape[0]
        device = x.device
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        # Prescribe the ocean channels BEFORE integrating: global t=0 is the one
        # point where a single tau per slot is correct (see impose_ocean).
        if self.nocean:
            x = self.impose_ocean(x, ocean_win, c_grid_win)

        h1 = zhat = tau_last = tau_eval = x_eval = None
        for i in range(num_steps):
            tau_cur = self.local_time(timesteps[i].expand(b))
            tau_next = self.local_time(timesteps[i + 1].expand(b))

            h1, zhat = self.heads(model, x, tau_cur, c_grid_win, c_scalar_win)
            d_cur = self._step(x, tau_cur, tau_cur, tau_next, h1, zhat)
            x_euler = x + d_cur
            x_eval, tau_eval = x, tau_cur
            # Keep the STEP-START latent estimate: the Heun corrector below
            # rebinds ``zhat`` to its own evaluation at tau_next, and the
            # Euler-Maruyama noise term wants the score at the step start.
            # Pairing tau_cur's Gamma with tau_next's zhat is not a small error
            # -- Gamma spans orders of magnitude across a step.
            zhat_start = zhat

            if self.solver == "heun" and i < num_steps - 1:
                h1_n, zhat_n = self.heads(
                    model, x_euler, tau_next, c_grid_win, c_scalar_win)
                d_next = self._step(
                    x_euler, tau_next, tau_cur, tau_next, h1_n, zhat_n)
                x = x + 0.5 * (d_cur + d_next)
                h1, zhat = h1_n, zhat_n
                x_eval, tau_eval = x_euler, tau_next
            else:
                x = x_euler

            # eps-family stochastic forcing (marginal-preserving in the exact-
            # score limit). eps_scale = 0 leaves the PF-ODE path bit-identical.
            eps = self._eps(tau_cur)
            if eps is not None:
                dtau = (tau_next - tau_cur).clamp(min=0.0)
                s = self.score(tau_cur, zhat_start).to(x.dtype)
                x = x + self.w5(eps * dtau) * s \
                    + self.w5((2.0 * eps * dtau).sqrt()) * self.get_noise(x)

            tau_last = tau_next

        # Denoised readout. It MUST be paired with the (x, tau) the heads were
        # actually evaluated at: y_hat = x + (1-beta) Delta_hat - Gamma(tau) zhat
        # only holds at that point, and Gamma varies by orders of magnitude
        # across a step, so pairing a lagged zhat with the endpoint Gamma is not
        # a small error -- it rescales the correction outright. Using the last
        # evaluation pair is free and self-consistent (it is exactly
        # E[y | x_eval], a legitimate conditional-mean estimate). Under the state
        # parameterization the readout is the head itself and the distinction is
        # moot. ``final_denoise`` buys the endpoint readout for one extra
        # evaluation per roll.
        if self.final_denoise:
            h1, zhat = self.heads(model, x, tau_last, c_grid_win, c_scalar_win)
            x_eval, tau_eval = x, tau_last
        y_hat = self.denoised(x_eval, tau_eval, h1, zhat)
        return x, y_hat

    def warmup_window(self, init_window):
        """Place W+1 oracle frames on the interpolant at the t = 0 staircase.

        init_window : (b, W+1, C, H, W) -- y_0 .. y_W. Slot w = 1..W is built
        from anchor y_{w-1} and target y_w at tau_w(0) = (W-w)/W, so the front
        slot enters nearly clean and the back slot at pure anchor + full noise.
        """
        if init_window.shape[1] != self.init_frames:
            raise ValueError(
                f"RSI oracle init expects {self.init_frames} frames "
                f"(W+1: slot 1 needs its anchor), got {init_window.shape[1]}. "
                f"Drivers read the count off `scheduler.init_frames`."
            )
        b = init_window.shape[0]
        tau0 = self.local_time(torch.zeros(b, device=init_window.device))
        anchors = init_window[:, :-1]
        targets = init_window[:, 1:]
        z = self.get_noise(targets)
        return self.interpolant(anchors, targets, tau0, z)

    def _fresh_slot(self, y_hat):
        """Back slot entering at tau = 0: anchor + Gamma(0) z.

        The anchor is the newest available state estimate -- the resolved slot W
        of the window just integrated. No slot ever starts from structureless
        noise; this is the single most consequential change relative to ERDM.
        """
        anchor = y_hat[:, -1:]
        tau0 = torch.zeros(anchor.shape[0], 1, device=anchor.device,
                           dtype=torch.float32)
        z = self.get_noise(anchor)
        if self.reduce_to_erdm:
            return self.gamma_apply(z, tau0)
        return anchor + self.gamma_apply(z, tau0)

    @torch.no_grad()
    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        """Rolling-window autoregressive sampler.

        init_window  : (b, W+1, C, H, W) oracle frames y_0..y_W. Under
            ``nocean > 0`` pass a BARE state stack -- it is padded here and the
            zeros are overwritten by the first imposition.
        c_grid_traj  : (b, T, c_grid, H, W) forcings over absolute future frames
        c_scalar_traj: (b, T, scalar_dim) or None
        horizon      : number of frames to forecast (emit)
        num_steps    : solver steps per emitted frame; int/None, or a sequence
            of length ``horizon``.

        Returns (b, horizon, C, H, W) -- ocean block included; call
        :meth:`strip_ocean` before handing frames to a state-sized consumer.
        """
        if isinstance(num_steps, (list, tuple)) and len(num_steps) != horizon:
            raise ValueError(
                f"num_steps schedule has length {len(num_steps)}, expected "
                f"horizon={horizon}"
            )

        x = self.warmup_window(self.pad_state(init_window))

        outputs = []
        for k in range(horizon):
            c_grid_win = self._gather_window(c_grid_traj, k)
            c_scalar_win = self._gather_window(c_scalar_traj, k)
            ocean_win = (
                self._gather_window(c_grid_traj, k + 1) if self.nocean else None
            )

            step_k = num_steps[k] if isinstance(num_steps, (list, tuple)) else num_steps
            x, y_hat = self.sample_window(
                model, x, c_grid_win, c_scalar_win, step_k, ocean_win=ocean_win)

            outputs.append(y_hat[:, 0])
            x = torch.cat([x[:, 1:], self._fresh_slot(y_hat)], dim=1)

        return torch.stack(outputs, dim=1)

    def forward(self, model, init_window, c_grid_traj, c_scalar_traj, horizon,
                num_steps=None):
        return self.sample_rollout(model, init_window, c_grid_traj, c_scalar_traj,
                                   horizon, num_steps)

    @torch.no_grad()
    def sample_rollout_generator(self, model, init_window, c_grid_traj,
                                 c_scalar_traj, horizon, num_steps=None,
                                 forcing_provider=None):
        """Streaming variant of :meth:`sample_rollout`, yielding ``(k, frame)``.

        This is what the 40-year climate rollout runs on: with a
        ``forcing_provider`` the GPU only ever holds the W-frame forcing window,
        never the full horizon. A provider may return a third element, the
        one-step-shifted window used to impose the true ocean fields;
        two-element returns stay valid.
        """
        device = init_window.device
        x = self.warmup_window(self.pad_state(init_window))

        for k in range(horizon):
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

            step_k = num_steps[k] if isinstance(num_steps, (list, tuple)) else num_steps
            x, y_hat = self.sample_window(
                model, x, c_grid_win, c_scalar_win, step_k, ocean_win=ocean_win)

            yield k, y_hat[:, 0]

            x = torch.cat([x[:, 1:], self._fresh_slot(y_hat)], dim=1)

    # ------------------------------------------------------------------
    # Streaming hooks (CombinedModule / rollout.py delegate to these)
    # ------------------------------------------------------------------
    def stream_init(self, init_window):
        """Open a streaming rollout. Returns the scheduler's rolling state.

        A 2-tuple ``(x, None)``: ERDM's streaming state is
        ``(x_bar, eps_prev)`` and the drivers (``rollout.py``, including its
        on-disk resume format) unpack two elements, so RSI keeps the arity and
        leaves the second slot empty -- it carries no noise history, the
        anchor is the previous roll's own resolved estimate.
        """
        return (self.warmup_window(self.pad_state(init_window)), None)

    def stream_step(self, model, state, c_grid_win, c_scalar_win, num_steps=None,
                    ocean_win=None):
        """Advance one emitted frame. Returns ``(emitted, new_state)``."""
        x = state[0]
        x, y_hat = self.sample_window(
            model, x, c_grid_win, c_scalar_win, num_steps, ocean_win=ocean_win)
        emitted = y_hat[:, 0]
        x = torch.cat([x[:, 1:], self._fresh_slot(y_hat)], dim=1)
        return emitted, (x, None)
