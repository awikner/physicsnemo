# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/x_DDC.py) for Phase 8f (F6).

import logging

import torch
import torch.nn as nn

from ._utils import get_log_uniform_t, power_sampler, sample_logit_normal

logger = logging.getLogger(__name__)


class DataDependentInterpolant(nn.Module):
    """
    Stochastic interpolant with data-dependent couplings (Albergo et al. 2310.03725).

    The interpolant is defined as:
        I_t = alpha_t * x0 + beta_t * x1

    where x0 = m(x1) + sigma * zeta is the data-dependent coupling
    (m is a corruption map, e.g. downsample-then-upsample, and zeta ~ N(0,I)).

    No additional gamma_t * z noise in the interpolant. All stochasticity
    comes from the data-dependent coupling x0 = m(x1) + sigma * zeta.
    """

    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='power',
                 l_max = 180,
                 noise = "spherical",
                 integrator = "exponential",
                 tau = 1.3,
                 model_last = False,
                 noise_scale_path = None):

        super().__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler
        self.tau = tau
        self.model_last = model_last
        self.noise_scale_path = noise_scale_path
        self.integrator = integrator

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
            "DataDependentInterpolant initialized: num_steps=%s, noise=%s, "
            "sigma_coef=%s, train_sampler=%s, integrator=%s",
            num_steps, noise, self.sigma_coef, self.train_sampler, self.integrator,
        )

    def get_noise(self, x):
        """Generate noise for the data-dependent coupling."""
        if self.generator is not None:
            noise = self.generator(x.shape[0], x.shape[1], device=x.device)
        else:
            noise = torch.randn_like(x)

        if self.noise_scales is not None:
            noise = noise * self.noise_scales

        return noise

    def compute_loss(self, model, x_lowres, x_highres):
        """
        Args:
            x_lowres: [b, c, h, w] — m(x1), the upsampled low-res (source base). Also passed in as conditioning
            x_highres: [b, c, h, w] — x1, ground truth (target distribution)
            model: predictor, called as model(I_t, t, cond=x_lowres)

        Returns:
            scalar loss
        """
        device = x_lowres.device

        # Data-dependent coupling: x0 = m(x1) + sigma * zeta
        zeta = self.get_noise(x_lowres)

        x0 = x_lowres + self.sigma_coef * zeta

        x1 = x_highres

        # sample timestep
        if self.train_sampler == 'logit_normal':
            t = sample_logit_normal(x0.shape[0], device=device)
        elif self.train_sampler == 'power':
            t = power_sampler(x0.shape[0], p=1.5, device=device)
        elif self.train_sampler == 'uniform':
            t = torch.rand(x0.shape[0], device=device)

        # Reshape for broadcasting: [b, 1, 1, 1]
        t_wide = t[:, None, None, None]

        # Interpolant: I_t = (1-t) * x0 + t * x1
        I_t = (1 - t_wide) * x0 + t_wide * x1

        x1_pred = model(I_t, x_lowres, t=t[:, None])

        loss = ((x1_pred - x1) ** 2).sum(dim=[1, 2, 3]).mean()

        return loss

    @torch.no_grad()
    def sample_exponential(self, model, x_lowres, num_steps=None):
        """
        Forward Euler ODE integration. Reparameterized for stability and x-prediction

        Draw zeta ~ N(0, I)
        X_0 = m(x1) + sigma * zeta
        define ratio r
        define dt_k = (1-r) / (1-t_k)
        this simplifies the Euler update to:
            x_{t+1} = r*x_t + (1-r)x_1

        originally:
            v_hat = (\\hat x_1 - x_t) / (1 - t)
            x_{t+1} = x_t + dt * v_hat => (1 - dt/(1-t)) x_t + dt/(1-t) * \\hat x_1

        Args:
            x_lowres: [b, c, h, w] — m(x1), upsampled low-res conditioning
            model: predictor
            num_steps: number of integration steps N

        Returns:
            [b, c, h, w] predicted high-res output
        """

        if num_steps is None:
            num_steps = self.num_steps

        # Starting point: X_0 = m(x1) + sigma * zeta
        zeta = self.get_noise(x_lowres)

        y = x_lowres + self.sigma_coef * zeta

        timesteps, ratio = get_log_uniform_t(n_t = num_steps - 1, scale = self.tau, device = x_lowres.device)

        ratio_batch = ratio.expand(x_lowres.shape[0], 1, 1, 1)

        if self.model_last:
            num_steps_euler = num_steps - 1
        else:
            num_steps_euler = num_steps

        for k in range(num_steps_euler):
            t_k = timesteps[k]
            t_batch = torch.full((x_lowres.shape[0], 1), t_k, device=x_lowres.device, dtype=x_lowres.dtype)

            x1_pred = model(y, x_lowres, t_batch)

            y = ratio_batch * y + (1-ratio_batch) * x1_pred

        # take last step w/o implied velocity and Euler step
        if self.model_last:
            t_batch = torch.full((x_lowres.shape[0], 1), timesteps[-1], device=x_lowres.device, dtype=x_lowres.dtype)
            y = model(y, x_lowres, t_batch)

        return y

    @torch.no_grad()
    def sample_uniform(self, model, x_lowres, num_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)

        if num_steps is None:
            num_steps = self.num_steps

        # Starting point: X_0 = m(x1) + sigma * zeta
        zeta = self.get_noise(x_lowres)

        xt = x_lowres + self.sigma_coef * zeta

        timesteps = torch.linspace(0, 1, num_steps + 1, device=xt.device)

        for i in range(num_steps):
            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(xt.shape[0])

            x1_pred = model(xt, x_lowres, t_curr_batch) # Predict x_1
            drift_curr = (x1_pred - xt) / (1 - t_curr_batch)

            xt = xt + dt * drift_curr

        return xt

    def sample(self, model, x_lowres, num_steps=None):
        if self.integrator == "exponential":
            return self.sample_exponential(model, x_lowres, num_steps=num_steps)
        elif self.integrator == "uniform":
            return self.sample_uniform(model, x_lowres, num_steps=num_steps)
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

    def forward(self, model, x, num_steps=None):
        return self.sample(model, x, num_steps=num_steps)
