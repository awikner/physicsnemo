# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/dynamic_interpolant.py) for Phase 8a.

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DriftScheduler(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 beta_fn="t",
                 noise="spherical",
                 l_max = 45,
                 noise_scale_path = None,
                 integrator="euler"
                 ):
        super(DriftScheduler, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef

        self.beta_fn = beta_fn

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
            "DriftScheduler initialized: num_steps=%s, sigma_coef=%s, noise=%s",
            self.num_steps, self.sigma_coef, noise,
        )

    def wide(self, t):
        return t[:, None, None, None]
    
    def alpha(self, t):
        return self.wide(1 - t)

    def alpha_dot(self, t):
        return self.wide(-1.0 * torch.ones_like(t))

    def beta(self, t):
        if self.beta_fn == "t":
            return self.wide(t)
        elif self.beta_fn == "t^2":
            return self.wide(t ** 2)

    def beta_dot(self, t):
        if self.beta_fn == "t":
            return self.wide(torch.ones_like(t))
        elif self.beta_fn == "t^2":
            return self.wide(2.0 * t)

    def sigma(self, t):
        return self.sigma_coef * self.wide(1 - t)

    def sigma_dot(self, t):
        return self.sigma_coef * self.wide(-1.0 * torch.ones_like(t))

    def I(self, x0, x1, t):
        return self.alpha(t) * x0 + self.beta(t) * x1

    def dIdt(self, x0, x1, t):
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    def get_noise(self, x):
        if self.generator is not None:
            noise = self.generator(x.shape[0], x.shape[1], device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        if self.noise_scales is not None:
            noise = noise * self.noise_scales
        return noise
    
    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def compute_loss(self, model, x, c_grid, c_scalar, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # y contains next prognostic state 

        device = x.device

        noise = self.get_noise(x)

        t = torch.rand(x.shape[0], device=device)

        sigma_t = self.sigma(t)          # shape (b, 1, 1, 1)
        sigma_dot_t = self.sigma_dot(t)  # shape (b, 1, 1, 1)
        W_t = self.wide(torch.sqrt(t))   # shape (b, 1, 1, 1)

        I = self.I(x, y, t)  # shape (b, d, nx, ny)
        dIdt = self.dIdt(x, y, t)  # shape (b, d, nx, ny)

        I_noised = I + sigma_t * W_t * noise
        target = dIdt + sigma_dot_t * W_t * noise

        pred = model(I_noised, x, t.view(-1, 1), c_grid, c_scalar)

        loss= self.image_sq_norm(pred - target).mean()

        return loss

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)

        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)
        # start y at source distribution, which is current state
        y = x.clone()

        for i in range(num_steps):
            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            v = model(y, x, t_curr_batch, c_grid, c_scalar) 

            noise = self.get_noise(x)

            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))
            diffusion_term = diffusion_scale * dW

            y = y + v * dt + diffusion_term

        return y
    
    def sample_debug_euler_v(self, model, x, c_grid, c_scalar, num_steps=None):
        # Same Euler-Maruyama integration as sample(), but records per-step
        # diagnostics for debugging.
        # Returns (y, x1_pred, debug):
        #   y        : final integrated state
        #   x1_pred  : None (the Euler integrator has no endpoint prediction)
        #   debug    : dict of per-step tensors (stacked along dim 0 over steps)

        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)
        # start y at source distribution, which is current state
        y = x.clone()

        # Per-step diagnostics. Spatial tensors keep only the first 6 channels
        # of the first batch element to keep saved files small.
        t_log = []
        drift_log = []
        xt_log = []
        dW_log = []
        diffusion_scale_log = []
        diffusion_term_log = []

        for i in range(num_steps):
            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            v = model(y, x, t_curr_batch, c_grid, c_scalar)

            noise = self.get_noise(x)

            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))
            diffusion_term = diffusion_scale * dW

            # Record state/drift/noise *before* taking the step.
            t_log.append(t_current.detach().cpu())
            drift_log.append(v[0, :6].detach().cpu())
            xt_log.append(y[0, :6].detach().cpu())
            dW_log.append(dW[0, :6].detach().cpu())
            diffusion_scale_log.append(diffusion_scale[0].detach().cpu())
            diffusion_term_log.append(diffusion_term[0, :6].detach().cpu())

            y = y + v * dt + diffusion_term

        debug = {
            "t": torch.stack(t_log),                              # (num_steps,)
            "drift": torch.stack(drift_log),                      # (num_steps, 6, nx, ny)
            "xt": torch.stack(xt_log),                            # (num_steps, 6, nx, ny)
            "dW": torch.stack(dW_log),                            # (num_steps, 6, nx, ny)
            "diffusion_scale": torch.stack(diffusion_scale_log),  # (num_steps, 1, 1, 1)
            "diffusion_term": torch.stack(diffusion_term_log),    # (num_steps, 6, nx, ny)
        }

        x1_pred = None
        return y, x1_pred, debug

    def forward(self, model, x, c_grid, c_scalar, num_steps=None):
        return self.sample(model, x, c_grid, c_scalar, num_steps)
