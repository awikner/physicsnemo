# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/x_interpolant.py) for Phase 8a.

import logging

import torch
import torch.nn as nn

from ._utils import get_log_uniform_t, power_sampler, sample_logit_normal

logger = logging.getLogger(__name__)


class DynamicInterpolant(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='uniform',
                 integrator='euler',
                 l_max = 180,
                 noise = "spherical",
                 noise_scale_path = None,
                 tau = 1.3,
                 t_final=0.999,
                 ):
        super(DynamicInterpolant, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler
        self.integrator = integrator
        self.t_final = t_final
        self.tau = tau

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
            "DynamicInterpolant initialized: num_steps=%s, noise=%s, integrator=%s, "
            "sigma_coef=%s, train_sampler=%s, tau=%s, t_final=%s, noise_scale_path=%s",
            num_steps, noise, integrator, self.sigma_coef, self.train_sampler,
            tau, t_final, noise_scale_path,
        )

    def wide(self, t):
        return t[:, None, None, None]
    
    def get_noise(self, x):
        if self.generator is None:
            return torch.randn_like(x, device=x.device)
        else:
            return self.generator(x.shape[0], x.shape[1], device=x.device)

    def compute_loss(self, model, x, c_grid, c_scalar, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # y contains next prognostic state

        device = x.device

        noise = self.get_noise(x)

        if self.noise_scales is not None:
            noise = noise * self.noise_scales

        # sample timestep
        if self.train_sampler == 'logit_normal':
            t = sample_logit_normal(x.shape[0], device=device)
        elif self.train_sampler == 'power':
            t = power_sampler(x.shape[0], p=2.0, device=device)
        elif self.train_sampler == 'uniform':
            t = torch.rand(x.shape[0], device=device)

        t = self.wide(t) 
        W_t = torch.sqrt(t) * noise
        X_t = (1-t) * x + t * y + (1-t) * self.sigma_coef * W_t

        pred_y = model(X_t, x, t.squeeze(dim=[1, 2, 3]), c_grid, c_scalar)

        loss = ((pred_y - y) ** 2).sum(dim=[1, 2, 3]).mean() 

        return loss

    def sample_uniform(self, model, x, c_grid, c_scalar, num_steps=None, return_model_last=False):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)
        # start y at source distribution, which is current state
        y = x.clone()
        W_t = torch.zeros_like(x)

        num_steps_drift = num_steps

        for i in range(num_steps_drift):
            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            y_pred = model(y, x, t_curr_batch, c_grid, c_scalar)  # Predict x_1
            drift_curr = (y_pred - x) - self.sigma_coef * W_t  # v_theta(t)

            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales
            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))

            # Euler predictor
            y_next_euler = y + drift_curr * dt + diffusion_scale * dW
            W_next       = W_t + dW

            y = y_next_euler
            W_t = W_next

        if return_model_last:
            return y, y_pred

        return y
    
    def sample_exponential(self, model, x, c_grid, c_scalar, num_steps=None, return_model_last=False):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps, ratio = get_log_uniform_t(n_t = num_steps, scale = self.tau, t_final=self.t_final, device = x.device)
        
        ratio_batch = ratio.expand(x.shape[0], 1, 1, 1)
        # start y at source distribution, which is current state
        xt = x.clone()

        num_steps_drift = num_steps

        for i in range(num_steps_drift):

            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            x1_pred = model(xt, x, t_curr_batch, c_grid, c_scalar)  # Predict x_1

            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales
            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))

            # Euler predictor
            x_next = ratio_batch * xt + (1 - ratio_batch) * x1_pred + diffusion_scale * dW

            xt = x_next

        if return_model_last:
            return xt, x1_pred
        
        return xt
    
    def sample(self, model, x, c_grid, c_scalar, num_steps=None, return_model_last=True):
        if self.integrator == "euler":
            return self.sample_uniform(model, x, c_grid, c_scalar,
                                        num_steps=num_steps,
                                        return_model_last=return_model_last)
        elif self.integrator == "exponential":
            return self.sample_exponential(model, x, c_grid, c_scalar,
                                        num_steps=num_steps,
                                        return_model_last=return_model_last)

    def forward(self, model, x, c_grid, c_scalar, num_steps=None):
        return self.sample(model, x, c_grid, c_scalar, num_steps)
    
    # only used during debugging
    def sample_debug_euler(self, model, x, c_grid, c_scalar, num_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)
        # start y at source distribution, which is current state
        xt = x.clone()

        num_steps_drift = num_steps

        metadata = []

        for i in range(num_steps_drift):
            save_dict = {}

            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            x1_pred = model(xt, x, t_curr_batch, c_grid, c_scalar)  # Predict x_1
            drift = (x1_pred - xt) / (1 - t_curr_batch)

            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales
            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))

            # Euler predictor
            x_next = xt + drift * dt + diffusion_scale * dW
            #W_next       = W_t + dW

            save_dict['t_current'] = t_current.cpu()
            save_dict['x1_pred'] = x1_pred[0, :6].cpu()
            save_dict['drift'] = drift[0, :6].cpu()
            save_dict['noise'] = noise[0, :6].cpu() 
            save_dict['dW'] = dW[0, :6].cpu()
            save_dict['diffusion_scale'] = diffusion_scale[0, :6].cpu()
            save_dict['xt'] = xt[0, :6].cpu()
            save_dict['x_next'] = x_next[0, :6].cpu()

            metadata.append(save_dict)

            xt = x_next

        # testing model_out
        #xt = x1_pred
        return xt, x1_pred, metadata
    
    # only used during debugging
    def sample_debug_onestep(self, model, x, c_grid, c_scalar, num_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)

        metadata = []

        t_current = timesteps[0]
        t_curr_batch = t_current.expand(x.shape[0])

        x1_pred = model(x, x, t_curr_batch, c_grid, c_scalar)  # Predict x_1

        return x1_pred, x1_pred, metadata
    
    def sample_debug_exponential(self, model, x, c_grid, c_scalar, num_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps, ratio = get_log_uniform_t(n_t = num_steps, scale = self.tau, t_final=self.t_final, device = x.device)
        
        ratio_batch = ratio.expand(x.shape[0], 1, 1, 1)
        # start y at source distribution, which is current state
        xt = x.clone()

        num_steps_drift = num_steps

        metadata = []

        for i in range(num_steps_drift):
            save_dict = {}

            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            x1_pred = model(xt, x, t_curr_batch, c_grid, c_scalar)  # Predict x_1
            drift = (x1_pred - xt) / (1 - t_curr_batch)

            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales
            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))

            # Euler predictor
            x_next = ratio_batch * xt + (1 - ratio_batch) * x1_pred + diffusion_scale * dW
            #W_next       = W_t + dW

            save_dict['t_current'] = t_current.cpu()
            save_dict['x1_pred'] = x1_pred[0, :6].cpu()
            save_dict['drift'] = drift[0, :6].cpu()
            save_dict['noise'] = noise[0, :6].cpu() 
            save_dict['dW'] = dW[0, :6].cpu()
            save_dict['diffusion_scale'] = diffusion_scale[0, :6].cpu()
            save_dict['xt'] = xt[0, :6].cpu()
            save_dict['x_next'] = x_next[0, :6].cpu()

            metadata.append(save_dict)

            xt = x_next
        
        # testing model_out
        #xt = x1_pred
        return xt, x1_pred, metadata
