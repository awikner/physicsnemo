# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/edm.py) for Phase 8a. Implementation is largely
# based on the code at https://github.com/yuanzhi-zhu/mini_edm.
#
# Note: the upstream class is a plain ``object``, not an ``nn.Module``,
# and has no learnable parameters or buffers. The ``EDMScheduler`` shim
# in this file's :mod:`__init__` wraps it in an ``nn.Module`` adapter
# so ``to(device)`` / ``state_dict()`` behave consistently with the
# other rolling schedulers in this package.

import torch


class EDMScheduler():
    def __init__(self, 
                 num_steps=10,
                 sigma_min=0.002,
                 sigma_max=80,
                 rho=7,
                 sigma_data=0.5,
                 P_mean=-1.2,
                 P_std=1.2,
                 ndim=2,
                 sde=False):
        
        self.skip_percent = 0
        self.noise_steps = num_steps
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.ndim = ndim
        self.sde = sde

    def batch_mult(self, x, y):
        if self.ndim == 2:
            return torch.einsum('b,bijk->bijk', x, y)
        elif self.ndim == 3:
            return torch.einsum('b,bijkl->bijkl', x, y)

    def compute_loss(self, x, y, model, **kwargs):
        # y shape [b nx ny d], is the future state
        # x, shape [b nx ny d], is the current state
        # cond is optional, shape [b cond_dim] is the conditioning vector

        rnd_normal = torch.randn([y.shape[0]], device=y.device) # shape [B]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp() # log-normal distribution
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 # loss weighting
        
        noise = torch.randn_like(y) 

        n = self.batch_mult(sigma, noise)

        D_yn = self.model_forward_wrapper(y + n, sigma, model, initial_cond=x, **kwargs)
        
        loss = self.batch_mult(weight, ((D_yn - y) ** 2))
        
        return loss.mean()

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma) 
    
    def model_forward_wrapper(self, x, sigma, model, initial_cond, **kwargs):
        """Wrapper for the model call"""
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        preconditioned_x = self.batch_mult(c_in, x)

        model_in = torch.cat([initial_cond, preconditioned_x], dim=-1) # b nx ny 2d

        model_output = model(model_in, sigma_t=c_noise, **kwargs)

        return self.batch_mult(c_skip, x) + self.batch_mult(c_out, model_output)
        

    def edm(self, x, sigma, model, initial_cond, **kwargs):
        if sigma.shape == torch.Size([]): # add batch dim
            sigma = sigma * torch.ones([x.shape[0]], device=x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), model, initial_cond=initial_cond, **kwargs)

    def sample(self, initial_cond, model, edm_solver="euler", **kwargs):
        """
        Main sample loop for EDMs
        initial_cond: the conditioning input, shape [b nx ny d]
        model: the neural network model
        """

        device = initial_cond.device
        edm_stoch = self.sde
        
        with torch.no_grad():
            # EDM sampling params
            num_steps=self.num_steps
            sigma_min=self.sigma_min
            sigma_max=self.sigma_max
            rho=self.rho

            # Time step discretization.
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
            
            x_next = torch.randn_like(initial_cond, device=device)

            x_next = x_next * t_steps[0]

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                
                # ============= Start Deterministic sampling =============
                if not edm_stoch:
                    t_hat = t_cur
                    x_hat = x_next
                # ============= End Deterministic sampling =============
                

                # # ============= Start stochastic sampling =============
                else:
                    noise = torch.randn_like(x_next, device=device)
                    
                    S_churn = 10
                    S_tmin = 0
                    S_tmax = 1e6
                    S_noise = 1

                    gamma = min(S_churn/self.noise_steps, 2**0.5 -1) if t_cur >= S_tmin and t_cur <= S_tmax else 0
                    noise = noise * S_noise
                    t_hat = t_cur + gamma * t_cur
                    x_hat = x_next + (t_hat**2 - t_cur**2)**0.5 * noise
                # # ============= End stochastic sampling =============

                # Euler step.
                denoised = self.edm(x_hat, t_hat, model, initial_cond=initial_cond, **kwargs)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                if edm_solver == 'heun':
                    # Apply 2nd order correction.
                    if i < num_steps - 1:
                        denoised = self.edm(x_next, t_next, model, initial_cond=initial_cond, **kwargs)
                        d_prime = (x_next - denoised) / t_next
                        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                                                
        return x_next