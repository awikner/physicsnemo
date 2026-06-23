# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/diffusion/utils.py) for Phase 8a. Pure-PyTorch (plus
# torch_harmonics for SphereNoiseGenerator) — no upstream-private
# imports.

import torch
import torch.nn as nn
from einops import rearrange

def get_log_uniform_t(t_final = 0.999, scale=1.3, n_t = 10, device = "cpu"):
    t_s = []
    t_0 = 0.0

    t_s.append(t_0)

    r = scale * (1-t_final)**(1/n_t)

    assert r < 1.0, "scale is too large for given t_final and n_t, resulting in r >= 1.0"

    for _ in range(n_t):
        delta_t = (1-r) * (1-t_0)
        t_s.append(t_0 + delta_t)
        t_0 = t_0 + delta_t

    return torch.tensor(t_s, device = device), torch.tensor(r, device = device)


def sample_logit_normal(shape, m=0.0, s=1.0, device='cpu', dtype=torch.float32):
    """
    Samples from a logit-normal distribution.
    
    Args:
        shape (tuple or int): The shape of the desired output tensor (e.g., batch size).
        m (float or torch.Tensor): Location parameter (mean of the underlying normal distribution).
                                   Negative biases towards data (p0), positive towards noise (p1).
        s (float or torch.Tensor): Scale parameter (standard deviation of the normal distribution).
        device (str or torch.device): Device to place the tensor on.
        dtype (torch.dtype): Data type of the tensor.
        
    Returns:
        torch.Tensor: Timestep samples 't' in the range (0, 1).
    """
    # 1. Sample u ~ N(m, s)
    # torch.randn generates samples from N(0, 1)
    u = torch.randn(shape, device=device, dtype=dtype)
    u = u * s + m
    
    # 2. Map it through the standard logistic function (sigmoid)
    # sigmoid(u) = 1 / (1 + exp(-u))
    t = torch.sigmoid(u)
    return t

def sample_power_law(n_steps, rho, device = 'cpu'):
    """
    Sample timesteps according to a power-law distribution.
    
    Args:
        n_steps (int): Number of timesteps to sample.
        rho (float): Power-law exponent. Higher values concentrate samples near 0.
    
    Returns:
        torch.Tensor: Timesteps sampled from the power-law distribution, in the range (0, 1).
    """
    n = torch.arange(0, n_steps, device=device, dtype=torch.float32)
    t = (1 -n / (n_steps-1)) ** rho
    
    # returns n_steps values from 1 to 0, with more concentration near 0 for higher rho
    return t

def power_sampler(batch_size, p=2.0, device = "cpu"):
    t = torch.rand(batch_size, device=device)
    return t ** p

class SphereNoiseGenerator(nn.Module):
    def __init__(self, l_max):
        super(SphereNoiseGenerator, self).__init__()
        from torch_harmonics import InverseRealSHT

        self.l_max = l_max
        self.isht = InverseRealSHT(lmax = l_max, nlat=l_max, nlon=l_max * 2, grid="equiangular")

    def forward(self, b, c, device, dtype=torch.complex64, l_max=None):
        # sample coefficient in the frequency domain
        # b: batch size, l_max: maximum degree
        # return: [b, l_max, l_max + 1] # coefficient for real harmonics
        if l_max is None:
            l_max = self.l_max
            coeffs = torch.randn(b*c, l_max, l_max + 1, device=device, dtype=dtype)
        else:
            assert l_max <= self.l_max
            coeffs = torch.randn(b*c, self.l_max, self.l_max + 1, device=device, dtype=dtype)
            # fill with zeros
            coeffs[:, l_max:, :] = 0

        noise = self.isht(coeffs)
        noise = rearrange(noise, '(b c) h w -> b c h w ', b=b, c=c)
        noise_means = torch.mean(noise, dim=(2, 3), keepdim=True)
        noise_stds = torch.std(noise, dim=(2, 3), keepdim=True)
        noise = (noise - noise_means) / noise_stds

        return noise

def compute_channel_variances(data: torch.Tensor, sigma_base: float = 1.0, gamma: float = 1.0) -> torch.Tensor:
    """
    Computes channel-specific noise variances based on the spectral 
    complexity (high-frequency energy) of each channel.
    
    Args:
        data: A PyTorch tensor of shape (C, X, Y) containing real-valued spatial fields.
        sigma_base: The baseline noise standard deviation (default: 1.0).
        gamma: Hyperparameter controlling how aggressively to scale based on complexity.
        
    Returns:
        sigma_c: A 1D tensor of shape (C,) with the target variance scale for each channel.
    """
    # Ensure data is at least 3D (C, X, Y)
    if data.dim() != 3:
        raise ValueError(f"Expected data to be 3D (C, X, Y), but got shape {data.shape}")
        
    C, X, Y = data.shape
    device = data.device
    
    # 1. Compute the 2D Fast Fourier Transform
    # We use fft2 for spatial data. 
    fft_data = torch.fft.fft2(data)
    
    # Shift the zero-frequency component to the center of the spectrum
    fft_shifted = torch.fft.fftshift(fft_data, dim=(-2, -1))
    
    # 2. Compute Power Spectral Density (PSD)
    # The power is the squared magnitude of the complex Fourier coefficients
    psd = torch.abs(fft_shifted)**2
    
    # 3. Create a 2D grid of radial wavenumbers (spatial frequencies)
    # Get the normalized frequencies for both spatial dimensions
    freq_x = torch.fft.fftshift(torch.fft.fftfreq(X))
    freq_y = torch.fft.fftshift(torch.fft.fftfreq(Y))
    
    # Create a 2D meshgrid of these frequencies
    grid_x, grid_y = torch.meshgrid(freq_x, freq_y, indexing='ij')
    
    # Calculate the radial wavenumber (Euclidean distance from the zero-frequency center)
    k = torch.sqrt(grid_x**2 + grid_y**2).to(device)
    
    # Expand k to match the shape of the PSD tensor (C, X, Y)
    k_expanded = k.unsqueeze(0).expand(C, -1, -1)
    
    # 4. Calculate Spectral Complexity for each channel
    # This is the spectral centroid: sum(k * PSD) / sum(PSD)
    numerator = torch.sum(k_expanded * psd, dim=(-2, -1))
    denominator = torch.sum(psd, dim=(-2, -1))
    
    # Add a small epsilon to the denominator to prevent division by zero on flat fields
    chi_c = numerator / (denominator + 1e-8)
    
    # 5. Compute the scaling factors
    # Find the reference complexity (mean across all 80 channels)
    chi_ref = torch.mean(chi_c)
    
    # Apply the power-law scaling scheme
    sigma_c = sigma_base * (chi_c / chi_ref)**gamma
    
    return sigma_c