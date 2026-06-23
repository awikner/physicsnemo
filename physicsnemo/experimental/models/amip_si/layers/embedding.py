# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/layers/embedding.py) for Phase 8a.

import math

import torch
import torch.nn as nn
from einops import repeat

def fourier_acyclic(x, num_freqs, min_freq=1.0, max_freq=1000.0):
    """Log-spaced Fourier features for a NON-periodic scalar (x ~ [0,1] or [-1,1])."""
    freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq),
                           num_freqs, device=x.device, dtype=x.dtype)
    ang = 2 * math.pi * x[..., None] * freqs              # (..., num_freqs)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)      # (..., 2*num_freqs)


def fourier_cyclic(x, num_harmonics):
    """Integer-harmonic features for a CYCLIC scalar normalized to [0,1].
    Integer freqs => identical values at x=0 and x=1 (no seam)."""
    k = torch.arange(1, num_harmonics + 1, device=x.device, dtype=x.dtype)
    ang = 2 * math.pi * x[..., None] * k
    return torch.cat([ang.sin(), ang.cos()], dim=-1)

class ScalarEmbed(nn.Module):
    def __init__(self, in_features, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
    def forward(self, feats):
        return self.net(feats)

class ScalarEmbedding(nn.Module):
    def __init__(self, dim=256, t_freqs=64, tod_harm=12, doy_harm=12, co2_freqs=24, use_t = True, use_co2=True):
        super().__init__()
        self.cfg = dict(t_freqs=t_freqs, tod_harm=tod_harm,
                        doy_harm=doy_harm, co2_freqs=co2_freqs)
        self.use_t = use_t
        self.use_co2 = use_co2

        self.in_dim = dim*2

        if use_t:
            self.t_embed   = ScalarEmbed(2 * t_freqs,   dim)
            self.in_dim += dim
        self.tod_embed = ScalarEmbed(2 * tod_harm,  dim)
        self.doy_embed = ScalarEmbed(2 * doy_harm,  dim)
        if use_co2:
            self.co2_embed = ScalarEmbed(2 * co2_freqs, dim)
            self.in_dim += dim
        
        self.out_proj = nn.Linear(self.in_dim, dim)

    def forward(self, t=None, c_scalar=None):
        sod = c_scalar[:, :1] # n 1
        doy = c_scalar[:, 1:2] # n 1
        sod = sod / 86400 # seconds in a day
        doy = doy / 365.25 # days in a year
        all_conds = []

        if self.use_t:
            e_t   = self.t_embed(fourier_acyclic(t, self.cfg['t_freqs'], 
                                                   min_freq=0.5, max_freq=1000.0))
            all_conds.append(e_t)

        e_tod = self.tod_embed(fourier_cyclic( sod, self.cfg['tod_harm']))
        all_conds.append(e_tod)
        e_doy = self.doy_embed(fourier_cyclic( doy, self.cfg['doy_harm']))
        all_conds.append(e_doy)
        
        if self.use_co2:
            co2 = c_scalar[:, 2:] # n 1
            e_co2 = self.co2_embed(fourier_acyclic(co2, self.cfg['co2_freqs'],
                                        min_freq=0.1, max_freq=8.0))
            all_conds.append(e_co2)
        
        cond = torch.cat(all_conds, dim=-1) # [b, cond_dim]
        out = self.out_proj(cond)
        return out.squeeze(1)

class FrequencyEmbedding(torch.nn.Module):
    """Periodic Embedding.

    Useful for inputs defined on the circle [0, 2pi)
    """

    def __init__(self, num_channels):
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(1, num_channels + 1), persistent=False
        )

    def forward(self, x):
        freqs = self.freqs[None, :, None, None]
        x = x[:, None, :, :]
        x = x * (2 * math.pi * freqs).to(x.dtype)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CalendarEmbedding(torch.nn.Module):
    """Time embedding assuming 365.25 day years.

    Args:
        calendar: (n, 2) when ``use_co2`` is False -> [second_of_day, day_of_year]
                  (n, 3) when ``use_co2`` is True  -> [second_of_day, day_of_year, co2]
    Returns:
        (n, embed_channels, nlat, nlon)
    """

    def __init__(self, nlon, nlat, embed_channels: int, use_co2: bool = True):
        super().__init__()

        lon = (torch.arange(nlon, dtype=torch.float32) + 0.5) / nlon * 360

        self.nlat = nlat
        self.nlon = nlon
        self.use_co2 = use_co2

        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)
        if use_co2: # Don't use periodic embedding for co2! We normalized sod/doy to be [0,1] so it is periodic on that interval, but co2 is not periodic.
            self.embed_co2 = nn.Sequential(
                nn.Linear(1, embed_channels),
                nn.GELU(),
                nn.Linear(embed_channels, embed_channels * 2))
            self.out_channels = embed_channels * 6
        else:
            self.embed_co2 = None
            self.out_channels = embed_channels * 4
        self.out_proj = torch.nn.Linear(self.out_channels, self.embed_channels)

    def forward(self, calendar):

        second_of_day = calendar[:, :1] # n 1
        day_of_year = calendar[:, 1:2] # n 1

        local_time = (second_of_day.unsqueeze(2) + self.lon * 86400 // 360) % 86400 # n 1 nlon


        a = self.embed_second(local_time / 86400) # n, embed_channels * 2, 1, nlon
        doy = day_of_year.unsqueeze(2) # n 1 1
        b = self.embed_day((doy / 365.25) % 1) # n, embed_channels * 2, 1, 1

        b = repeat(b, "n c 1 1 -> n c 1 nlon", nlon=self.nlon) # n, embed_channels * 2, 1, nlon

        if self.use_co2:
            co2 = calendar[:, 2:] # n 1
            c = self.embed_co2(co2) # n, embed_channels * 2
            c = repeat(c, "n c -> n c 1 nlon", nlon=self.nlon) # n, embed_channels * 2, 1, nlon
            out = torch.cat([a, b, c], dim=1) # n, embed_channels * 6, 1, nlon
        else:
            out = torch.cat([a, b], dim=1)

        out = self.out_proj(out.transpose(1, 3)).transpose(1, 3) # n, embed_channels, 1, nlon

        # repeat to n c nlat nlon
        out = out.expand(-1, -1, self.nlat, -1)

        return out
    