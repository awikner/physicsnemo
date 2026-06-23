# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/layers/positional_encoding.py) for Phase 8a.
# Pure tensor-ops module — no upstream-private imports, no edits needed.

import math

import numpy as np
import torch
from einops import rearrange
from torch import nn


# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base_freq=10000):
        super().__init__()
        inv_freq = 1. / (base_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, coordinates, scale=1):
        t = coordinates
        t = t * scale
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [..., n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [..., n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, cos_freqs, sin_freqs):
    return (t * cos_freqs) + (rotate_half(t) * sin_freqs)


def apply_2d_rotary_pos_emb(t,
                            cos_freqs_x, sin_freqs_x,
                            cos_freqs_y, sin_freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, cos_freqs_x, sin_freqs_x),
                      apply_rotary_pos_emb(t_y, cos_freqs_y, sin_freqs_y)), dim=-1)


def apply_3d_rotary_pos_emb(t,
                            cos_freqs_x, sin_freqs_x,
                            cos_freqs_y, sin_freqs_y,
                            cos_freqs_z, sin_freqs_z):

    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y, t_z = t[..., :d//3], t[..., d//3:2*d//3], t[..., 2*d//3:]

    return torch.cat((apply_rotary_pos_emb(t_x, cos_freqs_x, sin_freqs_x),
                      apply_rotary_pos_emb(t_y, cos_freqs_y, sin_freqs_y),
                      apply_rotary_pos_emb(t_z, cos_freqs_z, sin_freqs_z)), dim=-1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, num_conds=1, cache_freqs=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size*num_conds, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size
        self.num_conds = num_conds
        self.cache_freqs = cache_freqs

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, freqs=None):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if freqs is None:
            half = dim // 2
            freqs = torch.exp(
                - math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half).to(device=t.device)

        args = torch.einsum('bc,d->bcd', t.float(), freqs) # [b, num_conds, dim//2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding, freqs

    def forward(self, t):
        assert t.shape[-1] == self.num_conds
        if self.cache_freqs and hasattr(self, 'freqs'):
            t_emb, _ = self.timestep_embedding(t, self.frequency_embedding_size, freqs=self.freqs)
        else:
            t_emb, freqs = self.timestep_embedding(t, self.frequency_embedding_size)
            if self.cache_freqs:
                self.freqs = freqs
        t_emb = rearrange(t_emb, 'b nc d -> b (nc d)')
        t_emb = self.mlp(t_emb)
        return t_emb


# Gaussian Fourier features
# code modified from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels,
                 mapping_size=256, scale=10, learnable=False):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale,
                               requires_grad=learnable)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    
class FourierEmb(nn.Module):
    def __init__(self, hidden_dim, in_dim):
        super().__init__()
        self.hidden_dim = hidden_dim    
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, hidden_dim//2)
        self.scale = 2 * torch.pi

    def forward(self, x):
        # x: [b, n, in_dim] 
        x = self.scale * self.linear(x)
        y = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return y


class RadialBesselBasis(nn.Module):
    # head-wise positional encoding
    def __init__(
            self,
            num_kernels,
            num_heads,
            enforce_periodicity=False,
            trainable=False,
            act_fn=None,
    ):
        super().__init__()
        freqs = torch.arange(1, num_kernels+1).float()
        self.freqs = nn.Parameter(freqs, requires_grad=trainable)
        self.weights = nn.Parameter(torch.randn(num_heads, num_kernels) / np.sqrt(num_kernels), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(num_heads) / num_heads, requires_grad=True)

        self.enforce_periodicity = enforce_periodicity
        self.num_heads = num_heads
        self.num_kernels = num_kernels
        self.act_fn = act_fn

    def forward(self, angle, cache=True):
        # angles [n x n] assuming in radians like [0, pi]
        if not cache or not hasattr(self, 'angle'):
            if self.enforce_periodicity:
                # theta = min(theta, 2pi - theta)
                angle = torch.min(angle, 2 * np.pi - angle)
            # add a small epsilon to the zero element in angle to avoid division by zero
            angle[angle == 0] = 1e-6
            if cache:
                self.angle = angle.detach()
        else:
            angle = self.angle
        theta = torch.einsum('i j, d -> i j d', angle, self.freqs)

        basis = torch.sin(theta) / theta * np.sqrt(2 / np.pi)
        decay = torch.einsum('i j c, h c -> h i j', basis, self.weights)
        decay = torch.einsum('h, h i j -> h i j', self.bias, decay)

        return self.act_fn(decay) if self.act_fn is not None else decay


