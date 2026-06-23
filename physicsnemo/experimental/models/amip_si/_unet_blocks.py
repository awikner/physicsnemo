# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two UNet building blocks (``ResBlock`` + ``AttentionBlock``) extracted
from /work/nvme/bdiu/awikner/amip @ commit 497827e
(modules/models/Unet.py lines 11–98) for Phase 8a.

The upstream ``Unet.UNet`` class itself is deferred (Phase 8f); only
these two blocks are required by :class:`ERDM` (the rolling-window UNet
backbone) and are vendored here to keep the dependency edge tight.
"""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers.conv import SphereConv2d, nonlinearity


class ResBlock(nn.Module):
    """Residual block with SphereConv2d and timestep conditioning via AdaGN.

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    t_emb_dim : int
        Timestep embedding dimension.
    dropout : float, optional, default=0.1
        Dropout rate after the second activation.
    num_groups : int, optional, default=16
        Group count for the two :class:`GroupNorm` layers.

    Forward
    -------
    x : torch.Tensor
        Tensor of shape :math:`(B, C_{in}, H, W)`.
    t_emb : torch.Tensor
        Embedding of shape :math:`(B, \\text{t\\_emb\\_dim})`.

    Outputs
    -------
    torch.Tensor
        Tensor of shape :math:`(B, C_{out}, H, W)`.
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, dropout=0.1, num_groups=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6)
        self.conv1 = SphereConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Timestep projection -> scale and shift for AdaGN.
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels * 2),
        )

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = SphereConv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Zero-init the last conv so the residual block starts as identity.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        # AdaGN: scale and shift after norm2.
        scale_shift = self.t_proj(t_emb)[:, :, None, None]  # [b, 2*out_channels, 1, 1]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with :func:`F.scaled_dot_product_attention`.

    Parameters
    ----------
    channels : int
        Number of channels (and the attention embedding dim).
    num_heads : int, optional, default=8
        Number of attention heads.
    num_groups : int, optional, default=16
        Group count for the leading :class:`GroupNorm`.

    Forward
    -------
    x : torch.Tensor
        Tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Tensor of the same shape as ``x``.
    """

    def __init__(self, channels, num_heads=8, num_groups=16):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Zero-init output projection.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        nh = self.num_heads
        head_dim = c // nh

        qkv = self.qkv(self.norm(x))  # [b, 3*c, h, w]
        qkv = rearrange(qkv, "b (three nh hd) h w -> three b nh (h w) hd", three=3, nh=nh, hd=head_dim)
        q, k, v = qkv.unbind(0)  # each [b, nh, h*w, head_dim]

        out = F.scaled_dot_product_attention(q, k, v)  # [b, nh, h*w, head_dim]
        out = rearrange(out, "b nh (h w) hd -> b (nh hd) h w", h=h, w=w)

        return x + self.proj(out)


__all__ = ["AttentionBlock", "ResBlock"]
