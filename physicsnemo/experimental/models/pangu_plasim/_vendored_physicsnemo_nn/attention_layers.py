# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Vendored + patched :class:`physicsnemo.nn.module.attention_layers.EarthAttention3D`.

Patch:

* **``use_sdpa: bool = True`` kwarg.** When ``True`` (the default in this
  vendored copy — matching the ai-rossby Pangu_Plasim ports' existing
  behavior), routes through :func:`torch.nn.functional.scaled_dot_product_attention`
  for the kernel-fused fast path. When ``False`` reverts to upstream's explicit
  ``q @ k.T → softmax → attn @ v`` path (bit-identical with the upstream
  class).

When the upstream PR adding ``use_sdpa`` lands, replace imports of this module
with ``physicsnemo.nn.module.attention_layers.EarthAttention3D`` and pass
``use_sdpa=True`` at the call site (the upstream default will be ``False`` for
backward compat with existing checkpoints' numerics).
"""

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn

from physicsnemo.nn.module.utils import get_earth_position_index


class EarthAttention3D(nn.Module):
    r"""3D-window self-attention with earth-specific position bias.

    Parameters
    ----------
    dim : int
        Number of input channels.
    input_resolution : tuple of int
        ``(Pl, Lat, Lon)`` padded resolution.
    window_size : tuple of int
        ``(Wpl, Wlat, Wlon)`` attention-window size.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool, optional, default=True
        Whether the attention projection learns a bias.
    qk_scale : float, optional, default=None
        Override default qk scale of :math:`\sqrt{d_{head}}^{-1}`.
    attn_drop : float, optional, default=0.0
        Dropout on attention weights.
    proj_drop : float, optional, default=0.0
        Dropout on output projection.
    use_sdpa : bool, optional, default=True
        Route attention through :func:`F.scaled_dot_product_attention` when
        ``True``; use the explicit matmul-softmax-matmul path when ``False``.

    Forward
    -------
    x : torch.Tensor
        Tokens of shape :math:`(B \cdot \text{nLon}, \text{nPl} \cdot \text{nLat},
        W_{pl} \cdot W_{lat} \cdot W_{lon}, C)`.
    mask : torch.Tensor, optional
        Shift-window attention mask of shape :math:`(\text{nLon},
        \text{nPl} \cdot \text{nLat}, W_{pl} W_{lat} W_{lon}, W_{pl} W_{lat} W_{lon})`.

    Outputs
    -------
    torch.Tensor
        Tokens of the same shape as ``x``.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_sdpa = use_sdpa
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )

        earth_position_index = get_earth_position_index(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = torch.unbind(qkv, 0)
        L = q.shape[-1]

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )
        earth_position_bias = (
            earth_position_bias.permute(3, 2, 0, 1).contiguous().unsqueeze(0)
        )

        if self.use_sdpa:
            if mask is not None:
                nLon = mask.shape[0]
                x = F.scaled_dot_product_attention(
                    q.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                    k.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                    v.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                    attn_mask=earth_position_bias.unsqueeze(0) + mask.unsqueeze(1).unsqueeze(0),
                    scale=self.scale,
                )
                x = x.view(-1, self.num_heads, nW_, N, L)
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=earth_position_bias, scale=self.scale
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + earth_position_bias
            if mask is not None:
                nLon = mask.shape[0]
                attn = attn.view(
                    B_ // nLon, nLon, self.num_heads, nW_, N, N
                ) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
