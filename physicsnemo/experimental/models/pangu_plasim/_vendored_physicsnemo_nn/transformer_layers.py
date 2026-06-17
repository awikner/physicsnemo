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

r"""Vendored + patched :class:`physicsnemo.nn.module.transformer_layers.Transformer3DBlock`
and :class:`physicsnemo.nn.module.transformer_layers.FuserLayer`.

Patches:

* **``vertical_windowing: bool = True`` kwarg on Transformer3DBlock and
  FuserLayer.** Defaults to ``True`` (upstream behavior — shift along all three
  axes). When ``False``, ``shift_size`` along the pressure-level axis is forced
  to 0 and ``self.roll`` is gated on ``shift_lat and shift_lon`` only — matching
  the PanguWeather v2.0 ``vertical_windowing=False`` mode the Pangu_Plasim
  configs use.

The shift-window attention mask is constructed via the locally fixed
:func:`._vendored_physicsnemo_nn.shift_window_mask.get_shift_window_mask` (see
that module for the Issue `#1599
<https://github.com/NVIDIA/physicsnemo/issues/1599>`_ longitude-cyclic fix).

When the upstream PRs for ``vertical_windowing`` and the mask fix land,
replace imports of this module with
``physicsnemo.nn.module.transformer_layers.{Transformer3DBlock,FuserLayer}``
and delete this file.
"""

from collections.abc import Sequence

import torch
from timm.layers import DropPath
from torch import nn

from physicsnemo.nn.module.utils import (
    crop3d,
    get_pad3d,
    window_partition,
    window_reverse,
)

from .attention_layers import EarthAttention3D
from .mlp_layers import Mlp
from .shift_window_mask import get_shift_window_mask


class Transformer3DBlock(nn.Module):
    r"""Single Earth-Specific 3D Swin transformer block.

    Parameters
    ----------
    dim : int
        Number of input channels.
    input_resolution : tuple of int
        ``(Pl, Lat, Lon)`` resolution.
    num_heads : int
        Number of attention heads.
    window_size : tuple of int, optional, default=(2, 6, 12)
        ``(Wpl, Wlat, Wlon)`` attention-window size.
    shift_size : tuple of int, optional, default=None
        Cyclic shift size; ``None`` derives a half-window shift along each
        axis (or only lat/lon when ``vertical_windowing=False``).
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden dim to ``dim``.
    qkv_bias : bool, optional, default=True
        Whether the attention projection learns a bias.
    qk_scale : float, optional, default=None
        Override default qk scale of :math:`\sqrt{d_{head}}^{-1}`.
    drop : float, optional, default=0.0
        Dropout rate inside the MLP and attention output.
    attn_drop : float, optional, default=0.0
        Dropout rate on attention weights.
    drop_path : float, optional, default=0.0
        Stochastic-depth rate.
    act_layer : type, optional, default=torch.nn.GELU
        Activation layer.
    norm_layer : type, optional, default=torch.nn.LayerNorm
        Norm layer.
    vertical_windowing : bool, optional, default=True
        When ``False``, the pressure-level axis does not participate in the
        window shift (forces ``shift_pl = 0``).
    use_sdpa : bool, optional, default=True
        Forwarded to :class:`EarthAttention3D` — controls SDPA vs explicit
        matmul attention path.

    Forward
    -------
    x : torch.Tensor
        Tokens of shape :math:`(B, Pl \cdot Lat \cdot Lon, \text{dim})`.

    Outputs
    -------
    torch.Tensor
        Tokens of the same shape as ``x``.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        vertical_windowing=True,
        use_sdpa=True,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        if vertical_windowing:
            shift_size = (
                (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
                if shift_size is None
                else shift_size
            )
        else:
            shift_size = (
                (0, window_size[1] // 2, window_size[2] // 2)
                if shift_size is None
                else shift_size
            )
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=use_sdpa,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        if vertical_windowing:
            self.roll = shift_pl and shift_lon and shift_lat
        else:
            self.roll = shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape
        shift_pl, shift_lat, shift_lon = self.shift_size

        if self.roll:
            shifted_x = torch.roll(
                x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad
            )
            x = torch.roll(
                shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
            )
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad
            )
            x = shifted_x

        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FuserLayer(nn.Module):
    r"""Stack of :class:`Transformer3DBlock` blocks.

    Alternates "shift-by-zero" and "shift-by-half-window" blocks per the
    standard Swin Transformer pattern.

    Parameters mirror :class:`Transformer3DBlock` (one ``depth`` blocks; the
    block constructor is called with ``shift_size=(0,0,0)`` for even ``i`` and
    ``shift_size=None`` (auto half-window) for odd ``i``).

    Parameters
    ----------
    dim : int
        Number of input channels.
    input_resolution : tuple of int
        ``(Pl, Lat, Lon)`` resolution at this stage.
    depth : int
        Number of stacked blocks.
    num_heads : int
        Number of attention heads.
    window_size : tuple of int
        ``(Wpl, Wlat, Wlon)`` 3D attention-window size.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden dim to embedding dim.
    qkv_bias : bool, optional, default=True
        Whether the attention projection learns a bias.
    qk_scale : float, optional, default=None
        Override default qk scale of :math:`\sqrt{d_{head}}^{-1}`.
    drop : float, optional, default=0.0
        Dropout rate inside the MLP and attention output.
    attn_drop : float, optional, default=0.0
        Dropout rate on attention weights.
    drop_path : float or list of float, optional, default=0.0
        Stochastic-depth rate. May be a per-block list.
    norm_layer : type, optional, default=torch.nn.LayerNorm
        Norm layer class.
    vertical_windowing : bool, optional, default=True
        Forwarded to each :class:`Transformer3DBlock`.
    use_sdpa : bool, optional, default=True
        Forwarded to each block's :class:`EarthAttention3D`.

    Forward
    -------
    x : torch.Tensor
        Tokens of shape :math:`(B, Pl \cdot Lat \cdot Lon, \text{dim})`.

    Outputs
    -------
    torch.Tensor
        Tokens of the same shape as ``x``.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        vertical_windowing=True,
        use_sdpa=True,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                Transformer3DBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, (list, Sequence))
                    and not isinstance(drop_path, str)
                    else drop_path,
                    norm_layer=norm_layer,
                    vertical_windowing=vertical_windowing,
                    use_sdpa=use_sdpa,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
