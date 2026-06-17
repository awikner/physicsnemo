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

r"""Faithful port of `PanguWeather v2.0
<https://github.com/198808xc/Pangu-Weather>`_ building blocks.

Module-private helpers for :class:`~physicsnemo.experimental.models.pangu_plasim.PanguPlasim`
and :class:`~physicsnemo.experimental.models.pangu_plasim.PanguPlasimLegacy`. Submodule names
and tensor shapes match the source so checkpoints translated from the original
PanguWeather repo load 1:1.

The original source had an opt-in NVIDIA Transformer Engine (``USE_TE``) code
path with unguarded references to a non-imported ``te`` module — i.e. broken
dead code. That path is **omitted here**. If/when TE support is wanted, it can
be reintroduced cleanly behind a proper ``check_version_spec`` guard per the
EXTERNAL_IMPORTS standard.
"""

import torch
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from torch import nn

# Phase A swap: pure utilities come from physicsnemo.nn.module.utils.
# get_shift_window_mask stays local — physicsnemo's version has the
# longitude-partitioning bug from issue #1599; Phase C vendors a fixed copy.
from physicsnemo.nn.module.utils import (
    crop3d,
    get_earth_position_index,
    get_pad3d,
    window_partition,
    window_reverse,
)

# Phase B swap: at downsample/upsample factor=2 the local implementation is
# bit-identical to physicsnemo.nn.{Down,Up}Sample3D (same modules, same forward).
# We dispatch through factory classes below so the factor=2 path uses upstream.
from physicsnemo.nn import DownSample3D as _UpstreamDownSample3D
from physicsnemo.nn import UpSample3D as _UpstreamUpSample3D

from ._pangu_utils import get_shift_window_mask


class Mask(nn.Module):
    r"""Element-wise multiplicative mask with optional additive fill.

    Stores ``mask`` and (optionally) ``mask_fill`` as non-trainable
    :class:`torch.nn.Parameter` so they live in ``state_dict`` and roundtrip
    through ``.mdlus`` checkpoints — matching the original PanguWeather
    behavior.

    Parameters
    ----------
    mask : torch.Tensor
        Mask of shape :math:`(H, W)` broadcast over ``(B, C, H, W)`` inputs.
    mask_fill : torch.Tensor, optional, default=None
        Additive fill of shape :math:`(C, H, W)`. When ``None``, only the
        multiplicative mask is applied.

    Forward
    -------
    x : torch.Tensor
        Tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        ``x * mask`` when ``mask_fill`` is ``None``; otherwise
        ``x * mask + mask_fill``.
    """

    def __init__(self, mask, mask_fill=None):
        super().__init__()
        self.mask = nn.parameter.Parameter(
            mask.unsqueeze(0).unsqueeze(0), requires_grad=False
        )
        if mask_fill is not None:
            self.mask_fill = nn.parameter.Parameter(
                mask_fill.unsqueeze(0), requires_grad=False
            )
        else:
            self.mask_fill = None

    def forward(self, x):
        if self.mask_fill is not None:
            return x * self.mask + self.mask_fill
        return x * self.mask


class _LocalDownSample3D(nn.Module):
    r"""Generalized parametric down-sample (any ``downsample_factor``).

    Used by :class:`DownSample` as the fallback for ``downsample_factor != 2``;
    at ``downsample_factor == 2`` the factory swaps in
    :class:`physicsnemo.nn.DownSample3D` instead (identical numerics, identical
    state-dict keys).

    Parameters and behavior match the original PanguWeather v2.0 pseudocode,
    parametrized on ``downsample_factor`` (the original only supported 2 in
    practice).
    """

    def __init__(self, in_dim, input_resolution, output_resolution, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.linear = nn.Linear(
            in_dim * (self.downsample_factor**2),
            in_dim * self.downsample_factor,
            bias=False,
        )
        self.norm = nn.LayerNorm((self.downsample_factor**2) * in_dim)

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "pressure-level dimension must not change in DownSample"
        h_pad = out_lat * self.downsample_factor - in_lat
        w_pad = out_lon * self.downsample_factor - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Pad lat/lon so they cleanly divide by `downsample_factor`.
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        # Fold each (downsample_factor, downsample_factor) tile into the channel axis.
        x = x.reshape(
            B,
            in_pl,
            out_lat,
            self.downsample_factor,
            out_lon,
            self.downsample_factor,
            C,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(
            B,
            out_pl * out_lat * out_lon,
            (self.downsample_factor**2) * C,
        )

        x = self.norm(x)
        x = self.linear(x)
        return x


class DownSample:
    r"""Pangu-Weather lat/lon down-sampling factory.

    For the common ``downsample_factor == 2`` path returns a
    :class:`physicsnemo.nn.DownSample3D` instance — the upstream reference
    implementation, bit-identical with the local generalized impl at
    ``factor=2`` (same submodule names ``linear``, ``norm``, ``pad``, same
    algorithm, same numerics). For any other factor returns the local
    :class:`_LocalDownSample3D` fallback.

    state_dict keys are identical across both paths (translated PanguWeather
    checkpoints continue to load), so this is the swap point for Phase B of
    ``pangu_plasim_reuse_plan.md``.

    Adapted from the `Pangu-Weather pseudocode
    <https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py>`_.

    Parameters
    ----------
    in_dim : int
        Number of input channels :math:`C`.
    input_resolution : tuple of int
        ``(Pl, Lat, Lon)`` pre-downsample resolution.
    output_resolution : tuple of int
        ``(Pl, Lat, Lon)`` post-downsample resolution. ``Pl`` must equal the
        input's.
    downsample_factor : int, optional, default=2
        Lat/lon down-sampling factor.

    Forward
    -------
    x : torch.Tensor
        Flattened tokens of shape :math:`(B, Pl \cdot Lat \cdot Lon, C)`.

    Outputs
    -------
    torch.Tensor
        Tokens of shape :math:`(B, Pl \cdot Lat_{out} \cdot Lon_{out},
        C \cdot \text{downsample\_factor})`.
    """

    def __new__(
        cls, in_dim, input_resolution, output_resolution, downsample_factor=2
    ):
        if downsample_factor == 2:
            return _UpstreamDownSample3D(in_dim, input_resolution, output_resolution)
        return _LocalDownSample3D(
            in_dim, input_resolution, output_resolution, downsample_factor
        )


class _LocalUpSample3D(nn.Module):
    r"""Generalized parametric up-sample (any ``upsample_factor``).

    Fallback for :class:`UpSample` when ``upsample_factor != 2``; at
    ``upsample_factor == 2`` the factory returns :class:`physicsnemo.nn.UpSample3D`
    instead (identical numerics, identical state-dict keys).
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.linear1 = nn.Linear(in_dim, out_dim * (upsample_factor**2), bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(
            B,
            in_pl,
            in_lat,
            in_lon,
            self.upsample_factor,
            self.upsample_factor,
            C // self.upsample_factor,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(
            B,
            in_pl,
            in_lat * self.upsample_factor,
            in_lon * self.upsample_factor,
            -1,
        )

        assert in_pl == out_pl, "pressure-level dimension must not change in UpSample"
        pad_h = in_lat * self.upsample_factor - out_lat
        pad_w = in_lon * self.upsample_factor - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :,
            :out_pl,
            pad_top : self.upsample_factor * in_lat - pad_bottom,
            pad_left : self.upsample_factor * in_lon - pad_right,
            :,
        ]
        x = x.reshape(
            x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4]
        )
        x = self.norm(x)
        x = self.linear2(x)
        return x


class UpSample:
    r"""Pangu-Weather lat/lon up-sampling factory (inverse of :class:`DownSample`).

    For the common ``upsample_factor == 2`` path returns a
    :class:`physicsnemo.nn.UpSample3D` instance — the upstream reference
    implementation, bit-identical with the local generalized impl at ``factor=2``
    (same submodule names ``linear1``, ``linear2``, ``norm``, same algorithm,
    same numerics). For any other factor returns the local
    :class:`_LocalUpSample3D` fallback.

    state_dict keys are identical across both paths (Phase B of
    ``pangu_plasim_reuse_plan.md``).

    Parameters
    ----------
    in_dim : int
        Number of input channels.
    out_dim : int
        Number of output channels.
    input_resolution : tuple of int
        ``(Pl, Lat, Lon)`` pre-upsample resolution.
    output_resolution : tuple of int
        ``(Pl, Lat, Lon)`` post-upsample resolution. ``Pl`` must equal the
        input's.
    upsample_factor : int, optional, default=2
        Lat/lon up-sampling factor.

    Forward
    -------
    x : torch.Tensor
        Flattened tokens of shape :math:`(B, Pl \cdot Lat \cdot Lon, C_{in})`.

    Outputs
    -------
    torch.Tensor
        Tokens of shape :math:`(B, Pl \cdot Lat_{out} \cdot Lon_{out}, C_{out})`.
    """

    def __new__(
        cls, in_dim, out_dim, input_resolution, output_resolution, upsample_factor=2
    ):
        if upsample_factor == 2:
            return _UpstreamUpSample3D(
                in_dim, out_dim, input_resolution, output_resolution
            )
        return _LocalUpSample3D(
            in_dim, out_dim, input_resolution, output_resolution, upsample_factor
        )


class EarthSpecificLayer(nn.Module):
    r"""Stack of :class:`EarthSpecificBlock` transformer layers.

    Alternates "shift-by-zero" and "shift-by-half-window" blocks per the
    standard Swin Transformer pattern.

    Parameters
    ----------
    dim : int
        Number of input channels.
    input_resolution : tuple of int
        ``(Pl, Lat, Lon)`` resolution at this stage.
    depth : int
        Number of stacked :class:`EarthSpecificBlock` blocks.
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
        Whether window shift includes the vertical (pressure) axis.
    checkpointing : int, optional, default=0
        Activation-checkpointing depth (kept for parent-model compatibility;
        not active inside this layer's forward).
    use_reentrant : bool, optional, default=False
        ``use_reentrant`` flag forwarded to activation checkpointing.

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
        checkpointing=0,
        use_reentrant=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.checkpointing = checkpointing
        self.use_reentrant = use_reentrant

        self.blocks = nn.ModuleList(
            [
                EarthSpecificBlock(
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
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    vertical_windowing=vertical_windowing,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class EarthSpecificBlock(nn.Module):
    r"""Single Earth-Specific 3D Swin transformer block.

    LayerNorm → padded 3D window attention with earth-specific position bias
    → drop-path → MLP. Shifted windows alternate per layer.

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
        axis (subject to ``vertical_windowing``).
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
        If ``False``, never shift along the vertical (pressure) axis.

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

        # Pad to a multiple of the window size in each axis.
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
            x_windows.shape[0],
            x_windows.shape[1],
            win_pl * win_lat * win_lon,
            C,
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(
            attn_windows.shape[0],
            attn_windows.shape[1],
            win_pl,
            win_lat,
            win_lon,
            C,
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
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
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

        x = x.permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    r"""Two-layer MLP used inside :class:`EarthSpecificBlock`.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    hidden_features : int, optional, default=None
        Hidden dim; defaults to ``in_features``.
    out_features : int, optional, default=None
        Output dim; defaults to ``in_features``.
    act_layer : type, optional, default=torch.nn.GELU
        Activation layer.
    drop : float, optional, default=0.0
        Dropout rate.

    Forward
    -------
    x : torch.Tensor
        Tokens of shape :math:`(B, N, \text{in\_features})`.

    Outputs
    -------
    torch.Tensor
        Tokens of shape :math:`(B, N, \text{out\_features})`.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
