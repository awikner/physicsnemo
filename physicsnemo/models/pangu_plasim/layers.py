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

"""Faithful port of PanguWeather v2.0 building blocks (adapted from the
Pangu-Weather architecture, https://github.com/198808xc/Pangu-Weather).
Names/shapes preserved for checkpoint weight-compatibility.
"""

import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

from ._pangu_utils import (
    get_earth_position_index,
    get_pad3d,
    get_shift_window_mask,
    window_partition,
    window_reverse,
    crop3d,
)


USE_TE = False



# # Conditional imports
# if USE_TE:
#     import transformer_engine.pytorch as te
#     from transformer_engine.common import recipe
#     from torch.cuda import amp

#     fp8_recipe = recipe.DelayedScaling(
#         fp8_format=recipe.Format.HYBRID,
#         amax_history_len=16,
#         amax_compute_algo="max"
#     )


class Mask(nn.Module):
    def __init__(self, mask, mask_fill = None):
        super().__init__()
        self.mask = nn.parameter.Parameter(mask.unsqueeze(0).unsqueeze(0), requires_grad=False)
        if type(mask_fill) is not type(None):
            self.mask_fill = nn.parameter.Parameter(mask_fill.unsqueeze(0), requires_grad=False)
        else:
            self.mask_fill = None

    def forward(self, x):
        if type(self.mask_fill) is not type(None):
            return x * self.mask + self.mask_fill
        else:
            return x * self.mask


class DownSample(nn.Module):
    """
    Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor

        if USE_TE:
            self.linear = te.Linear(in_dim * (self.downsample_factor ** 2), in_dim * self.downsample_factor, bias=False)
            self.norm = te.LayerNorm((self.downsample_factor ** 2) * in_dim)
        else:
            self.linear = nn.Linear(in_dim * (self.downsample_factor ** 2), in_dim * self.downsample_factor, bias=False)
            self.norm = nn.LayerNorm((self.downsample_factor ** 2) * in_dim)


        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * self.downsample_factor - in_lat
        w_pad = out_lon * self.downsample_factor - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, self.downsample_factor, out_lon, self.downsample_factor, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, (self.downsample_factor ** 2) * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class UpSample(nn.Module):
    """
    Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution, upsample_factor=2):
        super().__init__()
        self.upsample_factor = upsample_factor

        if USE_TE:
            self.linear1 = te.Linear(in_dim, out_dim * (upsample_factor ** 2), bias=False)
            self.linear2 = te.Linear(out_dim, out_dim, bias=False)
            self.norm = te.LayerNorm(out_dim)
        else:
            self.linear1 = nn.Linear(in_dim, out_dim * (upsample_factor ** 2), bias=False)
            self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
            self.norm = nn.LayerNorm(out_dim)

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, self.upsample_factor, self.upsample_factor, C // self.upsample_factor).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * self.upsample_factor, in_lon * self.upsample_factor, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * self.upsample_factor - out_lat
        pad_w = in_lon * self.upsample_factor - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top: self.upsample_factor * in_lat - pad_bottom, pad_left: self.upsample_factor * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x



class EarthSpecificLayer(nn.Module): #BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer = nn.LayerNorm, vertical_windowing = True, checkpointing = 0,
                 use_reentrant = False): # Using TE here is not working.
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        norm_layer = te.LayerNorm if USE_TE else nn.LayerNorm
        self.checkpointing = checkpointing
        self.use_reentrant = use_reentrant


        self.blocks = nn.ModuleList([
            EarthSpecificBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer, vertical_windowing = vertical_windowing)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            # if self.checkpointing > 2 and train:
            #     x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            # else:
            x = blk(x)
        return x



# CHANGE SO THAT I CAN REPLACE THE EARTHSPECIFIC LAYER NORMALIZATION SCHEME WITH TE. Must be contiguous before applying the normalization.

class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer = nn.LayerNorm,
                 vertical_windowing = True):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        if vertical_windowing:
            shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2) if shift_size is None else shift_size
        else:
            shift_size = (0, window_size[1] // 2, window_size[2] // 2) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        norm_layer = te.LayerNorm if USE_TE else nn.LayerNorm

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        # Ensure x is contiguous before normalization (TE CHANGE)
        if USE_TE:
            x = self.norm1(x.contiguous())
        else:
            x = self.norm1(x)

        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size

        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)

        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            # B * Pl * Lat * Lon * C
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        if USE_TE:
            x = x + self.drop_path(self.mlp(self.norm2(x.contiguous())))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros((window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                        self.type_of_windows, num_heads)
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH



        earth_position_index = get_earth_position_index(window_size)  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        if USE_TE:
            self.qkv = te.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = te.Linear(dim, dim)
            # self.dpa = DotProductAttention(num_attention_heads=num_heads, kv_channels=dim // num_heads, num_gqa_groups=num_heads)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        # Mem efficient attention doesn't have permute
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = torch.unbind(qkv, 0)
        L = q.shape[-1]

        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows, -1
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1).contiguous().unsqueeze(0)  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        #with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        if mask is not None:
            nLon = mask.shape[0]
            x = F.scaled_dot_product_attention(q.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                                                    k.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                                                    v.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                                                    attn_mask=earth_position_bias.unsqueeze(0) + \
                                                        mask.unsqueeze(1).unsqueeze(0),
                                                    scale = self.scale)
            x = x.view(-1, self.num_heads, nW_, N, L)
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=earth_position_bias, scale=self.scale)

        x = x.permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if USE_TE:
            self.fc1 = te.Linear(in_features, hidden_features)
            self.fc2 = te.Linear(hidden_features, out_features)
        else:
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
