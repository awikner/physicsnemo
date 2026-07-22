# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from the ArchesWeather deterministic backbone in INRIA/geoarches
# (BSD-3-Clause, Copyright (c) 2024-2025 ARCHES team @ INRIA),
# which itself adapts WeatherLearn
# (https://github.com/lizhuoq/WeatherLearn) and the Pangu-Weather pseudocode
# (https://github.com/198808xc/Pangu-Weather). The vertical token count
# (``zdim``) is generalized to a parameter so the same backbone serves our
# 17-level / 1-degree ERA5 grid (patch (2,3,3) -> zdim 10) in addition to the
# original 13-level / 1.5-degree grid (patch (2,2,2) -> zdim 8). The
# cross-level ("axial") attention is vendored here as a plain MHA over the Z
# axis (the ``num_dimensions=1`` special case of lucidrains'
# ``axial_attention``) so the model has no extra pip dependency.

from __future__ import annotations

import math

import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

# --------------------------------------------------------------------------- #
# WeatherLearn / Pangu window helpers (verbatim port; window_size-generic).
# --------------------------------------------------------------------------- #


def get_earth_position_index(window_size):
    """Position index that reuses symmetric position-bias parameters.

    Port of WeatherLearn's ``get_earth_position_index`` (from the Pangu-Weather
    pseudocode). ``window_size`` is ``(win_pl, win_lat, win_lon)``. Returns a
    ``(win_pl*win_lat*win_lon, win_pl*win_lat*win_lon)`` index tensor.
    """
    win_pl, win_lat, win_lon = window_size
    coords_zi = torch.arange(win_pl)
    coords_zj = -torch.arange(win_pl) * win_pl
    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat
    coords_w = torch.arange(win_lon)

    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w], indexing="ij"))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w], indexing="ij"))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat

    position_index = coords.sum(-1)
    return position_index


def get_pad3d(input_resolution, window_size):
    """Symmetric zero-pad amounts to make ``input_resolution`` window-divisible.

    Returns ``(left, right, top, bottom, front, back)`` for ``nn.ZeroPad3d``.
    Verbatim port of WeatherLearn's ``get_pad3d``.
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def crop3d(x: torch.Tensor, resolution):
    """Center-crop a ``(B, C, Pl, Lat, Lon)`` tensor back to ``resolution``."""
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front
    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]


def window_partition(x: torch.Tensor, window_size):
    """(B, Pl, Lat, Lon, C) -> (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)."""
    B, Pl, Lat, Lon, C = x.shape
    win_pl, win_lat, win_lon = window_size
    x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
    windows = (
        x.permute(0, 5, 1, 3, 2, 4, 6, 7)
        .contiguous()
        .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
    )
    return windows


def window_reverse(windows, window_size, Pl, Lat, Lon):
    """Inverse of :func:`window_partition`."""
    win_pl, win_lat, win_lon = window_size
    B = int(windows.shape[0] / (Lon / win_lon))
    x = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return x


# --------------------------------------------------------------------------- #
# Sub-pixel (PixelShuffle) decoder init.
# --------------------------------------------------------------------------- #


def ICNR_init(tensor, initializer, upscale_factor=2, *args, **kwargs):  # noqa: N802
    """ICNR initialization for a PixelShuffle conv weight (checkerboard-free)."""
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, (
        f"The size of the first dimension: tensor.shape[0] = {tensor.shape[0]} "
        f"is not divisible by square of upscale_factor: {upscale_factor}"
    )
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    new_tensor = sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)
    tensor.data.copy_(new_tensor)


# --------------------------------------------------------------------------- #
# Timestep (month / hour) embedder — DiT style.
# --------------------------------------------------------------------------- #


class TimestepEmbedder(nn.Module):
    """Embeds a scalar (month or hour) into a ``hidden_size`` vector (DiT)."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# --------------------------------------------------------------------------- #
# Up / Down sampling (Pangu) + MLP.
# --------------------------------------------------------------------------- #


class UpSample(nn.Module):
    """Pangu up-sampling (Z untouched, lat/lon x2)."""

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """Pangu down-sampling (Z untouched, lat/lon /2)."""

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon
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
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)
        x = self.norm(x)
        x = self.linear(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --------------------------------------------------------------------------- #
# Cross-Level ("axial") attention — vendored num_dimensions=1 case.
# --------------------------------------------------------------------------- #


class AxialPositionalEmbedding1D(nn.Module):
    """Learned per-Z-token positional embedding (lucidrains, single axis)."""

    def __init__(self, dim: int, zdim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, zdim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, zdim, dim)
        return x + self.pos


class CrossLevelAttention(nn.Module):
    """Multi-head self-attention over the vertical (Z) token axis.

    This is exactly the ``num_dimensions=1`` / ``sum_axial_out=True`` case of
    lucidrains' ``AxialAttention`` (a single self-attention over the one axial
    dimension), reimplemented so the model needs no ``axial_attention`` dep.
    Bias-free q/k/v projections match the reference ``SelfAttention``.
    """

    def __init__(self, dim: int, heads: int = 8, dim_heads: int | None = None):
        super().__init__()
        self.dim_heads = dim_heads if dim_heads is not None else dim // heads
        self.heads = heads
        hidden_dim = self.dim_heads * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, dim) where t is the number of vertical tokens.
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        b, t, _ = q.shape
        h, e = self.heads, self.dim_heads

        def merge_heads(z):
            return z.reshape(b, t, h, e).transpose(1, 2).reshape(b * h, t, e)

        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum("bie,bje->bij", q, k) * (e ** -0.5)
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bij,bje->bie", attn, v)
        out = out.reshape(b, h, t, e).transpose(1, 2).reshape(b, t, h * e)
        return self.to_out(out)


# --------------------------------------------------------------------------- #
# 3D window attention with earth position bias.
# --------------------------------------------------------------------------- #


class EarthAttention3D(nn.Module):
    """3D window attention with earth position bias (shifted & non-shifted)."""

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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthSpecificBlock(nn.Module):
    """3D Swin transformer block with adaLN conditioning + cross-level attention."""

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
        axis_attn=False,
        roll_type=0,
        zdim=8,
        act_layer=nn.GELU,
        mlp_layer=Mlp,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 5) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.roll_type = roll_type

        self.norm1 = norm_layer(dim)
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
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.roll = roll_type > 0
        attn_mask = None  # ArchesWeather deliberately uses no shift attn-mask.

        if axis_attn:
            self.axis_pos = AxialPositionalEmbedding1D(dim=dim, zdim=zdim)
            self.axis_attn = CrossLevelAttention(dim=dim, heads=8)

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, dt=1):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
            x = x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        x = x.view(B, Pl, Lat, Lon, C)
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
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
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = shifted_x

        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, Pl * Lat * Lon, C)

        if hasattr(self, "axis_attn"):
            x2 = x.reshape(B, Pl, Lat * Lon, C).movedim(2, 1).flatten(0, 1)  # (B*Lat*Lon, Pl, C)
            x2 = self.axis_pos(x2)
            x2 = self.axis_attn(x2)
            x2 = x2.reshape(B, Lat * Lon, Pl, C).movedim(1, 2).flatten(1, 2)  # (B, Pl*Lat*Lon, C)

        if isinstance(dt, torch.Tensor):
            dt = dt[:, :, None]
        if c is None:
            x = shortcut + dt * self.drop_path(x)
            if hasattr(self, "axis_attn"):
                x = x + dt * self.drop_path(x2)
            x = x + dt * self.drop_path(self.mlp(self.norm2(x)))
        else:
            if hasattr(self, "axis_attn"):
                x = x + dt * self.drop_path(x2)
            x = shortcut + gate_msa[:, None, :] * self.drop_path(x)
            mlp_input = self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            x = x + self.drop_path(gate_mlp[:, None, :] * self.mlp(mlp_input))
        return x


class BasicLayer(nn.Module):
    """A stack of ``depth`` :class:`EarthSpecificBlock`s (alternating roll)."""

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
        zdim=8,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                EarthSpecificBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    roll_type=(i % 2),
                    zdim=zdim,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, *args, **kwargs):
        for blk in self.blocks:
            x = blk(x, *args, **kwargs)
        return x


class CondBasicLayer(BasicLayer):
    """:class:`BasicLayer` with one zero-init adaLN modulation per stage."""

    def __init__(self, *args, dim=192, cond_dim=32, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond_emb=None):
        c = self.adaLN_modulation(cond_emb)
        return super().forward(x, c)


class LinVert(nn.Module):
    """Column (vertical) mixing over the full Z stack — zdim-parametrized."""

    def __init__(self, in_features, zdim=8, drop=0.0, **kwargs):
        super().__init__()
        self.zdim = zdim
        self.fc1 = nn.Linear(zdim * in_features, zdim * in_features)

    def forward(self, x: torch.Tensor):
        shortcut = x
        z = self.zdim
        x2 = (
            shortcut.reshape((shortcut.shape[0], z, -1, shortcut.shape[-1]))
            .movedim(1, -2)
            .flatten(-2, -1)
        )  # B, lat*lon, z*C
        x2 = self.fc1(x2)
        x2 = (
            x2.reshape((x2.shape[0], -1, z, shortcut.shape[-1])).movedim(-2, 1).flatten(1, 2)
        )  # B, z*lat*lon, C
        return shortcut + x2
