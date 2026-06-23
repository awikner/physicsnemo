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

After Phase 6 Track A this module is a thin glue layer over upstream
:mod:`physicsnemo.nn`:

* ``Mask`` — the only class with no upstream analogue.
* ``DownSample`` / ``UpSample`` — factor=2 dispatch factories that delegate
  to :class:`physicsnemo.nn.DownSample3D` / :class:`physicsnemo.nn.UpSample3D`
  for the common path (and a local generalized impl for non-factor-2 cases).
* ``EarthSpecificLayer`` / ``EarthSpecificBlock`` — subclasses of upstream
  :class:`physicsnemo.nn.module.transformer_layers.FuserLayer` /
  :class:`physicsnemo.nn.module.transformer_layers.Transformer3DBlock` that
  pin the Pangu-Weather defaults (``cyclic_longitude=True``,
  ``use_sdpa=True``, ``mlp_layer=PanguMlp``).
* ``PanguMlp`` — local timm-style two-layer MLP (``fc1`` / ``fc2`` names)
  so PanguWeather ``.tar`` checkpoints translate cleanly.

The ``_vendored_physicsnemo_nn`` sub-package that previously held patched
copies of these blocks has been deleted — the patches (Issue #1599
cyclic-longitude mask fix, ``vertical_windowing`` kwarg, ``use_sdpa``
opt-in) all landed upstream as kwargs that default to historical behavior.
"""

import torch
from torch import nn

# Phase B: at downsample/upsample factor=2 the local implementation is
# bit-identical to physicsnemo.nn.{Down,Up}Sample3D (same modules, same forward).
# DownSample / UpSample below dispatch to these for the factor=2 path.
from physicsnemo.nn import DownSample3D as _UpstreamDownSample3D
from physicsnemo.nn import UpSample3D as _UpstreamUpSample3D


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



# Phase 6 Track A migration: source the Earth-Specific transformer blocks
# directly from upstream physicsnemo.nn (the three patches that previously
# lived in ``_vendored_physicsnemo_nn`` — Issue #1599 cyclic-longitude mask
# fix, ``vertical_windowing`` kwarg, ``use_sdpa`` opt-in — have landed
# upstream). The vendored sub-package is gone.
#
# Two callouts:
# 1. ``PanguMlp`` is a local timm-style two-layer MLP (``fc1`` / ``fc2``)
#    that PanguWeather checkpoints expect by name; upstream's
#    :class:`physicsnemo.nn.module.mlp_layers.Mlp` uses a ``layers.0`` /
#    ``layers.2`` sequential layout instead. We inject ``PanguMlp`` via
#    upstream's new ``mlp_layer=`` kwarg so the translated state-dict
#    keys still load cleanly.
# 2. ``EarthSpecificLayer`` / ``EarthSpecificBlock`` are thin subclasses
#    of upstream ``FuserLayer`` / ``Transformer3DBlock`` that pin
#    Pangu-Weather's defaults (``cyclic_longitude=True``, ``use_sdpa=True``,
#    ``mlp_layer=PanguMlp``). The model code keeps using these aliases —
#    upstream's defaults stay the historical non-Pangu behavior.
from physicsnemo.nn.module.attention_layers import EarthAttention3D
from physicsnemo.nn.module.transformer_layers import (
    FuserLayer as _UpstreamFuserLayer,
    Transformer3DBlock as _UpstreamTransformer3DBlock,
)
from physicsnemo.nn.module.utils.shift_window_mask import get_shift_window_mask


class PanguMlp(nn.Module):
    r"""Two-layer MLP with PanguWeather's ``fc1`` / ``fc2`` parameter names.

    Identical numerics to upstream :class:`physicsnemo.nn.module.mlp_layers.Mlp`
    (linear → activation → dropout → linear → dropout). The parameter
    names differ so PanguWeather ``.tar`` checkpoints translate cleanly.
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


# Back-compat alias for callers that imported ``Mlp`` from this module.
Mlp = PanguMlp


class EarthSpecificBlock(_UpstreamTransformer3DBlock):
    r"""Upstream :class:`Transformer3DBlock` with Pangu-Weather defaults pinned.

    Pins ``cyclic_longitude=True``, ``use_sdpa=True``, and
    ``mlp_layer=PanguMlp``. The model code constructs this class directly
    by name so callers don't need to pass the kwargs every time.
    """

    def __init__(self, *args, mlp_layer=None, **kwargs):
        kwargs.setdefault("cyclic_longitude", True)
        kwargs.setdefault("use_sdpa", True)
        super().__init__(
            *args,
            mlp_layer=mlp_layer if mlp_layer is not None else PanguMlp,
            **kwargs,
        )


class EarthSpecificLayer(_UpstreamFuserLayer):
    r"""Upstream :class:`FuserLayer` with Pangu-Weather defaults pinned.

    Same defaults pinning as :class:`EarthSpecificBlock` — ``cyclic_longitude``,
    ``use_sdpa``, and ``mlp_layer=PanguMlp``.
    """

    def __init__(self, *args, mlp_layer=None, **kwargs):
        kwargs.setdefault("cyclic_longitude", True)
        kwargs.setdefault("use_sdpa", True)
        super().__init__(
            *args,
            mlp_layer=mlp_layer if mlp_layer is not None else PanguMlp,
            **kwargs,
        )


# Back-compat aliases for callers that imported the upstream class names.
FuserLayer = EarthSpecificLayer
Transformer3DBlock = EarthSpecificBlock
