# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from the ArchesWeather deterministic backbone in INRIA/geoarches
# (BSD-3-Clause, Copyright (c) 2024-2025 ARCHES team @ INRIA), paper
# "ArchesWeather & ArchesWeatherGen" (arXiv:2412.12971). The model is adapted
# to the ai-rossby PLASIM/ERA5 variable-routing convention so the same trainer
# (examples/weather/ai_rossby/train.py) handles SFNO and ArchesWeather without
# touching train_step, and to our 17-level / 1-degree ERA5 grid via a
# generalized vertical token count (zdim). See docs/dev/context/.

r"""ArchesWeather deterministic 3D Swin U-Net weather model (physicsnemo port).

The model is a 3D Swin U-Net with per-block cross-level ("axial") attention and
adaLN month/hour conditioning. It consumes the *current* state and the state
*24 h earlier* (channel-concatenated), and predicts the next state one 24 h step
ahead. Unlike SFNO/Pangu it has **no** SST/SIC/TISR boundary forcing — the
``varying_boundary`` tensor is accepted for signature compatibility and ignored;
seasonality enters only through the month/hour adaLN condition.

Forward contract matches :class:`physicsnemo.experimental.models.sfno_plasim.SfnoPlasim`
so the trainer can swap models without changing ``train_step`` — the extra
``surface_prev_in`` / ``upper_air_prev_in`` / ``calendar`` kwargs are supplied by
the datapipe (``prev_state_steps`` + ``calendar_encoding='month_hour'``) and
passed through the optional-kwarg helper in ``train_loop`` / ``validate`` /
``inference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

from .archesweather_layers import (
    CondBasicLayer,
    DownSample,
    ICNR_init,
    LinVert,
    Mlp,
    TimestepEmbedder,
    UpSample,
)

try:  # SwiGLU MLP (timm) — matches the ArchesWeather-M flagship recipe.
    from timm.layers.mlp import SwiGLU
except Exception:  # pragma: no cover — older timm layouts
    from timm.models.layers import SwiGLU


@dataclass
class MetaData(ModelMetaData):
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    amp_gpu: bool = True
    bf16: bool = True
    onnx: bool = False


class WeatherEncodeDecodeLayer(nn.Module):
    """Patch-embed encoder + PixelShuffle/ICNR decoder for the 3D backbone.

    Differs from the geoarches original in two ways: (1) the constant boundary
    masks are a *forward input* (standardized here with per-channel buffers)
    rather than a baked ``.pt`` file, matching the physicsnemo variable-routing
    convention; (2) the vertical token count follows from ``patch_size`` +
    ``img_size`` (17 levels, patch depth 2 -> pad to 18 -> 9 level tokens; plus
    the surface token -> zdim = 10).
    """

    def __init__(
        self,
        img_size=(17, 180, 360),
        emb_dim=192,
        out_emb_dim=2 * 192,  # skip-concat doubles the decoder input dim
        patch_size=(2, 3, 3),
        surface_ch=4,
        level_ch=5,
        n_concatenated_states=1,  # the t-24h previous state
        constant_dims=2,  # land_sea_mask + geopotential_at_surface
        diag_ch=0,  # diagnostic head channels (precip, OLR); 0 = no head
    ) -> None:
        super().__init__()
        self.img_size = tuple(img_size)
        self.emb_dim = emb_dim
        self.out_emb_dim = out_emb_dim
        self.patch_size = tuple(patch_size)
        self.surface_ch = surface_ch
        self.level_ch = level_ch
        self.n_concatenated_states = n_concatenated_states
        self.constant_dims = constant_dims

        # Per-channel standardization of the constant boundary masks. Filled
        # from config (compute_const_stats) or left as identity for smoke tests.
        self.register_buffer("const_mean", torch.zeros(constant_dims))
        self.register_buffer("const_std", torch.ones(constant_dims))

        surface_ch_in = constant_dims + surface_ch + n_concatenated_states * surface_ch
        level_ch_in = level_ch + n_concatenated_states * level_ch

        self.level_proj = nn.Conv3d(
            level_ch_in, emb_dim, kernel_size=patch_size, stride=patch_size
        )
        self.surface_proj = nn.Conv2d(
            surface_ch_in, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )

        l_pad = patch_size[0] - img_size[0] % patch_size[0]
        if l_pad == patch_size[0]:
            l_pad = 0  # already divisible
        level_pads = [l_pad // 2, l_pad - l_pad // 2]
        self.level_padder = nn.ZeroPad3d((0, 0, 0, 0, *level_pads))
        self._odd_lat = img_size[1] % 2 == 1

        # Decoder: PixelShuffle upscale of patch_size[-1] with ICNR init.
        u = patch_size[-1]
        self.surface_deconv = nn.Conv2d(
            out_emb_dim, surface_ch * u**2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.level_deconv = nn.Conv2d(
            out_emb_dim // patch_size[0],
            level_ch * u**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixelshuffle = nn.PixelShuffle(u)
        ICNR_init(self.surface_deconv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=u)
        ICNR_init(self.level_deconv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=u)

        # Optional diagnostic head (e.g. 24h precip + OLR): a parallel decoder
        # branch off the same surface-token plane. Added 2026-07-27; absent from
        # geoarches ArchesWeather-M, so old checkpoints load with strict=False
        # (warm start) leaving only this branch randomly initialized.
        self.diag_ch = int(diag_ch)
        if self.diag_ch > 0:
            self.diag_deconv = nn.Conv2d(
                out_emb_dim, diag_ch * u**2, kernel_size=3, stride=1, padding=1, bias=False
            )
            ICNR_init(self.diag_deconv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=u)

    def _standardize_constant(self, constant: torch.Tensor) -> torch.Tensor:
        mean = self.const_mean.to(constant.device)[:, None, None]
        std = self.const_std.to(constant.device)[:, None, None]
        return (constant - mean) / std

    def encode(
        self,
        surface: torch.Tensor,  # (B, surface_ch, H, W)
        level: torch.Tensor,  # (B, level_ch, L, H, W)
        constant: torch.Tensor,  # (constant_dims, H, W) or (B, constant_dims, H, W)
        prev_surface: Optional[torch.Tensor] = None,
        prev_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = surface.shape[0]
        if constant.ndim == 3:
            constant = constant.unsqueeze(0).expand(bs, -1, -1, -1)
        constant = self._standardize_constant(constant)

        if self._odd_lat:
            surface = surface[..., :-1, :]
            level = level[..., :-1, :]
            constant = constant[..., :-1, :]

        surface = torch.cat([surface, constant], dim=1)

        if self.n_concatenated_states and prev_surface is not None:
            if self._odd_lat:
                prev_surface = prev_surface[..., :-1, :]
                prev_level = prev_level[..., :-1, :]
            surface = torch.cat([surface, prev_surface], dim=1)
            level = torch.cat([level, prev_level], dim=1)

        surface = self.surface_proj(surface)
        level = self.level_proj(self.level_padder(level))

        x = torch.cat([surface.unsqueeze(2), level], dim=2)  # (B, emb, zdim, lat', lon')
        return x

    def decode(self, x: torch.Tensor):
        p0 = self.patch_size[0]
        surface, level = x[:, :, 0], x[:, :, 1:]

        output_surface = self.surface_deconv(surface)
        output_surface = self.pixelshuffle(output_surface)
        output_surface = output_surface.unsqueeze(-3)

        output_diag = None
        if self.diag_ch > 0:
            output_diag = self.pixelshuffle(self.diag_deconv(surface)).unsqueeze(-3)

        # Split each level token back into p0 vertical planes, then drop the
        # zero-pad plane(s) at the front (faithful to the geoarches convention).
        n_tokens = level.shape[2]
        level = level.reshape(level.shape[0], level.shape[1] // p0, p0, *level.shape[2:]).flatten(
            2, 3
        )
        n_drop = p0 * n_tokens - self.img_size[0]
        level = level[:, :, n_drop:]
        level = level.movedim(-3, 1).flatten(0, 1)

        output_level = self.level_deconv(level)
        output_level = self.pixelshuffle(output_level)
        output_level = output_level.reshape(
            -1, self.img_size[0], *output_level.shape[1:]
        ).movedim(1, -3)

        if self._odd_lat:
            # Put back the removed south-pole row (nearest-neighbour).
            output_surface = torch.cat([output_surface, output_surface[..., -1:, :]], dim=-2)
            output_level = torch.cat([output_level, output_level[..., -1:, :]], dim=-2)
            if output_diag is not None:
                output_diag = torch.cat([output_diag, output_diag[..., -1:, :]], dim=-2)

        output_surface = output_surface.squeeze(-3)  # (B, surface_ch, H, W)
        if output_diag is not None:
            output_diag = output_diag.squeeze(-3)  # (B, diag_ch, H, W)
        return output_surface, output_level, output_diag


class ArchesWeatherCondBackbone(nn.Module):
    """3D Swin U-Net backbone with adaLN conditioning (zdim-parametrized)."""

    def __init__(
        self,
        tensor_size=(10, 60, 120),
        emb_dim=192,
        cond_dim=256,
        num_heads=(6, 12, 12, 6),
        window_size=(1, 6, 10),
        droppath_coeff=0.2,
        depth_multiplier=2,
        dropout=0.0,
        mlp_ratio=4.0,
        use_skip=True,
        first_interaction_layer="linear",
        gradient_checkpointing=False,
        mlp_layer="swiglu",
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_skip = use_skip
        self.first_interaction_layer = first_interaction_layer
        self.gradient_checkpointing = gradient_checkpointing
        drop_path = np.linspace(0, droppath_coeff / depth_multiplier, 8 * depth_multiplier).tolist()

        self.zdim = tensor_size[0]
        self.layer1_shape = tensor_size[1:]
        self.layer2_shape = (self.layer1_shape[0] // 2, self.layer1_shape[1] // 2)

        if first_interaction_layer == "linear":
            self.interaction_layer = LinVert(in_features=emb_dim, zdim=self.zdim)

        layer_args = dict(
            cond_dim=cond_dim,
            window_size=window_size,
            act_layer=nn.GELU,
            drop=dropout,
            mlp_layer=Mlp,
            mlp_ratio=mlp_ratio,
            zdim=self.zdim,
            axis_attn=True,  # cross-level attention in every block
        )
        if mlp_layer == "swiglu":
            layer_args["mlp_ratio"] = mlp_ratio * 2 / 3
            layer_args["mlp_layer"] = SwiGLU

        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[0],
            drop_path=drop_path[: 2 * depth_multiplier],
            **layer_args,
            **kwargs,
        )
        self.downsample = DownSample(
            in_dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            output_resolution=(self.zdim, *self.layer2_shape),
        )
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[1],
            drop_path=drop_path[2 * depth_multiplier :],
            **layer_args,
            **kwargs,
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[2],
            drop_path=drop_path[2 * depth_multiplier :],
            **layer_args,
            **kwargs,
        )
        self.upsample = UpSample(
            emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape)
        )
        out_dim = emb_dim if not self.use_skip else 2 * emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[3],
            drop_path=drop_path[: 2 * depth_multiplier],
            **layer_args,
            **kwargs,
        )

    def forward(self, x, cond_emb, **kwargs):
        import torch.utils.checkpoint as gradient_checkpoint

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        if self.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, cond_emb)
        skip = x
        x = self.downsample(x)
        x = self.layer2(x, cond_emb)
        if self.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(self.layer3, x, cond_emb, use_reentrant=False)
        else:
            x = self.layer3(x, cond_emb)
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb)

        output = x.transpose(1, 2).reshape(x.shape[0], -1, self.zdim, *self.layer1_shape)
        return output


class ArchesWeather(Module):
    r"""ArchesWeather-M deterministic weather model (physicsnemo Module).

    Constructor takes the same variable-group + geometry kwargs as
    :class:`SfnoPlasim` plus ArchesWeather architecture kwargs, so the ai-rossby
    trainer builds it via ``Module.instantiate`` with no registry changes.

    Parameters
    ----------
    surface_variables, upper_air_variables, constant_boundary_variables,
    varying_boundary_variables, diagnostic_variables : list of str (see below)
        Variable-group channel names (same convention as :class:`SfnoPlasim`).
        A non-empty ``diagnostic_variables`` (e.g. 24h precip + OLR) attaches a
        diagnostic decoder head — an ai-rossby extension over geoarches
        ArchesWeather-M; empty keeps the original head-less architecture.
        ``varying_boundary_variables`` is accepted for a uniform config but the
        boundary tensor is ignored at forward time.
    levels : list of float
        Vertical pressure levels (17 for our ERA5 recipe).
    horizontal_resolution : list of int
        ``[lat, lon]`` grid shape.
    emb_dim : int, default 192
    depth_multiplier : int, default 2
    num_heads : sequence of int, default (6, 12, 12, 6)
    window_size : sequence of int, default (1, 6, 10)
    droppath_coeff : float, default 0.2
    cond_dim : int, default 256
    patch_size : sequence of int, default (2, 3, 3)
    mlp_layer : str, default "swiglu"
    use_prev_state : bool, default True
    const_mean, const_std : sequence of float, optional
        Per-channel standardization stats for the constant boundary masks. When
        omitted the masks are passed through unscaled (identity) — fine for
        shape/smoke tests but set them from the store for training.

    Forward
    -------
    ``forward(surface_in, constant_boundary, varying_boundary, upper_air_in,
    *, surface_prev_in=None, upper_air_prev_in=None, calendar=None, ...)`` ->
    ``(out_surface, out_upper_air, out_diagnostic, 0, 0, 0)`` —
    ``out_diagnostic`` is a scalar 0 when the model has no diagnostic head.
    ``varying_boundary`` and the ``target_*`` / ``train`` kwargs are accepted
    and ignored.
    """

    def __init__(
        self,
        *,
        surface_variables: list,
        upper_air_variables: list,
        constant_boundary_variables: list,
        varying_boundary_variables: list = (),
        levels: list,
        horizontal_resolution: list,
        diagnostic_variables: list = (),
        emb_dim: int = 192,
        depth_multiplier: int = 2,
        num_heads: Sequence[int] = (6, 12, 12, 6),
        window_size: Sequence[int] = (1, 6, 10),
        droppath_coeff: float = 0.2,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        cond_dim: int = 256,
        patch_size: Sequence[int] = (2, 3, 3),
        mlp_layer: str = "swiglu",
        use_prev_state: bool = True,
        add_input_state: bool = False,
        gradient_checkpointing: bool = False,
        const_mean: Optional[Sequence[float]] = None,
        const_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(meta=MetaData())

        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        # Diagnostic head (precip + OLR) — an ai-rossby extension over
        # geoarches ArchesWeather-M (which has none). Empty list = no head,
        # bit-identical to the original architecture.
        self.diagnostic_variables = list(diagnostic_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.has_diagnostic = len(self.diagnostic_variables) > 0
        self.use_prev_state = bool(use_prev_state)
        self.add_input_state = bool(add_input_state)

        n_surface = len(self.surface_variables)
        n_level = len(self.upper_air_variables)
        n_const = len(self.constant_boundary_variables)
        n_levels = len(self.levels)
        lat, lon = int(horizontal_resolution[0]), int(horizontal_resolution[1])

        self.embedder = WeatherEncodeDecodeLayer(
            img_size=(n_levels, lat, lon),
            emb_dim=emb_dim,
            out_emb_dim=2 * emb_dim,
            patch_size=tuple(patch_size),
            surface_ch=n_surface,
            level_ch=n_level,
            n_concatenated_states=1 if self.use_prev_state else 0,
            constant_dims=n_const,
            diag_ch=len(self.diagnostic_variables),
        )
        if const_mean is not None:
            self.embedder.const_mean.copy_(torch.tensor(const_mean, dtype=torch.float32))
        if const_std is not None:
            self.embedder.const_std.copy_(torch.tensor(const_std, dtype=torch.float32))

        # Token grid: surface token (z=0) + padded-level tokens.
        padded_levels = n_levels + (
            (patch_size[0] - n_levels % patch_size[0]) % patch_size[0]
        )
        zdim = 1 + padded_levels // patch_size[0]
        tokens_lat = lat // patch_size[1] - (1 if lat % 2 else 0) * 0  # even grid: exact
        if lat % 2 == 1:
            tokens_lat = (lat - 1) // patch_size[1]
        tokens_lon = lon // patch_size[2]

        self.backbone = ArchesWeatherCondBackbone(
            tensor_size=(zdim, tokens_lat, tokens_lon),
            emb_dim=emb_dim,
            cond_dim=cond_dim,
            num_heads=tuple(num_heads),
            window_size=tuple(window_size),
            droppath_coeff=droppath_coeff,
            depth_multiplier=depth_multiplier,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            gradient_checkpointing=gradient_checkpointing,
        )

        # adaLN conditioning embedders (month 1-12, hour 0-23).
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)

    @property
    def upper_air_variable_names(self) -> list:
        return list(self.upper_air_variables)

    def _cond_emb(self, calendar: Optional[torch.Tensor], batch_size: int, device) -> torch.Tensor:
        if calendar is not None:
            month = calendar[:, 0].to(device).float()
            hour = calendar[:, 1].to(device).float()
        else:
            month = torch.ones(batch_size, device=device)
            hour = torch.zeros(batch_size, device=device)
        return self.month_embedder(month) + self.hour_embedder(hour)

    def forward(
        self,
        surface_in: torch.Tensor,
        constant_boundary: torch.Tensor,
        varying_boundary: Optional[torch.Tensor] = None,
        upper_air_in: torch.Tensor = None,
        target_surface: Optional[torch.Tensor] = None,
        target_upper_air: Optional[torch.Tensor] = None,
        train: bool = False,
        *,
        surface_prev_in: Optional[torch.Tensor] = None,
        upper_air_prev_in: Optional[torch.Tensor] = None,
        calendar: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ):
        del target_surface, target_upper_air, train, varying_boundary, return_latent

        bs = surface_in.shape[0]
        if self.use_prev_state and surface_prev_in is None:
            # No previous frame available (e.g. first rollout step at archive
            # start) — fall back to persistence (prev = current).
            surface_prev_in = surface_in
            upper_air_prev_in = upper_air_in

        x = self.embedder.encode(
            surface_in,
            upper_air_in,
            constant_boundary,
            prev_surface=surface_prev_in,
            prev_level=upper_air_prev_in,
        )
        cond_emb = self._cond_emb(calendar, bs, surface_in.device)
        x = self.backbone(x, cond_emb)
        out_surface, out_upper_air, out_diag = self.embedder.decode(x)

        if self.add_input_state:
            # Residual applies to the prognostic state only; the diagnostic
            # head is a direct prediction (no input diagnostic state exists).
            out_surface = out_surface + surface_in
            out_upper_air = out_upper_air + upper_air_in

        zero = torch.tensor(0.0, device=surface_in.device, dtype=surface_in.dtype)
        if out_diag is None:
            out_diag = zero
        return (out_surface, out_upper_air, out_diag, zero, zero, zero)
