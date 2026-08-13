# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""``DiTAE`` — the x_DDC downscaler's denoiser (Phase 12h).

Vendored from amip_v2 ``modules/models/DiTAE.py`` @ ``e0b7b60``.

**This is the only x_DDC denoiser amip_v2 has.** Upstream deleted the
convolutional variant: ``modules/ae_module.py`` raises ``NotImplementedError``
for any ``decoder_type`` other than ``"dit"``, and ``modules/models/`` holds
only ``DiTAE`` and ``RollingDiT``. This fork's :class:`XDDCUNet` is the v1
denoiser and stays for the frozen v1 family; a v2-trained x_DDC checkpoint —
such as ``x_DDC_42_2026-08-07T09-34-49``, whose config says
``decoder_type: dit`` with ``in_channels 302`` / ``out_channels 151`` — needs
this class.

Architecturally it is :class:`AmipDiT` with the conditioning removed: patch
embed, 2D RoPE on the lat/lon patch grid, self-attention DiT blocks, unpatchify
head — but no gridded (``c_grid``) or scalar (``c_scalar``) conditioning and no
cross-attention, because the downscaler is only ever called as
``model(x_noised, cond, t)``. The conditioning it *does* need — the upsampled
low-resolution state — arrives concatenated on the channel axis, which is why
``in_channels`` is twice the physical state width (302 = 2 x 151).

Submodule names are deliberately identical to upstream's
(``patch_embed_main`` / ``t_embedder`` / ``sa_blocks`` / ``unpatchify_layer`` /
``rope_lat`` / ``rope_lon``), so a translated state dict maps key-for-key and
the translator has no renaming to do for this backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from .dit import DiTBlock
from .layers.patchify import PatchEmbed
from .layers.positional_encoding import RotaryEmbedding, TimestepEmbedder
from .layers.unpatchify import Unpatchify, sphere_pad

__all__ = ["DiTAE"]


class DiTAE(nn.Module):
    r"""Patchified diffusion-transformer denoiser for the x_DDC downscaler.

    Parameters
    ----------
    in_channels : int, optional, default=302
        Width of ``cat([x_noised, cond], dim=1)`` — twice the physical state
        width for the standard 151-channel AMIP contract.
    out_channels : int, optional, default=151
        Physical state width.
    dim, num_heads, num_blocks, patch_size, dropout
        Transformer geometry.
    nlat, nlon : int
        The **full-resolution** grid the downscaler emits (180x360 for AMIP).
    unpatch : str, optional, default="vanilla"
        Only ``"vanilla"`` exists upstream; anything else raises.
    """

    def __init__(
        self,
        in_channels: int = 302,
        out_channels: int = 151,
        dim: int = 1024,
        num_heads: int = 16,
        num_blocks: int = 20,
        patch_size: int = 4,
        nlat: int = 180,
        nlon: int = 360,
        dropout: float = 0.0,
        unpatch: str = "vanilla",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.nlat = nlat
        self.nlon = nlon
        self.dropout = dropout
        self.unpatch = unpatch

        # Pad the spatial dims up to a multiple of patch_size. 180/360 at p=4
        # divide exactly (pad_* == 0); the generic padding is kept so other
        # (nlat, nlon, patch_size) combinations also work.
        self.nlat_pad = math.ceil(nlat / patch_size) * patch_size
        self.nlon_pad = math.ceil(nlon / patch_size) * patch_size
        self.pad_lat = self.nlat_pad - nlat
        self.pad_lon = self.nlon_pad - nlon

        # Polar padding split top/bottom, circular padding split left/right.
        self.pad_lat_top = math.ceil(self.pad_lat / 2)
        self.pad_lat_bottom = self.pad_lat - self.pad_lat_top
        self.pad_lon_left = math.ceil(self.pad_lon / 2)
        self.pad_lon_right = self.pad_lon - self.pad_lon_left

        self.grid_x = self.nlat_pad // patch_size
        self.grid_y = self.nlon_pad // patch_size
        self.with_poles = False

        self.patch_embed_main = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            hidden_size=dim,
            flatten=False,
        )

        # 2D RoPE: one RotaryEmbedding per spatial axis, each over half the head dim.
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        self.t_embedder = TimestepEmbedder(dim)

        self.sa_blocks = nn.ModuleList(
            [
                DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        if unpatch != "vanilla":
            raise ValueError(
                f"unknown unpatch mode {unpatch!r} (only 'vanilla' is supported)"
            )
        self.unpatchify_layer = Unpatchify(
            grid_size=(self.grid_x, self.grid_y),
            patch_size=(patch_size, patch_size),
            in_dim=dim,
            out_dim=out_channels,
            cond_dim=dim,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Re-zero the AdaLN modulation outputs; ``apply`` above overwrote them.
        for block in self.sa_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    @torch.no_grad()
    def get_grid(self, nlat: int, nlon: int, device):
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)
        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        return lat, lon

    @torch.no_grad()
    def compute_rope_freqs(self, device):
        """2D RoPE cos/sin for the patch grid, cached after the first call.

        Uses physical lat/lon at patch centers, so the model encodes geographic
        position rather than integer indices.
        """
        if hasattr(self, "_rope_cos_lat") and self._rope_cos_lat.device == device:
            return (
                self._rope_cos_lat,
                self._rope_sin_lat,
                self._rope_cos_lon,
                self._rope_sin_lon,
            )

        lat, lon = self.get_grid(self.nlat_pad, self.nlon_pad, device)

        # Average to patch centers: [nlat_pad] -> [grid_x], [nlon_pad] -> [grid_y].
        lat_patches = lat.reshape(self.grid_x, self.patch_size).mean(dim=1)
        lon_patches = lon.reshape(self.grid_y, self.patch_size).mean(dim=1)

        lat_grid, lon_grid = torch.meshgrid(lat_patches, lon_patches, indexing="ij")
        lat_seq = lat_grid.reshape(-1)
        lon_seq = lon_grid.reshape(-1)

        freqs_lat = self.rope_lat(lat_seq.unsqueeze(0))  # [1, n, dim_head//2]
        freqs_lon = self.rope_lon(lon_seq.unsqueeze(0))

        self._rope_cos_lat = freqs_lat.cos()
        self._rope_sin_lat = freqs_lat.sin()
        self._rope_cos_lon = freqs_lon.cos()
        self._rope_sin_lon = freqs_lon.sin()

        return (
            self._rope_cos_lat,
            self._rope_sin_lat,
            self._rope_cos_lon,
            self._rope_sin_lon,
        )

    def forward(
        self, x_noised: torch.Tensor, cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""``(x_noised, cond, t) -> (b, out_channels, nlat, nlon)``.

        Parameters
        ----------
        x_noised : torch.Tensor
            ``(b, c, nlat, nlon)`` — the interpolant :math:`I_t`, channel-first.
        cond : torch.Tensor
            ``(b, c, nlat, nlon)`` — the conditioning (upsampled low-res state).
        t : torch.Tensor
            ``(b,)`` or ``(b, 1)`` diffusion time.
        """
        nlat, nlon = self.nlat, self.nlon

        x_input = torch.cat([x_noised, cond], dim=1)

        # Circular in longitude, polar in latitude.
        x_input = sphere_pad(
            x_input,
            padding=(
                self.pad_lon_left,
                self.pad_lon_right,
                self.pad_lat_top,
                self.pad_lat_bottom,
            ),
        )

        # PatchEmbed is channel-last.
        x_nhwc = x_input.permute(0, 2, 3, 1)
        x = self.patch_embed_main(x_nhwc)
        x = rearrange(x, "b ny nx c -> b (ny nx) c")

        rope = self.compute_rope_freqs(x.device)

        if len(t.shape) == 1:
            t = t[:, None]
        t_emb = self.t_embedder(t)

        for block in self.sa_blocks:
            x = block(x, t_emb, *rope)

        # The head emits channel-last; permute back and crop the padding.
        x = self.unpatchify_layer(x, t_emb)
        x = x.permute(0, 3, 1, 2)
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = x[
                :,
                :,
                self.pad_lat_top : self.pad_lat_top + nlat,
                self.pad_lon_left : self.pad_lon_left + nlon,
            ]
        return x
