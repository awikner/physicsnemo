# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/models/Unet.py, class ``UNet``) for Phase 8f (F6). Renamed
# ``XDDCUNet`` here to disambiguate from a bare ``UNet`` name and to
# pair with :class:`~.wrappers.XDDCWrapper`. The two building blocks
# (``ResBlock`` / ``AttentionBlock``) are already vendored in
# :mod:`._unet_blocks` (used by :class:`~.erdm_unet.ERDM` too) — this
# module only adds the encoder/decoder/bottleneck assembly.
#
# Only the ``decoder_type: unet`` denoiser is vendored. Upstream's
# ``x_DDC`` also supports ``decoder_type: dit`` (``modules/models/DiTAE.py``,
# a DiT-style autoencoder) — deferred: neither of the two real Midway3
# x_DDC checkpoints in scope for Phase 8f use it (their configs carry
# only a ``decoder:`` UNet block, no ``dit:`` block, and
# ``decoder_type`` defaults to ``"unet"`` upstream when the key is
# absent). Vendor DiTAE if/when a ``decoder_type: dit`` checkpoint
# needs translating.

r"""x_DDC super-resolution cascade — denoiser backbone.

x_DDC ("data-dependent-coupling" diffusion) restores full-resolution
detail from a blurry low-res field. Training (the upstream
``AutoencoderModule``) degrades real full-res truth via
bilinear-downsample-then-upsample (:class:`~.layers.bilinear.BilinearEncoder`
+ :class:`~.layers.bilinear.BilinearDecoder`) and asks the denoiser to
restore the original signal; at inference (:class:`~.wrappers.CombinedModule`)
the low-res field instead comes from a real forecaster's prediction,
bilinear-upsampled to full resolution.

Unlike :class:`~.dit.AmipDiT` / :class:`~.rolling_dit.RollingDiT` /
:class:`~.erdm_unet.ERDM`, this backbone has **no** ``c_grid`` /
``c_scalar`` conditioning path and **no** window/temporal axis — it's a
plain per-frame 2D denoising UNet conditioned only on the diffusion
timestep ``t`` and the low-res field itself (concatenated channel-wise
with the noised input, exactly like upstream's ``torch.cat([x_noised,
cond], dim=1)``).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module as _PNeMoModule

from ._unet_blocks import AttentionBlock, ResBlock
from .layers.conv import DCDownsample, DCUpsample, SphereConv2d, nonlinearity
from .layers.positional_encoding import TimestepEmbedder


@dataclass
class MetaData(ModelMetaData):
    """Phase 8f default ModelMetaData for :class:`XDDCUNet`."""

    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_gpu: bool = False
    bf16: bool = False
    onnx: bool = False


class XDDCUNet(_PNeMoModule):
    r"""Denoising UNet for the x_DDC super-resolution cascade.

    Parameters
    ----------
    in_channels : int
        Channels of ``concat(x_noised, cond)`` (twice the state channel
        count — matches upstream's ``in_channels: 302 # 86*2`` convention).
    out_channels : int
        Predicted (high-res) state channel count.
    model_channels : int, optional, default=256
        Base channel width.
    channel_mult : tuple of int, optional, default=(1, 2, 4)
        Per-level channel multipliers.
    num_res_blocks : int, optional, default=3
        Residual blocks per level (encoder); decoder gets one extra per
        level for the skip-connection merge.
    attn_levels : tuple of int, optional, default=()
        Which encoder/decoder levels (0-indexed) get self-attention.
    num_heads : int, optional, default=8
        Attention heads.
    dropout : float, optional, default=0.0
        Dropout rate inside each :class:`~._unet_blocks.ResBlock`.
    t_emb_dim : int, optional, default=256
        Timestep embedding dimension.
    num_groups : int, optional, default=16
        Group count for every :class:`~torch.nn.GroupNorm`.

    Forward
    -------
    x_noised : torch.Tensor
        Tensor of shape :math:`(B, C, H, W)` — noised interpolant.
    cond : torch.Tensor
        Tensor of shape :math:`(B, C, H, W)` — low-res conditioning
        field (bilinear-upsampled to full resolution upstream of this
        call).
    t : torch.Tensor
        Tensor of shape :math:`(B, 1)` — diffusion timestep in
        :math:`[0, 1]`.

    Outputs
    -------
    torch.Tensor
        Tensor of shape :math:`(B, C_{out}, H, W)` — predicted
        high-res target.

    Examples
    --------
    >>> model = XDDCUNet(
    ...     in_channels=8, out_channels=4, model_channels=16,
    ...     channel_mult=(1, 2), num_res_blocks=1, num_groups=4,
    ... )
    >>> x_noised = torch.randn(2, 4, 16, 32)
    >>> cond = torch.randn(2, 4, 16, 32)
    >>> t = torch.rand(2, 1)
    >>> out = model(x_noised, cond, t)
    >>> out.shape
    torch.Size([2, 4, 16, 32])
    """

    __model_checkpoint_version__ = "1.0"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 256,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 3,
        attn_levels: tuple = (),
        num_heads: int = 8,
        dropout: float = 0.0,
        t_emb_dim: int = 256,
        num_groups: int = 16,
    ):
        super().__init__(meta=MetaData())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_mult = tuple(channel_mult)
        self.num_levels = len(self.channel_mult)

        # Timestep embedding.
        self.t_embedder = TimestepEmbedder(t_emb_dim)

        # Input projection.
        self.input_conv = SphereConv2d(
            in_channels, model_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # ── Encoder ──
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = model_channels
        enc_channels = [ch]  # track channels for skip connections

        for level in range(self.num_levels):
            out_ch = model_channels * self.channel_mult[level]
            use_attn = level in attn_levels

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                block_in = ch if i == 0 else out_ch
                level_blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout, num_groups))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
                ch = out_ch
                enc_channels.append(ch)

            self.enc_blocks.append(level_blocks)

            if level < self.num_levels - 1:
                self.downsamples.append(DCDownsample(ch, ch))
                enc_channels.append(ch)
            else:
                self.downsamples.append(None)

        # ── Bottleneck ──
        self.mid_block1 = ResBlock(ch, ch, t_emb_dim, dropout, num_groups)
        self.mid_attn = AttentionBlock(ch, num_heads, num_groups)
        self.mid_block2 = ResBlock(ch, ch, t_emb_dim, dropout, num_groups)

        # ── Decoder ──
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * self.channel_mult[level]
            use_attn = level in attn_levels

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = enc_channels.pop()
                block_in = ch + skip_ch
                level_blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout, num_groups))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
                ch = out_ch

            self.dec_blocks.append(level_blocks)

            if level > 0:
                self.upsamples.append(DCUpsample(ch, ch))
            else:
                self.upsamples.append(None)

        # Output.
        self.out_norm = nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-6)
        self.out_conv = SphereConv2d(ch, out_channels, kernel_size=(3, 3), padding=(1, 1))
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.initialize_weights()

    def initialize_weights(self):
        """Xavier-uniform for linear/conv layers, standard init for norms.

        Preserves the zero-inits already applied to output-critical
        layers.
        """
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, SphereConv2d)):
                if name.endswith(("out_conv", "conv2", "proj")):
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x_noised: torch.Tensor, cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t)  # (B, t_emb_dim)

        x = torch.cat([x_noised, cond], dim=1)
        x = self.input_conv(x)

        # ── Encoder ──
        skips = [x]
        for level in range(self.num_levels):
            for block in self.enc_blocks[level]:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                    skips.append(x)
                else:
                    x = block(x)

            if self.downsamples[level] is not None:
                x = self.downsamples[level](x)
                skips.append(x)

        # ── Bottleneck ──
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # ── Decoder ──
        for level_idx, _level in enumerate(reversed(range(self.num_levels))):
            for block in self.dec_blocks[level_idx]:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x, t_emb)
                else:
                    x = block(x)

            if self.upsamples[level_idx] is not None:
                x = self.upsamples[level_idx](x)

        # Output.
        x = self.out_norm(x)
        x = nonlinearity(x)
        x = self.out_conv(x)

        return x


__all__ = ["XDDCUNet"]
