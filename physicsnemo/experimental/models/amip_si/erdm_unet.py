# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/models/ERDM.py) for Phase 8a. The two helpers ``ResBlock``
# and ``AttentionBlock`` (originally in upstream ``modules.models.Unet``)
# are vendored in :mod:`._unet_blocks` — only the two helpers are
# imported, not the full UNet class (deferred to Phase 8f).

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataclasses import dataclass

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module as _PNeMoModule

from ._unet_blocks import AttentionBlock, ResBlock
from .layers.conv import DCDownsample, DCUpsample, SphereConv2d, nonlinearity
from .layers.embedding import CalendarEmbedding
from .layers.positional_encoding import TimestepEmbedder


@dataclass
class MetaData(ModelMetaData):
    """Phase 8a default ModelMetaData for :class:`ERDM` (UNet backbone)."""

    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_gpu: bool = False
    bf16: bool = False
    onnx: bool = False

# ---------------------------------------------------------------------------
# ERDM backbone (Elucidated Rolling Diffusion Model, https://arxiv.org/abs/2506.20024)
#
# A 2D ADM-style U-Net augmented with explicit temporal processing, following the
# latent-video-diffusion strategy described in the paper:
#   - Spatial conv / spatial-attention blocks act PER FRAME: the window dimension W
#     is folded into the batch (b*W, c, h, w), so convolutions are purely spatial.
#   - After each U-Net block we apply CAUSAL temporal attention over the W frames:
#     the spatial dimensions are folded into the batch (b*h*w, W, c) so attention is
#     purely temporal. Causal masking lets each frame attend only to itself and the
#     earlier (less-noised / nearer-term) frames in the window.
#   - The per-frame noise level (ln(sigma)/4) conditions both the spatial blocks
#     (AdaGN inside ResBlock) and the temporal blocks (adaLN), per the paper.
#   - The state is bilinearly interpolated from the native grid (nlat, nlon) up to a
#     working grid (nlat_work, nlon_work) divisible by size_mult, run through the U-Net
#     with no zero padding, then interpolated back down to the native grid.
#   - Grid-aligned forcings (c_grid) and the calendar (c_scalar) condition the model
#     SPATIALLY: forcings are embedded with a strided conv (interpolated full-res ->
#     working grid) and the calendar is expanded to the working grid via
#     CalendarEmbedding, then both are concatenated onto x_noised along the channel dim.
#
# Forward contract (called by modules/diffusion/erdm.py::ERDMScheduler):
#     F = model(x_noised, c_noise, c_grid, c_scalar)
#       x_noised : (b, W, C, H, W)        preconditioned noised window (c_in * x_bar)
#       c_noise  : (b, W)                 per-frame noise label ln(sigma)/4
#       c_grid   : (b, W, c_grid, Hf, Wf) per-frame forcings (or None)
#       c_scalar : (b, W, scalar_dim)     per-frame calendar (or None)
#       returns  : (b, W, C, H, W)        raw network output F_theta
#
# No clean conditioning frame is fed in: at the first window the nearly-clean
# front frames of the partially noised rolling window provide the conditioning.
# ---------------------------------------------------------------------------


class CausalTemporalAttention(nn.Module):
    """Causal self-attention over the window (temporal) axis.

    Spatial dims are folded into the batch so attention runs purely over the W
    frames. adaLN modulation injects the per-frame noise embedding (analogous to
    the AdaGN used in the spatial U-Net blocks).
    """

    def __init__(self, channels, t_emb_dim, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, 2 * channels))
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

        # Start as identity: zero output projection and zero adaLN modulation.
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, t_emb, b, W):
        """x: (b*W, c, h, w); t_emb: (b*W, t_emb_dim). Returns (b*W, c, h, w)."""
        _, c, h, w = x.shape
        nh = self.num_heads

        # Tokens over time: (b, h*w, W, c)
        seq = rearrange(x, '(b t) c h w -> b (h w) t c', b=b, t=W)
        hn = self.norm(seq)

        # Per-(frame) adaLN modulation, broadcast over the spatial token axis.
        scale, shift = self.adaln(t_emb).chunk(2, dim=-1)          # (b*W, c)
        scale = rearrange(scale, '(b t) c -> b 1 t c', b=b, t=W)
        shift = rearrange(shift, '(b t) c -> b 1 t c', b=b, t=W)
        hn = hn * (1 + scale) + shift

        qkv = self.qkv(hn)                                          # (b, hw, W, 3c)
        q, k, v = rearrange(
            qkv, 'b s t (three nh hd) -> three (b s) nh t hd', three=3, nh=nh
        ).unbind(0)                                                 # each (b*hw, nh, W, hd)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (b*hw, nh, W, hd)
        out = rearrange(out, '(b s) nh t hd -> b s t (nh hd)', b=b)
        out = self.proj(out)                                       # (b, hw, W, c)
        out = rearrange(out, 'b (h w) t c -> (b t) c h w', h=h, w=w)

        return x + out


class ERDM(_PNeMoModule):
    def __init__(
        self,
        in_channels,                  # per-frame state channels C
        out_channels=None,            # defaults to in_channels
        model_channels=128,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_levels=(2,),             # levels with spatial self-attention
        num_heads=8,
        temporal_num_heads=8,
        dropout=0.0,
        t_emb_dim=256,
        num_groups=16,
        nlat=45,                      # native latent grid the model receives / returns
        nlon=90,
        nlat_work=48,                 # working grid (divisible by size_mult) the U-Net runs on
        nlon_work=96,
        c_grid_dim=0,                 # number of forcing channels (0 disables)
        c_grid_embed_dim=4,           # forcing conv-embedding channels
        c_grid_downsample=4,          # strided-conv factor: full-res c_grid -> working grid
        scalar_dim=0,                 # calendar dim (0 disables)
        c_scalar_embed_dim=4,         # calendar grid-embedding channels
        **kwargs,                     # tolerate extra config keys
    ):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.channel_mult = channel_mult
        self.num_levels = len(channel_mult)
        # Spatial dims must be a multiple of this for clean factor-2 down/up-sampling.
        self.size_mult = 2 ** (self.num_levels - 1)
        self.nlat = nlat
        self.nlon = nlon
        # The state is bilinearly interpolated up to (nlat_work, nlon_work) so all
        # down/up-sampling is exact, then interpolated back down at the end. This
        # avoids zero padding and keeps everything on the data grid.
        self.nlat_work = nlat_work
        self.nlon_work = nlon_work
        assert nlat_work % self.size_mult == 0 and nlon_work % self.size_mult == 0, \
            f"working grid ({nlat_work}, {nlon_work}) must be divisible by size_mult={self.size_mult}"
        self.c_grid_dim = c_grid_dim
        self.c_grid_embed_dim = c_grid_embed_dim
        self.c_grid_downsample = c_grid_downsample
        self.scalar_dim = scalar_dim
        self.c_scalar_embed_dim = c_scalar_embed_dim

        # Per-frame noise embedding.
        self.t_embedder = TimestepEmbedder(t_emb_dim)

        # Per-frame input: [x_noised ; conv-embedded c_grid ; calendar grid].
        first_in = in_channels

        # Grid forcings: strided conv from (nlat_work*ds, nlon_work*ds) down to the
        # working grid (as in DiT), replacing bilinear interpolation of the forcings.
        if c_grid_dim > 0:
            self.c_grid_embed = nn.Conv2d(c_grid_dim, c_grid_embed_dim,
                                          kernel_size=c_grid_downsample,
                                          stride=c_grid_downsample)
            first_in += c_grid_embed_dim
        else:
            self.c_grid_embed = None

        # Calendar: expand scalars to the working grid and concat onto x_noised.
        if scalar_dim > 0:
            self.scalar_embedder = CalendarEmbedding(nlon=nlon_work, nlat=nlat_work,
                                                     embed_channels=c_scalar_embed_dim,
                                                     use_co2=(scalar_dim >= 3))
            first_in += c_scalar_embed_dim
        else:
            self.scalar_embedder = None

        self.input_conv = SphereConv2d(first_in, model_channels, kernel_size=(3, 3), padding=(1, 1))

        def temporal(ch):
            return CausalTemporalAttention(ch, t_emb_dim, temporal_num_heads)

        # ── Encoder ──
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        enc_channels = [ch]
        for level in range(self.num_levels):
            out_ch = model_channels * channel_mult[level]
            use_attn = level in attn_levels
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                block_in = ch if i == 0 else out_ch
                level_blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout, num_groups))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
                    print("using spatial attn at level", level)
                level_blocks.append(temporal(out_ch))  # causal temporal attn after each block
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
        self.mid_temporal = temporal(ch)
        self.mid_block2 = ResBlock(ch, ch, t_emb_dim, dropout, num_groups)
        self.mid_temporal2 = temporal(ch)

        # ── Decoder ──
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * channel_mult[level]
            use_attn = level in attn_levels
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = enc_channels.pop()
                block_in = ch + skip_ch
                level_blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout, num_groups))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
                level_blocks.append(temporal(out_ch))
                ch = out_ch
            self.dec_blocks.append(level_blocks)
            if level > 0:
                self.upsamples.append(DCUpsample(ch, ch))
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-6)
        self.out_conv = SphereConv2d(ch, self.out_channels, kernel_size=(3, 3), padding=(1, 1))
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.initialize_weights()

    def initialize_weights(self):
        """Xavier-uniform for linear/conv layers, preserving deliberate zero-inits."""
        for name, module in self.named_modules():
            # Skip layers that were deliberately zero-initialized.
            if name.endswith(('out_conv', 'conv2', 'proj')) or 'adaln' in name:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d, SphereConv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, x_noised, c_noise, c_grid=None, c_scalar=None):
        b, W, C, H, Wd = x_noised.shape
        Hw, Ww = self.nlat_work, self.nlon_work

        # Bilinearly interpolate the state up to the working grid (divisible by
        # size_mult) so all down/up-sampling is exact; no zero padding needed.
        x = F.interpolate(x_noised.flatten(0, 1), size=(Hw, Ww),
                          mode='bilinear', align_corners=False)   # (b*W, C, Hw, Ww)
        x = x.unflatten(0, (b, W))                                # (b, W, C, Hw, Ww)

        # Per-frame input: concat [x_noised_w, c_grid_emb_w, c_scalar_emb_w] along channels.
        feats = [x]
        if self.c_grid_embed is not None:
            # Interpolate full-res forcings to (Hw*ds, Ww*ds), then strided-conv to (Hw, Ww).
            cg = F.interpolate(c_grid.flatten(0, 1),
                               size=(Hw * self.c_grid_downsample, Ww * self.c_grid_downsample),
                               mode='bilinear', align_corners=False)
            cg = self.c_grid_embed(cg)                            # (b*W, c_grid_embed, Hw, Ww)
            cg = cg.unflatten(0, (b, W))                          # (b, W, c_grid_embed, Hw, Ww)
            feats.append(cg)
        if self.scalar_embedder is not None:
            cs = self.scalar_embedder(c_scalar.reshape(b * W, self.scalar_dim))  # (b*W, emb, Hw, Ww)
            cs = cs.unflatten(0, (b, W))                          # (b, W, c_scalar_embed, Hw, Ww)
            feats.append(cs)
        x = torch.cat(feats, dim=2)                                # (b, W, in, Hw, Ww)
        x = x.flatten(0, 1)                                        # (b*W, in, Hw, Ww)

        # Per-frame noise embedding.
        t_emb = self.t_embedder(c_noise.reshape(b * W, 1))         # (b*W, t_emb_dim)

        x = self.input_conv(x)

        def run(block, x):
            if isinstance(block, ResBlock):
                return block(x, t_emb)
            elif isinstance(block, CausalTemporalAttention):
                return block(x, t_emb, b, W)
            else:  # spatial AttentionBlock
                return block(x)

        # ── Encoder ──
        skips = [x]
        for level in range(self.num_levels):
            for block in self.enc_blocks[level]:
                x = run(block, x)
                if isinstance(block, ResBlock):
                    skips.append(x)
            if self.downsamples[level] is not None:
                x = self.downsamples[level](x)
                skips.append(x)

        # ── Bottleneck ──
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_temporal(x, t_emb, b, W)
        x = self.mid_block2(x, t_emb)
        x = self.mid_temporal2(x, t_emb, b, W)

        # ── Decoder ──
        for level_idx in range(self.num_levels):
            for block in self.dec_blocks[level_idx]:
                if isinstance(block, ResBlock):
                    x = torch.cat([x, skips.pop()], dim=1)
                x = run(block, x)
            if self.upsamples[level_idx] is not None:
                x = self.upsamples[level_idx](x)

        x = self.out_norm(x)
        x = nonlinearity(x)
        x = self.out_conv(x)                                       # (b*W, C, Hw, Ww)

        # Interpolate back down from the working grid to the native data grid.
        x = F.interpolate(x, size=(H, Wd), mode='bilinear', align_corners=False)
        return x.unflatten(0, (b, W))                             # (b, W, C, H, W)
