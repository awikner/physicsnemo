# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/models/RollingDiT.py) for Phase 8a.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataclasses import dataclass

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module as _PNeMoModule

from .dit import DiTBlock
from .layers.embedding import CalendarEmbedding
from .layers.patchify import PatchEmbed
from .layers.positional_encoding import RotaryEmbedding, TimestepEmbedder
from .layers.unpatchify import Unpatchify


@dataclass
class MetaData(ModelMetaData):
    """Phase 8a default ModelMetaData for :class:`RollingDiT`."""

    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_gpu: bool = False
    bf16: bool = False
    onnx: bool = False

# ---------------------------------------------------------------------------
# Rolling Diffusion Transformer (RollingDiT)
#
# A DiT adapted to the rolling-window emulator (RFM, see modules/diffusion/rfm.py).
# The window of W frames is processed by alternating two attention axes:
#
#   (1) SPATIAL self-attention, per frame. The window axis W is folded into the
#       batch (b*W, n, dim), so each frame attends fully over its own n = nlat*nlon
#       spatial tokens. No patching (patch_size = 1): one token per grid cell, with
#       2D RoPE on the (lat, lon) grid. This reuses DiT's DiTBlock verbatim.
#
#   (2) CAUSAL self-attention over the window, after each spatial block. The spatial
#       tokens are folded into the batch (b*n, W, dim) and attention runs purely
#       over the W slots with a causal mask: slot w attends only to slots 0..w.
#       Slot 0 is the FRONT (oldest/cleanest), slot W-1 the BACK (newest/noisiest),
#       so a noisy back frame attends to the cleaner front frames (self-conditioning)
#       while a front frame can never look forward into a noisier future frame.
#
#   (3) Grid forcings (c_grid) and the calendar (c_scalar) are injected ONCE, at the
#       input, concatenated per-frame onto the state channels (same c_grid conv
#       embedder and CalendarEmbedding as DiT). Because each frame only carries its
#       own forcing at the input and the temporal attention is causal, an earlier
#       frame can never reach a later frame's (future) forcing.
#
#   (4) The per-frame flow-time conditions both attention axes via AdaLN, using the
#       same TimestepEmbedder as DiT. It is batched simply by folding the window into
#       the batch: t (b, W) -> (b*W, 1) -> (b*W, dim). See the note in forward().
#
# Forward contract (matches the RFM backbone contract; modules/diffusion/rfm.py):
#     u = model(z, t, c_grid, c_scalar)
#       z        : (b, W, C, nlat, nlon)        the interpolant window (fed directly)
#       t        : (b, W)                        per-frame flow-time in [0, 1]
#       c_grid   : (b, W, c_grid_dim, Hf, Wf)    per-frame forcings (or None)
#       c_scalar : (b, W, scalar_dim)            per-frame calendar (or None)
#       returns u: (b, W, C, nlat, nlon)         predicted per-frame velocity x1 - eps
# ---------------------------------------------------------------------------


class CausalTemporalBlock(nn.Module):
    """Causal self-attention over the window (temporal) axis with AdaLN-Zero.

    Operates on tokens shaped (b*W, n, dim): the spatial tokens are folded into the
    batch so attention runs purely over the W frames, masked causally. The per-frame
    flow-time embedding modulates (shift/scale/gate) exactly as in DiTBlock, and the
    zero-init gate makes the block an identity at initialization.
    """

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)

        # AdaLN-Zero: (shift, scale, gate) for the temporal attention.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb, b, W, n):
        """x: (b*W, n, dim); t_emb: (b*W, dim). Returns (b*W, n, dim)."""
        dim = self.dim

        # Per-frame modulation, broadcast over the n spatial tokens.
        shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=-1)  # (b*W, dim)
        h = self.norm(x)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)               # (b*W, n, dim)

        # Fold spatial tokens into the batch, expose the window axis: (b*n, W, dim).
        h = h.view(b, W, n, dim).permute(0, 2, 1, 3).reshape(b * n, W, dim)

        qkv = self.qkv(h).reshape(b * n, W, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (b*n, heads, W, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "bn heads W hd -> bn W (heads hd)")
        out = self.attn_out(out)                          # (b*n, W, dim)

        # Restore (b*W, n, dim) and apply the gated residual.
        out = out.view(b, n, W, dim).permute(0, 2, 1, 3).reshape(b * W, n, dim)
        return x + gate.unsqueeze(1) * out


class RollingDiT(_PNeMoModule):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dim=384,
                 num_heads=8,
                 temporal_num_heads=8,
                 num_blocks=8,
                 nlat=45,
                 nlon=90,
                 dropout=0.0,
                 scalar_dim=2,
                 c_grid_dim=0,
                 c_grid_embed_dim=32,
                 c_scalar_embed_dim=16,
                 c_grid_downsample=4,
                 **kwargs):                  # tolerate extra config keys
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.nlat = nlat
        self.nlon = nlon
        # No patching: one token per grid cell.
        self.patch_size = 1
        self.grid_x = nlat
        self.grid_y = nlon
        self.with_poles = False
        self.c_grid_dim = c_grid_dim
        self.scalar_dim = scalar_dim

        # Per-frame input: [state ; conv-embedded c_grid ; calendar grid].
        patch_in_channels = in_channels

        # Grid forcings: strided conv from full-res (e.g. 180x360) to the latent grid.
        if c_grid_dim > 0 and c_grid_downsample > 0:
            self.c_grid_embed = nn.Conv2d(c_grid_dim, c_grid_embed_dim,
                                          kernel_size=c_grid_downsample,
                                          stride=c_grid_downsample)
            patch_in_channels += c_grid_embed_dim
        elif c_grid_dim > 0:
            self.c_grid_embed = None         # forcings assumed already at latent res
            patch_in_channels += c_grid_dim
        else:
            self.c_grid_embed = None

        # Calendar embedding at the latent (nlat, nlon) grid.
        if scalar_dim > 0:
            self.scalar_embedder = CalendarEmbedding(nlon=nlon, nlat=nlat,
                                                     embed_channels=c_scalar_embed_dim,
                                                     use_co2=(scalar_dim >= 3))
            patch_in_channels += c_scalar_embed_dim
        else:
            self.scalar_embedder = None

        self.patch_embed_main = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=patch_in_channels,
            hidden_size=dim,
            flatten=False)

        # 2D RoPE: one RotaryEmbedding per spatial axis, each over half the head dim.
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        # Per-frame flow-time embedding (shared with the spatial/temporal AdaLN).
        self.t_embedder = TimestepEmbedder(dim)

        # Alternating spatial (per-frame, RoPE) and causal-temporal blocks.
        self.spatial_blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout) for _ in range(num_blocks)]
        )
        self.temporal_blocks = nn.ModuleList(
            [CausalTemporalBlock(dim, temporal_num_heads, dropout=dropout) for _ in range(num_blocks)]
        )

        self.unpatchify_layer = Unpatchify(
            grid_size=(self.grid_x, self.grid_y),
            patch_size=(self.patch_size, self.patch_size),
            in_dim=dim,
            out_dim=self.out_channels,
            cond_dim=dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-init all AdaLN modulations so every block starts as an identity.
        for block in self.spatial_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.temporal_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init the output head so the model predicts zero velocity at init.
        final = self.unpatchify_layer.out_layer
        nn.init.constant_(final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(final.linear.weight, 0)
        nn.init.constant_(final.linear.bias, 0)

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)
        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        return lat, lon

    @torch.no_grad()
    def compute_rope_freqs(self, device):
        """2D RoPE cos/sin frequencies over the (nlat, nlon) grid (cached per device)."""
        if hasattr(self, '_rope_cos_lat') and self._rope_cos_lat.device == device:
            return (self._rope_cos_lat, self._rope_sin_lat,
                    self._rope_cos_lon, self._rope_sin_lon)

        lat, lon = self.get_grid(self.nlat, self.nlon, device)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')   # [nlat, nlon]
        lat_seq = lat_grid.reshape(-1)                                 # [n]
        lon_seq = lon_grid.reshape(-1)                                 # [n]

        freqs_lat = self.rope_lat(lat_seq.unsqueeze(0))   # [1, n, dim_head//2]
        freqs_lon = self.rope_lon(lon_seq.unsqueeze(0))

        self._rope_cos_lat = freqs_lat.cos()
        self._rope_sin_lat = freqs_lat.sin()
        self._rope_cos_lon = freqs_lon.cos()
        self._rope_sin_lon = freqs_lon.sin()
        return (self._rope_cos_lat, self._rope_sin_lat,
                self._rope_cos_lon, self._rope_sin_lon)

    def forward(self, z, t, c_grid=None, c_scalar=None):
        b, W, C, H, Wd = z.shape
        n = self.nlat * self.nlon

        # Fold the window into the batch so spatial layers act per-frame.
        x = z.reshape(b * W, C, H, Wd)

        # ── (3) Inject forcings ONCE, concatenated per-frame onto the input. ──
        feats = [x]
        if c_grid is not None and self.c_grid_dim > 0:
            cg = c_grid.reshape(b * W, *c_grid.shape[2:])
            if self.c_grid_embed is not None:
                cg = self.c_grid_embed(cg)            # (b*W, c_grid_embed_dim, nlat, nlon)
            feats.append(cg)
        if self.scalar_embedder is not None and c_scalar is not None:
            cs = self.scalar_embedder(c_scalar.reshape(b * W, self.scalar_dim))  # (b*W, emb, nlat, nlon)
            feats.append(cs)
        x_input = torch.cat(feats, dim=1)             # (b*W, patch_in_channels, nlat, nlon)

        # Patchify (1x1): channel-last in, [b*W, nlat, nlon, dim] out, then flatten.
        x_nhwc = x_input.permute(0, 2, 3, 1)
        x = self.patch_embed_main(x_nhwc)             # (b*W, nlat, nlon, dim)
        x = rearrange(x, 'bw ny nx c -> bw (ny nx) c')  # (b*W, n, dim)

        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = self.compute_rope_freqs(x.device)

        # ── (4) Per-frame flow-time embedding via folding the window into batch. ──
        # TimestepEmbedder expects (B, num_conds=1) and is agnostic to B, so the
        # batched window simply reshapes t (b, W) -> (b*W, 1) -> (b*W, dim).
        t_emb = self.t_embedder(t.reshape(b * W, 1))  # (b*W, dim)

        # ── (1)+(2) Alternate per-frame spatial and causal-temporal attention. ──
        for sblock, tblock in zip(self.spatial_blocks, self.temporal_blocks):
            x = sblock(x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
            x = tblock(x, t_emb, b, W, n)

        x = self.unpatchify_layer(x, t_emb)           # (b*W, nlat, nlon, out_channels)
        x = x.permute(0, 3, 1, 2)                     # (b*W, out_channels, nlat, nlon)
        return x.reshape(b, W, self.out_channels, H, Wd)
