# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/models/DiT.py) for Phase 8a. Imports rewritten to the
# local vendored ``layers`` package; no logic changes.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from dataclasses import dataclass

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module as _PNeMoModule

from .layers.cross_attention import CrossAttention
from .layers.embedding import CalendarEmbedding
from .layers.patchify import PatchEmbed
from .layers.positional_encoding import (
    RotaryEmbedding,
    TimestepEmbedder,
    apply_2d_rotary_pos_emb,
)
from .layers.unpatchify import Unpatchify, sphere_pad


@dataclass
class MetaData(ModelMetaData):
    """Phase 8a default ModelMetaData for :class:`DiT`.

    fp32-only at first commit. ``_NativeMetaData`` advertising
    ``cuda_graphs=True`` + ``bf16=True`` follows in Phase 8f.
    """

    jit: bool = False
    cuda_graphs: bool = False  # iterative diffusion sampling + dynamic shapes
    amp: bool = False
    amp_gpu: bool = False
    bf16: bool = False
    onnx: bool = False

class DiTBlock(nn.Module):
    """
    Vanilla self-attention transformer block with AdaLN-Zero timestep conditioning
    and 2D Rotary Position Embedding (RoPE).

    Input/output shape: [b, n, dim] where n = (nlat//p) * (nlon//p).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)

        # Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # AdaLN-Zero modulation: 6 * dim for (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

        # Zero-init the modulation output
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon):
        """
        Args:
            x: [b, n, dim]
            t_emb: [b, dim] timestep embedding
            rope_cos_lat: [1, n, dim_head//2] cosine freqs for latitude
            rope_sin_lat: [1, n, dim_head//2] sine freqs for latitude
            rope_cos_lon: [1, n, dim_head//2] cosine freqs for longitude
            rope_sin_lon: [1, n, dim_head//2] sine freqs for longitude
        """
        # AdaLN modulation parameters
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # [b, 1, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa

        b, n, c = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, heads, n, dim_head]
        q, k, v = qkv.unbind(0)

        # Apply 2D RoPE to q and k
        q = apply_2d_rotary_pos_emb(q, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
        k = apply_2d_rotary_pos_emb(k, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        h = self.attn_out(h) # [b, n, dim]

        x = x + gate_msa * h

        # FFN with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class DiTCrossAttentionBlock(nn.Module):
    """
    DiT block: self-attention -> cross-attention -> MLP, each with AdaLN-Zero
    timestep conditioning. Self-attention and cross-attention both use 2D RoPE.

    Context tokens are expected to share the same patch grid as the queries so
    that RoPE frequencies apply to both sides.
    """

    def __init__(self, dim, context_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        dim_head = dim // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)

        # Cross-attention (queries from main stream, keys/values from context)
        self.norm_ca_q = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_ca_kv = nn.LayerNorm(context_dim, eps=1e-6)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # AdaLN-Zero: 9 * dim for (shift/scale/gate) x (self-attn, cross-attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, context, t_emb,
                rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon):
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # [b, 1, 9*dim]
        (shift_msa, scale_msa, gate_msa,
         shift_ca,  scale_ca,  gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(9, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa

        b, n, c = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, heads, n, dim_head]
        q, k, v = qkv.unbind(0)

        q = apply_2d_rotary_pos_emb(q, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
        k = apply_2d_rotary_pos_emb(k, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        h = self.attn_out(h)

        x = x + gate_msa * h

        # Cross-attention with AdaLN
        h = self.norm_ca_q(x)
        h = h * (1 + scale_ca) + shift_ca
        ctx = self.norm_ca_kv(context)
        h = self.cross_attn(h, ctx,
                            rope_cos_lat, rope_sin_lat,
                            rope_cos_lon, rope_sin_lon)
        x = x + gate_ca * h

        # FFN with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class AmipDiT(_PNeMoModule):
    """
    Patchified Diffusion Transformer for stochastic interpolant velocity prediction.

    Class name carries the ``Amip`` prefix to disambiguate from the
    upstream :class:`physicsnemo.models.dit.DiT` registered under the
    base ``DiT`` name in :class:`physicsnemo.core.registry.ModelRegistry`.

    Architecture:
    - PatchEmbed for main input
    - 2D Rotary Position Embedding (RoPE) on lat/lon patch grid
    - N blocks of DiTBlock (self-attn with AdaLN + RoPE)
    - Unpatchify: dim -> out_channels @ nlat x nlon
    - Zero-initialized output projection
    """

    def __init__(self,
                 in_channels=249,
                 out_channels=249,
                 dim=384,
                 num_heads=8,
                 num_blocks=8,
                 patch_size=2,
                 nlat=45,
                 nlon=90,
                 dropout=0.0,
                 scalar_dim=1,
                 c_grid_dim=0,
                 c_grid_embed_dim=4,
                 c_scalar_embed_dim=4,
                 c_grid_downsample=4,
                 num_ca_blocks = 0,
                 num_output_blocks = 0,):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.nlat = nlat
        self.nlon = nlon
        self.dropout = dropout
        self.c_grid_dim = c_grid_dim

        # Pad spatial dims to be divisible by patch_size              # test case if nlat,nlon = 45, 90
        self.nlat_pad = math.ceil(nlat / patch_size) * patch_size     # 45/2 = 23 * 2 = 46
        self.nlon_pad = math.ceil(nlon / patch_size) * patch_size     # 90/2 = 45 * 2 = 90
        self.pad_lat = self.nlat_pad - nlat                           # 46 - 45 = 1
        self.pad_lon = self.nlon_pad - nlon                           # 90 - 90 = 0

        # Polar padding for latitude (split top/bottom), circular for longitude (split left/right)
        self.pad_lat_top = math.ceil(self.pad_lat / 2)                # 1/2 = 0.5 -> 1
        self.pad_lat_bottom = self.pad_lat - self.pad_lat_top         # 1 - 1 = 0
        self.pad_lon_left = math.ceil(self.pad_lon / 2)               # 0/2 = 0
        self.pad_lon_right = self.pad_lon - self.pad_lon_left         # 0 - 0 = 0

        self.grid_x = self.nlat_pad // patch_size
        self.grid_y = self.nlon_pad // patch_size
        self.with_poles = False

        # c_grid downsampling: (c_grid_dim, 180, 360) -> (c_grid_embed_dim, 45, 90)
        if c_grid_downsample > 0:
            self.c_grid_embed = nn.Conv2d(c_grid_dim, c_grid_embed_dim,
                                          kernel_size=c_grid_downsample,
                                          stride=c_grid_downsample)
            patch_in_channels = in_channels + c_grid_embed_dim
        else:
            self.c_grid_embed = None
            patch_in_channels = in_channels + c_grid_dim

        # Calendar embedding produced directly at latent (zlat, zlon) res so
        # it can be concatenated with c_grid_emb along the channel dim.
        if scalar_dim > 0:
            self.scalar_embedder = CalendarEmbedding(nlon=nlon, nlat=nlat, 
                                                     embed_channels=c_scalar_embed_dim, 
                                                     use_co2 = (scalar_dim >= 3))
        else:
            self.scalar_embedder = None

        if self.scalar_embedder is not None:
            patch_in_channels += c_scalar_embed_dim

        # Cross-attention context embedder. Projects the concatenated
        # (c_grid_emb, c_scalar_emb) — shape [b, 2*c_grid_embed_dim, zlat, zlon] —
        # to [b, n_patches, dim] on the same patch grid as the main stream so
        # self- and cross-attention share the same 2D RoPE.
        self.num_ca_blocks = num_ca_blocks
        if num_ca_blocks > 0:
            ca_in_channels = 0
            if c_grid_dim > 0:
                ca_in_channels += c_grid_embed_dim if c_grid_downsample > 0 else c_grid_dim
            if self.scalar_embedder is not None:
                ca_in_channels += c_scalar_embed_dim
            assert ca_in_channels > 0, "num_ca_blocks > 0 requires c_grid or scalar conditioning"
            self.ca_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=ca_in_channels,
                hidden_size=dim,
                flatten=False,
            )
        else:
            self.ca_embed = None

        self.patch_embed_main = PatchEmbed(
            patch_size=patch_size,
            in_chans=patch_in_channels,
            hidden_size=dim,
            flatten=False)

        # 2D RoPE: one RotaryEmbedding per spatial axis
        # Each axis gets half the head dimension
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        self.t_embedder = TimestepEmbedder(dim)

        # Transformer blocks. The final `num_ca_blocks` are cross-attention
        # blocks that attend to the (c_grid, c_scalar) context; the rest are
        # plain self-attention DiT blocks.
        assert num_ca_blocks <= num_blocks, "num_ca_blocks must be <= num_blocks"
        num_sa_only = num_blocks - num_ca_blocks
        sa_blocks = []
        for _ in range(num_sa_only):
            sa_blocks.append(DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout))
        for _ in range(num_ca_blocks):
            sa_blocks.append(DiTCrossAttentionBlock(
                dim=dim,
                context_dim=dim,
                num_heads=num_heads,
                mlp_ratio=4,
                dropout=dropout,
            ))

        for _ in range(num_output_blocks):
            sa_blocks.append(DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout))

        self.sa_blocks = nn.ModuleList(sa_blocks)

        self.unpatchify_layer = Unpatchify(
            grid_size=(self.grid_x, self.grid_y),
            patch_size=(patch_size, patch_size),
            in_dim=dim,
            out_dim=out_channels,
            cond_dim=dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Re-zero-init AdaLN modulation outputs (apply overwrites them).
        # Both DiTBlock and DiTCrossAttentionBlock expose `adaLN_modulation`.
        for block in self.sa_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        """Compute 2D RoPE cos/sin frequencies for the patch grid.

        Uses physical lat/lon coordinates at patch centers so the model
        encodes actual geographic position rather than integer indices.

        Returns cached buffers after first call.
        """
        if hasattr(self, '_rope_cos_lat') and self._rope_cos_lat.device == device:
            return (self._rope_cos_lat, self._rope_sin_lat,
                    self._rope_cos_lon, self._rope_sin_lon)

        # Get physical coordinates at padded resolution
        lat, lon = self.get_grid(self.nlat_pad, self.nlon_pad, device)

        # Average pool to patch centers: [nlat_pad] -> [grid_x], [nlon_pad] -> [grid_y]
        lat_patches = lat.reshape(self.grid_x, self.patch_size).mean(dim=1)  # [grid_x]
        lon_patches = lon.reshape(self.grid_y, self.patch_size).mean(dim=1)  # [grid_y]

        # Create 2D grid of patch positions and flatten to sequence
        # lat_grid[i,j] = lat of patch (i,j), lon_grid[i,j] = lon of patch (i,j)
        lat_grid, lon_grid = torch.meshgrid(lat_patches, lon_patches, indexing='ij')  # [grid_x, grid_y]
        lat_seq = lat_grid.reshape(-1)  # [n]
        lon_seq = lon_grid.reshape(-1)  # [n]

        # Compute RoPE frequencies: [n, dim_head//2] -> cos/sin each [1, n, dim_head//2]
        freqs_lat = self.rope_lat(lat_seq.unsqueeze(0))  # [1, n, dim_head//2]
        freqs_lon = self.rope_lon(lon_seq.unsqueeze(0))  # [1, n, dim_head//2]

        self._rope_cos_lat = freqs_lat.cos()
        self._rope_sin_lat = freqs_lat.sin()
        self._rope_cos_lon = freqs_lon.cos()
        self._rope_sin_lon = freqs_lon.sin()

        return (self._rope_cos_lat, self._rope_sin_lat,
                self._rope_cos_lon, self._rope_sin_lon)

    def forward(self, x_noised, cond, t, c_grid = None, c_scalar = None):
        """
        Args:
            x_noised: [b, c, zlat, zlon] — interpolant I_t (channel-first from assemble_input)
            cond: [b, c, zlat, zlon] — conditional information (current state)
            t: [b, 1] — diffusion timestep
            c_grid: optional [b, c_grid_dim, nlat, nlon] grid-aligned conditioning (e.g. land/ocean mask)
            c_scalar: optional [b, scalar_dim] non-spatial conditioning (e.g. time embeddings)

        Returns:
            [b, c, nlat, nlon] — predicted output (channel-first)
        """

        nlat, nlon = self.nlat, self.nlon

        # Embed grid-aligned and scalar conditioning to (c, zlat, zlon).
        if self.c_grid_embed is not None and c_grid is not None:
            c_grid_emb = self.c_grid_embed(c_grid)  # [b, c_grid_embed_dim, zlat, zlon]
        else:
            c_grid_emb = None

        if self.scalar_embedder is not None and c_scalar is not None:
            c_scalar_emb = self.scalar_embedder(c_scalar)  # [b, c_scalar_embed_dim, nlat, nlon]
        else:
            c_scalar_emb = None

        cond_list = [x_noised, cond]
        if c_grid_emb is not None:
            cond_list.append(c_grid_emb)
        if c_scalar_emb is not None:
            cond_list.append(c_scalar_emb)
        x_input = torch.cat(cond_list, dim=1)

        # Pad spatial dims to be divisible by patch_size
        # Circular padding in longitude, polar padding in latitude
        x_input = sphere_pad(x_input, padding=(self.pad_lon_left, self.pad_lon_right, self.pad_lat_top, self.pad_lat_bottom))

        # Convert channel-first to channel-last for PatchEmbed: [b, c, h, w] -> [b, h, w, c]
        x_nhwc = x_input.permute(0, 2, 3, 1)

        # Patchify: [b, h, w, c] -> [b, h//p, w//p, dim]
        x = self.patch_embed_main(x_nhwc)

        # Flatten spatial dims for sequence processing: [b, h//p, w//p, dim] -> [b, n, dim]
        x = rearrange(x, 'b ny nx c -> b (ny nx) c')

        # Compute 2D RoPE frequencies for patch grid
        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = self.compute_rope_freqs(x.device)

        # Build cross-attention context tokens on the same patch grid.
        context = None
        if self.ca_embed is not None:
            ctx_parts = []
            if c_grid_emb is not None:
                ctx_parts.append(c_grid_emb)
            if c_scalar_emb is not None:
                ctx_parts.append(c_scalar_emb)
            assert len(ctx_parts) > 0, "num_ca_blocks > 0 requires c_grid or c_scalar input"
            ctx_input = torch.cat(ctx_parts, dim=1)
            ctx_input = sphere_pad(ctx_input, padding=(self.pad_lon_left, self.pad_lon_right, self.pad_lat_top, self.pad_lat_bottom))
            ctx_nhwc = ctx_input.permute(0, 2, 3, 1)
            context = self.ca_embed(ctx_nhwc)  # [b, h//p, w//p, dim]
            context = rearrange(context, 'b ny nx c -> b (ny nx) c')

        # Timestep embedding
        if len(t.shape) == 1:
            t = t[:, None]
        
        # additionally pass scalars into AdaLN
        t_emb = self.t_embedder(t)  # [b, dim]

        # Transformer blocks with RoPE. First blocks are pure self-attention;
        # the final num_ca_blocks additionally cross-attend to the context.
        for block in self.sa_blocks:
            if isinstance(block, DiTCrossAttentionBlock):
                x = block(x, context, t_emb,
                          rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
            else:
                x = block(x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        x = self.unpatchify_layer(x, t_emb)

        # Convert back to channel-first: [b, h, w, c] -> [b, c, h, w]
        x = x.permute(0, 3, 1, 2)
        
        # Crop back to original spatial dims
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = x[:, :, self.pad_lat_top:self.pad_lat_top + nlat, self.pad_lon_left:self.pad_lon_left + nlon]

        return x

