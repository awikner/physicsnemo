# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from the amip repo @ commit 497827e
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
from .layers.input_embed import RollingDiTInputEmbed
from .layers.output_head import RollingDiTOutputHead
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


class CausalForcingCrossAttentionBlock(nn.Module):
    r"""Temporal causal cross-attention from the hidden state to the forcings.

    Mirrors :class:`CausalTemporalBlock`'s folding and AdaLN-Zero structure,
    but the queries come from the hidden state while keys/values come from a
    per-location stream of ``c_grid`` embeddings. Attention runs purely over
    time, per spatial location: query frame ``w`` attends to in-window forcings
    ``0..w`` causally. This gives the prescribed boundary forcings (SST, sea
    ice, …) a dedicated pathway to condition the prediction at depth — the
    repo's actual attend-to-the-forcings mechanism, as opposed to their
    contribution as input channels.

    Tokens are ``(b*W, n, dim)``; forcing KV is ``(b*n, W, kv_in_dim)``. The
    zero-init gate makes the block an identity at initialization, so adding it
    to a trained model is checkpoint-compatible.
    """

    def __init__(self, dim, kv_in_dim, num_heads, window_size, dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(dim, dim, bias=False)

        # Project the embedded forcing into the model dim, then standard k/v.
        self.kv_proj = nn.Linear(kv_in_dim, dim)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)

        # Learned temporal positional tables over the W window frames (key and
        # query sides). Both zero-init — no signal until learned.
        self.temporal_pos = nn.Parameter(torch.zeros(window_size, dim))
        self.query_pos = nn.Parameter(torch.zeros(window_size, dim))

        # AdaLN-Zero: (shift, scale, gate) for the cross-attention.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, forcing_kv, t_emb, b, W, n, attn_mask):
        """``x``: (b*W, n, dim); ``forcing_kv``: (b*n, W, kv_in_dim);
        ``attn_mask``: (W, W) bool. Returns (b*W, n, dim)."""
        dim = self.dim
        heads = self.num_heads
        hd = dim // heads

        # Per-frame modulation, broadcast over the n spatial tokens.
        shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=-1)
        h = self.norm_q(x)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Fold spatial tokens into the batch, expose the window (query) axis.
        h = h.view(b, W, n, dim).permute(0, 2, 1, 3).reshape(b * n, W, dim)
        q = self.to_q(h) + self.query_pos[:W].unsqueeze(0)

        # Keys/values from the forcing stream, with the temporal positional code.
        kv = self.kv_proj(forcing_kv) + self.temporal_pos[:W].unsqueeze(0)
        kv = self.norm_kv(kv)
        k = self.to_k(kv)
        v = self.to_v(kv)

        q = q.reshape(b * n, W, heads, hd).transpose(1, 2)
        k = k.reshape(b * n, W, heads, hd).transpose(1, 2)
        v = v.reshape(b * n, W, heads, hd).transpose(1, 2)

        mask = attn_mask.view(1, 1, W, W)  # broadcast over b*n, heads
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, "bn heads W hd -> bn W (heads hd)")
        out = self.attn_out(out)

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
                 window_size=6,
                 input_embed=None,
                 output_head=None,
                 global_cond=False,
                 c_grid_cross_layers=0,
                 c_grid_cross_heads=8,
                 state_layout=None,
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

        self.window_size = int(window_size)

        # Phase 12f: predicted ocean channels occupy a block at the tail of
        # in/out_channels (it arrives here inside ``state_layout``, derived by
        # the wrapper). Recorded before the projections are built so the
        # legacy-mode guard below can refuse them.
        self.nocean = int((state_layout or {}).get("nocean", 0) or 0)

        # ── Input projection (Phase 12e) ──────────────────────────────────
        # ``mode="legacy"`` reproduces the pre-12e module tree EXACTLY (same
        # submodules, same names, same shapes) so trained checkpoints load
        # unchanged; any other mode routes everything through one
        # RollingDiTInputEmbed and leaves the legacy submodules unbuilt.
        embed_cfg = (
            dict(input_embed)
            if isinstance(input_embed, dict)
            else ({} if input_embed is None else {"mode": input_embed})
        )
        self.input_embed_mode = str(embed_cfg.pop("mode", "legacy"))

        if self.input_embed_mode == "legacy":
            self.input_embed = None
            # Per-frame input: [state ; conv-embedded c_grid ; calendar grid].
            patch_in_channels = in_channels

            # Grid forcings: strided conv from full-res (e.g. 180x360) to the latent grid.
            if c_grid_dim > 0 and c_grid_downsample > 0:
                self.c_grid_embed = nn.Conv2d(c_grid_dim, c_grid_embed_dim,
                                              kernel_size=c_grid_downsample,
                                              stride=c_grid_downsample)
                patch_in_channels += c_grid_embed_dim
            elif c_grid_dim > 0:
                self.c_grid_embed = None     # forcings assumed already at latent res
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
            forcing_kv_dim = (
                c_grid_embed_dim if self.c_grid_embed is not None else c_grid_dim
            )
        else:
            self.c_grid_embed = None
            self.scalar_embedder = None
            self.patch_embed_main = None
            # ``state_layout`` carries nsurface / ndiagnostic / nlevels /
            # n_upper_air / nocean, derived by the wrapper from its variable
            # lists so the contract is stated once (never restated in a config).
            self.input_embed = RollingDiTInputEmbed(
                dim=dim, in_channels=in_channels, nlat=nlat, nlon=nlon,
                c_grid_dim=c_grid_dim, scalar_dim=scalar_dim,
                c_grid_downsample=c_grid_downsample,
                **{**(dict(state_layout) if state_layout else {}), **embed_cfg})
            forcing_kv_dim = self.input_embed.kv_dim

        # 2D RoPE: one RotaryEmbedding per spatial axis, each over half the head dim.
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        # Per-frame flow-time embedding (shared with the spatial/temporal AdaLN).
        #
        # ``global_cond`` additionally routes the *globally uniform* forcings —
        # day-of-year and (when present) the trend scalar — into that same
        # AdaLN vector, so they modulate shift/scale/gate of every block rather
        # than only contributing input channels. Over a 40-year run that scalar
        # is the signal that has to move the climate, and one number reaching
        # the model through a handful of input channels is a weak lever
        # compared with conditioning all ``num_blocks`` blocks on it.
        self.global_cond = bool(global_cond)
        # doy + (trend scalar when the calendar carries one). Second-of-day is
        # deliberately excluded: it is longitude-dependent, so it belongs in the
        # gridded calendar embedding, and it is identically zero for daily data.
        self.n_global_cond = (1 + int(scalar_dim >= 3)) if self.global_cond else 0
        if self.global_cond and scalar_dim < 2:
            raise ValueError("global_cond needs a calendar (scalar_dim >= 2)")
        self.t_embedder = TimestepEmbedder(dim, num_conds=1 + self.n_global_cond)

        # Alternating spatial (per-frame, RoPE) and causal-temporal blocks.
        self.spatial_blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout) for _ in range(num_blocks)]
        )
        self.temporal_blocks = nn.ModuleList(
            [CausalTemporalBlock(dim, temporal_num_heads, dropout=dropout) for _ in range(num_blocks)]
        )

        # Temporal causal cross-attention to the in-window forcing stream, on
        # the LAST ``c_grid_cross_layers`` blocks. Keyed by block index so the
        # state dict says which blocks carry it.
        self.c_grid_cross_layers = int(c_grid_cross_layers)
        self.forcing_blocks = nn.ModuleDict()
        if self.c_grid_cross_layers > 0:
            if c_grid_dim <= 0:
                raise ValueError("c_grid cross-attention requires c_grid_dim > 0")
            if not 0 < self.c_grid_cross_layers <= num_blocks:
                raise ValueError(
                    f"c_grid_cross_layers={self.c_grid_cross_layers} must be in "
                    f"(0, num_blocks={num_blocks}]"
                )
            if not forcing_kv_dim:
                raise ValueError(
                    "c_grid cross-attention needs a non-empty boundary embedding "
                    "(set d_boundary > 0 in budget mode, or c_grid_embed_dim > 0)"
                )
            for i in range(num_blocks - self.c_grid_cross_layers, num_blocks):
                self.forcing_blocks[str(i)] = CausalForcingCrossAttentionBlock(
                    dim, forcing_kv_dim, c_grid_cross_heads, self.window_size,
                    dropout=dropout)

        # ── Output projection (Phase 12e) ─────────────────────────────────
        head_cfg = (
            dict(output_head)
            if isinstance(output_head, dict)
            else ({} if output_head is None else {"mode": output_head})
        )
        self.output_head_mode = str(head_cfg.pop("mode", "legacy"))
        if self.output_head_mode == "legacy":
            self.output_head = None
            self.unpatchify_layer = Unpatchify(
                grid_size=(self.grid_x, self.grid_y),
                patch_size=(self.patch_size, self.patch_size),
                in_dim=dim,
                out_dim=self.out_channels,
                cond_dim=dim)
        else:
            self.unpatchify_layer = None
            self.output_head = RollingDiTOutputHead(
                dim=dim, out_channels=self.out_channels, nlat=nlat, nlon=nlon,
                cond_dim=dim,
                **{**(dict(state_layout) if state_layout else {}), **head_cfg})

        if self.nocean:
            # The legacy projections cannot carry the ocean block: PatchEmbed's
            # and Unpatchify's widths are exactly what a trained state dict
            # stores, so widening them would silently discard the trained input
            # projection and output head rather than extend them. Refuse here
            # instead of building a model that cannot load its own checkpoint.
            legacy = [
                name
                for name, mode in (
                    ("input_embed", self.input_embed_mode),
                    ("output_head", self.output_head_mode),
                )
                if mode == "legacy"
            ]
            if legacy:
                raise ValueError(
                    f"ocean_state_variables adds {self.nocean} channel(s), "
                    f"which the legacy {' and '.join(legacy)} cannot carry: "
                    f"their PatchEmbed / Unpatchify widths are checkpoint "
                    f"state-dict shapes, so widening them would discard the "
                    f"trained projection. Set input_embed.mode / "
                    f"output_head.mode to a non-legacy variant."
                )

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

        # Zero-init the forcing cross-attention gates so each block is an
        # identity at init (adding them to a trained model is a no-op).
        for block in self.forcing_blocks.values():
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init the output head so the model predicts zero at init. Both
        # heads must be re-zeroed HERE, after the tree-wide xavier pass above
        # would otherwise have overwritten the init done in their __init__.
        if self.unpatchify_layer is not None:
            final = self.unpatchify_layer.out_layer
            nn.init.constant_(final.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final.linear.weight, 0)
            nn.init.constant_(final.linear.bias, 0)
        if self.output_head is not None:
            nn.init.constant_(self.output_head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.output_head.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.output_head.gate[-1].weight, 0)
            nn.init.constant_(self.output_head.gate[-1].bias, 0)
            for expert in self.output_head.experts:
                # Last op of each path only, so every weight still gets
                # gradient from step 1 (see output_head's module note).
                if hasattr(expert, "zero_init_last"):
                    expert.zero_init_last()
                else:
                    nn.init.constant_(expert.weight, 0)
                    nn.init.constant_(expert.bias, 0)
            self.output_head.zero_init_ocean()
        if self.input_embed is not None:
            self.input_embed.zero_init_last()

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

    def _global_cond_cols(self, cs_flat, like):
        """``(B, n_global_cond)`` globally-uniform forcings for the AdaLN vector.

        Column 0 is day-of-year mapped to ``[0, 1)``; column 1 (when the
        calendar carries a trend scalar) is that already-z-scored scalar.
        Second-of-day is excluded on purpose — it is longitude-dependent, so it
        belongs in the gridded calendar embedding, and it is identically zero
        for daily-average data anyway.
        """
        if cs_flat is None:
            return like.new_zeros(like.shape[0], self.n_global_cond)
        cols = [(cs_flat[:, 1:2] / 365.25) % 1.0]
        if self.n_global_cond > 1:
            cols.append(cs_flat[:, 2:3])
        return torch.cat(cols, dim=-1).to(like.dtype)

    def _forcing_mask(self, W, device):
        """``(W, W)`` causal bool attend-mask for the forcing cross-attention.

        Query frame ``w`` attends to in-window forcing keys ``0..w``; True =
        may attend. Cached per (W, device).
        """
        cache = getattr(self, "_forcing_mask_cache", None)
        if cache is not None and cache[0] == W and cache[1].device == device:
            return cache[1]
        mask = torch.tril(torch.ones(W, W, dtype=torch.bool, device=device))
        self._forcing_mask_cache = (W, mask)
        return mask

    def forward(self, z, t, c_grid=None, c_scalar=None):
        b, W, C, H, Wd = z.shape
        n = self.nlat * self.nlon

        # Fold the window into the batch so spatial layers act per-frame.
        x = z.reshape(b * W, C, H, Wd)

        cg_flat = None
        if c_grid is not None and self.c_grid_dim > 0:
            cg_flat = c_grid.reshape(b * W, *c_grid.shape[2:])
        cs_flat = None
        if c_scalar is not None and self.scalar_dim > 0:
            cs_flat = c_scalar.reshape(b * W, self.scalar_dim)

        # ── (3) Inject forcings ONCE, per-frame, on the input. ──
        cg_embed = None                               # reused as the KV stream
        if self.input_embed is not None:
            x, cg_embed = self.input_embed(x, cg_flat, cs_flat)   # (b*W, n, dim)
        else:
            feats = [x]
            if cg_flat is not None:
                cg = cg_flat
                if self.c_grid_embed is not None:
                    cg = self.c_grid_embed(cg)        # (b*W, c_grid_embed_dim, nlat, nlon)
                cg_embed = cg
                feats.append(cg)
            if self.scalar_embedder is not None and cs_flat is not None:
                cs = self.scalar_embedder(cs_flat)    # (b*W, emb, nlat, nlon)
                feats.append(cs)
            x_input = torch.cat(feats, dim=1)         # (b*W, patch_in_channels, nlat, nlon)

            # Patchify (1x1): channel-last in, [b*W, nlat, nlon, dim] out, then flatten.
            x_nhwc = x_input.permute(0, 2, 3, 1)
            x = self.patch_embed_main(x_nhwc)         # (b*W, nlat, nlon, dim)
            x = rearrange(x, 'bw ny nx c -> bw (ny nx) c')  # (b*W, n, dim)

        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = self.compute_rope_freqs(x.device)

        # ── (4) Per-frame flow-time embedding via folding the window into batch. ──
        # TimestepEmbedder expects (B, num_conds) and is agnostic to B, so the
        # batched window simply reshapes t (b, W) -> (b*W, 1) -> (b*W, dim).
        t_cond = t.reshape(b * W, 1)
        if self.global_cond:
            t_cond = torch.cat(
                [t_cond, self._global_cond_cols(cs_flat, t_cond)], dim=-1
            )
        t_emb = self.t_embedder(t_cond)               # (b*W, dim)

        # ── Build the in-window forcing key/value stream for the cross-attention. ──
        # Per spatial location, the W in-window forcing embeddings: (b*n, W, Ce).
        forcing_kv = None
        forcing_mask = None
        if len(self.forcing_blocks) > 0 and cg_embed is not None:
            Ce = cg_embed.shape[1]
            cg_win = cg_embed.view(b, W, Ce, self.nlat, self.nlon)
            # Fold (lat, lon) into the batch to match the hidden-state order.
            forcing_kv = cg_win.permute(0, 3, 4, 1, 2).reshape(b * n, W, Ce)
            forcing_mask = self._forcing_mask(W, x.device)

        # ── (1)+(2) Alternate per-frame spatial and causal-temporal attention,
        #            with forcing cross-attention on the last N blocks. ──
        for i, (sblock, tblock) in enumerate(zip(self.spatial_blocks, self.temporal_blocks)):
            if forcing_kv is not None and str(i) in self.forcing_blocks:
                x = self.forcing_blocks[str(i)](
                    x, forcing_kv, t_emb, b, W, n, forcing_mask
                )
            x = sblock(x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
            x = tblock(x, t_emb, b, W, n)

        head = self.unpatchify_layer if self.output_head is None else self.output_head
        x = head(x, t_emb)                            # (b*W, nlat, nlon, out_channels)
        x = x.permute(0, 3, 1, 2)                     # (b*W, out_channels, nlat, nlon)
        return x.reshape(b, W, self.out_channels, H, Wd)
