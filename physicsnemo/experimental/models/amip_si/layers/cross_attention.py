# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/layers/cross_attention.py) for Phase 8a. The upstream's
# `CrossAttentionBlock` (which depended on the `modules.layers.old.MLP`
# helper) is dropped — only `CrossAttention` is consumed by our 3
# vendored backbones (DiTCrossAttentionBlock instantiates it).

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .positional_encoding import apply_2d_rotary_pos_emb


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional 2D RoPE on q and k.

    Uses F.scaled_dot_product_attention. Queries come from `x`, keys/values
    from `context`. If RoPE frequencies are provided they must correspond to
    the same grid for both the query and context token sequences (i.e. the
    context is expected to live on the same patch grid as the queries).
    """

    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context,
                rope_cos_lat=None, rope_sin_lat=None,
                rope_cos_lon=None, rope_sin_lon=None):
        b, nq, _ = x.shape
        nk = context.shape[1]
        h, d = self.heads, self.dim_head

        q = self.to_q(x).reshape(b, nq, h, d).transpose(1, 2)   # [b, h, nq, d]
        k = self.to_k(context).reshape(b, nk, h, d).transpose(1, 2)
        v = self.to_v(context).reshape(b, nk, h, d).transpose(1, 2)

        if rope_cos_lat is not None:
            q = apply_2d_rotary_pos_emb(q, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
            k = apply_2d_rotary_pos_emb(k, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# CrossAttentionBlock (upstream lines 54–95) intentionally dropped — it
# depended on the upstream `modules.layers.old.fa_basics.MLP` helper and
# is unused by our three vendored backbones.
