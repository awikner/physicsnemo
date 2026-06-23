# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored building blocks for the AMIP diffusion backbones.

Sourced from /work/nvme/bdiu/awikner/amip @ commit 497827e
(modules/layers/*.py) and lightly edited to drop upstream-private
imports (``modules.layers.old.*``) and adopt local relative imports.
"""

from .conv import (
    DCDownsample,
    DCUpsample,
    ResnetBlock,
    SphereConv2d,
    nonlinearity,
)
from .cross_attention import CrossAttention
from .embedding import CalendarEmbedding, FrequencyEmbedding
from .patchify import PatchEmbed
from .positional_encoding import (
    GaussianFourierFeatureTransform,
    RotaryEmbedding,
    TimestepEmbedder,
    apply_2d_rotary_pos_emb,
    apply_3d_rotary_pos_emb,
    apply_rotary_pos_emb,
)
from .unpatchify import FinalLayer, Unpatchify, modulate_fused, sphere_pad

__all__ = [
    "CalendarEmbedding",
    "CrossAttention",
    "DCDownsample",
    "DCUpsample",
    "FinalLayer",
    "FrequencyEmbedding",
    "GaussianFourierFeatureTransform",
    "PatchEmbed",
    "ResnetBlock",
    "RotaryEmbedding",
    "SphereConv2d",
    "TimestepEmbedder",
    "Unpatchify",
    "apply_2d_rotary_pos_emb",
    "apply_3d_rotary_pos_emb",
    "apply_rotary_pos_emb",
    "modulate_fused",
    "nonlinearity",
    "sphere_pad",
]
