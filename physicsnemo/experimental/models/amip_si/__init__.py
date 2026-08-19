# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMIP stochastic-interpolant / rolling-diffusion backbones (Phase 8a).

Three backbones vendored from the amip repo @ commit 497827e
with imports rewritten to the local ``layers/`` package and wrapped in
:class:`physicsnemo.Module`:

* :class:`DiT` — single-step Patchified Diffusion Transformer (used by SI / SI_X / EDM).
* :class:`RollingDiT` — temporal rolling-window DiT with causal temporal attention
  (used by RFM and ERDM).
* :class:`ERDM` — ADM-style UNet with causal temporal attention (UNet variant for ERDM).

The diffusion schedulers that drive these backbones live at
:mod:`physicsnemo.experimental.diffusion`.

CUDA-graph + bf16 friendly metadata advertised in Phase 8f follow-ups.
"""

from ._attention import (
    attention_dtype,
    get_attention_dtype,
    set_attention_dtype,
    maybe_disable_cudnn_sdp,
)
from .dit import AmipDiT, DiTBlock, DiTCrossAttentionBlock
from .erdm_unet import ERDM, CausalTemporalAttention
from .rolling_dit import CausalTemporalBlock, RollingDiT
from .wrappers import (
    AmipDiTWrapper,
    CombinedModule,
    ERDMWrapper,
    RollingDiTWrapper,
    XDDCWrapper,
)
from .dit_ae import DiTAE
from .x_ddc import XDDCUNet

# Back-compat alias — upstream amip used the bare name ``DiT``. Use
# :class:`AmipDiT` in new code; the registry-collision with
# :class:`physicsnemo.models.dit.DiT` is what motivated the rename.
DiT = AmipDiT

__all__ = [
    "maybe_disable_cudnn_sdp",
    "attention_dtype",
    "get_attention_dtype",
    "set_attention_dtype",
    "AmipDiT",
    "AmipDiTWrapper",
    "CausalTemporalAttention",
    "CausalTemporalBlock",
    "CombinedModule",
    "DiT",
    "DiTBlock",
    "DiTCrossAttentionBlock",
    "ERDM",
    "ERDMWrapper",
    "RollingDiT",
    "RollingDiTWrapper",
    "DiTAE",
    "XDDCUNet",
    "XDDCWrapper",
]
