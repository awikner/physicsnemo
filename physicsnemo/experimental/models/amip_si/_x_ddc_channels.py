# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (common/utils.py:assemble_input / disassemble_input) for Phase 8f (F6).
#
# Upstream's x_DDC channel order is ``(surface, diagnostic, multilevel)``
# — note diagnostic comes *before* the flattened upper-air block, unlike
# the SI/SI_X/ERDM/RFM wrappers' ``(surface, upper_air, diagnostic)``
# convention in :mod:`.wrappers`. :class:`~.wrappers.XDDCWrapper` must
# use *this* channel order to load real x_DDC checkpoint weights
# correctly.

from typing import Optional

import torch
from einops import rearrange


def assemble_input(
    surface: torch.Tensor,
    multilevel: torch.Tensor,
    diagnostic: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(surface, multilevel, diagnostic) -> x`` — flattens + concatenates.

    ``multilevel`` is ``(B, C, L, H, W)``, flattened to ``(B, C*L, H, W)``
    before concatenation. Channel order is ``(surface, [diagnostic,]
    multilevel)`` — diagnostic precedes multilevel when present.
    """
    multilevel = rearrange(multilevel, "b c l h w -> b (c l) h w")
    if diagnostic is None:
        return torch.cat((surface, multilevel), dim=1)
    return torch.cat((surface, diagnostic, multilevel), dim=1)


def disassemble_input(
    x: torch.Tensor,
    nsurface: int = 6,
    ndiagnostic: int = 15,
    nlevels: int = 13,
    use_diagnostic: bool = True,
) -> tuple:
    """Inverse of :func:`assemble_input`."""
    if use_diagnostic:
        surface = x[:, :nsurface]
        diagnostic = x[:, nsurface : nsurface + ndiagnostic]
        multilevel = x[:, nsurface + ndiagnostic :]
    else:
        surface = x[:, :nsurface]
        multilevel = x[:, nsurface:]

    multilevel = rearrange(multilevel, "b (c l) h w -> b c l h w", l=nlevels)

    if use_diagnostic:
        return surface, multilevel, diagnostic
    return surface, multilevel
