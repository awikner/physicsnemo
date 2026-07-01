# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/layers/bilinear.py) for Phase 8f (F6). Plain (non-``nn.Module``)
# classes — ``F.interpolate`` has no learnable parameters, matching
# upstream's own choice not to subclass ``nn.Module`` here.

from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange


class BilinearDownsample:
    def __init__(self, downsample_factor: int = 4):
        self.downsample_factor = downsample_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, scale_factor=1 / self.downsample_factor, mode="bilinear", align_corners=False
        )


class BilinearEncoder:
    r"""Downsample ``(surface, multilevel, diagnostic)`` by ``downsample_factor``.

    Used by the x_DDC super-resolution cascade to produce the low-res
    "forecaster-shaped" input from a full-res field (during autoencoder
    training, the low-res field is downsampled directly from the
    full-res truth; at inference the low-res field instead comes from
    a real forecaster's prediction — see :class:`~.wrappers.CombinedModule`).
    """

    def __init__(self, downsample_factor: int = 4):
        self.downsample_factor = downsample_factor

    def __call__(
        self,
        surface: torch.Tensor,
        multilevel: torch.Tensor,
        diagnostic: Optional[torch.Tensor] = None,
        assemble: bool = False,
    ) -> Any:
        z_surface, z_multilevel, z_diagnostic = self.forward(surface, multilevel, diagnostic)
        if assemble:
            from .._x_ddc_channels import assemble_input

            return assemble_input(z_surface, z_multilevel, z_diagnostic)
        return z_surface, z_multilevel, z_diagnostic

    def forward(
        self,
        surface: torch.Tensor,
        multilevel: torch.Tensor,
        diagnostic: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # surface: (B, C, H, W); multilevel: (B, C, L, H, W); diagnostic: (B, C, H, W)
        nlevels = multilevel.shape[2]
        multilevel = rearrange(multilevel, "b c l h w -> b (c l) h w")

        surface = F.interpolate(
            surface, scale_factor=1 / self.downsample_factor, mode="bilinear", align_corners=False
        )
        multilevel = F.interpolate(
            multilevel, scale_factor=1 / self.downsample_factor, mode="bilinear", align_corners=False
        )
        multilevel = rearrange(multilevel, "b (c l) h w -> b c l h w", l=nlevels)

        if diagnostic is not None:
            diagnostic = F.interpolate(
                diagnostic, scale_factor=1 / self.downsample_factor, mode="bilinear", align_corners=False
            )

        return surface, multilevel, diagnostic


class BilinearDecoder:
    r"""Upsample ``(surface, multilevel, diagnostic)`` by ``downsample_factor``.

    Inverse of :class:`BilinearEncoder` — brings a low-res field back
    up to full resolution (blurrily; the x_DDC diffusion denoiser
    restores the missing detail).
    """

    def __init__(self, downsample_factor: int = 4):
        self.downsample_factor = downsample_factor

    def __call__(
        self,
        surface: torch.Tensor,
        multilevel: torch.Tensor,
        diagnostic: Optional[torch.Tensor] = None,
    ) -> Any:
        return self.forward(surface, multilevel, diagnostic)

    def forward(
        self,
        surface: torch.Tensor,
        multilevel: torch.Tensor,
        diagnostic: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        nlevels = multilevel.shape[2]
        multilevel = rearrange(multilevel, "b c l h w -> b (c l) h w")

        surface = F.interpolate(
            surface, scale_factor=self.downsample_factor, mode="bilinear", align_corners=False
        )
        multilevel = F.interpolate(
            multilevel, scale_factor=self.downsample_factor, mode="bilinear", align_corners=False
        )
        multilevel = rearrange(multilevel, "b (c l) h w -> b c l h w", l=nlevels)

        if diagnostic is not None:
            diagnostic = F.interpolate(
                diagnostic, scale_factor=self.downsample_factor, mode="bilinear", align_corners=False
            )

        return surface, multilevel, diagnostic
