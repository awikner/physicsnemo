# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from amip_v2 @ e0b7b60 (modules/layers/output_head.py) for
# Phase 12e. Adapted only in imports + docstring style; the module/parameter
# names are preserved so upstream-trained checkpoints translate 1:1.

r"""σ-conditioned mixture output head for :class:`RollingDiT` (Phase 12e).

Why this is not a normal decoder head: the ERDM scheduler wraps the network in
EDM preconditioning, ``D = c_skip * x_bar + c_out * F``, so the quantity the
head regresses is

.. math::
    F_{target} = (y - c_{skip}\,\bar{x}) / c_{out}
               = a(\sigma)\,y + b(\sigma)\,\varepsilon

and in a rolling window the ``W`` frames carry **different σ in the same
forward pass** (0.007 → 500 at global ``t=0`` for the reference config). So a
single shared head must emit near-white noise for the almost-clean front slot
and a smooth field for the max-noise back slot *simultaneously*, with only the
AdaLN conditioning on ``c_noise = ln σ / 4`` to tell them apart.

The legacy head (:class:`~.unpatchify.Unpatchify`) conditions σ as a
shift/scale in *hidden* space ahead of one **fixed** ``Linear(dim → C)`` — the
output matrix is identical at every σ. But the target above is a σ- and
channel-dependent blend of two different readouts, which this module
parameterises directly:

.. math::
    out_c = \sum_k \left( 1/K + gate_{k,c}(c_{noise}) \right) (W_k h)_c

``num_experts=1`` is a per-output-channel σ-conditioned gain (the cheap
option, which also absorbs a mismatch between ``scheduler.sigma_data`` and the
true per-channel data std); ``num_experts=2`` matches ``F_target``'s
two-regime structure, letting one expert take the ``a(σ)·y`` branch and the
other ``b(σ)·ε``.

``decoder='column'`` is orthogonal and mirrors
:class:`~.input_embed.ColumnStateEncoder` on the way out: the pressure levels
share one ``Linear(d_level → n_upper_air)`` instead of ``nlevels ×
n_upper_air`` independent columns.

.. note:: **Init discipline.** Every path zero-inits its *last* op only, so
    ``F ≡ 0`` at init (giving ``D = c_skip · x_bar``, the intended soft start)
    while every weight still receives gradient from step 1. Zero-initialising
    a whole ``Linear → act → Linear`` chain is a permanent fixed point, not a
    soft start.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["ColumnDecoder", "RollingDiTOutputHead"]


class ColumnDecoder(nn.Module):
    r"""``(B, n, dim)`` → ``(B, n, out_channels)`` on the assembly contract.

    Inverse in spirit of :class:`~.input_embed.ColumnStateEncoder`: separate
    surface and diagnostic readouts, plus a level-shared ``Linear(d_level →
    n_upper_air)`` applied at all ``nlevels`` levels with a learned per-level
    embedding.
    """

    def __init__(
        self,
        dim: int,
        out_channels: int,
        nsurface: int,
        ndiagnostic: int,
        nlevels: int,
        n_upper_air: int,
        d_level: int = 16,
    ) -> None:
        super().__init__()
        if nsurface + ndiagnostic + nlevels * n_upper_air != out_channels:
            raise ValueError(
                f"state layout {nsurface}+{ndiagnostic}+{nlevels}*{n_upper_air} "
                f"does not add up to out_channels={out_channels}"
            )
        self.nsurface, self.ndiagnostic = nsurface, ndiagnostic
        self.nlevels, self.n_upper_air = nlevels, n_upper_air
        self.out_channels = out_channels

        self.surface_proj = nn.Linear(dim, nsurface)
        self.diagnostic_proj = nn.Linear(dim, ndiagnostic)
        self.to_levels = nn.Linear(dim, nlevels * d_level)
        self.level_embed = nn.Parameter(torch.zeros(nlevels, d_level))
        self.level_proj = nn.Linear(d_level, n_upper_air)  # shared across levels

        nn.init.normal_(self.level_embed, std=0.02)
        self.zero_init_last()

    def zero_init_last(self) -> None:
        """Zero the last op of each path (``to_levels`` stays live on purpose)."""
        for m in (self.surface_proj, self.diagnostic_proj, self.level_proj):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sfc = self.surface_proj(x)
        diag = self.diagnostic_proj(x)
        ua = rearrange(self.to_levels(x), "b n (l d) -> b n l d", l=self.nlevels)
        ua = F.gelu(ua + self.level_embed)
        ua = rearrange(self.level_proj(ua), "b n l v -> b n (l v)")
        return torch.cat([sfc, diag, ua], dim=-1)


class RollingDiTOutputHead(nn.Module):
    r"""``(B, n, dim)`` + conditioning → ``(B, nlat, nlon, out_channels)``.

    Drop-in for :class:`~.unpatchify.Unpatchify` at ``patch_size=1`` (the only
    size :class:`RollingDiT` uses), with the same ``head(x, cond)`` signature.

    Parameters
    ----------
    dim : int
        Hidden width of the incoming tokens.
    out_channels : int
        Total output channels, *including* any predicted-ocean tail block.
    nlat, nlon : int
        Token grid, for the final reshape.
    cond_dim : int
        Width of the AdaLN conditioning vector (the flow-time embedding).
    num_experts : int, optional, default=1
        Readouts to mix. ``1`` = σ-conditioned per-channel gain; ``2`` matches
        the signal/noise two-regime structure of the target.
    decoder : {"flat", "column"}, optional, default="flat"
        ``"column"`` shares one per-level readout across pressure levels.
    nsurface, ndiagnostic, nlevels, n_upper_air : int, optional
        Required by ``decoder="column"``; derived from the data config.
    nocean : int, optional, default=0
        Predicted-ocean channels (Phase 12f). They form a block at the tail of
        ``out_channels`` and get their **own** experts + gate.
    d_level : int, optional, default=16
        Per-level width inside the column decoder.
    """

    def __init__(
        self,
        dim: int,
        out_channels: int,
        nlat: int,
        nlon: int,
        cond_dim: int,
        num_experts: int = 1,
        decoder: str = "flat",
        nsurface: int | None = None,
        ndiagnostic: int | None = None,
        nlevels: int | None = None,
        n_upper_air: int | None = None,
        nocean: int = 0,
        d_level: int = 16,
    ) -> None:
        super().__init__()
        if decoder not in ("flat", "column"):
            raise ValueError(f"unknown decoder {decoder!r}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        self.nlat, self.nlon = nlat, nlon
        self.out_channels = out_channels
        self.num_experts = num_experts

        # Predicted ocean channels are a tail block that gets its own experts +
        # gate; the existing tensors stay sized on the state width. This is the
        # load-bearing detail for warm starts: sizing `experts` / `gate` on
        # out_channels instead would change their shapes, and
        # load_partial_weights drops shape-mismatched keys — silently resetting
        # a trained readout to F == 0, announced only by a "skipped N keys" line.
        self.nocean = int(nocean or 0)
        self.n_state_out = out_channels - self.nocean
        if self.n_state_out <= 0:
            raise ValueError(
                f"nocean={self.nocean} leaves nothing of out_channels={out_channels}"
            )

        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # SiLU-first so the single zero-init weight is still trainable (the
        # DiTBlock convention).
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        if decoder == "column":
            missing = [
                k
                for k, v in dict(
                    nsurface=nsurface,
                    ndiagnostic=ndiagnostic,
                    nlevels=nlevels,
                    n_upper_air=n_upper_air,
                ).items()
                if v is None
            ]
            if missing:
                raise ValueError(
                    f"decoder 'column' needs {missing} (set from the data config)"
                )
            self.experts = nn.ModuleList(
                [
                    ColumnDecoder(
                        dim,
                        self.n_state_out,
                        nsurface,
                        ndiagnostic,
                        nlevels,
                        n_upper_air,
                        d_level=d_level,
                    )
                    for _ in range(num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [nn.Linear(dim, self.n_state_out) for _ in range(num_experts)]
            )
            for m in self.experts:
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

        # Per-expert, per-output-channel gate driven by the noise level.
        # Zero-init, so the head starts as the plain mean of its experts.
        self.gate = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, num_experts * self.n_state_out)
        )
        nn.init.constant_(self.gate[-1].weight, 0)
        nn.init.constant_(self.gate[-1].bias, 0)

        # Ocean block: same mixture form on its own weights. It must be
        # σ-conditioned like the rest — F_target for an ocean channel is also
        # the a(σ)y + b(σ)ε blend, so an unconditioned Linear could not emit
        # the back-frame noise regime.
        if self.nocean:
            self.ocean_experts = nn.ModuleList(
                [nn.Linear(dim, self.nocean) for _ in range(num_experts)]
            )
            self.ocean_gate = nn.Sequential(
                nn.SiLU(), nn.Linear(cond_dim, num_experts * self.nocean)
            )
        else:
            self.ocean_experts = None
            self.ocean_gate = None

    def zero_init_ocean(self) -> None:
        """Zero the ocean readout so it contributes nothing at init.

        Called from ``RollingDiT.initialize_weights`` *after* its tree-wide
        ``xavier_uniform_`` pass, which would otherwise undo an init done in
        ``__init__``. Last op of each path only (see the module note).
        """
        if not self.nocean:
            return
        for m in list(self.ocean_experts) + [self.ocean_gate[-1]]:
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, n, dim)``; ``cond``: ``(B, cond_dim)`` → ``(B, nlat, nlon, C)``."""
        shift, scale = self.adaLN_modulation(cond).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm_final(x) * (1 + scale) + shift

        g = self.gate(cond).view(-1, 1, self.num_experts, self.n_state_out)
        out = sum(
            (1.0 / self.num_experts + g[:, :, k]) * expert(h)
            for k, expert in enumerate(self.experts)
        )

        if self.nocean:
            # Reuses the modulated ``h``, so the ocean readout sees the same
            # LayerNorm + AdaLN shift/scale the state channels do.
            go = self.ocean_gate(cond).view(-1, 1, self.num_experts, self.nocean)
            ocean = sum(
                (1.0 / self.num_experts + go[:, :, k]) * expert(h)
                for k, expert in enumerate(self.ocean_experts)
            )
            out = torch.cat([out, ocean], dim=-1)

        return out.view(out.shape[0], self.nlat, self.nlon, self.out_channels)
