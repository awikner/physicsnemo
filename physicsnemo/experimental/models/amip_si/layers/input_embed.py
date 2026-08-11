# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from amip_v2 @ e0b7b60 (modules/layers/input_embed.py) for
# Phase 12e. Adapted only in imports + docstring style; module/parameter names
# are preserved so upstream-trained checkpoints translate 1:1.
#
# One naming note: upstream's CalendarEmbedding takes ``use_trend_scalar``
# where the fork's (Phase-8a) vendored copy takes ``use_co2`` — a pure rename
# upstream made when the slot generalised from CO2 to any trend scalar. The
# fork keeps ``use_co2`` because that name is baked into the frozen legacy path
# and its translated checkpoints; the budget path below always passes False.

r"""Budgeted input projection for :class:`RollingDiT` (Phase 12e).

AMIP is a *forced* problem: the prognostic state is what the model must
invent, while the boundary conditions (SST, sea ice, TOA insolation, CO₂,
land–sea mask, surface geopotential, calendar) are read from ground truth at
every step of a 40-year rollout and are therefore the only thing anchoring the
trajectory. How much of the hidden width they get is a design decision, not
something to leave to the accident of raw channel counts.

The legacy projection (``mode="legacy"``, still the default here) is::

    cat([ x(151) , conv4x4(c_grid)(64) , CalendarEmbedding(c_scalar)(16) ])
      -> Conv1x1(231 -> dim)

with three problems: **no budget** (at init the state owns 151/231 = 65% of
the input variance and the calendar 7%, with CO₂ one of six blocks inside
those 16 channels — so the single scalar driving a climate-change run reaches
the model through ~2 effective channels of 1024); **mixed spaces** (raw
z-scored physics glued to already non-linearly embedded forcings); and
**contract-blindness** (a flat 1×1 conv discards the
``[surface | diagnostic | upper air]`` layout, and the 5 upper-air variables
are the same physics at all 26 levels).

``mode="budget"`` gives each source its own encoder and its own explicitly
sized, disjoint slice of ``dim``::

    token = [ E_state(x) | E_boundary(c_grid) | E_scalar(c_scalar) ]
              d_state         d_boundary          d_calendar        = dim

each RMS-normalised before the concat, so how *loud* a source is follows the
budget rather than its channel count. Sub-switches: ``state_encoder``
(``flat`` | ``column``), ``boundary_encoder`` (``conv1`` | ``conv2``),
``boundary_pool_stats``, ``boundary_static_bias``, ``d_co2`` / ``co2_linear``,
``source_norm`` — see the parameter docs.

``mode="legacy"`` leaves :class:`RollingDiT` bit-identical to its pre-12e
forward, so existing checkpoints still load.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .embedding import CalendarEmbedding
from .unpatchify import sphere_pad

__all__ = [
    "BoundaryEncoder",
    "ColumnStateEncoder",
    "RollingDiTInputEmbed",
    "ScalarForcingEmbedder",
    "SourceNorm",
]


def _round_to(x, multiple: int = 8, minimum: int = 8) -> int:
    """Round ``x`` up to a multiple of ``multiple``, at least ``minimum``."""
    x = max(int(x), minimum)
    return ((x + multiple - 1) // multiple) * multiple


class SourceNorm(nn.Module):
    r"""Channel-wise RMS norm over a ``(B, C, H, W)`` map, with a learned gain.

    Applied to every source before the concat so the relative loudness of
    state vs. boundary vs. calendar is set by the channel budget and the
    learned gain, not by how many raw channels each source happens to carry.
    Gain is per-channel and initialised to 1, so this is a pure re-scaling at
    init.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        scale = xf.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return (xf * scale).to(dtype) * self.weight.view(1, -1, 1, 1)


class BoundaryEncoder(nn.Module):
    r"""``(B, c_in, Hf, Wf)`` forcings → ``(B, d_out, nlat, nlon)``.

    Also the key/value source for :class:`RollingDiT`'s causal forcing
    cross-attention, so ``d_out`` doubles as ``c_grid_kv_dim``.

    The forcings arrive at native resolution (e.g. 180×360) while the state
    lives on the token grid (45×90); ``downsample`` is the ratio.

    Parameters
    ----------
    depth : {"conv1", "conv2"}
        ``conv1`` is one strided conv straight to ``d_out`` (the legacy shape,
        kept so a run can isolate the budget's effect from the encoder's).
        ``conv2`` is strided conv → GELU → 3×3 conv with spherical padding.
    pool_stats : bool
        Concatenate the **exact** area mean and std of each raw field over
        every coarse cell. The mean keeps the physical SST/SIC value reachable
        without passing through a learned kernel; the std is sub-grid
        heterogeneity — for the land–sea mask literally coastline fraction, for
        surface geopotential orographic roughness. A stride-4 conv can
        represent neither. Automatically inert at ``downsample == 1`` (there is
        nothing to pool over).
    static_bias : bool
        A learned ``(d_out, nlat, nlon)`` field added to the slice. The land
        map never changes and the model has no absolute spatial embedding
        otherwise (RoPE is relative), so this is a cheap geographic prior.
        Zero-init: no signal until it earns one.
    """

    def __init__(
        self,
        c_in: int,
        d_out: int,
        nlat: int,
        nlon: int,
        downsample: int = 4,
        depth: str = "conv2",
        pool_stats: bool = True,
        static_bias: bool = True,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        if depth not in ("conv1", "conv2"):
            raise ValueError(f"unknown boundary_encoder {depth!r}")
        self.c_in = c_in
        self.d_out = d_out
        self.downsample = int(downsample) if downsample else 1
        self.depth = depth
        self.pool_stats = bool(pool_stats) and self.downsample > 1

        k = self.downsample
        hidden = hidden or d_out

        if depth == "conv1":
            self.down = (
                nn.Conv2d(c_in, d_out, kernel_size=k, stride=k)
                if k > 1
                else nn.Conv2d(c_in, d_out, kernel_size=1)
            )
            self.refine = None
        else:
            self.down = (
                nn.Conv2d(c_in, hidden, kernel_size=k, stride=k)
                if k > 1
                else nn.Conv2d(c_in, hidden, kernel_size=1)
            )
            stats_ch = 2 * c_in if self.pool_stats else 0
            # 3x3 with spherical padding: periodic in longitude, pole-flipped
            # in latitude, so a coastline at the dateline is not a seam.
            self.refine = nn.Conv2d(hidden + stats_ch, d_out, kernel_size=3)

        self.static_bias = (
            nn.Parameter(torch.zeros(d_out, nlat, nlon)) if static_bias else None
        )

    def _pooled_stats(self, cg: torch.Tensor) -> torch.Tensor:
        """Exact area mean and std of each raw field over one coarse cell."""
        k = self.downsample
        m = F.avg_pool2d(cg, k)
        v = F.avg_pool2d(cg * cg, k) - m * m
        return torch.cat([m, v.clamp_min(0).sqrt()], dim=1)

    def forward(self, cg: torch.Tensor) -> torch.Tensor:
        h = self.down(cg)
        if self.refine is not None:
            h = F.gelu(h)
            if self.pool_stats:
                h = torch.cat([h, self._pooled_stats(cg)], dim=1)
            h = self.refine(sphere_pad(h, (1, 1, 1, 1)))
        if self.static_bias is not None:
            h = h + self.static_bias.unsqueeze(0)
        return h


class ScalarForcingEmbedder(nn.Module):
    r"""``(B, scalar_dim)`` calendar + trend scalar → ``(B, d_out, nlat, nlon)``.

    Wraps :class:`~.embedding.CalendarEmbedding` for the periodic part (local
    solar time, day of year) and gives the trend scalar its **own** reserved
    ``d_co2`` channels instead of letting it be one sixth of a 16-wide bundle.

    The scalar arrives z-scored against the training-period statistics — CO₂ or
    the ocean-mean SST anomaly, whichever ``data.scalar_forcing`` selected (the
    ``co2`` attribute names are kept because they are checkpoint keys).
    ``co2_linear`` keeps that head affine so a scenario run outside the
    normalisation range extrapolates monotonically instead of saturating.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        scalar_dim: int,
        d_out: int,
        d_co2: int | None = None,
        co2_linear: bool = True,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.use_co2 = scalar_dim >= 3
        if self.use_co2:
            d_co2 = _round_to(d_out // 2 if d_co2 is None else d_co2, 8, 8)
            if d_co2 >= d_out:
                raise ValueError(
                    f"d_co2={d_co2} must be smaller than d_calendar={d_out}"
                )
        else:
            d_co2 = 0
        self.d_co2 = d_co2
        self.d_time = d_out - d_co2
        self.d_out = d_out

        # NOTE: the fork's CalendarEmbedding spells this kwarg ``use_co2``
        # (upstream renamed it ``use_trend_scalar``); False either way here —
        # the trend scalar gets its own head below.
        self.time = CalendarEmbedding(
            nlon=nlon, nlat=nlat, embed_channels=self.d_time, use_co2=False
        )
        if self.use_co2:
            self.co2 = (
                nn.Linear(1, d_co2)
                if co2_linear
                else nn.Sequential(
                    nn.Linear(1, d_co2), nn.GELU(), nn.Linear(d_co2, d_co2)
                )
            )
        else:
            self.co2 = None

    def forward(self, c_scalar: torch.Tensor) -> torch.Tensor:
        out = self.time(c_scalar[:, :2])  # (B, d_time, nlat, nlon)
        if self.co2 is not None:
            c = self.co2(c_scalar[:, 2:3])  # (B, d_co2)
            c = c[:, :, None, None].expand(-1, -1, out.shape[2], out.shape[3])
            out = torch.cat([out, c], dim=1)
        return out


class ColumnStateEncoder(nn.Module):
    r"""Contract-aware state encoder: ``(B, C, H, W)`` → ``(B, d_out, H, W)``.

    Mirrors the assembly contract ``[surface | diagnostic | upper air
    (level-major, 1000 hPa first)]``. The upper-air block is ``nlevels``
    repetitions of the *same* ``n_upper_air`` physical variables, so it gets
    one shared ``Linear(n_upper_air → d_level)`` applied at every level plus a
    learned per-level embedding saying which pressure the slice came from —
    far fewer parameters than a flat ``130 → d`` projection, and it encodes
    "temperature is temperature at 850 hPa and at 500 hPa" directly.
    """

    def __init__(
        self,
        in_channels: int,
        d_out: int,
        nsurface: int,
        ndiagnostic: int,
        nlevels: int,
        n_upper_air: int,
        d_surface: int | None = None,
        d_diagnostic: int | None = None,
        d_level: int = 16,
    ) -> None:
        super().__init__()
        if nsurface + ndiagnostic + nlevels * n_upper_air != in_channels:
            raise ValueError(
                f"state layout {nsurface}+{ndiagnostic}+{nlevels}*{n_upper_air} "
                f"does not add up to in_channels={in_channels}"
            )
        self.nsurface = nsurface
        self.ndiagnostic = ndiagnostic
        self.nlevels = nlevels
        self.n_upper_air = n_upper_air

        # Default split is proportional to raw channel count, rounded up to a
        # multiple of 8 (already giving the 6 surface channels a bit more than
        # their share). Override either to weight the near-surface fields people
        # actually read — 2 m temperature, precipitation — more heavily.
        d_surface = _round_to(
            d_surface if d_surface is not None else d_out * nsurface / in_channels, 8, 8
        )
        d_diagnostic = _round_to(
            d_diagnostic
            if d_diagnostic is not None
            else d_out * ndiagnostic / in_channels,
            8,
            8,
        )
        d_upper = d_out - d_surface - d_diagnostic
        if d_upper <= 0:
            raise ValueError(
                f"d_surface={d_surface} + d_diagnostic={d_diagnostic} leaves "
                f"nothing of d_state={d_out} for the upper air"
            )
        self.d_surface, self.d_diagnostic, self.d_upper = (
            d_surface,
            d_diagnostic,
            d_upper,
        )

        self.surface_proj = nn.Conv2d(nsurface, d_surface, 1)
        self.diagnostic_proj = nn.Conv2d(ndiagnostic, d_diagnostic, 1)
        # Shared across levels; the level embedding breaks the symmetry.
        self.level_proj = nn.Linear(n_upper_air, d_level)
        self.level_embed = nn.Parameter(torch.zeros(nlevels, d_level))
        self.upper_proj = nn.Conv2d(nlevels * d_level, d_upper, 1)
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ns, nd, L, V = (
            self.nsurface,
            self.ndiagnostic,
            self.nlevels,
            self.n_upper_air,
        )
        sfc = self.surface_proj(x[:, :ns])
        diag = self.diagnostic_proj(x[:, ns : ns + nd])

        ua = rearrange(x[:, ns + nd :], "b (l v) h w -> b l h w v", l=L, v=V)
        ua = self.level_proj(ua) + self.level_embed[None, :, None, None, :]
        ua = F.gelu(ua)
        ua = rearrange(ua, "b l h w d -> b (l d) h w")
        ua = self.upper_proj(ua)

        return torch.cat([sfc, diag, ua], dim=1)


class RollingDiTInputEmbed(nn.Module):
    r"""Budgeted input projection: ``forward(x, c_grid, c_scalar)``.

    Returns ``(tokens, boundary_features)`` where ``tokens`` is
    ``(B, nlat*nlon, dim)`` and ``boundary_features`` is
    ``(B, d_boundary, nlat, nlon)`` (or ``None``) — the key/value stream for
    the causal forcing cross-attention. ``B`` is the already-folded ``b * W``.

    A source absent at call time still occupies its slice, filled with zeros,
    so the token width never depends on what was passed.

    Parameters
    ----------
    d_boundary, d_calendar : int, optional
        Explicit channel budget. ``None`` takes the default (``dim // 4`` and
        ``dim // 8``); an explicit ``0`` turns that source off entirely.
        ``d_state`` is the remainder of ``dim``.
    d_co2 : int, optional
        Reserved channels for the trend scalar inside ``d_calendar``.
    state_encoder : {"flat", "column"}
    boundary_encoder : {"conv1", "conv2"}
    nocean : int, optional, default=0
        Predicted-ocean channels (Phase 12f) at the tail of ``in_channels``.
        They get their own zero-init projection summed into the state slice
        rather than widening the state encoder's input — which keeps every
        existing tensor's shape (including ``ColumnStateEncoder``'s
        ``d_surface`` / ``d_diagnostic`` defaults, computed from its
        ``in_channels``) so a checkpoint trained without ocean channels
        warm-starts with zero skipped keys.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        nlat: int,
        nlon: int,
        c_grid_dim: int = 0,
        scalar_dim: int = 0,
        c_grid_downsample: int = 4,
        d_boundary: int | None = None,
        d_calendar: int | None = None,
        d_co2: int | None = None,
        state_encoder: str = "flat",
        boundary_encoder: str = "conv2",
        boundary_pool_stats: bool = True,
        boundary_static_bias: bool = True,
        co2_linear: bool = True,
        source_norm: bool = True,
        nsurface: int | None = None,
        ndiagnostic: int | None = None,
        nlevels: int | None = None,
        n_upper_air: int | None = None,
        nocean: int = 0,
        d_surface: int | None = None,
        d_diagnostic: int | None = None,
        d_level: int = 16,
    ) -> None:
        super().__init__()
        if state_encoder not in ("flat", "column"):
            raise ValueError(f"unknown state_encoder {state_encoder!r}")
        self.dim = dim
        self.nlat, self.nlon = nlat, nlon
        self.c_grid_dim = c_grid_dim
        self.scalar_dim = scalar_dim

        self.nocean = int(nocean or 0)
        self.n_state = in_channels - self.nocean
        if self.n_state <= 0:
            raise ValueError(
                f"nocean={self.nocean} leaves nothing of in_channels={in_channels}"
            )

        # ── Channel budget ────────────────────────────────────────────────
        self.d_boundary = (
            0
            if c_grid_dim <= 0 or d_boundary == 0
            else _round_to(dim // 4 if d_boundary is None else d_boundary)
        )
        self.d_calendar = (
            0
            if scalar_dim <= 0 or d_calendar == 0
            else _round_to(dim // 8 if d_calendar is None else d_calendar)
        )
        self.d_state = dim - self.d_boundary - self.d_calendar
        if self.d_state <= 0:
            raise ValueError(
                f"d_boundary={self.d_boundary} + d_calendar={self.d_calendar} "
                f"exceeds dim={dim}"
            )

        # ── State ─────────────────────────────────────────────────────────
        # Catch a config that bumped in_channels without setting the ocean
        # variables (or vice versa) HERE, where the numbers are named —
        # otherwise ``flat`` fails as an opaque conv shape error at the first
        # batch and ``column`` blames its own check for someone else's typo.
        if None not in (nsurface, ndiagnostic, nlevels, n_upper_air):
            if nsurface + ndiagnostic + nlevels * n_upper_air != self.n_state:
                raise ValueError(
                    f"state layout {nsurface}+{ndiagnostic}+{nlevels}*"
                    f"{n_upper_air} does not add up to in_channels"
                    f"({in_channels}) - nocean({self.nocean}) = {self.n_state}"
                )
        if state_encoder == "column":
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
                    f"state_encoder 'column' needs {missing} (set from the data config)"
                )
            self.state_embed = ColumnStateEncoder(
                self.n_state,
                self.d_state,
                nsurface,
                ndiagnostic,
                nlevels,
                n_upper_air,
                d_surface=d_surface,
                d_diagnostic=d_diagnostic,
                d_level=d_level,
            )
        else:
            self.state_embed = nn.Conv2d(self.n_state, self.d_state, 1)

        # ── Ocean (Phase 12f) — zero-init via zero_init_last() ────────────
        self.ocean_embed = (
            nn.Conv2d(self.nocean, self.d_state, 1) if self.nocean else None
        )

        # ── Boundary ──────────────────────────────────────────────────────
        self.boundary_embed = (
            BoundaryEncoder(
                c_grid_dim,
                self.d_boundary,
                nlat,
                nlon,
                downsample=c_grid_downsample,
                depth=boundary_encoder,
                pool_stats=boundary_pool_stats,
                static_bias=boundary_static_bias,
            )
            if self.d_boundary > 0
            else None
        )

        # ── Calendar / trend scalar ───────────────────────────────────────
        self.scalar_embed = (
            ScalarForcingEmbedder(
                nlat,
                nlon,
                scalar_dim,
                self.d_calendar,
                d_co2=d_co2,
                co2_linear=co2_linear,
            )
            if self.d_calendar > 0
            else None
        )

        # ── Per-source normalisation ──────────────────────────────────────
        if source_norm:
            self.norm_state = SourceNorm(self.d_state)
            self.norm_boundary = (
                SourceNorm(self.d_boundary) if self.d_boundary else None
            )
            self.norm_calendar = (
                SourceNorm(self.d_calendar) if self.d_calendar else None
            )
        else:
            self.norm_state = self.norm_boundary = self.norm_calendar = None

    # ------------------------------------------------------------------ #

    def zero_init_last(self) -> None:
        """Zero the ocean projection so it contributes nothing at init.

        Called from ``RollingDiT.initialize_weights`` *after* its tree-wide
        ``xavier_uniform_`` pass, which would otherwise overwrite an init done
        in ``__init__`` — turning the intended exact no-op into a random
        perturbation of a warm-started checkpoint.
        """
        if self.ocean_embed is not None:
            nn.init.constant_(self.ocean_embed.weight, 0)
            nn.init.constant_(self.ocean_embed.bias, 0)

    @property
    def kv_dim(self) -> int:
        """Width of the boundary stream exposed to the forcing cross-attention."""
        return self.d_boundary

    def describe(self) -> dict:
        """Budget summary, for logging and the input-embed benchmark."""
        d = {
            "dim": self.dim,
            "d_state": self.d_state,
            "d_boundary": self.d_boundary,
            "d_calendar": self.d_calendar,
        }
        if self.nocean:
            d["nocean"] = self.nocean
        if self.scalar_embed is not None:
            d["d_calendar_time"] = self.scalar_embed.d_time
            d["d_calendar_co2"] = self.scalar_embed.d_co2
        if isinstance(self.state_embed, ColumnStateEncoder):
            d["d_surface"] = self.state_embed.d_surface
            d["d_diagnostic"] = self.state_embed.d_diagnostic
            d["d_upper_air"] = self.state_embed.d_upper
        return d

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        c_grid: torch.Tensor | None = None,
        c_scalar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``x``: ``(B, in_channels, nlat, nlon)``; ``c_grid``: ``(B, c_grid_dim, Hf, Wf)``.

        ``x``'s trailing ``nocean`` channels are the predicted-ocean block;
        they are projected separately and summed into the state slice *before*
        ``SourceNorm``, so the slice's loudness is still set by its budget.
        """
        h = self.state_embed(x[:, : self.n_state])
        if self.ocean_embed is not None:
            h = h + self.ocean_embed(x[:, self.n_state :])
        feats = [h]
        if self.norm_state is not None:
            feats[0] = self.norm_state(feats[0])

        boundary = None
        if self.d_boundary > 0:
            if c_grid is not None:
                boundary = self.boundary_embed(c_grid)
                if self.norm_boundary is not None:
                    boundary = self.norm_boundary(boundary)
            else:
                boundary = x.new_zeros(
                    x.shape[0], self.d_boundary, self.nlat, self.nlon
                )
            feats.append(boundary)

        if self.d_calendar > 0:
            if c_scalar is not None:
                cal = self.scalar_embed(c_scalar)
                if self.norm_calendar is not None:
                    cal = self.norm_calendar(cal)
            else:
                cal = x.new_zeros(x.shape[0], self.d_calendar, self.nlat, self.nlon)
            feats.append(cal)

        tokens = torch.cat(feats, dim=1)  # (B, dim, nlat, nlon)
        tokens = rearrange(tokens, "b c ny nx -> b (ny nx) c")
        return tokens, boundary
