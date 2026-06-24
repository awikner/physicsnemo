# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Channel-routing wrappers for the AMIP diffusion backbones.

The bare backbones (:class:`AmipDiT`, :class:`RollingDiT`, :class:`ERDM`)
take flat channel-stacked tensors. Real recipes consume structured
sample dicts emitted by :class:`ClimateZarrDataset`
(``surface_in``, ``upper_air_in (C, L, H, W)``, ``constant_boundary``,
``varying_boundary``, ``diagnostic``, ``calendar``). The wrappers in this
module bridge the two:

* :class:`AmipDiTWrapper` wraps :class:`AmipDiT` for single-step diffusion
  recipes (SI / SI_X). Per-sample ``pack`` returns
  ``(x, y, c_grid, c_scalar)`` ready for ``scheduler.compute_loss(model, …)``.
* :class:`RollingDiTWrapper` wraps :class:`RollingDiT` for rolling-window
  recipes (RFM). Same ``pack`` API but the leading axis is ``(B, W, …)``.
* :class:`ERDMWrapper` wraps :class:`ERDM` (UNet variant) for ERDM — same
  rolling-window pack shape as :class:`RollingDiTWrapper`.

Each wrapper is a :class:`physicsnemo.Module` so it round-trips through
``.mdlus`` and stays trainable end-to-end. Its ``forward`` delegates
verbatim to the underlying backbone — schedulers (which call ``model(…)``)
work transparently with the wrapper instance.

Channel layout convention (matches upstream amip):

* ``x`` (prognostic state, fed back at next step) =
  ``concat(surface_in, upper_air_in.flatten(C,L), diagnostic)`` along the
  channel axis. ``diagnostic`` is predicted but NOT autoregressed at
  inference — the recipe drops it before feeding back.
* ``c_grid`` = ``concat(constant_boundary, varying_boundary)`` along the
  channel axis. Constant boundaries are broadcast to ``(B, …)`` /
  ``(B, W, …)`` before concat.
* ``c_scalar`` = ``sample["calendar"]`` — the
  ``(second_of_day, day_of_year)`` vector emitted by
  :class:`ClimateZarrDataset` when ``emit_calendar=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module as _PNeMoModule

from .dit import AmipDiT
from .erdm_unet import ERDM
from .rolling_dit import RollingDiT


@dataclass
class MetaData(ModelMetaData):
    """Default ModelMetaData shared by all three wrappers."""

    jit: bool = False
    cuda_graphs: bool = False  # iterative diffusion sampling + dynamic shapes
    amp: bool = False
    amp_gpu: bool = False
    bf16: bool = False
    onnx: bool = False


# ---------------------------------------------------------------------------
# Channel-routing helpers (shared)
# ---------------------------------------------------------------------------


def _broadcast_constant(constant: torch.Tensor, batch_dim_shape: tuple[int, ...]) -> torch.Tensor:
    """Broadcast a constant-boundary tensor across the leading shape.

    Input: ``(C, H, W)`` (no batch dim in the cached sample) or already
    batched ``(B, C, H, W)`` / ``(B, W, C, H, W)``.
    Output: matches ``(*batch_dim_shape, C, H, W)``.
    """
    if constant.ndim == 3:
        # (C, H, W) — expand across the requested leading dims.
        for _ in batch_dim_shape:
            constant = constant.unsqueeze(0)
        return constant.expand(*batch_dim_shape, *constant.shape[-3:])
    return constant


def _flatten_upper_air(upper_air: torch.Tensor) -> torch.Tensor:
    """Reshape ``(B, C_u, L, H, W)`` -> ``(B, C_u * L, H, W)``.

    Or ``(B, W, C_u, L, H, W)`` -> ``(B, W, C_u * L, H, W)`` for rolling.
    """
    leading = upper_air.shape[:-4]
    Cu, L, H, Wd = upper_air.shape[-4:]
    return upper_air.reshape(*leading, Cu * L, H, Wd)


def _unflatten_upper_air(
    flat: torch.Tensor, num_vars: int, num_levels: int
) -> torch.Tensor:
    """Inverse of :func:`_flatten_upper_air`."""
    leading = flat.shape[:-3]
    _, H, Wd = flat.shape[-3:]
    return flat.reshape(*leading, num_vars, num_levels, H, Wd)


# ---------------------------------------------------------------------------
# Single-step wrapper (SI / SI_X) — wraps AmipDiT
# ---------------------------------------------------------------------------


class AmipDiTWrapper(_PNeMoModule):
    r"""Single-step diffusion wrapper around :class:`AmipDiT`.

    Pack / unpack semantics — see the module docstring. The wrapper
    instance is callable with the bare-backbone signature
    ``forward(x_noised, cond, t, c_grid, c_scalar)``, so
    ``scheduler.compute_loss(wrapper, …)`` and
    ``scheduler.sample(wrapper, …)`` work transparently.

    Parameters
    ----------
    surface_variables, upper_air_variables, diagnostic_variables : list[str]
        Prognostic channel names (used for pack/unpack).
    constant_boundary_variables, varying_boundary_variables : list[str]
        Conditioning channel names — concatenated into ``c_grid``.
    levels : list[float]
        Pressure levels (used to size the flattened upper-air block).
    horizontal_resolution : (int, int)
        ``(nlat, nlon)``.
    scalar_dim : int, optional, default=2
        Length of the calendar / c_scalar vector. ``2`` matches
        :meth:`ClimateZarrDataset._calendar_vector`.
    dit_kwargs : dict, optional
        Forwarded to :class:`AmipDiT` (``dim``, ``num_heads``, etc.).
    """

    def __init__(
        self,
        *,
        surface_variables: Sequence[str],
        upper_air_variables: Sequence[str],
        diagnostic_variables: Sequence[str] = (),
        constant_boundary_variables: Sequence[str] = (),
        varying_boundary_variables: Sequence[str] = (),
        levels: Sequence[float],
        horizontal_resolution: Sequence[int],
        scalar_dim: int = 2,
        dit_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.scalar_dim = int(scalar_dim)

        self.num_surface = len(self.surface_variables)
        self.num_upper_air_vars = len(self.upper_air_variables)
        self.num_diagnostic = len(self.diagnostic_variables)
        self.num_levels = len(self.levels)
        self.num_constant_boundary = len(self.constant_boundary_variables)
        self.num_varying_boundary = len(self.varying_boundary_variables)

        self.in_channels = (
            self.num_surface
            + self.num_upper_air_vars * self.num_levels
            + self.num_diagnostic
        )
        self.c_grid_dim = self.num_constant_boundary + self.num_varying_boundary

        nlat, nlon = self.horizontal_resolution

        # AmipDiT's ``in_channels`` is the PatchEmbed channel count and
        # bakes in the [x_noised, cond] concat assumption — see the
        # upstream amip config ``in_channels: 302  # 151*2``. So we pass
        # ``2 * state_channels`` for in_channels and the real
        # ``state_channels`` for out_channels. (When wrapped, the *outer*
        # wrapper's MetaData is what ``Module.save`` / ``from_checkpoint``
        # use; the backbone is a regular submodule.)
        dit_kwargs = dict(dit_kwargs or {})
        dit_kwargs.setdefault("in_channels", 2 * self.in_channels)
        dit_kwargs.setdefault("out_channels", self.in_channels)
        dit_kwargs.setdefault("scalar_dim", self.scalar_dim)
        dit_kwargs.setdefault("c_grid_dim", self.c_grid_dim)
        dit_kwargs.setdefault("nlat", nlat)
        dit_kwargs.setdefault("nlon", nlon)
        self.backbone = AmipDiT(**dit_kwargs)

    # ------------------------------------------------------------------ #
    # Forward — delegates to the backbone so schedulers work transparently.
    # ------------------------------------------------------------------ #

    def forward(self, x_noised, cond, t, c_grid=None, c_scalar=None):
        return self.backbone(x_noised, cond, t, c_grid=c_grid, c_scalar=c_scalar)

    # ------------------------------------------------------------------ #
    # Pack / unpack — recipe-facing helpers.
    # ------------------------------------------------------------------ #

    def pack_state(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        r"""``sample -> x [B, C, H, W]`` (concat surface + upper_air + diag).

        ``sample`` is a single sample dict from
        :class:`ClimateZarrDataset` (no batch dim) OR a batched dict from
        the DataLoader. The leading axes (``B`` or empty) are preserved.
        """
        parts: list[torch.Tensor] = [sample["surface_in"]]
        if self.num_upper_air_vars > 0:
            parts.append(_flatten_upper_air(sample["upper_air_in"]))
        if self.num_diagnostic > 0:
            parts.append(sample["diagnostic"])
        # Channel axis is the third-from-last for batched samples,
        # second-from-last for unbatched. ``cat(dim=-3)`` works for both
        # because surface_in is always shape ``(*B, C, H, W)``.
        return torch.cat(parts, dim=-3)

    def unpack_state(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""``x [B, C, H, W] -> {surface_in, upper_air_in, diagnostic}``."""
        idx = 0
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = x.narrow(-3, idx, self.num_surface)
        idx += self.num_surface
        if self.num_upper_air_vars > 0:
            ua_flat = x.narrow(
                -3, idx, self.num_upper_air_vars * self.num_levels
            )
            out["upper_air_in"] = _unflatten_upper_air(
                ua_flat, self.num_upper_air_vars, self.num_levels
            )
            idx += self.num_upper_air_vars * self.num_levels
        if self.num_diagnostic > 0:
            out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
            idx += self.num_diagnostic
        return out

    def pack_c_grid(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        r"""``sample -> c_grid [B, C_g, H, W]``.

        Constant boundaries are broadcast across the batch axis when the
        cached tensor has no leading ``B``.
        """
        if self.c_grid_dim == 0:
            return None
        surface_in = sample.get("surface_in")
        batch_shape = surface_in.shape[:-3] if surface_in is not None else ()
        parts: list[torch.Tensor] = []
        if self.num_constant_boundary > 0:
            const = _broadcast_constant(sample["constant_boundary"], batch_shape)
            parts.append(const)
        if self.num_varying_boundary > 0:
            parts.append(sample["varying_boundary"])
        return torch.cat(parts, dim=-3)


# ---------------------------------------------------------------------------
# Rolling-window wrappers (RFM, ERDM)
# ---------------------------------------------------------------------------


class _RollingPackUnpackMixin:
    """Shared pack/unpack for rolling backbones."""

    def pack_window_state(
        self, window_sample: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        r"""``window_sample -> y [B, W, C, H, W]`` (rolling-window pack).

        Input fields are expected to be already stacked along a leading
        ``window`` axis — i.e. ``surface_in.shape == (B, W, C_s, H, W)``,
        ``upper_air_in.shape == (B, W, C_u, L, H, W)``. The
        :class:`SequenceDataset` helper in
        :mod:`physicsnemo.experimental.datapipes.climate.sequence` produces
        this layout from per-frame samples.
        """
        parts: list[torch.Tensor] = [window_sample["surface_in"]]
        if self.num_upper_air_vars > 0:
            parts.append(_flatten_upper_air(window_sample["upper_air_in"]))
        if self.num_diagnostic > 0:
            parts.append(window_sample["diagnostic"])
        return torch.cat(parts, dim=-3)

    def unpack_window_state(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""``x [B, W, C, H, W] -> {surface_in, upper_air_in, diagnostic}``."""
        idx = 0
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = x.narrow(-3, idx, self.num_surface)
        idx += self.num_surface
        if self.num_upper_air_vars > 0:
            ua_flat = x.narrow(
                -3, idx, self.num_upper_air_vars * self.num_levels
            )
            out["upper_air_in"] = _unflatten_upper_air(
                ua_flat, self.num_upper_air_vars, self.num_levels
            )
            idx += self.num_upper_air_vars * self.num_levels
        if self.num_diagnostic > 0:
            out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
            idx += self.num_diagnostic
        return out

    def pack_window_c_grid(
        self, window_sample: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        r"""``window_sample -> c_grid [B, W, C_g, H, W]``."""
        if self.c_grid_dim == 0:
            return None
        surface_in = window_sample.get("surface_in")
        batch_shape = surface_in.shape[:-3] if surface_in is not None else ()
        parts: list[torch.Tensor] = []
        if self.num_constant_boundary > 0:
            const = _broadcast_constant(
                window_sample["constant_boundary"], batch_shape
            )
            parts.append(const)
        if self.num_varying_boundary > 0:
            parts.append(window_sample["varying_boundary"])
        return torch.cat(parts, dim=-3)


class RollingDiTWrapper(_PNeMoModule, _RollingPackUnpackMixin):
    r"""Rolling-window diffusion wrapper around :class:`RollingDiT`.

    Same channel-group bookkeeping as :class:`AmipDiTWrapper` but the pack
    operates on ``(B, W, ...)`` window samples (drive via
    :class:`SequenceDataset`).
    """

    def __init__(
        self,
        *,
        surface_variables: Sequence[str],
        upper_air_variables: Sequence[str],
        diagnostic_variables: Sequence[str] = (),
        constant_boundary_variables: Sequence[str] = (),
        varying_boundary_variables: Sequence[str] = (),
        levels: Sequence[float],
        horizontal_resolution: Sequence[int],
        scalar_dim: int = 2,
        rolling_dit_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.scalar_dim = int(scalar_dim)

        self.num_surface = len(self.surface_variables)
        self.num_upper_air_vars = len(self.upper_air_variables)
        self.num_diagnostic = len(self.diagnostic_variables)
        self.num_levels = len(self.levels)
        self.num_constant_boundary = len(self.constant_boundary_variables)
        self.num_varying_boundary = len(self.varying_boundary_variables)

        self.in_channels = (
            self.num_surface
            + self.num_upper_air_vars * self.num_levels
            + self.num_diagnostic
        )
        self.c_grid_dim = self.num_constant_boundary + self.num_varying_boundary

        nlat, nlon = self.horizontal_resolution

        rolling_dit_kwargs = dict(rolling_dit_kwargs or {})
        # RollingDiT only takes ``x_noised`` (no separate ``cond``) so the
        # PatchEmbed in_channels equals state_channels (unlike AmipDiT).
        rolling_dit_kwargs.setdefault("in_channels", self.in_channels)
        rolling_dit_kwargs.setdefault("out_channels", self.in_channels)
        rolling_dit_kwargs.setdefault("scalar_dim", self.scalar_dim)
        rolling_dit_kwargs.setdefault("c_grid_dim", self.c_grid_dim)
        rolling_dit_kwargs.setdefault("nlat", nlat)
        rolling_dit_kwargs.setdefault("nlon", nlon)
        # Default to no spatial downsampling — recipes that want a smaller
        # latent grid can override with c_grid_downsample > 1.
        rolling_dit_kwargs.setdefault("c_grid_downsample", 1)
        self.backbone = RollingDiT(**rolling_dit_kwargs)

    def forward(self, z, t, c_grid=None, c_scalar=None):
        return self.backbone(z, t, c_grid=c_grid, c_scalar=c_scalar)


class ERDMWrapper(_PNeMoModule, _RollingPackUnpackMixin):
    r"""Rolling-window diffusion wrapper around :class:`ERDM` (UNet variant)."""

    def __init__(
        self,
        *,
        surface_variables: Sequence[str],
        upper_air_variables: Sequence[str],
        diagnostic_variables: Sequence[str] = (),
        constant_boundary_variables: Sequence[str] = (),
        varying_boundary_variables: Sequence[str] = (),
        levels: Sequence[float],
        horizontal_resolution: Sequence[int],
        scalar_dim: int = 2,
        erdm_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.scalar_dim = int(scalar_dim)

        self.num_surface = len(self.surface_variables)
        self.num_upper_air_vars = len(self.upper_air_variables)
        self.num_diagnostic = len(self.diagnostic_variables)
        self.num_levels = len(self.levels)
        self.num_constant_boundary = len(self.constant_boundary_variables)
        self.num_varying_boundary = len(self.varying_boundary_variables)

        self.in_channels = (
            self.num_surface
            + self.num_upper_air_vars * self.num_levels
            + self.num_diagnostic
        )
        self.c_grid_dim = self.num_constant_boundary + self.num_varying_boundary

        nlat, nlon = self.horizontal_resolution

        erdm_kwargs = dict(erdm_kwargs or {})
        erdm_kwargs.setdefault("in_channels", self.in_channels)
        erdm_kwargs.setdefault("out_channels", self.in_channels)
        erdm_kwargs.setdefault("scalar_dim", self.scalar_dim)
        erdm_kwargs.setdefault("c_grid_dim", self.c_grid_dim)
        erdm_kwargs.setdefault("nlat", nlat)
        erdm_kwargs.setdefault("nlon", nlon)
        # ERDM additionally needs nlat_work / nlon_work (interpolation grid).
        # Default to the same resolution; recipes that want the working
        # grid to differ can pass it through erdm_kwargs.
        erdm_kwargs.setdefault("nlat_work", nlat)
        erdm_kwargs.setdefault("nlon_work", nlon)
        self.backbone = ERDM(**erdm_kwargs)

    def forward(self, x_noised, c_noise, c_grid=None, c_scalar=None):
        return self.backbone(x_noised, c_noise, c_grid=c_grid, c_scalar=c_scalar)


__all__ = [
    "AmipDiTWrapper",
    "ERDMWrapper",
    "RollingDiTWrapper",
]
