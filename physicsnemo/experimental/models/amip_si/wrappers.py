# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
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
from .layers.bilinear import BilinearDecoder, BilinearEncoder
from .rolling_dit import RollingDiT
from .dit_ae import DiTAE
from .x_ddc import XDDCUNet


@dataclass
class MetaData(ModelMetaData):
    """Default ModelMetaData shared by all three wrappers.

    Phase 8f (F3) flips ``amp``/``bf16`` to ``True`` — the fp32-vs-bf16
    benchmark (``benchmarks/physicsnemo/experimental/models/amip_si/RESULTS.md``)
    confirms bf16 autocast training doesn't tank convergence vs. fp32.
    ``amp_gpu``/``amp_cpu`` are left unset (base class default ``None``)
    so :meth:`ModelMetaData.__post_init__` derives them from ``amp``
    instead of hardcoding them off. ``cuda_graphs`` stays ``False``
    permanently — the diffusion loop's iterative ``sample()`` is not
    CUDA-graph friendly (dynamic step counts + host-side control flow).
    """

    jit: bool = False
    cuda_graphs: bool = False  # iterative diffusion sampling + dynamic shapes
    amp: bool = True
    bf16: bool = True
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
# Channel layouts (Phase 12b) — see docs/dev/phase12_implementation_plan.md.
#
# ``"fork"``  the Phase-8 fork order: state ``[surface | upper_air | diag]``
#             with the upper-air block variable-major in config level order,
#             c_grid ``[constant | varying]``. No upstream checkpoint was
#             trained on this order — it exists for fork-trained artifacts
#             and the frozen v1-family wrappers.
# ``"v1"``    upstream amip v1 (``common/utils.py @ 497827e``): state
#             ``[surface | diagnostic | upper_air]`` with the upper-air
#             block variable-major in config level order, c_grid
#             ``[varying | constant]`` (``assemble_forcing(forcing,
#             invariant)``). Real v1-trained checkpoints expect this.
# ``"v2"``    upstream amip_v2 (@ ``e0b7b60``): same group order as v1 but
#             the upper-air block is LEVEL-MAJOR with the level axis
#             flipped so 1000 hPa leads::
#
#                 rearrange(ua.flip(level_axis), "... c l h w -> ... (l c) h w")
# ---------------------------------------------------------------------------

_CHANNEL_LAYOUTS = ("fork", "v1", "v2")


def _flatten_upper_air_v2(upper_air: torch.Tensor) -> torch.Tensor:
    """amip_v2 upper-air pack: level-major, level axis flipped.

    ``(*B, C_u, L, H, W) -> (*B, L * C_u, H, W)`` — bit-parity with
    upstream ``assemble_input``'s
    ``rearrange(multilevel.flip(2), "b c l h w -> b (l c) h w")``.
    """
    leading = upper_air.shape[:-4]
    Cu, L, H, Wd = upper_air.shape[-4:]
    flipped = upper_air.flip(-3)
    return flipped.transpose(-4, -3).reshape(*leading, L * Cu, H, Wd)


def _unflatten_upper_air_v2(
    flat: torch.Tensor, num_vars: int, num_levels: int
) -> torch.Tensor:
    """Inverse of :func:`_flatten_upper_air_v2` (back to config level order)."""
    leading = flat.shape[:-3]
    _, H, Wd = flat.shape[-3:]
    stacked = flat.reshape(*leading, num_levels, num_vars, H, Wd)
    return stacked.transpose(-4, -3).flip(-3)


def _validate_channel_layout(layout: str, allowed: tuple[str, ...]) -> str:
    if layout not in allowed:
        raise ValueError(
            f"channel_layout={layout!r} not supported; expected one of {allowed}"
        )
    return layout


# ---------------------------------------------------------------------------
# Muon param-group helper (shared) — see F1 in phase8f_completion_plan.md.
# ---------------------------------------------------------------------------


def _muon_groups(
    muon_weights: list[torch.nn.Parameter],
    adamw_params: list[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    muon_lr_multiplier: float,
    adam_betas: tuple[float, float],
) -> list[dict]:
    """Assemble the two-group list consumed by ``muon.MuonWithAuxAdam``.

    Matches upstream amip's convention (``modules/train_module.py``):
    the Muon group runs at ``lr * muon_lr_multiplier``; the aux-AdamW
    group runs at the base ``lr`` with ``betas=adam_betas``.
    """
    return [
        dict(
            params=muon_weights,
            use_muon=True,
            lr=lr * muon_lr_multiplier,
            weight_decay=weight_decay,
        ),
        dict(
            params=adamw_params,
            use_muon=False,
            lr=lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        ),
    ]


# ---------------------------------------------------------------------------
# Single-step wrapper (SI / SI_X) — wraps AmipDiT
# ---------------------------------------------------------------------------


class AmipDiTWrapper(_PNeMoModule):
    r"""Single-step diffusion wrapper around :class:`AmipDiT`.

    .. note:: **Frozen on the amip-v1 contract (Phase 12).** Upstream
        amip_v2 deleted the single-step families (SI / SI_X / EDM); this
        wrapper receives no amip_v2 (``"v2"``) layout. It does accept a
        ``channel_layout`` kwarg with two v1-era contracts (a Phase 12b
        correctness fix):

        * ``"fork"`` (default) — the historical Phase-8 fork packing:
          state ``[surface | upper_air | diag]``, c_grid
          ``[constant | varying]``. No upstream checkpoint was trained
          on this order.
        * ``"v1"`` — upstream amip v1's real training contract: state
          ``[surface | diag | upper_air]`` (upper-air variable-major in
          config level order), c_grid ``[varying | constant]``. **Use
          this for checkpoints translated from real v1 Lightning ckpts**
          (``--source-contract v1``) — under ``"fork"`` their
          channel-indexed projections see permuted channels.

        See ``docs/dev/phase12_implementation_plan.md`` ("dual-contract
        seam").

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
        scalar_routed_boundary_variables: Sequence[str] = (),
        levels: Sequence[float],
        horizontal_resolution: Sequence[int],
        scalar_dim: int = 2,
        channel_layout: str = "fork",
        dit_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        # Frozen family: only the two v1-era contracts, never "v2".
        self.channel_layout = _validate_channel_layout(
            channel_layout, ("fork", "v1")
        )
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
        # Channels the ForcingAssembler pops out of the gridded stream and
        # appends to the calendar row. Same contract as RollingDiTWrapper's
        # (added 2026-08-14): a SUBSET of ``varying_boundary_variables`` — the
        # stored order, which the NaN-fill still needs in full — subtracted from
        # ``c_grid_dim`` so the backbone is sized for what actually arrives.
        #
        # Required by the real v1 ``wCO2`` checkpoints: their config lists 4
        # varying channels while the backbone's ``c_grid_embed`` is
        # ``Conv2d(5 -> N)`` = 2 constant + 3 gridded, because
        # ``global_mean_co2`` rides the calendar row instead (``scalar_dim: 3``).
        # Without this the wrapper sizes c_grid_dim at 6 and the checkpoint
        # cannot load. Additive: an empty list reproduces the previous
        # arithmetic exactly, so the frozen configs are untouched.
        self.scalar_routed_boundary_variables = list(
            scalar_routed_boundary_variables
        )
        unknown = [
            v
            for v in self.scalar_routed_boundary_variables
            if v not in self.varying_boundary_variables
        ]
        if unknown:
            raise ValueError(
                f"scalar_routed_boundary_variables {unknown} are not in "
                f"varying_boundary_variables {self.varying_boundary_variables}"
            )
        self.num_varying_boundary = len(self.varying_boundary_variables) - len(
            self.scalar_routed_boundary_variables
        )

        self.in_channels = (
            self.num_surface
            + self.num_upper_air_vars * self.num_levels
            + self.num_diagnostic
        )
        self.c_grid_dim = self.num_constant_boundary + self.num_varying_boundary

        if self.scalar_routed_boundary_variables:
            # Every calendar encoding this datapipe emits is width 2, so a
            # routed channel has to be paid for with a matching scalar_dim.
            # Raising here keeps the model config and the assembler — which read
            # the same two keys — from drifting apart.
            expected = 2 + len(self.scalar_routed_boundary_variables)
            if self.scalar_dim != expected:
                raise ValueError(
                    f"scalar_dim={self.scalar_dim} does not match "
                    f"2 (calendar) + {len(self.scalar_routed_boundary_variables)} "
                    f"routed boundary channel(s) = {expected}; set "
                    f"scalar_dim: {expected} in the model config"
                )

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
        # ``c_grid_downsample=1`` keeps the c_grid embedding at the same
        # spatial resolution as ``x_noised`` so the cat at AmipDiT.forward
        # line 461 aligns. Upstream amip uses ``c_grid_downsample=4`` paired
        # with a recipe-side pre-downsample of ``x_noised + cond`` to the
        # patch grid; our recipe doesn't do that, so we keep both streams
        # at native res.
        dit_kwargs.setdefault("c_grid_downsample", 1)
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
        # Group order is layout-dependent (Phase 12b correctness fix):
        # "fork" = [surface | upper_air | diag]; "v1" = upstream's
        # [surface | diag | upper_air]. The upper-air block is
        # variable-major in config level order under both.
        parts: list[torch.Tensor] = [sample["surface_in"]]
        if self.channel_layout == "fork":
            if self.num_upper_air_vars > 0:
                parts.append(_flatten_upper_air(sample["upper_air_in"]))
            if self.num_diagnostic > 0:
                parts.append(sample["diagnostic"])
        else:  # "v1"
            if self.num_diagnostic > 0:
                parts.append(sample["diagnostic"])
            if self.num_upper_air_vars > 0:
                parts.append(_flatten_upper_air(sample["upper_air_in"]))
        # Channel axis is the third-from-last for batched samples,
        # second-from-last for unbatched. ``cat(dim=-3)`` works for both
        # because surface_in is always shape ``(*B, C, H, W)``.
        return torch.cat(parts, dim=-3)

    def unpack_state(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""``x [B, C, H, W] -> {surface_in, upper_air_in, diagnostic}``."""
        n_ua = self.num_upper_air_vars * self.num_levels
        idx = 0
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = x.narrow(-3, idx, self.num_surface)
        idx += self.num_surface
        if self.channel_layout == "fork":
            if self.num_upper_air_vars > 0:
                out["upper_air_in"] = _unflatten_upper_air(
                    x.narrow(-3, idx, n_ua),
                    self.num_upper_air_vars,
                    self.num_levels,
                )
                idx += n_ua
            if self.num_diagnostic > 0:
                out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
                idx += self.num_diagnostic
        else:  # "v1"
            if self.num_diagnostic > 0:
                out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
                idx += self.num_diagnostic
            if self.num_upper_air_vars > 0:
                out["upper_air_in"] = _unflatten_upper_air(
                    x.narrow(-3, idx, n_ua),
                    self.num_upper_air_vars,
                    self.num_levels,
                )
                idx += n_ua
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
        const = None
        if self.num_constant_boundary > 0:
            const = _broadcast_constant(sample["constant_boundary"], batch_shape)
        if self.channel_layout == "fork":
            if const is not None:
                parts.append(const)
            if self.num_varying_boundary > 0:
                parts.append(sample["varying_boundary"])
        else:  # "v1" — upstream assemble_forcing(forcing, invariant)
            if self.num_varying_boundary > 0:
                parts.append(sample["varying_boundary"])
            if const is not None:
                parts.append(const)
        return torch.cat(parts, dim=-3)

    def muon_param_groups(
        self,
        *,
        lr: float,
        weight_decay: float = 0.01,
        muon_lr_multiplier: float = 10.0,
        adam_betas: tuple[float, float] = (0.9, 0.95),
    ) -> list[dict]:
        r"""Split :class:`AmipDiT` parameters into Muon vs. aux-AdamW groups.

        Mirrors upstream amip's ``get_dit_muon_param_groups()``. The
        ``>=2D`` weight matrices of the self-/cross-attention DiT blocks
        (``backbone.sa_blocks``) go to Muon; block biases/norms plus
        *all* parameters of the patch embed, time embedder, unpatchify
        head, and the optional c_grid / scalar / cross-attention context
        embedders go to aux AdamW.

        Returns a two-entry list of ``dict(params=..., use_muon=...)``
        consumable by ``muon.MuonWithAuxAdam(param_groups)``.
        """
        block_params = list(self.backbone.sa_blocks.parameters())
        hidden_weights = [p for p in block_params if p.ndim >= 2]
        hidden_gains_biases = [p for p in block_params if p.ndim < 2]

        nonhidden_modules = [
            self.backbone.patch_embed_main,
            self.backbone.t_embedder,
            self.backbone.unpatchify_layer,
        ]
        if self.backbone.c_grid_embed is not None:
            nonhidden_modules.append(self.backbone.c_grid_embed)
        if self.backbone.scalar_embedder is not None:
            nonhidden_modules.append(self.backbone.scalar_embedder)
        if self.backbone.ca_embed is not None:
            nonhidden_modules.append(self.backbone.ca_embed)
        nonhidden_params = [p for m in nonhidden_modules for p in m.parameters()]

        return _muon_groups(
            hidden_weights,
            hidden_gains_biases + nonhidden_params,
            lr=lr,
            weight_decay=weight_decay,
            muon_lr_multiplier=muon_lr_multiplier,
            adam_betas=adam_betas,
        )


# ---------------------------------------------------------------------------
# Rolling-window wrappers (RFM, ERDM)
# ---------------------------------------------------------------------------


class _RollingPackUnpackMixin:
    """Shared pack/unpack for rolling backbones.

    Layout-aware (Phase 12b): the ``channel_layout`` attribute selects
    the packing contract (see ``_CHANNEL_LAYOUTS``). The class default
    is ``"fork"`` — the historical Phase-8 order — so the frozen
    :class:`ERDMWrapper` is bit-identical to its pre-12b behavior;
    :class:`RollingDiTWrapper` exposes it as a constructor kwarg
    defaulting to ``"v2"``.
    """

    channel_layout: str = "fork"
    #: Predicted ocean channels (Phase 12f). 0 for the frozen families;
    #: :class:`RollingDiTWrapper` derives it from ``ocean_state_variables``.
    num_ocean: int = 0

    def state_layout(self) -> dict[str, int]:
        r"""Block sizes of the packed channel axis (upstream ``state_layout``).

        Mirrors amip_v2 ``common/utils.py:state_layout`` so a
        contract-aware projection (Phase 12e) can address the blocks.
        ``nocean`` (Phase 12f) counts the predicted ocean channels, which
        occupy a block at the TAIL of the state axis — after ``diagnostic``
        / ``upper_air``, so the state blocks keep their offsets and a
        ``nocean=0`` checkpoint's projections stay addressable.
        """
        return {
            "nsurface": self.num_surface,
            "ndiagnostic": self.num_diagnostic,
            "nlevels": self.num_levels,
            "n_upper_air": self.num_upper_air_vars,
            "nocean": self.num_ocean,
        }

    @property
    def forcing_lag(self) -> int:
        r"""Frames by which the forcing window lags the state window.

        ``0`` for ``"fork"`` (own-time forcing — the Phase-8 convention the
        frozen configs are pinned to), ``1`` for ``"v1"`` / ``"v2"``, where
        upstream aligns window slot ``w`` (the state at step ``w+1``) with
        the forcing at step ``w``: *denoising the frame at time T uses the
        boundary forcing from T-1*.

        Derived from the layout rather than configured, because the two
        conventions differ by a whole-window shift — every tensor keeps its
        shape, so a mismatch trains silently against the wrong forcings. Read
        by the recipe when it builds
        :class:`~physicsnemo.experimental.datapipes.climate.SequenceDataset`.
        """
        return 0 if self.channel_layout == "fork" else 1

    def _flatten_ua(self, upper_air: torch.Tensor) -> torch.Tensor:
        if self.channel_layout == "v2":
            return _flatten_upper_air_v2(upper_air)
        return _flatten_upper_air(upper_air)

    def _unflatten_ua(self, flat: torch.Tensor) -> torch.Tensor:
        if self.channel_layout == "v2":
            return _unflatten_upper_air_v2(
                flat, self.num_upper_air_vars, self.num_levels
            )
        return _unflatten_upper_air(
            flat, self.num_upper_air_vars, self.num_levels
        )

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

        Group order is layout-dependent: ``[surface | upper_air | diag]``
        for ``"fork"``, ``[surface | diag | upper_air]`` for ``"v1"`` /
        ``"v2"`` (upstream ``assemble_input``).
        """
        parts: list[torch.Tensor] = [window_sample["surface_in"]]
        if self.channel_layout == "fork":
            if self.num_upper_air_vars > 0:
                parts.append(self._flatten_ua(window_sample["upper_air_in"]))
            if self.num_diagnostic > 0:
                parts.append(window_sample["diagnostic"])
        else:  # "v1" / "v2" — upstream group order
            if self.num_diagnostic > 0:
                parts.append(window_sample["diagnostic"])
            if self.num_upper_air_vars > 0:
                parts.append(self._flatten_ua(window_sample["upper_air_in"]))
        return torch.cat(parts, dim=-3)

    def unpack_window_state(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""``x [B, W, C, H, W] -> {surface_in, upper_air_in, diagnostic}``.

        Under ``num_ocean > 0`` the trailing ocean block is dropped: every
        block is read at its own width from the front, so the tail simply
        falls off the end. Equivalent to
        :meth:`~physicsnemo.experimental.diffusion.erdm.ERDMScheduler.strip_ocean`
        for unpack purposes — the predicted SST is a diagnostic byproduct,
        while the *forcing* SST that the writers record comes from the
        boundary data.
        """
        n_ua = self.num_upper_air_vars * self.num_levels
        idx = 0
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = x.narrow(-3, idx, self.num_surface)
        idx += self.num_surface
        if self.channel_layout == "fork":
            if self.num_upper_air_vars > 0:
                out["upper_air_in"] = self._unflatten_ua(x.narrow(-3, idx, n_ua))
                idx += n_ua
            if self.num_diagnostic > 0:
                out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
                idx += self.num_diagnostic
        else:  # "v1" / "v2"
            if self.num_diagnostic > 0:
                out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
                idx += self.num_diagnostic
            if self.num_upper_air_vars > 0:
                out["upper_air_in"] = self._unflatten_ua(x.narrow(-3, idx, n_ua))
                idx += n_ua
        return out

    #: Frameless alias: every block is read at ``narrow(-3, ...)`` and the
    #: upper-air unflatten keys off ``shape[:-3]``, so the same code unpacks a
    #: single ``(B, C, H, W)`` frame. The rolling drivers
    #: (``validate_diffusion.py``, ``inference.py``) score one emitted frame at
    #: a time and call it under this name, which the rolling wrappers did not
    #: previously define — an ``AttributeError`` on the first scored frame of
    #: any rolling validation or inference run (fixed alongside Phase 12f,
    #: whose ocean tail this call is also what drops).
    unpack_state = unpack_window_state

    def pack_window_c_grid(
        self, window_sample: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        r"""``window_sample -> c_grid [B, W, C_g, H, W]``.

        Order is layout-dependent: ``[constant | varying]`` for
        ``"fork"``; ``[varying | constant]`` for ``"v1"`` / ``"v2"``
        (upstream ``assemble_forcing(forcing, invariant)``).
        """
        if self.c_grid_dim == 0:
            return None
        surface_in = window_sample.get("surface_in")
        batch_shape = surface_in.shape[:-3] if surface_in is not None else ()
        parts: list[torch.Tensor] = []
        const = None
        if self.num_constant_boundary > 0:
            const = _broadcast_constant(
                window_sample["constant_boundary"], batch_shape
            )
        if self.channel_layout == "fork":
            if const is not None:
                parts.append(const)
            if self.num_varying_boundary > 0:
                parts.append(window_sample["varying_boundary"])
        else:  # "v1" / "v2" — forcing (varying) first, invariant (constant) last
            if self.num_varying_boundary > 0:
                parts.append(window_sample["varying_boundary"])
            if const is not None:
                parts.append(const)
        return torch.cat(parts, dim=-3)


class RollingDiTWrapper(_PNeMoModule, _RollingPackUnpackMixin):
    r"""Rolling-window diffusion wrapper around :class:`RollingDiT`.

    .. note:: **Channel layout (Phase 12b).** ``channel_layout`` selects
        the packing contract and travels with the ``.mdlus`` args:

        * ``"v2"`` (default) — upstream amip_v2: state ``[surface | diag |
          upper_air]`` with the upper-air block level-major, 1000 hPa
          first; c_grid ``[varying | constant]``.
        * ``"v1"`` — upstream amip v1: same group order, upper-air
          variable-major in config level order. Use for checkpoints
          translated from real v1 Lightning ckpts
          (``--source-contract v1``).
        * ``"fork"`` — the Phase-8 fork order (``[surface | upper_air |
          diag]``, c_grid ``[constant | varying]``). Pinned by the frozen
          ``conf/model/amip_rfm.yaml``; no upstream checkpoint matches it.

    .. note:: **Scalar-routed forcings (Phase 12d).**
        ``scalar_routed_boundary_variables`` names channels that
        :class:`~physicsnemo.experimental.datapipes.climate.ForcingAssembler`
        pops out of ``varying_boundary`` and appends to the calendar row
        (upstream amip_v2's CO2 handling: ``c_grid_dim`` 6 → 5,
        ``scalar_dim`` 2 → 3). List them here as a subset of
        ``varying_boundary_variables`` — ``c_grid_dim`` is reduced
        accordingly and ``scalar_dim`` is validated against
        ``2 + n_routed``, so a config that sizes the model differently from
        the data pipeline fails at construction instead of silently
        mis-packing.

    .. note:: **Predicted ocean channels (Phase 12f).**
        ``ocean_state_variables`` names gridded forcings (SST, sea ice) the
        forecaster should *also predict*. They widen ``in_channels`` /
        ``out_channels`` by a block at the TAIL of the state axis, and
        :attr:`ocean_grid_indices` says where in the assembled ``c_grid``
        their truth is read from — derived here, from the same variable
        lists that build the pack, so the training target and the
        inference-imposed field cannot disagree. Requires a varying-first
        c_grid layout (``"v1"`` / ``"v2"``) and a non-legacy
        input_embed/output_head; both are checked at construction.

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
        channel_layout: str = "v2",
        scalar_routed_boundary_variables: Sequence[str] = (),
        ocean_state_variables: Sequence[str] = (),
        rolling_dit_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        self.channel_layout = _validate_channel_layout(
            channel_layout, _CHANNEL_LAYOUTS
        )
        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.scalar_dim = int(scalar_dim)
        # Phase 12d.13: channels the ForcingAssembler pops out of the gridded
        # stream and appends to the calendar row. They are listed here as a
        # SUBSET of ``varying_boundary_variables`` (the stored order — the
        # NaN-fill still needs the full list) and are subtracted from
        # ``c_grid_dim`` so the backbone is sized for what actually arrives.
        self.scalar_routed_boundary_variables = list(
            scalar_routed_boundary_variables
        )
        unknown = [
            v
            for v in self.scalar_routed_boundary_variables
            if v not in self.varying_boundary_variables
        ]
        if unknown:
            raise ValueError(
                f"scalar_routed_boundary_variables {unknown} are not in "
                f"varying_boundary_variables {self.varying_boundary_variables}"
            )

        self.num_surface = len(self.surface_variables)
        self.num_upper_air_vars = len(self.upper_air_variables)
        self.num_diagnostic = len(self.diagnostic_variables)
        self.num_levels = len(self.levels)
        self.num_constant_boundary = len(self.constant_boundary_variables)
        self.num_varying_boundary = len(self.varying_boundary_variables) - len(
            self.scalar_routed_boundary_variables
        )

        # Phase 12f: predicted ocean channels. Validated against the *active*
        # varying names (scalar-routed channels removed) because that is the
        # stream that reaches c_grid — asking to predict ``global_mean_co2``
        # after it has been popped into the calendar row has no field to read.
        self.ocean_state_variables = list(ocean_state_variables)
        self.num_ocean = len(self.ocean_state_variables)
        self._validate_ocean_state_variables()

        self.num_state_channels = (
            self.num_surface
            + self.num_upper_air_vars * self.num_levels
            + self.num_diagnostic
        )
        # ``in_channels`` is the model's channel width: state + ocean tail.
        self.in_channels = self.num_state_channels + self.num_ocean
        self.c_grid_dim = self.num_constant_boundary + self.num_varying_boundary

        if self.scalar_routed_boundary_variables:
            # Every calendar encoding this datapipe emits is width 2
            # (second_of_day, day_of_year) or (month, hour), so a routed
            # channel must be paid for with a matching scalar_dim. Failing
            # loudly here is what keeps the model config and the assembler
            # from drifting apart (they read the same two config keys).
            expected = 2 + len(self.scalar_routed_boundary_variables)
            if self.scalar_dim != expected:
                raise ValueError(
                    f"scalar_dim={self.scalar_dim} does not match "
                    f"2 (calendar) + {len(self.scalar_routed_boundary_variables)} "
                    f"routed boundary channel(s) = {expected}; set "
                    f"scalar_dim: {expected} in the model config"
                )

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
        # Phase 12e: the contract-aware projections (``state_encoder: column``,
        # ``decoder: column``) need the state block sizes. DERIVE them from the
        # wrapper's own variable lists rather than have a config restate them —
        # the upstream ``state_layout`` lesson, so the two can never drift.
        rolling_dit_kwargs.setdefault("state_layout", self.state_layout())
        self.backbone = RollingDiT(**rolling_dit_kwargs)

    # -- Predicted ocean channels (Phase 12f) --------------------------------

    @property
    def active_varying_boundary_variables(self) -> list[str]:
        r"""Varying-boundary names in the order they reach ``c_grid``.

        The stored list minus the scalar-routed channels (upstream
        ``grid_forcing_names``). This is the ordering
        :class:`~physicsnemo.experimental.datapipes.climate.ForcingAssembler`
        emits, so it is what indexes the assembled forcing stream.
        """
        routed = set(self.scalar_routed_boundary_variables)
        return [v for v in self.varying_boundary_variables if v not in routed]

    @property
    def ocean_grid_indices(self) -> list[int]:
        r"""Channel indices of the ocean variables within ``c_grid``.

        Mirrors upstream ``AMIPDataset.ocean_grid_indices``. Valid against
        the assembled ``c_grid`` as well as the bare varying stream because
        the ``"v1"`` / ``"v2"`` c_grid layout puts varying first, so the
        varying channels are a prefix — which is exactly why
        :meth:`_validate_ocean_state_variables` rejects ``"fork"``.

        Derived, never configured: the scheduler reads the training target
        and the inference-imposed field through this one list.
        """
        names = self.active_varying_boundary_variables
        return [names.index(v) for v in self.ocean_state_variables]

    def _validate_ocean_state_variables(self) -> None:
        """Fail at construction, not mid-epoch, on an unreachable channel."""
        if not self.num_ocean:
            return
        if self.channel_layout == "fork":
            raise ValueError(
                "ocean_state_variables needs a varying-first c_grid layout so "
                "the forcing indices also address the assembled c_grid; "
                'channel_layout="fork" puts the constant boundary first. Use '
                'channel_layout="v2".'
            )
        if self.num_varying_boundary == 0:
            raise ValueError(
                "ocean_state_variables needs gridded forcings to read the "
                "truth from, but no varying boundary channels reach c_grid"
            )
        names = self.active_varying_boundary_variables
        for v in self.ocean_state_variables:
            if v in names:
                continue
            if v in self.scalar_routed_boundary_variables:
                raise ValueError(
                    f"ocean_state_variables cannot contain {v!r}: it is "
                    f"scalar-routed (popped out of the gridded forcings into "
                    f"the calendar row), so there is no field to predict"
                )
            raise ValueError(
                f"ocean_state_variables entry {v!r} is not one of the gridded "
                f"forcing channels {names}"
            )
        dupes = [
            v for v in set(self.ocean_state_variables)
            if self.ocean_state_variables.count(v) > 1
        ]
        if dupes:
            raise ValueError(
                f"ocean_state_variables has duplicate entries {sorted(dupes)}; "
                f"each predicted ocean channel needs its own state channel"
            )

    def forward(self, z, t, c_grid=None, c_scalar=None):
        return self.backbone(z, t, c_grid=c_grid, c_scalar=c_scalar)

    def muon_param_groups(
        self,
        *,
        lr: float,
        weight_decay: float = 0.01,
        muon_lr_multiplier: float = 10.0,
        adam_betas: tuple[float, float] = (0.9, 0.95),
    ) -> list[dict]:
        r"""Split :class:`RollingDiT` parameters into Muon vs. aux-AdamW groups.

        Mirrors upstream amip's ``get_rolling_dit_muon_param_groups()``.

        **Muon**: the ``>=2D`` weight matrices of the hidden transformer
        blocks — per-frame spatial, causal-temporal, and (Phase 12e) the
        causal forcing cross-attention blocks.

        **aux AdamW**: all biases / 1-D params from those blocks; the
        forcing blocks' learned positional tables (``temporal_pos`` /
        ``query_pos``, which are 2-D but position tables rather than
        matmul weights, so Muon's orthogonalisation is inappropriate); and
        *all* parameters of the input projection (legacy patch-embed +
        c_grid/scalar embedders, or the Phase-12e
        :class:`RollingDiTInputEmbed` that replaces them), the time
        embedder, and whichever output head is built.

        Every parameter of the backbone lands in exactly one group — see
        ``test_rolling_dit_features.py``, which asserts that for the
        all-features-on configuration.
        """
        block_params = list(self.backbone.spatial_blocks.parameters()) + list(
            self.backbone.temporal_blocks.parameters()
        )

        # Forcing cross-attention: Linear weights to Muon, but keep the
        # (window_size, dim) positional tables on AdamW.
        forcing_pos = []
        for name, p in self.backbone.forcing_blocks.named_parameters():
            if name.endswith("temporal_pos") or name.endswith("query_pos"):
                forcing_pos.append(p)
            else:
                block_params.append(p)

        hidden_weights = [p for p in block_params if p.ndim >= 2]
        hidden_gains_biases = [p for p in block_params if p.ndim < 2]
        hidden_gains_biases += forcing_pos

        # The whole input/output projection is AdamW territory regardless of
        # which variant was built; ``None`` entries are the modes that did not
        # build that submodule.
        nonhidden_modules = [
            self.backbone.patch_embed_main,
            self.backbone.t_embedder,
            self.backbone.unpatchify_layer,
            getattr(self.backbone, "input_embed", None),
            getattr(self.backbone, "output_head", None),
            self.backbone.c_grid_embed,
            self.backbone.scalar_embedder,
        ]
        nonhidden_params = [
            p for m in nonhidden_modules if m is not None for p in m.parameters()
        ]

        return _muon_groups(
            hidden_weights,
            hidden_gains_biases + nonhidden_params,
            lr=lr,
            weight_decay=weight_decay,
            muon_lr_multiplier=muon_lr_multiplier,
            adam_betas=adam_betas,
        )


class ERDMWrapper(_PNeMoModule, _RollingPackUnpackMixin):
    r"""Rolling-window diffusion wrapper around :class:`ERDM` (UNet variant).

    .. note:: **Frozen on the amip-v1 contract (Phase 12).** Upstream
        amip_v2 deleted the ERDMUnet backbone (the ERDM *scheduler*
        survives, paired with :class:`RollingDiT` — see
        ``conf/model/amip_erdm_v2.yaml``). This wrapper never receives
        the ``"v2"`` layout; its ``channel_layout`` kwarg carries the two
        v1-era contracts (Phase 12b correctness fix): ``"fork"``
        (default — historical Phase-8 packing) and ``"v1"`` (upstream
        v1's real training contract — **use for checkpoints translated
        from real v1 Lightning ckpts**, ``--source-contract v1``). See
        ``docs/dev/phase12_implementation_plan.md`` ("dual-contract
        seam").
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
        channel_layout: str = "fork",
        erdm_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        # Frozen family: only the two v1-era contracts, never "v2".
        self.channel_layout = _validate_channel_layout(
            channel_layout, ("fork", "v1")
        )
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
        erdm_kwargs.setdefault("c_grid_downsample", 1)
        self.backbone = ERDM(**erdm_kwargs)

    def forward(self, x_noised, c_noise, c_grid=None, c_scalar=None):
        return self.backbone(x_noised, c_noise, c_grid=c_grid, c_scalar=c_scalar)

    def muon_param_groups(
        self,
        *,
        lr: float,
        weight_decay: float = 0.01,
        muon_lr_multiplier: float = 10.0,
        adam_betas: tuple[float, float] = (0.9, 0.95),
    ) -> list[dict]:
        r"""Split :class:`ERDM` (UNet) parameters into Muon vs. aux-AdamW groups.

        Mirrors upstream amip's ``get_erdm_muon_param_groups()``. The
        ``>=2D`` weights of the encoder/decoder blocks, down/up-samples,
        bottleneck blocks, and causal temporal-attention layers go to
        Muon; their biases/1-D params, plus *all* parameters of the
        input/output projections and the noise/forcing/calendar
        embedders, go to aux AdamW.
        """
        muon_modules = [
            self.backbone.enc_blocks,
            self.backbone.dec_blocks,
            self.backbone.downsamples,
            self.backbone.upsamples,
            self.backbone.mid_block1,
            self.backbone.mid_attn,
            self.backbone.mid_block2,
            self.backbone.mid_temporal,
            self.backbone.mid_temporal2,
        ]
        muon_weights: list[torch.nn.Parameter] = []
        adamw_from_muon_modules: list[torch.nn.Parameter] = []
        for mod in muon_modules:
            for p in mod.parameters():
                if p.ndim >= 2:
                    muon_weights.append(p)
                else:
                    adamw_from_muon_modules.append(p)

        adamw_modules = [
            self.backbone.input_conv,
            self.backbone.out_norm,
            self.backbone.out_conv,
            self.backbone.t_embedder,
        ]
        if self.backbone.c_grid_embed is not None:
            adamw_modules.append(self.backbone.c_grid_embed)
        if self.backbone.scalar_embedder is not None:
            adamw_modules.append(self.backbone.scalar_embedder)
        adamw_params = [p for mod in adamw_modules for p in mod.parameters()]
        adamw_params += adamw_from_muon_modules

        return _muon_groups(
            muon_weights,
            adamw_params,
            lr=lr,
            weight_decay=weight_decay,
            muon_lr_multiplier=muon_lr_multiplier,
            adam_betas=adam_betas,
        )


# ---------------------------------------------------------------------------
# x_DDC super-resolution cascade wrapper + CombinedModule (Phase 8f, F6)
# ---------------------------------------------------------------------------


class XDDCWrapper(_PNeMoModule):
    r"""x_DDC super-resolution cascade wrapper around :class:`XDDCUNet`.

    Unlike :class:`AmipDiTWrapper` / :class:`RollingDiTWrapper` /
    :class:`ERDMWrapper`, x_DDC has **no** ``c_grid`` / ``c_scalar``
    conditioning — the "condition" passed to the backbone *is* the
    low-res field itself (bilinear-upsampled back to full resolution).
    Channel order matches upstream's ``common.utils.assemble_input``
    convention: ``(surface, [diagnostic,] upper_air)`` — diagnostic
    precedes the flattened upper-air block, unlike the other wrappers'
    ``(surface, upper_air, diagnostic)`` order. Getting this order
    right matters for loading real x_DDC checkpoint weights correctly.

    .. note:: **Channel layout (Phase 12b).** ``channel_layout`` selects
        the upper-air block layout inside that group order and travels
        with the ``.mdlus`` args: ``"v2"`` (default) = level-major,
        1000 hPa first (upstream amip_v2); ``"v1"`` = variable-major in
        config level order (real v1 x_DDC checkpoints —
        ``--source-contract v1`` at translation time).

    Parameters
    ----------
    surface_variables, upper_air_variables, diagnostic_variables : list[str]
        Prognostic channel names (used for pack/unpack). Same
        full-resolution grid for both the noised input and the
        upsampled-low-res conditioning field.
    levels : list[float]
        Pressure levels (used to size the flattened upper-air block).
    horizontal_resolution : (int, int)
        ``(nlat, nlon)`` — the *full* (high) resolution grid.
    downsample_factor : int, optional, default=4
        Bilinear down/up-sample factor used to build the low-res
        conditioning field from a full-res field (matches upstream's
        ``x_DDC.encoder.downsample_factor``).
    unet_kwargs : dict, optional
        Forwarded to :class:`XDDCUNet` (``model_channels``,
        ``channel_mult``, etc.).
    """

    def __init__(
        self,
        *,
        surface_variables: Sequence[str],
        upper_air_variables: Sequence[str],
        diagnostic_variables: Sequence[str] = (),
        levels: Sequence[float],
        horizontal_resolution: Sequence[int],
        downsample_factor: int = 4,
        channel_layout: str = "v2",
        decoder_type: str = "unet",
        unet_kwargs: dict | None = None,
        dit_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
        # x_DDC's group order always matched upstream ([surface | diag |
        # upper_air]), so only the upper-air block layout varies: "v1" =
        # variable-major config order, "v2" = level-major 1000-hPa-first.
        # There is no "fork" layout here.
        self.channel_layout = _validate_channel_layout(
            channel_layout, ("v1", "v2")
        )
        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.downsample_factor = int(downsample_factor)

        self.num_surface = len(self.surface_variables)
        self.num_upper_air_vars = len(self.upper_air_variables)
        self.num_diagnostic = len(self.diagnostic_variables)
        self.num_levels = len(self.levels)

        self.in_channels = (
            self.num_surface
            + self.num_upper_air_vars * self.num_levels
            + self.num_diagnostic
        )

        # Which denoiser backbone (Phase 12h). ``unet`` is the v1 convolutional
        # one this fork ported in Phase 8f and the frozen v1 family loads;
        # ``dit`` is :class:`DiTAE`, the ONLY x_DDC denoiser amip_v2 still has
        # (upstream's ``ae_module.py`` raises NotImplementedError for anything
        # else), so v2-trained x_DDC checkpoints need it. Both take the same
        # in/out convention: ``in_channels`` is the concat(x_noised, cond) count
        # (twice the state width), ``out_channels`` the bare state width.
        if decoder_type not in ("unet", "dit"):
            raise ValueError(
                f"decoder_type must be 'unet' or 'dit', got {decoder_type!r}"
            )
        self.decoder_type = decoder_type
        if decoder_type == "unet":
            if dit_kwargs:
                raise ValueError(
                    "dit_kwargs given with decoder_type='unet'; pick one "
                    "backbone so the checkpoint's key layout is unambiguous"
                )
            backbone_kwargs = dict(unet_kwargs or {})
        else:
            if unet_kwargs:
                raise ValueError(
                    "unet_kwargs given with decoder_type='dit'; pick one "
                    "backbone so the checkpoint's key layout is unambiguous"
                )
            backbone_kwargs = dict(dit_kwargs or {})
        backbone_kwargs.setdefault("in_channels", 2 * self.in_channels)
        backbone_kwargs.setdefault("out_channels", self.in_channels)
        if decoder_type == "dit":
            # DiTAE emits the FULL-resolution grid, so it needs it explicitly —
            # unlike the UNet, which infers spatial size from its input.
            nlat, nlon = self.horizontal_resolution
            backbone_kwargs.setdefault("nlat", nlat)
            backbone_kwargs.setdefault("nlon", nlon)
        self.backbone = (
            XDDCUNet(**backbone_kwargs)
            if decoder_type == "unet"
            else DiTAE(**backbone_kwargs)
        )

        self.downsampler = BilinearEncoder(downsample_factor=self.downsample_factor)
        self.upsampler = BilinearDecoder(downsample_factor=self.downsample_factor)

    def forward(self, x_noised, cond, t):
        return self.backbone(x_noised, cond, t)

    # ------------------------------------------------------------------ #
    # Pack / unpack — recipe-facing helpers. Channel order (surface,
    # diagnostic, upper_air) matches upstream's assemble_input, NOT the
    # (surface, upper_air, diagnostic) order the other wrappers use.
    # ------------------------------------------------------------------ #

    def state_layout(self) -> dict[str, int]:
        r"""Block sizes of the packed channel axis (upstream ``state_layout``)."""
        return {
            "nsurface": self.num_surface,
            "ndiagnostic": self.num_diagnostic,
            "nlevels": self.num_levels,
            "n_upper_air": self.num_upper_air_vars,
            "nocean": 0,
        }

    def pack_state(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        r"""``sample -> x [B, C, H, W]`` (concat surface + diagnostic + upper_air)."""
        flatten_ua = (
            _flatten_upper_air_v2
            if self.channel_layout == "v2"
            else _flatten_upper_air
        )
        parts: list[torch.Tensor] = [sample["surface_in"]]
        if self.num_diagnostic > 0:
            parts.append(sample["diagnostic"])
        if self.num_upper_air_vars > 0:
            parts.append(flatten_ua(sample["upper_air_in"]))
        return torch.cat(parts, dim=-3)

    def unpack_state(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""``x [B, C, H, W] -> {surface_in, diagnostic, upper_air_in}``."""
        unflatten_ua = (
            _unflatten_upper_air_v2
            if self.channel_layout == "v2"
            else _unflatten_upper_air
        )
        idx = 0
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = x.narrow(-3, idx, self.num_surface)
        idx += self.num_surface
        if self.num_diagnostic > 0:
            out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
            idx += self.num_diagnostic
        if self.num_upper_air_vars > 0:
            ua_flat = x.narrow(
                -3, idx, self.num_upper_air_vars * self.num_levels
            )
            out["upper_air_in"] = unflatten_ua(
                ua_flat, self.num_upper_air_vars, self.num_levels
            )
            idx += self.num_upper_air_vars * self.num_levels
        return out

    def downsample_then_upsample(
        self, sample: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        r"""Full-res ``sample -> packed low-res-then-upsampled "cond" field.

        Matches upstream's ``AutoencoderModule.encode`` (minus the
        optional training-noise injection): bilinear-downsamples then
        immediately bilinear-upsamples back to full resolution,
        producing the blurry conditioning field the scheduler denoises
        against during standalone x_DDC training. At inference inside
        :class:`CombinedModule`, the conditioning field instead comes
        from a real forecaster's low-res prediction upsampled the same
        way — see :meth:`CombinedModule.forward`.
        """
        surface = sample["surface_in"]
        upper_air = sample.get("upper_air_in")
        diagnostic = sample.get("diagnostic") if self.num_diagnostic > 0 else None
        z_surface, z_upper, z_diag = self.downsampler(surface, upper_air, diagnostic)
        z_surface, z_upper, z_diag = self.upsampler(z_surface, z_upper, z_diag)
        return self.pack_state(
            {"surface_in": z_surface, "upper_air_in": z_upper, "diagnostic": z_diag}
        )


class CombinedModule(_PNeMoModule):
    r"""Two-stage forecaster + x_DDC downscaler composition (Phase 8f, F6).

    Frozen, evaluation-only composition matching upstream's
    ``CombinedModule``: a low-res forecaster
    (:class:`AmipDiTWrapper` / :class:`RollingDiTWrapper` /
    :class:`ERDMWrapper`) predicts the next state at its own (lower)
    resolution; the prediction is bilinear-upsampled to full resolution
    and fed to the x_DDC downscaler (:class:`XDDCWrapper` +
    :class:`~physicsnemo.experimental.diffusion.DataDependentInterpolant`)
    as the low-res conditioning field, producing the final full-res
    forecast.

    Both sub-modules are loaded from independently-trained checkpoints
    (matches upstream — there is no standalone "Combined" checkpoint).
    This composition is **not trained end-to-end**; use :meth:`eval`
    and :meth:`forward` for inference only.

    Parameters
    ----------
    forecaster
        A trained forecaster wrapper (:class:`AmipDiTWrapper` etc.),
        operating at its own (low) resolution.
    forecaster_scheduler
        The diffusion scheduler paired with ``forecaster`` (e.g.
        :class:`~physicsnemo.experimental.diffusion.DynamicInterpolant`).
    downscaler
        A trained :class:`XDDCWrapper`, operating at full resolution.
    downscaler_scheduler
        The :class:`~physicsnemo.experimental.diffusion.DataDependentInterpolant`
        paired with ``downscaler``.
    """

    def __init__(
        self,
        *,
        forecaster: _PNeMoModule,
        forecaster_scheduler,
        downscaler: XDDCWrapper,
        downscaler_scheduler,
    ):
        super().__init__(meta=MetaData())
        self.forecaster = forecaster
        self.forecaster_scheduler = forecaster_scheduler
        self.downscaler = downscaler
        self.downscaler_scheduler = downscaler_scheduler

    @torch.no_grad()
    def forward(
        self,
        sample: dict[str, torch.Tensor],
        *,
        forecaster_num_steps: int | None = None,
        downscaler_num_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""Low-res ``sample`` dict -> full-res forecast dict.

        ``sample`` is shaped for the *forecaster's* resolution (its own
        ``pack_state`` / ``pack_c_grid`` contract) — the same layout
        used to train/validate the forecaster standalone. The
        downscaler's input is never taken from ``sample`` directly; it
        is always the forecaster's own prediction, upsampled.
        """
        forecaster = (
            self.forecaster.module
            if hasattr(self.forecaster, "module")
            else self.forecaster
        )
        x = forecaster.pack_state(sample)
        c_grid = forecaster.pack_c_grid(sample)
        c_scalar = sample.get("calendar")
        forecast_lowres = self.forecaster_scheduler.sample(
            self.forecaster, x, c_grid, c_scalar, num_steps=forecaster_num_steps
        )
        # Some schedulers (e.g. DynamicInterpolant with its
        # return_model_last=True default) return (y, model_last_pred)
        # instead of a plain tensor — take the first element either way.
        if isinstance(forecast_lowres, tuple):
            forecast_lowres = forecast_lowres[0]
        highres = self._downscale(
            forecast_lowres, forecaster=forecaster, num_steps=downscaler_num_steps
        )
        return self.downscaler.unpack_state(highres)

    # ------------------------------------------------------------------
    # Rolling-window streaming (Phase 12h) — upstream's windowed_init /
    # windowed_step, for driving an ERDM forecaster frame by frame.
    # ------------------------------------------------------------------

    def _downscale(
        self,
        lowres_packed: torch.Tensor,
        *,
        forecaster=None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        r"""One packed low-res state -> one packed full-res state.

        Unpack at the forecaster's contract, bilinear-upsample, repack at the
        downscaler's, and sample. Shared by :meth:`forward` and
        :meth:`windowed_step` so the single-step and streaming paths cannot
        drift in how they cross the resolution boundary.
        """
        if forecaster is None:
            forecaster = (
                self.forecaster.module
                if hasattr(self.forecaster, "module")
                else self.forecaster
            )
        lowres_state = forecaster.unpack_state(lowres_packed)
        diagnostic = (
            lowres_state.get("diagnostic")
            if self.downscaler.num_diagnostic > 0
            else None
        )
        surface_up, upper_up, diag_up = self.downscaler.upsampler(
            lowres_state["surface_in"], lowres_state.get("upper_air_in"), diagnostic
        )
        cond = self.downscaler.pack_state(
            {"surface_in": surface_up, "upper_air_in": upper_up, "diagnostic": diag_up}
        )
        highres = self.downscaler_scheduler.sample(
            self.downscaler, cond, num_steps=num_steps
        )
        if isinstance(highres, tuple):
            highres = highres[0]
        return highres

    @torch.no_grad()
    def windowed_init(self, init_window: torch.Tensor):
        r"""Prime the ERDM rolling window for a streaming rollout.

        Schedule-matched noising of the oracle window at global diffusion time
        ``t=0`` — the same warm-up
        :meth:`~physicsnemo.experimental.diffusion.ERDMScheduler.sample_rollout`
        does internally, exposed so a driver can emit frames one at a time (and
        checkpoint between them) instead of materialising the whole horizon.

        ``init_window`` is ``(b, W, C, h, w)`` at the forecaster's resolution.
        Pass a **bare state** window under ``nocean > 0``: it is zero-padded
        here and the first roll's imposition overwrites it, so a driver reading
        a state store never has to invent SST / sea-ice values.

        Returns ``(x_bar, eps_prev)`` — the noised window and the last frame's
        noise latent, which seeds the AR(1) chain.
        """
        sch = self.forecaster_scheduler
        b = init_window.shape[0]
        init_window = sch.pad_state(init_window)
        sigma0 = sch.sigma_schedule(torch.zeros(b, device=init_window.device))
        eps_win = sch.temporal_noise(init_window)
        x_bar = init_window + sch.w5(sigma0) * eps_win
        return x_bar, eps_win[:, -1:]

    @torch.no_grad()
    def windowed_step(
        self,
        x_bar: torch.Tensor,
        eps_prev: torch.Tensor,
        c_grid_win: torch.Tensor | None,
        c_scalar_win: torch.Tensor | None,
        num_steps: int | None = None,
        *,
        ocean_win: torch.Tensor | None = None,
        downscaler_num_steps: int | None = None,
    ):
        r"""Advance the window by one emitted frame and downscale it.

        One inner ODE sweep (front frame -> clean), downscale the emitted
        low-res frame, then shift the window forward and append a fresh
        max-noise frame continuing the AR(1) chain.

        ``ocean_win`` is the forcing window **shifted forward one step**, from
        which the true ocean fields are imposed (Phase 12f); ``None`` disables
        it. Returns ``(y_highres, x_bar, eps_prev)``.
        """
        sch = self.forecaster_scheduler
        # A rolling state saved by a run with a different channel count would
        # slice silently wrong, and a streaming driver may resume from disk.
        expected = getattr(self.forecaster, "in_channels", None)
        if expected is not None and x_bar.shape[2] != expected:
            raise ValueError(
                f"rolling-window state has {x_bar.shape[2]} channels but this "
                f"forecaster emits {expected}: the state came from a run with a "
                f"different ocean_state_variables. Start the rollout fresh "
                f"rather than resuming."
            )
        x_bar = sch.sample_window(
            self.forecaster, x_bar, c_grid_win, c_scalar_win, num_steps,
            ocean_win=ocean_win,
        )
        # Strip the predicted ocean block before the downscaler: it is a
        # pretrained, state-width model and must not see the extra channels.
        emitted = sch.strip_ocean(x_bar[:, 0])
        y_highres = self._downscale(emitted, num_steps=downscaler_num_steps)
        # Shift forward, appending a fresh max-noise back frame.
        eps_prev = sch.temporal_noise_next(eps_prev)
        x_bar = torch.cat([x_bar[:, 1:], eps_prev * sch.sigma_max], dim=1)
        return y_highres, x_bar, eps_prev


__all__ = [
    "AmipDiTWrapper",
    "CombinedModule",
    "ERDMWrapper",
    "RollingDiTWrapper",
    "XDDCWrapper",
]
