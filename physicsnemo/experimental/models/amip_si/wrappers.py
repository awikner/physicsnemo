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
from .layers.bilinear import BilinearDecoder, BilinearEncoder
from .rolling_dit import RollingDiT
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
        The ``>=2D`` weight matrices of the per-frame spatial blocks and
        the causal-temporal blocks go to Muon; the rest (block
        biases/norms plus *all* parameters of the patch embed, time
        embedder, unpatchify head, and the optional c_grid / scalar
        embedders) go to aux AdamW.
        """
        block_params = list(self.backbone.spatial_blocks.parameters()) + list(
            self.backbone.temporal_blocks.parameters()
        )
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
        nonhidden_params = [p for m in nonhidden_modules for p in m.parameters()]

        return _muon_groups(
            hidden_weights,
            hidden_gains_biases + nonhidden_params,
            lr=lr,
            weight_decay=weight_decay,
            muon_lr_multiplier=muon_lr_multiplier,
            adam_betas=adam_betas,
        )


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
        unet_kwargs: dict | None = None,
    ):
        super().__init__(meta=MetaData())
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

        unet_kwargs = dict(unet_kwargs or {})
        # Upstream's ``in_channels`` is the concat(x_noised, cond) count
        # (twice the state channel count); ``out_channels`` is the bare
        # state channel count — same convention as AmipDiTWrapper's
        # dit_kwargs.setdefault for in_channels/out_channels.
        unet_kwargs.setdefault("in_channels", 2 * self.in_channels)
        unet_kwargs.setdefault("out_channels", self.in_channels)
        self.backbone = XDDCUNet(**unet_kwargs)

        self.downsampler = BilinearEncoder(downsample_factor=self.downsample_factor)
        self.upsampler = BilinearDecoder(downsample_factor=self.downsample_factor)

    def forward(self, x_noised, cond, t):
        return self.backbone(x_noised, cond, t)

    # ------------------------------------------------------------------ #
    # Pack / unpack — recipe-facing helpers. Channel order (surface,
    # diagnostic, upper_air) matches upstream's assemble_input, NOT the
    # (surface, upper_air, diagnostic) order the other wrappers use.
    # ------------------------------------------------------------------ #

    def pack_state(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        r"""``sample -> x [B, C, H, W]`` (concat surface + diagnostic + upper_air)."""
        parts: list[torch.Tensor] = [sample["surface_in"]]
        if self.num_diagnostic > 0:
            parts.append(sample["diagnostic"])
        if self.num_upper_air_vars > 0:
            parts.append(_flatten_upper_air(sample["upper_air_in"]))
        return torch.cat(parts, dim=-3)

    def unpack_state(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""``x [B, C, H, W] -> {surface_in, diagnostic, upper_air_in}``."""
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
            out["upper_air_in"] = _unflatten_upper_air(
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
        lowres_state = forecaster.unpack_state(forecast_lowres)

        surface = lowres_state["surface_in"]
        upper_air = lowres_state.get("upper_air_in")
        diagnostic = (
            lowres_state.get("diagnostic") if self.downscaler.num_diagnostic > 0 else None
        )
        surface_up, upper_up, diag_up = self.downscaler.upsampler(
            surface, upper_air, diagnostic
        )
        cond = self.downscaler.pack_state(
            {"surface_in": surface_up, "upper_air_in": upper_up, "diagnostic": diag_up}
        )

        highres = self.downscaler_scheduler.sample(
            self.downscaler, cond, num_steps=downscaler_num_steps
        )
        return self.downscaler.unpack_state(highres)


__all__ = [
    "AmipDiTWrapper",
    "CombinedModule",
    "ERDMWrapper",
    "RollingDiTWrapper",
    "XDDCWrapper",
]
