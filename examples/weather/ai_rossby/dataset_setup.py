# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""One construction site for the sample→model-input pipeline (Phase 12d.13).

Upstream amip_v2 has a single ``AMIPDataset.forcing_from_raw`` that every
entrypoint calls, "so the channel contract cannot fork between training and
a 40-year rollout". The fork spreads the same work over three composable
pieces (NaN-fill → normalizer → :class:`ForcingAssembler`), and before this
module each of the five consumers assembled them by hand — which had already
produced two real divergences:

* **Order.** ``train_diffusion.py`` composed ``nan_fill(normalizer(sample))``,
  i.e. it substituted *physical-unit* fill values (e.g. SST 270 K) into an
  already z-scored tensor — a ~+20σ constant over every masked gridpoint.
  ``inference.py`` / ``climatology_cli.py`` / ``train.py`` all fill first and
  normalize after, which is the order :class:`NanFillTransform`'s own
  contract specifies.
* **Scope.** ``train.py`` / ``inference.py`` fill masked NaN in the
  *prognostic surface* and *diagnostic* groups too (PLASIM/ERA5 SST carries
  land-NaN, which otherwise reaches the loss); ``train_diffusion.py`` and
  ``climatology_cli.py`` filled only the boundaries.

:func:`build_forcing_pipeline` is now the only place those choices are made.

Two normalization placements exist in the recipes and are both first-class
here — the split is explicit instead of accidental:

``normalize_in_dataset=True``
    The dataset transform normalizes (``train_diffusion``, and any recipe
    whose validator reads already-normalized samples). Chain:
    ``extras → nan_fill → normalizer → assembler``.
``normalize_in_dataset=False``
    The recipe normalizes at use, per batch, on device (``inference``,
    ``climatology_cli``). Chain: ``extras → nan_fill``; the recipe then
    calls :meth:`ForcingPipeline.finalize` where it used to call the
    normalizer, which applies ``normalizer → assembler``.

Either way the composed order is identical, and
:meth:`ForcingPipeline.assert_matches` cross-checks the resulting channel
contract against the model wrapper so a config mismatch fails loudly at
startup rather than silently mis-packing ``c_grid`` / ``c_scalar``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("ai_rossby.dataset_setup")


def _cfg_list(node, key: str) -> list:
    value = node.get(key, None) if node is not None else None
    return list(value) if value else []


def build_nan_fill(cfg: DictConfig, *, strict: bool = False):
    """The one :class:`NanFillTransform` definition for every recipe.

    Fills boundary **and** prognostic-surface / diagnostic NaN with the same
    per-variable mask values, in physical units (so it must run before the
    normalizer). Smoothing knobs come from ``cfg.dataset``:
    ``smooth_nan_boundaries`` / ``smooth_sigma`` / ``smooth_kernel_size`` /
    ``smooth_n_iters`` (Phase 12d.14).
    """
    from physicsnemo.experimental.datapipes.climate import NanFillTransform

    data, model = cfg.dataset, cfg.model
    return NanFillTransform(
        constant_boundary_variables=_cfg_list(model, "constant_boundary_variables"),
        varying_boundary_variables=_cfg_list(model, "varying_boundary_variables"),
        surface_variables=_cfg_list(model, "surface_variables"),
        diagnostic_variables=_cfg_list(model, "diagnostic_variables"),
        fill_values=dict(
            OmegaConf.to_container(data.get("nan_fill_values", {}), resolve=True) or {}
        ),
        default=float(data.get("nan_fill_default", 0.0)),
        strict=strict,
        smooth_nan_boundaries=bool(data.get("smooth_nan_boundaries", False)),
        smooth_sigma=float(data.get("smooth_sigma", 1.5)),
        smooth_kernel_size=int(data.get("smooth_kernel_size", 5)),
        smooth_n_iters=int(data.get("smooth_n_iters", 10)),
    )


def build_forcing_assembler(cfg: DictConfig, *, strict: bool = True):
    """The one :class:`ForcingAssembler` definition (CO₂-style scalar routing).

    Reads ``cfg.model.scalar_routed_boundary_variables`` — the *same* config
    key the model wrapper reads to shrink ``c_grid_dim`` — so the pipeline
    and the model cannot be sized from different lists.
    """
    from physicsnemo.experimental.datapipes.climate import ForcingAssembler

    model = cfg.model
    return ForcingAssembler(
        varying_boundary_variables=_cfg_list(model, "varying_boundary_variables"),
        constant_boundary_variables=_cfg_list(model, "constant_boundary_variables"),
        scalar_routed_variables=_cfg_list(model, "scalar_routed_boundary_variables"),
        calendar_dim=2,
        strict=strict,
    )


class _NormalizeAndRoute:
    """``normalizer(batch)`` then scalar routing; delegates all else.

    See :meth:`ForcingPipeline.as_normalizer`. Kept deliberately thin: the
    routing is the only added behavior, and ``__getattr__`` delegation means
    ``denormalize_state`` / ``to`` / buffers all behave as before.
    """

    def __init__(self, normalizer, assembler) -> None:
        self._normalizer = normalizer
        self._assembler = assembler

    def __call__(self, batch: dict) -> dict:
        if self._normalizer is not None:
            batch = self._normalizer(batch)
        return self._assembler(batch)

    def to(self, device):
        if self._normalizer is not None and hasattr(self._normalizer, "to"):
            self._normalizer.to(device)
        return self

    def __getattr__(self, name):  # pragma: no cover - thin delegation
        return getattr(self._normalizer, name)


@dataclass
class ForcingPipeline:
    """The composed sample→model-input contract for one recipe run."""

    nan_fill: Any
    assembler: Any
    normalizer: Optional[Any] = None
    normalize_in_dataset: bool = True
    extra_transforms: Sequence[Callable] = ()

    @property
    def dataset_transform(self) -> Callable:
        """What to assign to ``dataset.transform`` (fixed, audited order)."""
        from physicsnemo.experimental.datapipes.climate import ComposeTransform

        stages: list[Callable] = list(self.extra_transforms)
        # Physical units first: the fill values are in the data's own units.
        stages.append(self.nan_fill)
        if self.normalize_in_dataset:
            if self.normalizer is not None:
                stages.append(self.normalizer)
            # Scalar routing reads the normalized boundary stream (upstream
            # takes its CO2 scalar off the z-scored tensor as well).
            if self.assembler.active:
                stages.append(self.assembler)
        return stages[0] if len(stages) == 1 else ComposeTransform(*stages)

    def finalize(self, batch: dict) -> dict:
        """Normalize + route a batch, for recipes that normalize at use.

        A drop-in replacement for a bare ``normalizer(batch)`` call: with no
        normalizer and no routed channels it is the identity, so recipes
        that gain nothing from the assembler are provably unchanged.
        """
        if self.normalize_in_dataset:
            # Already applied inside the dataset transform.
            return batch
        if self.normalizer is not None:
            batch = self.normalizer(batch)
        if self.assembler.active:
            batch = self.assembler(batch)
        return batch

    def as_normalizer(self):
        """A normalizer-shaped proxy that also applies scalar routing.

        For recipes that pass a normalizer *object* down into rollout helpers
        which use it for both directions (``normalizer(batch)`` to normalize
        inputs and ``normalizer.denormalize_state(...)`` for physical-unit
        metrics). Calling the proxy runs :meth:`finalize`; every other
        attribute delegates to the wrapped normalizer, so no call site needs
        to change. Returns the bare normalizer when no routing is active.
        """
        if self.normalize_in_dataset or not self.assembler.active:
            return self.normalizer
        return _NormalizeAndRoute(self.normalizer, self.assembler)

    def assert_matches(self, wrapper, *, name: str = "model") -> None:
        """Fail loudly when the model's channel contract ≠ the pipeline's.

        The anti-fork guard: compares the width of the ``c_grid`` /
        ``c_scalar`` the pipeline will actually produce against what the
        wrapper was constructed for. Wrappers without those attributes
        (deterministic Pangu / SFNO) are skipped.
        """
        inner = getattr(wrapper, "module", wrapper)
        for attr, expected in (
            ("c_grid_dim", self.assembler.c_grid_dim),
            ("scalar_dim", self.assembler.scalar_dim),
        ):
            actual = getattr(inner, attr, None)
            if actual is None:
                continue
            if int(actual) != int(expected):
                raise ValueError(
                    f"forcing contract mismatch: {name}.{attr}={actual} but the "
                    f"data pipeline produces {expected} "
                    f"(constant={self.assembler.constant_boundary_variables}, "
                    f"varying={self.assembler.varying_boundary_variables}, "
                    f"scalar_routed={self.assembler.scalar_routed_variables}). "
                    f"Check model.scalar_routed_boundary_variables / scalar_dim."
                )
        logger.info(
            "forcing contract OK: c_grid_dim=%d, scalar_dim=%d, scalar_routed=%s",
            self.assembler.c_grid_dim,
            self.assembler.scalar_dim,
            self.assembler.scalar_routed_variables or "[]",
        )


def build_forcing_pipeline(
    cfg: DictConfig,
    *,
    normalizer=None,
    normalize_in_dataset: bool = True,
    extra_transforms: Sequence[Callable] = (),
    nan_fill_strict: bool = False,
) -> ForcingPipeline:
    """Build the recipe's :class:`ForcingPipeline` (see module docstring)."""
    return ForcingPipeline(
        nan_fill=build_nan_fill(cfg, strict=nan_fill_strict),
        assembler=build_forcing_assembler(cfg),
        normalizer=normalizer,
        normalize_in_dataset=normalize_in_dataset,
        extra_transforms=tuple(extra_transforms),
    )
