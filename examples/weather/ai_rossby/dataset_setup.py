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


def _abs_path(p):
    """Resolve a config path against the ORIGINAL cwd (hydra chdirs into the run
    dir), matching the recipes' ``_resolve_path``. ``None`` passes through."""
    if not p:
        return None
    from hydra.utils import to_absolute_path

    return to_absolute_path(str(p))

logger = logging.getLogger("ai_rossby.dataset_setup")


def _cfg_list(node, key: str) -> list:
    value = node.get(key, None) if node is not None else None
    return list(value) if value else []


class VaryingBoundarySubset:
    """Select the model's varying-boundary channels out of the store's set.

    Shared version of the slice ``inference.py`` and ``train.py`` each carried
    privately (2026-08-14). Chains **first**, ahead of the NaN-fill and the
    normalizer, because both of those are sized from the MODEL's list.

    The case that forced this into the shared path: the real v1 ``SI`` (non-CO2)
    checkpoint lists 3 varying channels while ``amip_dailyavg_coarse`` serves 4
    — upstream's own run simply never fed ``global_mean_co2``. Without a slice,
    ``NanFillTransform`` indexes a 4-channel tensor against a 3-entry fill
    vector and dies with a bare ``IndexError``. Note the dropped channel is the
    store's FIRST, so "take the leading N" would silently mis-assign every
    forcing; the indices have to come from name lookup.
    """

    def __init__(self, indices: Sequence[int]) -> None:
        import torch

        self._idx = torch.tensor(list(indices), dtype=torch.long)

    def __call__(self, sample: dict) -> dict:
        import torch

        out = dict(sample)
        for key in ("varying_boundary", "varying_boundary_seq", "varying_boundary_next_seq"):
            v = out.get(key)
            if v is not None and torch.is_tensor(v):
                out[key] = v.index_select(v.ndim - 3, self._idx)
        return out


def model_varying_pre_rescaler(cfg: DictConfig) -> list[str]:
    """The model's varying list as the STORE supplies it, before the SST rescaler.

    ``cfg.model.varying_boundary_variables`` is written in POST-rescaler order —
    ``amip_erdm_fancy`` lists ``sea_surface_temperature_anomaly``, which no store
    holds: :class:`SSTRescaler` derives it inside the assembler. But the two
    consumers of that list, :func:`resolve_varying_subset` and the normalizer,
    both run BEFORE the assembler, on raw store channels.

    Comparing the two stages directly is what broke the fancy contract: the
    derived name is absent from the store, the subset test
    (``set(model).issubset(set(store))``) fails, no slice is applied, and
    ``global_mean_co2`` — which this contract replaces with the SST trend scalar
    — survives into the grid. The result is
    ``c_grid_dim=6 but the data pipeline produces 7``, which is why
    ``amip_erdm_fancy`` could not be trained at all (verified 2026-08-19 on
    DeltaAI: it fails identically to ``amip_rsi_fancy``). The static config
    gates never caught it because they never build a pipeline, and the
    checkpoint translator reconstructs the order arithmetically without one.

    Inverts :func:`~physicsnemo.experimental.datapipes.climate.sst_forcing.grid_forcing_names`:
    ``append`` inserted the anomaly after the SST channel, so drop it;
    ``replace`` substituted it for the SST channel, so put SST back.
    """
    from physicsnemo.experimental.datapipes.climate.sst_forcing import (
        SST_ANOMALY_CHANNEL_NAME,
        SST_VARIABLE_NAMES,
    )

    names = [str(v) for v in _cfg_list(cfg.model, "varying_boundary_variables")]
    data = cfg.get("dataset") or {}
    mode = str(data.get("sst_anomaly_channel", "none") or "none")
    if mode == "none" or SST_ANOMALY_CHANNEL_NAME not in names:
        return names
    if mode == "append":
        return [n for n in names if n != SST_ANOMALY_CHANNEL_NAME]
    # "replace": the anomaly occupies the absolute SST channel's slot. Restore
    # whichever SST name the store actually uses, so the subset can find it.
    store_sst = next(iter(SST_VARIABLE_NAMES))
    return [store_sst if n == SST_ANOMALY_CHANNEL_NAME else n for n in names]


def resolve_varying_subset(
    cfg: DictConfig, store_varying: Sequence[str] | None
) -> Optional[list[int]]:
    """Indices of the model's varying channels within the store's, or ``None``.

    ``None`` means "no slice needed" — the lists match, or there is nothing to
    compare against. A model list that is **not** a subset is left alone
    deliberately: that is a genuine misconfiguration, and the width checks
    downstream (``ForcingPipeline.assert_matches``) name it better than a
    silent reordering would.
    """
    # PRE-rescaler: the slice runs on store channels, so a derived pseudo-channel
    # must not be looked for there (see model_varying_pre_rescaler).
    model_varying = model_varying_pre_rescaler(cfg)
    store = [str(v) for v in (store_varying or [])]
    if not model_varying or not store or model_varying == store:
        return None
    if not set(model_varying).issubset(set(store)):
        logger.warning(
            f"model varying_boundary_variables {model_varying} is not a subset of "
            f"the store's {store}; leaving the stream unsliced"
        )
        return None
    indices = [store.index(v) for v in model_varying]
    logger.info(
        f"varying-boundary subset active: model uses {model_varying} of {store} "
        f"(indices={indices})"
    )
    return indices


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


def build_forcing_assembler(cfg: DictConfig, *, strict: bool = True,
                            sst_rescaler=None):
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
        sst_rescaler=sst_rescaler,
    )


def resolve_scalar_forcing(cfg: DictConfig) -> Optional[str]:
    """Which trend scalar occupies the calendar row's third slot, or ``None``.

    Mirrors upstream ``AMIPDataset._resolve_scalar_forcing``. In this fork the
    CO₂ route is spelled ``model.scalar_routed_boundary_variables`` (12d), so
    that list is what ``auto`` inspects and what ``global_mean_sst`` conflicts
    with: **both claim the same slot**, so asking for the SST scalar while CO₂
    is still routed is an error rather than a silent overwrite.
    """
    from physicsnemo.experimental.datapipes.climate import SST_VARIABLE_NAMES

    data = cfg.dataset
    routed = _cfg_list(cfg.model, "scalar_routed_boundary_variables")
    has_co2 = "global_mean_co2" in routed

    choice = str(data.get("scalar_forcing", "auto") or "auto").lower()
    valid = ("auto", "none", "co2", "global_mean_sst")
    if choice not in valid:
        raise ValueError(f"scalar_forcing must be one of {valid}, got {choice!r}")

    if choice == "auto":
        return "co2" if has_co2 else None
    if choice == "none":
        return None
    if choice == "co2":
        if not has_co2:
            raise ValueError(
                "scalar_forcing: co2 needs 'global_mean_co2' in "
                f"model.scalar_routed_boundary_variables, got {routed}"
            )
        return "co2"
    # global_mean_sst
    if has_co2:
        raise ValueError(
            "scalar_forcing: global_mean_sst conflicts with the routed "
            "'global_mean_co2' channel — both occupy the calendar row's third "
            "slot. Drop global_mean_co2 from "
            "model.scalar_routed_boundary_variables (and from the model's "
            "varying_boundary_variables) to use the SST scalar."
        )
    varying = _cfg_list(cfg.model, "varying_boundary_variables")
    if not any(v in SST_VARIABLE_NAMES for v in varying):
        raise ValueError(
            f"scalar_forcing: global_mean_sst needs an SST channel in "
            f"varying_boundary_variables, got {varying}"
        )
    return "global_mean_sst"


def build_sst_rescaler(cfg: DictConfig, *, normalizer):
    """The Phase 12g ``sst_rescaler`` hook, or ``None`` when unused.

    Needs the normalizer: the rescaler recovers physical kelvin by inverting the
    z-score of the SST channel, so it reads that channel's ``(mean, std)`` from
    the same statistics the pipeline normalizes with — by NAME, not by position.
    """
    from physicsnemo.experimental.datapipes.climate import (
        SSTForcing,
        SSTRescaler,
    )

    data = cfg.dataset
    scalar = resolve_scalar_forcing(cfg)
    forcing = SSTForcing.from_config(
        data,
        requires_scalar=(scalar == "global_mean_sst"),
        path=_abs_path(data.get("sst_climatology_path", None)),
    )
    if forcing is None:
        return None
    if normalizer is None:
        raise ValueError(
            "sst_anomaly_channel / scalar_forcing: global_mean_sst need the "
            "normalizer's SST statistics to recover kelvin; build the pipeline "
            "with normalizer=..."
        )

    names = getattr(normalizer, "varying_boundary_variables", None) or _cfg_list(
        cfg.model, "varying_boundary_variables"
    )
    idx = SSTForcing.sst_index(names)
    rescaler = SSTRescaler(
        forcing,
        names,
        sst_mean=float(normalizer.varying_mean.reshape(-1)[idx]),
        sst_std=float(normalizer.varying_std.reshape(-1)[idx]),
        emit_scalar=(scalar == "global_mean_sst"),
    )
    logger.info("%s", forcing.describe())
    logger.info(
        "SST rescaler: channel %d of %s -> grid names %s",
        idx,
        list(names),
        rescaler.grid_forcing_names,
    )
    return rescaler


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
    store_varying_variables: Sequence[str] | None = None,
) -> ForcingPipeline:
    """Build the recipe's :class:`ForcingPipeline` (see module docstring).

    ``store_varying_variables`` is the store's own varying-boundary list. Pass
    it whenever the dataset is available (it is, at every call site — the
    pipeline is built after the dataset): when the model consumes a strict
    subset, a :class:`VaryingBoundarySubset` slice is prepended so the NaN-fill
    and normalizer — both sized from the model's list — see the right width.
    Callers that pass it must also align the normalizer to the model's list
    (``ClimateNormalizer.from_dataset(..., varying_boundary_variables=...)``),
    since the slice runs before it.
    """
    subset = resolve_varying_subset(cfg, store_varying_variables)
    if subset is not None:
        extra_transforms = (VaryingBoundarySubset(subset), *tuple(extra_transforms))
    # Phase 12g: the SST rescaler runs inside the assembler, before the CO2 pop
    # (upstream's step-2 ordering), and is a no-op unless the dataset config asks
    # for the anomaly channel or the global_mean_sst scalar.
    return ForcingPipeline(
        nan_fill=build_nan_fill(cfg, strict=nan_fill_strict),
        assembler=build_forcing_assembler(
            cfg, sst_rescaler=build_sst_rescaler(cfg, normalizer=normalizer)
        ),
        normalizer=normalizer,
        normalize_in_dataset=normalize_in_dataset,
        extra_transforms=tuple(extra_transforms),
    )
