# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: All rights reserved.

r"""Long-horizon climate eval suite — fused scorers, both model families.

One rollout, many metrics. The predecessor (eval_diffusion.py's five
validator subclasses) re-ran the SAME full rollout once per metric family —
measured at 4 x 2.77 h of identical 4-year rollouts on Midway where one
serves, with each block scoring a different stochastic realization because
the global RNG advanced between validators. Here
:class:`~validate_diffusion.DiffusionRolloutValidator` performs the rollout
once and hands every scored (frame, channel-group) to a list of pluggable
scorers via a shared :class:`~validate_diffusion.StepContext` (ensemble
reductions and denormalization computed once, so N scorers cost no extra
collectives).

Scorers
-------
* :class:`ClimatologyScorer` — time-mean + per-bin climatology maps, bias
  maps, AND the lat-weighted global bias scalars (the old BiasValidator was a
  full second rollout for an 8-line reduction; it is absorbed here).
* :class:`QBOScorer` — tropical-band stratospheric U timeseries + period
  estimates.
* :class:`FluxSeriesScorer` — lat-weighted global-mean per-step series for
  named flux channels, as ``(horizon,)`` tensors correctly reduced across
  ranks (the old validator kept rank-local Python lists whose length also
  scaled with the IC batch count).
* spread / spread-skill are not a scorer at all: the drive's own spread
  metrics fill whenever ``ensemble_size > 1`` and
  :func:`derive_spread_skill` post-processes the flat dict (the old
  EnsembleEnvelopeValidator dissolves).

Model families
--------------
The driver is generation-agnostic. Diffusion checkpoints (wrappers exposing
``pack_state`` — the same dispatch signal inference.py uses) sample through
the scheduler instantiated from ``cfg.loss``. Deterministic checkpoints
(SfnoPlasim / PanguPlasim / ArchesWeather, trained by train.py) step through
:class:`deterministic_adapter.DeterministicStepAdapter`; their QBO/flux
variable names come from ``cfg.model`` via :class:`VariableCatalog`, because
the Pangu classes expose only channel counts. ``cfg.loss`` is never
instantiated on the deterministic path (deterministic loss configs carry no
``_target_``).

Ensembles: ONE total ``eval_suite.ensemble_size`` for the whole suite. All
metrics then describe the E-member ensemble mean (echoed in
``results["config"]`` — never compare results across differing E). Members
split evenly across ranks under a multi-rank launch (see
validate_diffusion's member-split machinery). Deterministic models require a
``gaussian_ic`` perturber for E > 1 — replicate_only would produce E
identical members and exactly zero spread.

Results are a ``torch.save`` dict, schema v2 (``schema_version: 2``): one
``rmse_acc`` flat block ({rmse|spread}_step{S}_{group} floats), ``stability``
(the scan over those traces), and one block per scorer. Partial progress
snapshots (rank 0, zero collectives, ``"_partial": True`` +
``"_progress"``) are written every ``partial_save_every_frames`` frames —
exact under member-split, rank-0's ICs under IC-split.
"""

from __future__ import annotations

import logging as _logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from climatology import (  # noqa: E402
    StreamingBinnedMean,
    StreamingTimeMean,
    lat_weighted_global_scalars,
)
from validate import Deterministic, ReplicateOnly  # noqa: E402
from validate_diffusion import (  # noqa: E402
    DiffusionRolloutValidator,
    StepContext,
)

logger = _logging.getLogger(__name__)


def _inner_wrapper(wrapper):
    return wrapper.module if hasattr(wrapper, "module") else wrapper


# ---------------------------------------------------------------------------
# Helpers (moved verbatim from eval_diffusion.py, which now aliases this
# module).
# ---------------------------------------------------------------------------


def scan_rmse_trace(
    rmse_acc: dict,
    *,
    jump_factor: float = 3.0,
    window: int = 50,
    max_jumps: int = 20,
) -> dict:
    """Instability scan over the per-step RMSE traces a rollout validator logs.

    The suite stores ``rmse_step<N>_<group>`` floats for every rollout step but
    nothing ever LOOKED at them (audited 2026-08-24): a mid-rollout blow-up or
    NaN would flow silently into the climatology accumulators. This scans each
    group's trace for (a) non-finite steps and (b) sudden jumps — a step
    exceeding ``jump_factor`` x the trailing-``window`` median. Returns

        {group: {"n_steps", "n_nonfinite", "first_nonfinite_step",
                 "jumps": [(step, value, trailing_median), ...]}}

    with ``jumps`` capped at ``max_jumps`` entries. Purely diagnostic — it
    flags, the accumulator-level finite guard (climatology._assert_finite) is
    what actually aborts.
    """
    import re as _re

    series: dict[str, list[tuple[int, float]]] = {}
    for key, val in rmse_acc.items():
        m = _re.match(r"^rmse_step(\d+)_(.+)$", str(key))
        if m:
            series.setdefault(m.group(2), []).append((int(m.group(1)), float(val)))

    out: dict[str, dict] = {}
    for group, pairs in series.items():
        pairs.sort()
        vals = [v for _, v in pairs]
        finite = [v for v in vals if math.isfinite(v)]
        first_nf = next((s for (s, v) in pairs if not math.isfinite(v)), None)
        jumps: list[tuple[int, float, float]] = []
        for i, (step, v) in enumerate(pairs):
            if i < 2 or not math.isfinite(v):
                continue
            hist = [x for x in vals[max(0, i - window):i] if math.isfinite(x)]
            if not hist:
                continue
            med = sorted(hist)[len(hist) // 2]
            if med > 0 and v > jump_factor * med and len(jumps) < max_jumps:
                jumps.append((step, v, med))
        out[group] = {
            "n_steps": len(pairs),
            "n_nonfinite": len(vals) - len(finite),
            "first_nonfinite_step": first_nf,
            "jumps": jumps,
        }
    return out


def _tropical_band_mask_and_weights(
    n_lat: int, band_deg: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mask, weights)`` for the ``[-band_deg, +band_deg]`` latitude band.

    Follows the same ``linspace(90, -90, n_lat)`` grid convention as
    :func:`validate.cos_lat_weights` / :func:`climatology.lat_weighted_global_scalars`.
    ``weights`` are ``cos(lat)``-weighted and normalized to sum to 1
    over the masked band.
    """
    phi = torch.linspace(math.pi / 2, -math.pi / 2, n_lat, device=device, dtype=dtype)
    lat_deg = phi * (180.0 / math.pi)
    mask = lat_deg.abs() <= band_deg
    weights = torch.cos(phi)[mask]
    weights = weights / weights.sum().clamp(min=1e-12)
    return mask, weights


def _estimate_period_months(timeseries: torch.Tensor, months_per_bin: float) -> float:
    """Zero-crossing period estimate (in months) for a 1-D binned timeseries.

    Demeans the series, counts sign changes, and reports twice the mean
    spacing between crossings (a full oscillation period = 2
    half-period crossings) in months. Returns ``nan`` when there are
    fewer than 2 crossings (series too short / no oscillation).
    """
    x = (timeseries - timeseries.mean()).detach().cpu()
    if x.numel() < 3:
        return float("nan")
    signs = torch.sign(x)
    signs[signs == 0] = 1.0
    # Treat the binned series as circular (last bin wraps to the first) —
    # correct for a climatological composite, where e.g. December
    # borders January, so a crossing spanning the wrap is a real one.
    signs = torch.cat([signs, signs[:1]])
    changes = (signs[1:] * signs[:-1] < 0).nonzero(as_tuple=True)[0]
    if changes.numel() < 2:
        return float("nan")
    spacings = (changes[1:] - changes[:-1]).float()
    return float(spacings.mean().item()) * 2.0 * float(months_per_bin)


def resolve_steps_per_bin(block, per_month: int, *, name: str, log=None) -> int:
    """Aggregation-bin width in model steps, derived from its duration.

    ``block.months_per_bin`` is the physical width; ``per_month`` comes from
    :func:`train_loop.steps_per_month`, i.e. from ``cfg.model.timedelta_hours``
    and the store's cadence. An explicit ``block.steps_per_bin`` still wins —
    some runs want a deliberately non-calendar bin — but a value that disagrees
    with the stated month count is warned about, because the config's own
    ``months_per_bin`` (and the QBO period estimate derived from it) then
    describes something the bins are not.

    Module-level rather than a closure in :func:`main` so the precedence is
    directly testable; it is the only thing deciding what the scorers bin by.
    """
    log = log or _logging.getLogger(__name__)
    months = float(block.get("months_per_bin", 1.0) or 1.0)
    derived = max(1, round(months * per_month))
    explicit = block.get("steps_per_bin", None)
    if explicit is None:
        return derived
    explicit = int(explicit)
    if explicit != derived:
        log.warning(
            f"{name}.steps_per_bin={explicit} overrides the {derived} steps "
            f"this run's {months}-month bin implies ({per_month} steps/month at "
            f"this model's timestep), so the bins are NOT {months} month(s) wide"
        )
    return explicit


def _perturber_scales(node) -> dict:
    """Config-or-dict -> plain dict. An empty DictConfig is FALSY, so the
    obvious ``to_container(node or {})`` hands to_container a plain dict and
    raises "Input cfg is not an OmegaConf config object"."""
    from omegaconf import OmegaConf as _OC

    if node is None:
        return {}
    if _OC.is_config(node):
        return dict(_OC.to_container(node, resolve=True) or {})
    return dict(node)


def _resolve_eval_sampler_num_steps(raw):
    """Mirror ``train_diffusion._build_validator``'s num_steps coercion."""
    if raw is None:
        return None
    from omegaconf import OmegaConf

    if OmegaConf.is_config(raw) or isinstance(raw, (list, tuple)):
        return [int(s) for s in raw]
    return int(raw)


def _to_cpu(obj):
    """Recursively move tensors to CPU for saving.

    The result tensors live on the compute device; saved as-is a rank-N file
    would pin ``cuda:N`` at load time (and a CUDA-less reader would need
    map_location gymnastics).
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_cpu(v) for v in obj]
        return t if isinstance(obj, list) else tuple(t)
    return obj


# ---------------------------------------------------------------------------
# Variable catalog — names for QBO/flux scorers, for BOTH families.
# ---------------------------------------------------------------------------


@dataclass
class VariableCatalog:
    """Variable names + levels, resolved once for the whole suite.

    ``from_cfg_model`` is the primary constructor: the model CONFIG carries
    the lists for every family, whereas the Pangu model classes expose only
    channel counts (``num_surface_vars`` etc.) — reading wrapper attributes
    would AttributeError for them. ``from_wrapper`` exists for stubs/tests.
    """

    surface: list = field(default_factory=list)
    upper_air: list = field(default_factory=list)
    diagnostic: list = field(default_factory=list)
    levels: list = field(default_factory=list)

    @classmethod
    def from_cfg_model(cls, cfg_model) -> "VariableCatalog":
        def _lst(key):
            v = cfg_model.get(key, None)
            return [str(x) for x in v] if v is not None else []

        levels = cfg_model.get("levels", None)
        return cls(
            surface=_lst("surface_variables"),
            upper_air=_lst("upper_air_variables"),
            diagnostic=_lst("diagnostic_variables"),
            levels=[float(x) for x in levels] if levels is not None else [],
        )

    @classmethod
    def from_wrapper(cls, wrapper) -> "VariableCatalog":
        inner = _inner_wrapper(wrapper)
        return cls(
            surface=list(getattr(inner, "surface_variables", [])),
            upper_air=list(getattr(inner, "upper_air_variables", [])),
            diagnostic=list(getattr(inner, "diagnostic_variables", [])),
            levels=list(getattr(inner, "levels", [])),
        )


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class ClimatologyScorer:
    """Time-mean + per-bin climatology maps, bias maps, global-bias scalars.

    Absorbs the old BiasValidator: ``global_bias`` (lat-weighted global
    scalars of the bias maps) is always emitted alongside the maps — it costs
    milliseconds and used to cost a second 2.77 h rollout. ``track_bins=False``
    skips the (memory-heavy, f64 x n_bins) binned aggregators and emits
    means/bias/global_bias only — the old "bias-only" configuration.
    """

    def __init__(self, *, n_bins: int = 12, steps_per_bin: int = 1,
                 track_bins: bool = True):
        self.n_bins = int(n_bins)
        self.steps_per_bin = max(1, int(steps_per_bin))
        self.track_bins = bool(track_bins)
        self._pred: dict = {}
        self._truth: dict = {}

    def bind(self, drive: DiffusionRolloutValidator) -> None:
        sample = drive.dataset[0]
        shapes = {"surface": tuple(sample["surface_in"].shape)}
        if drive.has_upper_air:
            shapes["upper_air"] = tuple(sample["upper_air_in"].shape)
        if drive.rmse_diagnostic is not None and "diagnostic" in sample:
            shapes["diagnostic"] = tuple(sample["diagnostic"].shape)

        def _aggs(shape):
            out = {"mean": StreamingTimeMean(shape, drive.device)}
            if self.track_bins:
                out["binned"] = StreamingBinnedMean(self.n_bins, shape, drive.device)
            return out

        self._pred = {kind: _aggs(sh) for kind, sh in shapes.items()}
        self._truth = {kind: _aggs(sh) for kind, sh in shapes.items()}

    def score_step(self, ctx: StepContext) -> None:
        if ctx.kind not in self._pred:
            return
        self._pred[ctx.kind]["mean"].update(ctx.pred_phys)
        self._truth[ctx.kind]["mean"].update(ctx.truth_phys)
        if self.track_bins:
            bin_value = (ctx.m_idx // self.steps_per_bin) % self.n_bins
            bin_idx = torch.full(
                (ctx.pred_phys.shape[0],), bin_value,
                dtype=torch.long, device=ctx.pred_phys.device,
            )
            self._pred[ctx.kind]["binned"].update(ctx.pred_phys, bin_idx)
            self._truth[ctx.kind]["binned"].update(ctx.truth_phys, bin_idx)

    def finalize(self, *, local_only: bool = False) -> dict[str, dict]:
        maps: dict[str, torch.Tensor] = {}
        global_bias: dict[str, torch.Tensor] = {}
        for kind in self._pred:
            pred_mean = self._pred[kind]["mean"].finalize(local_only=local_only)
            truth_mean = self._truth[kind]["mean"].finalize(local_only=local_only)
            bias = pred_mean - truth_mean
            maps[f"{kind}_pred_mean"] = pred_mean
            maps[f"{kind}_truth_mean"] = truth_mean
            maps[f"{kind}_bias"] = bias
            if self.track_bins:
                maps[f"{kind}_pred_binned"] = self._pred[kind]["binned"].finalize(
                    local_only=local_only
                )
                maps[f"{kind}_truth_binned"] = self._truth[kind]["binned"].finalize(
                    local_only=local_only
                )
            global_bias[kind] = lat_weighted_global_scalars(bias)
        return {"climatology": maps, "global_bias": global_bias}


class QBOScorer:
    """Tropical zonal-mean stratospheric U-wind timeseries + period estimate.

    Names/levels resolve against the :class:`VariableCatalog` (BY VALUE for
    levels), so Pangu-family checkpoints — whose classes expose only channel
    counts — work as long as ``cfg.model`` lists the variables.
    """

    def __init__(self, *, catalog: VariableCatalog,
                 u_variable_name: str = "u_component_of_wind",
                 qbo_levels: Sequence[float] = (10.0, 30.0, 50.0),
                 tropical_band_deg: float = 30.0,
                 steps_per_bin: int = 120,
                 months_per_bin: float = 1.0,
                 n_bins: Optional[int] = None):
        if u_variable_name not in catalog.upper_air:
            raise ValueError(
                f"u_variable_name={u_variable_name!r} not in the catalog's "
                f"upper_air variables {list(catalog.upper_air)}"
            )
        self.u_idx = list(catalog.upper_air).index(u_variable_name)
        levels = list(catalog.levels)
        missing = [lvl for lvl in qbo_levels if lvl not in levels]
        if missing:
            raise ValueError(
                f"qbo_levels {missing} not found in catalog levels={levels}"
            )
        self.qbo_levels = list(qbo_levels)
        self.level_indices = [levels.index(lvl) for lvl in self.qbo_levels]
        self.tropical_band_deg = float(tropical_band_deg)
        self.steps_per_bin = max(1, int(steps_per_bin))
        self.months_per_bin = float(months_per_bin)
        self._n_bins_cfg = n_bins
        self.n_bins = 0
        self.lat_mask = self.lat_weights = None
        self._pred = self._truth = None

    def bind(self, drive: DiffusionRolloutValidator) -> None:
        self.n_bins = (
            int(self._n_bins_cfg) if self._n_bins_cfg is not None
            else drive.horizon // self.steps_per_bin + 1
        )
        self.lat_mask, self.lat_weights = _tropical_band_mask_and_weights(
            drive.n_lat, self.tropical_band_deg, drive.device, torch.float32
        )
        shape = (len(self.level_indices),)
        self._pred = StreamingBinnedMean(self.n_bins, shape, drive.device)
        self._truth = StreamingBinnedMean(self.n_bins, shape, drive.device)

    def _band_mean(self, field_phys: torch.Tensor) -> torch.Tensor:
        """``(B, Cu, L, H, W) -> (B, n_levels)`` zonal + tropical-band mean."""
        u_field = field_phys[:, self.u_idx][:, self.level_indices]
        zonal = u_field.mean(dim=-1)
        masked = zonal[:, :, self.lat_mask]
        return (masked * self.lat_weights.to(masked.dtype)).sum(dim=-1)

    def score_step(self, ctx: StepContext) -> None:
        if ctx.kind != "upper_air":
            return
        pred_band = self._band_mean(ctx.pred_phys)
        truth_band = self._band_mean(ctx.truth_phys)
        bin_value = (ctx.m_idx // self.steps_per_bin) % self.n_bins
        bin_idx = torch.full(
            (pred_band.shape[0],), bin_value,
            dtype=torch.long, device=pred_band.device,
        )
        self._pred.update(pred_band, bin_idx)
        self._truth.update(truth_band, bin_idx)

    def finalize(self, *, local_only: bool = False) -> dict[str, dict]:
        pred_ts = self._pred.finalize(local_only=local_only)
        truth_ts = self._truth.finalize(local_only=local_only)
        block: dict = {
            "qbo_pred_timeseries": pred_ts,
            "qbo_truth_timeseries": truth_ts,
        }
        for i, lvl in enumerate(self.qbo_levels):
            block[f"qbo_period_months_pred_hPa{int(lvl)}"] = _estimate_period_months(
                pred_ts[:, i], self.months_per_bin
            )
            block[f"qbo_period_months_truth_hPa{int(lvl)}"] = _estimate_period_months(
                truth_ts[:, i], self.months_per_bin
            )
        return {"qbo": block}


class FluxSeriesScorer:
    """Lat-weighted global-mean per-step series for named flux channels.

    ``(horizon,)``-shaped f64 sum + count buffers keyed by ``m_idx``,
    all-reduced at finalize — correct under member-split (identical values on
    every rank; the sum/count ratio is invariant) AND under IC-split (a true
    combined mean). The old validator kept rank-local Python lists whose
    length scaled with the IC batch count and were never reduced.
    """

    def __init__(self, *, catalog: VariableCatalog, flux_variables: Sequence[str]):
        self.flux_variables = list(flux_variables)
        self._flux_index: dict[str, tuple[str, int]] = {}
        for name in self.flux_variables:
            if name in catalog.surface:
                self._flux_index[name] = ("surface", list(catalog.surface).index(name))
            elif name in catalog.diagnostic:
                self._flux_index[name] = (
                    "diagnostic", list(catalog.diagnostic).index(name)
                )
            else:
                raise ValueError(
                    f"flux variable {name!r} not found in the catalog's surface "
                    f"or diagnostic variables"
                )
        self._sums: dict = {}
        self._counts: dict = {}

    def bind(self, drive: DiffusionRolloutValidator) -> None:
        n = drive.horizon
        for name in self.flux_variables:
            self._sums[name] = {
                "pred": torch.zeros(n, device=drive.device, dtype=torch.float64),
                "truth": torch.zeros(n, device=drive.device, dtype=torch.float64),
            }
            self._counts[name] = torch.zeros(
                n, device=drive.device, dtype=torch.float64
            )

    def score_step(self, ctx: StepContext) -> None:
        for name, (grp, idx) in self._flux_index.items():
            if grp != ctx.kind:
                continue
            pred_scalars = lat_weighted_global_scalars(ctx.pred_phys[:, idx])
            truth_scalars = lat_weighted_global_scalars(ctx.truth_phys[:, idx])
            self._sums[name]["pred"][ctx.m_idx] += pred_scalars.double().sum()
            self._sums[name]["truth"][ctx.m_idx] += truth_scalars.double().sum()
            self._counts[name][ctx.m_idx] += float(pred_scalars.numel())

    def finalize(self, *, local_only: bool = False) -> dict[str, dict]:
        from validate import _all_reduce_sum

        pred_out, truth_out = {}, {}
        for name in self.flux_variables:
            ps = self._sums[name]["pred"].clone()
            ts = self._sums[name]["truth"].clone()
            c = self._counts[name].clone()
            if not local_only:
                _all_reduce_sum(ps)
                _all_reduce_sum(ts)
                _all_reduce_sum(c)
            denom = c.clamp(min=1.0)
            pred_out[name] = (ps / denom).float()
            truth_out[name] = (ts / denom).float()
        return {
            "global_mean": {
                "flux_pred_series": pred_out,
                "flux_truth_series": truth_out,
            }
        }


def check_deterministic_ensemble(perturber, ensemble_size: int) -> None:
    """A deterministic model needs REAL perturbations to have an ensemble.

    ``replicate_only`` (or the default ``None`` -> ReplicateOnly resolution)
    relies on the model's own stochasticity — a diffusion property. On a
    deterministic model every member is then identical: spread is exactly
    zero and the "ensemble" is E copies of one forecast.
    """
    if ensemble_size > 1 and (
        perturber is None or isinstance(perturber, (ReplicateOnly, Deterministic))
    ):
        raise ValueError(
            f"ensemble_size={ensemble_size} with a deterministic model "
            f"requires perturber=gaussian_ic: replicate_only would produce "
            f"{ensemble_size} identical members and exactly zero spread."
        )


def derive_spread_skill(rmse_acc: dict, log_steps: Sequence[int]) -> dict[str, float]:
    """spread/skill ratios from the flat metric dict — pure post-processing.

    Emitted by the runner whenever ``ensemble_size > 1`` (the drive's spread
    metrics fill automatically then); the old EnsembleEnvelopeValidator ran a
    whole extra rollout to compute exactly this.
    """
    ratios: dict[str, float] = {}
    for group in ("surface", "upper_air", "diagnostic"):
        for step in log_steps:
            spread_key = f"spread_step{step}_{group}"
            rmse_key = f"rmse_step{step}_{group}"
            if spread_key in rmse_acc and rmse_key in rmse_acc:
                rmse = rmse_acc[rmse_key]
                ratios[f"spread_skill_ratio_step{step}_{group}"] = (
                    rmse_acc[spread_key] / rmse if rmse > 0 else float("nan")
                )
    return ratios


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _is_combined_model_cfg(model_cfg) -> bool:
    """True when cfg.model is a two-block cascade config (amip_combined*.yaml):
    ``forecaster:`` and ``downscaler:`` each carrying {model, sampler,
    checkpoint}. Such a config cannot go through build_model()."""
    try:
        return (
            model_cfg is not None
            and "forecaster" in model_cfg
            and "downscaler" in model_cfg
        )
    except TypeError:
        return False


#: Upstream amip_v2's headline fields (bias.py HEADLINE_LEVELS + HEADLINE_2D):
#: [label, variable name, pressure level (hPa) or None].
DEFAULT_HEADLINE_VARIABLES = (
    ("z500", "geopotential", 500),
    ("u250", "u_component_of_wind", 250),
    ("t850", "temperature", 850),
    ("q850", "specific_humidity", 850),
    ("t2m", "2m_temperature", None),
    ("prate", "PRATEsfc_24h", None),
    ("q2m", "2m_specific_humidity", None),
    ("u10m", "10m_u_component_of_wind", None),
    ("v10m", "10m_v_component_of_wind", None),
)


def compute_headline_bias(clim_maps: dict, catalog: VariableCatalog,
                          spec) -> dict:
    """Upstream bias.py's headline statistics from the finalized bias maps.

    For each ``(label, name, level)``: lat-weighted MEAN and lat-weighted RMSE
    of the time-mean bias map — cos-lat weights normalized to mean 1, the same
    convention as upstream's ``headline_bias`` (and as
    ``lat_weighted_global_scalars``, which is reused for both reductions:
    ``rmse = sqrt(weighted_mean(bias**2))``). Levels are resolved BY VALUE
    against the catalog (never by hard index — the level axis is the model
    config's list); names fall back surface -> diagnostic, and an unknown
    name or level raises rather than silently skipping.
    """
    out: dict = {}
    levels = [float(lv) for lv in catalog.levels]
    for entry in spec:
        label, name = str(entry[0]), str(entry[1])
        level = entry[2] if len(entry) > 2 else None
        if level is not None:
            if name not in catalog.upper_air:
                raise ValueError(
                    f"headline {label}: {name!r} not in upper_air variables "
                    f"{catalog.upper_air}"
                )
            if float(level) not in levels:
                raise ValueError(
                    f"headline {label}: level {level} hPa not in the model's "
                    f"levels {levels}"
                )
            bias = clim_maps["upper_air_bias"][
                catalog.upper_air.index(name), levels.index(float(level))
            ]
        elif name in catalog.surface:
            bias = clim_maps["surface_bias"][catalog.surface.index(name)]
        elif name in catalog.diagnostic:
            if "diagnostic_bias" not in clim_maps:
                raise ValueError(
                    f"headline {label}: {name!r} is a diagnostic but the "
                    f"climatology maps carry no diagnostic_bias block"
                )
            bias = clim_maps["diagnostic_bias"][catalog.diagnostic.index(name)]
        else:
            raise ValueError(
                f"headline {label}: {name!r} not in surface "
                f"{catalog.surface} or diagnostic {catalog.diagnostic}"
            )
        b = bias.unsqueeze(0).double()
        mean = float(lat_weighted_global_scalars(b)[0])
        rmse = float(lat_weighted_global_scalars(b ** 2)[0].sqrt())
        out[label] = {"mean_bias": mean, "rmse_bias": rmse}
    return out


def format_headline_table(headline: dict) -> list[str]:
    """Fixed-width lines mirroring upstream bias.py's print_headline."""
    rows = ["Headline climatological bias (lat-weighted, predicted - truth):"]
    for label, s in headline.items():
        rows.append(
            f"  {label:<6} mean {s['mean_bias']:+12.5g}   "
            f"rmse {s['rmse_bias']:12.5g}"
        )
    return rows


class EvalSuiteRunner:
    """One rollout, every scorer, save-as-you-go, schema-v2 results.

    Partial snapshots use ``finalize(local_only=True)`` everywhere — ZERO
    collectives, so they cannot desync ranks whatever the IC/member split —
    and are exact under member-split (each rank's accumulators hold identical
    cross-rank-reduced content) or rank-0's-ICs-only under IC-split (flagged
    in ``_progress.rank_local_only``). The one true collective finalize runs
    exactly once at the end, on every rank; only rank 0 writes.

    Known caveat (pre-existing): ``climatology._assert_finite`` raising on
    ONE rank mid-run desyncs the per-step member-split collectives — the
    surviving ranks block until the backend times out. A loud single-rank
    error is still preferable to silently NaN results.
    """

    def __init__(self, drive: DiffusionRolloutValidator, scorers: Sequence, *,
                 output_path: str, partial_save_every_frames: int = 0,
                 rank: int = 0, config_echo: Optional[dict] = None, log=None,
                 headline_spec=None, catalog: Optional[VariableCatalog] = None):
        self.drive = drive
        self.scorers = list(scorers)
        self.output_path = str(output_path)
        self.partial_every = int(partial_save_every_frames)
        self.rank = int(rank)
        self.config_echo = dict(config_echo or {})
        self.headline_spec = list(headline_spec) if headline_spec else None
        self.catalog = catalog
        self.log = log or logger
        self._frames_scored = 0
        if self.partial_every > 0:
            drive.on_frame_scored = self._on_frame_scored

    # -- partial snapshots ------------------------------------------------ #
    def _on_frame_scored(self, k: int) -> None:
        self._frames_scored += 1
        if self.rank != 0:
            return
        if self._frames_scored % self.partial_every != 0:
            return
        results = self._assemble(partial=True)
        torch.save(_to_cpu(results), self.output_path)

    def _assemble(self, *, partial: bool, rmse_acc: Optional[dict] = None) -> dict:
        local_only = partial
        if rmse_acc is None:
            rmse_acc = self.drive._finalize(local_only=local_only)
        results: dict = {
            "schema_version": 2,
            "config": dict(self.config_echo),
            "rmse_acc": rmse_acc,
            "stability": scan_rmse_trace(rmse_acc),
        }
        for scorer in self.scorers:
            results.update(scorer.finalize(local_only=local_only))
        if (
            self.headline_spec
            and self.catalog is not None
            and "climatology" in results
        ):
            results["headline"] = compute_headline_bias(
                results["climatology"], self.catalog, self.headline_spec
            )
        if self.drive.ensemble_size > 1:
            results["spread_skill"] = derive_spread_skill(
                rmse_acc, self.drive.log_steps
            )
        if partial:
            results["_partial"] = True
            results["_progress"] = {
                "frames_scored": self._frames_scored,
                "rank_local_only": not self.drive.member_split
                and self.drive._world_size > 1,
            }
        return results

    # -- the run ----------------------------------------------------------- #
    def run(self, model) -> dict:
        if self.rank == 0:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        rmse_acc = self.drive.run(model)          # collective finalize inside
        results = self._assemble(partial=False, rmse_acc=rmse_acc)
        if self.rank == 0:
            for group, s in results["stability"].items():
                if s["n_nonfinite"]:
                    self.log.warning(
                        f"stability: {group} trace has {s['n_nonfinite']} "
                        f"NON-FINITE step(s), first at step "
                        f"{s['first_nonfinite_step']} — the rollout overflowed"
                    )
                for step, val, med in s["jumps"]:
                    self.log.warning(
                        f"stability: {group} RMSE jump at step {step} — "
                        f"{val:.4g} vs trailing median {med:.4g}"
                    )
            if "headline" in results:
                for line in format_headline_table(results["headline"]):
                    self.log.info(line)
            torch.save(_to_cpu(results), self.output_path)
            self.log.info(f"wrote eval suite results to {self.output_path}")
        return results


# ---------------------------------------------------------------------------
# Generation-agnostic Hydra driver
# ---------------------------------------------------------------------------


def _build_eval_dataset(cfg, log):
    """The shared dataset, preferring the VALIDATION store when configured.

    climatology_cli.py and inference.py already prefer ``val_zarr_path``; the
    old eval_diffusion silently rolled on the training store. The fallback
    keeps every existing eval command working.
    """
    from omegaconf import open_dict

    from train_diffusion import _build_dataset

    cfg2 = cfg.copy()
    val_path = cfg.dataset.get("val_zarr_path", None)
    if val_path:
        with open_dict(cfg2.dataset):
            cfg2.dataset.zarr_path = val_path
            val_bnd = cfg.dataset.get("val_boundary_zarr_path", None)
            if val_bnd:
                cfg2.dataset.boundary_zarr_path = val_bnd
        log.info(f"eval store: dataset.val_zarr_path = {val_path}")
    else:
        log.info(f"eval store: dataset.zarr_path = {cfg.dataset.zarr_path} "
                 f"(no val_zarr_path configured)")
    return _build_dataset(cfg2), cfg2


def main(cfg) -> None:
    """Runs the eval suite. Wrapped by :func:`cli` (kept as a plain function
    so it stays unit-testable without invoking Hydra)."""
    import warnings

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from physicsnemo.utils.logging import PythonLogger
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.datapipes.climate import ClimateNormalizer

    from inference import _model_optional_kwarg_names  # noqa: E402
    from train import _resolve_path, build_model  # noqa: E402
    from train_diffusion import _make_perturber  # noqa: E402
    from train_loop import (  # noqa: E402
        adopt_ocean_contract,
        assert_checkpoint_dir_contract,
        model_step_rows,
        steps_per_month,
    )

    from deterministic_adapter import (  # noqa: E402
        DeterministicPackShim,
        DeterministicStepAdapter,
    )

    DistributedManager.initialize()
    dist = DistributedManager()
    log = PythonLogger("climate_eval_suite")
    # Per-rank seed: decorrelates the schedulers' randn_like member noise
    # across ranks under member-split ensembles, and makes single-GPU evals
    # reproducible.
    torch.manual_seed(int(cfg.seed) + dist.rank)

    eval_cfg = cfg.get("eval_suite", None)
    if eval_cfg is None:
        raise ValueError(
            "eval_suite.* config block missing — select "
            "validation=eval_suite on the Hydra command line. (That config "
            "carries a '# @package eval_suite' directive on its first line, "
            "which is what routes it here from the validation group; if this "
            "raises with validation=eval_suite selected, check that the "
            "directive is still line 1 of conf/validation/eval_suite.yaml.)"
        )

    horizon = int(eval_cfg.horizon)
    ensemble_size = int(eval_cfg.get("ensemble_size", 1))
    perturber = (
        _make_perturber(
            str(eval_cfg.perturber),
            _perturber_scales(eval_cfg.get("perturber_scales", None)),
        )
        if eval_cfg.get("perturber", None) is not None
        else None
    )

    # Upstream-parity cascade hooks — populated only on the combined path.
    frame_transform = None
    unpack_wrapper = None
    init_downsample_factor = None

    combined_spec = cfg.model if _is_combined_model_cfg(cfg.model) else None
    if combined_spec is not None:
        # ── Combined (forecaster + downscaler) cascade ──────────────────────
        # Upstream amip_v2's bias protocol: the coarse forecaster streams, each
        # emitted frame is downscaled to the full grid, and the climatology is
        # scored there. cfg.model is the two-block {forecaster, downscaler}
        # config (amip_combined*.yaml); the DATASET, catalog and step
        # resolution all key off the FORECASTER's model config, loaded the
        # same way rollout.py's cascade does. No import cycle: rollout.py
        # never imports this module.
        from omegaconf import open_dict

        from rollout import _build_stage, _load_group  # noqa: E402
        from physicsnemo.experimental.models.amip_si import CombinedModule

        f_model_cfg = _load_group("model", str(combined_spec.forecaster.model))
        cfg_eff = cfg.copy()
        with open_dict(cfg_eff):
            cfg_eff.model = f_model_cfg
        raw_ds, ds_cfg = _build_eval_dataset(cfg_eff, log)
        forecaster, f_sched = _build_stage(
            combined_spec.forecaster, device=dist.device, log=log
        )
        downscaler, d_sched = _build_stage(
            combined_spec.downscaler, device=dist.device, log=log
        )
        adopt_ocean_contract(f_sched, forecaster)
        combined = CombinedModule(
            forecaster=forecaster,
            forecaster_scheduler=f_sched,
            downscaler=downscaler,
            downscaler_scheduler=d_sched,
        ).to(dist.device)
        combined.eval()
        wrapper = forecaster
        model_cfg_for_catalog = f_model_cfg
        scheduler = f_sched
        drive_model = forecaster
        is_diffusion = True
        sampler_num_steps = _resolve_eval_sampler_num_steps(
            eval_cfg.get("sampler_num_steps", None)
        )
        _d_steps = eval_cfg.get("downscaler_num_steps", None)
        _d_steps = int(_d_steps) if _d_steps is not None else None

        def frame_transform(x, _c=combined, _s=f_sched, _n=_d_steps):
            # Strip the predicted ocean tail, then downscale to the scoring
            # grid — parity with CombinedModule.windowed_step (the tail is a
            # diagnostic block the downscaler was never trained on).
            return _c._downscale(_s.strip_ocean(x), num_steps=_n)

        unpack_wrapper = downscaler
        # The forecaster runs at the coarse grid but the drive's dataset is
        # the full-res store: downsample the oracle init window's STATE by
        # the downscaler's own factor (the coarse store's build operator).
        init_downsample_factor = int(
            eval_cfg.get("init_downsample_factor", None)
            or getattr(downscaler, "downsample_factor", 4)
        )
        if getattr(raw_ds, "forcing_pipeline", None) is not None:
            raw_ds.forcing_pipeline.assert_matches(
                forecaster, name="model.forecaster"
            )
        ckpt_dir = (
            f"{combined_spec.forecaster.checkpoint} + "
            f"{combined_spec.downscaler.checkpoint}"
        )
        catalog = VariableCatalog.from_cfg_model(model_cfg_for_catalog)
        has_diagnostic = len(catalog.diagnostic) > 0
    else:
        raw_ds, ds_cfg = _build_eval_dataset(cfg, log)
        wrapper = build_model(cfg.model).to(dist.device)
        ckpt_dir = _resolve_path(str(eval_cfg.checkpoint_dir))
        assert_checkpoint_dir_contract(wrapper, ckpt_dir, log=log)
        loaded_epoch = load_checkpoint(ckpt_dir, models=wrapper, device=dist.device)
        log.info(f"loaded checkpoint epoch={loaded_epoch} from {ckpt_dir}")
        wrapper.eval()
        if getattr(raw_ds, "forcing_pipeline", None) is not None:
            # No-op for models without a c_grid contract (deterministic families).
            raw_ds.forcing_pipeline.assert_matches(wrapper, name="cfg.model")

        catalog = VariableCatalog.from_cfg_model(cfg.model)
        has_diagnostic = len(catalog.diagnostic) > 0

        # ── family dispatch ──────────────────────────────────────────────
        # Diffusion wrappers expose a pack surface: pack_state (single-step
        # AmipDiTWrapper) OR pack_window_state (the rolling
        # _RollingPackUnpackMixin family — RollingDiTWrapper/ERDMWrapper have
        # NO pack_state, which is why inference.py's pack_state-only signal
        # is insufficient here; caught by the fused-suite regression on the
        # RSI fancy checkpoint, Midway job 54834641). Deterministic families
        # (SFNO/Pangu/ArchesWeather) expose neither.
        _inner = _inner_wrapper(wrapper)
        is_diffusion = hasattr(_inner, "pack_state") or hasattr(
            _inner, "pack_window_state"
        )
        if is_diffusion:
            import hydra as _hydra

            scheduler = _hydra.utils.instantiate(cfg.loss).to(dist.device)
            adopt_ocean_contract(scheduler, wrapper)
            drive_model = wrapper
            sampler_num_steps = _resolve_eval_sampler_num_steps(
                eval_cfg.get("sampler_num_steps", None)
            )
        else:
            # cfg.loss is never instantiated here — deterministic loss configs
            # (mse.yaml etc.) carry no _target_ and would break instantiate.
            probe = raw_ds[0]
            shim = DeterministicPackShim(
                wrapper,
                catalog=catalog,
                n_constant=int(probe["constant_boundary"].shape[0]),
                n_varying=int(probe["varying_boundary"].shape[0]),
                has_diagnostic=has_diagnostic and "diagnostic" in probe,
            ).to(dist.device)
            scheduler = DeterministicStepAdapter(
                shim, optional_kwargs=_model_optional_kwarg_names(wrapper)
            )
            drive_model = shim
            sampler_num_steps = None
            if eval_cfg.get("sampler_num_steps", None) is not None:
                log.info("sampler_num_steps ignored for a deterministic checkpoint")
            check_deterministic_ensemble(perturber, ensemble_size)

    # The scoring-side denormalizer must live on the MODEL's level set: when
    # the pressure-level subset fired in _build_dataset (17-level model on the
    # 18-level archive), the rollout tensors carry the model's levels, and an
    # 18-level std would fail the broadcast (or worse, misalign silently if
    # the counts happened to match).
    norm_kwargs = {}
    data_levels = list(getattr(raw_ds, "pressure_levels", []) or [])
    if catalog.levels and data_levels:
        model_set = set(map(float, catalog.levels))
        if model_set != set(map(float, data_levels)) and model_set.issubset(
            set(map(float, data_levels))
        ):
            norm_kwargs["pressure_levels"] = [float(lv) for lv in catalog.levels]
    normalizer = ClimateNormalizer.from_dataset(
        raw_ds,
        mean_path=_resolve_path(ds_cfg.dataset.mean_path),
        std_path=_resolve_path(ds_cfg.dataset.std_path),
        normalize_constant_boundary=bool(
            ds_cfg.dataset.get("normalize_constant_boundary", False)
        ),
        normalize_diagnostic=bool(ds_cfg.dataset.get("normalize_diagnostic", False)),
        **norm_kwargs,
    ).to(dist.device)

    if dist.world_size > 1 and ensemble_size == 1 and dist.rank == 0:
        log.warning(
            f"world_size={dist.world_size} but eval_suite.ensemble_size=1: "
            f"ranks beyond the IC count idle. Set ensemble_size to a "
            f"multiple of world_size to split members across GPUs."
        )

    # ── scorers from the config blocks ─────────────────────────────────────
    per_month = steps_per_month(ds_cfg, raw_ds)
    scorers: list = []

    clim_cfg = eval_cfg.get("climatology", None)
    bias_cfg = eval_cfg.get("bias", None)
    clim_on = clim_cfg is not None and bool(clim_cfg.get("enabled", False))
    bias_on = bias_cfg is not None and bool(bias_cfg.get("enabled", False))
    if bias_on and bias_cfg.get("n_bins", None) is not None and clim_on:
        if int(bias_cfg.n_bins) != int(clim_cfg.get("n_bins", 12)):
            log.warning(
                "bias.n_bins is DEPRECATED (bias is computed from the "
                "climatology scorer's maps in the fused suite) and disagrees "
                "with climatology.n_bins — the climatology value is used."
            )
    if clim_on or bias_on:
        block = clim_cfg if clim_on else bias_cfg
        steps = resolve_steps_per_bin(
            block, per_month, name="climatology" if clim_on else "bias", log=log
        )
        log.info(f"climatology: {steps} steps/bin")
        scorers.append(
            ClimatologyScorer(
                n_bins=int(block.get("n_bins", 12)),
                steps_per_bin=steps,
                # bias-only configuration: means/bias/global_bias without the
                # heavy per-bin aggregators.
                track_bins=clim_on,
            )
        )

    qbo_cfg = eval_cfg.get("qbo", None)
    if qbo_cfg is not None and bool(qbo_cfg.get("enabled", False)):
        steps = resolve_steps_per_bin(qbo_cfg, per_month, name="qbo", log=log)
        log.info(f"qbo: {steps} steps/bin")
        scorers.append(
            QBOScorer(
                catalog=catalog,
                u_variable_name=str(
                    qbo_cfg.get("u_variable_name", "u_component_of_wind")
                ),
                qbo_levels=[float(x) for x in qbo_cfg.get("levels", [10, 30, 50])],
                steps_per_bin=steps,
                months_per_bin=float(qbo_cfg.get("months_per_bin", 1.0) or 1.0),
            )
        )

    gm_cfg = eval_cfg.get("global_mean", None)
    if gm_cfg is not None and bool(gm_cfg.get("enabled", False)):
        scorers.append(
            FluxSeriesScorer(
                catalog=catalog,
                flux_variables=[str(x) for x in gm_cfg.flux_variables],
            )
        )

    ens_cfg = eval_cfg.get("ensemble_envelope", None)
    if ens_cfg is not None and bool(ens_cfg.get("enabled", False)):
        log.warning(
            "eval_suite.ensemble_envelope is REMOVED: spread and spread_skill "
            "are emitted automatically whenever the top-level ensemble_size "
            "> 1 (one shared rollout). Its ensemble_size/perturber keys are "
            "IGNORED — set the top-level eval_suite.ensemble_size instead."
        )

    # ── calendar-pinned initial condition ──────────────────────────────────
    ic_indices = None
    ic_date = eval_cfg.get("ic_date", None)
    ic_resolved_time = None
    if ic_date:
        # Reuse inference.py's date->global-index machinery (cftime- and
        # datetime64-aware; multi-year stores concatenated chronologically).
        from inference import _full_time_coord, resolve_init_schedule

        y, m, d = (int(x) for x in str(ic_date).split("-"))
        times = _full_time_coord(raw_ds)
        matches = resolve_init_schedule(
            times, months=[m], days=[d], hours=[0], years=[y]
        )
        if len(matches) != 1:
            raise ValueError(
                f"eval_suite.ic_date={ic_date} resolved to {len(matches)} "
                f"row(s) in the store — expected exactly one 00Z frame."
            )
        ic_indices = matches
        ic_resolved_time = str(times[matches[0]])
        log.info(
            f"ic_date {ic_date} -> global row {matches[0]} ({ic_resolved_time})"
        )

    # ── headline (upstream bias.py's scalar table) ──────────────────────────
    headline_cfg = eval_cfg.get("headline", None)
    headline_spec = None
    if headline_cfg is not None and bool(headline_cfg.get("enabled", False)):
        raw_spec = headline_cfg.get("variables", None)
        headline_spec = (
            [list(e) for e in raw_spec]
            if raw_spec is not None
            else [list(e) for e in DEFAULT_HEADLINE_VARIABLES]
        )
        if not any(
            isinstance(s, ClimatologyScorer) for s in scorers
        ):
            raise ValueError(
                "eval_suite.headline needs the climatology (or bias) scorer "
                "enabled — the headline reduces its finalized bias maps."
            )

    # ── drive + runner ─────────────────────────────────────────────────────
    drive = DiffusionRolloutValidator(
        raw_ds,
        wrapper=drive_model,
        inference_scheduler=scheduler,
        log_steps=list(range(1, horizon + 1)),
        horizon=horizon,
        device=dist.device,
        has_diagnostic=has_diagnostic,
        max_initial_conditions=int(eval_cfg.get("max_initial_conditions", 1)),
        ic_stride=int(eval_cfg.get("ic_stride", 1)),
        step_size=model_step_rows(ds_cfg, raw_ds),
        batch_size=int(eval_cfg.get("batch_size", 1)),
        normalizer=normalizer,
        sampler_num_steps=sampler_num_steps,
        seed=int(cfg.seed),
        ensemble_size=ensemble_size,
        perturber=perturber,
        split_ensemble_across_ranks=(dist.world_size > 1 and ensemble_size > 1),
        scorers=scorers,
        ic_indices=ic_indices,
        frame_transform=frame_transform,
        unpack_wrapper=unpack_wrapper,
        init_downsample_factor=init_downsample_factor,
    )
    runner = EvalSuiteRunner(
        drive,
        scorers,
        output_path=_resolve_path(str(eval_cfg.output_path)),
        partial_save_every_frames=int(
            eval_cfg.get("partial_save_every_frames", 0) or 0
        ),
        rank=dist.rank,
        config_echo={
            "horizon": horizon,
            "ensemble_size": ensemble_size,
            "perturber": (
                str(eval_cfg.perturber)
                if eval_cfg.get("perturber", None) is not None else None
            ),
            "max_initial_conditions": int(eval_cfg.get("max_initial_conditions", 1)),
            "ic_stride": int(eval_cfg.get("ic_stride", 1)),
            "step_size": model_step_rows(ds_cfg, raw_ds),
            "sampler_num_steps": sampler_num_steps,
            "model_family": (
                "combined" if combined_spec is not None
                else "diffusion" if is_diffusion else "deterministic"
            ),
            "checkpoint_dir": str(ckpt_dir),
            "zarr_path": str(ds_cfg.dataset.zarr_path),
            "ic_date": str(ic_date) if ic_date else None,
            "ic_index": ic_indices[0] if ic_indices else None,
            "ic_resolved_time": ic_resolved_time,
            "downscaler_num_steps": (
                int(eval_cfg.downscaler_num_steps)
                if eval_cfg.get("downscaler_num_steps", None) is not None
                else None
            ),
            "init_downsample_factor": init_downsample_factor,
            "scoring_grid": (
                "x".join(str(int(v)) for v in raw_ds[0]["surface_in"].shape[-2:])
            ),
            # The 1996-2001 parity span lies INSIDE both checkpoints'
            # 1979-2015 training window — same as upstream's own protocol;
            # recorded so a saved .pt is honest about in-sample scoring.
            "in_sample_note": (
                "IC+horizon lie inside the checkpoints' training span"
                if ic_date else None
            ),
        },
        log=log,
        headline_spec=headline_spec,
        catalog=catalog,
    )
    runner.run(drive_model)


def cli() -> None:
    import hydra

    # ABSOLUTE config_path: with a relative one, hydra resolves it against the
    # task function's defining MODULE — and since main() lives here (an
    # imported module when invoked through the eval_diffusion.py alias, i.e.
    # not __main__), hydra switches to module-based resolution and demands
    # conf/ be an importable package ("Primary config module 'conf' not
    # found... __init__.py"). The absolute path keeps BOTH entrypoints on
    # plain file-based resolution (regression: Midway job 54953980).
    hydra.main(
        version_base="1.2",
        config_path=str(Path(__file__).resolve().parent / "conf"),
        config_name="config",
    )(main)()


if __name__ == "__main__":
    cli()
