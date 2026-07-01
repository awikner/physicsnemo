# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Long-horizon climate eval suite for the AMIP diffusion recipes (Phase 8f, F5).

:class:`~validate_diffusion.DiffusionRolloutValidator` already knows how
to drive a (single-step or rolling-window) diffusion model
autoregressively from an IC, dispatch ensemble perturbation, and score
discrete lead times. The five validators here subclass it and simply
change *what gets scored*: instead of a handful of ``log_steps``, every
emitted frame of a long (e.g. one validation year) rollout is scored,
and the per-frame ``(pred, truth)`` pairs are routed into the Phase 4c
streaming aggregators (:mod:`climatology`) instead of (or in addition
to) RMSE/ACC/spread. This reuses the rollout mechanics, IC selection,
ensemble handling, denormalization, and DDP-safety already implemented
and tested in :mod:`validate_diffusion`.

* :class:`ClimatologyValidator` — per-variable time-mean (+ per-bin,
  e.g. monthly, climatology) over a long rollout vs. the *dataset's own
  ground truth* trajectory. Matches upstream amip's own climatology
  convention (``evals/`` scripts compare against the ERA5-driven AMIP
  targets in the same Zarr, not a separately-shipped external oracle
  file).
* :class:`BiasValidator` — signed lat-weighted global mean bias per
  channel group, derived from :class:`ClimatologyValidator`'s bias
  field via :func:`climatology.lat_weighted_global_scalars`.
* :class:`QBOValidator` — 30°S-30°N zonal-mean U-wind at a few
  stratospheric pressure levels (10/30/50 hPa by default), binned to a
  monthly timeseries, plus a simple zero-crossing period estimate for
  pred vs. truth.
* :class:`GlobalMeanTimeseriesValidator` — lat-weighted global-mean
  per-step timeseries for a caller-specified set of flux channels
  (e.g. TOA / surface radiative fluxes), pred vs. truth.
* :class:`EnsembleEnvelopeValidator` — spread/skill ratio per (step,
  channel group), purely a ratio of the parent's own RMSE + spread
  aggregators — no new aggregation needed.

All five accept every constructor argument :class:`DiffusionRolloutValidator`
does except ``log_steps`` (fixed internally to score every emitted
frame).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from climatology import (  # noqa: E402
    StreamingBinnedMean,
    StreamingTimeMean,
    lat_weighted_global_scalars,
)
from validate_diffusion import DiffusionRolloutValidator  # noqa: E402


def _ensemble_mean(pred_ensemble: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    """``(B*E, ...) -> (B, ...)`` ensemble-mean reduction (identity if E=1)."""
    if ensemble_size <= 1:
        return pred_ensemble
    n_ic = pred_ensemble.shape[0] // ensemble_size
    rest = pred_ensemble.shape[1:]
    return pred_ensemble.view(n_ic, ensemble_size, *rest).mean(dim=1)


def _inner_wrapper(wrapper):
    return wrapper.module if hasattr(wrapper, "module") else wrapper


# ---------------------------------------------------------------------------
# ClimatologyValidator / BiasValidator
# ---------------------------------------------------------------------------


class ClimatologyValidator(DiffusionRolloutValidator):
    r"""Per-variable time-mean (+ per-bin climatology) over a long rollout.

    Scores every emitted frame of the rollout (``log_steps`` is fixed to
    ``1..horizon`` internally) and accumulates a :class:`~climatology.StreamingTimeMean`
    and :class:`~climatology.StreamingBinnedMean` per channel group, for
    both the prediction and the dataset's ground truth ("oracle")
    trajectory — same accumulator classes Phase 4c's
    ``climatology_cli.py`` uses for the deterministic recipe.

    Parameters
    ----------
    n_bins
        Number of climatology bins (e.g. 12 for monthly, 365 for daily).
    steps_per_bin
        Number of dataset steps per bin (e.g. 120 six-hourly steps ≈ 1
        month). Bin index for emitted frame ``k`` is
        ``(k // steps_per_bin) % n_bins``.
    Remaining parameters match :class:`DiffusionRolloutValidator`
    (everything except ``log_steps``).
    """

    def __init__(
        self,
        dataset,
        *,
        wrapper,
        inference_scheduler,
        horizon: int,
        device: torch.device,
        n_bins: int = 12,
        steps_per_bin: int = 1,
        **kwargs,
    ):
        super().__init__(
            dataset,
            wrapper=wrapper,
            inference_scheduler=inference_scheduler,
            log_steps=list(range(1, int(horizon) + 1)),
            horizon=horizon,
            device=device,
            **kwargs,
        )
        self.n_bins = int(n_bins)
        self.steps_per_bin = max(1, int(steps_per_bin))

        sample = dataset[0]
        shapes: dict[str, tuple[int, ...]] = {
            "surface": tuple(sample["surface_in"].shape)
        }
        if self.has_upper_air:
            shapes["upper_air"] = tuple(sample["upper_air_in"].shape)
        if self.rmse_diagnostic is not None and "diagnostic" in sample:
            shapes["diagnostic"] = tuple(sample["diagnostic"].shape)
        self._clim_shapes = shapes

        def _agg_pair(shape):
            return {
                "mean": StreamingTimeMean(shape, device),
                "binned": StreamingBinnedMean(self.n_bins, shape, device),
            }

        self._clim_pred = {kind: _agg_pair(sh) for kind, sh in shapes.items()}
        self._clim_truth = {kind: _agg_pair(sh) for kind, sh in shapes.items()}

    def _score_step(self, m_idx, pred_ensemble, truth, kind):
        super()._score_step(m_idx, pred_ensemble, truth, kind)
        if kind not in self._clim_pred:
            return
        pred_mean = _ensemble_mean(pred_ensemble, self.ensemble_size)
        pred_phys, truth_phys = self._denorm_pred_truth(kind, pred_mean, truth)
        bin_value = (m_idx // self.steps_per_bin) % self.n_bins
        bin_idx = torch.full(
            (pred_phys.shape[0],), bin_value, dtype=torch.long, device=pred_phys.device
        )
        self._clim_pred[kind]["mean"].update(pred_phys)
        self._clim_truth[kind]["mean"].update(truth_phys)
        self._clim_pred[kind]["binned"].update(pred_phys, bin_idx)
        self._clim_truth[kind]["binned"].update(truth_phys, bin_idx)

    def _finalize_climatology(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for kind in self._clim_pred:
            pred_mean = self._clim_pred[kind]["mean"].finalize()
            truth_mean = self._clim_truth[kind]["mean"].finalize()
            out[f"{kind}_pred_mean"] = pred_mean
            out[f"{kind}_truth_mean"] = truth_mean
            out[f"{kind}_bias"] = pred_mean - truth_mean
            out[f"{kind}_pred_binned"] = self._clim_pred[kind]["binned"].finalize()
            out[f"{kind}_truth_binned"] = self._clim_truth[kind]["binned"].finalize()
        return out

    def run(self, model, *, epoch: int = 0) -> dict:
        """Returns ``{"rmse_acc": {...float...}, "climatology": {...tensor...}}``."""
        rmse_acc = super().run(model, epoch=epoch)
        return {"rmse_acc": rmse_acc, "climatology": self._finalize_climatology()}


class BiasValidator(ClimatologyValidator):
    r"""Signed lat-weighted global mean bias vs. climatology.

    Thin extension of :class:`ClimatologyValidator` — reduces the
    per-pixel ``{group}_bias`` field (pred_mean - truth_mean) to a
    lat-weighted global scalar per channel (and level, for upper-air)
    via :func:`climatology.lat_weighted_global_scalars`.
    """

    def run(self, model, *, epoch: int = 0) -> dict:
        result = super().run(model, epoch=epoch)
        global_bias: dict[str, torch.Tensor] = {}
        for key, bias_field in result["climatology"].items():
            if not key.endswith("_bias"):
                continue
            group = key[: -len("_bias")]
            global_bias[group] = lat_weighted_global_scalars(bias_field)
        result["global_bias"] = global_bias
        return result


# ---------------------------------------------------------------------------
# QBOValidator
# ---------------------------------------------------------------------------


def _tropical_band_mask_and_weights(
    n_lat: int, band_deg: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mask, weights)`` for the ``[-band_deg, +band_deg]`` latitude band.

    Follows the same ``linspace(90, -90, n_lat)`` grid convention as
    :func:`validate.cos_lat_weights` / :func:`climatology.lat_weighted_global_scalars`.
    ``weights`` are ``cos(lat)``-weighted and normalized to sum to 1
    over the masked band.
    """
    import math

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


class QBOValidator(DiffusionRolloutValidator):
    r"""Tropical zonal-mean stratospheric U-wind vs. dataset ground truth.

    Reduces upper-air U-wind (channel ``u_variable_name``) at the
    requested pressure levels to a ``30°S-30°N`` zonal + lat-weighted
    mean scalar per emitted frame, bins it to a monthly (by default)
    timeseries for both prediction and truth, and reports a simple
    zero-crossing QBO-period estimate for each.

    Parameters
    ----------
    u_variable_name
        Name of the zonal-wind channel in ``wrapper.upper_air_variables``.
    qbo_levels
        Pressure levels (must match entries in ``wrapper.levels``
        exactly, same units — hPa in the AMIP configs).
    tropical_band_deg
        Half-width of the latitude band (default 30, i.e. 30°S-30°N).
    steps_per_bin, months_per_bin
        Bin width in dataset steps, and how many real-world months that
        many steps span (defaults assume 6-hourly steps: 120 steps ≈ 1
        month). Override both for other cadences.
    """

    def __init__(
        self,
        dataset,
        *,
        wrapper,
        inference_scheduler,
        horizon: int,
        device: torch.device,
        u_variable_name: str = "ua",
        qbo_levels: Sequence[float] = (10.0, 30.0, 50.0),
        tropical_band_deg: float = 30.0,
        steps_per_bin: int = 120,
        months_per_bin: float = 1.0,
        n_bins: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            wrapper=wrapper,
            inference_scheduler=inference_scheduler,
            log_steps=list(range(1, int(horizon) + 1)),
            horizon=horizon,
            device=device,
            **kwargs,
        )
        inner = _inner_wrapper(wrapper)
        if u_variable_name not in inner.upper_air_variables:
            raise ValueError(
                f"u_variable_name={u_variable_name!r} not in "
                f"wrapper.upper_air_variables={list(inner.upper_air_variables)}"
            )
        self.u_idx = list(inner.upper_air_variables).index(u_variable_name)
        levels = list(inner.levels)
        missing = [lvl for lvl in qbo_levels if lvl not in levels]
        if missing:
            raise ValueError(
                f"qbo_levels {missing} not found in wrapper.levels={levels}"
            )
        self.qbo_levels = list(qbo_levels)
        self.level_indices = [levels.index(lvl) for lvl in self.qbo_levels]
        self.steps_per_bin = max(1, int(steps_per_bin))
        self.months_per_bin = float(months_per_bin)
        self.n_bins = int(n_bins) if n_bins is not None else (
            self.horizon // self.steps_per_bin + 1
        )

        sample = dataset[0]
        n_lat = sample["surface_in"].shape[-2]
        self.lat_mask, self.lat_weights = _tropical_band_mask_and_weights(
            n_lat, tropical_band_deg, device, torch.float32
        )

        shape = (len(self.level_indices),)
        self._qbo_pred = StreamingBinnedMean(self.n_bins, shape, device)
        self._qbo_truth = StreamingBinnedMean(self.n_bins, shape, device)

    def _band_mean(self, field_phys: torch.Tensor) -> torch.Tensor:
        """``(B, Cu, L, H, W) -> (B, n_levels)`` zonal + tropical-band mean."""
        u_field = field_phys[:, self.u_idx]  # (B, L, H, W)
        u_field = u_field[:, self.level_indices]  # (B, n_levels, H, W)
        zonal = u_field.mean(dim=-1)  # (B, n_levels, H)
        masked = zonal[:, :, self.lat_mask]  # (B, n_levels, H_band)
        weights = self.lat_weights.to(masked.dtype)
        return (masked * weights).sum(dim=-1)  # (B, n_levels)

    def _score_step(self, m_idx, pred_ensemble, truth, kind):
        super()._score_step(m_idx, pred_ensemble, truth, kind)
        if kind != "upper_air":
            return
        pred_mean = _ensemble_mean(pred_ensemble, self.ensemble_size)
        pred_phys, truth_phys = self._denorm_pred_truth(kind, pred_mean, truth)
        pred_band = self._band_mean(pred_phys)
        truth_band = self._band_mean(truth_phys)
        bin_value = (m_idx // self.steps_per_bin) % self.n_bins
        bin_idx = torch.full(
            (pred_band.shape[0],), bin_value, dtype=torch.long, device=pred_band.device
        )
        self._qbo_pred.update(pred_band, bin_idx)
        self._qbo_truth.update(truth_band, bin_idx)

    def run(self, model, *, epoch: int = 0) -> dict:
        metrics = super().run(model, epoch=epoch)
        pred_ts = self._qbo_pred.finalize()  # (n_bins, n_levels)
        truth_ts = self._qbo_truth.finalize()
        metrics["qbo_pred_timeseries"] = pred_ts
        metrics["qbo_truth_timeseries"] = truth_ts
        for i, lvl in enumerate(self.qbo_levels):
            metrics[f"qbo_period_months_pred_hPa{int(lvl)}"] = _estimate_period_months(
                pred_ts[:, i], self.months_per_bin
            )
            metrics[f"qbo_period_months_truth_hPa{int(lvl)}"] = _estimate_period_months(
                truth_ts[:, i], self.months_per_bin
            )
        return metrics


# ---------------------------------------------------------------------------
# GlobalMeanTimeseriesValidator
# ---------------------------------------------------------------------------


class GlobalMeanTimeseriesValidator(DiffusionRolloutValidator):
    r"""Lat-weighted global-mean per-step timeseries for named flux channels.

    Useful for surface-energy-budget / TOA-flux sanity checks (e.g.
    ``flux_variables=["rsdt", "rsut", "rlut"]`` for net TOA radiation).
    Each name is looked up (in order) against
    ``wrapper.surface_variables`` then ``wrapper.diagnostic_variables``.

    ``run()`` returns per-name ``(horizon,)`` tensors under
    ``flux_pred_series`` / ``flux_truth_series``, in addition to the
    parent's RMSE/ACC/spread dict.
    """

    def __init__(
        self,
        dataset,
        *,
        wrapper,
        inference_scheduler,
        horizon: int,
        device: torch.device,
        flux_variables: Sequence[str],
        **kwargs,
    ):
        super().__init__(
            dataset,
            wrapper=wrapper,
            inference_scheduler=inference_scheduler,
            log_steps=list(range(1, int(horizon) + 1)),
            horizon=horizon,
            device=device,
            **kwargs,
        )
        inner = _inner_wrapper(wrapper)
        self.flux_variables = list(flux_variables)
        self._flux_index: dict[str, tuple[str, int]] = {}
        for name in self.flux_variables:
            if name in inner.surface_variables:
                self._flux_index[name] = ("surface", list(inner.surface_variables).index(name))
            elif name in inner.diagnostic_variables:
                self._flux_index[name] = ("diagnostic", list(inner.diagnostic_variables).index(name))
            else:
                raise ValueError(
                    f"flux variable {name!r} not found in wrapper.surface_variables "
                    f"or wrapper.diagnostic_variables"
                )
        self._pred_series: dict[str, list] = {n: [] for n in self.flux_variables}
        self._truth_series: dict[str, list] = {n: [] for n in self.flux_variables}

    def _score_step(self, m_idx, pred_ensemble, truth, kind):
        super()._score_step(m_idx, pred_ensemble, truth, kind)
        pred_mean = _ensemble_mean(pred_ensemble, self.ensemble_size)
        pred_phys, truth_phys = self._denorm_pred_truth(kind, pred_mean, truth)
        for name, (grp, idx) in self._flux_index.items():
            if grp != kind:
                continue
            pred_field = pred_phys[:, idx]  # (B, H, W)
            truth_field = truth_phys[:, idx]
            pred_scalar = float(lat_weighted_global_scalars(pred_field).mean().item())
            truth_scalar = float(lat_weighted_global_scalars(truth_field).mean().item())
            self._pred_series[name].append(pred_scalar)
            self._truth_series[name].append(truth_scalar)

    def run(self, model, *, epoch: int = 0) -> dict:
        metrics = super().run(model, epoch=epoch)
        metrics["flux_pred_series"] = {
            k: torch.tensor(v) for k, v in self._pred_series.items()
        }
        metrics["flux_truth_series"] = {
            k: torch.tensor(v) for k, v in self._truth_series.items()
        }
        return metrics


# ---------------------------------------------------------------------------
# EnsembleEnvelopeValidator
# ---------------------------------------------------------------------------


class EnsembleEnvelopeValidator(DiffusionRolloutValidator):
    r"""Spread/skill ratio across an ensemble rollout.

    Requires ``ensemble_size > 1``. Purely a post-hoc ratio of the
    parent's own RMSE (skill) and :class:`~validate_diffusion.StreamingLatWeightedSpread`
    (spread) metrics — no new aggregation. A well-calibrated ensemble
    has ``spread / skill ≈ 1``.
    """

    def __init__(
        self,
        dataset,
        *,
        wrapper,
        inference_scheduler,
        horizon: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(
            dataset,
            wrapper=wrapper,
            inference_scheduler=inference_scheduler,
            log_steps=list(range(1, int(horizon) + 1)),
            horizon=horizon,
            device=device,
            **kwargs,
        )
        if self.ensemble_size <= 1:
            raise ValueError(
                "EnsembleEnvelopeValidator requires ensemble_size > 1 "
                f"(got {self.ensemble_size})"
            )

    def run(self, model, *, epoch: int = 0) -> dict:
        metrics = super().run(model, epoch=epoch)
        ratios: dict[str, float] = {}
        for group in ("surface", "upper_air", "diagnostic"):
            for step in self.log_steps:
                spread_key = f"spread_step{step}_{group}"
                rmse_key = f"rmse_step{step}_{group}"
                if spread_key in metrics and rmse_key in metrics:
                    rmse = metrics[rmse_key]
                    ratios[f"spread_skill_ratio_step{step}_{group}"] = (
                        metrics[spread_key] / rmse if rmse > 0 else float("nan")
                    )
        metrics.update(ratios)
        return metrics


__all__ = [
    "ClimatologyValidator",
    "BiasValidator",
    "QBOValidator",
    "GlobalMeanTimeseriesValidator",
    "EnsembleEnvelopeValidator",
]


# ---------------------------------------------------------------------------
# Hydra entrypoint — selects + runs the aggregators enabled in
# ``conf/validation/eval_suite.yaml`` against a trained diffusion checkpoint.
# ---------------------------------------------------------------------------


def _resolve_eval_sampler_num_steps(raw):
    """Mirror ``train_diffusion._build_validator``'s num_steps coercion."""
    if raw is None:
        return None
    from omegaconf import OmegaConf

    if OmegaConf.is_config(raw) or isinstance(raw, (list, tuple)):
        return [int(s) for s in raw]
    return int(raw)


def main(cfg) -> None:
    """Runs the eval suite. Wrapped by ``hydra.main`` below (kept as a
    plain function so it stays unit-testable without invoking Hydra)."""
    import warnings

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    from physicsnemo.utils.logging import PythonLogger
    from validate import Deterministic, GaussianIC, ReplicateOnly

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.datapipes.climate import ClimateNormalizer

    from train import _resolve_path, build_model  # noqa: E402
    from train_diffusion import _build_dataset  # noqa: E402

    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("amip_eval_suite")

    eval_cfg = cfg.get("eval_suite", None)
    if eval_cfg is None:
        raise ValueError(
            "eval_suite.* config block missing — select "
            "validation=eval_suite on the Hydra command line."
        )

    raw_ds = _build_dataset(cfg)
    wrapper = build_model(cfg.model).to(dist.device)
    ckpt_dir = _resolve_path(str(eval_cfg.checkpoint_dir))
    loaded_epoch = load_checkpoint(ckpt_dir, models=wrapper, device=dist.device)
    logger.info(f"loaded checkpoint epoch={loaded_epoch} from {ckpt_dir}")
    wrapper.eval()

    # The same scheduler class serves both training loss (compute_loss)
    # and inference sampling (sample / sample_rollout) — see
    # train_diffusion.py's _build_validator, which reuses the training
    # stage's scheduler instance as the validator's inference_scheduler.
    import hydra as _hydra

    scheduler = _hydra.utils.instantiate(cfg.loss).to(dist.device)

    normalizer = ClimateNormalizer.from_dataset(
        raw_ds,
        mean_path=_resolve_path(cfg.dataset.mean_path),
        std_path=_resolve_path(cfg.dataset.std_path),
        normalize_constant_boundary=bool(
            cfg.dataset.get("normalize_constant_boundary", False)
        ),
        normalize_diagnostic=bool(cfg.dataset.get("normalize_diagnostic", False)),
    ).to(dist.device)

    has_diagnostic = (
        cfg.model.get("diagnostic_variables") is not None
        and len(list(cfg.model.diagnostic_variables)) > 0
    )
    horizon = int(eval_cfg.horizon)
    sampler_num_steps = _resolve_eval_sampler_num_steps(
        eval_cfg.get("sampler_num_steps", None)
    )

    base_kwargs = dict(
        wrapper=wrapper,
        inference_scheduler=scheduler,
        horizon=horizon,
        device=dist.device,
        has_diagnostic=has_diagnostic,
        max_initial_conditions=int(eval_cfg.get("max_initial_conditions", 1)),
        ic_stride=int(eval_cfg.get("ic_stride", 1)),
        batch_size=int(eval_cfg.get("batch_size", 1)),
        normalizer=normalizer,
        sampler_num_steps=sampler_num_steps,
        seed=int(cfg.seed),
    )

    results: dict = {}

    clim_cfg = eval_cfg.get("climatology", None)
    if clim_cfg is not None and bool(clim_cfg.get("enabled", False)):
        v = ClimatologyValidator(
            raw_ds,
            n_bins=int(clim_cfg.get("n_bins", 12)),
            steps_per_bin=int(clim_cfg.get("steps_per_bin", 1)),
            **base_kwargs,
        )
        results["climatology"] = v.run(wrapper, epoch=0)
        logger.info("climatology validator done")

    bias_cfg = eval_cfg.get("bias", None)
    if bias_cfg is not None and bool(bias_cfg.get("enabled", False)):
        v = BiasValidator(
            raw_ds,
            n_bins=int(bias_cfg.get("n_bins", 12)),
            steps_per_bin=int(bias_cfg.get("steps_per_bin", 1)),
            **base_kwargs,
        )
        results["bias"] = v.run(wrapper, epoch=0)
        logger.info("bias validator done")

    qbo_cfg = eval_cfg.get("qbo", None)
    if qbo_cfg is not None and bool(qbo_cfg.get("enabled", False)):
        v = QBOValidator(
            raw_ds,
            u_variable_name=str(qbo_cfg.get("u_variable_name", "ua")),
            qbo_levels=list(qbo_cfg.get("levels", [10.0, 30.0, 50.0])),
            steps_per_bin=int(qbo_cfg.get("steps_per_bin", 120)),
            months_per_bin=float(qbo_cfg.get("months_per_bin", 1.0)),
            **base_kwargs,
        )
        results["qbo"] = v.run(wrapper, epoch=0)
        logger.info("qbo validator done")

    gm_cfg = eval_cfg.get("global_mean", None)
    if gm_cfg is not None and bool(gm_cfg.get("enabled", False)):
        v = GlobalMeanTimeseriesValidator(
            raw_ds,
            flux_variables=list(gm_cfg.flux_variables),
            **base_kwargs,
        )
        results["global_mean"] = v.run(wrapper, epoch=0)
        logger.info("global_mean validator done")

    ens_cfg = eval_cfg.get("ensemble_envelope", None)
    if ens_cfg is not None and bool(ens_cfg.get("enabled", False)):
        ensemble_size = int(ens_cfg.get("ensemble_size", 4))
        perturber_kind = str(ens_cfg.get("perturber", "replicate_only")).lower()
        if perturber_kind in ("replicate_only", "replicateonly", "replicate"):
            perturber = ReplicateOnly()
        elif perturber_kind in ("gaussian_ic", "gaussianic", "gaussian"):
            from omegaconf import OmegaConf as _OmegaConf

            perturber = GaussianIC(
                scales=dict(
                    _OmegaConf.to_container(
                        ens_cfg.get("perturber_scales", {}), resolve=True
                    )
                    or {}
                )
            )
        else:
            perturber = Deterministic()
        ens_kwargs = dict(base_kwargs)
        v = EnsembleEnvelopeValidator(
            raw_ds,
            ensemble_size=ensemble_size,
            perturber=perturber,
            **ens_kwargs,
        )
        results["ensemble_envelope"] = v.run(wrapper, epoch=0)
        logger.info("ensemble_envelope validator done")

    output_path = _resolve_path(str(eval_cfg.output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    logger.info(f"wrote eval suite results to {output_path}")


if __name__ == "__main__":
    import hydra

    hydra.main(version_base="1.2", config_path="conf", config_name="config")(main)()
