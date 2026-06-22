# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Climatological validation CLI for ai_rossby rollouts (Phase 4c).

For a multi-year autoregressive rollout starting at a single (or small
set of) initial condition(s), this script accumulates:

* **Time-mean** of prediction and truth per pixel → climatological
  *bias* field = pred_mean − truth_mean.
* **Time-variance** of prediction and truth (Chan / Welford) → climate
  *variance bias* = pred_var − truth_var.
* **Per-bin mean** for the day-of-year (or any user-supplied bin
  function) → climatological *daily climatology* for prediction and
  truth, with the per-day bias as their difference.

Memory at any moment is ``O(n_bins × C × H × W)`` for the aggregators
plus one rollout-window working set on GPU — independent of how long
the rollout runs. Targets multi-year (≥ 1 year, 1460+ steps at 6 h
cadence) rollouts.

Usage::

    python climatology_cli.py \\
        model=sfno_plasim_5412 \\
        dataset=plasim_sim52_train_val \\
        +climatology.checkpoint_dir=./outputs/sfno_run/checkpoints \\
        +climatology.output_path=./outputs/sfno_run/climatology.nc \\
        +climatology.ic_start=[0] \\
        +climatology.max_step=1440      # 1 year at 6h cadence on PLASIM
        +climatology.steps_per_bin=4    # 6h × 4 = 1 day → daily climatology
        +climatology.n_bins=360         # PLASIM 360-day calendar

The script emits a NetCDF with these fields:
  * ``pred_{surface,upper_air,diagnostic}_mean`` — time-mean of forecast
  * ``truth_{...}_mean`` — time-mean of reference
  * ``bias_{...}`` — pred mean − truth mean
  * ``pred_{...}_var``, ``truth_{...}_var``, ``var_bias_{...}``
  * ``pred_{...}_daily_clim``, ``truth_{...}_daily_clim``, ``daily_bias_{...}``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.datapipes.plasim import (
        NanFillTransform,
        PlasimClimateDataset,
        PlasimNormalizer,
    )

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climatology import (
    StreamingBinnedMean,
    StreamingTimeMean,
    StreamingTimeVariance,
)
from validate import Deterministic, GaussianIC, Perturber, ReplicateOnly


def _resolve_path(p: Optional[str]) -> Optional[str]:
    return to_absolute_path(p) if p else None


def _build_perturber(name: str, scales: dict) -> Perturber:
    kind = str(name).lower()
    if kind in ("deterministic", "off", "none"):
        return Deterministic()
    if kind in ("replicate", "replicate_only", "stochastic_model"):
        return ReplicateOnly()
    if kind in ("gaussian_ic", "ic_gaussian", "gaussian"):
        if not scales:
            raise ValueError("gaussian_ic requires climatology.perturber_scales={var: std,...}")
        return GaussianIC(scales=dict(scales))
    raise ValueError(f"unknown perturber={name!r}")


def _fetch_input_at(
    dataset: PlasimClimateDataset,
    t: int,
    normalizer: Optional[PlasimNormalizer],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return the dataset's input frame at time ``t`` (no target needed).

    Re-uses the dataset's lead-1 lookup and discards the target half.
    Adds a leading batch dim of 1 for shape parity with model forwards.
    """
    if t + 1 < dataset.n_time:
        raw = dataset[(t, 1)]
    else:
        raw = dataset[(dataset.n_time - 2, 1)]
    out = {
        "surface_in": raw["surface_in"].unsqueeze(0).to(device),
        "upper_air_in": raw["upper_air_in"].unsqueeze(0).to(device),
        "varying_boundary": raw["varying_boundary"].unsqueeze(0).to(device),
        "constant_boundary": raw["constant_boundary"].to(device),
    }
    if "diagnostic" in raw and isinstance(raw["diagnostic"], torch.Tensor):
        out["diagnostic"] = raw["diagnostic"].unsqueeze(0).to(device)
    if normalizer is not None:
        out = normalizer(out)
    return out


def _make_aggregator_set(
    *,
    n_bins: int,
    shapes: dict[str, tuple[int, ...]],
    device: torch.device,
    track_bins: bool,
) -> dict[str, dict]:
    """Build the ``(mean, variance, [binned])`` triple per channel group.

    Returned dict keyed by group name (``surface`` / ``upper_air`` /
    ``diagnostic``); each value is a dict ``{mean, var, binned}`` with
    the aggregator instances. ``binned`` is ``None`` when track_bins
    is False or memory is tight.
    """
    out: dict[str, dict] = {}
    for grp, sh in shapes.items():
        slot: dict = {
            "mean": StreamingTimeMean(sh, device),
            "var": StreamingTimeVariance(sh, device),
        }
        if track_bins:
            slot["binned"] = StreamingBinnedMean(n_bins, sh, device)
        else:
            slot["binned"] = None
        out[grp] = slot
    return out


def _update_set(
    pred_agg: dict, truth_agg: dict, pred: torch.Tensor, truth: torch.Tensor, bin_idx: torch.Tensor
) -> None:
    """Push ``(pred, truth)`` into the matching aggregator slot.

    Both tensors should be ``(B, *shape)`` already on the aggregator's
    device + dtype.
    """
    pred_agg["mean"].update(pred)
    truth_agg["mean"].update(truth)
    pred_agg["var"].update(pred)
    truth_agg["var"].update(truth)
    if pred_agg["binned"] is not None:
        pred_agg["binned"].update(pred, bin_idx)
        truth_agg["binned"].update(truth, bin_idx)


@torch.no_grad()
def run_climatology(
    model: torch.nn.Module,
    dataset: PlasimClimateDataset,
    *,
    normalizer: Optional[PlasimNormalizer],
    device: torch.device,
    ic_indices: Sequence[int],
    max_step: int,
    n_bins: int,
    steps_per_bin: int,
    ensemble_size: int = 1,
    perturber: Optional[Perturber] = None,
    has_diagnostic: bool = False,
    seed: int = 0,
    track_bins: bool = True,
    logger=None,
) -> dict:
    """Drive a long rollout and accumulate climatological statistics.

    Returns a dict ``{"pred": {...}, "truth": {...}, "shapes": {...},
    "bin_counts": {...}}`` whose ``"mean"``, ``"var"``, ``"binned"``
    entries are torch tensors on CPU (float32) — ready to drop into an
    xarray output.
    """
    if perturber is None:
        perturber = Deterministic() if ensemble_size == 1 else ReplicateOnly()
    rng = torch.Generator(device=device).manual_seed(seed)

    # Layout introspection from a single dataset frame.
    sample = dataset[0]
    surface_shape = tuple(sample["surface_in"].shape)       # (C_s, H, W)
    upper_shape = tuple(sample["upper_air_in"].shape)        # (C_u, L, H, W)
    diag_shape = (
        tuple(sample["diagnostic"].shape)
        if has_diagnostic and "diagnostic" in sample
        else None
    )

    shapes = {"surface": surface_shape, "upper_air": upper_shape}
    if diag_shape is not None:
        shapes["diagnostic"] = diag_shape

    pred_set = _make_aggregator_set(
        n_bins=n_bins, shapes=shapes, device=device, track_bins=track_bins
    )
    truth_set = _make_aggregator_set(
        n_bins=n_bins, shapes=shapes, device=device, track_bins=track_bins
    )

    for ic_pos, ic in enumerate(ic_indices):
        if logger is not None:
            logger.info(
                f"climatology rollout {ic_pos+1}/{len(ic_indices)} from IC {ic} "
                f"({max_step} steps × {ensemble_size} ensemble)"
            )
        init = _fetch_input_at(dataset, int(ic), normalizer, device)
        state = perturber(init, ensemble_size, generator=rng)
        const_boundary = state.get("constant_boundary")

        for k in range(1, max_step + 1):
            input_boundary = state["varying_boundary"]
            out = model(
                state["surface_in"],
                const_boundary,
                input_boundary,
                state["upper_air_in"],
            )
            if has_diagnostic:
                next_surface, next_upper, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper = out[0], out[1]
                next_diag = None

            # Fetch truth at time ic+k (the state AT that time).
            truth_t = _fetch_input_at(dataset, int(ic) + int(k), normalizer, device)

            # Bin index: which day-of-year bin does this step land in?
            # bin = (k // steps_per_bin) % n_bins.
            bin_value = (int(k) // max(steps_per_bin, 1)) % n_bins
            bin_idx_pred = torch.full(
                (next_surface.shape[0],), bin_value, dtype=torch.long, device=device
            )
            bin_idx_truth = torch.full(
                (truth_t["surface_in"].shape[0],),
                bin_value,
                dtype=torch.long,
                device=device,
            )

            _update_set(
                pred_set["surface"], truth_set["surface"],
                next_surface, truth_t["surface_in"], bin_idx_pred,
            )
            _update_set(
                pred_set["upper_air"], truth_set["upper_air"],
                next_upper, truth_t["upper_air_in"], bin_idx_pred,
            )
            if next_diag is not None and "diagnostic" in pred_set:
                _update_set(
                    pred_set["diagnostic"], truth_set["diagnostic"],
                    next_diag,
                    truth_t.get("diagnostic", torch.zeros_like(next_diag)),
                    bin_idx_pred,
                )

            # Advance state. The next step's boundary marches forward.
            next_boundary = truth_t["varying_boundary"]
            if ensemble_size > 1:
                next_boundary = next_boundary.repeat_interleave(ensemble_size, dim=0)
            state = {
                "surface_in": next_surface,
                "upper_air_in": next_upper,
                "constant_boundary": const_boundary,
                "varying_boundary": next_boundary,
            }

    # Finalize.
    finalized = {"pred": {}, "truth": {}, "shapes": shapes, "bin_counts": {}}
    for grp in shapes:
        pred = pred_set[grp]
        truth = truth_set[grp]
        pmean = pred["mean"].finalize().cpu()
        tmean = truth["mean"].finalize().cpu()
        pm, pv = pred["var"].finalize()
        tm, tv = truth["var"].finalize()
        finalized["pred"][grp] = {"mean": pmean, "var": pv.cpu()}
        finalized["truth"][grp] = {"mean": tmean, "var": tv.cpu()}
        if pred["binned"] is not None:
            finalized["pred"][grp]["binned"] = pred["binned"].finalize().cpu()
            finalized["truth"][grp]["binned"] = truth["binned"].finalize().cpu()
            finalized["bin_counts"][grp] = pred["binned"].counts_per_bin.cpu()
    return finalized


def _agg_to_xarray(
    aggregated: dict,
    *,
    ic_indices: Sequence[int],
    max_step: int,
    ensemble_size: int,
    n_bins: int,
    steps_per_bin: int,
    surface_variables: Sequence[str],
    upper_air_variables: Sequence[str],
    diagnostic_variables: Sequence[str],
    levels: Sequence[float],
    lat: np.ndarray,
    lon: np.ndarray,
    has_diagnostic: bool,
) -> xr.Dataset:
    """Convert the aggregator dict to an xarray dataset with named coords."""
    coords = {
        "lat": ("lat", lat.astype(np.float32)),
        "lon": ("lon", lon.astype(np.float32)),
        "surface_var": ("surface_var", np.asarray(list(surface_variables))),
        "upper_air_var": ("upper_air_var", np.asarray(list(upper_air_variables))),
        "level": ("level", np.asarray(list(levels), dtype=np.float32)),
        "bin": ("bin", np.arange(n_bins, dtype=np.int64)),
    }
    if has_diagnostic and "diagnostic" in aggregated["shapes"]:
        coords["diag_var"] = ("diag_var", np.asarray(list(diagnostic_variables)))

    data_vars: dict = {}

    def _add(group: str, suffix: str, dim_pattern: tuple[str, ...], arr: torch.Tensor):
        data_vars[f"{suffix}_{group}"] = (dim_pattern, arr.numpy())

    # Surface (no level dim).
    surf_dims = ("surface_var", "lat", "lon")
    _add("surface", "pred_mean", surf_dims, aggregated["pred"]["surface"]["mean"])
    _add("surface", "truth_mean", surf_dims, aggregated["truth"]["surface"]["mean"])
    _add(
        "surface",
        "bias",
        surf_dims,
        aggregated["pred"]["surface"]["mean"] - aggregated["truth"]["surface"]["mean"],
    )
    _add("surface", "pred_var", surf_dims, aggregated["pred"]["surface"]["var"])
    _add("surface", "truth_var", surf_dims, aggregated["truth"]["surface"]["var"])
    _add(
        "surface",
        "var_bias",
        surf_dims,
        aggregated["pred"]["surface"]["var"] - aggregated["truth"]["surface"]["var"],
    )
    if "binned" in aggregated["pred"]["surface"]:
        binned_dims = ("bin",) + surf_dims
        _add("surface", "pred_daily_clim", binned_dims, aggregated["pred"]["surface"]["binned"])
        _add("surface", "truth_daily_clim", binned_dims, aggregated["truth"]["surface"]["binned"])
        _add(
            "surface",
            "daily_bias",
            binned_dims,
            aggregated["pred"]["surface"]["binned"]
            - aggregated["truth"]["surface"]["binned"],
        )

    # Upper-air (with level dim).
    upper_dims = ("upper_air_var", "level", "lat", "lon")
    _add("upper_air", "pred_mean", upper_dims, aggregated["pred"]["upper_air"]["mean"])
    _add("upper_air", "truth_mean", upper_dims, aggregated["truth"]["upper_air"]["mean"])
    _add(
        "upper_air",
        "bias",
        upper_dims,
        aggregated["pred"]["upper_air"]["mean"] - aggregated["truth"]["upper_air"]["mean"],
    )
    _add("upper_air", "pred_var", upper_dims, aggregated["pred"]["upper_air"]["var"])
    _add("upper_air", "truth_var", upper_dims, aggregated["truth"]["upper_air"]["var"])
    _add(
        "upper_air",
        "var_bias",
        upper_dims,
        aggregated["pred"]["upper_air"]["var"] - aggregated["truth"]["upper_air"]["var"],
    )
    if "binned" in aggregated["pred"]["upper_air"]:
        binned_dims = ("bin",) + upper_dims
        _add("upper_air", "pred_daily_clim", binned_dims, aggregated["pred"]["upper_air"]["binned"])
        _add(
            "upper_air",
            "truth_daily_clim",
            binned_dims,
            aggregated["truth"]["upper_air"]["binned"],
        )
        _add(
            "upper_air",
            "daily_bias",
            binned_dims,
            aggregated["pred"]["upper_air"]["binned"]
            - aggregated["truth"]["upper_air"]["binned"],
        )

    if has_diagnostic and "diagnostic" in aggregated["shapes"]:
        diag_dims = ("diag_var", "lat", "lon")
        _add("diagnostic", "pred_mean", diag_dims, aggregated["pred"]["diagnostic"]["mean"])
        _add("diagnostic", "truth_mean", diag_dims, aggregated["truth"]["diagnostic"]["mean"])
        _add(
            "diagnostic",
            "bias",
            diag_dims,
            aggregated["pred"]["diagnostic"]["mean"]
            - aggregated["truth"]["diagnostic"]["mean"],
        )
        _add("diagnostic", "pred_var", diag_dims, aggregated["pred"]["diagnostic"]["var"])
        _add("diagnostic", "truth_var", diag_dims, aggregated["truth"]["diagnostic"]["var"])
        _add(
            "diagnostic",
            "var_bias",
            diag_dims,
            aggregated["pred"]["diagnostic"]["var"]
            - aggregated["truth"]["diagnostic"]["var"],
        )
        if "binned" in aggregated["pred"]["diagnostic"]:
            binned_dims = ("bin",) + diag_dims
            _add(
                "diagnostic",
                "pred_daily_clim",
                binned_dims,
                aggregated["pred"]["diagnostic"]["binned"],
            )
            _add(
                "diagnostic",
                "truth_daily_clim",
                binned_dims,
                aggregated["truth"]["diagnostic"]["binned"],
            )
            _add(
                "diagnostic",
                "daily_bias",
                binned_dims,
                aggregated["pred"]["diagnostic"]["binned"]
                - aggregated["truth"]["diagnostic"]["binned"],
            )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["ic_indices"] = np.asarray(list(ic_indices), dtype=np.int64)
    ds.attrs["max_step"] = int(max_step)
    ds.attrs["ensemble_size"] = int(ensemble_size)
    ds.attrs["n_bins"] = int(n_bins)
    ds.attrs["steps_per_bin"] = int(steps_per_bin)
    for grp, counts in aggregated["bin_counts"].items():
        ds.attrs[f"bin_counts_{grp}"] = counts.numpy().astype(np.int64)
    return ds


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("ai_rossby_climatology_cli")

    ccfg = cfg.get("climatology", None)
    if ccfg is None:
        raise ValueError(
            "climatology.* config block missing — add "
            "+climatology.checkpoint_dir, +climatology.output_path, "
            "+climatology.max_step, +climatology.ic_start, "
            "+climatology.steps_per_bin, +climatology.n_bins"
        )

    if dist.rank != 0 and dist.world_size > 1:
        logger.warning("climatology_cli runs on rank 0 only; exiting on others")
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # --- Load model ---------------------------------------------------------
    flat = OmegaConf.to_container(cfg.model, resolve=True) or {}
    name = str(flat["name"])
    module_path = str(flat["module"])
    args = {k: v for k, v in flat.items() if k not in {"name", "module", "target", "model_type"}}
    model = Module.instantiate(
        {"__name__": name, "__module__": module_path, "__args__": args}
    ).to(dist.device)
    model.eval()
    ckpt_dir = _resolve_path(str(ccfg.checkpoint_dir))
    loaded_epoch = load_checkpoint(ckpt_dir, models=model, device=dist.device)
    logger.info(f"loaded checkpoint epoch={loaded_epoch} from {ckpt_dir}")

    # --- Dataset + normalizer ----------------------------------------------
    data = cfg.dataset
    val_zarr_path = _resolve_path(
        data.val_zarr_path if data.val_zarr_path else data.zarr_path
    )
    base_ds = PlasimClimateDataset(
        val_zarr_path,
        boundary_zarr_path=_resolve_path(data.boundary_zarr_path),
        yearly_repeating_boundary=bool(data.yearly_repeating_boundary),
        leap_boundary_zarr_path=_resolve_path(data.leap_boundary_zarr_path),
        non_leap_boundary_zarr_path=_resolve_path(data.non_leap_boundary_zarr_path),
    )
    normalizer = PlasimNormalizer.from_dataset(
        base_ds,
        mean_path=_resolve_path(data.mean_path),
        std_path=_resolve_path(data.std_path),
        normalize_constant_boundary=bool(data.get("normalize_constant_boundary", False)),
        normalize_diagnostic=bool(data.get("normalize_diagnostic", False)),
    ).to(dist.device)
    nan_fill = NanFillTransform(
        constant_boundary_variables=list(cfg.model.constant_boundary_variables),
        varying_boundary_variables=list(cfg.model.varying_boundary_variables),
        fill_values=dict(OmegaConf.to_container(data.nan_fill_values, resolve=True) or {}),
        default=float(data.nan_fill_default),
    )
    base_ds.transform = nan_fill

    # --- Roll out + aggregate ----------------------------------------------
    ic_indices = list(ccfg.ic_start)
    max_step = int(ccfg.max_step)
    ensemble_size = int(ccfg.get("ensemble_size", 1))
    n_bins = int(ccfg.get("n_bins", 365))
    steps_per_bin = int(ccfg.get("steps_per_bin", 1))
    track_bins = bool(ccfg.get("track_bins", True))
    perturber = _build_perturber(
        str(ccfg.get("perturber", "deterministic")),
        OmegaConf.to_container(ccfg.get("perturber_scales", {}), resolve=True) or {},
    )

    aggregated = run_climatology(
        model,
        base_ds,
        normalizer=normalizer,
        device=dist.device,
        ic_indices=ic_indices,
        max_step=max_step,
        n_bins=n_bins,
        steps_per_bin=steps_per_bin,
        ensemble_size=ensemble_size,
        perturber=perturber,
        has_diagnostic=getattr(model, "has_diagnostic", False),
        seed=int(cfg.seed) + 2027,
        track_bins=track_bins,
        logger=logger,
    )

    # --- Output -------------------------------------------------------------
    surface_variables = list(base_ds.layout.surface_variables)
    upper_air_variables = list(
        base_ds.layout.sigma_upper_air_variables
        + base_ds.layout.pressure_upper_air_variables
    )
    diagnostic_variables = list(base_ds.layout.diagnostic_variables)
    n_levels = aggregated["shapes"]["upper_air"][1]
    sigma_levels = list(getattr(base_ds, "sigma_levels", []))
    pressure_levels = list(getattr(base_ds, "pressure_levels", []))
    if sigma_levels and len(sigma_levels) == n_levels:
        levels_coord = sigma_levels
    elif pressure_levels and len(pressure_levels) == n_levels:
        levels_coord = pressure_levels
    else:
        levels_coord = list(range(n_levels))

    lat_arr = np.asarray(base_ds._ds["lat"].values, dtype=np.float32)
    lon_arr = np.asarray(base_ds._ds["lon"].values, dtype=np.float32)

    ds = _agg_to_xarray(
        aggregated,
        ic_indices=ic_indices,
        max_step=max_step,
        ensemble_size=ensemble_size,
        n_bins=n_bins,
        steps_per_bin=steps_per_bin,
        surface_variables=surface_variables,
        upper_air_variables=upper_air_variables,
        diagnostic_variables=diagnostic_variables,
        levels=levels_coord,
        lat=lat_arr,
        lon=lon_arr,
        has_diagnostic=getattr(model, "has_diagnostic", False),
    )

    output_path = _resolve_path(str(ccfg.output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith(".zarr"):
        ds.to_zarr(output_path, mode="w", zarr_format=3, consolidated=True)
    else:
        ds.to_netcdf(output_path, mode="w")
    logger.info(f"wrote {output_path}")


if __name__ == "__main__":
    main()
