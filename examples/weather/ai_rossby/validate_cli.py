# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""After-the-fact validation CLI for ai_rossby predictions (Phase 4b).

Reads a prediction NetCDF (or Zarr) written by ``inference.py`` plus a
reference dataset Zarr, computes per-step / per-channel-group
lat-weighted RMSE and (optionally) anomaly correlation + ensemble
CRPS, and emits a summary JSON + a markdown table.

Memory pattern: one IC × one step × channel-group at a time. The
streaming aggregators from the Phase 4a :mod:`validate` module are
reused (single-batch updates accumulate per (step, channel)).

Usage::

    python validate_cli.py \\
        +validation_cli.predictions=/path/to/preds.nc \\
        +validation_cli.reference_zarr=/path/to/13.zarr \\
        +validation_cli.output_json=/path/to/scores.json \\
        +validation_cli.climatology_zarr=/path/to/climatology.zarr   # optional

The script doesn't load the model — it only needs the predictions, the
reference time series, and the dataset stats (for normalization). Pair
its config with the same ``dataset=...`` Hydra group used by training,
or override ``validation_cli.{normalize,mean_path,std_path}`` explicitly.
"""

from __future__ import annotations

import json
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
        PlasimClimateDataset,
        PlasimNormalizer,
    )

from physicsnemo.utils.logging import PythonLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import (
    StreamingLatWeightedACC,
    StreamingLatWeightedRMSE,
    cos_lat_weights,
)


def _resolve_path(p: Optional[str]) -> Optional[str]:
    return to_absolute_path(p) if p else None


def _open_predictions(path: str) -> xr.Dataset:
    if path.endswith(".zarr"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _open_reference(
    zarr_path: str,
    *,
    mean_path: Optional[str],
    std_path: Optional[str],
    normalize_constant_boundary: bool,
    normalize_diagnostic: bool,
    constant_boundary_variables: Sequence[str],
    varying_boundary_variables: Sequence[str],
    diagnostic_variables: Sequence[str],
) -> tuple[PlasimClimateDataset, Optional[PlasimNormalizer]]:
    """Open the reference dataset and (optionally) its normalizer.

    The normalizer is applied to truth tensors so that RMSE is computed
    in the same normalized space as during training. If mean_path /
    std_path aren't supplied, RMSE is computed in raw units (typically
    larger numbers; useful for unit-bearing diagnostic vars).
    """
    ds = PlasimClimateDataset(zarr_path)
    if not (mean_path and std_path):
        return ds, None
    normalizer = PlasimNormalizer.from_dataset(
        ds,
        mean_path=mean_path,
        std_path=std_path,
        normalize_constant_boundary=normalize_constant_boundary,
        normalize_diagnostic=normalize_diagnostic,
    )
    return ds, normalizer


def _resolve_truth_at(
    dataset: PlasimClimateDataset,
    ic: int,
    step: int,
    normalizer: Optional[PlasimNormalizer],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fetch the reference sample at time ``ic + step`` and normalize it.

    The dataset's (t, lead=1) lookup returns surface_in = state(t), so to
    get the truth at ic+step we look up at time ic+step directly. Falls
    back to the last valid frame when we'd index past the end (the
    inference run's IC selection should normally prevent this).

    Returns a dict with keys ``surface_in`` ((C_s, H, W)),
    ``upper_air_in`` ((C_u, L, H, W)) and ``diagnostic`` when present
    in the layout.
    """
    target = int(ic) + int(step)
    if target + 1 < dataset.n_time:
        raw = dataset[(target, 1)]
    else:
        raw = dataset[(dataset.n_time - 2, 1)]
    truth = {
        "surface_in": raw["surface_in"].unsqueeze(0).to(device),
        "upper_air_in": raw["upper_air_in"].unsqueeze(0).to(device),
    }
    if "diagnostic" in raw and isinstance(raw["diagnostic"], torch.Tensor):
        truth["diagnostic"] = raw["diagnostic"].unsqueeze(0).to(device)
    if normalizer is not None:
        truth = normalizer(truth)
    return truth


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("ai_rossby_validate_cli")
    vcfg = cfg.get("validation_cli", None)
    if vcfg is None:
        raise ValueError(
            "validation_cli.* block missing; add "
            "+validation_cli.predictions=<.nc> and "
            "+validation_cli.reference_zarr=<.zarr>."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(vcfg.get("use_cuda", True)) else "cpu"
    )

    # --- Open predictions ---------------------------------------------------
    pred_path = _resolve_path(str(vcfg.predictions))
    pred_ds = _open_predictions(pred_path)
    logger.info(f"loaded predictions from {pred_path}")
    logger.info(f"  shape: {dict(pred_ds.sizes)}")

    n_ic = int(pred_ds.sizes["ic"])
    n_ens = int(pred_ds.sizes["ensemble"])
    max_step = int(pred_ds.sizes["step"])
    H = int(pred_ds.sizes["lat"])
    W = int(pred_ds.sizes["lon"])

    # --- Open reference + normalizer ----------------------------------------
    data = cfg.dataset
    ref_zarr = _resolve_path(str(vcfg.reference_zarr))
    ref_dataset, normalizer = _open_reference(
        ref_zarr,
        mean_path=_resolve_path(data.get("mean_path", None)),
        std_path=_resolve_path(data.get("std_path", None)),
        normalize_constant_boundary=bool(data.get("normalize_constant_boundary", False)),
        normalize_diagnostic=bool(data.get("normalize_diagnostic", False)),
        constant_boundary_variables=list(cfg.model.get("constant_boundary_variables", [])),
        varying_boundary_variables=list(cfg.model.get("varying_boundary_variables", [])),
        diagnostic_variables=list(cfg.model.get("diagnostic_variables", [])),
    )
    if normalizer is not None:
        normalizer = normalizer.to(device)

    n_surface = int(pred_ds.sizes["surface_var"])
    n_upper = int(pred_ds.sizes["upper_air_var"])
    has_diag = "pred_diagnostic" in pred_ds
    n_diag = int(pred_ds.sizes["diag_var"]) if has_diag else 0

    # --- Streaming metric accumulators --------------------------------------
    lat_w = cos_lat_weights(H, device, torch.float32)
    rmse_surface = StreamingLatWeightedRMSE(
        n_steps=max_step, n_channels=n_surface, device=device,
    )
    rmse_upper = StreamingLatWeightedRMSE(
        n_steps=max_step, n_channels=n_upper, device=device,
    )
    rmse_diag = (
        StreamingLatWeightedRMSE(
            n_steps=max_step, n_channels=n_diag, device=device,
        )
        if has_diag
        else None
    )

    # Optional ACC against climatology — `climatology_zarr` must hold
    # mean fields with the same channel layout as the reference.
    acc_surface = acc_upper = acc_diag = None
    if vcfg.get("climatology_zarr", None):
        clim_ds = xr.open_zarr(_resolve_path(str(vcfg.climatology_zarr)))
        # Pull the surface/upper/diag means; broadcasting handles
        # (C, [L,] H, W) → (B, ...). Cast to float32 on device.
        # NOTE: this expects the climatology to be in NORMALIZED units
        # if normalizer is set, or RAW units otherwise. The user is
        # responsible for matching basis.
        surf_clim = torch.from_numpy(
            np.stack([clim_ds[v].values for v in pred_ds["surface_var"].values], axis=0)
        ).to(device, dtype=torch.float32)
        acc_surface = StreamingLatWeightedACC(
            n_steps=max_step, n_channels=n_surface, climatology=surf_clim, device=device,
        )

    # --- Iterate over (IC, step) and update metrics -------------------------
    ic_values = pred_ds["ic"].values
    for ic_pos in range(n_ic):
        ic = int(ic_values[ic_pos])
        for k in range(1, max_step + 1):
            truth = _resolve_truth_at(ref_dataset, ic, k, normalizer, device=device)
            # Ensemble-mean reduction for RMSE/ACC.
            pred_surface = torch.from_numpy(
                np.asarray(pred_ds["pred_surface"].values[ic_pos, :, k - 1, :, :, :])
            ).to(device, dtype=torch.float32).mean(dim=0).unsqueeze(0)
            pred_upper = torch.from_numpy(
                np.asarray(pred_ds["pred_upper_air"].values[ic_pos, :, k - 1, :, :, :, :])
            ).to(device, dtype=torch.float32).mean(dim=0).unsqueeze(0)
            rmse_surface.update(k - 1, pred_surface, truth["surface_in"], lat_w)
            rmse_upper.update(k - 1, pred_upper, truth["upper_air_in"], lat_w)
            if acc_surface is not None:
                acc_surface.update(k - 1, pred_surface, truth["surface_in"], lat_w)
            if has_diag and rmse_diag is not None and "diagnostic" in truth:
                pred_diag = torch.from_numpy(
                    np.asarray(pred_ds["pred_diagnostic"].values[ic_pos, :, k - 1, :, :, :])
                ).to(device, dtype=torch.float32).mean(dim=0).unsqueeze(0)
                rmse_diag.update(k - 1, pred_diag, truth["diagnostic"], lat_w)

    # --- Finalize + emit ---------------------------------------------------
    rmse_s = rmse_surface.finalize().cpu().numpy()        # (n_steps, n_surface)
    rmse_u = rmse_upper.finalize().cpu().numpy()
    rmse_d = rmse_diag.finalize().cpu().numpy() if rmse_diag is not None else None
    acc_s = acc_surface.finalize().cpu().numpy() if acc_surface is not None else None

    surface_names = [str(s) for s in pred_ds["surface_var"].values]
    upper_names = [str(s) for s in pred_ds["upper_air_var"].values]
    diag_names = (
        [str(s) for s in pred_ds["diag_var"].values] if has_diag else []
    )
    step_indices = pred_ds["step"].values.tolist()

    summary: dict = {
        "n_ic": n_ic,
        "ensemble_size": n_ens,
        "max_step": max_step,
        "step_indices": step_indices,
        "rmse_surface": {
            v: rmse_s[:, i].tolist() for i, v in enumerate(surface_names)
        },
        "rmse_upper_air": {
            v: rmse_u[:, i].tolist() for i, v in enumerate(upper_names)
        },
    }
    if rmse_d is not None:
        summary["rmse_diagnostic"] = {
            v: rmse_d[:, i].tolist() for i, v in enumerate(diag_names)
        }
    if acc_s is not None:
        summary["acc_surface"] = {
            v: acc_s[:, i].tolist() for i, v in enumerate(surface_names)
        }

    out_json = _resolve_path(str(vcfg.output_json))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"wrote {out_json}")

    # Markdown summary (channel-mean per step).
    md_path = vcfg.get("output_md", None)
    if md_path:
        md_path = _resolve_path(str(md_path))
        lines = [
            "# ai_rossby validate_cli summary",
            "",
            f"* predictions: `{pred_path}`",
            f"* reference: `{ref_zarr}`",
            f"* ICs: {n_ic}, ensemble_size: {n_ens}, max_step: {max_step}",
            "",
            "## Per-step channel-mean RMSE",
            "",
            "| step | surface | upper_air"
            + (" | diagnostic" if rmse_d is not None else "")
            + " |",
            "|---|---|---"
            + ("|---|" if rmse_d is not None else "|"),
        ]
        for si, s in enumerate(step_indices):
            row = [
                f"{int(s)}",
                f"{rmse_s[si].mean():.4f}",
                f"{rmse_u[si].mean():.4f}",
            ]
            if rmse_d is not None:
                row.append(f"{rmse_d[si].mean():.4f}")
            lines.append("| " + " | ".join(row) + " |")
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()
