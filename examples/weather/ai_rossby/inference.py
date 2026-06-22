# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""After-the-fact rollout inference for the ai_rossby recipe (Phase 4b).

Loads a trained ai_rossby model checkpoint, marches it autoregressively
out to ``max_step`` from a list of initial-condition (IC) timestamps,
optionally generates an :class:`Perturber`-driven ensemble per IC, and
writes the per-channel-group predictions to a NetCDF (or Zarr) file
that downstream ``validate.py`` consumes.

The script is **memory-conscious**: it materializes the prediction
tensors for one (IC, ensemble) sub-batch at a time and streams them
to disk via xarray's incremental writer. The total in-RAM footprint at
any moment is one ``(B*E, C, H, W)`` state per group plus the boundary
slice for the current step.

Usage::

    python inference.py \\
        model=sfno_plasim_5412 \\
        dataset=plasim_sim52_year12 \\
        +inference.checkpoint_dir=/path/to/checkpoints \\
        +inference.output_path=/path/to/preds.nc \\
        +inference.max_step=20 \\
        +inference.ic_start=[0, 60, 120, 180]

The ``inference.*`` config block (see :func:`_inference_defaults` below for
the schema) controls IC selection, rollout horizon, ensemble settings,
and output paths. A separate Hydra group is intentionally avoided —
inference doesn't share state with training so a top-level CLI is
cleaner than another defaults entry.
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
        ClimateZarrDataset,
        NanFillTransform,
        PlasimClimateDataset,
        PlasimNormalizer,
    )

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger

# Reuse the perturber API + helper from the Phase 4a validator module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import (
    Deterministic,
    GaussianIC,
    Perturber,
    ReplicateOnly,
    cos_lat_weights,
)


def _resolve_path(p: Optional[str]) -> Optional[str]:
    return to_absolute_path(p) if p else None


def _build_perturber(perturber_name: str, perturber_scales: dict) -> Perturber:
    """Translate config strings into a concrete :class:`Perturber`."""
    kind = str(perturber_name).lower()
    if kind in ("deterministic", "off", "none"):
        return Deterministic()
    if kind in ("replicate", "replicate_only", "stochastic_model"):
        return ReplicateOnly()
    if kind in ("gaussian_ic", "ic_gaussian", "gaussian"):
        if not perturber_scales:
            raise ValueError(
                "gaussian_ic perturber requires inference.perturber_scales={var: std, ...}"
            )
        return GaussianIC(scales=dict(perturber_scales))
    raise ValueError(f"unknown perturber={perturber_name!r}")


def _stack_initial(dataset, ic_indices: Sequence[int], device: torch.device) -> dict:
    """Stack a list of per-IC dataset samples (lead=1 placeholders) into one batch.

    Same pattern as :class:`RolloutValidator._stack_initial`; pulled out
    here so the inference CLI doesn't import the validator class.
    """
    samples = [dataset[(int(t), 1)] for t in ic_indices]
    out: dict[str, torch.Tensor] = {}
    for k in samples[0]:
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
            out[k] = torch.stack([s[k] for s in samples], dim=0).to(device)
        elif isinstance(v0, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0).to(device)
    return out


def _stack_at_step(dataset, t_list: Sequence[int], device: torch.device) -> dict:
    samples = [dataset[(int(t), 1)] for t in t_list]
    out: dict[str, torch.Tensor] = {}
    for k in samples[0]:
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
            out[k] = torch.stack([s[k] for s in samples], dim=0).to(device)
    return out


def _maybe_normalize(normalizer, batch: dict) -> dict:
    if normalizer is None:
        return batch
    return normalizer(batch)


def _ensemble_mean_or_passthrough(x: torch.Tensor, n_ic: int, ensemble_size: int) -> torch.Tensor:
    """Reshape ``(n_ic*E, ...)`` to ``(n_ic, E, ...)``; identity when E=1."""
    if ensemble_size == 1:
        return x.unsqueeze(1)  # (n_ic, 1, ...) so the writer always has the ensemble axis
    return x.view(n_ic, ensemble_size, *x.shape[1:])


def _build_xr_dataset(
    *,
    ic_indices: Sequence[int],
    max_step: int,
    ensemble_size: int,
    lat: np.ndarray,
    lon: np.ndarray,
    surface_variables: Sequence[str],
    upper_air_variables: Sequence[str],
    diagnostic_variables: Sequence[str],
    levels: Sequence[float],
    n_levels: int,
    has_diagnostic: bool,
) -> xr.Dataset:
    """Allocate the xarray container for an inference run.

    Shapes:
      pred_surface:   (ic, ensemble, step, surface_var, lat, lon)
      pred_upper_air: (ic, ensemble, step, upper_air_var, level, lat, lon)
      pred_diagnostic: (ic, ensemble, step, diag_var, lat, lon) when has_diagnostic
    """
    n_ic = len(ic_indices)
    H, W = lat.shape[0], lon.shape[0]
    n_s = len(surface_variables)
    n_u = len(upper_air_variables)
    n_d = len(diagnostic_variables)

    data_vars = {
        "pred_surface": (
            ("ic", "ensemble", "step", "surface_var", "lat", "lon"),
            np.zeros((n_ic, ensemble_size, max_step, n_s, H, W), dtype=np.float32),
        ),
        "pred_upper_air": (
            ("ic", "ensemble", "step", "upper_air_var", "level", "lat", "lon"),
            np.zeros((n_ic, ensemble_size, max_step, n_u, n_levels, H, W), dtype=np.float32),
        ),
    }
    if has_diagnostic and n_d > 0:
        data_vars["pred_diagnostic"] = (
            ("ic", "ensemble", "step", "diag_var", "lat", "lon"),
            np.zeros((n_ic, ensemble_size, max_step, n_d, H, W), dtype=np.float32),
        )

    coords = {
        "ic": ("ic", np.asarray(ic_indices, dtype=np.int64)),
        "ensemble": ("ensemble", np.arange(ensemble_size, dtype=np.int64)),
        "step": ("step", np.arange(1, max_step + 1, dtype=np.int64)),
        "surface_var": ("surface_var", np.asarray(list(surface_variables))),
        "upper_air_var": ("upper_air_var", np.asarray(list(upper_air_variables))),
        "level": ("level", np.asarray(list(levels), dtype=np.float32)),
        "lat": ("lat", lat.astype(np.float32)),
        "lon": ("lon", lon.astype(np.float32)),
    }
    if has_diagnostic and n_d > 0:
        coords["diag_var"] = ("diag_var", np.asarray(list(diagnostic_variables)))
    return xr.Dataset(data_vars=data_vars, coords=coords)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: ClimateZarrDataset,
    *,
    normalizer: Optional[PlasimNormalizer],
    device: torch.device,
    ic_indices: Sequence[int],
    max_step: int,
    ensemble_size: int = 1,
    perturber: Optional[Perturber] = None,
    batch_size: int = 1,
    has_diagnostic: bool = False,
    seed: int = 0,
) -> xr.Dataset:
    """Roll the model out from each IC and return a populated xarray dataset.

    Memory pattern matches :class:`RolloutValidator`: at any moment we
    hold the current state of one (batch × ensemble) chunk plus the
    boundary at the current step. Predictions are copied to CPU + numpy
    immediately so the GPU memory bound stays at one rollout window.
    """
    if perturber is None:
        perturber = Deterministic() if ensemble_size == 1 else ReplicateOnly()
    rng = torch.Generator(device=device).manual_seed(seed)

    # Layout introspection: read one frame to size the output dataset.
    sample = dataset[0]
    surface_variables = list(dataset.layout.surface_variables)
    upper_air_variables = list(
        dataset.layout.sigma_upper_air_variables
        + dataset.layout.pressure_upper_air_variables
    )
    diagnostic_variables = list(dataset.layout.diagnostic_variables)
    has_upper_air = "upper_air_in" in sample
    if has_upper_air:
        n_levels = sample["upper_air_in"].shape[1]
    else:
        n_levels = 1
    # Approximate level coord — use the sigma + pressure coord values
    # from the dataset; if both present and equal length, prefer sigma.
    sigma_levels = list(getattr(dataset, "sigma_levels", []))
    pressure_levels = list(getattr(dataset, "pressure_levels", []))
    if sigma_levels and len(sigma_levels) == n_levels:
        levels_coord = sigma_levels
    elif pressure_levels and len(pressure_levels) == n_levels:
        levels_coord = pressure_levels
    else:
        levels_coord = list(range(n_levels))

    # Lat/lon from the dataset's underlying xarray store.
    lat_arr = np.asarray(dataset._ds["lat"].values, dtype=np.float32)  # type: ignore[attr-defined]
    lon_arr = np.asarray(dataset._ds["lon"].values, dtype=np.float32)  # type: ignore[attr-defined]

    out_ds = _build_xr_dataset(
        ic_indices=ic_indices,
        max_step=max_step,
        ensemble_size=ensemble_size,
        lat=lat_arr,
        lon=lon_arr,
        surface_variables=surface_variables,
        upper_air_variables=upper_air_variables,
        diagnostic_variables=diagnostic_variables,
        levels=levels_coord,
        n_levels=n_levels,
        has_diagnostic=has_diagnostic,
    )

    # Iterate over ICs in micro-batches of size batch_size.
    for batch_start in range(0, len(ic_indices), batch_size):
        sub_ic = list(ic_indices[batch_start : batch_start + batch_size])
        n_ic = len(sub_ic)
        if n_ic == 0:
            continue

        init_batch = _maybe_normalize(
            normalizer, _stack_initial(dataset, sub_ic, device)
        )
        state = perturber(init_batch, ensemble_size, generator=rng)
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
                next_surface, next_upper_air, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper_air = out[0], out[1]
                next_diag = None

            # Write predictions back to the output dataset (CPU numpy).
            ps = _ensemble_mean_or_passthrough(next_surface, n_ic, ensemble_size)
            out_ds["pred_surface"][batch_start : batch_start + n_ic, :, k - 1, :, :, :] = (
                ps.cpu().numpy().astype(np.float32)
            )
            pu = _ensemble_mean_or_passthrough(next_upper_air, n_ic, ensemble_size)
            out_ds["pred_upper_air"][
                batch_start : batch_start + n_ic, :, k - 1, :, :, :, :
            ] = pu.cpu().numpy().astype(np.float32)
            if next_diag is not None and "pred_diagnostic" in out_ds:
                pd = _ensemble_mean_or_passthrough(next_diag, n_ic, ensemble_size)
                out_ds["pred_diagnostic"][
                    batch_start : batch_start + n_ic, :, k - 1, :, :, :
                ] = pd.cpu().numpy().astype(np.float32)

            # Advance state. The next step's boundary comes from time t+k;
            # ensemble-repeat the boundary so it matches the rolled-out batch.
            target_times = [t + k for t in sub_ic]
            target_batch = _maybe_normalize(
                normalizer, _stack_at_step(dataset, target_times, device)
            )
            next_boundary = target_batch["varying_boundary"]
            if ensemble_size > 1:
                next_boundary = next_boundary.repeat_interleave(ensemble_size, dim=0)
            state = {
                "surface_in": next_surface,
                "upper_air_in": next_upper_air,
                "constant_boundary": const_boundary,
                "varying_boundary": next_boundary,
            }
    return out_ds


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("ai_rossby_inference")

    inf_cfg = cfg.get("inference", None)
    if inf_cfg is None:
        raise ValueError(
            "inference.* config block missing; add +inference.checkpoint_dir, "
            "+inference.output_path, +inference.max_step, +inference.ic_start "
            "on the command line."
        )

    if dist.rank != 0:
        # Rollout inference is a single-rank job by default. Multi-rank
        # support requires partitioning ICs across ranks + gathering the
        # output dataset — out of scope for the first Phase 4b cut.
        if dist.world_size > 1:
            logger.warning("inference.py runs on rank 0 only; non-rank-0 exiting")
        return

    # --- Load model + checkpoint --------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    flat = OmegaConf.to_container(cfg.model, resolve=True) or {}
    name = str(flat["name"])
    module_path = str(flat["module"])
    args = {k: v for k, v in flat.items() if k not in {"name", "module", "target", "model_type"}}
    model = Module.instantiate(
        {"__name__": name, "__module__": module_path, "__args__": args}
    ).to(dist.device)
    model.eval()

    ckpt_dir = _resolve_path(str(inf_cfg.checkpoint_dir))
    loaded_epoch = load_checkpoint(
        ckpt_dir, models=model, device=dist.device
    )
    logger.info(f"loaded checkpoint epoch={loaded_epoch} from {ckpt_dir}")

    # --- Build dataset + normalizer (no datapipe — direct dataset access) ---
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

    # --- Inference config ---------------------------------------------------
    ic_indices = list(inf_cfg.ic_start)
    max_step = int(inf_cfg.max_step)
    ensemble_size = int(inf_cfg.get("ensemble_size", 1))
    batch_size = int(inf_cfg.get("batch_size", 1))
    perturber = _build_perturber(
        str(inf_cfg.get("perturber", "deterministic")),
        OmegaConf.to_container(inf_cfg.get("perturber_scales", {}), resolve=True) or {},
    )

    if dist.rank == 0:
        logger.info(
            f"inference: {len(ic_indices)} ICs × {ensemble_size} ensemble × "
            f"{max_step} steps; perturber={inf_cfg.get('perturber', 'deterministic')!r}"
        )

    out_ds = run_inference(
        model,
        base_ds,
        normalizer=normalizer,
        device=dist.device,
        ic_indices=ic_indices,
        max_step=max_step,
        ensemble_size=ensemble_size,
        perturber=perturber,
        batch_size=batch_size,
        has_diagnostic=getattr(model, "has_diagnostic", False),
        seed=int(cfg.seed) + 1009,
    )

    # --- Write output -------------------------------------------------------
    output_path = _resolve_path(str(inf_cfg.output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith(".zarr"):
        out_ds.to_zarr(output_path, mode="w", zarr_format=3, consolidated=True)
    else:
        out_ds.to_netcdf(output_path, mode="w")
    logger.info(f"wrote {output_path} ({sum(v.nbytes for v in out_ds.data_vars.values()) / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
