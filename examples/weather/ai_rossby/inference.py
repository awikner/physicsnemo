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
from async_writer import (
    AsyncForecastWriter,
    format_time_for_filename,
    make_forecast_filename,
    subset_forecast_dataset,
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


def _build_per_ic_dataset(
    *,
    ic_index: int,
    n_frames: int,
    ensemble_size: int,
    lat: np.ndarray,
    lon: np.ndarray,
    surface_variables: Sequence[str],
    upper_air_variables: Sequence[str],
    diagnostic_variables: Sequence[str],
    levels: Sequence[float],
    n_levels: int,
    has_diagnostic: bool,
    time_values: Optional[Sequence] = None,
    ic_time=None,
    ensemble_save_mode: str = "members",
) -> xr.Dataset:
    r"""Allocate one xarray container per IC.

    Two on-disk layouts are supported via ``ensemble_save_mode``:

    * ``"members"`` (default) — full ensemble axis preserved.
      Shapes::

          pred_surface:   (ensemble, frame, surface_var, lat, lon)
          pred_upper_air: (ensemble, frame, upper_air_var, level, lat, lon)
          pred_diagnostic: (ensemble, frame, diag_var, lat, lon)

    * ``"summary"`` — per-frame ensemble mean + std, no ensemble axis.
      Shapes::

          pred_surface_mean / pred_surface_std:
              (frame, surface_var, lat, lon)
          pred_upper_air_mean / pred_upper_air_std:
              (frame, upper_air_var, level, lat, lon)
          pred_diagnostic_mean / pred_diagnostic_std:
              (frame, diag_var, lat, lon)

      The ``ensemble_size`` attr is retained so downstream consumers can
      back out the per-pixel error estimate. With ``ensemble_size = 1``
      the std fields are written as zeros — equivalent to passing the
      single member through unchanged.

    ``n_frames = max_step + 1`` — frame 0 is the IC, frames 1..max_step
    are the rollout predictions.

    ``time_values`` holds the dataset's time coord at the corresponding
    indices ``ic..ic+max_step``; when provided, the output carries a
    proper ``time`` coord (cftime aware) alongside the integer ``frame``
    index. ``frame == 0`` is the IC.
    """
    if ensemble_save_mode not in ("members", "summary"):
        raise ValueError(
            f"ensemble_save_mode must be 'members' or 'summary', got "
            f"{ensemble_save_mode!r}"
        )

    H, W = lat.shape[0], lon.shape[0]
    n_s = len(surface_variables)
    n_u = len(upper_air_variables)
    n_d = len(diagnostic_variables)

    data_vars: dict = {}
    if ensemble_save_mode == "members":
        data_vars["pred_surface"] = (
            ("ensemble", "frame", "surface_var", "lat", "lon"),
            np.zeros((ensemble_size, n_frames, n_s, H, W), dtype=np.float32),
        )
        data_vars["pred_upper_air"] = (
            ("ensemble", "frame", "upper_air_var", "level", "lat", "lon"),
            np.zeros(
                (ensemble_size, n_frames, n_u, n_levels, H, W), dtype=np.float32
            ),
        )
        if has_diagnostic and n_d > 0:
            data_vars["pred_diagnostic"] = (
                ("ensemble", "frame", "diag_var", "lat", "lon"),
                np.zeros((ensemble_size, n_frames, n_d, H, W), dtype=np.float32),
            )
    else:  # summary
        for kind, name, shape in (
            ("surface", "pred_surface", (n_frames, n_s, H, W)),
            (
                "upper_air",
                "pred_upper_air",
                (n_frames, n_u, n_levels, H, W),
            ),
        ):
            if kind == "surface":
                dims = ("frame", "surface_var", "lat", "lon")
            else:
                dims = ("frame", "upper_air_var", "level", "lat", "lon")
            data_vars[f"{name}_mean"] = (dims, np.zeros(shape, dtype=np.float32))
            data_vars[f"{name}_std"] = (dims, np.zeros(shape, dtype=np.float32))
        if has_diagnostic and n_d > 0:
            dims = ("frame", "diag_var", "lat", "lon")
            data_vars["pred_diagnostic_mean"] = (
                dims,
                np.zeros((n_frames, n_d, H, W), dtype=np.float32),
            )
            data_vars["pred_diagnostic_std"] = (
                dims,
                np.zeros((n_frames, n_d, H, W), dtype=np.float32),
            )

    coords = {
        "frame": ("frame", np.arange(n_frames, dtype=np.int64)),
        "surface_var": ("surface_var", np.asarray(list(surface_variables))),
        "upper_air_var": ("upper_air_var", np.asarray(list(upper_air_variables))),
        "level": ("level", np.asarray(list(levels), dtype=np.float32)),
        "lat": ("lat", lat.astype(np.float32)),
        "lon": ("lon", lon.astype(np.float32)),
    }
    if ensemble_save_mode == "members":
        coords["ensemble"] = (
            "ensemble",
            np.arange(ensemble_size, dtype=np.int64),
        )
    if has_diagnostic and n_d > 0:
        coords["diag_var"] = ("diag_var", np.asarray(list(diagnostic_variables)))
    if time_values is not None:
        coords["time"] = ("frame", np.asarray(list(time_values)))
    if ic_time is not None:
        # 0-d scalar coord — gives users `ds.ic_time` for the actual
        # datetime the IC was drawn from (in the dataset's cftime
        # calendar). Duplicates ds.time.isel(frame=0) but is much
        # cheaper to inspect and survives an upper_air-only subset
        # that drops the frame axis.
        coords["ic_time"] = np.asarray(ic_time)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["ic_index"] = int(ic_index)
    ds.attrs["ensemble_size"] = int(ensemble_size)
    ds.attrs["max_step"] = int(n_frames - 1)
    ds.attrs["frame_zero_is_ic"] = 1
    ds.attrs["ensemble_save_mode"] = ensemble_save_mode
    if ic_time is not None:
        ds.attrs["ic_time"] = str(ic_time)
    return ds


def _write_per_ic_frame(
    ds: xr.Dataset,
    frame: int,
    *,
    ensemble_save_mode: str,
    ensemble_size: int,
    surface: torch.Tensor,
    upper_air: torch.Tensor,
    diagnostic: Optional[torch.Tensor] = None,
) -> None:
    r"""Write one rollout frame's predictions into the per-IC dataset.

    Inputs are flat ``(B*E, C, [L,] H, W)`` tensors in physical units
    (the caller is responsible for denormalization). ``B`` is exactly 1
    here — the per-IC writer hands one IC at a time — so the
    reshape to ``(E, ...)`` is direct.
    """

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    surface_e = surface.view(ensemble_size, *surface.shape[1:])
    upper_e = upper_air.view(ensemble_size, *upper_air.shape[1:])
    diag_e = (
        diagnostic.view(ensemble_size, *diagnostic.shape[1:])
        if diagnostic is not None
        else None
    )

    if ensemble_save_mode == "members":
        ds["pred_surface"][:, frame, ...] = _np(surface_e)
        ds["pred_upper_air"][:, frame, ...] = _np(upper_e)
        if diag_e is not None and "pred_diagnostic" in ds:
            ds["pred_diagnostic"][:, frame, ...] = _np(diag_e)
        return

    # summary mode — per-pixel mean + std across the ensemble axis.
    s_mean = surface_e.mean(dim=0)
    u_mean = upper_e.mean(dim=0)
    if ensemble_size > 1:
        s_std = surface_e.std(dim=0, unbiased=False)
        u_std = upper_e.std(dim=0, unbiased=False)
    else:
        s_std = torch.zeros_like(s_mean)
        u_std = torch.zeros_like(u_mean)
    ds["pred_surface_mean"][frame, ...] = _np(s_mean)
    ds["pred_surface_std"][frame, ...] = _np(s_std)
    ds["pred_upper_air_mean"][frame, ...] = _np(u_mean)
    ds["pred_upper_air_std"][frame, ...] = _np(u_std)
    if diag_e is not None and "pred_diagnostic_mean" in ds:
        d_mean = diag_e.mean(dim=0)
        d_std = (
            diag_e.std(dim=0, unbiased=False)
            if ensemble_size > 1
            else torch.zeros_like(d_mean)
        )
        ds["pred_diagnostic_mean"][frame, ...] = _np(d_mean)
        ds["pred_diagnostic_std"][frame, ...] = _np(d_std)


def _extract_layout(dataset: PlasimClimateDataset) -> dict:
    """Layout introspection (run once per inference run).

    Returns surface/upper_air/diagnostic variable names + level coord +
    lat/lon arrays + time coord — everything the per-IC builder needs.
    """
    sample = dataset[0]
    surface_variables = list(dataset.layout.surface_variables)
    upper_air_variables = list(
        dataset.layout.sigma_upper_air_variables
        + dataset.layout.pressure_upper_air_variables
    )
    diagnostic_variables = list(dataset.layout.diagnostic_variables)
    n_levels = sample["upper_air_in"].shape[1] if "upper_air_in" in sample else 1
    sigma_levels = list(getattr(dataset, "sigma_levels", []))
    pressure_levels = list(getattr(dataset, "pressure_levels", []))
    if sigma_levels and len(sigma_levels) == n_levels:
        levels_coord = sigma_levels
    elif pressure_levels and len(pressure_levels) == n_levels:
        levels_coord = pressure_levels
    else:
        levels_coord = list(range(n_levels))
    lat_arr = np.asarray(dataset._ds["lat"].values, dtype=np.float32)  # type: ignore[attr-defined]
    lon_arr = np.asarray(dataset._ds["lon"].values, dtype=np.float32)  # type: ignore[attr-defined]
    time_arr = (
        np.asarray(dataset._ds["time"].values)  # type: ignore[attr-defined]
        if "time" in dataset._ds.coords  # type: ignore[attr-defined]
        else None
    )
    return {
        "surface_variables": surface_variables,
        "upper_air_variables": upper_air_variables,
        "diagnostic_variables": diagnostic_variables,
        "n_levels": n_levels,
        "levels_coord": levels_coord,
        "lat": lat_arr,
        "lon": lon_arr,
        "time": time_arr,
    }


@torch.no_grad()
def run_inference_streaming_per_ic(
    model: torch.nn.Module,
    dataset: PlasimClimateDataset,
    *,
    normalizer: Optional[PlasimNormalizer],
    device: torch.device,
    ic_indices: Sequence[int],
    max_step: int,
    writer: AsyncForecastWriter,
    output_dir: str,
    model_name: str,
    run_name: str,
    output_format: str = "zarr",
    ensemble_size: int = 1,
    perturber: Optional[Perturber] = None,
    has_diagnostic: bool = False,
    seed: int = 0,
    save_variables: Optional[dict] = None,
    logger=None,
) -> list[str]:
    r"""Roll each IC out and hand one xr.Dataset per IC to the writer.

    Frame 0 of each per-IC dataset is the **initial condition** (the
    observed state at ``ic_index``); frames 1..max_step are the
    autoregressive predictions. The on-disk values are in **physical
    units** — the rollout runs in normalized space (matching the
    training-time loss frame) but predictions are de-normalized via the
    ``normalizer`` immediately before being written into the per-IC
    Dataset. When ``normalizer`` is ``None`` the rollout already runs
    in raw units, so the saved values are also raw.

    Each per-IC dataset carries a scalar ``ic_time`` coord (and matching
    ``ic_time`` attr as an ISO string) when the source dataset has a
    time coord. ``ds.ic_time`` equals ``ds.time.isel(frame=0)`` but
    survives subsetting that drops the ``frame`` axis.

    Returns the list of paths submitted to the writer (the actual
    flush happens asynchronously — call ``writer.wait_all()`` to
    block).

    ``save_variables``, when not ``None``, is a dict with optional keys
    ``surface``, ``upper_air``, ``upper_air_levels``, ``diagnostic`` —
    each a list of names / levels to keep on disk. Filter happens
    *after* the rollout populates the full Dataset, *before* the writer
    submits it, so the GPU work is unaffected but the on-disk payload
    shrinks. ``None`` (or absent key) keeps that group whole; ``[]``
    drops the group entirely. See
    :func:`async_writer.subset_forecast_dataset` for full semantics.
    """
    if perturber is None:
        perturber = Deterministic() if ensemble_size == 1 else ReplicateOnly()
    rng = torch.Generator(device=device).manual_seed(seed)
    layout = _extract_layout(dataset)

    paths: list[str] = []
    for ic in ic_indices:
        ic = int(ic)
        n_frames = max_step + 1

        # Time slice for this IC (cftime-aware when available).
        if layout["time"] is not None:
            t_slice = layout["time"][ic : ic + n_frames]
            t_start = layout["time"][ic]
            t_end = layout["time"][ic + max_step] if (ic + max_step) < len(layout["time"]) else None
        else:
            t_slice = None
            t_start = ic
            t_end = ic + max_step

        ds = _build_per_ic_dataset(
            ic_index=ic,
            n_frames=n_frames,
            ensemble_size=ensemble_size,
            lat=layout["lat"],
            lon=layout["lon"],
            surface_variables=layout["surface_variables"],
            upper_air_variables=layout["upper_air_variables"],
            diagnostic_variables=layout["diagnostic_variables"],
            levels=layout["levels_coord"],
            n_levels=layout["n_levels"],
            has_diagnostic=has_diagnostic,
            time_values=t_slice,
            ic_time=(t_start if layout["time"] is not None else None),
        )

        # Initial state at ic — populate frame 0 of the output AND seed
        # the rollout. The rollout itself runs in NORMALIZED space
        # (matches the training-time loss frame), but the on-disk
        # tensors are de-normalized back to physical units so users
        # can interpret + plot the saved fields directly. When
        # normalizer is None the rollout is already in raw units and
        # denormalize_state is a no-op.
        init_batch = _maybe_normalize(
            normalizer, _stack_initial(dataset, [ic], device)
        )
        # IC → frame 0 (physical units).
        ic_phys = (
            normalizer.denormalize_state(
                surface=init_batch["surface_in"],
                upper_air=init_batch["upper_air_in"],
                diagnostic=init_batch.get("diagnostic"),
            )
            if normalizer is not None
            else {
                "surface": init_batch["surface_in"],
                "upper_air": init_batch["upper_air_in"],
                "diagnostic": init_batch.get("diagnostic"),
            }
        )
        ds["pred_surface"][:, 0, :, :, :] = (
            ic_phys["surface"].cpu().numpy().astype(np.float32)
        )
        ds["pred_upper_air"][:, 0, :, :, :, :] = (
            ic_phys["upper_air"].cpu().numpy().astype(np.float32)
        )
        if has_diagnostic and "pred_diagnostic" in ds and ic_phys.get("diagnostic") is not None:
            ds["pred_diagnostic"][:, 0, :, :, :] = (
                ic_phys["diagnostic"].cpu().numpy().astype(np.float32)
            )

        state = perturber(init_batch, ensemble_size, generator=rng)
        const_boundary = state.get("constant_boundary")

        for k in range(1, max_step + 1):
            out = model(
                state["surface_in"],
                const_boundary,
                state["varying_boundary"],
                state["upper_air_in"],
            )
            if has_diagnostic:
                next_surface, next_upper, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper = out[0], out[1]
                next_diag = None

            # Denormalize the rollout output once and write to disk;
            # the rollout itself continues with the normalized tensors.
            phys = (
                normalizer.denormalize_state(
                    surface=next_surface,
                    upper_air=next_upper,
                    diagnostic=next_diag,
                )
                if normalizer is not None
                else {
                    "surface": next_surface,
                    "upper_air": next_upper,
                    "diagnostic": next_diag,
                }
            )

            ps = _ensemble_mean_or_passthrough(phys["surface"], 1, ensemble_size)
            ds["pred_surface"][:, k, :, :, :] = (
                ps[0].cpu().numpy().astype(np.float32)
            )
            pu = _ensemble_mean_or_passthrough(phys["upper_air"], 1, ensemble_size)
            ds["pred_upper_air"][:, k, :, :, :, :] = (
                pu[0].cpu().numpy().astype(np.float32)
            )
            if phys.get("diagnostic") is not None and "pred_diagnostic" in ds:
                pd = _ensemble_mean_or_passthrough(phys["diagnostic"], 1, ensemble_size)
                ds["pred_diagnostic"][:, k, :, :, :] = (
                    pd[0].cpu().numpy().astype(np.float32)
                )

            # Advance boundary for next step (time t+k → input boundary).
            target_batch = _maybe_normalize(
                normalizer, _stack_at_step(dataset, [ic + k], device)
            )
            next_boundary = target_batch["varying_boundary"]
            if ensemble_size > 1:
                next_boundary = next_boundary.repeat_interleave(ensemble_size, dim=0)
            state = {
                "surface_in": next_surface,
                "upper_air_in": next_upper,
                "constant_boundary": const_boundary,
                "varying_boundary": next_boundary,
            }

        # Apply on-disk subset before handing to the writer.
        if save_variables:
            ds = subset_forecast_dataset(
                ds,
                surface=save_variables.get("surface"),
                upper_air=save_variables.get("upper_air"),
                upper_air_levels=save_variables.get("upper_air_levels"),
                diagnostic=save_variables.get("diagnostic"),
            )

        # Hand the populated dataset to the async writer.
        fname = make_forecast_filename(
            model_name=model_name,
            run_name=run_name,
            start_time=format_time_for_filename(t_start),
            end_time=format_time_for_filename(t_end if t_end is not None else (ic + max_step)),
            extension=output_format,
        )
        path = str(Path(output_dir) / fname)
        if logger is not None:
            logger.info(f"  → submitting IC {ic} → {fname} (writer in_flight={writer.in_flight})")
        writer.submit(path, ds)
        paths.append(path)

    return paths


def _is_diffusion_model(model) -> bool:
    """Detect whether the model is one of the Phase 8 diffusion wrappers.

    The detection key is ``pack_state`` — every diffusion wrapper exposes
    this helper (single-step *and* rolling). The deterministic
    PanguPlasim / SFNO recipes don't, so the test is exact.
    """
    inner = model.module if hasattr(model, "module") else model
    return hasattr(inner, "pack_state")


def _build_inference_scheduler(cfg: DictConfig, device: torch.device):
    """Instantiate the inference-time diffusion scheduler.

    Order of precedence:

    1. ``cfg.sampler._target_`` is set → instantiate it. This is the
       Phase 8d Q2 = b path: users explicitly choose the inference
       sampler family, decoupled from training.
    2. ``cfg.sampler._target_`` is ``null`` (the ``sampler=from_loss``
       sentinel) → instantiate ``cfg.loss`` instead, so the inference
       scheduler is the same family as the training scheduler.
    """
    sampler_cfg = cfg.get("sampler", None)
    target = (
        sampler_cfg.get("_target_", None) if sampler_cfg is not None else None
    )
    if target is None or target == "null":
        sched = hydra.utils.instantiate(cfg.loss)
    else:
        sched = hydra.utils.instantiate(sampler_cfg)
    if hasattr(sched, "to"):
        sched = sched.to(device)
    return sched


def _stack_window_initial(
    dataset, ic: int, W: int, device: torch.device
) -> dict:
    """Stack ``W`` consecutive frames ending at ``ic`` into ``(1, W, …)``.

    Pairs with the rolling wrappers' ``pack_window_state`` —
    ``surface_in`` ends up shaped ``(1, W, C_s, H, W)`` and
    ``upper_air_in`` ends up shaped ``(1, W, C_u, L, H, W)``.
    """
    if W <= 0:
        raise ValueError(f"window size must be > 0, got {W}")
    frames = [dataset[(int(ic - W + 1 + i), 1)] for i in range(W)]
    out: dict[str, torch.Tensor] = {}
    for k, v0 in frames[0].items():
        if isinstance(v0, torch.Tensor) and v0.dim() >= 1:
            out[k] = (
                torch.stack([f[k] for f in frames], dim=0)
                .unsqueeze(0)
                .to(device)
            )
        elif isinstance(v0, torch.Tensor):
            out[k] = (
                torch.stack([f[k] for f in frames], dim=0)
                .unsqueeze(0)
                .to(device)
            )
    return out


def _denorm_named(normalizer, kind: str, t: torch.Tensor) -> torch.Tensor:
    """Pass ``t`` through ``normalizer.denormalize_state`` for one group.

    Mirrors the deterministic recipe's per-group denorm; ``normalizer``
    can be ``None`` (raw-unit rollout), in which case the tensor passes
    through unchanged.
    """
    if normalizer is None:
        return t
    return normalizer.denormalize_state(**{kind: t})[kind]


@torch.no_grad()
def run_diffusion_inference_streaming_per_ic(
    model: torch.nn.Module,
    dataset: ClimateZarrDataset,
    *,
    scheduler,
    normalizer,
    device: torch.device,
    ic_indices: Sequence[int],
    max_step: int,
    writer: AsyncForecastWriter,
    output_dir: str,
    model_name: str,
    run_name: str,
    output_format: str = "zarr",
    ensemble_size: int = 1,
    perturber: Optional[Perturber] = None,
    has_diagnostic: bool = False,
    sampler_num_steps: Optional[int] = None,
    seed: int = 0,
    save_variables: Optional[dict] = None,
    ensemble_save_mode: str = "members",
    logger=None,
) -> list[str]:
    r"""Diffusion rollout — per-IC streaming, ensemble-aware.

    Mirrors :func:`run_inference_streaming_per_ic` (the deterministic
    branch) but the inner step is a full diffusion sample:

    * Single-step schedulers (``DriftScheduler``, ``DynamicInterpolant``,
      ``EDMSchedulerModule``) — autoregressive: one
      ``scheduler.sample(model, x, c_grid, c_scalar, num_steps=…)`` call
      per emitted frame, unpacked back into structured channels by the
      wrapper between steps.
    * Rolling schedulers (``ERDMScheduler``, ``RFMScheduler``) — one
      ``scheduler.sample_rollout(model, init_window, c_grid_traj,
      c_scalar_traj, horizon=max_step, num_steps=…)`` call that emits
      the full ``max_step`` window at once.

    The dataset must be opened with ``emit_calendar=True`` so each
    sample carries the ``calendar`` tensor the wrapper needs for
    ``c_scalar``. Per Q3 = a (rolling models), frame 0 of the per-IC
    output is the IC's *last* window frame in physical units; frames
    1..max_step are the predicted future states. Per Q4 = a, the
    default perturber is :class:`ReplicateOnly` for
    ``ensemble_size > 1`` (the ensemble axis is driven by the
    sampler's internal noise, not IC perturbation).
    """
    if perturber is None:
        perturber = Deterministic() if ensemble_size == 1 else ReplicateOnly()
    rng = torch.Generator(device=device).manual_seed(seed)

    inner_model = model.module if hasattr(model, "module") else model

    # Detect single-step vs window-rollout scheduler.
    window_mode = hasattr(scheduler, "sample_rollout")
    window_size = (
        int(getattr(scheduler, "window_size", 0)) if window_mode else 0
    )

    layout = _extract_layout(dataset)
    paths: list[str] = []

    for ic in ic_indices:
        ic = int(ic)
        n_frames = max_step + 1

        if layout["time"] is not None:
            t_slice = layout["time"][ic : ic + n_frames]
            t_start = layout["time"][ic]
            t_end = (
                layout["time"][ic + max_step]
                if (ic + max_step) < len(layout["time"])
                else None
            )
        else:
            t_slice = None
            t_start = ic
            t_end = ic + max_step

        ds = _build_per_ic_dataset(
            ic_index=ic,
            n_frames=n_frames,
            ensemble_size=ensemble_size,
            lat=layout["lat"],
            lon=layout["lon"],
            surface_variables=layout["surface_variables"],
            upper_air_variables=layout["upper_air_variables"],
            diagnostic_variables=layout["diagnostic_variables"],
            levels=layout["levels_coord"],
            n_levels=layout["n_levels"],
            has_diagnostic=has_diagnostic,
            time_values=t_slice,
            ic_time=(t_start if layout["time"] is not None else None),
            ensemble_save_mode=ensemble_save_mode,
        )

        # Frame 0 = IC observation (physical units). For single-step
        # rollouts the IC is at time ``ic``; for window rollouts the IC
        # frame on disk is the *last* of the W oracle frames (Q3 = a).
        ic_phys_state = _maybe_normalize(
            normalizer, _stack_initial(dataset, [ic], device)
        )
        ic_surface = _denorm_named(normalizer, "surface", ic_phys_state["surface_in"])
        ic_upper = _denorm_named(
            normalizer, "upper_air", ic_phys_state["upper_air_in"]
        )
        ic_diag = (
            _denorm_named(normalizer, "diagnostic", ic_phys_state["diagnostic"])
            if has_diagnostic and "diagnostic" in ic_phys_state
            else None
        )
        # Tile the IC across the ensemble axis for the "members" layout
        # (frame 0 is identical across members) — keeps the file
        # self-describing without forcing the caller to special-case
        # frame 0. The summary writer handles the broadcast internally.
        ic_surface_e = ic_surface.repeat_interleave(ensemble_size, dim=0)
        ic_upper_e = ic_upper.repeat_interleave(ensemble_size, dim=0)
        ic_diag_e = (
            ic_diag.repeat_interleave(ensemble_size, dim=0)
            if ic_diag is not None
            else None
        )
        _write_per_ic_frame(
            ds,
            frame=0,
            ensemble_save_mode=ensemble_save_mode,
            ensemble_size=ensemble_size,
            surface=ic_surface_e,
            upper_air=ic_upper_e,
            diagnostic=ic_diag_e,
        )

        # --- Rollout ---------------------------------------------------- #
        if window_mode:
            # Build the oracle initial window [ic - W + 1 .. ic], normalize,
            # replicate across the ensemble, pack.
            init_window = _maybe_normalize(
                normalizer, _stack_window_initial(dataset, ic, window_size, device)
            )
            init_window = perturber(init_window, ensemble_size, generator=rng)
            init_y = inner_model.pack_window_state(init_window)

            # Build c_grid_traj + c_scalar_traj over [ic - W + 1 .. ic + max_step - 1].
            traj_len = window_size + max_step - 1
            traj_frames = [
                _maybe_normalize(
                    normalizer,
                    _stack_at_step(
                        dataset, [ic - window_size + 1 + j], device
                    ),
                )
                for j in range(traj_len)
            ]

            def _stack_traj(key: str) -> torch.Tensor:
                return torch.stack([f[key] for f in traj_frames], dim=1)

            const_boundary = traj_frames[0]["constant_boundary"]
            c_grid_traj = inner_model.pack_window_c_grid(
                {
                    "surface_in": _stack_traj("surface_in"),
                    "constant_boundary": const_boundary,
                    "varying_boundary": _stack_traj("varying_boundary"),
                }
            )
            c_scalar_traj = _stack_traj("calendar")
            if ensemble_size > 1:
                c_grid_traj = c_grid_traj.repeat_interleave(
                    ensemble_size, dim=0
                )
                c_scalar_traj = c_scalar_traj.repeat_interleave(
                    ensemble_size, dim=0
                )

            traj_pred = scheduler.sample_rollout(
                model,
                init_y,
                c_grid_traj,
                c_scalar_traj,
                horizon=max_step,
                num_steps=sampler_num_steps,
            )  # (E, max_step, C_packed, H, W)

            for k in range(1, max_step + 1):
                packed = traj_pred[:, k - 1]
                unpacked = inner_model.unpack_state(packed)
                surface_phys = _denorm_named(
                    normalizer, "surface", unpacked["surface_in"]
                )
                upper_phys = _denorm_named(
                    normalizer, "upper_air", unpacked["upper_air_in"]
                )
                diag_phys = (
                    _denorm_named(
                        normalizer, "diagnostic", unpacked["diagnostic"]
                    )
                    if has_diagnostic and "diagnostic" in unpacked
                    else None
                )
                _write_per_ic_frame(
                    ds,
                    frame=k,
                    ensemble_save_mode=ensemble_save_mode,
                    ensemble_size=ensemble_size,
                    surface=surface_phys,
                    upper_air=upper_phys,
                    diagnostic=diag_phys,
                )
        else:
            # Single-step autoregressive diffusion.
            state = perturber(ic_phys_state, ensemble_size, generator=rng)
            const_boundary = state.get("constant_boundary")
            x = inner_model.pack_state(state)

            for k in range(1, max_step + 1):
                c_grid = inner_model.pack_c_grid(state)
                c_scalar = state["calendar"]
                x_next = scheduler.sample(
                    model, x, c_grid, c_scalar, num_steps=sampler_num_steps
                )
                unpacked = inner_model.unpack_state(x_next)
                surface_phys = _denorm_named(
                    normalizer, "surface", unpacked["surface_in"]
                )
                upper_phys = _denorm_named(
                    normalizer, "upper_air", unpacked["upper_air_in"]
                )
                diag_phys = (
                    _denorm_named(
                        normalizer, "diagnostic", unpacked["diagnostic"]
                    )
                    if has_diagnostic and "diagnostic" in unpacked
                    else None
                )
                _write_per_ic_frame(
                    ds,
                    frame=k,
                    ensemble_save_mode=ensemble_save_mode,
                    ensemble_size=ensemble_size,
                    surface=surface_phys,
                    upper_air=upper_phys,
                    diagnostic=diag_phys,
                )

                # Advance: next state's surface/upper_air/diag come from
                # the diffusion sample. Boundary + calendar march to the
                # next step using the dataset sample at t+k.
                if k < max_step:
                    next_step = _maybe_normalize(
                        normalizer,
                        _stack_at_step(dataset, [ic + k], device),
                    )
                    next_var_boundary = next_step["varying_boundary"]
                    next_calendar = next_step["calendar"]
                    if ensemble_size > 1:
                        next_var_boundary = next_var_boundary.repeat_interleave(
                            ensemble_size, dim=0
                        )
                        next_calendar = next_calendar.repeat_interleave(
                            ensemble_size, dim=0
                        )
                    state = {
                        "surface_in": unpacked["surface_in"],
                        "constant_boundary": const_boundary,
                        "varying_boundary": next_var_boundary,
                        "calendar": next_calendar,
                    }
                    if "upper_air_in" in unpacked:
                        state["upper_air_in"] = unpacked["upper_air_in"]
                    if "diagnostic" in unpacked:
                        state["diagnostic"] = unpacked["diagnostic"]
                    x = x_next

        if save_variables:
            ds = subset_forecast_dataset(
                ds,
                surface=save_variables.get("surface"),
                upper_air=save_variables.get("upper_air"),
                upper_air_levels=save_variables.get("upper_air_levels"),
                diagnostic=save_variables.get("diagnostic"),
            )

        fname = make_forecast_filename(
            model_name=model_name,
            run_name=run_name,
            start_time=format_time_for_filename(t_start),
            end_time=format_time_for_filename(
                t_end if t_end is not None else (ic + max_step)
            ),
            extension=output_format,
        )
        path = str(Path(output_dir) / fname)
        if logger is not None:
            logger.info(
                f"  → submitting diffusion IC {ic} → {fname} "
                f"(writer in_flight={writer.in_flight}, mode={ensemble_save_mode})"
            )
        writer.submit(path, ds)
        paths.append(path)

    return paths


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
    # Diffusion wrappers need a ``calendar`` tensor on each sample; turn
    # the emit on when the model exposes ``pack_state``. Off for the
    # deterministic recipes (preserves existing behavior bit-for-bit).
    is_diffusion = _is_diffusion_model(model)
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
        emit_calendar=is_diffusion,
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

    output_dir = _resolve_path(str(inf_cfg.output_dir))
    output_format = str(inf_cfg.get("output_format", "zarr"))
    max_in_flight = int(inf_cfg.get("writer_max_in_flight", 4))
    num_writers = int(inf_cfg.get("writer_num_workers", 2))
    save_variables = (
        OmegaConf.to_container(inf_cfg.save_variables, resolve=True)
        if "save_variables" in inf_cfg
        else None
    )
    ensemble_save_mode = str(inf_cfg.get("ensemble_save_mode", "members"))

    if is_diffusion:
        scheduler = _build_inference_scheduler(cfg, dist.device)
        sampler_num_steps_raw = inf_cfg.get("sampler_num_steps", None)
        sampler_num_steps = (
            int(sampler_num_steps_raw)
            if sampler_num_steps_raw is not None
            else None
        )
    else:
        scheduler = None
        sampler_num_steps = None

    if dist.rank == 0:
        logger.info(
            f"inference: {len(ic_indices)} ICs × {ensemble_size} ensemble × "
            f"{max_step} steps; perturber={inf_cfg.get('perturber', 'deterministic')!r}; "
            f"output_dir={output_dir} (format={output_format}, "
            f"writer_max_in_flight={max_in_flight}, num_writers={num_writers}, "
            f"mode={'diffusion' if is_diffusion else 'deterministic'}, "
            f"save_mode={ensemble_save_mode})"
        )
        if is_diffusion:
            logger.info(
                f"diffusion: sampler={type(scheduler).__name__}, "
                f"sampler_num_steps={sampler_num_steps}"
            )

    # Async writer overlaps disk I/O with GPU rollout — the next IC's
    # forecast runs while the previous one is being flushed to disk.
    with AsyncForecastWriter(
        max_in_flight=max_in_flight, num_workers=num_writers
    ) as writer:
        if is_diffusion:
            paths = run_diffusion_inference_streaming_per_ic(
                model,
                base_ds,
                scheduler=scheduler,
                normalizer=normalizer,
                device=dist.device,
                ic_indices=ic_indices,
                max_step=max_step,
                writer=writer,
                output_dir=output_dir,
                model_name=str(cfg.model.name),
                run_name=str(cfg.run_name),
                output_format=output_format,
                ensemble_size=ensemble_size,
                perturber=perturber,
                has_diagnostic=getattr(model, "has_diagnostic", False)
                or bool(getattr(cfg.model, "diagnostic_variables", [])),
                sampler_num_steps=sampler_num_steps,
                seed=int(cfg.seed) + 1009,
                save_variables=save_variables,
                ensemble_save_mode=ensemble_save_mode,
                logger=logger,
            )
        else:
            paths = run_inference_streaming_per_ic(
                model,
                base_ds,
                normalizer=normalizer,
                device=dist.device,
                ic_indices=ic_indices,
                max_step=max_step,
                writer=writer,
                output_dir=output_dir,
                model_name=str(cfg.model.name),
                run_name=str(cfg.run_name),
                output_format=output_format,
                ensemble_size=ensemble_size,
                perturber=perturber,
                has_diagnostic=getattr(model, "has_diagnostic", False),
                seed=int(cfg.seed) + 1009,
                save_variables=save_variables,
                logger=logger,
            )
        # __exit__ calls wait_all(): final flush + raise on any worker error.
    logger.info(f"wrote {len(paths)} per-IC forecast files to {output_dir}")
    for p in paths:
        logger.info(f"  {p}")


if __name__ == "__main__":
    main()
