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
from async_writer import (
    AsyncForecastWriter,
    format_time_for_filename,
    make_forecast_filename,
    subset_forecast_dataset,
)
from climatology import (
    StreamingBinnedMean,
    StreamingBinnedVariance,
    StreamingTimeMean,
    StreamingTimeVariance,
    lat_weighted_global_scalars,
)
from validate import Deterministic, GaussianIC, Perturber, ReplicateOnly


def _resolve_path(p: Optional[str]) -> Optional[str]:
    return to_absolute_path(p) if p else None


class _NullWriterContext:
    """Stand-in for AsyncForecastWriter when forecast-chunk dumping is off.

    Lets the ``with writer_ctx as forecast_writer:`` pattern stay
    unconditional in main() — yields ``None`` so downstream guards on
    ``forecast_writer is not None`` make the call a no-op.
    """

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


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
    track_binned_variance: bool = False,
) -> dict[str, dict]:
    """Build the ``(mean, variance, [binned_mean, binned_var])`` aggregators per group.

    Returned dict keyed by group name (``surface`` / ``upper_air`` /
    ``diagnostic``); each value is a dict
    ``{mean, var, binned, binned_var}``. ``binned``/``binned_var``
    are ``None`` when their respective flags are off (memory tight).
    """
    out: dict[str, dict] = {}
    for grp, sh in shapes.items():
        slot: dict = {
            "mean": StreamingTimeMean(sh, device),
            "var": StreamingTimeVariance(sh, device),
        }
        slot["binned"] = StreamingBinnedMean(n_bins, sh, device) if track_bins else None
        slot["binned_var"] = (
            StreamingBinnedVariance(n_bins, sh, device)
            if (track_bins and track_binned_variance)
            else None
        )
        out[grp] = slot
    return out


def _update_set(
    pred_agg: dict, truth_agg: dict, pred: torch.Tensor, truth: torch.Tensor, bin_idx: torch.Tensor
) -> None:
    """Push ``(pred, truth)`` into the matching aggregator slot."""
    pred_agg["mean"].update(pred)
    truth_agg["mean"].update(truth)
    pred_agg["var"].update(pred)
    truth_agg["var"].update(truth)
    if pred_agg["binned"] is not None:
        pred_agg["binned"].update(pred, bin_idx)
        truth_agg["binned"].update(truth, bin_idx)
    if pred_agg["binned_var"] is not None:
        pred_agg["binned_var"].update(pred, bin_idx)
        truth_agg["binned_var"].update(truth, bin_idx)


# ---------------------------------------------------------------------------
# Forecast chunk buffer (optional non-overlapping time-range zarr dump)
# ---------------------------------------------------------------------------


class _ForecastChunkBuffer:
    r"""Accumulates rollout frames for one non-overlapping chunk file.

    Each chunk represents a contiguous time range
    ``[t_start_step, t_start_step + chunk_steps)``. When the buffer
    fills to ``chunk_steps`` frames it's converted to an xarray
    Dataset, handed to the writer, and reset for the next chunk.

    The IC frame (step 0) lands in the *first* chunk only — so the
    chunk file is self-contained for replay but no timestamp is
    duplicated across files (chunk 1 starts at step ``chunk_steps + 1``
    when the IC was put in chunk 0).

    Memory: ``O(ensemble × chunk_steps × Σ_groups C_g × H × W × 4 B)``.
    For PLASIM at 53 channels × 64 × 128 × E=1 × chunk_steps=40 (10
    days at 6 h cadence) that's ~70 MB per buffer instance — fits a
    single chunk in flight while the next one fills.
    """

    def __init__(
        self,
        *,
        chunk_steps: int,
        ensemble_size: int,
        layout: dict,
        has_diagnostic: bool,
    ):
        self.chunk_steps = int(chunk_steps)
        self.ensemble_size = int(ensemble_size)
        self.layout = layout
        self.has_diagnostic = has_diagnostic
        self._buffer: dict[str, list] = {}
        self._frame_meta: list[dict] = []  # one per frame: {"step": k, "time_value": ...}
        self._chunk_idx = 0

    def _reset(self) -> None:
        self._buffer = {"surface": [], "upper_air": []}
        if self.has_diagnostic:
            self._buffer["diagnostic"] = []
        self._frame_meta = []

    def append(
        self,
        *,
        step: int,
        time_value,
        pred_surface_np: np.ndarray,
        pred_upper_np: np.ndarray,
        pred_diag_np: Optional[np.ndarray],
    ) -> None:
        if not self._buffer:
            self._reset()
        self._buffer["surface"].append(pred_surface_np)
        self._buffer["upper_air"].append(pred_upper_np)
        if self.has_diagnostic and pred_diag_np is not None:
            self._buffer["diagnostic"].append(pred_diag_np)
        self._frame_meta.append({"step": int(step), "time_value": time_value})

    @property
    def is_full(self) -> bool:
        return len(self._frame_meta) >= self.chunk_steps

    @property
    def is_empty(self) -> bool:
        return len(self._frame_meta) == 0

    def to_dataset(
        self,
        *,
        ic_index: int,
        run_id: str,
        ic_time=None,
    ) -> xr.Dataset:
        """Stack the buffered frames into an xr.Dataset for the writer."""
        if self.is_empty:
            raise RuntimeError("flush() called on empty chunk buffer")
        n_frames = len(self._frame_meta)
        # Each buffer entry is shape (ensemble, ...). Stack along the frame axis.
        surf_arr = np.stack(self._buffer["surface"], axis=1)  # (E, frame, C_s, H, W)
        upper_arr = np.stack(self._buffer["upper_air"], axis=1)
        data_vars = {
            "pred_surface": (
                ("ensemble", "frame", "surface_var", "lat", "lon"),
                surf_arr,
            ),
            "pred_upper_air": (
                ("ensemble", "frame", "upper_air_var", "level", "lat", "lon"),
                upper_arr,
            ),
        }
        if self.has_diagnostic and "diagnostic" in self._buffer:
            diag_arr = np.stack(self._buffer["diagnostic"], axis=1)
            data_vars["pred_diagnostic"] = (
                ("ensemble", "frame", "diag_var", "lat", "lon"),
                diag_arr,
            )

        coords = {
            "ensemble": ("ensemble", np.arange(self.ensemble_size, dtype=np.int64)),
            "frame": ("frame", np.arange(n_frames, dtype=np.int64)),
            "step": ("frame", np.asarray([m["step"] for m in self._frame_meta], dtype=np.int64)),
            "surface_var": ("surface_var", np.asarray(list(self.layout["surface_variables"]))),
            "upper_air_var": ("upper_air_var", np.asarray(list(self.layout["upper_air_variables"]))),
            "level": ("level", np.asarray(list(self.layout["levels_coord"]), dtype=np.float32)),
            "lat": ("lat", self.layout["lat"].astype(np.float32)),
            "lon": ("lon", self.layout["lon"].astype(np.float32)),
        }
        if self.has_diagnostic and "diagnostic" in self._buffer:
            coords["diag_var"] = (
                "diag_var",
                np.asarray(list(self.layout["diagnostic_variables"])),
            )
        # Time coord if available (cftime-aware).
        if any(m["time_value"] is not None for m in self._frame_meta):
            coords["time"] = (
                "frame",
                np.asarray([m["time_value"] for m in self._frame_meta]),
            )

        if ic_time is not None:
            # 0-d scalar coord — actual datetime the IC was drawn from.
            coords["ic_time"] = np.asarray(ic_time)
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs["ic_index"] = int(ic_index)
        ds.attrs["chunk_index"] = int(self._chunk_idx)
        ds.attrs["chunk_steps_target"] = int(self.chunk_steps)
        ds.attrs["run_id"] = str(run_id)
        ds.attrs["step_start"] = int(self._frame_meta[0]["step"])
        ds.attrs["step_end"] = int(self._frame_meta[-1]["step"])
        if ic_time is not None:
            ds.attrs["ic_time"] = str(ic_time)
        return ds

    def flush(
        self,
        *,
        writer: AsyncForecastWriter,
        output_dir: str,
        ic_index: int,
        model_name: str,
        run_name: str,
        extension: str,
        save_variables: Optional[dict] = None,
        ic_time=None,
    ) -> Optional[str]:
        """If non-empty, build dataset, submit, advance chunk counter.

        Returns the path that was submitted (or None when empty).

        When ``save_variables`` is non-empty it filters the on-disk
        payload down to a subset of channels / levels — same dict
        contract as the inference path (see
        :func:`async_writer.subset_forecast_dataset`). ``None`` keeps
        the chunk intact.

        ``ic_time``, when provided, is attached as a scalar coord +
        attribute so downstream consumers can recover the actual IC
        datetime without indexing the (possibly subsetted) frame axis.
        """
        if self.is_empty:
            return None
        run_id = f"{model_name}__{run_name}__ic{ic_index}"
        ds = self.to_dataset(ic_index=ic_index, run_id=run_id, ic_time=ic_time)
        if save_variables:
            ds = subset_forecast_dataset(
                ds,
                surface=save_variables.get("surface"),
                upper_air=save_variables.get("upper_air"),
                upper_air_levels=save_variables.get("upper_air_levels"),
                diagnostic=save_variables.get("diagnostic"),
            )
        t_start = self._frame_meta[0]["time_value"]
        t_end = self._frame_meta[-1]["time_value"]
        # Fall back to step indices when there's no time coord.
        start_token = (
            format_time_for_filename(t_start)
            if t_start is not None
            else f"step{self._frame_meta[0]['step']}"
        )
        end_token = (
            format_time_for_filename(t_end)
            if t_end is not None
            else f"step{self._frame_meta[-1]['step']}"
        )
        fname = make_forecast_filename(
            model_name=model_name,
            run_name=f"{run_name}__ic{ic_index}",
            start_time=start_token,
            end_time=end_token,
            extra=f"chunk{self._chunk_idx:04d}",
            extension=extension,
        )
        path = str(Path(output_dir) / fname)
        writer.submit(path, ds)
        self._chunk_idx += 1
        self._reset()
        return path


def _extract_layout(dataset: PlasimClimateDataset, has_diagnostic: bool) -> dict:
    """Layout introspection shared by the run + chunk buffer."""
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
    track_binned_variance: bool = False,
    forecast_chunk_steps: Optional[int] = None,
    forecast_writer: Optional[AsyncForecastWriter] = None,
    forecast_output_dir: Optional[str] = None,
    model_name: str = "model",
    run_name: str = "run",
    forecast_extension: str = "zarr",
    include_ic_in_forecast: bool = True,
    forecast_save_variables: Optional[dict] = None,
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

    layout = _extract_layout(dataset, has_diagnostic)
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
        n_bins=n_bins,
        shapes=shapes,
        device=device,
        track_bins=track_bins,
        track_binned_variance=track_binned_variance,
    )
    truth_set = _make_aggregator_set(
        n_bins=n_bins,
        shapes=shapes,
        device=device,
        track_bins=track_bins,
        track_binned_variance=track_binned_variance,
    )

    # Optional chunked-forecast dumping. Requires both the chunk size and
    # the writer + output dir to be configured.
    dump_chunks = (
        forecast_chunk_steps is not None
        and forecast_chunk_steps > 0
        and forecast_writer is not None
        and forecast_output_dir is not None
    )

    all_forecast_paths: list[str] = []
    for ic_pos, ic in enumerate(ic_indices):
        if logger is not None:
            logger.info(
                f"climatology rollout {ic_pos+1}/{len(ic_indices)} from IC {ic} "
                f"({max_step} steps × {ensemble_size} ensemble); "
                f"dump_chunks={dump_chunks} (chunk_steps={forecast_chunk_steps})"
            )

        # Per-IC chunk buffer. The IC frame (step 0) is included in the
        # first chunk only — frames in later chunks start at step
        # chunk_steps+1 to keep timestamps non-overlapping across files.
        chunk_buf: Optional[_ForecastChunkBuffer] = None
        if dump_chunks:
            chunk_buf = _ForecastChunkBuffer(
                chunk_steps=int(forecast_chunk_steps),
                ensemble_size=ensemble_size,
                layout=layout,
                has_diagnostic=has_diagnostic,
            )
        # IC datetime (cftime-aware) when the dataset has a time coord —
        # plumbed into every chunk written for this IC.
        ic_time_value = (
            layout["time"][int(ic)]
            if layout["time"] is not None and int(ic) < len(layout["time"])
            else None
        )

        init = _fetch_input_at(dataset, int(ic), normalizer, device)
        state = perturber(init, ensemble_size, generator=rng)
        const_boundary = state.get("constant_boundary")

        # Include IC at step 0 of chunk 0 when requested.
        if dump_chunks and include_ic_in_forecast:
            ic_surface_np = init["surface_in"].cpu().numpy().astype(np.float32)
            ic_upper_np = init["upper_air_in"].cpu().numpy().astype(np.float32)
            ic_diag_np = (
                init.get("diagnostic", None)
            )
            ic_diag_np = (
                ic_diag_np.cpu().numpy().astype(np.float32)
                if ic_diag_np is not None
                else None
            )
            t_ic = layout["time"][int(ic)] if layout["time"] is not None else None
            # Broadcast IC across the ensemble axis if needed.
            if ensemble_size > 1:
                ic_surface_np = np.broadcast_to(
                    ic_surface_np, (ensemble_size,) + ic_surface_np.shape[1:]
                ).copy()
                ic_upper_np = np.broadcast_to(
                    ic_upper_np, (ensemble_size,) + ic_upper_np.shape[1:]
                ).copy()
                if ic_diag_np is not None:
                    ic_diag_np = np.broadcast_to(
                        ic_diag_np, (ensemble_size,) + ic_diag_np.shape[1:]
                    ).copy()
            chunk_buf.append(
                step=0,
                time_value=t_ic,
                pred_surface_np=ic_surface_np,
                pred_upper_np=ic_upper_np,
                pred_diag_np=ic_diag_np,
            )

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

            # Reduce the ensemble axis before the climatology aggregator:
            # one prediction frame per step, matched to the one truth frame.
            # The full ensemble is still available via the chunked-forecast
            # dump (which keeps the ensemble axis).
            pred_surface_mean = (
                next_surface
                if ensemble_size == 1
                else next_surface.mean(dim=0, keepdim=True)
            )
            pred_upper_mean = (
                next_upper
                if ensemble_size == 1
                else next_upper.mean(dim=0, keepdim=True)
            )
            pred_diag_mean = (
                next_diag
                if (next_diag is None or ensemble_size == 1)
                else next_diag.mean(dim=0, keepdim=True)
            )
            bin_idx = torch.full((1,), bin_value, dtype=torch.long, device=device)

            _update_set(
                pred_set["surface"], truth_set["surface"],
                pred_surface_mean, truth_t["surface_in"], bin_idx,
            )
            _update_set(
                pred_set["upper_air"], truth_set["upper_air"],
                pred_upper_mean, truth_t["upper_air_in"], bin_idx,
            )
            if pred_diag_mean is not None and "diagnostic" in pred_set:
                _update_set(
                    pred_set["diagnostic"], truth_set["diagnostic"],
                    pred_diag_mean,
                    truth_t.get("diagnostic", torch.zeros_like(pred_diag_mean)),
                    bin_idx,
                )

            # Optional forecast chunk write — runs after aggregator update
            # so the aggregator state always reflects "all frames ever
            # observed", independent of whether the chunk has been flushed.
            # Chunk dumps are written in PHYSICAL UNITS (the aggregator
            # statistics above stay in normalized space — that's a separate
            # axis from the on-disk chunk payload).
            if chunk_buf is not None:
                t_k = (
                    layout["time"][int(ic) + int(k)]
                    if layout["time"] is not None
                    and int(ic) + int(k) < len(layout["time"])
                    else None
                )
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
                # next_surface has leading dim (ensemble,) when ensemble_size>1,
                # else (1, ...). Cast to (E, ...) numpy.
                ns_np = phys["surface"].cpu().numpy().astype(np.float32)
                nu_np = phys["upper_air"].cpu().numpy().astype(np.float32)
                nd_np = (
                    phys["diagnostic"].cpu().numpy().astype(np.float32)
                    if phys.get("diagnostic") is not None
                    else None
                )
                chunk_buf.append(
                    step=k,
                    time_value=t_k,
                    pred_surface_np=ns_np,
                    pred_upper_np=nu_np,
                    pred_diag_np=nd_np,
                )
                if chunk_buf.is_full:
                    path = chunk_buf.flush(
                        writer=forecast_writer,
                        output_dir=forecast_output_dir,
                        ic_index=int(ic),
                        model_name=model_name,
                        run_name=run_name,
                        extension=forecast_extension,
                        save_variables=forecast_save_variables,
                        ic_time=ic_time_value,
                    )
                    if path is not None:
                        all_forecast_paths.append(path)

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

        # End of rollout for this IC — flush whatever's left in the buffer.
        if chunk_buf is not None and not chunk_buf.is_empty:
            path = chunk_buf.flush(
                writer=forecast_writer,
                output_dir=forecast_output_dir,
                ic_index=int(ic),
                model_name=model_name,
                run_name=run_name,
                extension=forecast_extension,
                save_variables=forecast_save_variables,
                ic_time=ic_time_value,
            )
            if path is not None:
                all_forecast_paths.append(path)

    # Finalize.
    finalized = {
        "pred": {},
        "truth": {},
        "shapes": shapes,
        "bin_counts": {},
        "forecast_paths": all_forecast_paths,
    }
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
        if pred.get("binned_var") is not None:
            pbm, pbv = pred["binned_var"].finalize()
            tbm, tbv = truth["binned_var"].finalize()
            finalized["pred"][grp]["binned_var"] = pbv.cpu()
            finalized["truth"][grp]["binned_var"] = tbv.cpu()
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

    # ---------- Lat-weighted global scalars ------------------------------
    # Tiny per-channel-(level) summary fields derived from the per-pixel
    # mean/var/bias fields. ``global_*`` vars carry no spatial dims so
    # they're cheap to plot/tabulate without re-loading the full grid.
    import torch as _torch
    for grp_name, dim_tail, dims_no_spatial in (
        ("surface", "surface_var", ("surface_var",)),
        ("upper_air", "upper_air_var", ("upper_air_var", "level")),
    ) + (
        (("diagnostic", "diag_var", ("diag_var",)),)
        if has_diagnostic and "diagnostic" in aggregated["shapes"]
        else ()
    ):
        for src_name, dst_name in (
            (f"pred_mean_{grp_name}", f"global_pred_mean_{grp_name}"),
            (f"truth_mean_{grp_name}", f"global_truth_mean_{grp_name}"),
            (f"bias_{grp_name}", f"global_bias_{grp_name}"),
            (f"pred_var_{grp_name}", f"global_pred_var_{grp_name}"),
            (f"truth_var_{grp_name}", f"global_truth_var_{grp_name}"),
            (f"var_bias_{grp_name}", f"global_var_bias_{grp_name}"),
        ):
            if src_name not in data_vars:
                continue
            arr = _torch.from_numpy(data_vars[src_name][1])
            scalars = lat_weighted_global_scalars(arr).numpy()
            data_vars[dst_name] = (dims_no_spatial, scalars)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["ic_indices"] = np.asarray(list(ic_indices), dtype=np.int64)
    ds.attrs["max_step"] = int(max_step)
    ds.attrs["ensemble_size"] = int(ensemble_size)
    ds.attrs["n_bins"] = int(n_bins)
    ds.attrs["steps_per_bin"] = int(steps_per_bin)
    for grp, counts in aggregated["bin_counts"].items():
        ds.attrs[f"bin_counts_{grp}"] = counts.numpy().astype(np.int64)
    if aggregated.get("forecast_paths"):
        ds.attrs["forecast_chunk_paths"] = np.asarray(
            list(aggregated["forecast_paths"]), dtype=object
        )
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
    track_binned_variance = bool(ccfg.get("track_binned_variance", False))
    perturber = _build_perturber(
        str(ccfg.get("perturber", "deterministic")),
        OmegaConf.to_container(ccfg.get("perturber_scales", {}), resolve=True) or {},
    )

    # Optional chunked forecast dumping with non-overlapping timestamps.
    forecast_chunk_steps = ccfg.get("forecast_chunk_steps", None)
    forecast_output_dir = ccfg.get("forecast_output_dir", None)
    forecast_extension = str(ccfg.get("forecast_extension", "zarr"))
    writer_max_in_flight = int(ccfg.get("writer_max_in_flight", 4))
    writer_num_workers = int(ccfg.get("writer_num_workers", 2))
    include_ic_in_forecast = bool(ccfg.get("include_ic_in_forecast", True))
    forecast_save_variables = (
        OmegaConf.to_container(ccfg.forecast_save_variables, resolve=True)
        if "forecast_save_variables" in ccfg
        else None
    )
    dump_forecast = (
        forecast_chunk_steps is not None
        and int(forecast_chunk_steps) > 0
        and forecast_output_dir is not None
    )
    forecast_output_dir = (
        _resolve_path(str(forecast_output_dir)) if dump_forecast else None
    )

    writer_ctx = (
        AsyncForecastWriter(
            max_in_flight=writer_max_in_flight, num_workers=writer_num_workers
        )
        if dump_forecast
        else _NullWriterContext()
    )

    with writer_ctx as forecast_writer:
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
            track_binned_variance=track_binned_variance,
            forecast_chunk_steps=(
                int(forecast_chunk_steps) if dump_forecast else None
            ),
            forecast_writer=forecast_writer if dump_forecast else None,
            forecast_output_dir=forecast_output_dir if dump_forecast else None,
            model_name=str(cfg.model.name),
            run_name=str(cfg.run_name),
            forecast_extension=forecast_extension,
            include_ic_in_forecast=include_ic_in_forecast,
            forecast_save_variables=forecast_save_variables,
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
