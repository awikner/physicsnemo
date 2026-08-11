#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-coarsen a full-resolution climate Zarr store (Phase 12c.10).

Produces the 45x90 forecaster-training store the amip_v2 rebaseline needs
(`docs/dev/phase12_implementation_plan.md` 12c): every channel is
bilinearly downsampled by ``--factor`` using the **exact** call upstream
amip_v2 uses to build its fast state store and to define the downscaler's
corruption operator (``modules/layers/bilinear.py`` /
``scripts/reassemble.py``)::

    F.interpolate(x, scale_factor=1/factor, mode="bilinear",
                  align_corners=False)

Matching this operator matters: the x_DDC downscaler's ``x0`` is *defined*
as "the blur the forecaster's grid implies", so a store coarsened any other
way would train the forecaster on a slightly different manifold than the
downscaler undoes. (Agreement is at float32 rounding — the kernel's SIMD
accumulation order shifts the last ulp with channel folding, and upstream
folds whole channel groups per frame while this tool works per variable.)

Semantics (mirrors upstream ``reassemble.py``):

* **State groups** (surface / pressure upper-air / diagnostic role lists)
  must be NaN-free — a NaN would bleed through the bilinear kernel, so the
  tool hard-fails if it finds one.
* **Boundary groups** (constant + varying) are NaN-filled first via the
  ``--mask-fill`` map (default matches upstream ``ERDM_co2.yaml``:
  SST-like -> 270.0 K, sea-ice-like -> 0.0), then coarsened.

  ``--smooth-boundaries`` (recommended, and what the PBS script uses)
  fills through the masked-Gaussian coast fade instead of a hard constant.
  This is **not cosmetic in a coarsening pipeline**: a 4x4 block straddling
  a coastline averages real ocean values with whatever fills the land side,
  so a hard 270 K fill drags coastal coarse cells cold (~10 K for a
  half-land block at 290 K SST) while the fade starts at the true coastal
  value and decays outward, landing much closer. Upstream never hits this
  because it never coarsens boundaries (see the resolution note below);
  smoothing is how we keep coastal coarse values honest.

  .. note:: **Resolution divergence from upstream (12c/12d seam).**
     Upstream's fast store keeps boundaries at **native 180x360** and lets
     the model's stride-4 conv reduce them (``c_grid_downsample: 4``); this
     tool coarsens them to 45x90 for a self-contained store paired with
     ``c_grid_downsample: 1``. Consequences: (a) the runtime
     ``smooth_nan_boundaries`` knobs are inert on the output (no NaN
     survives — hence doing the fade *here*), and (b) within-cell boundary
     variance is gone, which is what upstream's 12e ``boundary_pool_stats``
     reads. Pointing ``boundary_zarr_path`` at the full-res store does NOT
     work around this today: constants load from the prognostic store while
     varying boundaries load from the boundary store, and the wrapper's
     ``pack_c_grid`` concatenates them (verified ``RuntimeError`` on the
     mismatch). Closing that is scoped to Phase 12e, where
     ``boundary_pool_stats`` forces the decision.
* **Extra variables** (the converter's not-routed archive-preservation
  channels) are skipped by default — the coarse store is a lean training
  artifact. ``--include-extras`` coarsens them too.
* Coarse ``lat`` / ``lon`` coords are block means of the source coords
  (what align_corners=False pixel averaging implies for pixel-centered
  grids: 1 deg centers -> 88.0, 84.0, ... for factor 4).
* All source attrs are copied; a ``coarsen_*`` provenance block records
  the source store, factor, interpolation call, and fill map.

Usage
-----

::

    python tools/data/amip/coarsen_zarr.py \\
      --input  $AI_ROSSBY_DATA/amip_dailyavg/1981.zarr \\
      --output $AI_ROSSBY_DATA/amip_dailyavg_coarse/1981.zarr \\
      --factor 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

logger = logging.getLogger("coarsen_zarr")

# Upstream amip_v2 ERDM_co2.yaml mask_fill defaults.
DEFAULT_MASK_FILL = {
    "sea_surface_temperature_monthly_interp": 270.0,
    "sea_surface_temperature": 270.0,
    "sea_ice_cover_monthly_interp": 0.0,
    "sea_ice_cover": 0.0,
}

COARSEN_PROVENANCE_VERSION = "1.0"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Full-res Zarr store.")
    p.add_argument("--output", type=Path, required=True, help="Coarse Zarr store.")
    p.add_argument("--factor", type=int, default=4, help="Downsample factor.")
    p.add_argument(
        "--time-block",
        type=int,
        default=64,
        help="Time steps processed (read -> interpolate -> write) per block. "
        "Must be a multiple of --time-chunk (region writes stay chunk-aligned).",
    )
    p.add_argument(
        "--time-chunk",
        type=int,
        default=8,
        help="Zarr chunk length along time. 8 aligns with the ERDM W+1 "
        "rolling-window read (W=6) so a window read touches ~one chunk per "
        "variable; the 12c benchmark showed 64-step chunks force a 64x "
        "over-read per random sample, erasing the coarse store's I/O win.",
    )
    p.add_argument(
        "--smooth-boundaries",
        action="store_true",
        help="Fill boundary NaN with the masked-Gaussian coast fade instead of "
        "a hard constant, BEFORE coarsening. Keeps coastal coarse cells from "
        "being dragged toward the fill value (see module docstring).",
    )
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--smooth-kernel-size", type=int, default=5)
    p.add_argument("--smooth-n-iters", type=int, default=10)
    p.add_argument(
        "--include-extras",
        action="store_true",
        help="Also coarsen the extra_* (not-routed) variables.",
    )
    p.add_argument(
        "--mask-fill",
        type=str,
        default=None,
        help=(
            "JSON object mapping variable name -> fill value for boundary "
            "NaN-fill before coarsening. Defaults to the upstream "
            "ERDM_co2.yaml map (SST->270, sea-ice->0)."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def coarsen_field(x: np.ndarray, factor: int) -> np.ndarray:
    """Bilinearly downsample the trailing (H, W) axes by ``factor``.

    Same mathematical operator as upstream amip_v2
    (``F.interpolate(mode="bilinear", align_corners=False)``); agreement
    is at float32 rounding (the kernel's SIMD accumulation order varies
    in the last ulp with channel folding, and upstream folds whole
    channel groups per frame while this tool works per variable).

    The leading time axis (if any) rides on the BATCH dim — batch is the
    kernel's outer loop, so results are independent of the tool's
    ``--time-block`` size (deterministic re-runs).

    Accepted shapes: ``(H, W)``, ``(T, H, W)``, ``(T, L, H, W)``.
    """
    x = np.ascontiguousarray(x)
    t = torch.from_numpy(x)
    if t.ndim == 2:
        t4 = t[None, None]
    elif t.ndim == 3:
        t4 = t[:, None]
    elif t.ndim == 4:
        t4 = t
    else:
        raise ValueError(f"expected 2-4 dims, got shape {x.shape}")
    with torch.no_grad():
        out = F.interpolate(
            t4, scale_factor=1.0 / factor, mode="bilinear", align_corners=False
        )
    H, W = x.shape[-2:]
    return out.numpy().reshape(*x.shape[:-2], H // factor, W // factor)


def _block_mean_coord(vals: np.ndarray, factor: int) -> np.ndarray:
    return vals.reshape(-1, factor).mean(axis=1).astype(vals.dtype)


def _role_lists(attrs: dict) -> tuple[list[str], list[str], list[str]]:
    """(state_vars, boundary_vars, extra_vars) from the store attrs."""
    state = (
        list(attrs.get("surface_variables", []))
        + list(attrs.get("pressure_upper_air_variables", []))
        + list(attrs.get("sigma_upper_air_variables", []))
        + list(attrs.get("diagnostic_variables", []))
    )
    boundary = list(attrs.get("constant_boundary_variables", [])) + list(
        attrs.get("varying_boundary_variables", [])
    )
    extras = list(attrs.get("extra_surface_variables", [])) + list(
        attrs.get("extra_pressure_upper_air_variables", [])
    )
    return state, boundary, extras


def coarsen_store(
    input_path: Path,
    output_path: Path,
    *,
    factor: int = 4,
    time_block: int = 64,
    time_chunk: int = 8,
    include_extras: bool = False,
    mask_fill: dict[str, float] | None = None,
    smooth_boundaries: bool = False,
    smooth_sigma: float = 1.5,
    smooth_kernel_size: int = 5,
    smooth_n_iters: int = 10,
) -> xr.Dataset:
    """Coarsen one per-year store; returns the (lazily-opened) output."""
    mask_fill = DEFAULT_MASK_FILL if mask_fill is None else dict(mask_fill)
    if time_block % time_chunk:
        raise ValueError(
            f"time_block={time_block} must be a multiple of "
            f"time_chunk={time_chunk} so region writes stay chunk-aligned"
        )

    src = xr.open_zarr(
        input_path,
        decode_times=CFDatetimeCoder(use_cftime=True),
        chunks=None,
    )
    attrs = dict(src.attrs)
    state_vars, boundary_vars, extra_vars = _role_lists(attrs)

    todo = state_vars + boundary_vars + (extra_vars if include_extras else [])
    missing = [v for v in todo if v not in src.data_vars]
    if missing:
        raise KeyError(f"store {input_path} is missing role-listed vars: {missing}")

    n_lat, n_lon = src.sizes["lat"], src.sizes["lon"]
    if n_lat % factor or n_lon % factor:
        raise ValueError(
            f"grid ({n_lat}, {n_lon}) not divisible by factor={factor}"
        )

    coarse_lat = _block_mean_coord(src["lat"].values, factor)
    coarse_lon = _block_mean_coord(src["lon"].values, factor)

    # --- Output skeleton (metadata + empty chunks; region-written below) ---
    coords: dict = {
        "time": src["time"],
        "lat": ("lat", coarse_lat),
        "lon": ("lon", coarse_lon),
    }
    if "pressure_level" in src.coords:
        coords["pressure_level"] = src["pressure_level"]

    n_time = src.sizes.get("time", 0)
    time_chunk = min(time_chunk, max(1, n_time))
    const_names = [v for v in todo if "time" not in src[v].dims]
    timed_names = [v for v in todo if "time" in src[v].dims]

    is_boundary = set(boundary_vars)

    def _prep(name: str, arr: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(arr)
        if not nan_mask.any():
            return arr
        if name in is_boundary or name in extra_vars:
            fill = mask_fill.get(name)
            if fill is None:
                raise ValueError(
                    f"{name!r} contains NaN but has no --mask-fill entry; "
                    f"a NaN would bleed through the bilinear kernel"
                )
            if smooth_boundaries:
                # Masked-Gaussian coast fade (same operator the runtime
                # NanFillTransform uses, so the store and an un-coarsened
                # boundary path agree). One variable per call, leading time
                # axis rides the conv batch dim.
                from physicsnemo.experimental.datapipes.climate.transforms import (
                    _smooth_fill_channel,
                )

                smoothed = _smooth_fill_channel(
                    torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32),
                    torch.from_numpy(nan_mask),
                    float(fill),
                    sigma=smooth_sigma,
                    kernel_size=smooth_kernel_size,
                    n_iters=smooth_n_iters,
                )
                return smoothed.numpy().astype(arr.dtype)
            return np.where(nan_mask, np.asarray(fill, dtype=arr.dtype), arr)
        raise ValueError(
            f"state variable {name!r} contains NaN — the daily-avg state "
            f"contract is NaN-free; refusing to coarsen"
        )

    # ``compute=False`` defers only dask-backed variables, so the skeleton
    # uses dask zeros for the time-varying arrays (metadata-only write) and
    # real eagerly-coarsened values for the tiny constants.
    import dask.array as da

    data_vars: dict = {}
    encoding: dict = {}
    for name in todo:
        dims = src[name].dims  # spatial dims are always the trailing two
        out_shape = tuple(
            (src.sizes[d] // factor) if d in ("lat", "lon") else src.sizes[d]
            for d in dims
        )
        chunks = tuple(
            time_chunk if d == "time" else out_shape[i]
            for i, d in enumerate(dims)
        )
        if name in const_names:
            coarse_const = coarsen_field(_prep(name, src[name].values), factor)
            data_vars[name] = (dims, coarse_const)
            logger.info("constant %s -> %s", name, coarse_const.shape)
        else:
            data_vars[name] = (
                dims,
                da.zeros(out_shape, dtype=src[name].dtype, chunks=chunks),
            )
        encoding[name] = {"chunks": chunks}

    out = xr.Dataset(data_vars, coords=coords)
    out.attrs = {
        **attrs,
        "coarsen_provenance_version": COARSEN_PROVENANCE_VERSION,
        "coarsen_source_store": str(input_path),
        "coarsen_factor": int(factor),
        "coarsen_interpolation": (
            "torch.nn.functional.interpolate(mode='bilinear', "
            "align_corners=False)  # == amip_v2 modules/layers/bilinear.py"
        ),
        "coarsen_mask_fill": json.dumps(mask_fill),
        "coarsen_boundary_fill": "smooth" if smooth_boundaries else "hard",
        "coarsen_smooth_params": json.dumps(
            {
                "sigma": smooth_sigma,
                "kernel_size": smooth_kernel_size,
                "n_iters": smooth_n_iters,
            }
            if smooth_boundaries
            else {}
        ),
        "coarsen_included_extras": bool(include_extras),
        "coarsen_time_chunk": int(time_chunk),
        "coarsen_boundary_note": (
            "boundary channels are NaN-filled (coarsen_mask_fill, see "
            "coarsen_boundary_fill for hard vs smooth) then coarsened to the "
            "state grid. Upstream keeps boundaries at native resolution with "
            "c_grid_downsample=4; pair THIS store with c_grid_downsample=1. "
            "See the 12c/12d seam note in the coarsen_zarr.py docstring."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    out.to_zarr(output_path, compute=False, encoding=encoding)

    # --- Time-varying: block-wise region writes ---
    for start in range(0, n_time, time_block):
        stop = min(start + time_block, n_time)
        block_vars = {}
        for name in timed_names:
            arr = _prep(name, src[name].isel(time=slice(start, stop)).values)
            block_vars[name] = (src[name].dims, coarsen_field(arr, factor))
        xr.Dataset(
            block_vars,
            coords={"time": src["time"].isel(time=slice(start, stop))},
        ).to_zarr(output_path, region={"time": slice(start, stop)})
        logger.info("time block %d:%d / %d written", start, stop, n_time)

    return xr.open_zarr(
        output_path, decode_times=CFDatetimeCoder(use_cftime=True)
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    mask_fill = json.loads(args.mask_fill) if args.mask_fill else None
    coarsen_store(
        args.input,
        args.output,
        factor=args.factor,
        time_block=args.time_block,
        time_chunk=args.time_chunk,
        include_extras=args.include_extras,
        mask_fill=mask_fill,
        smooth_boundaries=args.smooth_boundaries,
        smooth_sigma=args.smooth_sigma,
        smooth_kernel_size=args.smooth_kernel_size,
        smooth_n_iters=args.smooth_n_iters,
    )
    logger.info("coarse store written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())