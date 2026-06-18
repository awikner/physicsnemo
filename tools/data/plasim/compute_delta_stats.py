#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a delta-std NetCDF for a PLASIM Zarr store.

For each prognostic variable in the store, compute the std of one-step
tendencies (``state_{t+1} - state_t``) across the time axis, then write a
NetCDF file matching the existing ``*_std_sigma.nc`` layout so it drops in
to :class:`PlasimNormalizer` when ``predict_delta=True``.

Usage
-----
::

    python tools/data/plasim/compute_delta_stats.py \\
      --zarr $AI_ROSSBY_TEST_DATA/plasim/smoke_month_t1.zarr \\
      --stride 1 \\
      --output $AI_ROSSBY_TEST_DATA/plasim/smoke_month_delta_std.nc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--zarr", type=Path, required=True, help="Source PLASIM Zarr store.")
    p.add_argument("--output", type=Path, required=True, help="Output NetCDF path.")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Tendency stride (1 = one-step). Use the model's training lead time.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def compute_delta_std(ds: xr.Dataset, stride: int) -> xr.Dataset:
    """For every time-dimensioned variable, compute std of ``x[t+s] - x[t]``.

    Important: subtract raw numpy values (not aligned DataArrays) so xarray's
    coordinate-alignment doesn't silently null the difference by matching
    timestamps between the two slices.
    """
    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue
        a = da.isel(time=slice(stride, None)).values
        b = da.isel(time=slice(None, -stride)).values
        diff = a - b
        # Reduce time/lat/lon — preserve level axis for upper-air vars.
        # diff dims match da's dims order (time, [level,] lat, lon).
        # We want std over time + lat + lon → preserve only level dim if present.
        non_time_dims = [d for d in da.dims if d != "time"]
        level_dims = [d for d in non_time_dims if d not in ("lat", "lon")]
        # diff shape order matches the dim order; compute std over reduce-axis idxs.
        reduce_axes = tuple(
            i
            for i, d in enumerate(da.dims)
            if d not in level_dims
        )
        std_vals = diff.std(axis=reduce_axes).astype("float32")
        if level_dims:
            out_vars[name] = xr.DataArray(std_vals, dims=tuple(level_dims))
        else:
            out_vars[name] = xr.DataArray(std_vals)
    coords = {}
    if "pressure_level" in ds.coords:
        coords["Z"] = ("Z", ds["pressure_level"].values.astype("float32"))
    if "sigma_level" in ds.coords:
        coords["Z_2"] = ("Z_2", ds["sigma_level"].values.astype("float32"))
    renamed: dict[str, xr.DataArray] = {}
    for name, arr in out_vars.items():
        if "pressure_level" in arr.dims:
            arr = arr.rename({"pressure_level": "Z"})
        if "sigma_level" in arr.dims:
            arr = arr.rename({"sigma_level": "Z_2"})
        renamed[name] = arr
    return xr.Dataset(renamed, coords=coords)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Opening source zarr %s", args.zarr)
    # cftime everywhere — uniform time coord across PLASIM (year 1, cftime
    # already) and ERA5/E3SM (post-1582, would otherwise decode to datetime64).
    ds = xr.open_zarr(
        args.zarr,
        consolidated=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    logger.info("Computing delta std (stride=%d) over %d timesteps", args.stride, ds.sizes["time"])
    delta = compute_delta_std(ds, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    delta.to_netcdf(args.output)
    logger.info("Wrote %s — %d variables", args.output, len(delta.data_vars))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
