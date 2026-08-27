#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Merge ThorCast's split PLASIM stats NetCDFs into one ai-rossby stats store.

ThorCast (``trainer_PLASIM_v4.py`` + ``data/data_loader_multifiles_v4.py``)
spreads normalization across **five** file pairs and computes a sixth group at
runtime. This repo's :class:`ClimateNormalizer` instead wants a *single*
mean/std pair addressed by variable name. This tool bridges the two so a
translated ThorCast checkpoint can be evaluated against the exact statistics it
was trained with.

Sources, and where each lands (names from ``PANGU_NEW_0171_no_soil.yaml``):

===========================  ==========================================
group                        ThorCast source
===========================  ==========================================
surface (``pl``, ``tas``)    ``surface_mean`` / ``surface_std``
upper air (sigma+pressure)   ``upper_air_mean`` / ``upper_air_std``
varying boundary             ``boundary_dir/boundary_mean`` / ``_std``
diagnostic (``pr_6h``)       ``diagnostic_mean`` / ``diagnostic_std``
constant boundary            *no file* — synthesized, see below
===========================  ==========================================

Two subtleties this tool exists to get right
--------------------------------------------

**1. Constant boundary is normalized from the maps themselves.** ThorCast's
``_load_constant_boundary_data`` does::

    data = stack([lsm, z0, sg])            # after NaN -> mask_fill
    data = (data - data.mean(dim=(1, 2))) / data.std(dim=(1, 2))

i.e. each static map is z-scored by its *own spatial* mean/std, with no stats
file involved. This tool recomputes those scalars and writes them as ordinary
entries, so the repo's existing stats-file path reproduces ThorCast bit-for-bit
with no library change and no weight surgery. Two details that would otherwise
introduce a silent bias:

* the NaN fill happens **before** the reduction, in that order;
* ``torch.std`` is **unbiased** (``N-1``), whereas ``numpy``/``xarray`` default
  to the population estimator. We pass ``ddof=1`` to match.

**2. Upper-air pressure levels are subset.** ThorCast's ``zg`` stats live on a
13-level ``Z`` coord but the model consumes only ``Z[3:13]``
(``levels_p = plev[3:13]``, and ``isel(Z=arange(3, 13))`` in ``load_mean_std``)
— i.e. 20000..100000 Pa, which is exactly the ``pressure_level`` coord the
PLASIM Zarr advertises. The subset is applied here so
:class:`ClimateNormalizer`'s match-levels-by-value lookup lines up.

``--pressure-levels full`` disables that subsetting, which is what the
PanguWeather target needs: its own ``data_12-132_*_sigma.nc`` carry all 13 ``Z``
levels and its flat YAML's ``levels:`` list selects the 10 by value at load
time. Pair it with ``--format netcdf`` to emit the mean/std **pair** that
PanguWeather resolves by bare filename against ``data_dir``, rather than the
single ai-rossby Zarr.

Variables present in a source file but not requested (``mrso``, ``mrro``,
``evap``, ``ts``, ``pr_12h``, ``mrso_climatology``) are dropped, so the output
describes exactly one model contract. ``sic`` legitimately carries
``mean=std=1`` upstream; that is preserved verbatim rather than "fixed".

Usage
-----
::

    python tools/data/plasim/build_thorcast_stats.py \
        --stats-dir /work/nvme/bdiu/awikner/thorcast-migration/stats \
        --output    /work/nvme/bdiu/awikner/thorcast-migration/normalization_12-111.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.normalization import (  # noqa: E402
    build_normalization_dataset,
    write_normalization_zarr,
)

logger = logging.getLogger(__name__)

# Defaults mirror config/PANGU_NEW_0171_no_soil.yaml (the config paired with the
# _no_soil_ff checkpoint). Filenames are ThorCast's, relative to --stats-dir.
DEFAULT_FILES = {
    "surface_mean": "data_12-111_sigma_surface_mean.nc",
    "surface_std": "data_12-111_sigma_surface_std.nc",
    "upper_air_mean": "data_12-111_sigma_mean_Z.nc",
    "upper_air_std": "data_12-111_sigma_std_Z.nc",
    "diagnostic_mean": "data_12-111_surface_mean.nc",
    "diagnostic_std": "data_12-111_surface_std.nc",
    "boundary_mean": "boundary_vars/plasim_test_51_150_boundary_mean.nc",
    "boundary_std": "boundary_vars/plasim_test_51_150_boundary_std.nc",
}
DEFAULT_CONSTANT_MAPS = {"lsm": "boundary_vars/lsm.nc",
                         "z0": "boundary_vars/z0.nc",
                         "sg": "boundary_vars/sg.nc"}

DEFAULT_SURFACE = ["pl", "tas"]
DEFAULT_SIGMA_UPPER_AIR = ["ta", "ua", "va", "hus"]
DEFAULT_PRESSURE_UPPER_AIR = ["zg"]
DEFAULT_VARYING_BOUNDARY = ["sst", "rsdt", "sic"]
DEFAULT_DIAGNOSTIC = ["pr_6h"]

# ThorCast keeps plev[3:13] of a 13-level pressure coord (data_loader_multifiles_v4
# line ~154). Expressed as a slice so a differently-sized source errors out loudly.
PRESSURE_SLICE = slice(3, 13)

_CFTIME = xr.coders.CFDatetimeCoder(use_cftime=True)


def _open(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"stats file not found: {path}")
    return xr.open_dataset(path, decode_times=_CFTIME)


def _take_scalars(ds: xr.Dataset, names: list[str], where: str) -> dict:
    """Pull named scalar variables out of a stats dataset."""
    out = {}
    for name in names:
        if name not in ds.data_vars:
            raise KeyError(
                f"{where}: variable {name!r} not found (has {sorted(ds.data_vars)})"
            )
        da = ds[name]
        if da.ndim != 0:
            raise ValueError(f"{where}: expected {name!r} to be scalar, got dims {da.dims}")
        out[name] = xr.DataArray(np.asarray(da.values, dtype="float64"))
    return out


def _take_levels(
    ds: xr.Dataset, names: list[str], dim: str, where: str,
    level_slice: Optional[slice] = None,
) -> dict:
    """Pull named level-resolved variables, optionally subsetting the level axis."""
    out = {}
    for name in names:
        if name not in ds.data_vars:
            raise KeyError(
                f"{where}: variable {name!r} not found (has {sorted(ds.data_vars)})"
            )
        da = ds[name]
        if da.dims != (dim,):
            raise ValueError(
                f"{where}: expected {name!r} on dim ({dim!r},), got {da.dims}"
            )
        if level_slice is not None:
            da = da.isel({dim: level_slice})
        out[name] = da.astype("float64")
    return out


def constant_boundary_scalars(
    stats_dir: Path,
    maps: dict,
    *,
    fill_values: dict,
    default_fill: float,
) -> tuple[dict, dict]:
    """Recompute ThorCast's on-the-fly spatial mean/std for the static maps.

    Reproduces ``_load_constant_boundary_data``: NaN-fill first, then reduce
    over the spatial dims with the **unbiased** (``N-1``) std that ``torch.std``
    uses by default.

    Returns
    -------
    tuple of dict
        ``(means, stds)``, each mapping variable name -> scalar
        :class:`xarray.DataArray`.
    """
    means, stds = {}, {}
    for var, rel in maps.items():
        path = stats_dir / rel
        ds = _open(path)
        if var not in ds.data_vars:
            raise KeyError(f"{path}: variable {var!r} not found (has {sorted(ds.data_vars)})")
        arr = np.asarray(ds[var].values, dtype="float64")
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            fill = float(fill_values.get(var, default_fill))
            arr = np.where(np.isnan(arr), fill, arr)
            logger.info("%s: filled %d NaN with %g before reducing", var, n_nan, fill)
        # ddof=1 to match torch.std's unbiased default.
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            raise ValueError(
                f"{var}: spatial std is {std!r} — normalizing by it would produce "
                f"inf/nan. Check {path}."
            )
        means[var] = xr.DataArray(np.float64(mean))
        stds[var] = xr.DataArray(np.float64(std))
        logger.info("%s: spatial mean=%.8g std=%.8g (n=%d)", var, mean, std, arr.size)
        ds.close()
    return means, stds


def build(
    stats_dir: Path,
    files: dict,
    constant_maps: dict,
    *,
    surface: list[str],
    sigma_upper_air: list[str],
    pressure_upper_air: list[str],
    varying_boundary: list[str],
    diagnostic: list[str],
    fill_values: dict,
    default_fill: float,
    pressure_slice: Optional[slice] = PRESSURE_SLICE,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Assemble the merged ``(mean_ds, std_ds)`` pair.

    ``pressure_slice=None`` keeps every level of the source ``Z`` coord (the
    PanguWeather convention); the default keeps ThorCast's ``Z[3:13]`` subset.
    """
    mean_vars: dict = {}
    std_vars: dict = {}

    sfc_m, sfc_s = _open(stats_dir / files["surface_mean"]), _open(stats_dir / files["surface_std"])
    mean_vars |= _take_scalars(sfc_m, surface, "surface_mean")
    std_vars |= _take_scalars(sfc_s, surface, "surface_std")

    dia_m, dia_s = _open(stats_dir / files["diagnostic_mean"]), _open(stats_dir / files["diagnostic_std"])
    mean_vars |= _take_scalars(dia_m, diagnostic, "diagnostic_mean")
    std_vars |= _take_scalars(dia_s, diagnostic, "diagnostic_std")

    bnd_m, bnd_s = _open(stats_dir / files["boundary_mean"]), _open(stats_dir / files["boundary_std"])
    mean_vars |= _take_scalars(bnd_m, varying_boundary, "boundary_mean")
    std_vars |= _take_scalars(bnd_s, varying_boundary, "boundary_std")

    ua_m, ua_s = _open(stats_dir / files["upper_air_mean"]), _open(stats_dir / files["upper_air_std"])
    mean_vars |= _take_levels(ua_m, sigma_upper_air, "Z_2", "upper_air_mean")
    std_vars |= _take_levels(ua_s, sigma_upper_air, "Z_2", "upper_air_std")
    mean_vars |= _take_levels(
        ua_m, pressure_upper_air, "Z", "upper_air_mean", level_slice=pressure_slice
    )
    std_vars |= _take_levels(
        ua_s, pressure_upper_air, "Z", "upper_air_std", level_slice=pressure_slice
    )

    const_m, const_s = constant_boundary_scalars(
        stats_dir, constant_maps, fill_values=fill_values, default_fill=default_fill
    )
    mean_vars |= const_m
    std_vars |= const_s

    mean_ds = xr.Dataset(mean_vars)
    std_ds = xr.Dataset(std_vars)
    for ds in (mean_ds, std_ds):
        # build_normalization_dataset reads the level coords off the mean side.
        if "Z_2" in ds.dims and "Z_2" not in ds.coords:
            ds.coords["Z_2"] = np.asarray(ua_m["Z_2"].values, dtype="float64")
        if "Z" in ds.dims and "Z" not in ds.coords:
            z = np.asarray(ua_m["Z"].values, dtype="float64")
            ds.coords["Z"] = z if pressure_slice is None else z[pressure_slice]
    return mean_ds, std_ds


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge ThorCast's split PLASIM stats into one ai-rossby store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stats-dir", type=Path, required=True,
                   help="Directory holding ThorCast's stats NetCDFs (and boundary_vars/).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output path. For --format zarr this is the store; for "
                        "--format netcdf it is a PREFIX and two files are written: "
                        "<prefix>_mean_sigma.nc and <prefix>_std_sigma.nc.")
    p.add_argument(
        "--format",
        choices=("zarr", "netcdf"),
        default="zarr",
        help="zarr = the ai-rossby unified store (single file, `stat` coord), "
             "consumed by ClimateNormalizer. netcdf = the PanguWeather v2.0 "
             "layout: a separate mean/std pair whose bare filenames go in the "
             "flat YAML's surface_mean/upper_air_mean/... keys.",
    )
    p.add_argument(
        "--pressure-levels",
        choices=("subset", "full"),
        default="subset",
        help="subset = keep only Z[3:13] (20000..100000 Pa), matching what the "
             "model consumes and what the PLASIM Zarr's `pressure_level` coord "
             "advertises — required for the ai-rossby store, whose normalizer "
             "matches levels by value against the data. full = keep all 13 "
             "levels, which is what PanguWeather's own stats files do (its "
             "config's `levels:` list selects the 10 by value at load time).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing output.")
    p.add_argument("--nan-fill-default", type=float, default=0.0,
                   help="Fill for NaN in the static maps when no per-var value is given.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    pressure_slice = PRESSURE_SLICE if args.pressure_levels == "subset" else None
    if args.format == "netcdf" and args.pressure_levels == "subset":
        logger.warning(
            "--format netcdf with --pressure-levels subset: PanguWeather's own "
            "stats files carry all 13 Z levels and select by value from the "
            "config's `levels:` list. Pass --pressure-levels full to match."
        )

    mean_ds, std_ds = build(
        args.stats_dir,
        DEFAULT_FILES,
        DEFAULT_CONSTANT_MAPS,
        surface=DEFAULT_SURFACE,
        sigma_upper_air=DEFAULT_SIGMA_UPPER_AIR,
        pressure_upper_air=DEFAULT_PRESSURE_UPPER_AIR,
        varying_boundary=DEFAULT_VARYING_BOUNDARY,
        diagnostic=DEFAULT_DIAGNOSTIC,
        fill_values={},
        default_fill=args.nan_fill_default,
        pressure_slice=pressure_slice,
    )
    logger.info(
        "merged %d variables (pressure levels: %s)",
        len(mean_ds.data_vars),
        "Z[3:13]" if pressure_slice else "all",
    )

    if args.format == "netcdf":
        # PanguWeather resolves the bare filenames in its flat YAML against
        # data_dir, and reads mean/std from separate files.
        mean_path = Path(f"{args.output}_mean_sigma.nc")
        std_path = Path(f"{args.output}_std_sigma.nc")
        for path in (mean_path, std_path):
            if path.exists() and not args.overwrite:
                logger.error("%s exists; pass --overwrite to replace", path)
                return 1
        mean_path.parent.mkdir(parents=True, exist_ok=True)
        # float32 to match the upstream files byte-for-byte in dtype.
        mean_ds.astype("float32").to_netcdf(mean_path)
        std_ds.astype("float32").to_netcdf(std_path)
        logger.info("wrote %s", mean_path)
        logger.info("wrote %s", std_path)
        return 0

    out_ds = build_normalization_dataset(
        mean_ds, std_ds, sigma_coord_name="Z_2", pressure_coord_name="Z"
    )
    write_normalization_zarr(
        out_ds,
        args.output,
        source_mean=args.stats_dir / DEFAULT_FILES["upper_air_mean"],
        source_std=args.stats_dir / DEFAULT_FILES["upper_air_std"],
        overwrite=args.overwrite,
    )
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
