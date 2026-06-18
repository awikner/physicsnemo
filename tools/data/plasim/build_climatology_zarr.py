#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert PLASIM climatology.nc + bias .npy directory to a unified Zarr store.

Reads:

* The PanguWeather PLASIM daily climatology NetCDF (e.g.
  ``sim52/sigma_data/climatology.nc``), which carries ``time`` (366 daily
  samples), ``lev`` (sigma), ``plev`` (pressure), ``lat``, ``lon`` and one data
  variable per channel.
* The flat directory of bias .npy files (e.g. ``sim52/bias/``), with the naming
  convention ``{var}[_{level}]_bias[_{H}z].npy`` (see
  :mod:`tools.data._common.bias`).

Emits one Zarr store under the unified ai-rossby schema. Variables present in
both the climatology and bias dir get all three of:

* ``{var}`` — daily climatology, dims ``(dayofyear, [level,] lat, lon)``.
* ``{var}_bias_annual`` — annual-mean bias, dims ``([level,] lat, lon)``.
* ``{var}_bias_diurnal`` — diurnal-cycle bias, dims ``(hour_of_day, [level,] lat, lon)``.

Variables present in only one source are still included; the missing arrays are
NaN-filled (the union convention requested in the Phase 3 follow-up planning).

The bias-file reads are parallelized via a process pool (sized via
``SLURM_CPUS_PER_TASK`` or ``--n-workers``).

Usage
-----
::

    python tools/data/plasim/build_climatology_zarr.py \\
      --climatology /work/.../sim52/sigma_data/climatology.nc \\
      --bias-dir    /work/.../sim52/bias \\
      --output      /work/.../plasim/climatology_bias.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.bias import scan_bias_dir  # noqa: E402
from _common.climatology_bias import (  # noqa: E402
    build_climatology_bias_dataset,
    write_climatology_bias_zarr,
)


logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PLASIM climatology + bias .npy dir to a unified Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--climatology",
        type=Path,
        required=True,
        help="Source daily climatology NetCDF (e.g. sim52/sigma_data/climatology.nc).",
    )
    p.add_argument(
        "--bias-dir",
        type=Path,
        default=None,
        help="Directory of bias .npy files. Omit for a climatology-only store.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Zarr path (created; pass --overwrite to replace).",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Process-pool size for parallel bias-.npy loading. "
        "0 autodetects via SLURM_CPUS_PER_TASK (fallback os.cpu_count()).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output store if it already exists.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging at DEBUG level.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    climatology = xr.open_dataset(args.climatology, use_cftime=True)
    logger.info(
        "loaded climatology %s (%d vars, dayofyear=%d, lat=%d, lon=%d)",
        args.climatology,
        len(climatology.data_vars),
        climatology.sizes.get("time", 0),
        climatology.sizes.get("lat", 0),
        climatology.sizes.get("lon", 0),
    )

    if args.bias_dir is not None:
        bias_groups = scan_bias_dir(args.bias_dir)
        logger.info(
            "scanned bias dir %s (%d variables, e.g. %s)",
            args.bias_dir,
            len(bias_groups),
            sorted(bias_groups)[:5],
        )
    else:
        bias_groups = {}
        logger.info("no bias dir supplied — climatology-only store.")

    out_ds = build_climatology_bias_dataset(
        climatology,
        bias_groups,
        sigma_dim="lev",
        pressure_dim="plev",
        n_workers=args.n_workers,
    )
    write_climatology_bias_zarr(
        out_ds,
        args.output,
        source_climatology=args.climatology,
        source_bias_dir=args.bias_dir,
        coord_convention="ai_rossby_v1",
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
