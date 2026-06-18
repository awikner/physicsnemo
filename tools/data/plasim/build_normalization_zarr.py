#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert PanguWeather PLASIM mean + std NetCDF files to a unified Zarr store.

Reads the PanguWeather v2.0 PLASIM mean / std NetCDF pair (e.g.
``sim52/h5/sigma_data/data_12-132_{mean,std}_sigma.nc``), where pressure-level
vars use the ``Z`` coord (Pa) and sigma-level vars use the ``Z_2`` coord
(unitless 0..1). Emits a single Zarr store with the unified ai-rossby schema:

* ``stat`` coord of length 2 (``mean``, ``std``)
* ``sigma_level`` coord (renamed from ``Z_2``)
* ``pressure_level`` coord (renamed from ``Z``)
* One variable per source channel, shape ``(stat[, level])``

The tendency (delta) std for ``predict_delta`` training is **not** included
here — it depends on the model timedelta and is generated separately by
``compute_delta_stats.py``.

Usage
-----
::

    python tools/data/plasim/build_normalization_zarr.py \\
      --mean /work/nvme/.../data_12-132_mean_sigma.nc \\
      --std  /work/nvme/.../data_12-132_std_sigma.nc \\
      --output /work/nvme/.../plasim/normalization_12-132.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

# tools/data/_common is a package layered next to tools/data/plasim; make it
# importable when this script is invoked directly via `python tools/data/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.normalization import (  # noqa: E402
    build_normalization_dataset,
    write_normalization_zarr,
)


logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PLASIM PanguWeather mean+std NetCDFs to a Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mean",
        type=Path,
        required=True,
        help="Source mean NetCDF (e.g. data_12-132_mean_sigma.nc).",
    )
    p.add_argument(
        "--std",
        type=Path,
        required=True,
        help="Source std NetCDF (e.g. data_12-132_std_sigma.nc).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Zarr store path (created; pass --overwrite to replace).",
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

    mean_ds = xr.open_dataset(args.mean)
    std_ds = xr.open_dataset(args.std)
    logger.info(
        "loaded mean (%d vars) + std (%d vars) from %s + %s",
        len(mean_ds.data_vars),
        len(std_ds.data_vars),
        args.mean,
        args.std,
    )

    out_ds = build_normalization_dataset(
        mean_ds,
        std_ds,
        sigma_coord_name="Z_2",
        pressure_coord_name="Z",
    )
    write_normalization_zarr(
        out_ds,
        args.output,
        source_mean=args.mean,
        source_std=args.std,
        coord_convention="ai_rossby_v1",
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
