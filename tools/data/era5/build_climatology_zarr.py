#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert ERA5 mean+std daily climatology NetCDFs to one climatology Zarr.

Reads:

* ``1979-2018_mean_climatology.nc`` — daily mean climatology, dims
  ``(time=366, plev=17, lat=180, lon=360)``.
* ``1979-2018_std_climatology.nc`` — same shape, climatological std.

ERA5 has no per-channel bias .npy directory (per user). The output Zarr's bias
arrays are therefore absent (only the climatology `{var}` arrays with the
`(stat, dayofyear, [level,] lat, lon)` shape land).

Reuses the shared
:func:`tools.data._common.climatology_bias.build_climatology_bias_dataset`
with ``bias_groups={}`` and both ``climatology_ds`` (mean) and
``std_climatology_ds`` populated.

Usage::

    python tools/data/era5/build_climatology_zarr.py \\
      --source-dir /work/hdd/bdiu/bgong1/data/h5data \\
      --output /work/nvme/bdiu/awikner/physicsnemo-zarr/era5/climatology_bias.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.climatology_bias import (  # noqa: E402
    build_climatology_bias_dataset,
    write_climatology_bias_zarr,
)


logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ERA5 daily climatology mean+std to a unified Zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/work/hdd/bdiu/bgong1/data/h5data"),
        help="Dir containing the ERA5 climatology mean+std NetCDFs.",
    )
    p.add_argument(
        "--mean-climatology",
        type=str,
        default="1979-2018_mean_climatology.nc",
        help="Filename for the daily mean climatology (relative to --source-dir).",
    )
    p.add_argument(
        "--std-climatology",
        type=str,
        default="1979-2018_std_climatology.nc",
        help="Filename for the daily std climatology (relative to --source-dir).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Zarr store path.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    mean_path = args.source_dir / args.mean_climatology
    std_path = args.source_dir / args.std_climatology
    for p in (mean_path, std_path):
        if not p.exists():
            raise FileNotFoundError(p)

    _cftime_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    mean_clim = xr.open_dataset(mean_path, decode_times=_cftime_decoder)
    std_clim = xr.open_dataset(std_path, decode_times=_cftime_decoder)
    logger.info(
        "loaded mean (%d vars) + std (%d vars) climatologies from %s",
        len(mean_clim.data_vars),
        len(std_clim.data_vars),
        args.source_dir,
    )

    # ERA5 uses `plev` as its pressure-level coord and no sigma. The shared
    # helper handles the missing sigma_dim cleanly.
    out_ds = build_climatology_bias_dataset(
        mean_clim,
        {},
        std_climatology_ds=std_clim,
        sigma_dim="lev",  # not present in ERA5 climatology; helper handles absence
        pressure_dim="plev",
        n_workers=0,
    )
    write_climatology_bias_zarr(
        out_ds,
        args.output,
        source_climatology=mean_path,
        source_bias_dir=None,
        coord_convention="ai_rossby_v1",
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
