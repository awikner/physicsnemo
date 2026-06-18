#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert E3SM PanguWeather mean+std NetCDFs to one normalization Zarr.

The E3SM source dir
(``/work/hdd/bdiu/awikner/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data/``)
contains:

* ``data_2015-2050_mean.nc`` — per-variable mean, with multi-level vars
  carrying a ``Z_2`` coord of 18 hybrid pressure levels (hPa).
* ``data_2015-2050_std.nc`` — same shape, std.

The shared
:func:`tools.data._common.normalization.build_normalization_dataset` writes
``Z_2`` as the ai-rossby ``pressure_level`` coord (hPa values preserved per
the user's E3SM convention answer #6).

Usage::

    python tools/data/e3sm/build_normalization_zarr.py \\
      --source-dir /work/hdd/bdiu/awikner/E3SM/.../h5/sigma_data \\
      --output /work/nvme/bdiu/awikner/physicsnemo-zarr/e3sm/normalization_2015-2050.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.normalization import (  # noqa: E402
    build_normalization_dataset,
    write_normalization_zarr,
)


logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert E3SM mean+std NetCDFs to a normalization Zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "/work/hdd/bdiu/awikner/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data"
        ),
    )
    p.add_argument("--mean", type=str, default="data_2015-2050_mean.nc")
    p.add_argument("--std", type=str, default="data_2015-2050_std.nc")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    mean_p = args.source_dir / args.mean
    std_p = args.source_dir / args.std
    for p in (mean_p, std_p):
        if not p.exists():
            raise FileNotFoundError(p)

    _cftime_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    mean_ds = xr.open_dataset(mean_p, decode_times=_cftime_decoder)
    std_ds = xr.open_dataset(std_p, decode_times=_cftime_decoder)
    logger.info("loaded %d vars (mean) + %d vars (std)", len(mean_ds.data_vars), len(std_ds.data_vars))

    # E3SM uses Z_2 for its 18-level hybrid pressure coord (in hPa); no sigma.
    out = build_normalization_dataset(
        mean_ds,
        std_ds,
        sigma_coord_name=None,
        pressure_coord_name="Z_2",
    )
    write_normalization_zarr(
        out,
        args.output,
        source_mean=mean_p,
        source_std=std_p,
        coord_convention="ai_rossby_v1",
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
