#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert ERA5 PanguWeather mean+std NetCDFs to one normalization Zarr.

The ERA5 source dir
(``/work/hdd/bdiu/bgong1/data/h5data/``) ships **per-variant** normalization
files. Each variant has its own naming convention:

* ``pangu_s2s`` (default): ``pangu_s2s_1979-2018_mean.nc`` (upper-air, Z) +
  ``pangu_s2s_1979-2018_surface_mean.nc`` (surface scalars), same suffix for std.
* ``pangu_s2s_withnino``: same plus the Nino34 lags
  (``..._surface_mean_withnino.nc``).
* ``pangu_s2s_log_precip``: log-precip-transformed surface stats
  (``..._surface_mean_log_precip.nc``).

This converter takes a ``--variant`` flag and reads the matching pair of
upper-air + surface files. The output Zarr's name should typically encode the
variant (e.g. ``normalization_pangu_s2s.zarr``) so multiple coexist in one
directory.

ERA5 has no sigma coord — only pressure levels. The shared
:func:`tools.data._common.normalization.build_normalization_dataset` handles
``sigma_coord_name=None`` cleanly.

Usage::

    python tools/data/era5/build_normalization_zarr.py \\
      --source-dir /work/hdd/bdiu/bgong1/data/h5data \\
      --variant pangu_s2s \\
      --output /work/nvme/bdiu/awikner/physicsnemo-zarr/era5/normalization_pangu_s2s.zarr
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


# Variant → (upper_air_mean, upper_air_std, surface_mean, surface_std).
# All paths are relative to ``--source-dir``.
ERA5_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "pangu_s2s": (
        "pangu_s2s_1979-2018_mean.nc",
        "pangu_s2s_1979-2018_std.nc",
        "pangu_s2s_1979-2018_surface_mean.nc",
        "pangu_s2s_1979-2018_surface_std.nc",
    ),
    "pangu_s2s_withnino": (
        "pangu_s2s_1979-2018_mean.nc",
        "pangu_s2s_1979-2018_std.nc",
        "pangu_s2s_1979-2018_surface_mean_withnino.nc",
        "pangu_s2s_1979-2018_surface_std_withnino.nc",
    ),
    "pangu_s2s_log_precip": (
        "pangu_s2s_1979-2018_mean.nc",
        "pangu_s2s_1979-2018_std.nc",
        "pangu_s2s_1979-2018_surface_mean_log_precip.nc",
        "pangu_s2s_1979-2018_surface_std_log_precip.nc",
    ),
    "pangu_s2s_withnino_log_precip": (
        "pangu_s2s_1979-2018_mean.nc",
        "pangu_s2s_1979-2018_std.nc",
        "pangu_s2s_1979-2018_surface_mean_withnino_log_precip.nc",
        "pangu_s2s_1979-2018_surface_std_withnino_log_precip.nc",
    ),
    "pangu_s2s_Z200": (
        # Z-only stats at 200 hPa — special-case for ablation; surface stats
        # are still pangu_s2s_1979-2018_surface_{mean,std}.nc.
        "pangu_s2s_1979-2018_mean_Z200.nc",
        "pangu_s2s_1979-2018_std_Z200.nc",
        "pangu_s2s_1979-2018_surface_mean.nc",
        "pangu_s2s_1979-2018_surface_std.nc",
    ),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ERA5 mean+std NetCDFs to a normalization Zarr (per variant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/work/hdd/bdiu/bgong1/data/h5data"),
        help="Dir containing the per-variant ERA5 mean/std NetCDFs.",
    )
    p.add_argument(
        "--variant",
        choices=sorted(ERA5_VARIANTS),
        default="pangu_s2s",
        help="Pangu_S2S variant naming convention to read.",
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

    upper_mean_n, upper_std_n, surf_mean_n, surf_std_n = ERA5_VARIANTS[args.variant]
    upper_mean_p = args.source_dir / upper_mean_n
    upper_std_p = args.source_dir / upper_std_n
    surf_mean_p = args.source_dir / surf_mean_n
    surf_std_p = args.source_dir / surf_std_n
    for p in (upper_mean_p, upper_std_p, surf_mean_p, surf_std_p):
        if not p.exists():
            raise FileNotFoundError(p)

    _cftime_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    upper_mean = xr.open_dataset(upper_mean_p, decode_times=_cftime_decoder)
    upper_std = xr.open_dataset(upper_std_p, decode_times=_cftime_decoder)
    surf_mean = xr.open_dataset(surf_mean_p, decode_times=_cftime_decoder)
    surf_std = xr.open_dataset(surf_std_p, decode_times=_cftime_decoder)

    # Merge the upper-air + surface var sets per (mean, std). The pangu_s2s
    # convention writes upper-air vars under the `Z` coord and surface vars
    # as scalars in the surface file.
    mean_ds = xr.merge([upper_mean, surf_mean])
    std_ds = xr.merge([upper_std, surf_std])
    logger.info(
        "merged %d upper-air + %d surface vars from variant %s",
        len(upper_mean.data_vars),
        len(surf_mean.data_vars),
        args.variant,
    )

    out_ds = build_normalization_dataset(
        mean_ds,
        std_ds,
        sigma_coord_name=None,  # ERA5 = pressure only
        pressure_coord_name="Z",
    )
    write_normalization_zarr(
        out_ds,
        args.output,
        source_mean=upper_mean_p,
        source_std=upper_std_p,
        coord_convention="ai_rossby_v1",
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
