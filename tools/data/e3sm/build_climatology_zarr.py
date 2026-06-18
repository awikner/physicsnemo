#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert E3SM climatology.nc + bias .npy dir to one climatology+bias Zarr.

E3SM source layout:

* ``…/sigma_data/climatology.nc`` — daily mean climatology (365 dayofyear
  samples). Uses ``lev`` (18 hybrid pressure levels in hPa) for upper-air vars
  and ``levgrnd`` (5 soil depths, ~7 mm to ~21 cm) for soil vars (``H2OSOI``,
  ``TSOI``).
* ``…/bias/`` — flat directory of bias .npy files (per-variable annual mean +
  4-hour diurnal cycle). 510 files vs PLASIM's 635.

E3SM lacks a separate std climatology (per user, the dataset has no std
file); the output Zarr's ``stat=std`` slot will be NaN-filled.

**Soil-level decomposition** (per user answer #7): E3SM's soil vars
(``H2OSOI``, ``TSOI``) have a ``levgrnd`` dim that doesn't match the unified
``sigma_level`` / ``pressure_level`` schema. We **split each soil var into
per-depth 2D channels** with the depth value baked into the channel name
(e.g. ``TSOI_0.007101``, ``TSOI_0.027925``, …). The resulting per-depth
channels are then treated as surface vars in the unified schema.

Reuses the shared
:func:`tools.data._common.climatology_bias.build_climatology_bias_dataset`
with ``sigma_dim="lev"`` (climatology source uses ``lev`` for pressure-system
upper-air; ai-rossby converter reads ``Z_2`` from the bias filenames as
pressure because all bias level values are ≥ 2). Note: PLASIM uses ``lev`` for
sigma, but E3SM's ``lev`` is hPa pressure values — we route it through the
``sigma_dim`` parameter to use the climatology converter's "promote source
``lev`` to ``sigma_level``" path, then **rename** ``sigma_level`` →
``pressure_level`` post-hoc since the values are pressure semantically.

Usage::

    python tools/data/e3sm/build_climatology_zarr.py \\
      --climatology /work/hdd/bdiu/.../sigma_data/climatology.nc \\
      --bias-dir    /work/hdd/bdiu/.../bias \\
      --output      /work/nvme/bdiu/awikner/physicsnemo-zarr/e3sm/climatology_bias.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.bias import scan_bias_dir  # noqa: E402
from _common.climatology_bias import (  # noqa: E402
    build_climatology_bias_dataset,
    write_climatology_bias_zarr,
)


logger = logging.getLogger(__name__)


def _decompose_soil_vars(
    ds: xr.Dataset, *, soil_dim: str = "levgrnd"
) -> xr.Dataset:
    """Split each soil-level var into separate ``{var}_{depth}`` 2D channels.

    E3SM's soil vars (``H2OSOI``, ``TSOI``) have a ``(time, levgrnd, lat, lon)``
    layout. The unified ai-rossby schema has no ``levgrnd`` coord; per the user
    answer #7 we promote each depth slice to a separate flat channel:
    ``H2OSOI(0.007101) → H2OSOI_0.007101(time, lat, lon)``.
    """
    if soil_dim not in ds.dims:
        return ds
    depths = ds[soil_dim].values
    to_drop = [v for v in ds.data_vars if soil_dim in ds[v].dims]
    new_vars: dict = {}
    for v in to_drop:
        for d in depths:
            new_name = f"{v}_{float(d)}"
            new_vars[new_name] = ds[v].sel({soil_dim: d}, drop=True)
    # drop_dims drops the soil dim AND all coords / data_vars that reference it
    # (including the index coord itself and *_bnds aux vars). Then assign the
    # per-depth flat replacements.
    out = ds.drop_dims(soil_dim)
    return out.assign(new_vars)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert E3SM climatology + bias dir to a unified Zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--climatology",
        type=Path,
        required=True,
        help="Source daily mean climatology NetCDF (e.g. .../sigma_data/climatology.nc).",
    )
    p.add_argument(
        "--bias-dir",
        type=Path,
        default=None,
        help="Optional bias .npy dir. Omit for climatology-only.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=0,
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

    _cftime_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    climatology = xr.open_dataset(args.climatology, decode_times=_cftime_decoder)
    logger.info(
        "loaded climatology %s (%d vars, time=%d, lat=%d, lon=%d)",
        args.climatology,
        len(climatology.data_vars),
        climatology.sizes.get("time", 0),
        climatology.sizes.get("lat", 0),
        climatology.sizes.get("lon", 0),
    )

    # Soil-level decomposition (E3SM-specific): split H2OSOI / TSOI by levgrnd.
    climatology = _decompose_soil_vars(climatology, soil_dim="levgrnd")
    logger.info(
        "post soil decomposition: %d vars (e.g. %s)",
        len(climatology.data_vars),
        [v for v in climatology.data_vars if "TSOI" in v or "H2OSOI" in v][:6],
    )

    if args.bias_dir is not None:
        bias_groups = scan_bias_dir(args.bias_dir)
        logger.info("scanned bias dir %s (%d vars)", args.bias_dir, len(bias_groups))
    else:
        bias_groups = {}

    # E3SM uses `lev` for its hybrid pressure coord; we route it through the
    # shared converter's `pressure_dim`. No sigma in E3SM (sigma_dim absent).
    out_ds = build_climatology_bias_dataset(
        climatology,
        bias_groups,
        std_climatology_ds=None,  # E3SM has no separate std climatology
        sigma_dim="not_present",  # signal the helper there's no sigma source
        pressure_dim="lev",
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
