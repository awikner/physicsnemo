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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OFFLINE: SEEPS climatology (p1, t2) from IMERG training years.

Per calendar month and gridpoint:

* ``p1``  — climatological probability of a dry day
  (precip <= ``--dry-threshold`` mm/day; 0.25 default, WB2 convention);
* ``t2``  — the light/heavy boundary: the 2/3 quantile of *wet-day*
  amounts (light is climatologically twice as likely as heavy). NaN where
  a gridpoint has no wet days (those cells fail the p1 validity window
  anyway);
* ``clim_mean`` — the mean precip (mm/day), used as the anomaly reference
  for the monthly lat-weighted ACC validation metric.

Output: small zarr ``(month, lat, lon)`` with vars ``p1`` / ``t2``,
consumed by ``seeps.SeepsClimatology``. Login-node safe: plain
xarray / zarr / numpy only.

Usage (Derecho)::

    python examples/weather/ai_rossbypalooza/tools/compute_seeps_climatology.py \\
        --imerg-root /glade/derecho/scratch/awikner/physicsnemo-zarr/imerg \\
        --years 2001-2018 \\
        --out /glade/derecho/scratch/awikner/physicsnemo-zarr/normalization/imerg_seeps_climatology.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger("compute_seeps_climatology")

SCRIPT_REL_PATH = (
    "examples/weather/ai_rossbypalooza/tools/compute_seeps_climatology.py"
)


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def month_fields(
    imerg_root: Path,
    years: list[int],
    month: int,
    *,
    var: str,
    dry_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(p1, t2, clim_mean, n_days) for one calendar month across all years."""
    chunks = []
    for year in years:
        store = imerg_root / f"{year}.zarr"
        if not store.exists():
            continue
        ds = xr.open_zarr(store, consolidated=True)
        try:
            da = ds[var].sel(time=ds["time.month"] == month)
            if da.sizes["time"]:
                chunks.append(da.values)
        finally:
            ds.close()
    if not chunks:
        raise ValueError(f"no data for month {month} under {imerg_root}")
    data = np.concatenate(chunks, axis=0)  # (T, H, W)
    finite = np.isfinite(data)
    n_finite = finite.sum(axis=0)
    dry = (data <= dry_threshold) & finite
    with np.errstate(invalid="ignore"):
        p1 = np.where(n_finite > 0, dry.sum(axis=0) / n_finite, np.nan)
    wet = np.where(finite & (data > dry_threshold), data, np.nan)
    # 2/3 quantile of wet-day amounts; all-NaN columns -> NaN (suppressed).
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice")
        t2 = np.nanquantile(wet, 2.0 / 3.0, axis=0)
        clim_mean = np.nanmean(np.where(finite, data, np.nan), axis=0)
    return (
        p1.astype("float32"),
        t2.astype("float32"),
        clim_mean.astype("float32"),
        data.shape[0],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--imerg-root", type=Path, required=True)
    p.add_argument("--years", required=True, help="e.g. 2001-2018")
    p.add_argument("--var", default="total_precipitation_24hr")
    p.add_argument("--dry-threshold", type=float, default=0.25,
                   help="dry-day threshold in mm/day (WB2 convention)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--commit", default="unknown")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    years = parse_years(args.years)
    # Grab the grid from the first available store.
    lat = lon = None
    for year in years:
        store = args.imerg_root / f"{year}.zarr"
        if store.exists():
            with xr.open_zarr(store, consolidated=True) as ds:
                lat = ds["lat"].values.astype("float32")
                lon = ds["lon"].values.astype("float32")
            break
    if lat is None:
        raise FileNotFoundError(f"no yearly stores under {args.imerg_root}")

    p1 = np.empty((12, lat.size, lon.size), dtype="float32")
    t2 = np.empty_like(p1)
    clim_mean = np.empty_like(p1)
    for month in range(1, 13):
        p1[month - 1], t2[month - 1], clim_mean[month - 1], n = month_fields(
            args.imerg_root,
            years,
            month,
            var=args.var,
            dry_threshold=args.dry_threshold,
        )
        logger.info("month %02d: %d days", month, n)

    ds = xr.Dataset(
        {
            "p1": (("month", "lat", "lon"), p1),
            "t2": (("month", "lat", "lon"), t2),
            "clim_mean": (("month", "lat", "lon"), clim_mean),
        },
        coords={
            "month": ("month", np.arange(1, 13, dtype="int32")),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs={
            "schema_version": "1.0",
            "source": str(args.imerg_root),
            "source_years": f"{years[0]}-{years[-1]}",
            "dry_threshold_mm": float(args.dry_threshold),
            "t2_definition": "2/3 quantile of wet-day amounts (light:heavy = 2:1)",
            "generator": f"{SCRIPT_REL_PATH}@{args.commit}",
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.out, mode="w", zarr_format=3, consolidated=True)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
