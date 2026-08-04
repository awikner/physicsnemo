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

"""OFFLINE: per-gridpoint climatological precip percentiles from IMERG.

For each gridpoint, the requested percentiles (default 75/90/95) of DAILY
precipitation over the season's days (default JJAS) across the training
years — including dry days, matching the project's "climatological
quantiles" target definition. Consumed as FSS exceedance-threshold maps by
``losses.load_precip_quantile_thresholds`` (percentile thresholds remove
frequency bias from the fractions so the FSS term isolates displacement
error; Roberts & Lean 2008).

Output: small zarr ``precip_quantile_mm (quantile, lat, lon)``.
Login-node safe: plain xarray / zarr / numpy only.

Usage (Derecho)::

    python examples/weather/ai_rossbypalooza/tools/compute_precip_quantiles.py \\
        --imerg-root /glade/derecho/scratch/awikner/physicsnemo-zarr/imerg \\
        --years 2000-2019 \\
        --out /glade/derecho/scratch/awikner/physicsnemo-zarr/normalization/imerg_precip_quantiles_jjas.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Bare import when run as a script (tools/ is sys.path[0]); relative when
# imported as tools.compute_precip_quantiles (the test path).
try:
    from compute_seeps_climatology import compact_years, parse_years
except ImportError:  # pragma: no cover - exercised via the package import
    from .compute_seeps_climatology import compact_years, parse_years

logger = logging.getLogger("compute_precip_quantiles")

SCRIPT_REL_PATH = (
    "examples/weather/ai_rossbypalooza/tools/compute_precip_quantiles.py"
)


def seasonal_quantiles(
    imerg_root: Path,
    years: list[int],
    *,
    var: str,
    months: list[int],
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(maps (Q, H, W), lat, lon, n_days) over the season across all years.

    ~2 440 samples per gridpoint for JJAS x 20 years at 1 degree — small
    enough to hold the season in memory in one piece (122 x 180 x 360
    float32 per year ≈ 32 MB).
    """
    chunks = []
    lat = lon = None
    for year in years:
        store = imerg_root / f"{year}.zarr"
        if not store.exists():
            logger.warning("missing %s — skipped", store)
            continue
        ds = xr.open_zarr(store, consolidated=True)
        try:
            if lat is None:
                lat = ds["lat"].values.astype("float32")
                lon = ds["lon"].values.astype("float32")
            da = ds[var].sel(time=ds["time.month"].isin(months))
            if da.sizes["time"]:
                chunks.append(da.values.astype("float32"))
                logger.info("%d: %d season days", year, da.sizes["time"])
        finally:
            ds.close()
    if not chunks:
        raise FileNotFoundError(f"no seasonal data under {imerg_root}")
    data = np.concatenate(chunks, axis=0)  # (T, H, W)
    # Quantiles over ALL season days (dry included), NaN cells excluded.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice")
        maps = np.nanquantile(
            np.where(np.isfinite(data), data, np.nan),
            [q / 100.0 for q in quantiles],
            axis=0,
        )
    return maps.astype("float32"), lat, lon, data.shape[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--imerg-root", type=Path, required=True)
    p.add_argument("--years", required=True, help="e.g. 2000-2019")
    p.add_argument("--var", default="total_precipitation_24hr")
    p.add_argument(
        "--months",
        default="6,7,8,9",
        help="season as comma-separated month numbers (default JJAS)",
    )
    p.add_argument(
        "--quantiles",
        default="75,90,95",
        help="percentiles to compute, comma-separated",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--commit", default="unknown")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    years = parse_years(args.years)
    months = [int(m) for m in args.months.split(",") if m.strip()]
    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip()]
    if sorted(quantiles) != quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError(f"quantiles must be strictly increasing, got {quantiles}")

    maps, lat, lon, n_days = seasonal_quantiles(
        args.imerg_root,
        years,
        var=args.var,
        months=months,
        quantiles=quantiles,
    )
    # Sanity: quantile maps must be monotone in q wherever finite.
    finite = np.isfinite(maps).all(axis=0)
    if not (np.diff(maps[:, finite], axis=0) >= 0).all():
        raise AssertionError("quantile maps are not monotone in q")
    logger.info(
        "%d season days; wet fraction at the lowest percentile map > 0: %.2f",
        n_days,
        float((maps[0] > 0).mean()),
    )

    ds = xr.Dataset(
        {"precip_quantile_mm": (("quantile", "lat", "lon"), maps)},
        coords={
            "quantile": ("quantile", np.asarray(quantiles, dtype="float64")),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs={
            "schema_version": "1.0",
            "source": str(args.imerg_root),
            "source_years": compact_years(years),
            "season_months": ",".join(str(m) for m in months),
            "definition": (
                "per-gridpoint percentiles of daily precip (mm/day) over all "
                "season days, dry days included"
            ),
            "n_season_days": int(n_days),
            "generator": f"{SCRIPT_REL_PATH}@{args.commit}",
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.out, mode="w", zarr_format=3, consolidated=True)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
