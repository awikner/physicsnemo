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

"""OFFLINE: IMERG precipitation normalization stats (shared mean/std).

Computes the global mean/std of ``total_precipitation_24hr`` over the IMERG
training years (finite cells only, optional month filter) and writes a tiny
combined stats zarr (``stat`` coord {mean, std}) consumed by
``datapipes/stats.py`` for every expert's precip channel AND the target.

Login-node safe: plain xarray / zarr / numpy only.

Usage (Derecho)::

    python examples/weather/ai_rossbypalooza/tools/compute_precip_norm.py \\
        --imerg-root /glade/derecho/scratch/awikner/physicsnemo-zarr/imerg \\
        --years 2001-2018 \\
        --out /glade/derecho/scratch/awikner/physicsnemo-zarr/normalization/imerg_precip_stats.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

logger = logging.getLogger("compute_precip_norm")

SCRIPT_REL_PATH = "examples/weather/ai_rossbypalooza/tools/compute_precip_norm.py"


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


def compact_years(years: list[int]) -> str:
    """"2000-2004,2010-2024" for a possibly non-contiguous year list.

    A plain first-to-last span would report a k-fold training set that holds
    out a middle block as the full range, i.e. provenance metadata claiming
    the held-out years were included.
    """
    runs: list[tuple[int, int]] = []
    for y in sorted(set(years)):
        if runs and y == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], y)
        else:
            runs.append((y, y))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)

def compute_stats(
    imerg_root: Path,
    years: list[int],
    *,
    var: str = "total_precipitation_24hr",
    months: list[int] | None = None,
    log_epsilon: float | None = None,
    log_units: str = "m",
) -> tuple[float, float, int]:
    """Streaming (count, sum, sumsq) over the yearly stores; finite cells only.

    With ``log_epsilon`` set, statistics are computed in the model-v1
    transformed space ``log(epsilon + P[log_units])`` (IMERG values are
    mm/day; ``log_units="m"`` divides by 1000 first).
    """
    transform = None
    if log_epsilon is not None:
        from datapipes.precip import LogPrecipTransform

        transform = LogPrecipTransform(epsilon=log_epsilon, units=log_units)
    count = 0
    total = 0.0
    total_sq = 0.0
    for year in years:
        store = imerg_root / f"{year}.zarr"
        if not store.exists():
            logger.warning("missing store %s (skipped)", store)
            continue
        ds = xr.open_zarr(store, consolidated=True)
        try:
            da = ds[var]
            if months:
                da = da.sel(time=da["time.month"].isin(months))
            vals = da.values
        finally:
            ds.close()
        finite = np.isfinite(vals)
        v = vals[finite].astype(np.float64)
        if transform is not None:
            v = transform.forward(np.clip(v, 0.0, None))
        count += v.size
        total += float(v.sum())
        total_sq += float((v**2).sum())
        logger.info("%s: %d finite cells", store.name, v.size)
    if count == 0:
        raise ValueError(f"no finite {var} data found under {imerg_root}")
    mean = total / count
    var_ = max(total_sq / count - mean**2, 0.0)
    return mean, float(np.sqrt(var_)), count


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--imerg-root", type=Path, required=True)
    p.add_argument("--years", required=True, help="e.g. 2001-2018")
    p.add_argument("--months", default=None,
                   help="optional month filter, e.g. 5,6,7,8,9 (default: all)")
    p.add_argument("--var", default="total_precipitation_24hr")
    p.add_argument("--log-epsilon", type=float, default=None,
                   help="model v1: compute stats in log(epsilon + P) space")
    p.add_argument("--log-units", default="m", choices=("m", "mm"),
                   help="units the log offset applies in (default m)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--commit", default="unknown")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    years = parse_years(args.years)
    months = (
        [int(m) for m in args.months.split(",")] if args.months else None
    )
    mean, std, count = compute_stats(
        args.imerg_root,
        years,
        var=args.var,
        months=months,
        log_epsilon=args.log_epsilon,
        log_units=args.log_units,
    )
    if args.log_epsilon is not None:
        units = f"log({args.log_epsilon:g} + P[{args.log_units}/24h])"
    else:
        units = "mm/day"
    logger.info("mean=%.6f std=%.6f (%s) n=%d", mean, std, units, count)

    attrs = {
        "schema_version": "1.0",
        "source": str(args.imerg_root),
        "source_years": compact_years(years),
        "source_months": str(months) if months else "all",
        "n_samples": count,
        "units": units,
        "generator": f"{SCRIPT_REL_PATH}@{args.commit}",
    }
    if args.log_epsilon is not None:
        attrs["transform"] = "log"
        attrs["log_epsilon"] = float(args.log_epsilon)
        attrs["log_units"] = args.log_units
    ds = xr.Dataset(
        {args.var: (("stat",), np.array([mean, std], dtype="float64"))},
        coords={"stat": ("stat", np.array(["mean", "std"]))},
        attrs=attrs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.out, mode="w", zarr_format=3, consolidated=True)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
