#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair the AMIP-family lat label: rewrite the COORDINATE to ascending, in place.

``amip_h5_to_zarr.py`` used to declare ``lat`` as N->S (89.5 .. -89.5) while the raw
AMIP / ERA5-daily-average archives store rows S->N (row 0 = South Pole) and the
converter copied them verbatim. Every store it wrote therefore has correct data under
a reversed label. This tool fixes the label and touches nothing else.

WHY RELABEL RATHER THAN FLIP THE DATA
    ``ClimateZarrDataset`` reads array order verbatim and never consults the lat
    VALUES, so relabelling leaves the read path at zero per-sample transformation --
    the optimum for data loading. Reversing ~2.7 TB of data instead buys no loading
    speed, and it would break bit-exact agreement with upstream amip_v2 and with
    every existing checkpoint (all trained on exactly these S->N bytes), and would
    force ``sst_climatology.npz`` to be refitted. E3SM already ships correct S->N
    labels, so this is not a new convention for the project.

    Contrast ``tools/data/era5/flip_lat_zarr.py``, which reverses only the DATA: that
    is for stores whose lat coord was already correct and whose rows were upside-down.
    The two tools are inverses of each other -- applying the wrong one doubles the bug.

Safety: idempotent and crash-safe. A store is skipped unless its data really is
upside-down relative to its label, which is re-verified from physical anchors here
rather than taken on trust, so pointing this at an already-correct store does
nothing. ``lat_relabelled_to_ascending`` guards against a second pass.

Usage::

    relabel_lat_ascending.py --stores "$AI_ROSSBY_DATA/amip_dailyavg/*.zarr"
    relabel_lat_ascending.py --stores "..." --dry-run      # report only
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _common.lat_orientation import (  # noqa: E402
    INSOL_KEYS,
    LAND_KEYS,
    OROG_KEYS,
    SOUTH_FIRST,
    TEMP_KEYS,
    combine,
    order_of_coord,
    vote_insolation,
    vote_land_mask,
    vote_orography,
    vote_temperature,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("relabel_lat")

GUARD = "lat_relabelled_to_ascending"


def _month_of(store: str, idx: int) -> int | None:
    """Calendar month of a timestep, needed to unlock the seasonal anchors."""
    try:
        import xarray as xr

        ds = xr.open_zarr(store, consolidated=None, decode_timedelta=False)
        t = np.asarray(ds["time"].values)[idx]
        return t.month if hasattr(t, "month") else int(str(t)[5:7])
    except Exception:
        return None


def data_row_order(g, lat: np.ndarray, store: str) -> tuple[str | None, dict[str, str]]:
    """Row order of the DATA, from physical anchors (label-independent)."""
    names = set(g.array_keys())
    votes: dict[str, str | None] = {}

    def plane(name: str, tidx: int | None) -> np.ndarray | None:
        a = g[name]
        try:
            if a.ndim == 2:
                return np.asarray(a[:], dtype="float64")
            if tidx is None:
                return None
            if a.ndim == 3:
                return np.asarray(a[tidx], dtype="float64")
            if a.ndim == 4:
                return np.asarray(a[tidx, 0], dtype="float64")
        except Exception:
            return None
        return None

    for keys, fn, label in ((LAND_KEYS, vote_land_mask, "land_mask"),
                            (OROG_KEYS, vote_orography, "orography")):
        k = next((n for n in keys if n in names), None)
        if k is None:
            continue
        f = plane(k, 0 if g[k].ndim >= 3 else None)
        if f is not None and f.shape[0] == lat.size:
            votes[f"{label}:{k}"] = fn(f)

    n_time = next((int(g[n].shape[0]) for n in names
                   if g[n].ndim >= 3 and n not in ("lat", "lon", "time")), 0)
    for keys, fn, label in ((INSOL_KEYS, vote_insolation, "insolation"),
                            (TEMP_KEYS, vote_temperature, "temperature")):
        k = next((n for n in keys if n in names and g[n].ndim >= 3), None)
        if k is None or not n_time:
            continue
        # Spread the seasonal samples across the year: a single day can sit inside
        # a warm-air intrusion, and index 0 is mid-winter where the contrast is
        # smallest. The vote itself abstains when the contrast is within noise.
        for idx in {0, n_time // 4, n_time // 2, (3 * n_time) // 4}:
            mo = _month_of(store, idx)
            if mo is None:
                continue
            f = plane(k, idx)
            if f is not None and f.shape[0] == lat.size:
                v = fn(f, mo)
                if v is not None:
                    votes[f"{label}:{k}@{idx}"] = v
    return combine(votes)


def relabel(path: str, dry_run: bool = False) -> str:
    base = os.path.basename(path)
    g = zarr.open_group(path, mode="r" if dry_run else "r+", use_consolidated=False)
    lat = np.asarray(g["lat"][:], dtype="float64")
    claimed = order_of_coord(lat)
    if g.attrs.get(GUARD):
        return f"{base}: already relabelled (lat {lat[0]:.1f}..{lat[-1]:.1f}); skip"

    order, votes = data_row_order(g, lat, path)
    if order is None:
        return f"{base}: SKIP — could not establish the data's row order (votes={votes})"
    if order == claimed:
        return (f"{base}: OK already — data and label both {claimed} "
                f"(votes={votes}); nothing to do")
    if order != SOUTH_FIRST:
        return (f"{base}: SKIP — data is {order} under a {claimed} label; this tool only "
                f"relabels S->N data (use tools/data/era5/flip_lat_zarr.py instead)")

    if dry_run:
        return (f"{base}: WOULD relabel lat {lat[0]:.1f}..{lat[-1]:.1f} ({claimed}) -> "
                f"{lat[-1]:.1f}..{lat[0]:.1f} (S->N), data untouched (votes={votes})")

    g["lat"][:] = lat[::-1].astype(g["lat"].dtype)
    g.attrs["lat_row_order"] = SOUTH_FIRST
    g.attrs[GUARD] = True
    g.attrs["lat_relabel_note"] = (
        "The lat coordinate was reversed to ascending (S->N) to match the data rows, "
        "which the converter copied verbatim from an S->N raw archive. DATA BYTES ARE "
        "UNCHANGED and remain bit-identical to upstream amip_v2. Previously this store "
        "declared N->S, which mislabelled every field. See "
        "docs/dev/context/lat-orientation-audit.md."
    )
    zarr.consolidate_metadata(g.store)
    new = np.asarray(zarr.open_group(path, mode="r")["lat"][:])
    return (f"{base}: relabelled -> lat {new[0]:.1f}..{new[-1]:.1f} "
            f"({order_of_coord(new)}), data untouched (votes={votes})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stores", nargs="+", required=True, help="store paths and/or globs")
    p.add_argument("--dry-run", action="store_true", help="report, change nothing")
    a = p.parse_args(argv)

    stores: list[str] = []
    for s in a.stores:
        stores.extend(sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s])
    stores = [s for s in stores if os.path.isdir(s)]
    if not stores:
        logger.error("no stores matched")
        return 2
    logger.info("%d store(s)%s", len(stores), " [dry run]" if a.dry_run else "")
    changed = skipped = failed = 0
    for s in stores:
        try:
            msg = relabel(s, a.dry_run)
        except Exception as e:  # noqa: BLE001
            failed += 1
            logger.error("%s: %s: %s", os.path.basename(s), type(e).__name__, e)
            continue
        logger.info("%s", msg)
        if "relabelled ->" in msg or "WOULD relabel" in msg:
            changed += 1
        else:
            skipped += 1
    logger.info("done: %d relabelled, %d skipped, %d failed", changed, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
