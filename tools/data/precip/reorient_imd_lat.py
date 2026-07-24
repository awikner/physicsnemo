#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reorient IMD yearly stores from S->N to N->S IN PLACE.

The IMD gauge-analysis zarr stores were originally written on IMD's native
S->N grid (lat 6.5 -> 38.5). They were internally CONSISTENT (data matched the
store's own lat coord), but every other product in the project (ERA5, Pangu
hindcasts, IMERG) is N->S. This script flips BOTH the ``lat`` coordinate AND
every lat-dimensioned data array together, so the store adopts the project
convention while staying internally consistent.

Contrast with ``tools/data/era5/flip_lat_zarr.py``, which flips ONLY the data:
that tool is for stores whose lat coord was already (correctly) N->S but whose
data rows were upside-down. Applying that tool here would have BROKEN IMD.

Crash-safe / idempotent: each data array is recorded in the
``reorient_data_flipped`` attr as it is flipped, the coord flip is guarded by
``reorient_coord_flipped``, and the whole store is finalized with
``lat_reoriented_to_NtoS``. An interrupted run resumes without double-flipping.

Applied 2026-07-24 to all 125 ``imd/*.zarr`` (1901-2025) on Stampede3;
verified value-aligned corr vs corrected ERA5 unchanged (as-is +0.58).
The converter (``tools/data/precip/h5_to_zarr.py``) now reorients IMD on
ingest, so re-converted stores need no post-hoc fix.

Usage::

    python tools/data/precip/reorient_imd_lat.py --root /path/to/imd
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reorient_imd")

GUARD = "lat_reoriented_to_NtoS"
COORDS = {"lat", "lon", "time"}


def reorient(path: str) -> None:
    base = os.path.basename(path)
    g = zarr.open_group(path, mode="r+", use_consolidated=False)
    if g.attrs.get(GUARD):
        la = np.asarray(g["lat"][:])
        logger.info("%s: already reoriented (lat %.1f..%.1f); skip", base, la[0], la[-1])
        return
    # 1) Flip each lat-dimensioned DATA array (per-array manifest -> resumable).
    done = set(g.attrs.get("reorient_data_flipped", []))
    for name in list(g.array_keys()):
        if name in COORDS or name in done:
            continue
        arr = g[name]
        dims = arr.metadata.dimension_names
        if dims and "lat" in dims:
            ax = list(dims).index("lat")
        elif arr.ndim == 3 and arr.shape[1] == g["lat"].shape[0]:
            ax = 1  # (time, lat, lon) fallback if dimension_names missing
        else:
            continue
        arr[:] = np.flip(np.asarray(arr[:]), axis=ax)
        done.add(name)
        g.attrs["reorient_data_flipped"] = sorted(done)
    # 2) Flip the lat COORD (separate flag).
    if not g.attrs.get("reorient_coord_flipped"):
        g["lat"][:] = np.asarray(g["lat"][:])[::-1]
        g.attrs["reorient_coord_flipped"] = True
    # 3) Finalize.
    g.attrs[GUARD] = True
    zarr.consolidate_metadata(g.store)
    ng = zarr.open_group(path, mode="r", use_consolidated=True)
    la = np.asarray(ng["lat"][:])
    logger.info(
        "%s: lat %.1f..%.1f (%s) data_flipped=%s",
        base, la[0], la[-1],
        "N->S" if la[0] > la[-1] else "S->N",
        ng.attrs.get("reorient_data_flipped"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Directory of IMD *.zarr yearly stores.")
    args = p.parse_args()
    stores = sorted(glob.glob(f"{args.root}/*.zarr"))
    logger.info("found %d IMD stores", len(stores))
    for s in stores:
        reorient(s)
    logger.info("DONE")


if __name__ == "__main__":
    main()
