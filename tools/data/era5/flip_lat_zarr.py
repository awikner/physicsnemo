#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flip a ClimateZarr store along latitude, in place.

`pangu_h5_to_zarr.py` hardcoded the lat coordinate as N->S (89.5 .. -89.5) but the
raw PanguWeather H5 archive stores data S->N (row 0 = South Pole / Antarctica), so
every field ended up upside-down relative to its lat label. This reverses each data
array along its lat axis so row 0 = North Pole, matching the (correct, untouched)
N->S lat coordinate -- and the SFNO spherical-harmonic-transform grid, which places
row 0 at the North Pole.

Coordinate arrays (lat/lon/time/pressure_level) are left untouched. Idempotent: a
`lat_flipped_to_NtoS` store attr guards against a double flip.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flip_lat")

COORDS = {"lat", "lon", "time", "pressure_level", "init_time", "lead_time"}
BLOCK = 60  # timesteps per read/flip/write block (bounds memory, keeps bulk I/O)


def flip_store(path: str) -> None:
    # use_consolidated=False: we mutate arrays + attrs then regenerate consolidated md.
    g = zarr.open_group(path, mode="r+", use_consolidated=False)
    if g.attrs.get("lat_flipped_to_NtoS"):
        logger.info("%s already flipped (lat_flipped_to_NtoS); skipping", path)
        return
    # Resumable: record each array as it is flipped, so an interrupted run resumes
    # (never double-flips a done array, never leaves a store half-flipped-unrecorded).
    done = set(g.attrs.get("lat_flipped_arrays", []))
    flipped = list(done)
    for name in list(g.array_keys()):
        if name in COORDS or name in done:
            continue
        arr = g[name]
        dims = arr.metadata.dimension_names
        if not dims or "lat" not in dims:
            continue
        lat_axis = list(dims).index("lat")
        if "time" in dims or "init_time" in dims:
            # Reverse in blocks along dim0 to bound memory while keeping bulk I/O
            # (per-timestep single-chunk writes are far too slow on Lustre).
            n = arr.shape[0]
            block = int(BLOCK)
            for s in range(0, n, block):
                e = min(s + block, n)
                arr[s:e] = np.flip(np.asarray(arr[s:e]), axis=lat_axis)
        else:
            arr[:] = np.flip(np.asarray(arr[:]), axis=lat_axis)
        flipped.append(name)
        g.attrs["lat_flipped_arrays"] = sorted(flipped)  # persist progress per array
    g.attrs["lat_flipped_to_NtoS"] = True
    zarr.consolidate_metadata(g.store)
    logger.info("%s: flipped %d arrays along lat: %s", path, len(flipped), flipped)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, help="Path to the .zarr store to flip in place.")
    args = p.parse_args()
    flip_store(args.store)


if __name__ == "__main__":
    main()
