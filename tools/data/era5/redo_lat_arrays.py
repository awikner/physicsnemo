#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive specific data arrays of an existing ERA5 store from the raw H5 archive,
flipping lat S->N -> N->S, and overwrite them in place.

Used to definitively repair stores left *half*-flipped by an interrupted
flip_lat_zarr.py run: the affected arrays are re-read from the source of truth
(the raw H5, always S->N) and rewritten correctly, regardless of their current
(unknown) flip state. 2D surface/diagnostic vars and 4D pressure-level upper-air
vars are both handled (ndim detected from the store). Sets lat_flipped_to_NtoS.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("redo_lat")


def _level_key(g, prefix: str, level: float, rtol: float = 1e-3) -> str:
    for k in g.keys():
        if k.startswith(prefix + "_"):
            try:
                if abs(float(k[len(prefix) + 1:]) - level) <= rtol * max(1.0, abs(level)):
                    return k
            except ValueError:
                pass
    raise KeyError(f"no H5 key matches {prefix}_<{level}>")


def _read_ts(job):
    path, var, is_upper, levels = job
    with h5py.File(path, "r") as f:
        g = f["input"]
        if is_upper:
            a = np.stack(
                [np.asarray(g[_level_key(g, var, l)][:], dtype="float32") for l in levels],
                axis=0,
            )
            return a[:, ::-1, :]  # (nlev, lat, lon), flip lat
        return np.asarray(g[var][:], dtype="float32")[::-1, :]  # (lat, lon), flip lat


def redo(store_path: str, h5_dir: str, year: int, variables: list[str], n_workers: int) -> None:
    st = zarr.open_group(store_path, mode="r+", use_consolidated=False)
    n = st["2m_temperature"].shape[0]
    levels = [float(x) for x in np.asarray(st["pressure_level"][:])]
    files = [f"{h5_dir}/{year}_{i:04d}.h5" for i in range(n)]
    for var in variables:
        is_upper = st[var].ndim == 4
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, arr in enumerate(
                ex.map(_read_ts, [(fp, var, is_upper, levels) for fp in files], chunksize=4)
            ):
                st[var][i] = arr
        logger.info("%d: re-derived %s (upper=%s)", year, var, is_upper)
    st.attrs["lat_flipped_to_NtoS"] = True
    st.attrs.pop("lat_flipped_arrays", None)
    zarr.consolidate_metadata(st.store)
    logger.info("%d: done (%s)", year, variables)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True)
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--vars", nargs="+", required=True)
    p.add_argument("--n-workers", type=int, default=8)
    args = p.parse_args()
    redo(args.store, args.h5_dir, args.year, args.vars, args.n_workers)


if __name__ == "__main__":
    main()
