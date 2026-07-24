#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Augment an existing ``era5/{year}.zarr`` store (built by an earlier reduced
``pangu_h5_to_zarr.py`` run that wrote only 4 prognostic surface vars) with the
full Pangu-Weather S2S feature set, so SFNO/ArchesWeather train on the same
variables the Pangu-S2S hindcasts carry.

Adds, per store, these arrays read index-aligned from the raw Pangu v2.0 H5
archive (``{year}_{idx:04d}.h5``, ``input/<var>``) — the archive is confirmed
time-index-identical to the store (same n_time, same 2m_temperature values):

  surface (new prognostic):   surface_pressure, skin_temperature,
                              soil_temperature_level_1, volumetric_soil_water_layer_1
  diagnostic (derived):       mean_top_net_long_wave_radiation_flux = -ULWRFtoa_24h

and reclassifies ``sea_surface_temperature`` from *varying boundary* to
*prognostic surface* (its array already exists in the store — attrs-only change).
MSLP and every other existing array/attr are left untouched. Idempotent: re-runs
overwrite the added arrays and rewrite the three group attrs.

Usage (one year):
    python augment_era5_full_surface.py --h5-dir /scratch/.../ERA5/h5 \\
        --store /scratch/.../physicsnemo-zarr/era5/2000.zarr --year 2000 --n-workers 32
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("augment_era5")

# store array name -> (raw H5 input/<key>, affine scale). offset assumed 0.
ADD_VARS: dict[str, tuple[str, float]] = {
    "surface_pressure": ("surface_pressure", 1.0),
    "skin_temperature": ("skin_temperature", 1.0),
    "soil_temperature_level_1": ("soil_temperature_level_1", 1.0),
    "volumetric_soil_water_layer_1": ("volumetric_soil_water_layer_1", 1.0),
    # ERA5 net TOA longwave (negative = outgoing) = -ULWRFtoa_24h (upward TOA LW).
    "mean_top_net_long_wave_radiation_flux": ("ULWRFtoa_24h", -1.0),
}
NEW_SURFACE = [
    "surface_pressure",
    "sea_surface_temperature",  # reclassified from varying boundary
    "skin_temperature",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_1",
]
NEW_DIAGNOSTIC = ["mean_top_net_long_wave_radiation_flux"]


def _read_one(path_str: str) -> dict:
    with h5py.File(path_str, "r") as f:
        g = f["input"]
        # Raw H5 is S->N; flip lat (axis 0 of the (lat,lon) field) to N->S so the
        # added arrays match the store's N->S orientation (see pangu_h5_to_zarr.py).
        return {
            name: (np.asarray(g[src][:], dtype="float32") * np.float32(scale))[::-1, :]
            for name, (src, scale) in ADD_VARS.items()
        }


def augment(h5_dir: Path, store_path: Path, year: int, n_workers: int) -> None:
    # use_consolidated=False so newly-created arrays are visible via getitem
    # (a consolidated-metadata view is frozen to the arrays it was built with);
    # we regenerate the consolidated metadata at the end.
    store = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)
    ref = store["2m_temperature"]
    n_time, n_lat, n_lon = ref.shape
    files = [str(h5_dir / f"{year}_{idx:04d}.h5") for idx in range(n_time)]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{year}: {len(missing)} of {n_time} H5 files missing (e.g. {missing[0]})"
        )
    logger.info("%d: %d timesteps, grid %dx%d", year, n_time, n_lat, n_lon)

    # (Re)create the target arrays with the store's surface chunking (1, lat, lon).
    for name in ADD_VARS:
        if name in store:
            del store[name]
        store.create_array(
            name,
            shape=(n_time, n_lat, n_lon),
            chunks=(1, n_lat, n_lon),
            dtype="float32",
            # REQUIRED: xarray (which the datapipe uses to open the store) needs
            # dimension_names in each Zarr v3 array's metadata to map dims. Arrays
            # created without it raise "Zarr object is missing the dimension_names
            # metadata" on xr.open_zarr.
            dimension_names=("time", "lat", "lon"),
        )

    # Stream reads across workers; write each timestep as it arrives (in order).
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for idx, payload in enumerate(ex.map(_read_one, files, chunksize=4)):
            for name in ADD_VARS:
                store[name][idx] = payload[name]
            if idx % 500 == 0:
                logger.info("  %d: wrote timestep %d/%d", year, idx, n_time)

    # Attrs: promote SST + new surface prognostics; drop SST from boundary; add OLR.
    surf = list(store.attrs.get("surface_variables", []))
    for v in NEW_SURFACE:
        if v not in surf:
            surf.append(v)
    vb = [v for v in store.attrs.get("varying_boundary_variables", []) if v != "sea_surface_temperature"]
    diag = list(store.attrs.get("diagnostic_variables", []))
    for v in NEW_DIAGNOSTIC:
        if v not in diag:
            diag.append(v)
    store.attrs["surface_variables"] = surf
    store.attrs["varying_boundary_variables"] = vb
    store.attrs["diagnostic_variables"] = diag
    store.attrs["augmented_full_surface"] = True

    zarr.consolidate_metadata(store.store)
    logger.info(
        "%d done: surface=%d %s | varying_boundary=%s | diagnostic=%d %s",
        year, len(surf), surf, vb, len(diag), diag,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5-dir", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--n-workers", type=int, default=16)
    args = p.parse_args()
    augment(args.h5_dir, args.store, args.year, args.n_workers)


if __name__ == "__main__":
    main()
