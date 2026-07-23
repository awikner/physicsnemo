# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

from physmetrics_weather.physics_metrics import (
    get_grid_cell_area,
    compute_dry_air_mass,
    compute_water_mass,
    derive_surface_pressure,
    _compute_tcwv,
    _find_var,
    _detect_level_dim,
    Q_NAMES,
)
from physmetrics_weather.run_all_metrics import load_static_fields, _get_ps

# TODO(multi-year): hardcoded to 2024.zarr below because that's the only year
# I had access to -- the hindcast archive is one Zarr store per year (e.g.
# hindcasts/pangu_s2s/2024.zarr, .../2025.zarr, ...), so MODELS below needs to
# glob for all available *.zarr under each model's dir and combine them along
# init_time (e.g. xr.open_mfdataset(..., concat_dim='init_time', combine='nested')
# or xr.concat over per-year xr.open_zarr calls) instead of a single hardcoded path.
# Output filenames also aren't year-labeled -- a multi-year rerun will silently
# overwrite the previous results/ file.
DATA = "/path/to/physicsnemo-zarr"  # adjust to your own data root
MODELS = {
    "pangu_s2s": f"{DATA}/hindcasts/pangu_s2s/2024.zarr",
    "sfno_s2s": f"{DATA}/hindcasts/sfno_s2s/2024.zarr",
}
ERA5_PATH = f"{DATA}/era5/2024.zarr"

_ds_cache = {}
_ds_static = None
_era5 = None
_era5_level_dim = None
_era5_area = None
_era5_mass_cache = {}  # valid_time -> (dry_Eg, water_kg), persists per worker


def _rename_latlon(ds):
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    return ds.rename(rename) if rename else ds


def _mass_at_snapshot(snap, level_dim, area):
    ps = _get_ps(snap, _ds_static, level_dim=level_dim)
    q_name = _find_var(snap, Q_NAMES)
    tcwv = _compute_tcwv(snap, ps, q_name=q_name, level_dim=level_dim)
    dry = compute_dry_air_mass(snap, ps, area, q_name=q_name, level_dim=level_dim, tcwv=tcwv)
    water = compute_water_mass(snap, ps, area, q_name=q_name, level_dim=level_dim, tcwv=tcwv)
    return dry, water


def _mass_at_snapshot_forced_derivation(snap, level_dim, area):
    """Force the hypsometric MSL-derived surface pressure, ignoring any direct
    surface_pressure variable -- lets us compare Pangu against itself under the
    same derivation sfno_s2s is forced to use, isolating method vs. model."""
    ps = derive_surface_pressure(snap, _ds_static)
    q_name = _find_var(snap, Q_NAMES)
    tcwv = _compute_tcwv(snap, ps, q_name=q_name, level_dim=level_dim)
    dry = compute_dry_air_mass(snap, ps, area, q_name=q_name, level_dim=level_dim, tcwv=tcwv)
    water = compute_water_mass(snap, ps, area, q_name=q_name, level_dim=level_dim, tcwv=tcwv)
    return dry, water


def _era5_mass(valid_time):
    key = np.datetime64(valid_time)
    if key in _era5_mass_cache:
        return _era5_mass_cache[key]
    snap = _era5.sel(time=key, method="nearest").load()
    dry, water = _mass_at_snapshot(snap, _era5_level_dim, _era5_area)
    _era5_mass_cache[key] = (dry, water)
    return dry, water


def _worker_init():
    global _ds_static, _era5, _era5_level_dim, _era5_area
    _era5 = _rename_latlon(xr.open_zarr(ERA5_PATH, decode_timedelta=True))
    _ds_static = load_static_fields(_era5)
    _era5_level_dim = _detect_level_dim(_era5)
    _era5_area = get_grid_cell_area(_era5)
    for model_name, path in MODELS.items():
        ds = _rename_latlon(xr.open_zarr(path, decode_timedelta=True))
        level_dim = _detect_level_dim(ds)
        area = get_grid_cell_area(ds)
        _ds_cache[model_name] = (ds, level_dim, area)


def _process_one_init(model_name, init):
    ds, level_dim, area = _ds_cache[model_name]
    ds_init = ds.sel(init_time=init)
    rows = []
    for ld in ds_init["lead_time"].values:
        lead_day = int(ld / np.timedelta64(1, "D"))
        snap = ds_init.sel(lead_time=ld).load()
        dry, water = _mass_at_snapshot(snap, level_dim, area)
        rows.append((model_name, lead_day, dry, water))

        if model_name == "pangu_s2s":
            dry_d, water_d = _mass_at_snapshot_forced_derivation(snap, level_dim, area)
            rows.append(("pangu_s2s_msl_derived", lead_day, dry_d, water_d))

        valid_time = init + ld
        era5_dry, era5_water = _era5_mass(valid_time)
        rows.append(("era5", lead_day, era5_dry, era5_water))
    return rows


def _process_chunk(chunk):
    all_rows = []
    for model_name, init in chunk:
        try:
            all_rows.extend(_process_one_init(model_name, init))
        except Exception as exc:
            print(f"  [FAILED] {model_name} init={init}: {exc}")
            continue
        print(f"  {model_name} init={init} done")
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    model_inits = {}
    for model_name, path in MODELS.items():
        ds_probe = _rename_latlon(xr.open_zarr(path, decode_timedelta=True))
        model_inits[model_name] = sorted(ds_probe["init_time"].values)

    all_inits = sorted(set().union(*[set(v) for v in model_inits.values()]))
    tasks = []
    for init in all_inits:
        for model_name, inits in model_inits.items():
            if init in inits:
                tasks.append((model_name, init))

    n_workers = args.workers
    chunk_size = max(1, -(-len(tasks) // n_workers))
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for i, fut in enumerate(as_completed(futures), 1):
            all_rows.extend(fut.result())
            print(f"[chunk {futures[fut]}] done ({i}/{len(chunks)} chunks complete)")

    df = pd.DataFrame(all_rows, columns=["model", "lead_day", "dry_mass_Eg", "water_mass_kg"])
    # era5 rows get emitted once per (model, init) task that shares that init/lead_day,
    # so era5 has ~2x the row count of the models (duplicated identical values across
    # pangu_s2s's and sfno_s2s's tasks for the same init) -- harmless for the mean/std,
    # just inflates era5's "n" column; drop duplicates to keep it honest.
    df = df.drop_duplicates()

    agg = df.groupby(["model", "lead_day"]).agg(
        dry_mass_Eg_mean=("dry_mass_Eg", "mean"),
        dry_mass_Eg_std=("dry_mass_Eg", "std"),
        water_mass_kg_mean=("water_mass_kg", "mean"),
        water_mass_kg_std=("water_mass_kg", "std"),
        n=("dry_mass_Eg", "count"),
    ).reset_index()

    out = Path(__file__).parent / "results" / "mass_trajectory.csv"
    agg.to_csv(out, index=False)
    print(f"Wrote {len(agg)} rows to {out}")


if __name__ == "__main__":
    main()
