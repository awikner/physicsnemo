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
    compute_drift_slope,
    _compute_tcwv,
    _find_var,
    _detect_level_dim,
    Q_NAMES,
    SP_NAMES,
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

DEFAULT_LEAD_DAYS = [8, 9, 10, 11, 12, 13, 14]
WINDOW_START_H = 12.0

# Populated once per worker process by _worker_init.
_ds_cache = {}
_era5 = None
_ds_static = None
_era5_level_dim = None
# Persists across every init a worker processes -> cache hits across nearby inits.
_era5_water_cache = {}


def _rename_latlon(ds):
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    return ds.rename(rename) if rename else ds


def _worker_init():
    global _era5, _ds_static, _era5_level_dim
    _era5 = _rename_latlon(xr.open_zarr(ERA5_PATH, decode_timedelta=True))
    _ds_static = load_static_fields(_era5)
    _era5_level_dim = _detect_level_dim(_era5)
    for model_name, path in MODELS.items():
        ds = _rename_latlon(xr.open_zarr(path, decode_timedelta=True))
        level_dim = _detect_level_dim(ds)
        area = get_grid_cell_area(ds)
        _ds_cache[model_name] = (ds, level_dim, area)


def _mass_pair(snap, ds_static, level_dim):
    ps = _get_ps(snap, ds_static, level_dim=level_dim)
    q_name = _find_var(snap, Q_NAMES)
    tcwv = _compute_tcwv(snap, ps, q_name=q_name, level_dim=level_dim)
    return ps, q_name, tcwv


def _era5_water_mass(valid_time, area):
    key = np.datetime64(valid_time)
    if key in _era5_water_cache:
        return _era5_water_cache[key]
    ref_snap = _era5.sel(time=key, method="nearest").load()
    ps_ref, q_ref, tcwv_ref = _mass_pair(ref_snap, _ds_static, _era5_level_dim)
    water_r = compute_water_mass(ref_snap, ps_ref, area, q_name=q_ref,
                                  level_dim=_era5_level_dim, tcwv=tcwv_ref)
    _era5_water_cache[key] = water_r
    return water_r


def _process_one_init(model_name, init, lead_days):
    ds, level_dim, area = _ds_cache[model_name]
    ds_init = ds.sel(init_time=init)
    lead_hours_all = ds_init["lead_time"].values / np.timedelta64(1, "h")
    max_lead_h = max(lead_days) * 24.0

    # Compute every model snapshot needed for the FULL window ONCE.
    mask_full = (lead_hours_all >= WINDOW_START_H) & (lead_hours_all <= max_lead_h)
    sel_leads_full = ds_init["lead_time"].values[mask_full]

    hours_full, dry_full, water_full = [], [], []
    for ld in sel_leads_full:
        snap = ds_init.sel(lead_time=ld).load()
        ps, q_name, tcwv = _mass_pair(snap, _ds_static, level_dim)
        dry = compute_dry_air_mass(snap, ps, area, q_name=q_name, level_dim=level_dim, tcwv=tcwv)
        water = compute_water_mass(snap, ps, area, q_name=q_name, level_dim=level_dim, tcwv=tcwv)
        h = float(ld / np.timedelta64(1, "h"))
        hours_full.append(h)
        dry_full.append(dry)
        water_full.append(water)

    hours_full = np.array(hours_full)
    dry_full = np.array(dry_full)
    water_full = np.array(water_full)

    valid_times_full = init + hours_full.astype("timedelta64[h]")
    water_ref_full = np.array([_era5_water_mass(vt, area) for vt in valid_times_full])

    rows = []
    for lead_d in lead_days:
        window_mask = hours_full <= lead_d * 24.0
        if window_mask.sum() < 2:
            continue

        hm = hours_full[window_mask]
        dm = dry_full[window_mask]
        wm = water_full[window_mask]
        wr = water_ref_full[window_mask]

        slope_dry = compute_drift_slope(hm, dm)
        slope_water = compute_drift_slope(hm, wm)
        slope_water_ref = compute_drift_slope(hm, wr)

        dry_drift_pct = (slope_dry / dm[0] * 100.0) if dm[0] else np.nan
        water_model_pct = (slope_water / wm[0] * 100.0) if wm[0] else np.nan
        water_ref_pct = (slope_water_ref / wr[0] * 100.0) if wr[0] else np.nan

        rows.append({
            "model": model_name,
            "init_time": str(init),
            "lead_days": lead_d,
            "dry_mass_drift_pct_per_day": dry_drift_pct,
            "water_mass_drift_pct_per_day": water_model_pct - water_ref_pct,
        })
    return rows


def _process_chunk(chunk, lead_days):
    all_rows = []
    for model_name, init in chunk:
        try:
            all_rows.extend(_process_one_init(model_name, init, lead_days))
        except Exception as exc:
            print(f"  [FAILED] {model_name} init={init}: {exc}")
        print(f"  {model_name} init={init} done (cache size={len(_era5_water_cache)})")
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--lead-days", type=str, default=None,
        help="Comma-separated lead days, e.g. '8,9,10,11,12,13,14' (default: 8-14)",
    )
    args = parser.parse_args()

    lead_days = (
        [int(x) for x in args.lead_days.split(",")]
        if args.lead_days else DEFAULT_LEAD_DAYS
    )

    model_inits = {}
    for model_name, path in MODELS.items():
        ds_probe = _rename_latlon(xr.open_zarr(path, decode_timedelta=True))
        if _find_var(ds_probe, SP_NAMES) is None:
            print(f"[warn] {model_name}: no direct surface_pressure variable — "
                  f"PS will be derived (hypsometric MSL + era5 static geopotential)")
        model_inits[model_name] = sorted(ds_probe["init_time"].values)

    # Interleave by init first, model second: pangu_s2s and sfno_s2s share the
    # same init dates, so adjacent tasks need the same era5 valid_times and are
    # likely to land in the same worker chunk -> cache hits across models too.
    all_inits = sorted(set().union(*[set(v) for v in model_inits.values()]))
    tasks = []
    for init in all_inits:
        for model_name, inits in model_inits.items():
            if init in inits:
                tasks.append((model_name, init))

    # Contiguous chunks (not round-robin): each worker gets temporally-adjacent
    # inits, so the era5 cache actually gets reused within a worker.
    n_workers = args.workers
    chunk_size = max(1, -(-len(tasks) // n_workers))
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_process_chunk, chunk, lead_days): i
                   for i, chunk in enumerate(chunks)}
        for i, fut in enumerate(as_completed(futures), 1):
            chunk_idx = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            print(f"[chunk {chunk_idx}] done ({i}/{len(chunks)} chunks complete, "
                  f"{len(rows)} rows from this chunk)")

    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent / "results" / "mass_drift_week2.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
