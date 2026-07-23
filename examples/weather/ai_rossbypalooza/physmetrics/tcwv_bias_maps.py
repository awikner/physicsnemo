# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

from physmetrics_weather.physics_metrics import (
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

DEFAULT_LEAD_DAYS = [8, 9, 10, 11, 12, 13, 14]

_ds_model = None
_model_level_dim = None
_era5 = None
_ds_static = None
_era5_level_dim = None
_model_path = None


def _rename_latlon(ds):
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    return ds.rename(rename) if rename else ds


def _worker_init(model_path):
    global _ds_model, _model_level_dim, _era5, _ds_static, _era5_level_dim
    _era5 = _rename_latlon(xr.open_zarr(ERA5_PATH, decode_timedelta=True))
    _ds_static = load_static_fields(_era5)
    _era5_level_dim = _detect_level_dim(_era5)
    _ds_model = _rename_latlon(xr.open_zarr(model_path, decode_timedelta=True))
    _model_level_dim = _detect_level_dim(_ds_model)


def _tcwv_snapshot(snap, level_dim):
    ps = _get_ps(snap, _ds_static, level_dim=level_dim)
    q_name = _find_var(snap, Q_NAMES)
    return _compute_tcwv(snap, ps, q_name=q_name, level_dim=level_dim)


def _process_init(init, lead_days):
    ds_init = _ds_model.sel(init_time=init)
    lead_hours_all = ds_init["lead_time"].values / np.timedelta64(1, "h")

    result = {}
    for lead_d in lead_days:
        target_h = lead_d * 24.0
        idx = int(np.argmin(np.abs(lead_hours_all - target_h)))
        if abs(lead_hours_all[idx] - target_h) > 1e-6:
            continue
        ld = ds_init["lead_time"].values[idx]

        snap = ds_init.sel(lead_time=ld).load()
        model_tcwv = _tcwv_snapshot(snap, _model_level_dim).values

        valid_time = init + np.timedelta64(int(target_h), "h")
        ref_snap = _era5.sel(time=valid_time, method="nearest").load()
        era5_tcwv = _tcwv_snapshot(ref_snap, _era5_level_dim).values

        result[lead_d] = (model_tcwv, era5_tcwv)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--lead-days", type=str, default=None,
        help="Comma-separated lead days, e.g. '8,9,10,11,12,13,14' (default: 8-14)",
    )
    parser.add_argument(
        "--model", type=str, choices=list(MODELS.keys()), default="pangu_s2s",
        help="Which hindcast model to compare against ERA5 (default: pangu_s2s)",
    )
    args = parser.parse_args()

    lead_days = (
        [int(x) for x in args.lead_days.split(",")]
        if args.lead_days else DEFAULT_LEAD_DAYS
    )
    model_path = MODELS[args.model]

    ds_probe = _rename_latlon(xr.open_zarr(model_path, decode_timedelta=True))
    inits = list(ds_probe["init_time"].values)
    lat = ds_probe["latitude"].values
    lon = ds_probe["longitude"].values

    sums = {ld: np.zeros((len(lat), len(lon))) for ld in lead_days}
    sums_ref = {ld: np.zeros((len(lat), len(lon))) for ld in lead_days}
    counts = {ld: 0 for ld in lead_days}

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init, initargs=(model_path,)
    ) as pool:
        futures = {pool.submit(_process_init, init, lead_days): init for init in inits}
        for i, fut in enumerate(as_completed(futures), 1):
            init = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"[{i}/{len(inits)}] init={init} FAILED: {exc}")
                continue
            for ld, (model_tcwv, era5_tcwv) in result.items():
                sums[ld] += model_tcwv
                sums_ref[ld] += era5_tcwv
                counts[ld] += 1
            print(f"[{i}/{len(inits)}] init={init} done")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"tcwv_bias_maps_{args.model}.npz"
    save_kwargs = {"lat": lat, "lon": lon}
    for ld in lead_days:
        n = counts[ld]
        if n == 0:
            continue
        save_kwargs[f"model_mean_lead{ld}"] = sums[ld] / n
        save_kwargs[f"era5_mean_lead{ld}"] = sums_ref[ld] / n
        save_kwargs[f"n_lead{ld}"] = n
    np.savez(out_path, **save_kwargs)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
