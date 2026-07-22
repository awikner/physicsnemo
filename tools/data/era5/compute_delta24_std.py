# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compute per-variable 24 h-increment std stats for the ArchesWeather loss.

The ArchesWeather-M "loss_delta_normalization" weights each channel's error by
``(pangu_std / delta24_std)^2`` where ``delta24_std`` is the per-variable SCALAR
std of the 24 h increment ``x(t+24h) - x(t)`` (all levels pooled per variable),
and ``pangu_std`` is the per-variable (surface) / per-(variable, level) (upper
air) normalization std.

This script computes ``delta24_std`` from the raw ERA5 year-stores and reads
``pangu_std`` from the normalization std-store, then writes the loss scaler JSON
consumed by ``ArchesWeatherLoss(delta_scaler_path=...)``:

    {"surface": {var: pangu_std/delta24_std, ...},
     "level":   {var: [pangu_std[l]/delta24_std for l in levels], ...}}

Plain xarray / numpy / zarr only — safe to run on a login node or via the
stampede3-cpu-job skill. Import physicsnemo is intentionally avoided.

Example
-------
    python tools/data/era5/compute_delta24_std.py \
        --data-dir $AI_ROSSBY_DATA/era5_train \
        --std-store $AI_ROSSBY_DATA/era5/normalization_pangu_s2s_std.zarr \
        --out-json $AI_ROSSBY_DATA/era5/archesweather_delta_scaler.json \
        --surface-vars 2m_temperature 10m_u_component_of_wind \
                       10m_v_component_of_wind mean_sea_level_pressure \
        --level-vars temperature u_component_of_wind v_component_of_wind \
                     specific_humidity geopotential \
        --levels 5 10 20 30 50 70 100 150 250 300 400 500 600 700 850 925 1000 \
        --step-24h 4 --sample-stride 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _open_years(data_dir: Path) -> list[xr.Dataset]:
    stores = sorted(data_dir.glob("*.zarr"))
    if not stores:
        raise SystemExit(f"no *.zarr stores under {data_dir}")
    return [xr.open_zarr(s, consolidated=True, decode_times=False) for s in stores]


def _delta_std_for_var(
    datasets: list[xr.Dataset], var: str, *, step_24h: int, stride: int
) -> float:
    """Scalar std of (x(t+24h) - x(t)) for one variable, all levels pooled.

    Increments are computed WITHIN each year-store only (no cross-store pairing),
    which is exact for interior samples and drops only the last ``step_24h``
    samples of each year — negligible for a std estimate.
    """
    acc_sq = 0.0
    acc_n = 0
    acc_sum = 0.0
    for ds in datasets:
        if var not in ds:
            continue
        da = ds[var]
        nt = da.sizes["time"]
        if nt <= step_24h:
            continue
        idx = np.arange(0, nt - step_24h, max(1, stride))
        for i in idx:
            a = np.asarray(da.isel(time=i + step_24h).values, dtype="float64")
            b = np.asarray(da.isel(time=i).values, dtype="float64")
            d = (a - b).ravel()
            d = d[np.isfinite(d)]
            acc_sq += float(np.sum(d * d))
            acc_sum += float(np.sum(d))
            acc_n += d.size
    if acc_n == 0:
        raise SystemExit(f"no samples accumulated for variable {var!r}")
    mean = acc_sum / acc_n
    var_ = acc_sq / acc_n - mean * mean
    return float(np.sqrt(max(var_, 0.0)))


def _pangu_std_surface(std_ds: xr.Dataset, var: str) -> float:
    v = np.asarray(std_ds[var].values, dtype="float64").ravel()
    return float(v[0]) if v.size else float("nan")


def _pangu_std_level(std_ds: xr.Dataset, var: str, levels: list[float]) -> list[float]:
    da = std_ds[var]
    if "pressure_level" not in da.dims:
        # scalar std applied to all levels
        s = float(np.asarray(da.values, dtype="float64").ravel()[0])
        return [s] * len(levels)
    store_levels = [float(x) for x in std_ds["pressure_level"].values]
    idx = {round(l, 3): i for i, l in enumerate(store_levels)}
    vals = np.asarray(da.values, dtype="float64")
    # move pressure_level to axis 0
    ax = da.dims.index("pressure_level")
    vals = np.moveaxis(vals, ax, 0)
    out = []
    for l in levels:
        key = round(float(l), 3)
        if key not in idx:
            raise SystemExit(f"level {l} not in std store levels {store_levels}")
        out.append(float(vals[idx[key]].ravel()[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--std-store", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--surface-vars", nargs="+", required=True)
    ap.add_argument("--level-vars", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", type=float, required=True)
    ap.add_argument("--step-24h", type=int, default=4, help="time steps per 24h (6h data -> 4)")
    ap.add_argument("--sample-stride", type=int, default=20, help="subsample stride over time")
    args = ap.parse_args()

    datasets = _open_years(args.data_dir)
    std_ds = xr.open_zarr(args.std_store, consolidated=True, decode_times=False)

    surface = {}
    for v in args.surface_vars:
        d = _delta_std_for_var(datasets, v, step_24h=args.step_24h, stride=args.sample_stride)
        p = _pangu_std_surface(std_ds, v)
        surface[v] = p / d if d > 0 else 0.0
        print(f"surface {v:35s} delta24_std={d:.6g} pangu_std={p:.6g} scaler={surface[v]:.6g}")

    level = {}
    for v in args.level_vars:
        d = _delta_std_for_var(datasets, v, step_24h=args.step_24h, stride=args.sample_stride)
        p_levels = _pangu_std_level(std_ds, v, args.levels)
        level[v] = [(p / d if d > 0 else 0.0) for p in p_levels]
        print(f"level   {v:35s} delta24_std={d:.6g} scaler[min,max]="
              f"[{min(level[v]):.4g},{max(level[v]):.4g}]")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump({"surface": surface, "level": level}, fh, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
