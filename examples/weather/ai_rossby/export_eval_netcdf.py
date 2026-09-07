# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Export an ``eval_suite.pt`` to labelled NetCDF.

The suite saves a torch dict, which is convenient in-process and awkward
everywhere else. This writes the two artifacts that are actually analysed:

* **Campaign A** -- the time- and member-averaged BIAS fields, as
  ``(variable, lat, lon)`` for surface/diagnostic and
  ``(variable, level, lat, lon)`` for upper air, alongside the pred and truth
  means they are differenced from.
* **Campaign B** -- the full daily ``t2m`` field as ``(time, lat, lon)`` plus
  the pred/truth global-mean series and the fitted trends.

**Latitude and longitude are READ from a store, never synthesized.** Row order
is not uniform across these archives (AMIP is S->N, era5/plasim are N->S), and
fabricating a coordinate from the grid shape is precisely how a field ends up
labelled upside-down relative to its data -- ``rollout.py::_highres_coords``
carries the same warning. The store must be at the SCORED grid: for Campaign A
that is the downscaler's 180x360 (``amip_dailyavg_boundary`` is native), not
the coarse store the forecaster rolls on.

Usage::

    python export_eval_netcdf.py \
        --results eval_bias5yr_e24/eval_suite.pt \
        --coords-zarr $AI_ROSSBY_DATA/amip_dailyavg_boundary \
        --model-config conf/model/amip_rsi_sst_pred.yaml \
        --out eval_bias5yr_e24/bias_fields.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

_GROUP_DIM = {"surface": "surface_var", "upper_air": "upper_air_var",
              "diagnostic": "diag_var"}


def _read_coords(zarr_path: str | Path, n_lat: int, n_lon: int):
    """lat/lon from the store's own coords, refusing a grid mismatch."""
    path = Path(zarr_path)
    if path.is_dir() and not str(path).endswith(".zarr"):
        years = sorted(path.glob("*.zarr"))
        if not years:
            raise FileNotFoundError(f"no per-year stores under {path}")
        path = years[0]
    ds = xr.open_zarr(path, consolidated=True, decode_times=False)
    lat = np.asarray(ds["lat"].values, dtype="float32")
    lon = np.asarray(ds["lon"].values, dtype="float32")
    if lat.shape[0] != n_lat or lon.shape[0] != n_lon:
        raise ValueError(
            f"{path} is {lat.shape[0]}x{lon.shape[0]} but the results are "
            f"{n_lat}x{n_lon}. Point --coords-zarr at a store on the SCORED "
            f"grid (a cascade scores at the downscaler's resolution)."
        )
    return lat, lon


def _var_names(model_cfg: dict, group: str, n: int) -> list[str]:
    key = {"surface": "surface_variables", "upper_air": "upper_air_variables",
           "diagnostic": "diagnostic_variables"}[group]
    names = [str(v) for v in (model_cfg.get(key) or [])]
    if len(names) != n:
        # Label positionally rather than refuse: the arrays are still correct,
        # and a wrong-length name list is a config mismatch worth surfacing
        # without losing the export.
        print(f"  WARNING {group}: config lists {len(names)} names for {n} "
              f"channels; falling back to indices")
        return [f"{group}_{i}" for i in range(n)]
    return names


def export_bias(results: dict, *, lat, lon, model_cfg: dict) -> xr.Dataset:
    clim = results.get("climatology")
    if not clim:
        raise ValueError("no 'climatology' block: this is not a Campaign A result")
    data, coords = {}, {"lat": ("lat", lat), "lon": ("lon", lon)}
    levels = [float(v) for v in (model_cfg.get("levels") or [])]

    for group in ("surface", "upper_air", "diagnostic"):
        bias = clim.get(f"{group}_bias")
        if bias is None:
            continue
        arr = bias.detach().cpu().numpy()
        vdim = _GROUP_DIM[group]
        coords[vdim] = (vdim, np.asarray(_var_names(model_cfg, group, arr.shape[0])))
        if group == "upper_air":
            if len(levels) == arr.shape[1]:
                coords["level"] = ("level", np.asarray(levels, dtype="float32"))
            dims = (vdim, "level", "lat", "lon")
        else:
            dims = (vdim, "lat", "lon")
        for kind in ("bias", "pred_mean", "truth_mean"):
            t = clim.get(f"{group}_{kind}")
            if t is not None:
                data[f"{group}_{kind}"] = (dims, t.detach().cpu().numpy())

    ds = xr.Dataset(data, coords=coords)
    for group in ("surface", "upper_air", "diagnostic"):
        name = f"{group}_bias"
        if name in ds:
            ds[name].attrs["long_name"] = (
                "time- and ensemble-member-mean bias (predicted - observed)"
            )
    cfg = results.get("config", {})
    ds.attrs.update({
        "description": "amip_v2-parity climatological bias, time- and "
                       "member-averaged (predicted - observed climatology)",
        "truth_source": str(cfg.get("truth_source", "")),
        "obs_climatology_dir": str(cfg.get("obs_climatology_dir", "")),
        "horizon_frames": int(cfg.get("horizon", 0) or 0),
        "ensemble_size": int(cfg.get("ensemble_size", 0) or 0),
        "ic_date": str(cfg.get("ic_date", "")),
        "ic_resolved_time": str(cfg.get("ic_resolved_time", "")),
        "weights_used": str(cfg.get("weights_used", "")),
        "checkpoint_epoch": str(cfg.get("epoch", cfg.get("checkpoint_epoch", ""))),
        "in_sample_note": str(cfg.get("in_sample_note") or ""),
        "lat_row_order": "S->N (ascending)" if lat[0] < lat[-1] else "N->S",
    })
    # The headline scalars, and their across-member spread when present.
    for label, st in (results.get("headline") or {}).items():
        ds.attrs[f"headline_{label}_mean_bias"] = float(st["mean_bias"])
        ds.attrs[f"headline_{label}_rmse_bias"] = float(st["rmse_bias"])
    ms = results.get("member_spread") or {}
    for group, blk in ms.items():
        ds.attrs[f"member_spread_{group}_n_members"] = int(blk["n_members"])
    return ds


def export_field_series(results: dict, *, lat, lon, cfg_step_hours: float = 24.0):
    fs = results.get("field_series")
    if not fs:
        raise ValueError("no 'field_series' block: this is not a Campaign B result")
    cfg = results.get("config", {})
    data, coords = {}, {"lat": ("lat", lat), "lon": ("lon", lon)}
    n_time = None
    for name, t in fs.items():
        if not hasattr(t, "shape"):
            continue
        arr = t.detach().cpu().numpy()
        n_time = arr.shape[0]
        data[name] = (("time", "lat", "lon"), arr)
    if n_time is None:
        raise ValueError("field_series carried no tensors (saved with local_only?)")
    # Frame k is the prediction valid at ic + (k+1) model steps -- the oracle
    # init window is NOT saved. Same convention amip_v2's meta.yaml records
    # verbatim, and getting it wrong shifts every date by one day.
    coords["time"] = ("time", np.arange(1, n_time + 1, dtype="int32"))
    ds = xr.Dataset(data, coords=coords)
    ds["time"].attrs["long_name"] = "model steps after the initial condition"
    ds["time"].attrs["note"] = (
        f"index k is the prediction valid at ic_date + k*{cfg_step_hours:g} h; "
        f"the oracle init window is not saved"
    )
    for name in list(data):
        ds[name].attrs["units"] = "K" if "temperature" in name else "unknown"

    gm = results.get("global_mean") or {}
    for which, key in (("flux_pred_series", "pred"), ("flux_truth_series", "truth")):
        for name, series in (gm.get(which) or {}).items():
            ds[f"{name}_global_mean_{key}"] = (
                ("time",), np.asarray(series.detach().cpu(), dtype="float64")
            )
    for key, entry in (results.get("trend") or {}).items():
        for label, fit in entry.items():
            if "slope" in fit:
                tag = f"trend_{key.replace(':', '_')}_{label}"
                ds.attrs[f"{tag}_K_per_decade"] = float(fit["slope"])
                ds.attrs[f"{tag}_stderr"] = float(fit["stderr"])
                ds.attrs[f"{tag}_r2"] = float(fit["r2"])
                ds.attrs[f"{tag}_n"] = int(fit["n"])
    ds.attrs.update({
        "description": "daily field series and global-mean trend",
        "ic_date": str(cfg.get("ic_date", "")),
        "ic_resolved_time": str(cfg.get("ic_resolved_time", "")),
        "weights_used": str(cfg.get("weights_used", "")),
        "quote_which_trend": "annual (the daily fit is BIASED by the seasonal "
                             "cycle: -0.0156 K/decade per K of amplitude)",
        "lat_row_order": "S->N (ascending)" if lat[0] < lat[-1] else "N->S",
    })
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--coords-zarr", required=True,
                    help="a store at the SCORED grid (never synthesized)")
    ap.add_argument("--model-config", required=True,
                    help="the forecaster's model yaml, for variable names/levels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-hours", type=float, default=24.0)
    a = ap.parse_args()

    r = torch.load(a.results, map_location="cpu", weights_only=False)
    model_cfg = yaml.safe_load(Path(a.model_config).read_text()) or {}

    clim = r.get("climatology")
    fs = r.get("field_series")
    if clim:
        shape = next(iter(clim.values())).shape
        lat, lon = _read_coords(a.coords_zarr, int(shape[-2]), int(shape[-1]))
        ds = export_bias(r, lat=lat, lon=lon, model_cfg=model_cfg)
    elif fs:
        t = next(v for v in fs.values() if hasattr(v, "shape"))
        lat, lon = _read_coords(a.coords_zarr, int(t.shape[-2]), int(t.shape[-1]))
        ds = export_field_series(r, lat=lat, lon=lon, cfg_step_hours=a.step_hours)
    else:
        raise SystemExit("results carry neither a climatology nor a field_series")

    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(a.out, encoding=enc)
    print(f"wrote {a.out}")
    for v in ds.data_vars:
        print(f"  {v:34s} {tuple(ds[v].shape)}  dims {ds[v].dims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
