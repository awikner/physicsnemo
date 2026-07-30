#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reformat DSI AIWP per-init hindcast stores into two archive formats.

Source layout (on the UChicago DSI cluster, ``/net/monsoon/marchakitus/...``):
one **per-init** store per initialization —

* Zarr v3 ``init_YYYYMMDD[THH].zarr`` with flat, level-baked variable names
  (``q_850``, ``z_500``, ``2t``, ``tp``, ``swvl1`` …). Each variable is
  dimensioned ``(time=1, <lead-axis>, lat, lon)`` where ``<lead-axis>`` is
  either ``prediction_timedelta`` (hours, 6-hourly) or
  ``prediction_timedelta_daily`` (days) — the split is per-variable and per
  store. Arrays are compressed with a ``sharding_indexed`` → ``bitround`` →
  ``blosc`` codec stack (needs ``import numcodecs.zarr3`` + zarr >= 3.1.3).
* NetCDF ``init_YYYYMMDDHH.nc`` (the WeatherBench-2 GraphCast archive) with a
  single 6-hourly ``step`` lead axis.

This tool writes, for a chosen model, the deterministic 2000-2024 hindcasts,
sliced to a lead-time window (default lead **days 7-21 inclusive** =
``[168, 504]`` h on the 6h axis and ``[7, 21]`` d on the daily axis), in two
formats — **native 0.25 deg, native flat channels, native cadence**:

**Format 1 — per-sample HDF5** (the PanguWeather-ERA5 convention consumed by
``tools/data/era5/pangu_h5_to_zarr.py``): one ``.h5`` per 6-hourly valid-time,
group ``input`` with flat keys ``input/<varname>`` (verbatim source names) plus
``input/time``; latitude stored **S->N**. The series is *ragged*: every 6h step
carries the 6h vars; daily vars appear only on the 00Z-aligned steps
(lead-hours divisible by 24 within the daily window). Path::

    {h5_root}/{model}/{YYYYMMDDHH}/{lead_hours:04d}.h5

**Format 2 — per-year consolidated Zarr v3** (mirrors
``tools/data/hindcast/consolidate_hindcasts.py`` /
``physicsnemo-zarr/hindcasts/pangu_s2s/{YYYY}.zarr``), but keeping the source's
**two native lead axes** and **flat channels**: 6h vars are
``(init_time, prediction_timedelta, lat, lon)`` and daily vars are
``(init_time, prediction_timedelta_daily, lat, lon)``; latitude stored **N->S**.
Path::

    {zarr_root}/{model}/{YYYY}.zarr

Login-node safety: imports only xarray/zarr/numcodecs/h5py/numpy/cftime/dask —
never ``physicsnemo`` (CUDA/Warp init can core-dump on a login node).

Usage (one model; merge multiple source dirs into one model with repeated
``--source-dir``, e.g. the AIFS-v1 daily + twice-weekly split)::

    python tools/data/hindcast/dsi_hindcast_to_formats.py \\
        --model graphcast_e2s --source-kind zarr \\
        --source-dir /net/monsoon/marchakitus/reforecast/forecasts_graphcast_e2s \\
        --out-root /net/scratch/awikner/hindcasts_dsi \\
        --years 2000-2024 --n-workers 64
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
# Register numcodecs' zarr-v3 codecs (bitround/blosc) so the sharded source
# arrays decode.  MUST precede any xarray.open_zarr of the source.
import numcodecs.zarr3  # noqa: E402,F401
import cftime  # noqa: E402
import xarray as xr  # noqa: E402

logger = logging.getLogger("dsi_hindcast")

HINDCAST_SCHEMA_VERSION = "1.0"
SCRIPT_REL_PATH = "tools/data/hindcast/dsi_hindcast_to_formats.py"
CALENDAR = "standard"

_LAT_DIMS = ("lat", "latitude", "y")
_LON_DIMS = ("lon", "longitude", "x")
_SIXH_AXES = ("prediction_timedelta", "step")       # units: hours
_DAILY_AXES = ("prediction_timedelta_daily",)       # units: days
_TIME_DIMS = ("time", "t", "valid_time")
# Names that are precipitation-like (recorded as diagnostic in store attrs).
_PRECIP = ("tp", "total_precipitation", "total_precipitation_6hr",
           "total_precipitation_24hr", "precip", "pr")

DEFAULT_LEAD_HOURS = (168, 504)   # lead days 7..21 inclusive, on the 6h axis
DEFAULT_LEAD_DAYS = (7, 21)       # lead days 7..21 inclusive, on the daily axis


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _find_dim(dims, candidates) -> Optional[str]:
    lower = {str(d).lower(): str(d) for d in dims}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _parse_init_dt(name: str) -> cftime.DatetimeGregorian:
    """Parse the init datetime from ``init_YYYYMMDD[THH]`` / ``init_YYYYMMDDHH``."""
    m = re.search(r"init_(\d{4})(\d{2})(\d{2})T?(\d{2})?", name)
    if not m:
        raise ValueError(f"no init_YYYYMMDD token in {name!r}")
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    h = int(m.group(4)) if m.group(4) else 0
    return cftime.DatetimeGregorian(y, mo, d, h)


def _init_stamp(dt) -> str:
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}"


def _valid_iso(init_dt, lead_hours: int) -> str:
    v = init_dt + __import__("datetime").timedelta(hours=int(lead_hours))
    return f"{v.year:04d}-{v.month:02d}-{v.day:02d}T{v.hour:02d}:{v.minute:02d}:{v.second:02d}"


def _parse_years(spec: str) -> range:
    lo, hi = (spec.split("-") + [None])[:2]
    lo = int(lo)
    hi = int(hi) if hi else lo
    return range(lo, hi + 1)


def _list_init_files(source_dirs, kind: str):
    """Return sorted list of (init_dt, path) across one or more source dirs.

    Asserts no duplicate init datetimes across dirs (the AIFS-v1 daily +
    twice-weekly merge must be date-disjoint; on any collision we raise rather
    than silently drop — the caller decides).
    """
    suffix = ".zarr" if kind == "zarr" else ".nc"
    seen: dict[tuple, Path] = {}
    out = []
    for sd in source_dirs:
        sd = Path(sd)
        for e in sorted(os.listdir(sd)):
            if not e.endswith(suffix) or not e.startswith("init_"):
                continue
            dt = _parse_init_dt(e)
            key = (dt.year, dt.month, dt.day, dt.hour)
            if key in seen:
                raise ValueError(
                    f"duplicate init date {key} across source dirs: "
                    f"{seen[key]} vs {sd / e} (merge must be date-disjoint)"
                )
            seen[key] = sd / e
            out.append((dt, sd / e))
    out.sort(key=lambda t: (t[0].year, t[0].month, t[0].day, t[0].hour))
    return out


def _open(path: Path, kind: str) -> xr.Dataset:
    if kind == "zarr":
        return xr.open_zarr(path, consolidated=False, decode_timedelta=False)
    return xr.open_dataset(path, decode_timedelta=False)


def _classify(ds: xr.Dataset):
    """Split data_vars into 6h vs daily groups by their lead axis.

    Returns (vars_6h, vars_daily, sixh_dim, daily_dim, lat_dim, lon_dim).
    """
    lat_dim = _find_dim(ds.dims, _LAT_DIMS)
    lon_dim = _find_dim(ds.dims, _LON_DIMS)
    vars_6h, vars_daily = [], []
    sixh_dim = daily_dim = None
    for v in ds.data_vars:
        dims = ds[v].dims
        s = _find_dim(dims, _SIXH_AXES)
        d = _find_dim(dims, _DAILY_AXES)
        if d is not None:
            vars_daily.append(v)
            daily_dim = d
        elif s is not None:
            vars_6h.append(v)
            sixh_dim = s
        # else: no recognized lead axis -> skip (constants); none in these stores
    return sorted(vars_6h), sorted(vars_daily), sixh_dim, daily_dim, lat_dim, lon_dim


def _window_index(coord_vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(coord_vals).ravel()
    return np.nonzero((v >= lo) & (v <= hi))[0]


def _probe(path: Path, kind: str, lead_hours, lead_days):
    """Open one init and return the store's schema + windowed lead coords."""
    ds = _open(path, kind)
    try:
        v6, vd, s_dim, d_dim, lat_dim, lon_dim = _classify(ds)
        lat = np.asarray(ds[lat_dim].values, dtype="float32")
        lon = np.asarray(ds[lon_dim].values, dtype="float32")
        sixh_vals = daily_vals = None
        if s_dim is not None:
            idx = _window_index(ds[s_dim].values, *lead_hours)
            sixh_vals = np.asarray(ds[s_dim].values).ravel()[idx].astype("int64")
        if d_dim is not None:
            idx = _window_index(ds[d_dim].values, *lead_days)
            daily_vals = np.asarray(ds[d_dim].values).ravel()[idx].astype("int64")
    finally:
        ds.close()
    return {
        "vars_6h": v6, "vars_daily": vd,
        "sixh_dim": s_dim, "daily_dim": d_dim,
        "lat_dim": lat_dim, "lon_dim": lon_dim,
        "lat": lat, "lon": lon,
        "sixh_vals": sixh_vals, "daily_vals": daily_vals,
    }


def _load_windowed(ds: xr.Dataset, schema, lead_hours, lead_days):
    """Return per-var windowed numpy arrays (lead, lat, lon), squeezing time."""
    out6, outd = {}, {}
    tdim = _find_dim(ds.dims, _TIME_DIMS)
    s_dim, d_dim = schema["sixh_dim"], schema["daily_dim"]
    lat_dim, lon_dim = schema["lat_dim"], schema["lon_dim"]
    if s_dim is not None:
        i6 = _window_index(ds[s_dim].values, *lead_hours)
    if d_dim is not None:
        idd = _window_index(ds[d_dim].values, *lead_days)
    for v in schema["vars_6h"]:
        da = ds[v]
        if tdim in da.dims:
            da = da.isel({tdim: 0})
        da = da.isel({s_dim: i6}).transpose(s_dim, lat_dim, lon_dim)
        out6[v] = np.asarray(da.values, dtype="float32")
    for v in schema["vars_daily"]:
        da = ds[v]
        if tdim in da.dims:
            da = da.isel({tdim: 0})
        da = da.isel({d_dim: idd}).transpose(d_dim, lat_dim, lon_dim)
        outd[v] = np.asarray(da.values, dtype="float32")
    return out6, outd


# --------------------------------------------------------------------------- #
# Format 1 (per-sample HDF5, ragged 6h series, lat S->N)
# --------------------------------------------------------------------------- #
def _write_h5(*, out6, outd, schema, init_dt, model, h5_root: Path) -> int:
    import h5py

    src_NtoS = float(schema["lat"][0]) > float(schema["lat"][-1])
    flip = src_NtoS  # h5 wants S->N; flip only if source is N->S

    def _sn(a):  # to S->N for the h5 array
        return a[..., ::-1, :] if flip else a

    sixh_vals = schema["sixh_vals"]
    daily_vals = schema["daily_vals"]
    init_dir = h5_root / model / _init_stamp(init_dt)   # dir = datetime string (no "init_" prefix)
    init_dir.mkdir(parents=True, exist_ok=True)

    # Iterate the 6h lead grid (union: daily-aligned steps fold in the daily
    # vars).  If a store has no 6h axis, iterate the daily grid instead.
    n = 0
    if sixh_vals is not None:
        daily_by_hour = {}
        if daily_vals is not None:
            daily_by_hour = {int(d) * 24: di for di, d in enumerate(daily_vals)}
        for j, L in enumerate(int(x) for x in sixh_vals):
            fpath = init_dir / f"{L:04d}.h5"
            with h5py.File(fpath, "w") as f:
                g = f.create_group("input")
                for v in schema["vars_6h"]:
                    g.create_dataset(v, data=_sn(out6[v][j]), dtype="f4")
                di = daily_by_hour.get(L)
                if di is not None:
                    for v in schema["vars_daily"]:
                        g.create_dataset(v, data=_sn(outd[v][di]), dtype="f4")
                g.create_dataset("time", data=np.bytes_(_valid_iso(init_dt, L)))
                g.attrs["init_time"] = _init_stamp(init_dt)
                g.attrs["lead_hours"] = int(L)
                g.attrs["model"] = model
            n += 1
    elif daily_vals is not None:
        for di, d in enumerate(int(x) for x in daily_vals):
            L = d * 24
            fpath = init_dir / f"{L:04d}.h5"
            with h5py.File(fpath, "w") as f:
                g = f.create_group("input")
                for v in schema["vars_daily"]:
                    g.create_dataset(v, data=_sn(outd[v][di]), dtype="f4")
                g.create_dataset("time", data=np.bytes_(_valid_iso(init_dt, L)))
                g.attrs["init_time"] = _init_stamp(init_dt)
                g.attrs["lead_hours"] = int(L)
                g.attrs["model"] = model
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Format 2 (per-year zarr, two lead axes, flat channels, lat N->S)
# --------------------------------------------------------------------------- #
def _create_year_store(*, store_path: Path, init_dts, schema, model,
                       source_dataset, lead_hours, lead_days, generator,
                       overwrite: bool):
    import dask.array as da

    src_NtoS = float(schema["lat"][0]) > float(schema["lat"][-1])
    lat = schema["lat"] if src_NtoS else schema["lat"][::-1]      # store N->S
    lon = schema["lon"]
    n_lat, n_lon = lat.size, lon.size
    v6, vd = schema["vars_6h"], schema["vars_daily"]
    # Canonical output axis names (source may call the 6h axis "step").
    s_dim, d_dim = "prediction_timedelta", "prediction_timedelta_daily"
    n_init = len(init_dts)

    coords = {
        "init_time": ("init_time", np.asarray(init_dts)),
        "lat": ("lat", np.asarray(lat, dtype="float32")),
        "lon": ("lon", np.asarray(lon, dtype="float32")),
    }
    data_vars = {}
    encoding = {}
    if v6:
        n6 = int(schema["sixh_vals"].size)
        coords[s_dim] = (s_dim, np.asarray(schema["sixh_vals"], dtype="int64"))
        for v in v6:
            data_vars[v] = ((("init_time", s_dim, "lat", "lon")),
                            da.zeros((n_init, n6, n_lat, n_lon),
                                     chunks=(1, n6, n_lat, n_lon), dtype="float32"))
            encoding[v] = {"chunks": (1, n6, n_lat, n_lon), "dtype": "float32"}
    if vd:
        nd = int(schema["daily_vals"].size)
        coords[d_dim] = (d_dim, np.asarray(schema["daily_vals"], dtype="int64"))
        for v in vd:
            data_vars[v] = ((("init_time", d_dim, "lat", "lon")),
                            da.zeros((n_init, nd, n_lat, n_lon),
                                     chunks=(1, nd, n_lat, n_lon), dtype="float32"))
            encoding[v] = {"chunks": (1, nd, n_lat, n_lon), "dtype": "float32"}

    diag = [v for v in (v6 + vd) if v.lower() in _PRECIP]
    attrs = {
        "hindcast_schema_version": HINDCAST_SCHEMA_VERSION,
        "model": model,
        "source_dataset": source_dataset,
        "calendar": CALENDAR,
        "n_init": n_init,
        "lead_window_hours": list(lead_hours),
        "lead_window_days": list(lead_days),
        "channel_variables_6h": list(v6),
        "channel_variables_daily": list(vd),
        "diagnostic_variables": diag,
        "note": "native 0.25deg, native flat channels (level baked in name), "
                "native cadence (two lead axes), lat N->S.",
        "generator": generator,
    }
    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    if s_dim in ds.coords:
        ds[s_dim].attrs.update({"units": "hours", "long_name": "forecast lead time"})
    if d_dim and d_dim in ds.coords:
        ds[d_dim].attrs.update({"units": "days", "long_name": "forecast lead time (daily)"})
    encoding["init_time"] = {
        "units": f"hours since {init_dts[0].year}-01-01 00:00:00",
        "calendar": CALENDAR, "dtype": "int64",
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(store_path, mode="w" if overwrite else "w-", zarr_format=3,
               consolidated=True, encoding=encoding, compute=False)


def _write_zarr_region(*, store_path: Path, index: int, out6, outd, schema):
    src_NtoS = float(schema["lat"][0]) > float(schema["lat"][-1])
    flip = not src_NtoS  # store wants N->S; flip only if source is S->N
    s_dim, d_dim = "prediction_timedelta", "prediction_timedelta_daily"

    def _ns(a):
        return a[..., ::-1, :] if flip else a

    region_vars = {}
    for v in schema["vars_6h"]:
        region_vars[v] = (("init_time", s_dim, "lat", "lon"), _ns(out6[v])[np.newaxis, ...])
    for v in schema["vars_daily"]:
        region_vars[v] = (("init_time", d_dim, "lat", "lon"), _ns(outd[v])[np.newaxis, ...])
    xr.Dataset(region_vars).to_zarr(store_path, region={"init_time": slice(index, index + 1)})


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _process_init(args: dict) -> dict:
    """Read one source init, slice the window, write Format 1 and/or Format 2."""
    warnings.filterwarnings("ignore")
    import numcodecs.zarr3  # noqa: F401  (register codecs in the worker too)

    path = Path(args["path"])
    kind = args["kind"]
    model = args["model"]
    lead_hours = tuple(args["lead_hours"])
    lead_days = tuple(args["lead_days"])
    formats = args["formats"]
    init_dt = _parse_init_dt(path.name)

    ds = _open(path, kind)
    try:
        v6, vd, s_dim, d_dim, lat_dim, lon_dim = _classify(ds)
        schema = {
            "vars_6h": v6, "vars_daily": vd, "sixh_dim": s_dim, "daily_dim": d_dim,
            "lat_dim": lat_dim, "lon_dim": lon_dim,
            "lat": np.asarray(ds[lat_dim].values, dtype="float32"),
            "lon": np.asarray(ds[lon_dim].values, dtype="float32"),
            "sixh_vals": (np.asarray(ds[s_dim].values).ravel()[
                _window_index(ds[s_dim].values, *lead_hours)].astype("int64")
                if s_dim is not None else None),
            "daily_vals": (np.asarray(ds[d_dim].values).ravel()[
                _window_index(ds[d_dim].values, *lead_days)].astype("int64")
                if d_dim is not None else None),
        }
        # consistency with the template's schema
        if set(v6) != set(args["exp_vars_6h"]) or set(vd) != set(args["exp_vars_daily"]):
            raise ValueError(f"{path.name}: variable set differs from template")
        out6, outd = _load_windowed(ds, schema, lead_hours, lead_days)
    finally:
        ds.close()

    probe = next(iter({**out6, **outd}.values()))
    finite = float(np.isfinite(probe).mean())

    n_h5 = 0
    if formats in ("both", "h5"):
        n_h5 = _write_h5(out6=out6, outd=outd, schema=schema, init_dt=init_dt,
                         model=model, h5_root=Path(args["h5_root"]))
    if formats in ("both", "zarr"):
        _write_zarr_region(store_path=Path(args["store_path"]),
                           index=int(args["index"]), out6=out6, outd=outd, schema=schema)
    return {"init": _init_stamp(init_dt), "n_h5": n_h5, "finite": finite}


# --------------------------------------------------------------------------- #
# Reverse path: expand a Format-2 store into per-sample h5 (run on Derecho)
# --------------------------------------------------------------------------- #
def expand_zarr_to_h5(store_path: Path, model: str, h5_root: Path) -> int:
    """Regenerate the ragged per-sample h5 series from a Format-2 store.

    Format-2 already holds the exact windowed arrays (init_time, lead, lat, lon,
    lat N->S), so this just replays each init through the same _write_h5 used on
    the forward path (which flips to S->N and folds daily vars into 00Z steps).
    """
    ds = xr.open_zarr(store_path, consolidated=True, use_cftime=True, decode_timedelta=False)
    try:
        v6 = list(ds.attrs.get("channel_variables_6h", []))
        vd = list(ds.attrs.get("channel_variables_daily", []))
        if not v6 and not vd:  # older stores without the attr: infer by dims
            for v in ds.data_vars:
                (v6 if "prediction_timedelta" in ds[v].dims else vd).append(str(v))
        schema = {
            "vars_6h": v6, "vars_daily": vd,
            "lat": np.asarray(ds["lat"].values, dtype="float32"),
            "lon": np.asarray(ds["lon"].values, dtype="float32"),
            "sixh_vals": (np.asarray(ds["prediction_timedelta"].values, dtype="int64") if v6 else None),
            "daily_vals": (np.asarray(ds["prediction_timedelta_daily"].values, dtype="int64") if vd else None),
        }
        inits = ds["init_time"].values
        n = 0
        for i in range(len(inits)):
            out6 = {v: np.asarray(ds[v].isel(init_time=i).values, dtype="float32") for v in v6}
            outd = {v: np.asarray(ds[v].isel(init_time=i).values, dtype="float32") for v in vd}
            n += _write_h5(out6=out6, outd=outd, schema=schema,
                           init_dt=inits[i], model=model, h5_root=h5_root)
        return n
    finally:
        ds.close()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(a: argparse.Namespace) -> None:
    if not a.source_kind or not a.source_dir:
        raise SystemExit("--source-kind and --source-dir are required (forward mode)")
    years = _parse_years(a.years)
    out_root = Path(a.out_root)
    h5_root = out_root / "h5"
    zarr_root = out_root / "zarr"
    lead_hours = tuple(a.lead_hours)
    lead_days = tuple(a.lead_days)
    generator = f"{SCRIPT_REL_PATH}@{a.commit}" if a.commit else SCRIPT_REL_PATH

    all_inits = _list_init_files(a.source_dir, a.source_kind)
    all_inits = [(dt, p) for dt, p in all_inits if dt.year in years]
    if a.limit_inits:
        all_inits = all_inits[: a.limit_inits]
    if not all_inits:
        raise SystemExit(f"no inits for {a.model} in years {a.years}")
    logger.info("%s: %d inits in %s across %d source dir(s)",
                a.model, len(all_inits), a.years, len(a.source_dir))

    by_year: dict[int, list] = {}
    for dt, p in all_inits:
        by_year.setdefault(dt.year, []).append((dt, p))

    n_workers = a.n_workers or max(1, (os.cpu_count() or 8) - 2)
    grand_h5 = 0
    for year in sorted(by_year):
        inits = sorted(by_year[year], key=lambda t: (t[0].month, t[0].day, t[0].hour))
        init_dts = [dt for dt, _ in inits]
        schema = _probe(inits[0][1], a.source_kind, lead_hours, lead_days)
        store_path = zarr_root / a.model / f"{year}.zarr"
        if a.formats in ("both", "zarr"):
            if store_path.exists() and not a.overwrite:
                raise FileExistsError(f"{store_path} exists; pass --overwrite")
            _create_year_store(store_path=store_path, init_dts=init_dts, schema=schema,
                               model=a.model, source_dataset=a.source_dataset,
                               lead_hours=lead_hours, lead_days=lead_days,
                               generator=generator, overwrite=a.overwrite)
        tasks = [{
            "path": str(p), "kind": a.source_kind, "model": a.model,
            "lead_hours": list(lead_hours), "lead_days": list(lead_days),
            "formats": a.formats, "h5_root": str(h5_root), "store_path": str(store_path),
            "index": i, "exp_vars_6h": schema["vars_6h"], "exp_vars_daily": schema["vars_daily"],
        } for i, (dt, p) in enumerate(inits)]

        year_h5 = 0
        finite_min = 1.0
        if n_workers <= 1:
            for t in tasks:
                r = _process_init(t)
                year_h5 += r["n_h5"]
                finite_min = min(finite_min, r["finite"])
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_process_init, t) for t in tasks]
                for fut in as_completed(futs):
                    r = fut.result()
                    year_h5 += r["n_h5"]
                    finite_min = min(finite_min, r["finite"])
        grand_h5 += year_h5

        if a.formats in ("both", "zarr"):
            try:
                import zarr
                zarr.consolidate_metadata(str(store_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("consolidate_metadata failed (non-fatal): %s", exc)
        logger.info("year %d: %d inits, %d h5 files, min finite=%.4f%s",
                    year, len(inits), year_h5, finite_min,
                    f" -> {store_path}" if a.formats != 'h5' else "")

    logger.info("DONE %s: %d inits total, %d h5 files, zarr under %s",
                a.model, len(all_inits), grand_h5, zarr_root / a.model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Output model archive name.")
    p.add_argument("--source-kind", choices=["zarr", "nc"], default=None)
    p.add_argument("--source-dir", action="append", default=None,
                   help="Source dir of init_* files; repeat to merge (date-disjoint).")
    p.add_argument("--expand-zarr", type=Path, default=None,
                   help="REVERSE MODE: a Format-2 {year}.zarr store to expand into "
                        "per-sample h5 under {out-root}/h5/{model}/ (run on Derecho).")
    p.add_argument("--out-root", required=True,
                   help="Output root; writes {root}/h5/... and {root}/zarr/...")
    p.add_argument("--years", default="2000-2024", help="Init-year filter, e.g. 2000-2024.")
    p.add_argument("--lead-hours", type=int, nargs=2, default=list(DEFAULT_LEAD_HOURS),
                   metavar=("LO", "HI"), help="Closed 6h-axis window in hours.")
    p.add_argument("--lead-days", type=int, nargs=2, default=list(DEFAULT_LEAD_DAYS),
                   metavar=("LO", "HI"), help="Closed daily-axis window in days.")
    p.add_argument("--formats", choices=["both", "h5", "zarr"], default="both")
    p.add_argument("--source-dataset", default="", help="Pass-through attr.")
    p.add_argument("--commit", default="", help="Git commit for the generator attr.")
    p.add_argument("--n-workers", type=int, default=0, help="0 = cpu_count-2.")
    p.add_argument("--limit-inits", type=int, default=0, help="Smoke-test cap.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.expand_zarr:
        n = expand_zarr_to_h5(a.expand_zarr, a.model, Path(a.out_root) / "h5")
        logger.info("expanded %s -> %d h5 files under %s/h5/%s",
                    a.expand_zarr, n, a.out_root, a.model)
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
