#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert daily precipitation per-day HDF5 files into one yearly Zarr store.

Two daily-precip sources are handled through the same code path (selected by
``--dataset-name`` / coord source):

``imerg``
    Global 180x360 daily ``total_precipitation_24hr`` (mm) on the ERA5 grid.
    The source H5 rows/cols are ASSUMED to already be in ERA5 order (lat
    89.5 -> -89.5 N->S, lon 0 -> 359 E); the exact lat/lon coords are copied
    from an ERA5 reference store (``--ref-store``) so IMERG aligns cell-for-cell
    with ERA5.  The assumption is recorded in the store attr
    ``grid_assumption`` and a cheap tropics-vs-poles sanity check is logged (and
    stored in ``grid_orientation_check``); a loud warning is emitted if it looks
    flipped.

``imd``
    India-region 33x35 daily rainfall on its native grid.  ``RAINFALL`` is
    renamed to ``total_precipitation_24hr``; lat/lon are taken verbatim from
    ``--coords-nc`` (``coordinates.nc``).  NaN (ocean / no-station cells,
    ~69% of the domain) is preserved as the store fill value.

Output store (shared ai-rossby ``ClimateZarrStoreLayout`` schema)
-----------------------------------------------------------------
* zarr v3, consolidated, float32, fill value ``NaN``.
* One data variable ``total_precipitation_24hr`` dimensioned ``(time, lat, lon)``,
  chunked ``(1, n_lat, n_lon)`` -- one chunk per day (a full field).
* Coords: ``time`` (cftime standard calendar, encoded ``hours since
  {year}-01-01 00:00:00``, built from each file's actual ``input/time`` so that
  NON-CONTIGUOUS / gappy date series are handled), ``lat`` (deg N), ``lon``
  (deg E).  No pressure_level / sigma_level (precip is a diagnostic surface
  field).
* Store attrs: the six ClimateZarr variable-group lists (only
  ``diagnostic_variables`` is populated -- ``["total_precipitation_24hr"]`` --
  all others empty), ``calendar`` (``"standard"``), ``data_timedelta_hours``
  (``24`` -- daily), plus the ERA5-style ``era5_zarr_schema_version`` /
  ``year_index`` / ``sample_range`` attrs.

Login-node safety
-----------------
This module imports only plain ``h5py`` / ``xarray`` / ``zarr`` / ``numpy`` /
``cftime`` -- it deliberately never imports ``physicsnemo`` (importing that on a
cluster login node can core-dump on CUDA / Warp init).  Run it directly on a
login node or as a CPU job.

Usage
-----
::

    # IMERG (global, ERA5 grid copied from a reference store)
    python tools/data/precip/h5_to_zarr.py \\
        --input-dir /glade/.../laude_data/IMERG_1p0 \\
        --source-var total_precipitation_24hr --out-var total_precipitation_24hr \\
        --dataset-name imerg --year 2000 --units "mm/day" \\
        --ref-store /glade/.../physicsnemo-zarr/era5/2000.zarr \\
        --out-store /glade/.../physicsnemo-zarr/imerg/2000.zarr

    # IMD (native India grid from coordinates.nc, RAINFALL -> total_precipitation_24hr)
    python tools/data/precip/h5_to_zarr.py \\
        --input-dir /glade/.../laude_data/IMD_1p0 \\
        --source-var RAINFALL --out-var total_precipitation_24hr \\
        --dataset-name imd --year 1901 --units "mm/day" \\
        --coords-nc /glade/.../laude_data/IMD_1p0/coordinates.nc \\
        --out-store /glade/.../physicsnemo-zarr/imd/1901.zarr
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import cftime
import h5py
import numpy as np
import xarray as xr

logger = logging.getLogger("precip_h5_to_zarr")

# Bumped independently of the ERA5 converter but kept on the same attr name so
# the shared ClimateZarr reader / registry tooling treats these like ERA5.
ERA5_ZARR_SCHEMA_VERSION = "1.0"
CALENDAR = "standard"
DATA_TIMEDELTA_HOURS = 24  # daily

# Filenames are YYYYMMDDHH.h5 (HH == 00 for these daily archives).
_FNAME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})\.h5$")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert daily precip per-day HDF5 files to one yearly Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dir of per-day YYYYMMDDHH.h5 files.",
    )
    p.add_argument(
        "--source-var",
        required=True,
        help="H5 dataset name under the 'input' group (e.g. "
        "total_precipitation_24hr or RAINFALL).",
    )
    p.add_argument(
        "--out-var",
        default="total_precipitation_24hr",
        help="Output variable name in the Zarr store.",
    )
    p.add_argument("--out-store", type=Path, required=True, help="Output Zarr store path.")
    p.add_argument("--year", type=int, required=True, help="Year to convert.")
    p.add_argument(
        "--coords-nc",
        type=Path,
        default=None,
        help="NetCDF with lat/lon coords (IMD native grid). If omitted, lat/lon "
        "are copied from --ref-store.",
    )
    p.add_argument(
        "--ref-store",
        type=Path,
        default=None,
        help="ERA5 Zarr store to copy lat/lon from (IMERG global grid).",
    )
    p.add_argument(
        "--dataset-name",
        choices=["imerg", "imd"],
        required=True,
        help="Source dataset id (controls the grid-orientation sanity check + attrs).",
    )
    p.add_argument("--units", default="mm/day", help="Units attr for the output variable.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output store.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _decode_time(time_value) -> cftime.DatetimeGregorian:
    """Parse the per-file time scalar into a cftime DatetimeGregorian (standard).

    Source H5 stores a ``numpy.datetime64`` bytes blob like
    ``b'2000-06-01T00:00:00.000000000'``.
    """
    if isinstance(time_value, bytes):
        s = time_value.decode()
    else:
        s = str(time_value)
    s = s.split(".")[0]  # drop fractional seconds
    if "T" in s:
        date_part, time_part = s.split("T", 1)
    else:
        date_part, time_part = (s.split(" ", 1) + ["00:00:00"])[:2]
    year, month, day = (int(x) for x in date_part.split("-"))
    hms = time_part.replace("Z", "").split(":")
    h = int(hms[0]) if len(hms) > 0 and hms[0] else 0
    m = int(hms[1]) if len(hms) > 1 and hms[1] else 0
    sec = int(hms[2]) if len(hms) > 2 and hms[2] else 0
    return cftime.DatetimeGregorian(year, month, day, h, m, sec)


def _list_year_files(input_dir: Path, year: int) -> list[tuple[str, Path]]:
    """Enumerate per-day H5 files whose filename date-token is in ``year``.

    Returns ``[(YYYYMMDDHH, path), ...]`` sorted by the filename token; the
    authoritative chronological order is re-derived from the real timestamps in
    :func:`read_year`.
    """
    matches: list[tuple[str, Path]] = []
    for p in sorted(input_dir.iterdir()):
        m = _FNAME_RE.match(p.name)
        if m and int(m.group(1)) == year:
            matches.append((m.group(0)[:-3], p))  # strip ".h5"
    matches.sort(key=lambda t: t[0])
    return matches


def _resolve_coords(args: argparse.Namespace, data_shape: tuple[int, int]) -> dict:
    """Determine (lat, lon) for the output store from --coords-nc or --ref-store."""
    n_lat, n_lon = data_shape
    if args.coords_nc is not None:
        cds = xr.open_dataset(args.coords_nc)
        try:
            lat = np.asarray(cds["lat"].values, dtype="float32")
            lon = np.asarray(cds["lon"].values, dtype="float32")
        finally:
            cds.close()
        src = f"coords-nc:{args.coords_nc}"
    elif args.ref_store is not None:
        rs = xr.open_zarr(args.ref_store, consolidated=True, decode_times=False)
        try:
            lat = np.asarray(rs["lat"].values, dtype="float32")
            lon = np.asarray(rs["lon"].values, dtype="float32")
        finally:
            rs.close()
        src = f"ref-store:{args.ref_store}"
    else:
        raise ValueError("one of --coords-nc or --ref-store is required")

    if lat.size != n_lat or lon.size != n_lon:
        raise ValueError(
            f"coord grid from {src} is ({lat.size}, {lon.size}) but data field is "
            f"({n_lat}, {n_lon}) -- grid mismatch; refusing to misalign."
        )
    return {"lat": lat, "lon": lon, "coord_source": src}


def read_year(input_dir: Path, year: int, source_var: str) -> dict:
    """Read all per-day files for ``year``: stack chronologically to (time, lat, lon)."""
    files = _list_year_files(input_dir, year)
    if not files:
        raise FileNotFoundError(f"no YYYYMMDDHH.h5 files for year {year} in {input_dir}")

    records: list[tuple[cftime.DatetimeGregorian, np.ndarray, str]] = []
    for token, path in files:
        with h5py.File(path, "r") as f:
            g = f["input"]
            if source_var not in g:
                raise KeyError(
                    f"{path.name}: 'input/{source_var}' not found (have {list(g.keys())})"
                )
            arr = np.asarray(g[source_var][:], dtype="float32")
            t = _decode_time(g["time"][()])
        if int(t.year) != int(year):
            logger.warning(
                "%s: file time %s year != --year %d; keeping (coord uses real time)",
                path.name, t, year,
            )
        records.append((t, arr, path.name))

    # Authoritative chronological order from the real timestamps (handles gaps
    # / any out-of-order filenames without assuming a contiguous daily series).
    records.sort(key=lambda r: (r[0].year, r[0].month, r[0].day, r[0].hour, r[0].minute))

    times = [r[0] for r in records]
    # Duplicate-timestamp guard (loud, non-fatal).
    seen = set()
    dups = []
    for t in times:
        key = (t.year, t.month, t.day, t.hour)
        if key in seen:
            dups.append(t)
        seen.add(key)
    if dups:
        logger.warning("year %d: %d duplicate timestamps present, e.g. %s",
                       year, len(dups), dups[0])

    data = np.stack([r[1] for r in records], axis=0)  # (time, lat, lon)
    n_lat, n_lon = data.shape[1], data.shape[2]
    logger.info(
        "year %d: %d files, field %dx%d, time %s .. %s",
        year, len(records), n_lat, n_lon, times[0], times[-1],
    )
    return {"times": times, "data": data, "n_files": len(records)}


def grid_orientation_check(data: np.ndarray, lat: np.ndarray) -> dict:
    """Cheap N->S orientation sanity check for a global precip field.

    Compares the time-mean precip in the deep tropics (|lat| < 15) against the
    polar bands (|lat| > 60).  Physical fields have MUCH more precip in the
    tropics (ITCZ) than at the poles, so ``tropics_mean > poles_mean`` should
    hold if rows line up with the lat coord.  Returns a dict of the computed
    means; the caller decides whether to warn.
    """
    field = np.nanmean(data, axis=0)  # (lat, lon) time-mean
    tropics = np.abs(lat) < 15.0
    poles = np.abs(lat) > 60.0
    trop_mean = float(np.nanmean(field[tropics, :])) if tropics.any() else float("nan")
    pole_mean = float(np.nanmean(field[poles, :])) if poles.any() else float("nan")
    global_mean = float(np.nanmean(field))
    looks_ok = np.isfinite(global_mean) and global_mean > 0.0 and (
        not (np.isfinite(trop_mean) and np.isfinite(pole_mean)) or trop_mean > pole_mean
    )
    return {
        "global_mean": global_mean,
        "tropics_mean": trop_mean,
        "poles_mean": pole_mean,
        "looks_ok": bool(looks_ok),
    }


def convert(args: argparse.Namespace) -> dict:
    out_store = Path(args.out_store)
    if out_store.exists() and not args.overwrite:
        raise FileExistsError(f"{out_store} exists; pass --overwrite to replace")

    year_data = read_year(args.input_dir, args.year, args.source_var)
    data = year_data["data"]
    times = year_data["times"]
    n_time, n_lat, n_lon = data.shape

    coords_res = _resolve_coords(args, (n_lat, n_lon))
    lat = coords_res["lat"]
    lon = coords_res["lon"]

    attrs: dict = {
        "era5_zarr_schema_version": ERA5_ZARR_SCHEMA_VERSION,
        "calendar": CALENDAR,
        "data_timedelta_hours": int(DATA_TIMEDELTA_HOURS),
        "dataset_name": args.dataset_name,
        "source_input_dir": str(args.input_dir),
        "coord_source": coords_res["coord_source"],
        # Six ClimateZarr variable-group lists -- precip is diagnostic-only.
        "surface_variables": [],
        "constant_boundary_variables": [],
        "varying_boundary_variables": [],
        "diagnostic_variables": [args.out_var],
        "pressure_upper_air_variables": [],
        "sigma_upper_air_variables": [],
        "year_index": int(args.year),
        # Half-open [lo, hi) index range of samples in this store (ERA5-style).
        "sample_range": [0, int(n_time)],
    }

    # Grid-orientation sanity check + assumption record (IMERG only -- IMD keeps
    # its native regional grid verbatim, no N->S assumption to test).
    orient: Optional[dict] = None
    if args.dataset_name == "imerg":
        attrs["grid_assumption"] = (
            "IMERG source rows/cols assumed already in ERA5 order (lat 89.5->-89.5 "
            "N->S, lon 0->359 E); lat/lon copied from --ref-store to align "
            "cell-for-cell with ERA5."
        )
        orient = grid_orientation_check(data, lat)
        attrs["grid_orientation_check"] = (
            f"global_mean={orient['global_mean']:.4f} mm/day, "
            f"tropics(|lat|<15)_mean={orient['tropics_mean']:.4f}, "
            f"poles(|lat|>60)_mean={orient['poles_mean']:.4f}, "
            f"looks_ok={orient['looks_ok']}"
        )
        if orient["looks_ok"]:
            logger.info(
                "IMERG grid orientation OK: %s", attrs["grid_orientation_check"]
            )
        else:
            logger.warning(
                "IMERG grid orientation SANITY CHECK FAILED / POSSIBLY FLIPPED: %s "
                "-- tropics mean should exceed poles mean; investigate before use.",
                attrs["grid_orientation_check"],
            )

    ds = xr.Dataset(
        {args.out_var: (("time", "lat", "lon"), data)},
        coords={
            "time": ("time", times),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs=attrs,
    )
    ds[args.out_var].attrs["units"] = args.units

    chunks = (1, n_lat, n_lon)
    encoding = {
        args.out_var: {"chunks": chunks, "dtype": "float32", "_FillValue": np.float32(np.nan)},
        "time": {
            "units": f"hours since {args.year}-01-01 00:00:00",
            "calendar": CALENDAR,
            "dtype": "int64",
        },
    }

    out_store.parent.mkdir(parents=True, exist_ok=True)
    logger.info("writing %s (%d time x %d lat x %d lon)", out_store, n_time, n_lat, n_lon)
    ds.to_zarr(
        out_store,
        mode="w" if args.overwrite else "w-",
        zarr_format=3,
        consolidated=True,
        encoding=encoding,
    )
    logger.info("done: %s", out_store)

    summary = {
        "out_store": str(out_store),
        "n_time": int(n_time),
        "n_lat": int(n_lat),
        "n_lon": int(n_lon),
        "n_files": int(year_data["n_files"]),
        "orientation": orient,
    }
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    convert(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
