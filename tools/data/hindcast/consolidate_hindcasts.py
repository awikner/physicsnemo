#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate per-init hindcast outputs into one yearly Zarr v3 store.

Two hindcast producers write **per-init** files; this script merges one
init-year's worth into a single consolidated store::

    {out_root}/{model}/{YYYY}.zarr

with ``model`` in ``{pangu_s2s, sfno_era5, archesweather_era5}`` and ``YYYY``
the init year (2000-2024).  See
``docs/dev/context/plan-2026-07-hindcasts-archesweather.md`` §4 for the full
spec this mirrors.

Login-node safety
-----------------
This module imports **only** plain ``xarray`` / ``zarr`` / ``numpy`` /
``cftime`` / ``dask`` — it deliberately never imports ``physicsnemo`` because
importing that package on a cluster login node can core-dump (CUDA / Warp
init).  Run it directly on a login node or as a CPU job.

Output store (zarr v3, consolidated)
------------------------------------
* ``zarr_format`` 3, consolidated (a root ``zarr.json`` is written), codecs
  = ``bytes(little-endian)`` + ``zstd(level 0)`` (the zarr-python 3 defaults,
  which match the ERA5 training stores), dtype float32, fill value ``NaN``.
* Dims: surface & diagnostic variables ``(init_time, lead_time, lat, lon)``;
  upper-air variables ``(init_time, lead_time, pressure_level, lat, lon)``.
* Chunking ``(1, n_lead, [n_levels,] n_lat, n_lon)`` — one chunk per
  (variable, init).
* Coords: ``init_time`` (cftime standard calendar, encoded hours-since-
  ``YYYY-01-01``), ``lead_time`` = int ``[0..n_lead-1]`` with attr units
  ``"days"``, ``lat`` / ``lon`` / ``pressure_level`` (from a reference
  training store when given, else from the input files).
* ``lead 0`` is the ERA5 initial condition (frame 0 of each input); leads
  ``1..N`` are the forecasts.

Input adapters
--------------
``--format pangu``
    Per-init NetCDF named ``pangu_plasim_{run}_24h_{N}step_{YYYYMMDDHH}.nc``.
    Dims ``(time=N+1, plev, lat, lon)``; ``time`` frame 0 is the IC.  Variable
    names are **detected** from each file and mapped to the canonical ERA5
    names via an alias table (short names like ``t2m`` / ``T`` are handled).
    The init datetime is parsed from the ``YYYYMMDDHH`` token in the filename.

``--format ai_rossby``
    Per-IC output of ``examples/weather/ai_rossby/inference.py`` via
    ``async_writer.py`` (``_build_per_ic_dataset``).  Layout::

        pred_surface   (ensemble, frame, surface_var, lat, lon)
        pred_upper_air (ensemble, frame, upper_air_var, level, lat, lon)
        pred_diagnostic(ensemble, frame, diag_var, lat, lon)   [optional]

    (``summary`` mode writes ``pred_*_mean`` / ``pred_*_std`` instead — the
    ``_mean`` fields are used.)  The ``ensemble`` (and, if present, ``ic``)
    axis is squeezed.  **Frame indexing**: that builder sets the store attr
    ``frame_zero_is_ic = 1`` and uses ``n_frames = max_step + 1`` with frame 0
    the IC, so the lead-0 IC is already on disk and no prepend is needed.  The
    ``--prepend-ic`` flag (and auto-detection from ``frame_zero_is_ic`` / the
    frame coord starting at 1) covers the alternate case of a producer that
    emits forecast steps only — a NaN lead-0 slot is then prepended and a
    loud warning logged (the IC would have to be sourced separately).

Feb-29
------
Any init whose datetime is Feb 29 is dropped (the legacy Pangu InferFilter
emits it in leap years).  Target is 95 inits/year.

Usage
-----
::

    python tools/data/hindcast/consolidate_hindcasts.py \\
        --format pangu \\
        --input-dir /scratch/.../pangu_s2s_hindcasts/.../2000/predictions \\
        --out-store /scratch/.../hindcasts/pangu_s2s/2000.zarr \\
        --model pangu_s2s --year 2000 \\
        --ref-store /scratch/.../era5/2000.zarr \\
        --checkpoint "S2S/2000/best_ckpt.tar" --source-dataset pangu_s2s \\
        --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import cftime
import numpy as np
import xarray as xr

logger = logging.getLogger("consolidate_hindcasts")

HINDCAST_SCHEMA_VERSION = "1.0"
SCRIPT_REL_PATH = "tools/data/hindcast/consolidate_hindcasts.py"
INIT_DAYS = (1, 5, 9, 13, 17, 21, 25, 29)
LEAD_TIME_HOURS = 24
INIT_SCHEDULE_STR = "monthly days 1,5,9,13,17,21,25,29 (no Feb 29) 00Z"
CALENDAR = "standard"

# --------------------------------------------------------------------------- #
# Variable-name detection (canonical -> accepted aliases, matched lowercase).
# The Pangu NetCDF writer's exact var names are not fixed across archives, so
# we detect + map rather than hardcode.  Classification into surface vs
# upper-air is done first by dimensionality (presence of a pressure-level
# dim); the alias tables then resolve the canonical name within the group.
# --------------------------------------------------------------------------- #
SURFACE_ALIASES: dict[str, list[str]] = {
    "2m_temperature": ["2m_temperature", "t2m", "2t", "tas", "temperature_2m", "air_temperature_2m"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10", "10u", "uas", "10m_u_wind"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10", "10v", "vas", "10m_v_wind"],
    "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl", "mslp", "air_pressure_at_mean_sea_level"],
    "surface_pressure": ["surface_pressure", "sp", "ps", "surface_air_pressure", "pres"],
}
DIAGNOSTIC_ALIASES: dict[str, list[str]] = {
    "total_precipitation_24hr": [
        "total_precipitation_24hr", "total_precipitation", "tp", "precip", "pr", "precipitation",
    ],
    "mean_top_net_long_wave_radiation_flux": [
        "mean_top_net_long_wave_radiation_flux", "mtnlwrf", "ttr", "olr",
        "top_net_long_wave_radiation", "mean_top_net_lw_radiation_flux",
    ],
}
UPPER_AIR_ALIASES: dict[str, list[str]] = {
    "temperature": ["temperature", "t", "ta", "air_temperature"],
    "u_component_of_wind": ["u_component_of_wind", "u", "ua", "eastward_wind"],
    "v_component_of_wind": ["v_component_of_wind", "v", "va", "northward_wind"],
    "specific_humidity": ["specific_humidity", "q", "hus", "humidity"],
    "geopotential": ["geopotential", "z", "zg", "gh", "geopotential_height"],
}


def _invert(aliases: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for canon, names in aliases.items():
        for n in names:
            out[n.lower()] = canon
    return out


SURF_LOOKUP = _invert(SURFACE_ALIASES)
DIAG_LOOKUP = _invert(DIAGNOSTIC_ALIASES)
UPPER_LOOKUP = _invert(UPPER_AIR_ALIASES)

# Candidate dim names (case-insensitive).
_TIME_DIMS = ("time", "step", "lead_time", "frame", "t", "valid_time")
_LEVEL_DIMS = ("plev", "pressure_level", "level", "lev", "isobaricinhpa", "pressure", "levels", "p")
_LAT_DIMS = ("lat", "latitude", "y")
_LON_DIMS = ("lon", "longitude", "x")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _find_dim(dims: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(d).lower(): str(d) for d in dims}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _map_name(name: str, lookup: dict[str, str]) -> Optional[str]:
    return lookup.get(str(name).lower())


def _order_group(present: dict[str, np.ndarray], canonical_order: Sequence[str]) -> dict[str, np.ndarray]:
    """Return ``present`` reordered by ``canonical_order`` (extras appended)."""
    out: dict[str, np.ndarray] = {}
    for c in canonical_order:
        if c in present:
            out[c] = present[c]
    for k in present:
        if k not in out:
            out[k] = present[k]
    return out


def _parse_dt_string(s: str) -> cftime.DatetimeGregorian:
    """Parse a datetime from a string (ISO-ish, YYYYMMDDTHHMM, or YYYYMMDDHH)."""
    s = str(s).strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):?(\d{2})?", s)
    if m:
        y, mo, d, h = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        mi = int(m.group(5)) if m.group(5) else 0
        return cftime.DatetimeGregorian(y, mo, d, h, mi)
    m = re.search(r"(\d{8})T(\d{2})(\d{2})", s)
    if m:
        ds8, h, mi = m.group(1), int(m.group(2)), int(m.group(3))
        return cftime.DatetimeGregorian(int(ds8[0:4]), int(ds8[4:6]), int(ds8[6:8]), h, mi)
    m = re.search(r"(\d{10})", s)
    if m:
        t = m.group(1)
        return cftime.DatetimeGregorian(int(t[0:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]))
    raise ValueError(f"could not parse a datetime from {s!r}")


def _is_feb29(dt) -> bool:
    return int(dt.month) == 2 and int(dt.day) == 29


def _dt_key(dt) -> tuple[int, int, int, int]:
    return (int(dt.year), int(dt.month), int(dt.day), int(dt.hour))


def _coord_values(ds: xr.Dataset, dim: Optional[str], *, kind: str) -> Optional[np.ndarray]:
    """Read a coord array for ``dim`` from ``ds``, synthesizing lat/lon if absent."""
    if dim is None:
        return None
    if dim in ds.coords:
        return np.asarray(ds[dim].values)
    if dim in ds.variables:
        return np.asarray(ds[dim].values)
    n = int(ds.sizes[dim])
    if kind == "lat":
        return np.linspace(89.5, -89.5, n, dtype="float32")
    if kind == "lon":
        return np.linspace(0.0, 360.0 * (n - 1) / n, n, dtype="float32")
    return None  # levels cannot be synthesized


# --------------------------------------------------------------------------- #
# Pangu per-init NetCDF adapter
# --------------------------------------------------------------------------- #
def _parse_pangu_init_dt(name: str) -> cftime.DatetimeGregorian:
    m = re.search(r"_(\d{10})\.nc$", name)
    if not m:
        # fall back to the last 10-digit run in the name
        groups = re.findall(r"\d{10}", name)
        if not groups:
            raise ValueError(f"no YYYYMMDDHH token in Pangu filename {name!r}")
        t = groups[-1]
    else:
        t = m.group(1)
    return cftime.DatetimeGregorian(int(t[0:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]))


def load_pangu_init(path: Path) -> dict:
    """Load one Pangu per-init NetCDF into the common intermediate dict."""
    ds = xr.open_dataset(path, decode_times=False)
    try:
        lat_dim = _find_dim(ds.dims, _LAT_DIMS)
        lon_dim = _find_dim(ds.dims, _LON_DIMS)
        time_dim = _find_dim(ds.dims, _TIME_DIMS)
        lev_dim = _find_dim(ds.dims, _LEVEL_DIMS)
        if lat_dim is None or lon_dim is None or time_dim is None:
            raise ValueError(
                f"{path.name}: could not find lat/lon/time dims among {list(ds.dims)}"
            )

        surface: dict[str, np.ndarray] = {}
        diagnostic: dict[str, np.ndarray] = {}
        upper: dict[str, np.ndarray] = {}
        unmapped: list[str] = []

        for vname, da in ds.data_vars.items():
            dims = da.dims
            if lat_dim not in dims or lon_dim not in dims:
                continue  # not a spatial field
            if lev_dim is not None and lev_dim in dims:
                canon = _map_name(vname, UPPER_LOOKUP)
                if canon is None:
                    unmapped.append(str(vname))
                    canon = str(vname)
                upper[canon] = (
                    da.transpose(time_dim, lev_dim, lat_dim, lon_dim).values.astype("float32")
                )
            else:
                canon = _map_name(vname, DIAG_LOOKUP)
                if canon is not None:
                    diagnostic[canon] = (
                        da.transpose(time_dim, lat_dim, lon_dim).values.astype("float32")
                    )
                    continue
                canon = _map_name(vname, SURF_LOOKUP)
                if canon is None:
                    unmapped.append(str(vname))
                    canon = str(vname)
                surface[canon] = (
                    da.transpose(time_dim, lat_dim, lon_dim).values.astype("float32")
                )

        lat = _coord_values(ds, lat_dim, kind="lat")
        lon = _coord_values(ds, lon_dim, kind="lon")
        plev = _coord_values(ds, lev_dim, kind="level")
    finally:
        ds.close()

    if unmapped:
        logger.warning("%s: unmapped variable(s) kept verbatim: %s", path.name, unmapped)
    if plev is not None:
        plev = np.asarray(plev, dtype="float64")
        if plev.size and np.nanmax(plev) > 2000.0:
            logger.info("%s: pressure levels look like Pa (max=%.1f); converting to hPa",
                        path.name, float(np.nanmax(plev)))
            plev = plev / 100.0
        plev = plev.astype("float32")

    surface = _order_group(surface, list(SURFACE_ALIASES))
    diagnostic = _order_group(diagnostic, list(DIAGNOSTIC_ALIASES))
    upper = _order_group(upper, list(UPPER_AIR_ALIASES))

    return {
        "init_dt": _parse_pangu_init_dt(path.name),
        "surface": surface,
        "diagnostic": diagnostic,
        "upper": upper,
        "lat": None if lat is None else np.asarray(lat, dtype="float32"),
        "lon": None if lon is None else np.asarray(lon, dtype="float32"),
        "plev": plev,
        "source": str(path),
    }


# --------------------------------------------------------------------------- #
# ai_rossby per-IC adapter (async_writer.py / _build_per_ic_dataset)
# --------------------------------------------------------------------------- #
def _ai_rossby_init_dt(ds: xr.Dataset, path: Path, frame_dim: str) -> cftime.DatetimeGregorian:
    ic_attr = ds.attrs.get("ic_time")
    if ic_attr:
        try:
            return _parse_dt_string(str(ic_attr))
        except ValueError:
            pass
    if "ic_time" in ds.coords:
        try:
            return _parse_dt_string(str(ds["ic_time"].values))
        except ValueError:
            pass
    if "time" in ds.coords and frame_dim in ds["time"].dims:
        try:
            return _parse_dt_string(str(np.asarray(ds["time"].values).ravel()[0]))
        except ValueError:
            pass
    return _parse_dt_string(path.name)


def _open_input(path: Path) -> xr.Dataset:
    if str(path).endswith(".zarr") or path.is_dir():
        return xr.open_zarr(path, consolidated=True, decode_timedelta=False)
    return xr.open_dataset(path, decode_timedelta=False)


def load_ai_rossby_init(path: Path, *, prepend_ic: bool = False) -> dict:
    """Load one ai_rossby per-IC store/file into the common intermediate dict."""
    ds = _open_input(path)
    try:
        for squeeze_dim in ("ic", "ensemble"):
            if squeeze_dim in ds.dims:
                if int(ds.sizes[squeeze_dim]) > 1:
                    logger.warning(
                        "%s: %s size=%d > 1; taking index 0 (deterministic member)",
                        Path(path).name, squeeze_dim, int(ds.sizes[squeeze_dim]),
                    )
                ds = ds.isel({squeeze_dim: 0}, drop=True)

        surf_da = ds.get("pred_surface", ds.get("pred_surface_mean"))
        upper_da = ds.get("pred_upper_air", ds.get("pred_upper_air_mean"))
        diag_da = ds.get("pred_diagnostic", ds.get("pred_diagnostic_mean"))
        if surf_da is None and upper_da is None:
            raise ValueError(
                f"{Path(path).name}: no pred_surface/pred_upper_air variables found "
                f"(have {list(ds.data_vars)})"
            )

        frame_dim = _find_dim(surf_da.dims if surf_da is not None else upper_da.dims, _TIME_DIMS)
        if frame_dim is None:
            raise ValueError(f"{Path(path).name}: no frame/step/time dim found")
        lat_dim = _find_dim(ds.dims, _LAT_DIMS)
        lon_dim = _find_dim(ds.dims, _LON_DIMS)

        # Frame-0 = IC detection.  _build_per_ic_dataset stamps
        # frame_zero_is_ic=1 and starts the frame coord at 0.
        frame_zero_is_ic = int(ds.attrs.get("frame_zero_is_ic", 0)) == 1
        frame_starts_at_zero = True
        if frame_dim in ds.coords:
            fv = np.asarray(ds[frame_dim].values)
            if fv.size:
                frame_starts_at_zero = int(fv.min()) == 0
        has_ic = frame_zero_is_ic or frame_starts_at_zero

        surface: dict[str, np.ndarray] = {}
        diagnostic: dict[str, np.ndarray] = {}
        upper: dict[str, np.ndarray] = {}

        if surf_da is not None:
            svar_dim = _find_dim(surf_da.dims, ("surface_var",)) or "surface_var"
            for name in [str(x) for x in ds[svar_dim].values]:
                surface[name] = (
                    surf_da.sel({svar_dim: name})
                    .transpose(frame_dim, lat_dim, lon_dim)
                    .values.astype("float32")
                )
        if diag_da is not None:
            dvar_dim = _find_dim(diag_da.dims, ("diag_var",)) or "diag_var"
            for name in [str(x) for x in ds[dvar_dim].values]:
                diagnostic[name] = (
                    diag_da.sel({dvar_dim: name})
                    .transpose(frame_dim, lat_dim, lon_dim)
                    .values.astype("float32")
                )
        plev = None
        if upper_da is not None:
            uvar_dim = _find_dim(upper_da.dims, ("upper_air_var",)) or "upper_air_var"
            lev_dim = _find_dim(upper_da.dims, _LEVEL_DIMS)
            for name in [str(x) for x in ds[uvar_dim].values]:
                upper[name] = (
                    upper_da.sel({uvar_dim: name})
                    .transpose(frame_dim, lev_dim, lat_dim, lon_dim)
                    .values.astype("float32")
                )
            if lev_dim is not None and lev_dim in ds.coords:
                plev = np.asarray(ds[lev_dim].values, dtype="float32")

        lat = _coord_values(ds, lat_dim, kind="lat")
        lon = _coord_values(ds, lon_dim, kind="lon")
        init_dt = _ai_rossby_init_dt(ds, Path(path), frame_dim)
    finally:
        ds.close()

    # Prepend a lead-0 IC slot if the producer emitted forecasts only.
    need_prepend = prepend_ic or (not has_ic)
    if need_prepend and has_ic and prepend_ic:
        logger.warning(
            "%s: --prepend-ic set but input already has a lead-0 IC frame "
            "(frame_zero_is_ic / frame starts at 0); NOT prepending.",
            Path(path).name,
        )
    elif need_prepend and not has_ic:
        logger.warning(
            "%s: input frame set starts at step 1 (no IC frame); prepending a "
            "NaN lead-0 slot. The lead-0 ERA5 IC is NOT present in this input "
            "and must be sourced separately if needed.",
            Path(path).name,
        )
        for grp in (surface, diagnostic, upper):
            for k, arr in list(grp.items()):
                pad = np.full((1,) + arr.shape[1:], np.nan, dtype="float32")
                grp[k] = np.concatenate([pad, arr], axis=0)

    return {
        "init_dt": init_dt,
        "surface": surface,
        "diagnostic": diagnostic,
        "upper": upper,
        "lat": None if lat is None else np.asarray(lat, dtype="float32"),
        "lon": None if lon is None else np.asarray(lon, dtype="float32"),
        "plev": plev,
        "source": str(path),
    }


# --------------------------------------------------------------------------- #
# Discovery / validation / coord resolution
# --------------------------------------------------------------------------- #
def _list_inputs(input_dir: Path, fmt: str) -> list[Path]:
    if fmt == "pangu":
        files = sorted(input_dir.glob("pangu_plasim_*.nc"))
        if not files:
            files = sorted(input_dir.glob("*.nc"))
        return files
    # ai_rossby
    files = sorted(input_dir.glob("*.zarr"))
    if not files:
        files = sorted(input_dir.glob("*.nc"))
    return files


def expected_schedule(year: int) -> list[tuple[int, int]]:
    """The (month, day) init slots expected for ``year`` (Feb 29 excluded)."""
    out: list[tuple[int, int]] = []
    for month in range(1, 13):
        for day in INIT_DAYS:
            if month == 2 and day == 29:
                continue  # never generated / always dropped
            try:
                cftime.DatetimeGregorian(year, month, day, 0)
            except (ValueError, TypeError):
                continue  # e.g. Feb 29 in a non-leap year
            out.append((month, day))
    return out


def validate_inits(inits: list[dict]) -> None:
    """Raise clearly if inits disagree on shapes / variable sets."""
    ref = inits[0]

    def _shapes(d: dict) -> dict:
        s = {}
        for grp in ("surface", "diagnostic", "upper"):
            for k, arr in d[grp].items():
                s[(grp, k)] = arr.shape
        return s

    ref_vars = {grp: sorted(ref[grp]) for grp in ("surface", "diagnostic", "upper")}
    ref_shapes = _shapes(ref)
    for it in inits[1:]:
        cur_vars = {grp: sorted(it[grp]) for grp in ("surface", "diagnostic", "upper")}
        if cur_vars != ref_vars:
            raise ValueError(
                f"variable sets differ:\n  {ref['source']}: {ref_vars}\n"
                f"  {it['source']}: {cur_vars}"
            )
        cur_shapes = _shapes(it)
        if cur_shapes != ref_shapes:
            diffs = {k: (ref_shapes[k], cur_shapes.get(k)) for k in ref_shapes
                     if ref_shapes[k] != cur_shapes.get(k)}
            raise ValueError(
                f"array shapes differ between {ref['source']} and {it['source']}: {diffs}"
            )


def resolve_coords(inits: list[dict], ref_store: Optional[Path]) -> dict:
    """Determine lat/lon/pressure_level for the output store.

    Input coords are authoritative for shape.  When ``--ref-store`` is given
    and its lat/lon (and pressure_level, if lengths match) align, its exact
    values are used so the hindcast grid matches the training grid bit-for-bit.
    """
    ref = inits[0]
    lat = ref["lat"]
    lon = ref["lon"]
    plev = ref["plev"]

    # Infer sizes from any array if a coord happens to be missing.
    n_lat = n_lon = n_lev = None
    for grp in ("surface", "diagnostic", "upper"):
        for arr in ref[grp].values():
            n_lat, n_lon = arr.shape[-2], arr.shape[-1]
            if grp == "upper":
                n_lev = arr.shape[-3]
    has_upper = len(ref["upper"]) > 0

    if lat is None and n_lat is not None:
        lat = np.linspace(89.5, -89.5, n_lat, dtype="float32")
    if lon is None and n_lon is not None:
        lon = np.linspace(0.0, 360.0 * (n_lon - 1) / n_lon, n_lon, dtype="float32")

    if ref_store is not None:
        try:
            rs = xr.open_zarr(ref_store, consolidated=True, decode_timedelta=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not open --ref-store %s: %s", ref_store, exc)
            rs = None
        if rs is not None:
            try:
                if "lat" in rs.coords and lat is not None and rs["lat"].size == lat.size:
                    lat = np.asarray(rs["lat"].values, dtype="float32")
                    logger.info("using lat coord from ref-store")
                if "lon" in rs.coords and lon is not None and rs["lon"].size == lon.size:
                    lon = np.asarray(rs["lon"].values, dtype="float32")
                    logger.info("using lon coord from ref-store")
                if ("pressure_level" in rs.coords and has_upper
                        and rs["pressure_level"].size == (n_lev or -1)):
                    plev = np.asarray(rs["pressure_level"].values, dtype="float32")
                    logger.info("using pressure_level coord from ref-store")
                elif has_upper and "pressure_level" in rs.coords:
                    logger.info(
                        "ref-store has %d pressure levels but inputs have %d; "
                        "keeping the input levels",
                        int(rs["pressure_level"].size), n_lev,
                    )
            finally:
                rs.close()

    if has_upper and plev is None:
        raise ValueError(
            "upper-air variables present but no pressure_level coord could be "
            "resolved from inputs or --ref-store; cannot build the store."
        )
    if lat is None or lon is None:
        raise ValueError("could not resolve lat/lon coords from inputs or --ref-store")
    return {"lat": lat, "lon": lon, "plev": plev}


# --------------------------------------------------------------------------- #
# Store writer (streaming per-init region writes to bound memory)
# --------------------------------------------------------------------------- #
def write_store(
    *,
    inits: list[dict],
    coords: dict,
    out_store: Path,
    year: int,
    model: str,
    checkpoint: str,
    source_dataset: str,
    boundary_clamped_inits: list[str],
    created: str,
    generator: str,
    overwrite: bool,
) -> dict:
    import dask.array as da  # local import — heavy, and only needed for writes

    ref = inits[0]
    surface_vars = list(ref["surface"])
    diag_vars = list(ref["diagnostic"])
    upper_vars = list(ref["upper"])

    n_init = len(inits)
    # n_lead from the frame axis of any array.
    probe = next(iter({**ref["surface"], **ref["diagnostic"], **ref["upper"]}.values()))
    n_lead = int(probe.shape[0])
    lat = np.asarray(coords["lat"], dtype="float32")
    lon = np.asarray(coords["lon"], dtype="float32")
    plev = coords["plev"]
    n_lat, n_lon = lat.size, lon.size
    n_lev = int(np.asarray(plev).size) if (upper_vars and plev is not None) else 0

    surf_shape = (n_init, n_lead, n_lat, n_lon)
    surf_chunks = (1, n_lead, n_lat, n_lon)
    up_shape = (n_init, n_lead, n_lev, n_lat, n_lon)
    up_chunks = (1, n_lead, n_lev, n_lat, n_lon)

    init_times = np.asarray([it["init_dt"] for it in inits])

    out_coords: dict = {
        "init_time": ("init_time", init_times),
        "lead_time": ("lead_time", np.arange(n_lead, dtype="int32")),
        "lat": ("lat", lat),
        "lon": ("lon", lon),
    }
    if upper_vars:
        out_coords["pressure_level"] = ("pressure_level", np.asarray(plev, dtype="float32"))

    data_vars: dict = {}
    for v in surface_vars + diag_vars:
        data_vars[v] = (("init_time", "lead_time", "lat", "lon"),
                        da.zeros(surf_shape, chunks=surf_chunks, dtype="float32"))
    for v in upper_vars:
        data_vars[v] = (("init_time", "lead_time", "pressure_level", "lat", "lon"),
                        da.zeros(up_shape, chunks=up_chunks, dtype="float32"))

    attrs = {
        "hindcast_schema_version": HINDCAST_SCHEMA_VERSION,
        "model": model,
        "checkpoint": checkpoint,
        "source_dataset": source_dataset,
        "init_schedule": INIT_SCHEDULE_STR,
        "lead_time_hours": LEAD_TIME_HOURS,
        "n_lead": n_lead,
        "calendar": CALENDAR,
        "boundary_clamped_inits": list(boundary_clamped_inits),
        "created": created,
        "generator": generator,
        "year_index": int(year),
        "surface_variables": surface_vars,
        "diagnostic_variables": diag_vars,
        "upper_air_variables": upper_vars,
        "pressure_upper_air_variables": upper_vars,
        "sigma_upper_air_variables": [],
        "constant_boundary_variables": [],
        "varying_boundary_variables": [],
    }

    ds = xr.Dataset(data_vars, coords=out_coords, attrs=attrs)
    ds["lead_time"].attrs.update({"units": "days", "long_name": "forecast lead time"})

    encoding: dict = {}
    for v in surface_vars + diag_vars:
        encoding[v] = {"chunks": surf_chunks, "dtype": "float32"}
    for v in upper_vars:
        encoding[v] = {"chunks": up_chunks, "dtype": "float32"}
    encoding["init_time"] = {
        "units": f"hours since {year}-01-01 00:00:00",
        "calendar": CALENDAR,
        "dtype": "int64",
    }

    out_store.parent.mkdir(parents=True, exist_ok=True)
    logger.info("allocating zarr template at %s (%d inits x %d leads)", out_store, n_init, n_lead)
    ds.to_zarr(
        out_store,
        mode="w" if overwrite else "w-",
        zarr_format=3,
        consolidated=True,
        encoding=encoding,
        compute=False,
    )

    # Stream one init at a time into its init_time region.
    for i, it in enumerate(inits):
        region_vars: dict = {}
        for v in surface_vars:
            region_vars[v] = (("init_time", "lead_time", "lat", "lon"),
                              it["surface"][v][np.newaxis, ...])
        for v in diag_vars:
            region_vars[v] = (("init_time", "lead_time", "lat", "lon"),
                              it["diagnostic"][v][np.newaxis, ...])
        for v in upper_vars:
            region_vars[v] = (("init_time", "lead_time", "pressure_level", "lat", "lon"),
                              it["upper"][v][np.newaxis, ...])
        xr.Dataset(region_vars).to_zarr(out_store, region={"init_time": slice(i, i + 1)})
    logger.info("wrote %d inits", n_init)

    # Re-consolidate (region writes don't touch the consolidated metadata, but
    # be explicit so downstream readers get a fresh zarr.json).
    try:
        import zarr
        zarr.consolidate_metadata(str(out_store))
    except Exception as exc:  # noqa: BLE001
        logger.warning("consolidate_metadata failed (non-fatal): %s", exc)

    return {
        "surface_variables": surface_vars,
        "diagnostic_variables": diag_vars,
        "upper_air_variables": upper_vars,
        "n_init": n_init,
        "n_lead": n_lead,
        "n_lev": n_lev,
        "n_lat": n_lat,
        "n_lon": n_lon,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    out_store = Path(args.out_store)
    if out_store.exists() and not args.overwrite:
        raise FileExistsError(f"{out_store} exists; pass --overwrite to replace")

    files = _list_inputs(input_dir, args.format)
    if not files:
        raise FileNotFoundError(f"no {args.format} inputs found in {input_dir}")
    logger.info("found %d candidate input(s) in %s", len(files), input_dir)

    loader = load_pangu_init if args.format == "pangu" else (
        lambda p: load_ai_rossby_init(p, prepend_ic=args.prepend_ic)
    )

    inits: list[dict] = []
    n_dropped_feb29 = 0
    n_wrong_year = 0
    for path in files:
        it = loader(path)
        dt = it["init_dt"]
        if _is_feb29(dt):
            n_dropped_feb29 += 1
            logger.info("dropping Feb-29 init: %s (%s)", dt, Path(path).name)
            continue
        if int(dt.year) != int(args.year):
            n_wrong_year += 1
            logger.warning("skipping init %s: year != --year %d (%s)", dt, args.year, Path(path).name)
            continue
        inits.append(it)

    if not inits:
        raise RuntimeError("no inits left after Feb-29 / year filtering")

    inits.sort(key=lambda d: _dt_key(d["init_dt"]))
    validate_inits(inits)

    # Schedule-completeness log (no silent truncation).
    expected = expected_schedule(int(args.year))
    got = {(int(it["init_dt"].month), int(it["init_dt"].day)) for it in inits}
    missing = [md for md in expected if md not in got]
    if missing:
        logger.warning(
            "%d/%d scheduled init slots MISSING for %d: %s",
            len(missing), len(expected), args.year,
            ", ".join(f"{m:02d}-{d:02d}" for m, d in missing),
        )
    else:
        logger.info("all %d scheduled init slots present for %d", len(expected), args.year)

    coords = resolve_coords(inits, Path(args.ref_store) if args.ref_store else None)

    generator = args.generator or (f"{SCRIPT_REL_PATH}@{args.commit}" if args.commit else SCRIPT_REL_PATH)
    summary = write_store(
        inits=inits,
        coords=coords,
        out_store=out_store,
        year=int(args.year),
        model=args.model,
        checkpoint=args.checkpoint or "",
        source_dataset=args.source_dataset or "",
        boundary_clamped_inits=list(args.boundary_clamped_inits or []),
        created=args.created or "",
        generator=generator,
        overwrite=args.overwrite,
    )

    # Spot-check a field's finite fraction.
    probe_name = (summary["surface_variables"] or summary["upper_air_variables"])[0]
    grp = "surface" if probe_name in inits[0]["surface"] else "upper"
    probe = inits[0][grp][probe_name]
    finite_frac = float(np.isfinite(probe).mean())

    logger.info("=" * 60)
    logger.info("SUMMARY for %s (%s, %d)", out_store, args.model, args.year)
    logger.info("  inputs found        : %d", len(files))
    logger.info("  inits kept          : %d", summary["n_init"])
    logger.info("  dropped Feb-29      : %d", n_dropped_feb29)
    logger.info("  skipped wrong-year  : %d", n_wrong_year)
    logger.info("  surface variables   : %s", summary["surface_variables"])
    logger.info("  diagnostic variables: %s", summary["diagnostic_variables"])
    logger.info("  upper-air variables : %s", summary["upper_air_variables"])
    logger.info("  output dims         : init_time=%d, lead_time=%d, pressure_level=%d, "
                "lat=%d, lon=%d", summary["n_init"], summary["n_lead"], summary["n_lev"],
                summary["n_lat"], summary["n_lon"])
    logger.info("  finite frac (%s @ init0) : %.4f", probe_name, finite_frac)
    logger.info("=" * 60)

    summary.update({
        "n_inputs_found": len(files),
        "n_dropped_feb29": n_dropped_feb29,
        "n_skipped_wrong_year": n_wrong_year,
        "n_missing_schedule": len(missing),
        "finite_fraction": finite_frac,
        "out_store": str(out_store),
    })
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Consolidate per-init hindcast outputs into a yearly Zarr v3 store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--format", choices=["pangu", "ai_rossby"], required=True)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory of per-init files (Pangu .nc) or per-IC stores (ai_rossby .zarr).")
    p.add_argument("--out-store", type=Path, required=True, help="Output Zarr store path.")
    p.add_argument("--model", required=True,
                   help="Model name, e.g. pangu_s2s / sfno_era5 / archesweather_era5.")
    p.add_argument("--year", type=int, required=True, help="Init year (2000-2024).")
    p.add_argument("--ref-store", type=Path, default=None,
                   help="ERA5 training zarr to copy lat/lon/pressure_level coords from.")
    p.add_argument("--checkpoint", default="", help="Checkpoint id/path (pass-through attr).")
    p.add_argument("--source-dataset", default="", help="Source dataset id (pass-through attr).")
    p.add_argument("--boundary-clamped-inits", nargs="*", default=[],
                   help="Init-time strings whose boundary forcing was clamped (pass-through attr).")
    p.add_argument("--created", default="",
                   help="Creation timestamp string (pass-through; NOT auto-filled).")
    p.add_argument("--commit", default="", help="Git commit hash for the generator attr.")
    p.add_argument("--generator", default="",
                   help="Override the full generator attr string (else script@commit).")
    p.add_argument("--prepend-ic", action="store_true",
                   help="Force prepending a lead-0 IC slot for ai_rossby inputs that emit "
                        "forecast steps only (auto-detected via frame_zero_is_ic otherwise).")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output store.")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
