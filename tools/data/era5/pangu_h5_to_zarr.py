#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert PanguWeather-style per-timestep ERA5 HDF5 files to one Zarr store.

The source layout
(``/work/hdd/bdiu/bgong1/data/h5data/{year}_{idx:04d}.h5``) carries one
sample per file with flat ``input/<varname>[_<pressure>]`` keys. Pressure-level
vars use a numeric suffix in **hPa** (``temperature_500.0``); surface vars are
flat (``2m_temperature``); single-level "boundary-like" vars (e.g.
``nino34_0``, ``soil_temperature_level_1``) keep their suffix in the
variable name and are written as separate surface channels — matching the
flat-channel convention used by the PLASIM and E3SM converters.

The output Zarr uses the shared ai-rossby :class:`ClimateZarrStoreLayout`:

* Coords: ``time`` (cftime), ``lat`` (deg N), ``lon`` (deg E),
  ``pressure_level`` (hPa).
* Per-variable arrays dimensioned by ``(time, [pressure_level,] lat, lon)``;
  constant boundaries are ``(lat, lon)``.
* Store ``attrs``: ``surface_variables``, ``constant_boundary_variables``,
  ``varying_boundary_variables``, ``diagnostic_variables``,
  ``pressure_upper_air_variables``, ``sigma_upper_air_variables`` (empty for
  ERA5 — pressure-only), ``calendar`` (``"standard"``),
  ``data_timedelta_hours`` (auto-detected from the file index spacing — ERA5
  is typically 6 h).

Channel groups default to a Pangu-Weather-style training config. Override via
``--channel-config <path-to-json>`` whose JSON dict supplies the same keys as
the default :data:`PANGU_ERA5_CHANNELS` mapping.

Usage::

    python tools/data/era5/pangu_h5_to_zarr.py \\
      --year 1979 --sample-range 0 1460 \\
      --output /work/nvme/bdiu/awikner/physicsnemo-zarr/era5/1979.zarr

cftime decoding is forced everywhere for consistency with the PLASIM
converter; the per-sample ``input/time`` scalar is parsed and reattached as a
cftime ``DatetimeGregorian`` value.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cftime
import h5py
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# tools/data/_common shares ai-rossby converter helpers across PLASIM/ERA5/E3SM.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.normalization import NORMALIZATION_SCHEMA_VERSION  # noqa: F401, E402  # re-imports for documentation

ERA5_ZARR_SCHEMA_VERSION = "1.0"


# Default Pangu-ERA5 channel groups. Pull from the standard Pangu-Weather
# v2.0 training feature set; the per-year converter writes every channel
# listed below. Override per-run with --channel-config <path>.
PANGU_ERA5_CHANNELS = {
    "surface_variables": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ],
    "constant_boundary_variables": [
        "land_sea_mask",
        "geopotential_at_surface",
    ],
    "varying_boundary_variables": [
        "sea_surface_temperature",
        "sea_ice_cover",
        "toa_incident_solar_radiation",
    ],
    "diagnostic_variables": [
        "total_precipitation_24hr",
    ],
    "pressure_upper_air_variables": [
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ],
    "sigma_upper_air_variables": [],
    # Levels: standard 13-level Pangu-Weather pressure set in hPa.
    "pressure_levels": [
        50.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        850.0,
        925.0,
        1000.0,
    ],
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ERA5 PanguWeather per-timestep HDF5 to a Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/work/hdd/bdiu/bgong1/data/h5data"),
        help="Dir containing ERA5 {year}_{idx:04d}.h5 per-timestep files.",
    )
    p.add_argument("--year", type=int, required=True, help="Year to convert.")
    p.add_argument(
        "--sample-range",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Half-open [LO, HI) range of file indices within the year. "
        "Defaults to all files for that year.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Zarr store path.",
    )
    p.add_argument(
        "--channel-config",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default Pangu-ERA5 channel groups.",
    )
    p.add_argument(
        "--data-timedelta-hours",
        type=int,
        default=6,
        help="Inter-sample time delta in hours.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Process-pool size for per-file H5 reads. "
        "0 autodetects via SLURM_CPUS_PER_TASK (fallback os.cpu_count()).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing output.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _resolve_n_workers(default: int = 32) -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    return max(1, os.cpu_count() or default)


def _list_files(input_dir: Path, year: int, sample_range: Optional[tuple[int, int]]):
    """Enumerate per-timestep H5 files for one year."""
    pattern = re.compile(rf"^{year}_(\d{{4}})\.h5$")
    matches: list[tuple[int, Path]] = []
    for p in input_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    matches.sort()
    if not matches:
        raise FileNotFoundError(f"no {year}_NNNN.h5 files in {input_dir}")
    if sample_range is not None:
        lo, hi = sample_range
        matches = [(i, p) for i, p in matches if lo <= i < hi]
    return matches


def _level_key(group: h5py.Group, prefix: str, level: float, rtol: float = 1e-3) -> str:
    """Find the H5 key in ``group`` that encodes ``prefix_<level>``.

    The level token can have trailing zeros, scientific notation, or excess
    precision — match by parsing the trailing numeric portion of each key with
    the requested ``prefix`` and tolerance.
    """
    candidates = []
    for k in group.keys():
        if not k.startswith(prefix + "_"):
            continue
        tail = k[len(prefix) + 1 :]
        try:
            value = float(tail)
        except ValueError:
            continue
        if abs(value - float(level)) <= rtol * abs(float(level) or 1.0):
            candidates.append(k)
    if not candidates:
        raise KeyError(
            f"no H5 key matches {prefix}_<{level}> in {sorted(group.keys())[:5]}..."
        )
    if len(candidates) > 1:
        raise KeyError(f"ambiguous {prefix}_{level}: {candidates}")
    return candidates[0]


def _decode_time(time_value) -> cftime.datetime:
    """Parse the ERA5 per-file time scalar into a cftime DatetimeGregorian.

    The source H5 stores a numpy.datetime64 bytes blob like
    ``b'1979-01-01T00:00:00.000000000'``; we coerce to ``YYYY-MM-DDTHH:MM``
    and rebuild as cftime to stay on the project-wide cftime path.
    """
    if isinstance(time_value, bytes):
        s = time_value.decode()
    else:
        s = str(time_value)
    s = s.split(".")[0]  # drop fractional seconds
    # Format: YYYY-MM-DDTHH:MM:SS (Z optional)
    if "T" in s:
        date_part, time_part = s.split("T", 1)
    else:
        date_part, time_part = s.split(" ", 1)
    year, month, day = (int(x) for x in date_part.split("-"))
    h, m, sec = (int(x) for x in time_part.replace("Z", "").split(":"))
    return cftime.DatetimeGregorian(year, month, day, h, m, sec)


def _read_one_file(
    path: Path,
    *,
    surface_vars: list[str],
    constant_boundary_vars: list[str],
    varying_boundary_vars: list[str],
    diagnostic_vars: list[str],
    pressure_upper_air_vars: list[str],
    pressure_levels: list[float],
    read_constants: bool,
) -> dict:
    """Worker: pull one H5 sample's data into per-var float32 ndarrays."""
    with h5py.File(path, "r") as f:
        g = f["input"]
        time = _decode_time(g["time"][()])
        out: dict = {"time": time}

        for v in surface_vars + varying_boundary_vars + diagnostic_vars:
            out[v] = np.asarray(g[v][:], dtype="float32")

        for v in pressure_upper_air_vars:
            stack = np.stack(
                [np.asarray(g[_level_key(g, v, lev)][:], dtype="float32") for lev in pressure_levels],
                axis=0,
            )
            out[v] = stack  # shape (n_levels, H, W)

        if read_constants:
            for v in constant_boundary_vars:
                out[f"_const_{v}"] = np.asarray(g[v][:], dtype="float32")

    return out


def _load_channel_config(path: Optional[Path]) -> dict:
    if path is None:
        return dict(PANGU_ERA5_CHANNELS)
    with open(path) as fh:
        user = json.load(fh)
    merged = dict(PANGU_ERA5_CHANNELS)
    merged.update(user)
    return merged


def convert(args: argparse.Namespace) -> None:
    channels = _load_channel_config(args.channel_config)
    surface_vars = list(channels["surface_variables"])
    constant_boundary_vars = list(channels["constant_boundary_variables"])
    varying_boundary_vars = list(channels["varying_boundary_variables"])
    diagnostic_vars = list(channels["diagnostic_variables"])
    pressure_upper_air_vars = list(channels["pressure_upper_air_variables"])
    pressure_levels = [float(x) for x in channels["pressure_levels"]]

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace")

    files = _list_files(args.input_dir, args.year, args.sample_range)
    logger.info(
        "year %d: %d sample files in [%d, %d)",
        args.year,
        len(files),
        files[0][0],
        files[-1][0] + 1,
    )

    n_workers = args.n_workers or _resolve_n_workers()
    logger.info("loading per-file H5 reads across %d workers", n_workers)

    # Read all per-file payloads in parallel. The first file additionally
    # provides the constant-boundary values (they don't vary in time so we
    # only read them once).
    payloads: dict[int, dict] = {}
    if n_workers <= 1 or len(files) <= 1:
        for idx, path in files:
            payloads[idx] = _read_one_file(
                path,
                surface_vars=surface_vars,
                constant_boundary_vars=constant_boundary_vars,
                varying_boundary_vars=varying_boundary_vars,
                diagnostic_vars=diagnostic_vars,
                pressure_upper_air_vars=pressure_upper_air_vars,
                pressure_levels=pressure_levels,
                read_constants=(idx == files[0][0]),
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {
                ex.submit(
                    _read_one_file,
                    path,
                    surface_vars=surface_vars,
                    constant_boundary_vars=constant_boundary_vars,
                    varying_boundary_vars=varying_boundary_vars,
                    diagnostic_vars=diagnostic_vars,
                    pressure_upper_air_vars=pressure_upper_air_vars,
                    pressure_levels=pressure_levels,
                    read_constants=(idx == files[0][0]),
                ): idx
                for idx, path in files
            }
            for i, fut in enumerate(as_completed(future_to_idx)):
                idx = future_to_idx[fut]
                payloads[idx] = fut.result()
                if (i + 1) % 200 == 0:
                    logger.info("loaded %d / %d", i + 1, len(files))

    # Assemble into the Zarr-bound xarray.Dataset.
    ordered_idxs = [i for i, _ in files]
    times = [payloads[i]["time"] for i in ordered_idxs]
    n_time = len(ordered_idxs)
    # The lat / lon grid for ERA5 is 180×360.
    sample = next(iter(payloads.values()))
    sample_surface = next(iter(v for k, v in sample.items() if k in surface_vars))
    n_lat, n_lon = sample_surface.shape

    coords = {
        "time": ("time", times),
        "lat": ("lat", np.linspace(89.5, -89.5, n_lat, dtype="float32")),
        "lon": ("lon", np.linspace(0.0, 360.0 * (n_lon - 1) / n_lon, n_lon, dtype="float32")),
        "pressure_level": ("pressure_level", np.asarray(pressure_levels, dtype="float32")),
    }

    data_vars: dict = {}
    for v in surface_vars + varying_boundary_vars + diagnostic_vars:
        arr = np.stack([payloads[i][v] for i in ordered_idxs], axis=0)
        data_vars[v] = (("time", "lat", "lon"), arr)
    for v in pressure_upper_air_vars:
        arr = np.stack([payloads[i][v] for i in ordered_idxs], axis=0)
        data_vars[v] = (("time", "pressure_level", "lat", "lon"), arr)
    # Constants — pull from the first file's payload.
    first_payload = payloads[files[0][0]]
    for v in constant_boundary_vars:
        data_vars[v] = (("lat", "lon"), first_payload[f"_const_{v}"])

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "era5_zarr_schema_version": ERA5_ZARR_SCHEMA_VERSION,
            "calendar": "standard",
            "data_timedelta_hours": int(args.data_timedelta_hours),
            "source_input_dir": str(args.input_dir),
            "surface_variables": surface_vars,
            "constant_boundary_variables": constant_boundary_vars,
            "varying_boundary_variables": varying_boundary_vars,
            "diagnostic_variables": diagnostic_vars,
            "pressure_upper_air_variables": pressure_upper_air_vars,
            "sigma_upper_air_variables": [],
            "year_index": int(args.year),
            "sample_range": [files[0][0], files[-1][0] + 1],
        },
    )
    logger.info("writing Zarr to %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"chunks": (1,) + ds[v].shape[1:]} for v in ds.data_vars if "time" in ds[v].dims}
    ds.to_zarr(
        args.output,
        mode="w" if args.overwrite else "w-",
        consolidated=True,
        zarr_format=3,
        encoding=encoding,
    )
    logger.info("done: %d vars, %d timesteps", len(ds.data_vars), n_time)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    convert(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
