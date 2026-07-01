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
    # Levels: FULL ERA5 source coverage — 18 pressure levels in hPa
    # (PanguWeather v2.0 ERA5 H5 archives include all 18). Model configs may
    # subset (e.g. SFNO_S2S drops 200 hPa to use 17 levels); the *archive*
    # captures everything so subsetting is a downstream choice.
    "pressure_levels": [
        5.0,
        10.0,
        20.0,
        30.0,
        50.0,
        70.0,
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
    p.add_argument(
        "--write-batch",
        type=int,
        default=50,
        help="Number of timesteps held in memory at once and written to Zarr "
        "in one region update. With the parallel reader the per-batch peak "
        "memory is ``(write_batch + n_workers)`` ordered payloads, not the "
        "full year. See amip_h5_to_zarr.py for the rationale.",
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
    write_batch = max(1, int(args.write_batch))
    logger.info(
        "loading per-file H5 reads across %d workers, write-batch=%d",
        n_workers, write_batch,
    )

    # --- Streaming write strategy ---------------------------------------
    # 1. Read the first file in the main process to capture constants +
    #    pin the (lat, lon) shape (the constants_first-file convention).
    # 2. Allocate a dask-zeros template Dataset spanning the full time
    #    axis and write only the Zarr metadata (compute=False) up front.
    # 3. Use ProcessPoolExecutor to read remaining files concurrently;
    #    consume futures in *index order* in batches of ``write_batch``,
    #    write each batch to its time-region. Peak resident payloads are
    #    bounded by ``write_batch + n_workers`` (futures in flight +
    #    batch buffer), not the full year.
    import time as _time

    import dask.array as da

    n_time = len(files)
    first_idx, first_path = files[0]
    logger.info("Reading first file for constants + reference shape")
    first_payload = _read_one_file(
        first_path,
        surface_vars=surface_vars,
        constant_boundary_vars=constant_boundary_vars,
        varying_boundary_vars=varying_boundary_vars,
        diagnostic_vars=diagnostic_vars,
        pressure_upper_air_vars=pressure_upper_air_vars,
        pressure_levels=pressure_levels,
        read_constants=True,
    )
    sample_surface = next(iter(first_payload[v] for v in surface_vars))
    n_lat, n_lon = sample_surface.shape
    n_levels = len(pressure_levels)

    # The first file's time we already have; remaining times will arrive
    # via the per-batch future drain below. The Zarr template needs the
    # full time coord up front — but Zarr v3 + xarray let us write the
    # time coord during the template write and then ``region=`` writes
    # only touch the data vars (not the coord). So we read all timestamps
    # in a cheap first pass before the heavy reads.
    logger.info("Pass 1/2: reading %d timestamps for the time coord", n_time)
    t_phase = _time.time()
    times: list[cftime.datetime] = [first_payload["time"]]
    for idx, path in files[1:]:
        with h5py.File(path, "r") as f:
            times.append(_decode_time(f["input"]["time"][()]))
    logger.info("  done in %.1fs", _time.time() - t_phase)

    coords = {
        "time": ("time", times),
        "lat": ("lat", np.linspace(89.5, -89.5, n_lat, dtype="float32")),
        "lon": (
            "lon",
            np.linspace(0.0, 360.0 * (n_lon - 1) / n_lon, n_lon, dtype="float32"),
        ),
        "pressure_level": ("pressure_level", np.asarray(pressure_levels, dtype="float32")),
    }

    surface_shape = (n_time, n_lat, n_lon)
    surface_chunks = (1, n_lat, n_lon)
    upper_shape = (n_time, n_levels, n_lat, n_lon)
    upper_chunks = (1, n_levels, n_lat, n_lon)

    data_vars: dict = {}
    for v in surface_vars + varying_boundary_vars + diagnostic_vars:
        data_vars[v] = (
            ("time", "lat", "lon"),
            da.zeros(surface_shape, chunks=surface_chunks, dtype="float32"),
        )
    for v in pressure_upper_air_vars:
        data_vars[v] = (
            ("time", "pressure_level", "lat", "lon"),
            da.zeros(upper_shape, chunks=upper_chunks, dtype="float32"),
        )
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
    logger.info("Allocating Zarr template at %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"chunks": surface_chunks if "pressure_level" not in ds[v].dims else upper_chunks} for v in ds.data_vars if "time" in ds[v].dims}
    ds.to_zarr(
        args.output,
        mode="w" if args.overwrite else "w-",
        consolidated=True,
        zarr_format=3,
        encoding=encoding,
        compute=False,
    )

    # --- Pass 2/2: stream batched reads → batched region writes ---------
    surface_3d_vars = surface_vars + varying_boundary_vars + diagnostic_vars

    def _write_batch_payloads(start: int, payloads_ordered: list[dict]) -> None:
        bsize = len(payloads_ordered)
        if bsize == 0:
            return
        buf_3d = {
            v: np.stack([p[v] for p in payloads_ordered], axis=0)
            for v in surface_3d_vars
        }
        buf_4d = {
            v: np.stack([p[v] for p in payloads_ordered], axis=0)
            for v in pressure_upper_air_vars
        }
        batch_vars: dict = {}
        for v in surface_3d_vars:
            batch_vars[v] = (("time", "lat", "lon"), buf_3d[v])
        for v in pressure_upper_air_vars:
            batch_vars[v] = (("time", "pressure_level", "lat", "lon"), buf_4d[v])
        batch_ds = xr.Dataset(batch_vars)
        batch_ds.to_zarr(
            args.output, region={"time": slice(start, start + bsize)}
        )
        logger.info("  wrote timesteps %d..%d / %d", start, start + bsize, n_time)

    logger.info(
        "Pass 2/2: streaming %d timesteps in batches of %d", n_time, write_batch
    )
    t_phase = _time.time()

    if n_workers <= 1 or n_time <= 1:
        # Serial path — read + write in batches, no PPE.
        batch_payloads: list[dict] = [first_payload]
        for k in range(1, n_time):
            _, path = files[k]
            payload = _read_one_file(
                path,
                surface_vars=surface_vars,
                constant_boundary_vars=constant_boundary_vars,
                varying_boundary_vars=varying_boundary_vars,
                diagnostic_vars=diagnostic_vars,
                pressure_upper_air_vars=pressure_upper_air_vars,
                pressure_levels=pressure_levels,
                read_constants=False,
            )
            batch_payloads.append(payload)
            if len(batch_payloads) == write_batch:
                _write_batch_payloads(k - len(batch_payloads) + 1, batch_payloads)
                batch_payloads = []
        if batch_payloads:
            _write_batch_payloads(n_time - len(batch_payloads), batch_payloads)
    else:
        # Parallel path — submit all futures up front; PPE keeps
        # ``n_workers`` reads in flight while we drain in order.
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures: list = [None] * n_time
            for k, (idx, path) in enumerate(files):
                if k == 0:
                    continue  # first payload already in hand
                futures[k] = ex.submit(
                    _read_one_file,
                    path,
                    surface_vars=surface_vars,
                    constant_boundary_vars=constant_boundary_vars,
                    varying_boundary_vars=varying_boundary_vars,
                    diagnostic_vars=diagnostic_vars,
                    pressure_upper_air_vars=pressure_upper_air_vars,
                    pressure_levels=pressure_levels,
                    read_constants=False,
                )

            for batch_start in range(0, n_time, write_batch):
                batch_end = min(batch_start + write_batch, n_time)
                batch_payloads = []
                for k in range(batch_start, batch_end):
                    if k == 0:
                        batch_payloads.append(first_payload)
                    else:
                        batch_payloads.append(futures[k].result())
                        futures[k] = None  # release for GC
                _write_batch_payloads(batch_start, batch_payloads)

    logger.info(
        "done: %d vars, %d timesteps (write phase %.1fs)",
        len(ds.data_vars), n_time, _time.time() - t_phase,
    )


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
