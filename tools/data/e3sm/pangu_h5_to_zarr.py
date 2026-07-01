#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert PanguWeather-style per-timestep E3SM HDF5 files to one Zarr store.

E3SM source layout
(``/work/hdd/bdiu/awikner/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data/``)
has one sample per ``{year}_{idx:04d}.h5`` file. Flat ``input/<var>[_<level>]``
keys; pressure-level vars use **hPa** in the level suffix (e.g.
``T_998.4964394917621``). Note:

* E3SM uses uppercase var names (``T``, ``U``, ``V``, ``Z3``, ``CLOUD``, etc.).
* Hybrid pressure levels are written as **hPa floats** with full source
  precision — the converter rounds to float32 at the unified ``pressure_level``
  coord per the ai-rossby schema convention.
* Calendar: 365-day no-leap (``noleap`` in cftime).
* Soil vars (``H2OSOI``, ``TSOI``) appear in the *climatology* file only and
  decompose into per-depth 2D channels (see
  :mod:`tools.data.e3sm.build_climatology_zarr`).
  The per-year converter does NOT process soil vars.

The output Zarr uses the shared ai-rossby
:class:`ClimateZarrStoreLayout`, identical schema to ERA5 and PLASIM stores.

Default channel groups follow the Pangu-Weather-style training feature set
adapted to E3SM var names. Override via ``--channel-config <json>``.

Usage::

    python tools/data/e3sm/pangu_h5_to_zarr.py \\
      --year 2015 --sample-range 0 1460 \\
      --output /work/nvme/bdiu/awikner/physicsnemo-zarr/e3sm/2015.zarr
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

# Default Pangu-E3SM channel groups (overrideable per-run via --channel-config).
# All pressure values are hPa.
PANGU_E3SM_CHANNELS = {
    "surface_variables": [
        "TREFHT",
        "U10",
        "PSL",
    ],
    "constant_boundary_variables": [
        "TOPO",
        "PCT_GLACIER",
        "PCT_NATVEG",
        "PFTDATA_MASK",
    ],
    "varying_boundary_variables": [
        "SST",
        "ICE",
        "sol_in",
    ],
    "diagnostic_variables": [
        "PRECT",
    ],
    "pressure_upper_air_variables": [
        "T",
        "U",
        "V",
        "RELHUM",
        "Z3",
    ],
    "sigma_upper_air_variables": [],
    # FULL E3SM hybrid-pressure coverage — all 18 levels (hPa). These are
    # the actual hybrid-pressure values written into the H5 keys (e.g.
    # `T_50.11779996521295`); the converter rounds to float32 for the unified
    # `pressure_level` coord. Model configs may subset; the archive keeps
    # everything so we don't drop stratospheric levels by default.
    "pressure_levels": [
        4.714998332947841,    # ~5
        10.655023096474308,   # ~10
        19.235455601758737,   # ~20
        28.79458853709195,    # ~30
        50.11779996521295,    # ~50
        69.59908688413749,    # ~70
        96.46377266572703,    # ~100
        145.04282239200347,   # ~150
        200.99889546355382,   # ~200
        256.72368590525895,   # ~250
        302.21364012188303,   # ~300
        385.999023919911,     # ~400
        492.46857402252755,   # ~500
        608.6437744215842,    # ~600
        713.7046383204334,    # ~700
        849.6612491105952,    # ~850
        925.5197481473349,    # ~925
        998.4964394917621,    # ~1000
    ],
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert E3SM PanguWeather per-timestep HDF5 to a Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/work/hdd/bdiu/awikner/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data"
        ),
        help="Dir containing the E3SM {year}_{idx:04d}.h5 per-timestep files.",
    )
    p.add_argument("--year", type=int, required=True)
    p.add_argument(
        "--sample-range",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Half-open [LO, HI) range of file indices within the year.",
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
        help="Optional JSON override for the default Pangu-E3SM channel groups.",
    )
    p.add_argument(
        "--data-timedelta-hours",
        type=int,
        default=6,
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
        "in one region update. See amip_h5_to_zarr.py for the rationale.",
    )
    p.add_argument("--overwrite", action="store_true")
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


def _level_key(group: h5py.Group, prefix: str, level: float, rtol: float = 5e-3) -> str:
    """Resolve ``prefix_<level>`` in an E3SM H5 group.

    E3SM's level tokens carry full float64 precision (e.g.
    ``T_998.4964394917621``); we match by absolute hPa with a loose tolerance
    against the requested 13-level Pangu set (50, 100, ..., 1000).
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
            candidates.append((abs(value - float(level)), k))
    if not candidates:
        raise KeyError(
            f"no H5 key matches {prefix}_<{level}> in {sorted(group.keys())[:5]}..."
        )
    candidates.sort()
    return candidates[0][1]


def _decode_time(time_value, *, year: int, idx: int, data_timedelta_hours: int) -> cftime.datetime:
    """Decode E3SM's per-file time scalar to a cftime ``DatetimeNoLeap``.

    The H5 may store the time as bytes/str; E3SM uses noleap calendar so we
    construct via dayofyear arithmetic from the per-file index (lo=0 → Jan 1
    00:00). Falls through to parsing the H5 scalar if it's a recognizable ISO
    timestamp.
    """
    if isinstance(time_value, bytes):
        s = time_value.decode()
    elif isinstance(time_value, str):
        s = time_value
    else:
        s = ""
    s = s.split(".")[0]
    if "T" in s:
        try:
            date_part, time_part = s.split("T", 1)
            y, mo, d = (int(x) for x in date_part.split("-"))
            h, m, sec = (int(x) for x in time_part.replace("Z", "").split(":"))
            return cftime.DatetimeNoLeap(y, mo, d, h, m, sec)
        except (ValueError, IndexError):
            pass
    # Fallback: synthesize from the file index.
    from datetime import timedelta

    total_hours = idx * data_timedelta_hours
    day_of_year = total_hours // 24
    hour = total_hours % 24
    base = cftime.DatetimeNoLeap(year, 1, 1, 0, 0, 0)
    return base + timedelta(days=day_of_year, hours=hour)


def _read_one_file(
    path: Path,
    idx: int,
    *,
    year: int,
    surface_vars: list[str],
    constant_boundary_vars: list[str],
    varying_boundary_vars: list[str],
    diagnostic_vars: list[str],
    pressure_upper_air_vars: list[str],
    pressure_levels: list[float],
    data_timedelta_hours: int,
    read_constants: bool,
) -> dict:
    with h5py.File(path, "r") as f:
        g = f["input"]
        time_raw = g["time"][()] if "time" in g else b""
        time = _decode_time(
            time_raw,
            year=year,
            idx=idx,
            data_timedelta_hours=data_timedelta_hours,
        )
        out: dict = {"time": time}
        for v in surface_vars + varying_boundary_vars + diagnostic_vars:
            out[v] = np.asarray(g[v][:], dtype="float32")
        for v in pressure_upper_air_vars:
            stack = np.stack(
                [np.asarray(g[_level_key(g, v, lev)][:], dtype="float32") for lev in pressure_levels],
                axis=0,
            )
            out[v] = stack
        if read_constants:
            for v in constant_boundary_vars:
                out[f"_const_{v}"] = np.asarray(g[v][:], dtype="float32")
    return out


def _load_channel_config(path: Optional[Path]) -> dict:
    if path is None:
        return dict(PANGU_E3SM_CHANNELS)
    with open(path) as fh:
        user = json.load(fh)
    merged = dict(PANGU_E3SM_CHANNELS)
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

    # --- Streaming write strategy (see amip_h5_to_zarr.py for rationale) ---
    import time as _time

    import dask.array as da

    n_time = len(files)

    # Pass 1: read first file (constants + reference shape) + all times.
    first_idx, first_path = files[0]
    logger.info("Reading first file for constants + reference shape")
    first_payload = _read_one_file(
        first_path,
        first_idx,
        year=args.year,
        surface_vars=surface_vars,
        constant_boundary_vars=constant_boundary_vars,
        varying_boundary_vars=varying_boundary_vars,
        diagnostic_vars=diagnostic_vars,
        pressure_upper_air_vars=pressure_upper_air_vars,
        pressure_levels=pressure_levels,
        data_timedelta_hours=args.data_timedelta_hours,
        read_constants=True,
    )
    sample_surface = next(iter(first_payload[v] for v in surface_vars))
    n_lat, n_lon = sample_surface.shape
    n_levels = len(pressure_levels)

    logger.info("Pass 1/2: reading %d timestamps for the time coord", n_time)
    t_phase = _time.time()
    times: list[cftime.datetime] = [first_payload["time"]]
    for idx, path in files[1:]:
        with h5py.File(path, "r") as f:
            time_raw = f["input"]["time"][()] if "time" in f["input"] else b""
        times.append(
            _decode_time(
                time_raw,
                year=args.year,
                idx=idx,
                data_timedelta_hours=args.data_timedelta_hours,
            )
        )
    logger.info("  done in %.1fs", _time.time() - t_phase)

    coords = {
        "time": ("time", times),
        "lat": ("lat", np.linspace(-89.5, 89.5, n_lat, dtype="float32")),
        "lon": ("lon", np.linspace(0.5, 359.5, n_lon, dtype="float32")),
        "pressure_level": (
            "pressure_level",
            np.asarray(pressure_levels, dtype="float32"),
        ),
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
            "e3sm_zarr_schema_version": "1.0",
            "calendar": "noleap",
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
    encoding = {
        v: {
            "chunks": surface_chunks
            if "pressure_level" not in ds[v].dims
            else upper_chunks
        }
        for v in ds.data_vars
        if "time" in ds[v].dims
    }
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
        batch_payloads: list[dict] = [first_payload]
        for k in range(1, n_time):
            idx, path = files[k]
            payload = _read_one_file(
                path,
                idx,
                year=args.year,
                surface_vars=surface_vars,
                constant_boundary_vars=constant_boundary_vars,
                varying_boundary_vars=varying_boundary_vars,
                diagnostic_vars=diagnostic_vars,
                pressure_upper_air_vars=pressure_upper_air_vars,
                pressure_levels=pressure_levels,
                data_timedelta_hours=args.data_timedelta_hours,
                read_constants=False,
            )
            batch_payloads.append(payload)
            if len(batch_payloads) == write_batch:
                _write_batch_payloads(k - len(batch_payloads) + 1, batch_payloads)
                batch_payloads = []
        if batch_payloads:
            _write_batch_payloads(n_time - len(batch_payloads), batch_payloads)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures: list = [None] * n_time
            for k, (idx, path) in enumerate(files):
                if k == 0:
                    continue
                futures[k] = ex.submit(
                    _read_one_file,
                    path,
                    idx,
                    year=args.year,
                    surface_vars=surface_vars,
                    constant_boundary_vars=constant_boundary_vars,
                    varying_boundary_vars=varying_boundary_vars,
                    diagnostic_vars=diagnostic_vars,
                    pressure_upper_air_vars=pressure_upper_air_vars,
                    pressure_levels=pressure_levels,
                    data_timedelta_hours=args.data_timedelta_hours,
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
                        futures[k] = None
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
