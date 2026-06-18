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
    logger.info("loading per-file H5 reads across %d workers", n_workers)

    payloads: dict[int, dict] = {}
    if n_workers <= 1 or len(files) <= 1:
        for idx, path in files:
            payloads[idx] = _read_one_file(
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
                read_constants=(idx == files[0][0]),
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {
                ex.submit(
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
                    read_constants=(idx == files[0][0]),
                ): idx
                for idx, path in files
            }
            for i, fut in enumerate(as_completed(future_to_idx)):
                idx = future_to_idx[fut]
                payloads[idx] = fut.result()
                if (i + 1) % 200 == 0:
                    logger.info("loaded %d / %d", i + 1, len(files))

    ordered_idxs = [i for i, _ in files]
    times = [payloads[i]["time"] for i in ordered_idxs]
    sample = next(iter(payloads.values()))
    sample_surface = next(iter(v for k, v in sample.items() if k in surface_vars))
    n_lat, n_lon = sample_surface.shape

    coords = {
        "time": ("time", times),
        "lat": ("lat", np.linspace(-89.5, 89.5, n_lat, dtype="float32")),
        "lon": ("lon", np.linspace(0.5, 359.5, n_lon, dtype="float32")),
        "pressure_level": ("pressure_level", np.asarray(pressure_levels, dtype="float32")),
    }
    data_vars: dict = {}
    for v in surface_vars + varying_boundary_vars + diagnostic_vars:
        arr = np.stack([payloads[i][v] for i in ordered_idxs], axis=0)
        data_vars[v] = (("time", "lat", "lon"), arr)
    for v in pressure_upper_air_vars:
        arr = np.stack([payloads[i][v] for i in ordered_idxs], axis=0)
        data_vars[v] = (("time", "pressure_level", "lat", "lon"), arr)
    first_payload = payloads[files[0][0]]
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
    logger.info("writing Zarr to %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        v: {"chunks": (1,) + ds[v].shape[1:]}
        for v in ds.data_vars
        if "time" in ds[v].dims
    }
    ds.to_zarr(
        args.output,
        mode="w" if args.overwrite else "w-",
        consolidated=True,
        zarr_format=3,
        encoding=encoding,
    )
    logger.info("done: %d vars, %d timesteps", len(ds.data_vars), len(times))


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
