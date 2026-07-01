#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert AMIP per-timestep HDF5 archives to the per-year Zarr layout
:class:`physicsnemo.experimental.datapipes.climate.ClimateZarrDataset` reads.

Sibling of ``tools/data/plasim/pangu_h5_to_zarr.py``. Differences from
PLASIM:

* **No sigma levels** — AMIP atmospheric variables live on a single 26-level
  pressure axis (hPa). The converter writes a ``pressure_level`` coord and
  no ``sigma_level`` coord. Sigma-related code paths from the PLASIM
  converter are absent.
* **Soil-as-surface** — the user-supplied channel config flattens the 4-level
  soil variables (``soil_temperature_level_{1..4}``,
  ``volumetric_soil_water_layer_{1..4}``) into ``surface_variables`` so they
  occupy 8 named entries rather than a third vertical axis. This matches
  PanguPlasim's existing land_variables convention.
* **Extra variables** — the new ``extra_variables`` config key holds variable
  names that get written to the Zarr (as ``(time, pressure_level, lat, lon)``
  cubes for upper-air, ``(time, lat, lon)`` for surface) but are NOT added
  to the role lists in the Zarr ``attrs``. Useful for "save the data but
  don't route it through any model group by default" — e.g. AMIP's
  ``vertical_velocity`` upper-air var.
* **Calendar** — defaults to ``standard`` (Gregorian) for AMIP's 1978–2024
  span; PLASIM defaulted to ``proleptic_gregorian``. Override via the
  config's ``calendar`` key if needed.

Reads a YAML config to pick up channel structure and runs ``{year}_NNNN.h5``
files from the source dir into a single per-year ``.zarr`` store with the
schema:

* ``time``: (T,) cftime datetimes parsed from the per-file ``input/time`` scalar.
* ``lat``, ``lon``: (H,), (W,) coordinate axes (taken from the YAML).
* ``pressure_level``: (L_p,) coordinate axis (hPa, from the YAML ``levels``).
* Surface vars: ``(time, lat, lon)``.
* Pressure-level upper-air vars: ``(time, pressure_level, lat, lon)``.
* Constant boundaries: ``(lat, lon)`` (assumed constant in time; first file used).
* Varying boundaries: ``(time, lat, lon)``.
* Diagnostic vars: ``(time, lat, lon)``.

Store ``attrs`` carry the channel-group lists, calendar, data timedelta,
the source config path, and schema version.

Usage
-----

::

    python tools/data/amip/amip_h5_to_zarr.py \\
      --config tools/data/amip/configs/amip_default.yaml \\
      --year 1981 \\
      --output /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr
import yaml

from physicsnemo.experimental.datapipes.climate import (
    CLIMATE_ZARR_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, required=True, help="AMIP channel-config YAML."
    )
    p.add_argument(
        "--config-key",
        default="AMIP",
        help="Top-level YAML key to read channel/level config from.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Source dir containing {year}_{idx:04d}.h5. Defaults to config's data_dir.",
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
        help="Output .zarr store path. Overwritten if it exists.",
    )
    p.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Number of timesteps per Zarr chunk along the time axis. The default "
        "of 1 (one chunk per timestep) is the fastest for random-access training "
        "workloads.",
    )
    p.add_argument(
        "--write-batch",
        type=int,
        default=50,
        help="Number of timesteps held in memory at once and written to Zarr "
        "in one region update. Smaller = lower peak memory, more zarr writes; "
        "larger = the inverse. 50 ≈ 2 GB peak for the AMIP 1981 layout — fits "
        "in any sane Slurm allocation while keeping the write overhead negligible.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _read_config(path: Path, key: str) -> dict:
    with path.open() as f:
        raw = yaml.safe_load(f)
    if key not in raw:
        raise KeyError(f"Config {path} has no top-level key {key!r}; got {list(raw)}")
    return raw[key]


def _list_files(
    input_dir: Path, year: int, sample_range: tuple[int, int] | None
) -> list[Path]:
    pattern = re.compile(rf"^{year}_(\d{{4}})\.h5$")
    found: list[tuple[int, Path]] = []
    for path in sorted(input_dir.iterdir()):
        m = pattern.match(path.name)
        if m:
            idx = int(m.group(1))
            found.append((idx, path))
    if not found:
        raise FileNotFoundError(
            f"No files matching {year}_NNNN.h5 in {input_dir}"
        )
    found.sort(key=lambda t: t[0])
    if sample_range is not None:
        lo, hi = sample_range
        found = [t for t in found if lo <= t[0] < hi]
        if not found:
            raise ValueError(
                f"No files in [{lo}, {hi}); available indices "
                f"{found[0][0] if found else '(none)'}..{found[-1][0] if found else ''}"
            )
    return [p for _, p in found]


def _level_key(group: h5py._hl.group.Group, prefix: str, level: float) -> str:
    """Find the HDF5 dataset name matching ``{prefix}_{level}`` within
    numerical tolerance — keys store the level as a Python-formatted float
    string that may not be reproducible by ``f"{level}"``.
    """
    rtol, atol = 1e-3, 1e-6
    candidates: list[tuple[float, str]] = []
    for name in group:
        if not name.startswith(prefix + "_"):
            continue
        try:
            val = float(name[len(prefix) + 1:])
        except ValueError:
            continue
        candidates.append((val, name))
    for val, name in candidates:
        if abs(val - level) <= atol + rtol * abs(level):
            return name
    available = sorted(v for v, _ in candidates)
    raise KeyError(
        f"No dataset matching {prefix}_{level}; available {prefix} levels: {available}"
    )


def _decode_time(time_dataset_value, calendar: str) -> cftime.datetime:
    """Decode the ``input/time`` scalar dataset's contents into a cftime
    datetime. AMIP stores it as ``YYYY-MM-DDTHH:MM:SS.nnnnnnnnn`` (nanosecond-
    resolution string)."""
    if isinstance(time_dataset_value, bytes):
        s = time_dataset_value.decode("ascii")
    else:
        s = str(time_dataset_value)
    s = s.replace("T", " ").strip()
    m = re.match(
        r"^(?P<y>-?\d+)-(?P<mo>\d+)-(?P<d>\d+)[ ]"
        r"(?P<h>\d+):(?P<mi>\d+):(?P<s>\d+(?:\.\d+)?)$",
        s,
    )
    if not m:
        raise ValueError(f"Unrecognized time string: {s!r}")
    y = int(m["y"])
    mo = int(m["mo"])
    d = int(m["d"])
    h = int(m["h"])
    mi = int(m["mi"])
    sec_f = float(m["s"])
    sec = int(sec_f)
    micros = int((sec_f - sec) * 1e6)
    cls = cftime._cftime.DATE_TYPES.get(calendar, cftime.DatetimeGregorian)
    return cls(y, mo, d, h, mi, sec, micros)


def _read_time_only(path: Path, calendar: str) -> cftime.datetime:
    """Read the per-file timestamp (no payload) for the cheap first pass.

    The streaming converter needs the full time axis up front to size
    the Zarr template. Reading just the ``input/time`` scalar is ~1000×
    cheaper than the full file read and decouples the time-axis build
    from the data-batch reads.
    """
    with h5py.File(path, "r") as f:
        return _decode_time(f["input"]["time"][()], calendar)


def _read_one_file(
    path: Path,
    *,
    surface_vars: list[str],
    pressure_upper_vars: list[str],
    constant_boundary_vars: list[str],
    varying_boundary_vars: list[str],
    diagnostic_vars: list[str],
    extra_surface_vars: list[str],
    extra_pressure_upper_vars: list[str],
    pressure_levels: list[float],
    calendar: str,
    include_constants: bool,
) -> dict[str, object]:
    """Read one AMIP per-timestep HDF5 file, return a dict of per-channel
    arrays + a decoded ``time`` value.
    """
    out: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        grp = f["input"]
        out["time"] = _decode_time(grp["time"][()], calendar)

        for v in surface_vars + varying_boundary_vars + diagnostic_vars + extra_surface_vars:
            if v not in grp:
                raise KeyError(f"{path}: missing surface/var dataset {v!r}")
            out[v] = grp[v][...]

        if include_constants:
            for v in constant_boundary_vars:
                if v not in grp:
                    raise KeyError(f"{path}: missing constant boundary {v!r}")
                out[v] = grp[v][...]

        # Reference shape for upper-air cube allocation.
        ref_shape = grp[surface_vars[0]].shape if surface_vars else (
            grp[constant_boundary_vars[0]].shape if constant_boundary_vars else None
        )
        if ref_shape is None and (pressure_upper_vars or extra_pressure_upper_vars):
            raise RuntimeError(
                "Cannot infer (lat, lon) shape: surface_variables and "
                "constant_boundary_variables are both empty but upper-air vars "
                "were requested."
            )

        for v in pressure_upper_vars + extra_pressure_upper_vars:
            cube = np.empty((len(pressure_levels), *ref_shape), dtype="float32")
            for k, lev in enumerate(pressure_levels):
                cube[k] = grp[_level_key(grp, v, lev)][...]
            out[v] = cube

    return out


def convert(
    config: dict,
    *,
    input_dir: Path,
    year: int,
    sample_range: tuple[int, int] | None,
    output: Path,
    time_chunk: int,
    write_batch: int = 50,
) -> None:
    surface_vars = list(config.get("surface_variables", []) or [])
    pressure_upper_vars = list(config.get("pressure_upper_air_variables", []) or [])
    constant_boundary_vars = list(config.get("constant_boundary_variables", []) or [])
    varying_boundary_vars = list(config.get("varying_boundary_variables", []) or [])
    diagnostic_vars = list(config.get("diagnostic_variables", []) or [])
    # extra_variables holds names written to the Zarr but NOT added to any
    # role-list attr; downstream dataset readers ignore them by default.
    extra_block = config.get("extra_variables", {}) or {}
    extra_surface_vars = list(extra_block.get("surface_variables", []) or [])
    extra_pressure_upper_vars = list(
        extra_block.get("pressure_upper_air_variables", []) or []
    )
    pressure_levels = [float(x) for x in config.get("levels", [])]
    calendar = config.get("calendar", "standard")
    horizontal_resolution = list(config["horizontal_resolution"])
    lat = np.array(config["lat"], dtype="float32")
    lon = np.array(config["lon"], dtype="float32")
    timedelta_hours = int(
        config.get("data_timedelta_hours", config.get("timedelta_hours", 6))
    )

    if lat.shape != (horizontal_resolution[0],):
        raise ValueError(
            f"lat length {lat.shape[0]} != horizontal_resolution[0]={horizontal_resolution[0]}"
        )
    if lon.shape != (horizontal_resolution[1],):
        raise ValueError(
            f"lon length {lon.shape[0]} != horizontal_resolution[1]={horizontal_resolution[1]}"
        )

    files = _list_files(input_dir, year, sample_range)
    logger.info(
        "Found %d files in %s for year %d, sample range %s",
        len(files), input_dir, year, sample_range,
    )

    # --- Streaming write strategy ---------------------------------------
    # 1. Read just timestamps from every file (cheap — ~1000× faster than
    #    the full payload read) so the time coord is fully known up front.
    # 2. Read the first file in full to (a) pin the (lat, lon) shape and
    #    (b) capture the constant-boundary fields.
    # 3. Allocate a dask-zeros template Dataset spanning the full time
    #    axis. Write it with ``compute=False`` — only the Zarr metadata
    #    is materialized, no data block reads.
    # 4. Iterate over the file list in batches of ``write_batch``
    #    timesteps. For each batch: read N files, build a per-batch
    #    Dataset, write it via ``to_zarr(region={"time": slice(...)})``.
    # Peak memory is bounded by ``write_batch`` × per-timestep payload
    # rather than scaling with the full year.

    import time as _time

    import dask.array as da

    n_time = len(files)
    logger.info("Pass 1/2: reading %d timestamps for the time coord", n_time)
    t_phase = _time.time()
    times: list[cftime.datetime] = []
    for k, path in enumerate(files):
        times.append(_read_time_only(path, calendar))
        if (k + 1) % 200 == 0:
            logger.info("  %d/%d timestamps read", k + 1, n_time)
    logger.info("  done in %.1fs", _time.time() - t_phase)

    logger.info("Reading first file for constants + reference shape")
    first = _read_one_file(
        files[0],
        surface_vars=surface_vars,
        pressure_upper_vars=pressure_upper_vars,
        constant_boundary_vars=constant_boundary_vars,
        varying_boundary_vars=varying_boundary_vars,
        diagnostic_vars=diagnostic_vars,
        extra_surface_vars=extra_surface_vars,
        extra_pressure_upper_vars=extra_pressure_upper_vars,
        pressure_levels=pressure_levels,
        calendar=calendar,
        include_constants=True,
    )
    constants: dict[str, np.ndarray] = {
        v: first[v] for v in constant_boundary_vars
    }

    n_lat = lat.shape[0]
    n_lon = lon.shape[0]
    n_levels = len(pressure_levels)

    # Build dask-backed template with the full time axis.
    time_chunk_eff = min(max(1, time_chunk), n_time)
    surface_chunks = (time_chunk_eff, n_lat, n_lon)
    upper_chunks = (time_chunk_eff, n_levels, n_lat, n_lon)
    surface_shape = (n_time, n_lat, n_lon)
    upper_shape = (n_time, n_levels, n_lat, n_lon)

    data_vars: dict[str, tuple] = {}
    for v in surface_vars + extra_surface_vars + varying_boundary_vars + diagnostic_vars:
        data_vars[v] = (
            ("time", "lat", "lon"),
            da.zeros(surface_shape, chunks=surface_chunks, dtype="float32"),
        )
    for v in pressure_upper_vars + extra_pressure_upper_vars:
        data_vars[v] = (
            ("time", "pressure_level", "lat", "lon"),
            da.zeros(upper_shape, chunks=upper_chunks, dtype="float32"),
        )
    for v in constant_boundary_vars:
        data_vars[v] = (("lat", "lon"), constants[v])

    coords: dict[str, object] = {
        "time": ("time", times),
        "lat": ("lat", lat),
        "lon": ("lon", lon),
    }
    if pressure_upper_vars or extra_pressure_upper_vars:
        coords["pressure_level"] = (
            "pressure_level",
            np.array(pressure_levels, dtype="float32"),
        )

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "climate_zarr_schema_version": CLIMATE_ZARR_SCHEMA_VERSION,
        "calendar": calendar,
        "data_timedelta_hours": timedelta_hours,
        "source_config": str(config.get("__source_path__", "")),
        # Role lists exclude extra_variables intentionally — the dataset
        # class won't route them through any group unless a downstream
        # YAML adds them back.
        "surface_variables": list(surface_vars),
        "constant_boundary_variables": list(constant_boundary_vars),
        "varying_boundary_variables": list(varying_boundary_vars),
        "diagnostic_variables": list(diagnostic_vars),
        "pressure_upper_air_variables": list(pressure_upper_vars),
        "sigma_upper_air_variables": [],
        # Document which extras landed in the Zarr (informational only).
        "extra_surface_variables": list(extra_surface_vars),
        "extra_pressure_upper_air_variables": list(extra_pressure_upper_vars),
        "year_index": int(year),
        "sample_range": list(sample_range) if sample_range else "all",
    }

    chunk_spec = {
        "time": time_chunk_eff,
        "lat": n_lat,
        "lon": n_lon,
    }
    if "pressure_level" in ds.sizes:
        chunk_spec["pressure_level"] = n_levels
    encoding: dict[str, dict] = {}
    for name in ds.data_vars:
        chunks = tuple(chunk_spec[d] for d in ds[name].dims)
        encoding[name] = {"chunks": chunks}

    logger.info("Allocating Zarr template at %s", output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    # ``compute=False`` writes only the Zarr metadata + creates empty
    # chunks; no dask graph is executed. Constants + coords are written
    # eagerly (they're small numpy arrays in ``ds``).
    ds.to_zarr(
        output,
        mode="w",
        encoding=encoding,
        zarr_format=3,
        consolidated=True,
        compute=False,
    )

    # --- Phase 2/2: stream batched data writes --------------------------
    logger.info(
        "Pass 2/2: streaming %d timesteps in batches of %d (peak mem ≈ "
        "%d × per-timestep payload)",
        n_time, write_batch, write_batch,
    )
    t_phase = _time.time()
    write_batch = max(1, int(write_batch))
    n_time_vars_3d = surface_vars + extra_surface_vars + varying_boundary_vars + diagnostic_vars
    n_time_vars_4d = pressure_upper_vars + extra_pressure_upper_vars

    for batch_start in range(0, n_time, write_batch):
        batch_end = min(batch_start + write_batch, n_time)
        bsize = batch_end - batch_start
        # Pre-allocate batch buffers — sized exactly to the batch slice
        # so the peak memory at any moment is ``write_batch`` timesteps,
        # not the full year.
        buf_3d = {
            v: np.empty((bsize, n_lat, n_lon), dtype="float32")
            for v in n_time_vars_3d
        }
        buf_4d = {
            v: np.empty((bsize, n_levels, n_lat, n_lon), dtype="float32")
            for v in n_time_vars_4d
        }
        for k_local, k_global in enumerate(range(batch_start, batch_end)):
            data = _read_one_file(
                files[k_global],
                surface_vars=surface_vars,
                pressure_upper_vars=pressure_upper_vars,
                constant_boundary_vars=constant_boundary_vars,
                varying_boundary_vars=varying_boundary_vars,
                diagnostic_vars=diagnostic_vars,
                extra_surface_vars=extra_surface_vars,
                extra_pressure_upper_vars=extra_pressure_upper_vars,
                pressure_levels=pressure_levels,
                calendar=calendar,
                include_constants=False,
            )
            for v in n_time_vars_3d:
                buf_3d[v][k_local] = data[v]
            for v in n_time_vars_4d:
                buf_4d[v][k_local] = data[v]

        # Build a region-shaped Dataset (no coords for time — already
        # written from the template, and including them re-triggers a
        # coord write that xarray will reject in region mode).
        batch_vars: dict[str, tuple] = {}
        for v in n_time_vars_3d:
            batch_vars[v] = (("time", "lat", "lon"), buf_3d[v])
        for v in n_time_vars_4d:
            batch_vars[v] = (("time", "pressure_level", "lat", "lon"), buf_4d[v])
        batch_ds = xr.Dataset(batch_vars)
        batch_ds.to_zarr(
            output,
            region={"time": slice(batch_start, batch_end)},
        )
        logger.info(
            "  wrote timesteps %d..%d / %d", batch_start, batch_end, n_time
        )

    logger.info(
        "Done. Total variables: %d, times: %d (write phase %.1fs)",
        len(ds.data_vars), n_time, _time.time() - t_phase,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    config = _read_config(args.config, args.config_key)
    config["__source_path__"] = str(args.config.resolve())
    input_dir = args.input_dir or Path(config["data_dir"])
    sample_range = tuple(args.sample_range) if args.sample_range else None
    convert(
        config,
        input_dir=input_dir,
        year=args.year,
        sample_range=sample_range,
        output=args.output,
        time_chunk=args.time_chunk,
        write_batch=args.write_batch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
