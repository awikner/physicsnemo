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

"""Convert PanguWeather v2.0 per-timestep PLASIM HDF5 files to a single PLASIM
Zarr store consumable by :class:`physicsnemo.experimental.datapipes.plasim.PlasimClimateDatapipe`.

Reads a PanguWeather YAML config to pick up channel structure (surface,
upper-air × sigma + pressure levels, constant/varying boundaries, diagnostic
variables) and a contiguous range of per-timestep ``{year}_{idx:04d}.h5``
files, then writes them into an xarray-backed Zarr v3 store with the schema:

* ``time``: (T,) cftime datetimes parsed from the per-file ``input/time`` scalar.
* ``lat``, ``lon``: (H,), (W,) coordinate axes (taken from the YAML).
* ``pressure_level``: (L_p,) coordinate axis (Pa, taken from the YAML ``levels``).
* ``sigma_level``: (L_s,) coordinate axis (taken from the YAML ``sigma_levels``).
* Surface variables: ``(time, lat, lon)``.
* Pressure-level upper-air variables: ``(time, pressure_level, lat, lon)``.
* Sigma-level upper-air variables: ``(time, sigma_level, lat, lon)``.
* Constant boundaries: ``(lat, lon)`` (assumed constant in time; first file used).
* Varying boundaries: ``(time, lat, lon)``.
* Diagnostic variables: ``(time, lat, lon)``.

Zarr store attributes record the calendar, the data timedelta, the source
config, the variable-group bookkeeping, and a schema version. The
:class:`PlasimClimateDatapipe` consumes the store via ``xarray.open_zarr`` and
introspects these attributes — no flag duplication at the datapipe site.

Usage
-----
::

    python tools/data/plasim/pangu_h5_to_zarr.py \\
      --config /work/nvme/bdiu/awikner/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_5412.yaml \\
      --config-key PLASIM \\
      --year 100 --sample-range 0 120 \\
      --output /work/nvme/bdiu/awikner/physicsnemo_test_data/plasim/smoke_month.zarr
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

logger = logging.getLogger(__name__)


PLASIM_ZARR_SCHEMA_VERSION = "1.0"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PanguWeather v2.0 PLASIM per-timestep HDF5 to a Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="PanguWeather YAML config.")
    p.add_argument(
        "--config-key",
        default="PLASIM",
        help="Top-level YAML key to read channel/level config from.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Source dir containing {year}_{idx:04d}.h5. Defaults to config's data_dir.",
    )
    p.add_argument("--year", type=int, required=True, help="Year-index to convert.")
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
        default=50,
        help="Number of timesteps per Zarr chunk along the time axis.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _read_config(path: Path, key: str) -> dict:
    with path.open() as f:
        raw = yaml.safe_load(f)
    if key not in raw:
        raise KeyError(f"Config {path} has no top-level key {key!r}; got {list(raw)}")
    return raw[key]


def _list_files(input_dir: Path, year: int, sample_range: tuple[int, int] | None) -> list[Path]:
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
    """Find the HDF5 dataset name matching ``{prefix}_{level}`` within numerical
    tolerance, since the keys store the level as a Python-formatted float string
    that may not be reproducible by ``f"{level}"``.
    """
    rtol, atol = 1e-3, 1e-6
    candidates: list[tuple[float, str]] = []
    for name in group:
        if not name.startswith(prefix + "_"):
            continue
        try:
            val = float(name[len(prefix) + 1 :])
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
    datetime. PanguWeather stores it as a string like ``2100-01-01 00:00:00``
    (encoding the calendar choice; here we just rely on the YAML's ``calendar``).
    """
    if isinstance(time_dataset_value, bytes):
        s = time_dataset_value.decode("ascii")
    else:
        s = str(time_dataset_value)
    # Accepts ISO-ish "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS".
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
    cls = cftime._cftime.DATE_TYPES.get(calendar, cftime.DatetimeProlepticGregorian)
    return cls(y, mo, d, h, mi, sec, micros)


def _read_one_file(
    path: Path,
    *,
    surface_vars: list[str],
    pressure_upper_vars: list[str],
    sigma_upper_vars: list[str],
    constant_boundary_vars: list[str],
    varying_boundary_vars: list[str],
    diagnostic_vars: list[str],
    pressure_levels: list[float],
    sigma_levels: list[float],
    calendar: str,
    include_constants: bool,
) -> dict[str, np.ndarray]:
    """Read one PLASIM per-timestep HDF5 file, return a dict of per-channel
    arrays + a decoded ``time`` value.
    """
    out: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        grp = f["input"]
        out["time"] = _decode_time(grp["time"][()], calendar)

        for v in surface_vars + varying_boundary_vars + diagnostic_vars:
            if v not in grp:
                raise KeyError(f"{path}: missing surface/var dataset {v!r}")
            out[v] = grp[v][...]

        if include_constants:
            for v in constant_boundary_vars:
                if v not in grp:
                    raise KeyError(f"{path}: missing constant boundary {v!r}")
                out[v] = grp[v][...]

        for v in pressure_upper_vars:
            cube = np.empty((len(pressure_levels), *grp[surface_vars[0]].shape), dtype="float32")
            for k, lev in enumerate(pressure_levels):
                cube[k] = grp[_level_key(grp, v, lev)][...]
            out[v] = cube

        for v in sigma_upper_vars:
            cube = np.empty((len(sigma_levels), *grp[surface_vars[0]].shape), dtype="float32")
            for k, lev in enumerate(sigma_levels):
                cube[k] = grp[_level_key(grp, v, lev)][...]
            out[v] = cube

    return out


def _split_upper_air_vars(
    upper_air_variables: list[str],
    use_sigma_levels: bool,
) -> tuple[list[str], list[str]]:
    """PanguWeather v2.0 PLASIM convention: when ``use_sigma_levels=True``,
    every upper-air variable except ``zg`` (geopotential height) lives on
    sigma levels; ``zg`` always lives on pressure levels.
    """
    pressure_vars: list[str] = []
    sigma_vars: list[str] = []
    for v in upper_air_variables:
        if v == "zg":
            pressure_vars.append(v)
        elif use_sigma_levels:
            sigma_vars.append(v)
        else:
            pressure_vars.append(v)
    return pressure_vars, sigma_vars


def convert(
    config: dict,
    *,
    input_dir: Path,
    year: int,
    sample_range: tuple[int, int] | None,
    output: Path,
    time_chunk: int,
) -> None:
    surface_vars = list(config["surface_variables"])
    upper_air_vars = list(config["upper_air_variables"])
    constant_boundary_vars = list(config.get("constant_boundary_variables", []) or [])
    varying_boundary_vars = list(config.get("varying_boundary_variables", []) or [])
    diagnostic_vars = list(config.get("diagnostic_variables", []) or [])
    pressure_levels = [float(x) for x in config.get("levels", [])]
    sigma_levels = [float(x) for x in config.get("sigma_levels", [])]
    use_sigma_levels = bool(config.get("use_sigma_levels", False))
    pressure_upper_vars, sigma_upper_vars = _split_upper_air_vars(
        upper_air_vars, use_sigma_levels
    )
    calendar = config.get("calendar", "proleptic_gregorian")
    horizontal_resolution = list(config["horizontal_resolution"])
    lat = np.array(config["lat"], dtype="float32")
    lon = np.array(config["lon"], dtype="float32")
    timedelta_hours = int(config.get("data_timedelta_hours", config.get("timedelta_hours", 6)))

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
        len(files),
        input_dir,
        year,
        sample_range,
    )

    times: list[cftime.datetime] = []
    surface_stacks: dict[str, list[np.ndarray]] = {v: [] for v in surface_vars}
    varying_stacks: dict[str, list[np.ndarray]] = {v: [] for v in varying_boundary_vars}
    diagnostic_stacks: dict[str, list[np.ndarray]] = {v: [] for v in diagnostic_vars}
    pressure_stacks: dict[str, list[np.ndarray]] = {v: [] for v in pressure_upper_vars}
    sigma_stacks: dict[str, list[np.ndarray]] = {v: [] for v in sigma_upper_vars}
    constants: dict[str, np.ndarray] = {}

    for k, path in enumerate(files):
        if k % 50 == 0:
            logger.info("Reading %d/%d %s", k + 1, len(files), path.name)
        data = _read_one_file(
            path,
            surface_vars=surface_vars,
            pressure_upper_vars=pressure_upper_vars,
            sigma_upper_vars=sigma_upper_vars,
            constant_boundary_vars=constant_boundary_vars,
            varying_boundary_vars=varying_boundary_vars,
            diagnostic_vars=diagnostic_vars,
            pressure_levels=pressure_levels,
            sigma_levels=sigma_levels,
            calendar=calendar,
            include_constants=(k == 0),
        )
        times.append(data["time"])
        for v in surface_vars:
            surface_stacks[v].append(data[v])
        for v in varying_boundary_vars:
            varying_stacks[v].append(data[v])
        for v in diagnostic_vars:
            diagnostic_stacks[v].append(data[v])
        for v in pressure_upper_vars:
            pressure_stacks[v].append(data[v])
        for v in sigma_upper_vars:
            sigma_stacks[v].append(data[v])
        if k == 0:
            for v in constant_boundary_vars:
                constants[v] = data[v]

    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, object] = {
        "time": ("time", times),
        "lat": ("lat", lat),
        "lon": ("lon", lon),
    }
    if pressure_upper_vars:
        coords["pressure_level"] = ("pressure_level", np.array(pressure_levels, dtype="float32"))
    if sigma_upper_vars:
        coords["sigma_level"] = ("sigma_level", np.array(sigma_levels, dtype="float32"))

    for v in surface_vars:
        data_vars[v] = xr.DataArray(np.stack(surface_stacks[v]), dims=("time", "lat", "lon"))
    for v in varying_boundary_vars:
        data_vars[v] = xr.DataArray(np.stack(varying_stacks[v]), dims=("time", "lat", "lon"))
    for v in diagnostic_vars:
        data_vars[v] = xr.DataArray(np.stack(diagnostic_stacks[v]), dims=("time", "lat", "lon"))
    for v in pressure_upper_vars:
        data_vars[v] = xr.DataArray(
            np.stack(pressure_stacks[v]),
            dims=("time", "pressure_level", "lat", "lon"),
        )
    for v in sigma_upper_vars:
        data_vars[v] = xr.DataArray(
            np.stack(sigma_stacks[v]),
            dims=("time", "sigma_level", "lat", "lon"),
        )
    for v in constant_boundary_vars:
        data_vars[v] = xr.DataArray(constants[v], dims=("lat", "lon"))

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "plasim_zarr_schema_version": PLASIM_ZARR_SCHEMA_VERSION,
        "calendar": calendar,
        "data_timedelta_hours": timedelta_hours,
        "source_config": str(config.get("__source_path__", "")),
        "surface_variables": list(surface_vars),
        "constant_boundary_variables": list(constant_boundary_vars),
        "varying_boundary_variables": list(varying_boundary_vars),
        "diagnostic_variables": list(diagnostic_vars),
        "pressure_upper_air_variables": list(pressure_upper_vars),
        "sigma_upper_air_variables": list(sigma_upper_vars),
        "year_index": int(year),
        "sample_range": list(sample_range) if sample_range else "all",
    }

    # Chunking — Zarr v3 takes chunk shapes per data_var via encoding; we don't
    # need dask-backed dask-array chunking (which would require the optional
    # `dask` dep), so we pass chunks via `encoding` and let xarray's serializer
    # split the numpy arrays at write time.
    chunk_spec = {
        "time": min(time_chunk, ds.sizes["time"]),
        "lat": ds.sizes["lat"],
        "lon": ds.sizes["lon"],
    }
    if "pressure_level" in ds.sizes:
        chunk_spec["pressure_level"] = ds.sizes["pressure_level"]
    if "sigma_level" in ds.sizes:
        chunk_spec["sigma_level"] = ds.sizes["sigma_level"]

    encoding: dict[str, dict] = {}
    for name in ds.data_vars:
        chunks = tuple(chunk_spec[d] for d in ds[name].dims)
        encoding[name] = {"chunks": chunks}

    logger.info("Writing %s", output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    ds.to_zarr(output, mode="w", encoding=encoding, zarr_format=3, consolidated=True)
    logger.info("Done. Total variables: %d, times: %d", len(ds.data_vars), ds.dims["time"])


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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
