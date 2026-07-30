#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OFFLINE: harmonize hindcast archives into the unified MoWE store schema.

One output schema for every expert, ``{out_root}/{model}/{YYYY}.zarr``:

* dims ``(init_time, lead_time, lat, lon)`` — ``lead_time`` in whole days as
  coordinate VALUES (0 = the IC where present), on the 1-degree IMERG/ERA5
  grid (``--ref-store``);
* every variable a flat 2-D field named by the ERA5 long name with an
  integer-style ``_{level}`` suffix for pressure levels
  (``geopotential_500``), with a per-variable ``units`` attr;
* accumulated / time-mean variables refer to the **preceding 24 h** in ERA5
  units (``total_precipitation_24hr`` in m,
  ``mean_top_net_long_wave_radiation_flux`` in W m**-2); leads whose 24 h
  window is unavailable (e.g. lead 0) are NaN.

Two source modes:

``--source dsi``
    0.25-degree DSI stores (``dsi_hindcast_to_formats.py`` Format 2):
    conservative regrid + rename (flat short names -> canonical) + sampling
    of 6h-axis variables at the 24h-multiple leads +
    ``total_precipitation_6hr`` -> trailing-24h accumulation. Multiple
    ``--src-root``s merge into ONE store per year (priority order: the
    first root wins per (init, variable); union of inits and variables;
    an ``init_source`` coord records each init's primary source).

``--source consolidated``
    1-degree consolidated stores (``consolidate_hindcasts.py``): subset to
    ``--variables`` (canonical names, ``name_level`` for upper air), flatten
    3-D pressure-level variables to per-level 2-D fields, relabel
    ``lead_time`` day indices as day values, NaN the lead-0 diagnostics.

Login-node safe: numpy / xarray / zarr / cftime only (no torch, dask, or
physicsnemo). Idempotent via ``.harmonize_done/<model>_<year>.done``
sentinels; a partial store without its sentinel is wiped and redone.

Usage (Derecho)::

    # DSI, merged graphcast (e2s wins over wb2):
    python harmonize_hindcasts.py --source dsi --model graphcast \\
        --src-root .../hindcasts_dsi/zarr/graphcast_e2s \\
        --src-root .../hindcasts_dsi/zarr/graphcast_wb2 \\
        --out-root .../hindcasts_mowe --ref-store .../imerg/2001.zarr \\
        --years 2000-2024 --n-workers 8

    # Consolidated subset:
    python harmonize_hindcasts.py --source consolidated --model pangu_s2s \\
        --src-root .../physicsnemo-zarr/hindcasts/pangu_s2s \\
        --variables-file mowe_subset_variables.txt \\
        --out-root .../hindcasts_mowe --ref-store .../imerg/2001.zarr
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cftime
import numpy as np
import xarray as xr
import zarr

_RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from datapipes.regrid import Regridder, grids_equal  # noqa: E402
from datapipes.variables import (  # noqa: E402
    CANONICAL_UNITS,
    harmonized_name,
    levels_match,
    parse_flat_name,
)

logger = logging.getLogger("harmonize_hindcasts")

SCRIPT_REL_PATH = "examples/weather/ai_rossbypalooza/tools/harmonize_hindcasts.py"
SCHEMA_VERSION = "1.0"
#: Variables with trailing-24h (accumulation / mean-rate) semantics.
TRAILING_24H_VARS = {
    "total_precipitation_24hr",
    "mean_top_net_long_wave_radiation_flux",
}
#: Native precip-like names and how they map (per-variable, per-axis).
PRECIP_6H_NAMES = {"total_precipitation_6hr", "tp_6hr"}


def _register_codecs() -> None:
    try:
        import numcodecs.zarr3  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"reading DSI stores requires numcodecs zarr3 codecs ({exc})"
        ) from exc


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _units_for(harmonized: str) -> str | None:
    parsed = parse_flat_name(harmonized)
    if parsed is None:
        return None
    return CANONICAL_UNITS.get(parsed[0])


def _open(store: Path) -> xr.Dataset:
    return xr.open_zarr(
        store,
        consolidated=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    )


# --------------------------------------------------------------------------- #
# DSI source: per-store variable plans
# --------------------------------------------------------------------------- #
class DsiSource:
    """One DSI yearly store: name mapping + per-(var, day) read/assemble.

    ``keep`` restricts the harmonized output to that set of names (the
    user's master variable list); None keeps everything mappable.
    """

    def __init__(self, label: str, store: Path, keep: set[str] | None = None) -> None:
        self.label = label
        self.store = store
        self.ds = _open(store)
        self.inits = {
            (t.year, t.month, t.day, t.hour): i
            for i, t in enumerate(self.ds["init_time"].values)
        }
        self.hours = (
            {int(v): i for i, v in enumerate(self.ds["prediction_timedelta"].values)}
            if "prediction_timedelta" in self.ds.coords
            else {}
        )
        self.days = (
            {int(v): i for i, v in enumerate(self.ds["prediction_timedelta_daily"].values)}
            if "prediction_timedelta_daily" in self.ds.coords
            else {}
        )
        v6 = [str(v) for v in self.ds.attrs.get("channel_variables_6h", [])]
        vd = [str(v) for v in self.ds.attrs.get("channel_variables_daily", [])]
        #: harmonized name -> (native name, axis)
        self.mapping: dict[str, tuple[str, str]] = {}
        for native, axis in [*[(v, "6h") for v in v6], *[(v, "daily") for v in vd]]:
            if native in PRECIP_6H_NAMES:
                out = "total_precipitation_24hr"
            else:
                parsed = parse_flat_name(native)
                if parsed is None:
                    logger.info("%s: unmapped variable '%s' skipped", store, native)
                    continue
                out = harmonized_name(*parsed)
            if keep is not None and out not in keep:
                logger.debug(
                    "%s: '%s' -> '%s' not in the variable list, dropped",
                    store,
                    native,
                    out,
                )
                continue
            if out in self.mapping:
                logger.warning(
                    "%s: '%s' duplicates harmonized '%s'", store, native, out
                )
                continue
            self.mapping[out] = (native, axis)

    def lead_days(self) -> list[int]:
        days = set(self.days)
        days.update(h // 24 for h in self.hours if h % 24 == 0)
        return sorted(days)

    def read(self, out_var: str, init_idx: int, days: list[int]) -> np.ndarray:
        """(n_days, H, W) at native resolution; NaN where unavailable."""
        native, axis = self.mapping[out_var]
        arr = self.ds[native].isel(init_time=init_idx).values  # (n_lead, H, W)
        h, w = arr.shape[-2:]
        out = np.full((len(days), h, w), np.nan, dtype=np.float32)
        if native in PRECIP_6H_NAMES:
            step = 6
            for k, d in enumerate(days):
                idx = [
                    self.hours.get(d * 24 - step * i)
                    for i in range(24 // step - 1, -1, -1)
                ]
                if all(i is not None for i in idx):
                    out[k] = arr[idx].sum(axis=0)
        elif axis == "6h":
            for k, d in enumerate(days):
                i = self.hours.get(d * 24)
                if i is not None:
                    out[k] = arr[i]
        else:
            for k, d in enumerate(days):
                i = self.days.get(d)
                if i is not None:
                    out[k] = arr[i]
        return out

    def close(self) -> None:
        self.ds.close()


def harmonize_dsi_year(
    src_stores: list[tuple[str, Path]],
    dst_store: Path,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    *,
    model: str,
    commit: str,
    overwrite: bool,
    keep: set[str] | None = None,
) -> dict:
    _register_codecs()
    sources = [DsiSource(label, p, keep=keep) for label, p in src_stores]
    try:
        # Union of harmonized variables; the first source listing a variable
        # defines it (values still come from the highest-priority source
        # that has the (init, variable)).
        variables: list[str] = []
        for s in sources:
            for v in s.mapping:
                if v not in variables:
                    variables.append(v)
        variables.sort()
        if not variables:
            raise ValueError(
                f"{dst_store}: no source variable survives the variable list"
            )
        days = sorted({d for s in sources for d in s.lead_days()})
        # Union of inits; primary source per init = first source having it.
        init_keys = sorted({k for s in sources for k in s.inits})
        year = init_keys[0][0]
        init_dts = [cftime.DatetimeGregorian(*k) for k in init_keys]
        init_src = [
            next(s.label for s in sources if k in s.inits) for k in init_keys
        ]
        n_lat, n_lon = len(dst_lat), len(dst_lon)

        src_grid = sources[0].ds["lat"].values.astype("float64")
        needs_regrid = not (
            grids_equal(src_grid, dst_lat)
            and grids_equal(sources[0].ds["lon"].values, dst_lon)
        )
        regridder = (
            Regridder(
                sources[0].ds["lat"].values,
                sources[0].ds["lon"].values,
                dst_lat,
                dst_lon,
            )
            if needs_regrid
            else None
        )
        for s in sources[1:]:
            if not grids_equal(
                s.ds["lat"].values.astype("float64"), src_grid
            ):
                raise ValueError(
                    f"{s.store}: source grids differ between merged roots"
                )

        attrs = _output_attrs(
            model=model,
            sources=[str(p) for _, p in src_stores],
            variables=variables,
            commit=commit,
            regridded=needs_regrid,
        )
        encoding = {
            v: {"chunks": (1, len(days), n_lat, n_lon), "dtype": "float32"}
            for v in variables
        }
        encoding["init_time"] = {
            "units": f"hours since {year}-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "int64",
        }
        dst_store.parent.mkdir(parents=True, exist_ok=True)
        nan_only: set[str] = set()
        for i, key in enumerate(init_keys):
            step_vars = {}
            for v in variables:
                src = next(
                    (s for s in sources if key in s.inits and v in s.mapping),
                    None,
                )
                if src is None:
                    slab = np.full(
                        (len(days), n_lat, n_lon), np.nan, dtype=np.float32
                    )
                else:
                    raw = src.read(v, src.inits[key], days)
                    slab = regridder(raw) if regridder is not None else raw
                if not np.isfinite(slab).any():
                    nan_only.add(v)
                step_vars[v] = (
                    ("init_time", "lead_time", "lat", "lon"),
                    slab[np.newaxis].astype(np.float32),
                )
            _append_init(
                dst_store,
                step_vars,
                init_dt=init_dts[i],
                init_source=init_src[i],
                days=days,
                lat=dst_lat,
                lon=dst_lon,
                attrs=attrs,
                encoding=encoding,
                first=(i == 0),
                overwrite=overwrite,
            )
        zarr.consolidate_metadata(str(dst_store))
        return {
            "store": str(dst_store),
            "n_init": len(init_keys),
            "variables": len(variables),
            "nan_only_vars_somewhere": sorted(nan_only),
        }
    finally:
        for s in sources:
            s.close()


# --------------------------------------------------------------------------- #
# Consolidated source (pangu/sfno subset + flatten)
# --------------------------------------------------------------------------- #
def _resolve_subset(ds: xr.Dataset, variables: list[str]) -> dict[str, tuple]:
    """harmonized name -> ("scalar", native) | ("level", native, level_idx)."""
    levels = (
        [float(v) for v in ds["pressure_level"].values]
        if "pressure_level" in ds.coords
        else []
    )
    out: dict[str, tuple] = {}
    for want in variables:
        if want in ds.data_vars:
            out[want] = ("scalar", want)
            continue
        parsed = parse_flat_name(want)
        if parsed is None or parsed[1] is None:
            raise ValueError(f"requested variable '{want}' not in the store")
        canonical, level = parsed
        if canonical not in ds.data_vars:
            raise ValueError(
                f"requested variable '{want}': '{canonical}' not in the store"
            )
        matches = [j for j, lv in enumerate(levels) if levels_match(lv, level)]
        if not matches:
            raise ValueError(
                f"requested variable '{want}': level {level} not in "
                f"pressure_level {levels}"
            )
        out[harmonized_name(canonical, level)] = ("level", canonical, matches[0])
    return out


def harmonize_consolidated_year(
    src_store: Path,
    dst_store: Path,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    *,
    model: str,
    variables: list[str],
    commit: str,
    overwrite: bool,
) -> dict:
    ds = _open(src_store)
    try:
        if not (
            grids_equal(ds["lat"].values.astype("float64"), dst_lat)
            and grids_equal(ds["lon"].values.astype("float64"), dst_lon)
        ):
            raise ValueError(f"{src_store} is not on the reference 1-deg grid")
        subset = _resolve_subset(ds, variables)
        out_vars = sorted(subset)
        days = [int(v) for v in ds["lead_time"].values]  # index == day value
        init_dts = list(ds["init_time"].values)
        year = int(init_dts[0].year)
        n_lat, n_lon = len(dst_lat), len(dst_lon)

        attrs = _output_attrs(
            model=model,
            sources=[str(src_store)],
            variables=out_vars,
            commit=commit,
            regridded=False,
        )
        encoding = {
            v: {"chunks": (1, len(days), n_lat, n_lon), "dtype": "float32"}
            for v in out_vars
        }
        encoding["init_time"] = {
            "units": f"hours since {year}-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "int64",
        }
        dst_store.parent.mkdir(parents=True, exist_ok=True)
        upper_natives = sorted({s[1] for s in subset.values() if s[0] == "level"})
        for i in range(len(init_dts)):
            # One chunk decode per 3-D native variable per init; every
            # requested level slices from the cached array.
            upper_cache = {
                nv: ds[nv].isel(init_time=i).values for nv in upper_natives
            }
            step_vars = {}
            for v in out_vars:
                spec = subset[v]
                if spec[0] == "scalar":
                    slab = ds[spec[1]].isel(init_time=i).values
                else:
                    slab = upper_cache[spec[1]][:, spec[2]]
                slab = np.asarray(slab, dtype=np.float32)
                if v in TRAILING_24H_VARS and days and days[0] == 0:
                    # Lead 0 is the IC: a trailing-24h forecast is undefined.
                    slab[0] = np.nan
                step_vars[v] = (
                    ("init_time", "lead_time", "lat", "lon"),
                    slab[np.newaxis],
                )
            _append_init(
                dst_store,
                step_vars,
                init_dt=init_dts[i],
                init_source=None,
                days=days,
                lat=dst_lat,
                lon=dst_lon,
                attrs=attrs,
                encoding=encoding,
                first=(i == 0),
                overwrite=overwrite,
            )
        zarr.consolidate_metadata(str(dst_store))
        return {
            "store": str(dst_store),
            "n_init": len(init_dts),
            "variables": len(out_vars),
        }
    finally:
        ds.close()


# --------------------------------------------------------------------------- #
# shared writer
# --------------------------------------------------------------------------- #
def _output_attrs(*, model, sources, variables, commit, regridded) -> dict:
    attrs = {
        "mowe_hindcast_schema_version": SCHEMA_VERSION,
        "model": model,
        "sources": list(sources),
        "calendar": "standard",
        "variables": list(variables),
        "note": (
            "unified MoWE hindcast schema: flat ERA5 long names "
            "(integer _level suffix), lead_time in whole days as values, "
            "accumulated/mean variables refer to the PRECEDING 24 h in "
            "ERA5 units (tp in m), lat N->S on the 1-deg IMERG/ERA5 grid."
        ),
        "generator": f"{SCRIPT_REL_PATH}@{commit}",
    }
    if regridded:
        attrs["regrid_method"] = "1d-conservative"
    return attrs


def _append_init(
    dst_store: Path,
    step_vars: dict,
    *,
    init_dt,
    init_source: str | None,
    days: list[int],
    lat: np.ndarray,
    lon: np.ndarray,
    attrs: dict,
    encoding: dict,
    first: bool,
    overwrite: bool,
) -> None:
    coords = {
        "init_time": ("init_time", np.asarray([init_dt])),
        "lead_time": ("lead_time", np.asarray(days, dtype="int64")),
        "lat": ("lat", np.asarray(lat, dtype="float32")),
        "lon": ("lon", np.asarray(lon, dtype="float32")),
    }
    if init_source is not None:
        coords["init_source"] = ("init_time", np.asarray([init_source]))
    step = xr.Dataset(step_vars, coords=coords, attrs=attrs)
    step["lead_time"].attrs.update(
        {"units": "days", "long_name": "forecast lead time (whole days; 0 = IC)"}
    )
    for v in step.data_vars:
        u = _units_for(v)
        if u is not None:
            step[v].attrs["units"] = u
        if v in TRAILING_24H_VARS:
            step[v].attrs["interval"] = "preceding 24 hours"
    if first:
        step.to_zarr(
            dst_store,
            mode="w" if overwrite else "w-",
            zarr_format=3,
            consolidated=False,
            encoding=encoding,
        )
    else:
        step.to_zarr(dst_store, append_dim="init_time", consolidated=False)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _sentinel(out_root: Path, model: str, year: int) -> Path:
    return out_root / ".harmonize_done" / f"{model}_{year}.done"


def _worker(job: dict) -> dict:
    dst = Path(job["dst"])
    try:
        if dst.exists() and not Path(job["sentinel"]).exists():
            logger.warning("removing partial store %s", dst)
            shutil.rmtree(dst)
        if job["source"] == "dsi":
            summary = harmonize_dsi_year(
                [(lbl, Path(p)) for lbl, p in job["src_stores"]],
                dst,
                np.asarray(job["dst_lat"]),
                np.asarray(job["dst_lon"]),
                model=job["model"],
                commit=job["commit"],
                overwrite=job["overwrite"],
                keep=set(job["variables"]) if job["variables"] else None,
            )
        else:
            summary = harmonize_consolidated_year(
                Path(job["src_stores"][0][1]),
                dst,
                np.asarray(job["dst_lat"]),
                np.asarray(job["dst_lon"]),
                model=job["model"],
                variables=job["variables"],
                commit=job["commit"],
                overwrite=job["overwrite"],
            )
        sentinel = Path(job["sentinel"])
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        summary["status"] = "ok"
        return summary
    except Exception as exc:  # noqa: BLE001 - reported to the driver
        return {"store": str(dst), "status": "error", "error": repr(exc)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source", choices=("dsi", "consolidated"), required=True)
    p.add_argument("--model", required=True,
                   help="output model name (e.g. graphcast, pangu_s2s)")
    p.add_argument("--src-root", action="append", required=True, type=Path,
                   help="source root(s) holding {YYYY}.zarr; for dsi, "
                        "repeatable in priority order (first wins)")
    p.add_argument("--src-label", action="append", default=None,
                   help="label per --src-root for the init_source coord "
                        "(default: root basename)")
    p.add_argument("--variables", nargs="+", default=None,
                   help="harmonized variable names to keep (required for "
                        "consolidated; optional filter for dsi)")
    p.add_argument("--variables-file", type=Path, default=None,
                   help="file with one variable name per line (same role)")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--ref-store", type=Path, required=True,
                   help="store providing the target lat/lon (an IMERG year)")
    p.add_argument("--years", default=None, help="e.g. 2000-2024")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--commit", default="unknown")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ref = _open(args.ref_store)
    dst_lat = ref["lat"].values.astype("float64")
    dst_lon = ref["lon"].values.astype("float64")
    ref.close()

    # The variable list is REQUIRED in consolidated mode (defines the
    # subset) and optional in dsi mode (filters the harmonized union down
    # to the master list; unmatched entries are simply absent per model).
    variables: list[str] = []
    if args.variables_file is not None:
        variables += [
            ln.strip()
            for ln in args.variables_file.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    if args.variables:
        variables += list(args.variables)
    if args.source == "consolidated":
        if not variables:
            raise SystemExit("consolidated mode needs --variables[-file]")
        if len(args.src_root) != 1:
            raise SystemExit("consolidated mode takes exactly one --src-root")

    labels = args.src_label or [r.name for r in args.src_root]
    if len(labels) != len(args.src_root):
        raise SystemExit("--src-label count must match --src-root count")

    years_filter = parse_years(args.years) if args.years else None
    all_years = sorted(
        {
            int(s.stem)
            for root in args.src_root
            for s in root.glob("*.zarr")
            if s.stem.isdigit()
        }
    )
    jobs = []
    skipped = 0
    for year in all_years:
        if years_filter and year not in years_filter:
            continue
        src_stores = [
            (lbl, str(root / f"{year}.zarr"))
            for lbl, root in zip(labels, args.src_root)
            if (root / f"{year}.zarr").exists()
        ]
        if not src_stores:
            continue
        sentinel = _sentinel(args.out_root, args.model, year)
        if sentinel.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append(
            {
                "source": args.source,
                "model": args.model,
                "src_stores": src_stores,
                "dst": str(args.out_root / args.model / f"{year}.zarr"),
                "dst_lat": dst_lat.tolist(),
                "dst_lon": dst_lon.tolist(),
                "variables": variables,
                "sentinel": str(sentinel),
                "commit": args.commit,
                "overwrite": args.overwrite,
            }
        )
    logger.info(
        "%s/%s: %d store-years to harmonize (%d already done)",
        args.source,
        args.model,
        len(jobs),
        skipped,
    )
    if not jobs:
        return 0
    failures = 0
    if args.n_workers <= 1:
        results = map(_worker, jobs)
        for r in results:
            logger.log(
                logging.INFO if r["status"] == "ok" else logging.ERROR, "%s", r
            )
            failures += r["status"] != "ok"
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = [pool.submit(_worker, j) for j in jobs]
            for fut in as_completed(futures):
                r = fut.result()
                logger.log(
                    logging.INFO if r["status"] == "ok" else logging.ERROR,
                    "%s",
                    r,
                )
                failures += r["status"] != "ok"
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
