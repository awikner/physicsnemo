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

"""OFFLINE: conservatively regrid DSI hindcast stores 0.25 deg -> 1 deg.

One-time preprocessing before MoWE training: every ``{src_root}/{model}/
{YYYY}.zarr`` (schema of ``tools/data/hindcast/dsi_hindcast_to_formats.py``
Format 2) is rewritten at 1 deg onto the IMERG/ERA5 grid as
``{out_root}/{model}/{YYYY}.zarr`` with identical dims/coords/attrs except
lat/lon, using separable 1-D conservative pooling
(``datapipes/regrid.py``). The training datapipe only reads these outputs.

Login-node safe: plain xarray / zarr / numpy only; never imports
physicsnemo, torch, or dask. Idempotent: a ``.regrid_done/<model>_<year>.done``
sentinel under ``out_root`` marks completed stores; finished stores are
skipped unless ``--overwrite``, and a partial store without its sentinel is
wiped and redone on the next run.

Usage (Derecho)::

    python examples/weather/ai_rossbypalooza/tools/regrid_dsi_to_1deg.py \\
        --src-root /glade/derecho/scratch/awikner/hindcasts_dsi/zarr \\
        --out-root /glade/derecho/scratch/awikner/hindcasts_dsi_1deg/zarr \\
        --model all --years 2000-2024 --n-workers 8 \\
        --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

# Recipe-local import: this script lives in <recipe>/tools/.
_RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from datapipes.regrid import Regridder, grids_equal  # noqa: E402

logger = logging.getLogger("regrid_dsi_to_1deg")

SCRIPT_REL_PATH = "examples/weather/ai_rossbypalooza/tools/regrid_dsi_to_1deg.py"
LEAD_DIMS = ("prediction_timedelta", "prediction_timedelta_daily")


def _register_codecs() -> None:
    """The DSI stores use numcodecs zarr3 codecs (bitround); register them."""
    try:
        import numcodecs.zarr3  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "reading DSI stores requires numcodecs>=0.13 with zarr3 codecs "
            f"(import numcodecs.zarr3 failed: {exc})"
        ) from exc


def default_target_grid() -> tuple[np.ndarray, np.ndarray]:
    """The 1-degree IMERG/ERA5 grid: lat N->S 89.5..-89.5, lon 0..359."""
    return (
        np.linspace(89.5, -89.5, 180).astype("float32"),
        np.arange(0.0, 360.0, 1.0).astype("float32"),
    )


def target_grid_from_ref(ref_store: Path) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_zarr(ref_store, consolidated=True, decode_times=False)
    try:
        return (
            ds["lat"].values.astype("float32"),
            ds["lon"].values.astype("float32"),
        )
    finally:
        ds.close()


def _sentinel(out_root: Path, model: str, year: int) -> Path:
    return out_root / ".regrid_done" / f"{model}_{year}.done"


def regrid_store(
    src_store: Path,
    dst_store: Path,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    *,
    commit: str = "unknown",
    overwrite: bool = False,
) -> dict:
    """Regrid one yearly store; returns a small summary dict.

    Writes init-by-init with ``append_dim`` (chunk-aligned, one chunk per
    (variable, init)) — deliberately dask-free so the tool runs anywhere
    plain xarray/zarr do. One worker owns one store, so sequential appends
    are safe.
    """
    _register_codecs()
    src = xr.open_zarr(
        src_store, consolidated=True, decode_times=False, decode_timedelta=False
    )
    try:
        src_lat = src["lat"].values.astype("float64")
        src_lon = src["lon"].values.astype("float64")
        if grids_equal(src_lat, dst_lat) and grids_equal(src_lon, dst_lon):
            raise ValueError(f"{src_store} is already on the target grid")
        regridder = Regridder(src_lat, src_lon, dst_lat, dst_lon)
        n_init = src.sizes["init_time"]
        n_dlat, n_dlon = len(dst_lat), len(dst_lon)

        var_dims: dict[str, tuple[str, ...]] = {}
        encoding: dict = {}
        for v in src.data_vars:
            dims = tuple(str(d) for d in src[v].dims)
            if dims[-2:] != ("lat", "lon") or dims[0] != "init_time":
                logger.warning("skipping %s with unexpected dims %s", v, dims)
                continue
            var_dims[v] = dims
            encoding[v] = {
                "chunks": (1, src.sizes[dims[1]], n_dlat, n_dlon),
                "dtype": "float32",
            }
        if not var_dims:
            raise ValueError(f"{src_store}: no regriddable variables found")

        attrs = dict(src.attrs)
        attrs["note"] = (
            "1 deg conservative regrid of the native 0.25 deg store; "
            "otherwise identical schema (two lead axes, flat channels, lat N->S)."
        )
        attrs["regridded_from"] = str(src_store)
        attrs["regrid_method"] = "1d-conservative"
        attrs["generator"] = f"{SCRIPT_REL_PATH}@{commit}"

        base_coords = {
            "lat": ("lat", np.asarray(dst_lat, dtype="float32")),
            "lon": ("lon", np.asarray(dst_lon, dtype="float32")),
        }
        for dim in LEAD_DIMS:
            if dim in src.coords:
                base_coords[dim] = (dim, src[dim].values)

        dst_store.parent.mkdir(parents=True, exist_ok=True)
        nan_vars: set[str] = set()
        for i in range(n_init):
            step_vars = {}
            for v, dims in var_dims.items():
                raw = src[v].isel(init_time=i).values  # (n_lead, H_src, W_src)
                out = regridder(raw).astype("float32")
                if np.isnan(out).any():
                    nan_vars.add(v)
                step_vars[v] = (dims, out[np.newaxis, ...])
            step = xr.Dataset(
                step_vars,
                coords={
                    "init_time": (
                        "init_time",
                        src["init_time"].values[i : i + 1],
                    ),
                    **base_coords,
                },
                attrs=attrs,
            )
            # decode_times=False keeps init_time raw ints; carry the source
            # units/calendar (and lead-axis) attrs through verbatim.
            step["init_time"].attrs.update(src["init_time"].attrs)
            for dim in LEAD_DIMS:
                if dim in step.coords:
                    step[dim].attrs.update(src[dim].attrs)
            if i == 0:
                step.to_zarr(
                    dst_store,
                    mode="w" if overwrite else "w-",
                    zarr_format=3,
                    consolidated=False,
                    encoding=encoding,
                )
            else:
                step.to_zarr(
                    dst_store, append_dim="init_time", consolidated=False
                )
        zarr.consolidate_metadata(str(dst_store))
        if nan_vars:
            logger.warning(
                "%s: NaNs present after regrid in %s", dst_store, sorted(nan_vars)
            )
        return {
            "store": str(dst_store),
            "n_init": n_init,
            "variables": len(var_dims),
            "nan_vars": sorted(nan_vars),
        }
    finally:
        src.close()


def _worker(job: dict) -> dict:
    src = Path(job["src"])
    dst = Path(job["dst"])
    try:
        if dst.exists() and not Path(job["sentinel"]).exists():
            # Partial store from an interrupted run: wipe and redo.
            import shutil

            logger.warning("removing partial store %s", dst)
            shutil.rmtree(dst)
        summary = regrid_store(
            src,
            dst,
            np.asarray(job["dst_lat"]),
            np.asarray(job["dst_lon"]),
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--src-root", type=Path, required=True,
                   help="root holding {model}/{YYYY}.zarr DSI stores")
    p.add_argument("--out-root", type=Path, required=True,
                   help="destination root for the 1 deg copies")
    p.add_argument("--model", action="append", required=True,
                   help="model subdir name; repeatable; 'all' discovers subdirs")
    p.add_argument("--years", default=None,
                   help="e.g. 2000-2024 or 2019,2020; default: all found")
    p.add_argument("--ref-store", type=Path, default=None,
                   help="copy the target lat/lon from this store (e.g. an "
                        "IMERG year) instead of the built-in 1 deg grid")
    p.add_argument("--n-workers", type=int, default=4,
                   help="process pool size over (model, year) stores")
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
    if args.ref_store is not None:
        dst_lat, dst_lon = target_grid_from_ref(args.ref_store)
    else:
        dst_lat, dst_lon = default_target_grid()

    models: list[str] = []
    for m in args.model:
        if m == "all":
            models.extend(
                sorted(
                    d.name
                    for d in args.src_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                )
            )
        else:
            models.append(m)

    jobs: list[dict] = []
    skipped = 0
    for model in models:
        stores = sorted((args.src_root / model).glob("*.zarr"))
        if not stores:
            logger.warning("no stores under %s", args.src_root / model)
        for store in stores:
            try:
                year = int(store.stem)
            except ValueError:
                logger.warning("skipping non-year store %s", store)
                continue
            if args.years is not None and year not in parse_years(args.years):
                continue
            sentinel = _sentinel(args.out_root, model, year)
            if sentinel.exists() and not args.overwrite:
                skipped += 1
                continue
            jobs.append(
                {
                    "src": str(store),
                    "dst": str(args.out_root / model / store.name),
                    "dst_lat": dst_lat.tolist(),
                    "dst_lon": dst_lon.tolist(),
                    "sentinel": str(sentinel),
                    "commit": args.commit,
                    "overwrite": args.overwrite,
                }
            )
    logger.info("%d stores to regrid (%d already done)", len(jobs), skipped)
    if not jobs:
        return 0

    failures = 0
    if args.n_workers <= 1:
        results = map(_worker, jobs)
        for r in results:
            level = logging.INFO if r["status"] == "ok" else logging.ERROR
            logger.log(level, "%s", r)
            failures += r["status"] != "ok"
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = [pool.submit(_worker, j) for j in jobs]
            for fut in as_completed(futures):
                r = fut.result()
                level = logging.INFO if r["status"] == "ok" else logging.ERROR
                logger.log(level, "%s", r)
                failures += r["status"] != "ok"
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
