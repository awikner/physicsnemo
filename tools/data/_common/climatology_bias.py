# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a unified climatology + annual-bias + diurnal-bias Zarr store.

The ai-rossby climatology-bias schema (used by PLASIM and E3SM; ERA5 omits the
bias arrays because no bias .npy directory exists for it):

* Coords: ``dayofyear`` (int 1..366 — 366 to accommodate leap years; the
  source climatology already has 366 daily samples), ``hour_of_day``
  (int ``[0, 6, 12, 18]``), ``sigma_level``, ``pressure_level``, ``lat``, ``lon``.
* Data variables, per source channel ``X``:

  - ``X``: daily climatology, dims ``(dayofyear, [level,] lat, lon)``.
  - ``X_bias_annual``: annual-mean bias, dims ``([level,] lat, lon)``.
  - ``X_bias_diurnal``: diurnal-cycle bias, dims ``(hour_of_day, [level,] lat, lon)``.

  Missing combinations are NaN-filled (the union convention requested in the
  Phase 3 follow-up planning).

* Attrs: ``schema_version``, ``source_climatology``, ``source_bias_dir``,
  ``coord_convention``.

The merging is a CPU-bound layout-assembly job: parsing the bias filename
convention, reshaping per-level (lat, lon) .npy files into ``(level, lat, lon)``
stacks, concat with the climatology — all numpy. We parallelize the bias-file
reads via a process pool sized by ``SLURM_CPUS_PER_TASK`` (or ``os.cpu_count``)
because the I/O is the long tail when there are hundreds of files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import xarray as xr

from .bias import (
    BiasFileSpec,
    VariableBiasGroup,
    collect_bias_paths,
    load_bias_arrays,
    scan_bias_dir,
)

logger = logging.getLogger(__name__)


CLIMATOLOGY_BIAS_SCHEMA_VERSION = "1.0"
DIURNAL_HOURS = (0, 6, 12, 18)


def _resolve_pool_size(default: int = 32) -> int:
    """Worker count: prefer ``SLURM_CPUS_PER_TASK``, else ``os.cpu_count()``."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    return max(1, os.cpu_count() or default)


def _climatology_layout_for_var(
    climatology: xr.Dataset,
    var: str,
    sigma_levels: Optional[np.ndarray],
    pressure_levels: Optional[np.ndarray],
    sigma_dim: str,
    pressure_dim: str,
) -> Optional[xr.DataArray]:
    """Pluck `var` from the climatology, rename its level coord to the target.

    After renaming the source level dim (``lev`` / ``plev``) to the unified
    name (``sigma_level`` / ``pressure_level``), the coord values are replaced
    with ``sigma_levels`` / ``pressure_levels`` so every variable in the output
    store shares the SAME coord values (otherwise float precision differences
    between the CDO climatology float64 coord and the bias-dir-derived float32
    coord would make xarray treat them as distinct levels and double the axis).

    Returns ``None`` if the source climatology has no such variable (the union
    convention will NaN-fill it).
    """
    if var not in climatology.data_vars:
        return None
    da = climatology[var]
    rename = {}
    if sigma_dim in da.dims:
        rename[sigma_dim] = "sigma_level"
    if pressure_dim in da.dims:
        rename[pressure_dim] = "pressure_level"
    if rename:
        da = da.rename(rename)
    # Replace renamed-coord values with the unified (float32-quantized) ones.
    new_coords: dict = {}
    if "sigma_level" in da.dims and sigma_levels is not None:
        if da.sizes["sigma_level"] != len(sigma_levels):
            # Climatology has fewer sigma levels than the unioned set; align
            # via nearest-neighbor reindex (NaN-fill missing slots).
            da = da.reindex(
                {"sigma_level": sigma_levels},
                method="nearest",
                tolerance=1e-4,
            )
        new_coords["sigma_level"] = ("sigma_level", sigma_levels)
    if "pressure_level" in da.dims and pressure_levels is not None:
        if da.sizes["pressure_level"] != len(pressure_levels):
            da = da.reindex(
                {"pressure_level": pressure_levels},
                method="nearest",
                tolerance=0.5,
            )
        new_coords["pressure_level"] = ("pressure_level", pressure_levels)
    if new_coords:
        da = da.assign_coords(**new_coords)
    # Rename the source `time` dim to `dayofyear` and replace its values with
    # the integer dayofyear index (1..N, N==366 for leap-year-padded sources).
    if "time" in da.dims:
        n = da.sizes["time"]
        da = da.rename({"time": "dayofyear"}).assign_coords(
            dayofyear=("dayofyear", np.arange(1, n + 1, dtype="int16"))
        )
    return da


def _empty_climatology_da(
    var: str,
    *,
    levels: Optional[np.ndarray],
    level_dim: Optional[str],
    n_dayofyear: int,
    n_lat: int,
    n_lon: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> xr.DataArray:
    """NaN-filled climatology array for a var that's only in the bias dir."""
    if levels is None or level_dim is None:
        shape = (n_dayofyear, n_lat, n_lon)
        dims = ("dayofyear", "lat", "lon")
        coords = {
            "dayofyear": ("dayofyear", np.arange(1, n_dayofyear + 1, dtype="int16")),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        }
    else:
        shape = (n_dayofyear, len(levels), n_lat, n_lon)
        dims = ("dayofyear", level_dim, "lat", "lon")
        coords = {
            "dayofyear": ("dayofyear", np.arange(1, n_dayofyear + 1, dtype="int16")),
            level_dim: (level_dim, levels),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        }
    return xr.DataArray(np.full(shape, np.nan, dtype="float32"), dims=dims, coords=coords)


def _classify_level(
    value: float,
    sigma_levels: Optional[np.ndarray],
    pressure_levels: Optional[np.ndarray],
) -> tuple[str, int]:
    """Pick ``"sigma_level"`` or ``"pressure_level"`` for a numeric level value.

    The PLASIM / E3SM bias dirs cleanly partition: sigma values are < 2
    (unitless 0..1), pressure values are ≥ 2 (Pa or hPa). Returns the axis name
    and the index into the corresponding level vector (closest match).
    """
    if sigma_levels is not None and value < 2.0:
        idx = int(np.argmin(np.abs(sigma_levels - float(value))))
        return "sigma_level", idx
    if pressure_levels is not None and value >= 2.0:
        idx = int(np.argmin(np.abs(pressure_levels - float(value))))
        return "pressure_level", idx
    raise ValueError(
        f"bias level {value} doesn't fit into the configured sigma or pressure "
        "axes; either the bias dir is mis-tagged or the dataset has an "
        "unhandled third level system."
    )


def _alloc_bias_buffers(
    *,
    levels: Optional[np.ndarray],
    level_dim: Optional[str],
    n_lat: int,
    n_lon: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Allocate empty ``(annual, diurnal)`` xarray buffers in the right layout."""
    if level_dim is None:
        annual_buf = np.full((n_lat, n_lon), np.nan, dtype="float32")
        diurnal_buf = np.full((len(DIURNAL_HOURS), n_lat, n_lon), np.nan, dtype="float32")
        annual_da = xr.DataArray(
            annual_buf, dims=("lat", "lon"),
            coords={"lat": ("lat", lat), "lon": ("lon", lon)},
        )
        diurnal_da = xr.DataArray(
            diurnal_buf, dims=("hour_of_day", "lat", "lon"),
            coords={
                "hour_of_day": ("hour_of_day", np.asarray(DIURNAL_HOURS, dtype="int8")),
                "lat": ("lat", lat),
                "lon": ("lon", lon),
            },
        )
    else:
        annual_buf = np.full((len(levels), n_lat, n_lon), np.nan, dtype="float32")
        diurnal_buf = np.full(
            (len(DIURNAL_HOURS), len(levels), n_lat, n_lon),
            np.nan,
            dtype="float32",
        )
        annual_da = xr.DataArray(
            annual_buf, dims=(level_dim, "lat", "lon"),
            coords={
                level_dim: (level_dim, levels),
                "lat": ("lat", lat),
                "lon": ("lon", lon),
            },
        )
        diurnal_da = xr.DataArray(
            diurnal_buf, dims=("hour_of_day", level_dim, "lat", "lon"),
            coords={
                "hour_of_day": ("hour_of_day", np.asarray(DIURNAL_HOURS, dtype="int8")),
                level_dim: (level_dim, levels),
                "lat": ("lat", lat),
                "lon": ("lon", lon),
            },
        )
    return annual_da, diurnal_da


def _bias_layouts_for_var(
    group: VariableBiasGroup,
    bias_arrays: dict[Path, np.ndarray],
    *,
    sigma_levels: Optional[np.ndarray],
    pressure_levels: Optional[np.ndarray],
    n_lat: int,
    n_lon: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict[str, xr.DataArray]:
    """Build the bias arrays for one variable, split by level system.

    A single source variable name can appear on sigma levels, pressure levels,
    both, or neither (surface). We emit:

    * Surface vars: ``{var}_bias_annual`` + ``{var}_bias_diurnal`` (no level dim).
    * Sigma-only vars: ``{var}_bias_annual_sigma`` + ``{var}_bias_diurnal_sigma``.
    * Pressure-only vars: ``{var}_bias_annual_pressure`` + ``{var}_bias_diurnal_pressure``.
    * Vars on both axes: BOTH the sigma and pressure variant pairs.

    NaN-fills any (level, hour) slot for which no bias file exists.
    """
    var = group.var
    # Classify each level value into "sigma" or "pressure", remember the source
    # path so we can pluck arrays out of bias_arrays at fill time.
    sigma_annual: dict[float, Path] = {}
    pressure_annual: dict[float, Path] = {}
    sigma_diurnal: dict[tuple[int, float], Path] = {}
    pressure_diurnal: dict[tuple[int, float], Path] = {}
    surface_annual: Optional[Path] = group.annual.get(None)
    surface_diurnal: dict[int, Path] = {
        h: p for (h, lv), p in group.diurnal.items() if lv is None
    }

    for level, path in group.annual.items():
        if level is None:
            continue
        axis, _ = _classify_level(level, sigma_levels, pressure_levels)
        (sigma_annual if axis == "sigma_level" else pressure_annual)[level] = path
    for (hour, level), path in group.diurnal.items():
        if level is None:
            continue
        axis, _ = _classify_level(level, sigma_levels, pressure_levels)
        if hour not in DIURNAL_HOURS:
            logger.warning("%s: hour %d not in %s; skipping", path, hour, DIURNAL_HOURS)
            continue
        (sigma_diurnal if axis == "sigma_level" else pressure_diurnal)[(hour, level)] = path

    out: dict[str, xr.DataArray] = {}

    def _populate(annual_paths, diurnal_paths, levels, level_dim, suffix):
        annual_da, diurnal_da = _alloc_bias_buffers(
            levels=levels,
            level_dim=level_dim,
            n_lat=n_lat,
            n_lon=n_lon,
            lat=lat,
            lon=lon,
        )
        for key, path in annual_paths.items():
            arr = bias_arrays[path]
            if arr.shape != (n_lat, n_lon):
                raise ValueError(
                    f"{path}: bias .npy shape {arr.shape} != ({n_lat}, {n_lon})"
                )
            if level_dim is None:
                annual_da.values[:] = arr
            else:
                idx = int(np.argmin(np.abs(levels - float(key))))
                annual_da.values[idx] = arr
        for key, path in diurnal_paths.items():
            arr = bias_arrays[path]
            if level_dim is None:
                hour = key
                h_idx = DIURNAL_HOURS.index(hour)
                diurnal_da.values[h_idx] = arr
            else:
                hour, level = key
                h_idx = DIURNAL_HOURS.index(hour)
                idx = int(np.argmin(np.abs(levels - float(level))))
                diurnal_da.values[h_idx, idx] = arr
        out[f"{var}_bias_annual{suffix}"] = annual_da
        out[f"{var}_bias_diurnal{suffix}"] = diurnal_da

    if surface_annual is not None or surface_diurnal:
        _populate(
            {None: surface_annual} if surface_annual is not None else {},
            surface_diurnal,
            None,
            None,
            "",
        )
    if sigma_annual or sigma_diurnal:
        if sigma_levels is None:
            raise ValueError(
                f"variable {var!r} has sigma-level bias files but the climatology "
                "has no sigma_level coord"
            )
        _populate(
            sigma_annual,
            sigma_diurnal,
            np.asarray(sigma_levels, dtype="float32"),
            "sigma_level",
            "_sigma",
        )
    if pressure_annual or pressure_diurnal:
        if pressure_levels is None:
            raise ValueError(
                f"variable {var!r} has pressure-level bias files but the "
                "climatology has no pressure_level coord"
            )
        _populate(
            pressure_annual,
            pressure_diurnal,
            np.asarray(pressure_levels, dtype="float32"),
            "pressure_level",
            "_pressure",
        )
    return out


def _tolerance_union(values: Iterable[float], tol: float = 1e-4) -> list[float]:
    """Sort ``values`` and collapse runs within ``tol`` to a single representative.

    Bias .npy filenames carry the source PLASIM-config full-precision sigma
    values (e.g. ``0.21085000783205032``), while the climatology file's `lev`
    coord stores a CDO-rounded float32 (e.g. ``0.21085``). At the physical-
    level semantic both refer to the same model layer; we collapse them so the
    unified store has one entry per layer. ``tol`` defaults to ``1e-4`` —
    larger than any precision artifact but far smaller than any real PLASIM
    sigma spacing (~0.05).
    """
    sortv = sorted(float(v) for v in values)
    out: list[float] = []
    for v in sortv:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def _union_levels_from_bias(
    bias_groups: dict[str, VariableBiasGroup],
    base_sigma: Optional[np.ndarray],
    base_pressure: Optional[np.ndarray],
    *,
    sigma_tol: float = 1e-4,
    pressure_tol: float = 0.5,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Union the climatology's level coords with the bias dir's level values.

    Bias .npy filenames may reference levels the climatology omits (e.g. PLASIM
    `ta_5000.0_bias.npy` exists on pressure even though the climatology has
    `ta` only on sigma). We pull all bias-file levels in, split by < 2.0 vs
    >= 2.0, and union with the climatology's per-axis vector. Tolerance-based
    deduplication collapses precision-artifact "near-duplicate" entries (the
    bias dir carries full float64 precision; CDO climatology coords come back
    as float32-rounded values that miss the last few digits).
    """
    sigma_in: list[float] = []
    pressure_in: list[float] = []
    if base_sigma is not None:
        sigma_in.extend(float(v) for v in base_sigma)
    if base_pressure is not None:
        pressure_in.extend(float(v) for v in base_pressure)
    for g in bias_groups.values():
        for level in g.levels:
            if level < 2.0:
                sigma_in.append(float(level))
            else:
                pressure_in.append(float(level))
    sigma_out = (
        np.asarray(_tolerance_union(sigma_in, sigma_tol), dtype="float32")
        if sigma_in
        else None
    )
    pressure_out = (
        np.asarray(_tolerance_union(pressure_in, pressure_tol), dtype="float32")
        if pressure_in
        else None
    )
    return sigma_out, pressure_out


def build_climatology_bias_dataset(
    climatology_ds: xr.Dataset,
    bias_groups: dict[str, VariableBiasGroup],
    *,
    sigma_dim: str = "lev",
    pressure_dim: str = "plev",
    n_workers: int = 0,
) -> xr.Dataset:
    """Assemble the unified climatology + bias xarray.Dataset.

    Parameters
    ----------
    climatology_ds : xarray.Dataset
        Source climatology (e.g. ``sim52/sigma_data/climatology.nc``), with
        ``time`` (366 daily samples), ``lev`` (sigma), ``plev`` (pressure),
        ``lat``, ``lon`` coords.
    bias_groups : dict[str, VariableBiasGroup]
        Output of :func:`tools.data._common.bias.scan_bias_dir`. Pass an empty
        dict to build a climatology-only store (the ERA5 case).
    sigma_dim, pressure_dim : str
        Names of the level coords in the source climatology. Default matches
        PLASIM / E3SM (``lev``, ``plev``).
    n_workers : int, optional
        Process-pool size for parallel bias-file loading. ``0`` autodetects
        via ``SLURM_CPUS_PER_TASK`` (fallback ``os.cpu_count()``).

    Returns
    -------
    xarray.Dataset
        Ready to write via ``.to_zarr`` with the unified schema.
    """
    if "time" not in climatology_ds.dims:
        raise ValueError("climatology dataset has no 'time' dim")
    n_dayofyear = climatology_ds.sizes["time"]
    # Keep lat/lon as the source dtype (typically float64) so the bias arrays'
    # coord values match the climatology's exactly under xarray merging.
    # float32 quantization of e.g. 87.8637988 differs from float64 and xarray
    # would treat the two as distinct coord values, doubling the lat/lon axes.
    lat = climatology_ds["lat"].values
    lon = climatology_ds["lon"].values
    n_lat = len(lat)
    n_lon = len(lon)

    # Per-axis level vectors: union of climatology coord + bias dir levels.
    # Bias dirs sometimes carry vars on level systems the climatology omits
    # (PLASIM: `ta` exists on pressure-level bias but only on sigma in the
    # climatology). The unified store includes both.
    base_sigma = (
        climatology_ds[sigma_dim].values if sigma_dim in climatology_ds.coords else None
    )
    base_pressure = (
        climatology_ds[pressure_dim].values
        if pressure_dim in climatology_ds.coords
        else None
    )
    sigma_levels, pressure_levels = _union_levels_from_bias(
        bias_groups, base_sigma, base_pressure
    )

    # Load all bias arrays in one fanout (cheap when the dir is empty).
    if n_workers == 0:
        n_workers = _resolve_pool_size()
    if bias_groups:
        all_paths = collect_bias_paths(bias_groups)
        logger.info(
            "loading %d bias .npy files across %d worker processes",
            len(all_paths),
            n_workers,
        )
        bias_arrays = load_bias_arrays(all_paths, max_workers=n_workers)
    else:
        bias_arrays = {}

    # Compute the union variable set.
    clim_vars = [v for v in climatology_ds.data_vars if v != "time_bnds"]
    bias_vars = list(bias_groups.keys())
    all_vars = sorted(set(clim_vars) | set(bias_vars))
    logger.info(
        "merging %d climatology vars + %d bias vars → %d unified vars",
        len(clim_vars),
        len(bias_vars),
        len(all_vars),
    )

    out_data: dict[str, xr.DataArray] = {}
    for var in all_vars:
        # 1) climatology: read from source or NaN-fill via the bias group's levels.
        clim_da = _climatology_layout_for_var(
            climatology_ds,
            var,
            sigma_levels=sigma_levels,
            pressure_levels=pressure_levels,
            sigma_dim=sigma_dim,
            pressure_dim=pressure_dim,
        )
        if clim_da is None:
            # Var exists only in the bias dir. Choose level dim from bias levels.
            group = bias_groups.get(var)
            if group and group.levels:
                level_arr = np.asarray(group.levels, dtype="float32")
                if sigma_levels is not None and np.all(level_arr < 2.0):
                    levels = sigma_levels
                    level_dim = "sigma_level"
                elif pressure_levels is not None and np.all(level_arr >= 2.0):
                    levels = pressure_levels
                    level_dim = "pressure_level"
                else:
                    levels, level_dim = None, None
            else:
                levels, level_dim = None, None
            clim_da = _empty_climatology_da(
                var,
                levels=levels,
                level_dim=level_dim,
                n_dayofyear=n_dayofyear,
                n_lat=n_lat,
                n_lon=n_lon,
                lat=lat,
                lon=lon,
            )
        out_data[var] = clim_da.astype("float32")

        # 2) bias_annual + bias_diurnal: from the bias group (NaN if absent).
        # The helper emits per-axis variables (`{var}_bias_annual_sigma`,
        # `{var}_bias_annual_pressure`, surface `{var}_bias_annual`) so a single
        # source-channel can carry biases on both level systems independently
        # (PLASIM `ta`, `ua`, `va`, `hus`).
        group = bias_groups.get(var)
        if group is None:
            continue
        out_data.update(
            _bias_layouts_for_var(
                group,
                bias_arrays,
                sigma_levels=sigma_levels,
                pressure_levels=pressure_levels,
                n_lat=n_lat,
                n_lon=n_lon,
                lat=lat,
                lon=lon,
            )
        )

    out_ds = xr.Dataset(out_data)
    return out_ds


def write_climatology_bias_zarr(
    out_ds: xr.Dataset,
    output_path: str | Path,
    *,
    source_climatology: Path,
    source_bias_dir: Optional[Path],
    coord_convention: str = "ai_rossby_v1",
    overwrite: bool = False,
) -> None:
    """Write the climatology+bias Zarr at ``output_path``, set provenance attrs."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output path {output_path} exists; pass overwrite=True to replace"
        )

    out_ds = out_ds.assign_attrs(
        climatology_bias_schema_version=CLIMATOLOGY_BIAS_SCHEMA_VERSION,
        source_climatology=str(source_climatology),
        source_bias_dir=str(source_bias_dir) if source_bias_dir is not None else "",
        coord_convention=coord_convention,
    )
    mode = "w" if overwrite else "w-"
    out_ds.to_zarr(output_path, mode=mode, consolidated=True, zarr_format=3)
    logger.info(
        "wrote climatology+bias Zarr to %s (vars=%d)",
        output_path,
        len(out_ds.data_vars),
    )
