#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Fit the day-of-year SST climatology that ``climate.sst_forcing`` reads back.

Port of amip_v2 ``scripts/make_sst_climatology.py`` @ ``e0b7b60`` onto this
fork's per-year Zarr stores. **The ``.npz`` key set is upstream's exactly**, so
artifacts are interchangeable in both directions.

Writes one small artifact (~2 MB at 180x360) holding

* ``harmonic_coeffs`` ``(1 + 2K, nlat, nlon)`` — per-gridpoint least-squares fit
  of ``a0 + sum_k [a_k cos(2 pi k t) + b_k sin(2 pi k t)]`` to the served SST
  field over the training years, ``t`` the fractional year.
* ``anom_std`` — one global scalar, the ocean-area-weighted RMS of the residual.
  The divisor that replaces the ~12.3 K absolute-SST std for the anomaly channel.
* ``gm_mean`` / ``gm_std`` — mean and std of the ocean-mean anomaly over the fit
  window, for z-scoring the ``global_mean_sst`` trend scalar.
* ``ocean_weight`` — ``cos(lat)`` over cells where the *source* SST is defined,
  normalised to sum to 1.
* ``anom_std_map`` / ``gm_series`` / ``gm_years`` — diagnostics, not read at
  training time.

The fit is deliberately restricted to the **training** years. Fitting over the
full record would put part of the warming into the reference climatology and
shrink the very signal this artifact exists to expose.

Single pass. The normal equations give the residual variance per gridpoint
without a second read (``SSE = y'y - c'X'y``), and the ocean-mean anomaly series
follows analytically from the per-frame ocean mean (``gm(t) = w.y(t) - X(t).(w.c)``).

Two things this port does differently, both because the source is different:

* **The ocean mask comes straight from the data.** Upstream had to probe the
  original HDF5 files because its fast store was already NaN-filled; this fork's
  ``amip_dailyavg_boundary`` store preserves NaN (that is why it exists), so the
  mask is read from the SST field itself. A store with no NaN at all is refused
  rather than silently treated as all-ocean — point ``--zarr`` at the
  NaN-preserving store, not the coarse one.
* **The fill the loader applies is applied here**, via the loader's own
  :class:`NanFillTransform`, so the fit sees the field training sees (upstream
  fits on its already-filled store). Keep the flags in sync with the dataset
  config; the artifact records them so a mismatch is auditable after the fact.

Usage::

    python tools/data/amip/make_sst_climatology.py \
        --zarr $AI_ROSSBY_DATA/amip_dailyavg_boundary \
        --year-start 1979 --year-end 2015 \
        --out $AI_ROSSBY_DATA/norm_stats/sst_climatology.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from physicsnemo.experimental.datapipes.climate.sst_forcing import (  # noqa: E402
    SST_VARIABLE_NAMES,
    harmonic_design_row,
    year_fraction,
)
from physicsnemo.experimental.datapipes.climate.transforms import (  # noqa: E402
    NanFillTransform,
)


def _sst_name(ds) -> str:
    for name in SST_VARIABLE_NAMES:
        if name in ds:
            return name
    raise ValueError(
        f"no SST variable in the store (looked for {SST_VARIABLE_NAMES}); "
        f"present: {sorted(ds.data_vars)}"
    )


def area_weights(nlat: int, nlon: int) -> np.ndarray:
    """``cos(lat)`` on the regular grid, as an ``(nlat, nlon)`` map.

    Symmetric about the equator, so it does not matter whether the store lists
    latitude ascending or descending.
    """
    lat = 90.0 - (np.arange(nlat) + 0.5) * (180.0 / nlat)
    return np.repeat(np.cos(np.deg2rad(lat))[:, None], nlon, axis=1)


def ocean_mask(sst, probe_indices) -> np.ndarray:
    """Cells where the source SST is defined, intersected over several times.

    Conservative on purpose: a cell counts as ocean only if SST is defined there
    at every probe, so a seasonal sea-ice edge does not drift into the weights.
    """
    mask = None
    for i in probe_indices:
        valid = ~np.isnan(np.asarray(sst[i]))
        mask = valid if mask is None else (mask & valid)
    return mask


def calendar_of(time_value) -> tuple[float, float]:
    """``(second_of_day, day_of_year)`` in **this fork's 0-indexed** convention.

    Mirrors ``ClimateZarrDataset._decompose_time`` (``dayofyr - 1``) so the fit
    and the runtime evaluation share a phase — the conversion to upstream's
    1-indexed ``year_fraction`` happens once, below.
    """
    try:
        doy = time_value.dayofyr - 1
        hour = time_value.hour
    except AttributeError:
        import pandas as pd

        ts = pd.Timestamp(str(time_value)).to_pydatetime()
        doy = ts.timetuple().tm_yday - 1
        hour = ts.hour
    return float(hour) * 3600.0, float(doy)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--zarr",
        required=True,
        help="directory of per-year {year}.zarr stores (use the NaN-preserving "
        "amip_dailyavg_boundary store, not the coarse one)",
    )
    p.add_argument("--year-start", type=int, required=True)
    p.add_argument(
        "--year-end", type=int, required=True, help="exclusive, as upstream"
    )
    p.add_argument("--harmonics", type=int, default=3, help="annual harmonics K")
    p.add_argument(
        "--stride",
        type=int,
        default=4,
        help="store rows between samples; 4 = one 24-hour model step over "
        "6-hourly rows (the AMIP archives), i.e. daily",
    )
    p.add_argument("--out", required=True)
    # Keep these matching the dataset config's NaN-fill block.
    p.add_argument("--nan-fill", type=float, default=270.0)
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--smooth-kernel-size", type=int, default=5)
    p.add_argument("--smooth-n-iters", type=int, default=10)
    args = p.parse_args()

    import torch
    import xarray as xr

    root = Path(args.zarr)
    years = list(range(args.year_start, args.year_end))
    stores = [root / f"{y}.zarr" for y in years]
    missing = [str(s) for s in stores if not s.is_dir()]
    if missing:
        raise SystemExit(f"missing stores: {missing[:4]}{'...' if len(missing) > 4 else ''}")

    # The loader's own fill, so the fit sees exactly the field training sees —
    # including the coast fade, which is what keeps the anomaly continuous
    # across the coastline instead of stamping the land-sea mask into it.
    fill = NanFillTransform(
        default=args.nan_fill,
        smooth_nan_boundaries=not args.no_smooth,
        smooth_sigma=args.smooth_sigma,
        smooth_kernel_size=args.smooth_kernel_size,
        smooth_n_iters=args.smooth_n_iters,
    )
    sst_fill = torch.full((1, 1, 1), float(args.nan_fill))

    n_coef = 1 + 2 * args.harmonics
    XtX = np.zeros((n_coef, n_coef))
    XtY = None
    sumsq = None
    rows_X: list[np.ndarray] = []
    rows_wy: list[float] = []
    rows_year: list[float] = []
    y_ref = None
    weight = wflat = None
    sst_var = None
    n_frames = 0

    for year, store in zip(years, stores):
        ds = xr.open_zarr(store, consolidated=True, decode_timedelta=False)
        name = _sst_name(ds)
        sst_var = sst_var or name
        if name != sst_var:
            raise SystemExit(f"{store} holds {name!r}, earlier stores hold {sst_var!r}")
        sst = ds[name]
        nt, nlat, nlon = sst.shape

        if weight is None:
            probes = [0, nt // 4, nt // 2, (3 * nt) // 4]
            ocean = ocean_mask(sst, probes)
            if ocean.all():
                raise SystemExit(
                    f"{store} has no NaN in {name!r}, so the ocean mask would be "
                    f"the whole globe. This is the NaN-FILLED store; fit on "
                    f"amip_dailyavg_boundary (NaN preserved) instead."
                )
            weight = area_weights(nlat, nlon) * ocean
            weight = weight / weight.sum()
            wflat = weight.reshape(-1)
            XtY = np.zeros((n_coef, nlat * nlon))
            sumsq = np.zeros(nlat * nlon)
            print(
                f"grid {nlat}x{nlon}; {ocean.mean():.3f} of cells carry SST in the "
                f"source archive"
            )

        times = ds["time"].values
        for i in range(0, nt, args.stride):
            frame = torch.from_numpy(np.asarray(sst[i], dtype="float32"))
            filled = fill._fill_boundary(frame.unsqueeze(0), sst_fill)[0]
            y = np.asarray(filled, dtype=np.float64)
            if y_ref is None:
                # Shift by the first frame before accumulating: the design has
                # an intercept so this only moves a0, and it keeps y'y off ~1e9
                # while the residual it must resolve is ~1e-1.
                y_ref = y.copy()
            y = (y - y_ref).reshape(-1)

            sod, doy0 = calendar_of(times[i])
            x = harmonic_design_row(year_fraction(sod, doy0 + 1.0), args.harmonics)[0]

            XtX += np.outer(x, x)
            XtY += np.outer(x, y)
            sumsq += y * y
            rows_X.append(x)
            rows_wy.append(float(wflat @ y))
            rows_year.append(float(year))
            n_frames += 1
        ds.close()
        print(f"  {year}: {nt} rows -> {len(range(0, nt, args.stride))} frames")

    X = np.stack(rows_X)
    wy = np.asarray(rows_wy)
    coeffs = np.linalg.solve(XtX, XtY)  # (n_coef, nlat*nlon)

    # SSE = y'y - c'X'y, exactly, per gridpoint.
    sse = np.maximum(sumsq - np.einsum("kc,kc->c", coeffs, XtY), 0.0)
    dof = max(n_frames - n_coef, 1)
    nlat, nlon = weight.shape
    anom_var_map = (sse / dof).reshape(nlat, nlon)
    anom_std = float(np.sqrt((anom_var_map.reshape(-1) * wflat).sum()))

    # Ocean-mean anomaly series, analytically: gm(t) = w.y(t) - X(t).(w.c)
    gm_series = wy - X @ (coeffs @ wflat)
    gm_mean, gm_std = float(gm_series.mean()), float(gm_series.std())

    # Undo the y_ref shift: it lives entirely in the intercept.
    coeffs[0] += y_ref.reshape(-1)
    coeffs = coeffs.reshape(n_coef, nlat, nlon)

    yr = np.asarray(rows_year)
    slope = np.polyfit(yr, gm_series, 1)[0]
    print(
        f"\n  anomaly std (ocean, area-weighted)  {anom_std:.4f} K"
        f"   [absolute-SST std for reference: ~12.3 K]"
    )
    print(f"  global-mean anomaly std             {gm_std:.4f} K")
    print(
        f"  global-mean anomaly trend in window {slope * 10:+.4f} K/decade"
        f"  = {slope * 10 / gm_std:+.3f} sigma/decade as a scalar input"
    )
    print(
        f"  same trend through the anomaly channel"
        f"           = {slope * 10 / anom_std:+.3f} sigma/decade"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        harmonic_coeffs=coeffs.astype(np.float32),
        n_harmonics=np.int32(args.harmonics),
        anom_std=np.float32(anom_std),
        anom_std_map=np.sqrt(anom_var_map).astype(np.float32),
        gm_mean=np.float32(gm_mean),
        gm_std=np.float32(gm_std),
        gm_series=gm_series.astype(np.float32),
        gm_years=yr.astype(np.float32),
        ocean_weight=weight.astype(np.float32),
        ocean_mask=(weight > 0),
        fit_year_start=np.int32(args.year_start),
        fit_year_end=np.int32(args.year_end),
        fit_stride=np.int32(args.stride),
        sst_variable=str(sst_var),
        source=str(root),
        # Fork addition: the fill the fit saw, so a later mismatch with the
        # dataset config is auditable rather than invisible.
        fill_value=np.float32(args.nan_fill),
        fill_smoothed=bool(not args.no_smooth),
        fill_smooth_sigma=np.float32(args.smooth_sigma),
        fill_smooth_kernel_size=np.int32(args.smooth_kernel_size),
        fill_smooth_n_iters=np.int32(args.smooth_n_iters),
    )
    print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) from {n_frames} frames")


if __name__ == "__main__":
    main()
