#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that a ClimateZarr store's LABELLED latitude matches its actual DATA.

``registry.py check`` counts stores; it cannot tell you a store is upside-down.
This does. It answers, per array, "is the data actually where the lat coordinate
says it is?", using only facts about the Earth -- never the look of the numbers.

Four checks, strongest first. Each is independent, and where they overlap they
cross-validate each other:

A. ABSOLUTE PHYSICAL ANCHORS (``tools/data/_common/lat_orientation.py``)
     land mask, surface geopotency, solstice insolation, seasonal temperature.
     Polar bands are selected BY LABEL (lat > +70 vs lat < -70), so an
     upside-down store fails every applicable test. Needs nothing but the store.

B. CROSS-ARRAY CONSISTENCY
     Arrays with no anchor of their own are compared against the anchored ones by
     ZONAL-ANOMALY pattern correlation, as-is vs lat-reversed. Removing the zonal
     mean strips the near-symmetric meridional profile and leaves the
     continent-dominated structure, which is strongly hemisphere-asymmetric, so
     the true pairing wins by a wide margin. Correlation MAGNITUDES are compared,
     because plenty of real field pairs are legitimately anticorrelated (surface
     pressure vs orography is r = -0.99). A pair only votes when the winning
     correlation is genuinely strong, so noise cannot masquerade as a finding.
     Winds additionally get a GEOSTROPHY check against the store's own
     geopotential; note that is a RELATIVE test (a whole-store flip reverses both
     dPhi/dy and the sign of f, and the two cancel), so it pins the wind to Z
     rather than to the globe.

C. TEMPORAL SEAM DETECTION
     field(t) vs field(t+1), as-is and lat-reversed, sampled ON the 60-timestep
     block boundaries that ``flip_lat_zarr.py`` writes in, plus interior
     controls. This is what catches an array left HALF flipped by an interrupted
     repair -- a store can carry ``lat_flipped_to_NtoS=True`` and still be broken
     across part of the year. Block edges are never dropped when subsampling.

D. EXACT RAW COMPARISON (``--h5-dir``, the gold standard)
     Bit-for-bit against the raw H5 source as-is and lat-reversed. The verdict is
     label-aware: the store's data order is the raw order, reversed iff the store
     reversed the rows, and THAT is what gets compared with the store's own lat
     direction -- E3SM legitimately labels S->N, so "kept the raw S->N rows" is
     correct there. Channels altered on ingest (NaN fill, derived fields) fall
     back to the zonal-anomaly correlation, and because a co-located raw copy is
     not always the exact archive a store was built from, correlation-only
     evidence needs a strong majority and never on its own reports a seam.

Verdict per array: OK / FLIPPED (data upside-down vs its label) / MIXED-IN-TIME /
UNKNOWN (no decisive test). Exit status is 1 if any array is not OK/UNKNOWN, so
this is usable as a gate.

Examples::

    # fast whole-store screen (1-2 reads per store) over every AMIP copy
    check_lat_orientation.py --anchors-only --stores "$AI_ROSSBY_DATA/amip/*.zarr"

    # full per-array audit, gold-standard, against the raw archive
    check_lat_orientation.py --stores "$AI_ROSSBY_DATA/era5/1989.zarr" \\
        --h5-dir /work/hdd/bdiu/bgong1/data/h5data --raw-order "S->N"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:  # normal in-repo use
    from tools.data._common.lat_orientation import (
        INSOL_KEYS,
        LAND_KEYS,
        MIN_TEMP_CONTRAST_K,
        NORTH_FIRST,
        OROG_KEYS,
        TEMP_KEYS,
        order_of_coord,
    )
except ImportError:  # running the file directly from tools/data/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data._common.lat_orientation import (  # type: ignore[no-redef]
        INSOL_KEYS,
        LAND_KEYS,
        MIN_TEMP_CONTRAST_K,
        NORTH_FIRST,
        OROG_KEYS,
        TEMP_KEYS,
        order_of_coord,
    )

COORDS = {
    "lat", "lon", "time", "pressure_level", "sigma_level", "level",
    "init_time", "lead_time", "latitude", "longitude", "time_bnds",
    "month", "quantile", "stat",
}
OMEGA, A_EARTH = 7.2921e-5, 6.371e6
#: A cross-array pair votes only when the winning |r| clears MINCORR and beats the
#: alternative by MARGIN. Loosening these invites confident-looking noise.
MARGIN, MINCORR = 0.15, 0.35
#: ``flip_lat_zarr.py`` rewrites in blocks of this many timesteps, so an
#: interrupted run leaves its seam on a multiple of it.
FLIP_BLOCK = 60


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def zonal_anomaly(a: np.ndarray) -> np.ndarray:
    """Strip the zonal mean, leaving the hemisphere-asymmetric eddy structure."""
    a = np.asarray(a, dtype="float64")
    with np.errstate(invalid="ignore"):
        return a - np.nanmean(a, axis=-1, keepdims=True)


def pattern_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = zonal_anomaly(a).ravel(), zonal_anomaly(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def relation_from_corr(c_asis: float, c_flip: float) -> tuple[str, float]:
    """Is this field oriented like the reference, or reversed?"""
    if not (np.isfinite(c_asis) and np.isfinite(c_flip)):
        return "UNKNOWN", float("nan")
    margin = abs(c_asis) - abs(c_flip)
    if abs(margin) < MARGIN or max(abs(c_asis), abs(c_flip)) < MINCORR:
        return "UNKNOWN", margin
    return ("SAME" if margin > 0 else "REVERSED"), margin


def compose(ref_verdict: str | None, rel: str | None) -> str:
    if ref_verdict not in ("OK", "FLIPPED") or rel not in ("SAME", "REVERSED"):
        return "UNKNOWN"
    if rel == "SAME":
        return ref_verdict
    return "FLIPPED" if ref_verdict == "OK" else "OK"


# --------------------------------------------------------------------------- #
# store access
# --------------------------------------------------------------------------- #
def open_store(path: str):
    import zarr

    try:
        return zarr.open_group(path, mode="r")
    except Exception:
        return zarr.open_group(path, mode="r", use_consolidated=False)


def lat_axis(arr, n_lat: int) -> int | None:
    try:
        dims = arr.metadata.dimension_names
    except Exception:
        dims = None
    if dims and "lat" in dims:
        return list(dims).index("lat")
    hits = [i for i, s in enumerate(arr.shape) if s == n_lat]
    if not hits:
        return None
    return hits[0] if len(hits) == 1 else hits[-2]


def decoded_times(path: str):
    try:
        import xarray as xr

        ds = xr.open_zarr(path, consolidated=None, decode_timedelta=False)
        if "time" not in ds.coords:
            return None
        return list(np.asarray(ds["time"].values))
    except Exception:
        return None


def month_day(t) -> tuple[int, int]:
    if hasattr(t, "month"):
        return t.month, t.day
    s = str(t)
    try:
        return int(s[5:7]), int(s[8:10])
    except Exception:
        return 0, 0


#: How far (in rough days) the nearest available timestep may sit from a seasonal
#: target date before the test is abandoned. A subset store -- an AMIP quarter, say
#: -- may contain no solstice at all, and without this the "nearest to 21 Jun" and
#: "nearest to 21 Dec" tests both land on the same edge timestep and contradict each
#: other on identical data.
SEASON_TOLERANCE_DAYS = 20


def nearest_time(times, month: int, day: int,
                 max_cost: float = SEASON_TOLERANCE_DAYS) -> int | None:
    if not times:
        return None
    best, best_i = 10**9, None
    for i, t in enumerate(times):
        mo, dy = month_day(t)
        if mo == 0:
            continue
        cost = abs((mo - month) * 31 + (dy - day))
        if cost < best:
            best, best_i = cost, i
    return best_i if best <= max_cost else None


def slice_2d(g, name: str, tidx: int | None, lev: int = 0) -> np.ndarray | None:
    """A single (lat, lon) plane, or None if the array cannot yield one."""
    arr = g[name]
    try:
        if arr.ndim == 2:
            return np.asarray(arr[:], dtype="float64")
        if tidx is None:
            return None
        if arr.ndim == 3:
            return np.asarray(arr[tidx], dtype="float64")
        if arr.ndim == 4:
            return np.asarray(arr[tidx, lev], dtype="float64")
    except Exception:
        return None
    return None


def raw_key_for(h5_keys, name: str, level: float | None) -> str | None:
    if level is None:
        return name if name in h5_keys else None
    for k in h5_keys:
        if not k.startswith(name + "_"):
            continue
        try:
            v = float(k[len(name) + 1:])
        except ValueError:
            continue
        if abs(v - level) <= 1e-6 + 1e-3 * abs(level):
            return k
    return None


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #
def audit_store(
    store: str,
    *,
    h5_dir: str | None = None,
    year: str | None = None,
    group: str = "input",
    raw_order: str | None = None,
    stride: int = 60,
    max_samples: int = 30,
    band: float = 70.0,
    anchors_only: bool = False,
) -> dict:
    g = open_store(store)
    names = sorted(g.array_keys())
    nameset = set(names)
    lat_name = "lat" if "lat" in nameset else ("latitude" if "latitude" in nameset else None)
    if lat_name is None:
        return {"store": store, "error": "no lat array", "arrays": {}}
    lat = np.asarray(g[lat_name][:], dtype="float64")
    n_lat = lat.size
    rep: dict = {
        "store": store,
        "lat_first": float(lat[0]),
        "lat_last": float(lat[-1]),
        "n_lat": int(n_lat),
        "lat_order": order_of_coord(lat),
        "attrs": {
            k: str(v)[:70]
            for k, v in g.attrs.items()
            if any(s in k.lower() for s in ("flip", "orient", "grid", "lat_row"))
        },
        "parts": {},
        "arrays": {},
    }

    data_arrays = [n for n in names if n not in COORDS and lat_axis(g[n], n_lat) is not None]
    rep["n_data_arrays"] = len(data_arrays)
    times = decoded_times(store)
    n_time = next((int(g[n].shape[0]) for n in data_arrays if g[n].ndim >= 3), 0)
    rep["n_time"] = n_time

    # Level choice is PER ARRAY: a PLASIM sigma store puts every upper-air variable
    # on sigma levels except zg, which is on pressure levels, so one global index
    # would read the wrong coordinate (and look up the wrong raw key) for zg.
    level_coords = {
        ln: np.asarray(g[ln][:], dtype="float64")
        for ln in ("pressure_level", "sigma_level", "level")
        if ln in nameset
    }

    def level_for(name: str) -> tuple[int, float | None]:
        arr = g[name]
        if arr.ndim != 4:
            return 0, None
        try:
            dims = arr.metadata.dimension_names
        except Exception:
            dims = None
        ln = next((d for d in (dims or []) if d in level_coords), None)
        if ln is None:
            ln = next((c for c, v in level_coords.items() if arr.shape[1] == v.size), None)
        if ln is None:
            return 0, None
        vals = level_coords[ln]
        i = int(vals.size // 2)  # mid-atmosphere, unit-agnostic
        return i, float(vals[i])

    verdicts: dict[str, dict] = {n: {} for n in data_arrays}
    detail: dict[str, dict] = {n: {} for n in data_arrays}

    # ---- A: absolute anchors, judged against the store's own labels ----------
    north, south = lat > band, lat < -band
    part_a = []
    if north.sum() and south.sum():

        def anchor(name: str, tidx, expect_south_greater: bool, label: str,
                   min_contrast: float = 0.0) -> None:
            f = slice_2d(g, name, tidx, level_for(name)[0])
            if f is None or f.shape[0] != n_lat:
                return
            n_val = float(np.nanmean(f[north]))
            s_val = float(np.nanmean(f[south]))
            if (not (np.isfinite(n_val) and np.isfinite(s_val)) or n_val == s_val
                    or abs(n_val - s_val) < min_contrast):
                v = "UNKNOWN"
            else:
                ok = (s_val > n_val) if expect_south_greater else (n_val > s_val)
                v = "OK" if ok else "FLIPPED"
            part_a.append(
                {"array": name, "test": label, "north": n_val, "south": s_val, "verdict": v}
            )
            if v != "UNKNOWN":
                verdicts.setdefault(name, {}).setdefault("A", []).append(v)
                detail.setdefault(name, {}).setdefault("anchors", []).append(f"{label}:{v}")

        for k in LAND_KEYS:
            if k in nameset:
                anchor(k, 0 if g[k].ndim >= 3 else None, True, "land-mask")
        for k in OROG_KEYS:
            if k in nameset:
                anchor(k, 0 if g[k].ndim >= 3 else None, True, "orography")
        for k in INSOL_KEYS:
            if k in nameset and g[k].ndim >= 3:
                for mo, expect_south in ((6, False), (12, True)):
                    i = nearest_time(times, mo, 21)
                    if i is not None:
                        anchor(k, i, expect_south, f"insolation-{mo:02d}")
        for k in TEMP_KEYS:
            if k in nameset and g[k].ndim >= 3:
                for mo, expect_south in ((1, True), (7, False)):
                    i = nearest_time(times, mo, 15)
                    if i is not None:
                        anchor(k, i, expect_south, f"temperature-{mo:02d}",
                               min_contrast=MIN_TEMP_CONTRAST_K)
    rep["parts"]["A"] = part_a

    anchored = {
        n: d["A"][0]
        for n, d in verdicts.items()
        if d.get("A") and len(set(d["A"])) == 1 and d["A"][0] in ("OK", "FLIPPED")
    }
    rep["references"] = sorted(anchored)

    # ---- B: cross-array consistency + geostrophy ------------------------------
    t_ref = nearest_time(times, 7, 15) if times else (n_time // 2 if n_time else None)
    if t_ref is None and n_time:
        t_ref = n_time // 2
    part_b = []
    if anchored and not anchors_only:
        ref_fields = []
        for r in sorted(anchored)[:5]:
            f = slice_2d(g, r, t_ref if g[r].ndim >= 3 else None, level_for(r)[0])
            if f is not None and f.shape[0] == n_lat:
                ref_fields.append((r, anchored[r], f))
        for n in data_arrays:
            f = slice_2d(g, n, t_ref if g[n].ndim >= 3 else None, level_for(n)[0])
            if f is None or f.shape[0] != n_lat:
                continue
            best = None
            for r_name, r_verdict, r_field in ref_fields:
                if r_name == n:
                    continue
                c_a, c_f = pattern_corr(f, r_field), pattern_corr(f, r_field[::-1])
                rel, margin = relation_from_corr(c_a, c_f)
                if rel == "UNKNOWN":
                    continue
                # Prefer the STRONGEST relationship, not the widest margin: a noisy
                # pair can show a wide margin between two weak correlations and
                # then outvote a genuinely well-correlated reference.
                strength = max(abs(c_a), abs(c_f))
                if best is None or strength > max(abs(best[4]), abs(best[5])):
                    best = (r_name, r_verdict, rel, margin, c_a, c_f)
            if best:
                r_name, r_verdict, rel, margin, c_a, c_f = best
                verdicts[n]["B"] = compose(r_verdict, rel)
                detail[n]["cross_array"] = {
                    "ref": r_name, "rel": rel, "corr_asis": c_a,
                    "corr_flipped": c_f, "margin": margin,
                }
                part_b.append({"array": n, "ref": r_name, "corr_asis": c_a,
                               "corr_flipped": c_f, "verdict": verdicts[n]["B"]})

    z_key = next((k for k in ("geopotential", "z", "Z") if k in nameset and g[k].ndim == 4), None)
    if z_key is not None and t_ref is not None and not anchors_only:
        phi = slice_2d(g, z_key, t_ref, level_for(z_key)[0])
        if phi is not None and phi.shape[0] == n_lat:
            f_cor = 2 * OMEGA * np.sin(np.deg2rad(lat))
            dphi_dy = np.gradient(phi, np.deg2rad(lat) * A_EARTH, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                u_geo = -dphi_dy / f_cor[:, None]
            extratropics = np.abs(lat) > 25
            for u_key in ("u_component_of_wind", "10m_u_component_of_wind", "ua", "U"):
                if u_key not in nameset:
                    continue
                u_field = slice_2d(g, u_key, t_ref, level_for(u_key)[0])
                if u_field is None or u_field.shape[0] != n_lat:
                    continue
                c_a = pattern_corr(u_field[extratropics], u_geo[extratropics])
                c_f = pattern_corr(u_field[::-1][extratropics], u_geo[extratropics])
                rel = (
                    ("SAME" if c_a > c_f else "REVERSED")
                    if (np.isfinite(c_a) and np.isfinite(c_f) and abs(c_a - c_f) > MARGIN)
                    else "UNKNOWN"
                )
                detail.setdefault(u_key, {})["geostrophy"] = {
                    "corr_asis": c_a, "corr_flipped": c_f,
                    "z_key": z_key, "relative_to_z": rel,
                }
                part_b.append({"array": u_key, "ref": f"geostrophic-u({z_key})",
                               "corr_asis": c_a, "corr_flipped": c_f, "relative_to_z": rel})
    rep["parts"]["B"] = part_b

    # ---- C: temporal seams ----------------------------------------------------
    part_c = []
    if n_time > 2 and not anchors_only:
        edges = sorted(
            {t for k in range(1, n_time // FLIP_BLOCK + 1)
             if (t := FLIP_BLOCK * k - 1) < n_time - 1}
        )
        interior = list(range(0, n_time - 1, max(1, stride // 3)))
        cap = max_samples * 3
        keep = set(edges)
        if len(keep) > cap:
            ed = sorted(keep)
            keep = {ed[i] for i in np.unique(np.linspace(0, len(ed) - 1, cap).astype(int))}
        room = cap - len(keep)
        extra = [t for t in interior if t not in keep]
        if room > 0 and extra:
            idx = np.unique(np.linspace(0, len(extra) - 1, min(room, len(extra))).astype(int))
            keep |= {extra[i] for i in idx}
        pairs = sorted(keep)
        for n in data_arrays:
            if g[n].ndim < 3:
                continue
            lev = level_for(n)[0]
            breaks, checked = [], 0
            for t in pairs:
                f0 = slice_2d(g, n, t, lev)
                f1 = slice_2d(g, n, t + 1, lev)
                if f0 is None or f1 is None:
                    continue
                c_a, c_f = pattern_corr(f0, f1), pattern_corr(f0, f1[::-1])
                if not (np.isfinite(c_a) and np.isfinite(c_f)):
                    continue
                checked += 1
                if c_f > c_a + MARGIN:
                    breaks.append({"t": t, "corr_asis": c_a, "corr_flipped": c_f})
            part_c.append({"array": n, "pairs_checked": checked,
                           "n_seams": len(breaks), "seams": breaks[:6]})
            detail[n]["temporal_uniform"] = not breaks
    rep["parts"]["C"] = part_c

    # ---- D: exact comparison against raw -------------------------------------
    part_d = []
    if h5_dir:
        import h5py

        yr = year or str(
            g.attrs.get("year_index") or os.path.basename(store).replace(".zarr", "")
        )
        files = sorted(glob.glob(f"{h5_dir}/{yr}_*.h5"))
        if files:
            with h5py.File(files[0], "r") as f:
                h5_keys = set(f[group].keys())
            try:
                offset = int(np.asarray(g.attrs.get("sample_range", [0])).ravel()[0])
            except Exception:
                offset = 0
            rep["raw_index_offset"] = offset
            samples = [t for t in range(0, n_time, max(1, stride)) if offset + t < len(files)]
            if len(samples) > max_samples:
                samples = [
                    samples[i]
                    for i in np.unique(np.linspace(0, len(samples) - 1, max_samples).astype(int))
                ]
            # The store's data order is the raw order, reversed iff the store
            # reversed the rows; that is what must match the store's own label.
            aligned = (raw_order == NORTH_FIRST) == (rep["lat_order"] == NORTH_FIRST)
            for n in data_arrays:
                ndim = g[n].ndim
                lev_i, lev_v = level_for(n)
                rk = raw_key_for(h5_keys, n, lev_v if ndim == 4 else None)
                if rk is None:
                    part_d.append({"array": n, "raw_key": None, "note": "absent-in-raw"})
                    continue
                rels: dict[str, int] = {}
                for t in ([None] if ndim == 2 else samples):
                    fi = 0 if t is None else offset + t
                    with h5py.File(files[fi], "r") as f:
                        raw = np.asarray(f[group][rk][:], dtype="float32")
                    z = slice_2d(g, n, t, lev_i)
                    if z is None or z.shape != raw.shape:
                        rels["shape-mismatch"] = rels.get("shape-mismatch", 0) + 1
                        continue
                    z32 = z.astype("float32")
                    if np.array_equal(z32, raw, equal_nan=True):
                        r = "asis-exact"
                    elif np.array_equal(z32, raw[::-1], equal_nan=True):
                        r = "flipped-exact"
                    else:
                        c_a = pattern_corr(z, raw)
                        c_f = pattern_corr(z, raw[::-1])
                        r = (
                            ("flipped-corr" if abs(c_f) > abs(c_a) else "asis-corr")
                            if abs(abs(c_a) - abs(c_f)) > MARGIN
                            else "ambiguous"
                        )
                    rels[r] = rels.get(r, 0) + 1
                part_d.append({"array": n, "raw_key": rk, "relations": rels})
                detail[n]["vs_raw"] = sorted(rels)
                if raw_order is not None:
                    verdicts[n]["D"] = _verdict_vs_raw(rels, aligned)
    rep["parts"]["D"] = part_d

    # ---- combine, by authority: D > A > geostrophy-composed > B --------------
    z_final = None
    if z_key is not None:
        zp = verdicts.get(z_key, {})
        za = zp.get("A")
        z_final = zp.get("D") or (za[0] if za and len(set(za)) == 1 else None) or zp.get("B")
    for n in data_arrays:
        parts = verdicts.get(n, {})
        a_votes = parts.get("A")
        a_v = a_votes[0] if a_votes and len(set(a_votes)) == 1 else ("CONFLICT" if a_votes else None)
        geo_rel = detail.get(n, {}).get("geostrophy", {}).get("relative_to_z")
        b_geo = compose(z_final, geo_rel) if (z_final and geo_rel) else None
        ordered = [
            ("D", parts.get("D")),
            ("A", a_v),
            ("Bgeo", b_geo if b_geo != "UNKNOWN" else None),
            ("B", parts.get("B")),
        ]
        final, source = "UNKNOWN", None
        for src, v in ordered:
            if v in ("OK", "FLIPPED"):
                final, source = v, src
                break
            if v and v != "UNKNOWN" and final == "UNKNOWN":
                final, source = v, src
        opinions = {s: v for s, v in ordered if v}
        disagree = sorted({v for v in opinions.values() if v in ("OK", "FLIPPED")})
        if detail.get(n, {}).get("temporal_uniform") is False:
            final = "MIXED-IN-TIME"
        elif len(disagree) > 1:
            final = "CONFLICT(" + "/".join(f"{s}={v}" for s, v in opinions.items()) + ")"
        rep["arrays"][n] = {"verdict": final, "decided_by": source,
                            "by_part": opinions, **detail.get(n, {})}

    vs = [v["verdict"] for v in rep["arrays"].values()]
    rep["store_verdict"] = (
        "NO-ARRAYS" if not vs
        else "OK" if all(v == "OK" for v in vs)
        else "ALL-FLIPPED" if all(v == "FLIPPED" for v in vs)
        else "OK-WITH-UNKNOWNS" if all(v in ("OK", "UNKNOWN") for v in vs)
        else "PROBLEM"
    )
    rep["bad_arrays"] = sorted(n for n, d in rep["arrays"].items() if d["verdict"] != "OK")
    return rep


def _verdict_vs_raw(rels: dict, aligned: bool) -> str:
    """Translate raw relations into a verdict.

    Bit-exact matches are trusted outright. Correlation-only matches are weaker
    evidence -- some channels were altered on ingest and a co-located raw copy is
    not always the exact archive a store was built from -- so they need a strong
    majority and never on their own report a seam.
    """
    kinds = set(rels)
    exact_flipped, exact_asis = "flipped-exact" in kinds, "asis-exact" in kinds
    asis_means = "OK" if aligned else "FLIPPED"
    flipped_means = "FLIPPED" if aligned else "OK"
    if exact_flipped and exact_asis:
        return "MIXED-IN-TIME"
    if exact_flipped:
        return flipped_means
    if exact_asis:
        return asis_means
    n_f, n_a = rels.get("flipped-corr", 0), rels.get("asis-corr", 0)
    total = n_f + n_a
    if total == 0:
        return "UNKNOWN"
    if n_f / total >= 0.8:
        return flipped_means
    if n_a / total >= 0.8:
        return asis_means
    return "UNKNOWN"


def _job(kwargs) -> dict:
    store = kwargs.pop("store")
    try:
        return audit_store(store, **kwargs)
    except Exception as e:  # noqa: BLE001
        return {"store": store, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-900:], "arrays": {}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stores", nargs="+", required=True, help="store paths and/or globs")
    p.add_argument("--h5-dir", default=None, help="raw archive dir; enables the exact check")
    p.add_argument("--year", default=None, help="raw file prefix (default: from the store)")
    p.add_argument("--raw-order", choices=[NORTH_FIRST, "S->N"], default=None,
                   help="TRUE row order of the raw archive; required to score --h5-dir")
    p.add_argument("--group", default="input", help="raw H5 group")
    p.add_argument("--stride", type=int, default=60, help="timestep stride for sampling")
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--band", type=float, default=70.0, help="polar band edge |lat|")
    p.add_argument("--anchors-only", action="store_true",
                   help="absolute anchors only (1-2 reads/store): fast whole-store screen")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    if a.h5_dir and not a.raw_order:
        p.error("--h5-dir needs --raw-order (establish it with the anchors first)")

    stores: list[str] = []
    for s in a.stores:
        stores.extend(sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s])
    stores = [s for s in stores if os.path.isdir(s)]
    if not stores:
        print("no stores matched", file=sys.stderr)
        return 2
    print(f"checking {len(stores)} store(s), {a.workers} workers"
          f"{', anchors only' if a.anchors_only else ''}"
          f"{f', vs raw {a.h5_dir} [{a.raw_order}]' if a.h5_dir else ''}", flush=True)

    jobs = [
        {"store": s, "h5_dir": a.h5_dir, "year": a.year, "group": a.group,
         "raw_order": a.raw_order, "stride": a.stride, "max_samples": a.max_samples,
         "band": a.band, "anchors_only": a.anchors_only}
        for s in stores
    ]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(_job, j) for j in jobs]
        for fu in as_completed(futures):
            r = fu.result()
            results.append(r)
            base = os.path.basename(r["store"])
            if "error" in r:
                print(f"  {base:22s} ERROR {r['error']}", flush=True)
            else:
                bad = ",".join(r["bad_arrays"][:6]) + ("..." if len(r["bad_arrays"]) > 6 else "")
                print(f"  {base:22s} lat {r['lat_first']:+7.2f}..{r['lat_last']:+7.2f} "
                      f"{r['lat_order']:5s} {r['store_verdict']:18s} {bad or '-'}", flush=True)
    results.sort(key=lambda r: r["store"])

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
        print(f"wrote {a.json_out}")

    tally: dict[str, int] = {}
    for r in results:
        tally[r.get("store_verdict", "ERROR")] = tally.get(r.get("store_verdict", "ERROR"), 0) + 1
    print(f"\nSUMMARY: {tally}")
    problems = sum(
        1 for r in results
        if r.get("store_verdict") not in ("OK", "OK-WITH-UNKNOWNS", "NO-ARRAYS")
        or "error" in r
    )
    if problems:
        print(f"{problems} store(s) need attention.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
