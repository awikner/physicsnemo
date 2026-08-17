#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reverse a single array over a TIMESTEP RANGE, to repair a half-flipped array.

``flip_lat_zarr.py`` rewrites in 60-timestep blocks, so an interrupted run leaves an
array correct over part of the year and upside-down over the rest -- while the store
still carries ``lat_flipped_to_NtoS=True`` and looks repaired. Three ERA5 stores are
in that state (see docs/dev/context/lat-orientation-audit.md):

    era5/1987  sea_surface_temperature  t=0..1459     (the whole year)
    era5/1988  temperature              t=1260..1463
    era5/1989  temperature              t=420..1459

``flip_lat_zarr.py`` cannot fix these: it flips whole arrays, which would just move
the damage to the other part of the year. ``redo_lat_arrays.py`` can, but only where
the raw archive is co-located. This tool works from the store alone.

It refuses to act on an unverified range. Before writing anything it checks, from the
data, that

  1. the requested range really IS upside-down -- the target's zonal-anomaly pattern
     inside the range matches a lat-REVERSED anchored reference array better than the
     as-is one, and
  2. the rest of the array is NOT -- a sample outside the range matches as-is,

so a mistyped range fails closed rather than corrupting good data. After writing it
re-checks both, and confirms the seam at each boundary is gone.

Reference arrays are the ones an absolute physical anchor can vouch for (land mask,
surface geopotential, solstice insolation, seasonal temperature), so "correct" here
means agreeing with the Earth, not with a neighbouring array.

Usage::

    repair_lat_ranges.py --store .../era5/1989.zarr --array temperature --range 420:1460
    repair_lat_ranges.py --store .../era5/1988.zarr --array temperature --range 1260:1464
    repair_lat_ranges.py --store .../era5/1987.zarr --array sea_surface_temperature \\
        --range 0:1460 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_lat_orientation as clo  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("repair_lat_ranges")

MANIFEST = "lat_range_repairs"
#: Timesteps per read/flip/write block: bounds memory while keeping bulk I/O.
IO_BLOCK = 60


def _anchored_references(store: str, band: float = 70.0) -> tuple[list[str], dict]:
    """Arrays whose orientation an absolute physical anchor can vouch for."""
    rep = clo.audit_store(store, band=band, anchors_only=True)
    refs = [
        n for n, d in rep["arrays"].items()
        if d["verdict"] == "OK" and d.get("decided_by") == "A"
    ]
    return refs, rep


def _orientation_at(g, array: str, t: int, refs: list[str], lev: int) -> tuple[str, float, float]:
    """Is ``array`` at timestep ``t`` oriented like the (correct) references?"""
    target = clo.slice_2d(g, array, t, lev)
    if target is None:
        return "UNKNOWN", float("nan"), float("nan")
    best = None
    for r in refs:
        if r == array:
            continue
        rt = None if g[r].ndim == 2 else t
        ref = clo.slice_2d(g, r, rt, 0)
        if ref is None or ref.shape != target.shape:
            continue
        c_a = clo.pattern_corr(target, ref)
        c_f = clo.pattern_corr(target, ref[::-1])
        rel, _margin = clo.relation_from_corr(c_a, c_f)
        if rel == "UNKNOWN":
            continue
        strength = max(abs(c_a), abs(c_f))
        if best is None or strength > best[3]:
            best = (rel, c_a, c_f, strength)
    if best is None:
        return "UNKNOWN", float("nan"), float("nan")
    rel, c_a, c_f, _s = best
    return ("OK" if rel == "SAME" else "FLIPPED"), c_a, c_f


def _seam_at(g, array: str, t: int, lev: int) -> bool:
    """Is there an orientation discontinuity between t-1 and t?"""
    if t <= 0 or t >= g[array].shape[0]:
        return False
    f0 = clo.slice_2d(g, array, t - 1, lev)
    f1 = clo.slice_2d(g, array, t, lev)
    if f0 is None or f1 is None:
        return False
    c_a = clo.pattern_corr(f0, f1)
    c_f = clo.pattern_corr(f0, f1[::-1])
    return bool(np.isfinite(c_a) and np.isfinite(c_f) and c_f > c_a + clo.MARGIN)


def repair(store: str, array: str, ranges: list[tuple[int, int]], *,
           dry_run: bool = False, force: bool = False, band: float = 70.0) -> int:
    refs, anchor_rep = _anchored_references(store, band)
    if not refs:
        logger.error("%s: no array here can be anchored physically, so nothing can "
                     "vouch for 'correct'; refusing", os.path.basename(store))
        return 2
    logger.info("anchored reference arrays: %s", refs)

    g = zarr.open_group(store, mode="r" if dry_run else "r+", use_consolidated=False)
    if array not in set(g.array_keys()):
        logger.error("%s: no array %r", store, array)
        return 2
    arr = g[array]
    n_time = int(arr.shape[0])
    # Level choice matters only for the diagnostics (the flip covers all levels), but
    # it decides whether they can conclude anything: an upper-tropospheric level
    # correlates weakly with the surface anchors, so try the SURFACE-MOST level first
    # (largest pressure / sigma), then mid-atmosphere, then the top.
    lev_candidates = [0]
    if arr.ndim == 4:
        for ln in ("pressure_level", "sigma_level", "level"):
            if ln in set(g.array_keys()):
                vals = np.asarray(g[ln][:], dtype="float64")
                lev_candidates = list(
                    dict.fromkeys(
                        [int(np.argmax(vals)), int(vals.size // 2), int(np.argmin(vals))]
                    )
                )
                break
    lev = lev_candidates[0]

    done = {json.dumps(x, sort_keys=True) for x in (g.attrs.get(MANIFEST) or [])}
    rc = 0
    for start, stop in ranges:
        tag = json.dumps({"array": array, "start": start, "stop": stop}, sort_keys=True)
        label = f"{os.path.basename(store)}/{array}[{start}:{stop}]"
        if tag in done:
            logger.info("%s: already repaired (manifest); skip", label)
            continue
        if not (0 <= start < stop <= n_time):
            logger.error("%s: range outside 0..%d", label, n_time)
            rc = 2
            continue

        # A partially applied range is mid-repair: its blocks are recorded, so the
        # pre-flight gate below would now see already-corrected data inside the range
        # and refuse. It was validated on the first attempt; resume instead.
        resuming = any(
            json.dumps({"array": array, "start": s, "stop": min(s + IO_BLOCK, stop)},
                       sort_keys=True) in done
            for s in range(start, stop, IO_BLOCK)
        )

        # --- pre-flight: the range must be wrong, and the rest must be right ---
        inside = start + (stop - start) // 2
        v_in, ca_in, cf_in = "UNKNOWN", float("nan"), float("nan")
        for cand in lev_candidates:
            v_in, ca_in, cf_in = _orientation_at(g, array, inside, refs, cand)
            if v_in != "UNKNOWN":
                lev = cand
                break
        if arr.ndim == 4:
            logger.info("%s: diagnostics use level index %d of %s",
                        label, lev, lev_candidates)
        outside = next((t for t in (stop + 5, start - 5, n_time - 1, 0)
                        if 0 <= t < n_time and not (start <= t < stop)), None)
        v_out = None
        if outside is not None:
            v_out, ca_out, cf_out = _orientation_at(g, array, outside, refs, lev)
            logger.info("%s: outside t=%d -> %s (corr as-is %+.3f, reversed %+.3f)",
                        label, outside, v_out, ca_out, cf_out)
        logger.info("%s: inside  t=%d -> %s (corr as-is %+.3f, reversed %+.3f)",
                    label, inside, v_in, ca_in, cf_in)

        if resuming:
            logger.info("%s: resuming a partially applied range (blocks already "
                        "recorded); pre-flight gate was passed on the first attempt",
                        label)
        if v_in != "FLIPPED" and not (force or resuming):
            logger.error("%s: the range does not look upside-down (verdict %s). "
                         "Refusing; pass --force only if you are certain.", label, v_in)
            rc = 2
            continue
        if v_out == "FLIPPED" and not (force or resuming):
            logger.error("%s: data OUTSIDE the range is also upside-down, so this is not "
                         "a partial flip — use flip_lat_zarr.py on the whole array. "
                         "Refusing.", label)
            rc = 2
            continue
        if dry_run:
            logger.info("%s: WOULD reverse %d timesteps along lat", label, stop - start)
            continue

        # --- apply, in blocks, recording EACH block as it lands ----------------
        # Progress is persisted per block, not per range: a write can fail partway
        # (a nearly-full Lustre returns EDQUOT intermittently), and re-running with
        # the same range must then resume rather than reverse what already landed.
        # Without this, a retry double-flips the completed blocks.
        lat_axis = arr.ndim - 2
        for s in range(start, stop, IO_BLOCK):
            e = min(s + IO_BLOCK, stop)
            block_tag = json.dumps(
                {"array": array, "start": s, "stop": e}, sort_keys=True
            )
            if block_tag in done:
                continue
            arr[s:e] = np.flip(np.asarray(arr[s:e]), axis=lat_axis)
            done.add(block_tag)
            g.attrs[MANIFEST] = [json.loads(x) for x in sorted(done)]
            zarr.consolidate_metadata(g.store)
        done.add(tag)
        g.attrs[MANIFEST] = [json.loads(x) for x in sorted(done)]
        g.attrs["lat_range_repair_note"] = (
            "Ranges listed in lat_range_repairs were reversed along lat to repair an "
            "array left half-flipped by an interrupted flip_lat_zarr.py run. Verified "
            "against physically anchored arrays before and after. See "
            "docs/dev/context/lat-orientation-audit.md."
        )
        zarr.consolidate_metadata(g.store)
        logger.info("%s: reversed %d timesteps", label, stop - start)

        # --- post-flight -------------------------------------------------------
        v_in2, ca2, cf2 = _orientation_at(g, array, inside, refs, lev)
        logger.info("%s: after -> %s (corr as-is %+.3f, reversed %+.3f)",
                    label, v_in2, ca2, cf2)
        if v_in2 != "OK":
            logger.error("%s: STILL not correct after the repair — investigate", label)
            rc = 1
        for boundary in (start, stop):
            if _seam_at(g, array, boundary, lev):
                logger.error("%s: a seam remains at t=%d", label, boundary)
                rc = 1
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store", required=True)
    p.add_argument("--array", required=True)
    p.add_argument("--range", dest="ranges", action="append", required=True,
                   metavar="START:STOP", help="half-open timestep range; repeatable")
    p.add_argument("--band", type=float, default=70.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="skip the pre-flight refusal (you had better be sure)")
    a = p.parse_args(argv)

    ranges: list[tuple[int, int]] = []
    for spec in a.ranges:
        lo, _, hi = spec.partition(":")
        ranges.append((int(lo), int(hi)))
    return repair(a.store, a.array, ranges, dry_run=a.dry_run,
                  force=a.force, band=a.band)


if __name__ == "__main__":
    sys.exit(main())
