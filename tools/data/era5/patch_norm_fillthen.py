#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patch fill-then-corrected mean/std into the ERA5 normalization stores, in place.

The ``normalization_pangu_s2s_{mean,std}.zarr`` stats were imported from
PanguWeather v2.0 ``.nc`` files (see ``build_normalization_zarr.py`` -- it
repackages, it does not compute). A 2026-07-24 audit against the actual era5/
archive (normalized std should be ~1) found four variables off:

* ``skin_temperature`` / ``soil_temperature_level_1`` -- imported stats from a
  different source distribution (std ~28/27 vs actual ~22/21; the imported mean
  ~270 is suspiciously the mask fill value). Normalized std was ~0.78.
* ``sea_surface_temperature`` / ``sea_ice_cover`` -- imported stats are
  ocean-only (nan-aware), but the training pipeline fills land NaN with the
  Pangu mask values (sst=270, sic=0) BEFORE normalizing, so the stats must be
  computed on the FILLED field ("fill-then"), not nan-aware.

The replacement values below were computed fill-then over the training years
1979-2018 (~monthly sampling of every year store). Post-patch, all four
variables normalize to std 0.97-1.01 on held-out data. Applied 2026-07-24 to
the stores on Stampede3 (master) and Midway (/scratch/midway2 training copy).

RULE: normalization stats for masked variables must fill NaN with the intended
mask value BEFORE computing mean/std, matching what the model actually sees.

Usage::

    python tools/data/era5/patch_norm_fillthen.py --root /path/to/era5
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr

# var: (mean, std) -- fill-then over era5/{1979..2018}.zarr
NEW_STATS = {
    "skin_temperature": (278.7409269512543, 22.15992946341383),
    "soil_temperature_level_1": (279.4315958472603, 21.30109405906873),
    "sea_surface_temperature": (281.13867839151646, 12.367769927038164),
    "sea_ice_cover": (0.11038578542353251, 0.2973422787968356),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", required=True,
        help="Directory holding normalization_pangu_s2s_{mean,std}.zarr.",
    )
    args = p.parse_args()
    gm = zarr.open_group(
        f"{args.root}/normalization_pangu_s2s_mean.zarr", mode="r+", use_consolidated=False
    )
    gs = zarr.open_group(
        f"{args.root}/normalization_pangu_s2s_std.zarr", mode="r+", use_consolidated=False
    )
    for v, (m, s) in NEW_STATS.items():
        om = float(np.asarray(gm[v]))
        osd = float(np.asarray(gs[v]))
        gm[v][...] = np.asarray(m, dtype=gm[v].dtype)
        gs[v][...] = np.asarray(s, dtype=gs[v].dtype)
        print(f"  {v:32s} mean {om:9.5g} -> {m:9.5g}   std {osd:9.5g} -> {s:9.5g}")
    # verify via a fresh consolidated read
    gm2 = zarr.open_group(
        f"{args.root}/normalization_pangu_s2s_mean.zarr", mode="r", use_consolidated=True
    )
    gs2 = zarr.open_group(
        f"{args.root}/normalization_pangu_s2s_std.zarr", mode="r", use_consolidated=True
    )
    ok = all(
        abs(float(np.asarray(gm2[v])) - m) < 1e-3 and abs(float(np.asarray(gs2[v])) - s) < 1e-3
        for v, (m, s) in NEW_STATS.items()
    )
    print(f"VERIFY_CONSOLIDATED_READBACK_OK={ok}  ROOT={args.root}")


if __name__ == "__main__":
    main()
