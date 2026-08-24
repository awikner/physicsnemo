# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
r"""Per-channel latent scale S for RSI on the ERDM_fancy (154-channel) contract.

Computes the normalized 1-step increment std — std(x_{t+1} - x_t) / std(x) —
per state channel, packed in the ``channel_layout: v2`` order of
``conf/model/amip_rsi_fancy.yaml``:

    [ surface(6) | diagnostics(15) |
      upper-air level-major, 1000 hPa first, [T,u,v,z,q] within level (130) |
      ocean(3) ]

Point ``loss.noise_scale_path`` at the output; with it, ``gamma_0``/``gamma_1``
are in INCREMENT units (the plan's Phase-A default: gamma_0=1.0 injects one
increment-std of latent per channel instead of a fixed z-scored amplitude that
over-corrupts slow channels ~30x — measured to be a large driver of the A2
training instability, docs/dev/context/rsi-h1-precond-instability.md).

Relation to ``make_noise_scales.py``: that tool packs through the wrapper
itself (authoritative for any contract), but it cannot fill the DERIVED
``sea_surface_temperature_anomaly`` channel, which exists only after the 12g
rescaler. This tool hardcodes the fancy pack (verified positionally against
the live RSI_LOSS_DIAG channel decode: surface_pressure ~0.017 at ch 1,
v@900 ~0.29 at ch 43, cloud/precip diagnostics as the healthy top-5) and
uses the SST ratio for the anomaly channel.

Plain zarr/numpy — safe on login nodes (no physicsnemo import).

Usage::

    python tools/data/amip/make_rsi_delta_scales.py \
        --data-root $AI_ROSSBY_DATA \
        --state-store amip_dailyavg_coarse_train7914 \
        --boundary-store amip_dailyavg_boundary_train7914 \
        --years 1985 1995 2005 \
        --out $AI_ROSSBY_DATA/norm_stats/sigma_c_fancy154.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr

SURFACE = ["skin_temperature", "surface_pressure", "2m_temperature",
           "2m_specific_humidity", "10m_u_component_of_wind",
           "10m_v_component_of_wind"]
DIAG = ["USWRFtoa_24h", "ULWRFtoa_24h", "USWRFsfc_24h", "ULWRFsfc_24h",
        "DSWRFsfc_24h", "DLWRFsfc_24h", "LHTFLsfc_24h", "SHTFLsfc_24h",
        "PRATEsfc_24h", "hcc_24h", "lcc_24h", "mcc_24h", "mn2t_24h",
        "mx2t_24h", "mxtpr_24h"]
UPPER = ["temperature", "u_component_of_wind", "v_component_of_wind",
         "geopotential", "specific_humidity"]
LEVELS = [5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 400,
          500, 600, 700, 800, 850, 875, 900, 925, 950, 975, 1000]
#: pack order: [sst, DERIVED anomaly (uses sst's ratio), sea ice]
OCEAN = ["sea_surface_temperature_monthly_interp",
         "sea_surface_temperature_anomaly",
         "sea_ice_cover_monthly_interp"]


def _ratio(a):
    """increment std / field std, NaN-tolerant (SST/ice are land-masked)."""
    s = np.nanstd(a)
    return np.nanstd(np.diff(a, axis=0)) / s if s > 0 else np.nan


def _mean_ratio(reads):
    r = [x for x in (_ratio(a) for a in reads) if np.isfinite(x)]
    return float(np.mean(r)) if r else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--state-store", default="amip_dailyavg_coarse_train7914")
    p.add_argument("--boundary-store", default="amip_dailyavg_boundary_train7914")
    p.add_argument("--years", type=int, nargs="+", default=[1985, 1995, 2005],
                   help="years to average over; spread them across the record")
    p.add_argument("--floor", type=float, default=0.01,
                   help="minimum scale — a zero would make Gamma singular")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.data_root)
    stores = [zarr.open(str(root / args.state_store / f"{y}.zarr"), mode="r")
              for y in args.years]
    bstores = [zarr.open(str(root / args.boundary_store / f"{y}.zarr"), mode="r")
               for y in args.years]

    vals, names = [], []
    for v in SURFACE + DIAG:
        vals.append(_mean_ratio([np.asarray(st[v]) for st in stores]))
        names.append(v)
    lv = {lev: i for i, lev in enumerate(LEVELS)}
    for lev in reversed(LEVELS):                      # 1000 hPa first
        for v in UPPER:
            # lazy per-level slice: (T, H, W), a few MB per read
            vals.append(_mean_ratio([st[v][:, lv[lev]] for st in stores]))
            names.append(f"{v}@{lev}")
    sst = _mean_ratio(
        [np.asarray(b["sea_surface_temperature_monthly_interp"]) for b in bstores])
    ice = _mean_ratio(
        [np.asarray(b["sea_ice_cover_monthly_interp"]) for b in bstores])
    vals += [sst, sst, ice]
    names += OCEAN

    vals = np.asarray(vals, dtype="float32")
    if len(vals) != 154:
        raise SystemExit(f"packed {len(vals)} channels, expected 154")
    bad = [names[i] for i in range(len(vals)) if not np.isfinite(vals[i])]
    if bad:
        raise SystemExit(f"non-finite increment ratio for {bad}")
    vals = np.clip(vals, args.floor, None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # bare (C, 1, 1) tensor: what `loss.noise_scale_path` torch.loads
    torch.save(torch.from_numpy(vals)[:, None, None].contiguous(), out)
    print(f"saved {out} (154,1,1); mean={vals.mean():.4f} "
          f"min={vals.min():.4f} max={vals.max():.4f}")
    for i in [1, 9, 14, 43, 151, 152, 153]:
        print(f"  ch {i:3d} {names[i]:45s} {vals[i]:.4f}")


if __name__ == "__main__":
    main()
