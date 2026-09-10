# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
r"""Per-channel latent scale S for RSI on the AMIP ``channel_layout: v2`` packs.

Computes the normalized L-step increment std -- std(x_{t+L} - x_t) / std(x) --
per state channel, packed in the ``channel_layout: v2`` order:

    [ surface(6) | diagnostics(15) |
      upper-air level-major, 1000 hPa first, [T,u,v,z,q] within level (130) |
      ocean tail ]

with the ocean tail set by ``--contract``:

    fancy     (154 ch, conf/model/amip_rsi_fancy.yaml):
              [sst, DERIVED sst anomaly (uses sst's ratio), sea ice]
    sst_pred  (153 ch, conf/model/amip_rsi_sst_pred.yaml): [sst, sea ice]

Point ``loss.noise_scale_path`` at the output; with it ``gamma_0``/``gamma_1``
are in INCREMENT units (gamma_0 = 1.0 injects one increment std of latent per
channel instead of a fixed z-scored amplitude that over-corrupts the slow
channels ~30x; docs/dev/context/rsi-h1-precond-instability.md).

**Lag and step.** ``--lag L`` is the anchor lag of the scheduler
(``RSIScheduler(anchor_lag=L)``), in MODEL steps: slot w interpolates from
frame w-L, so Gamma(0) has to be the L-step increment scale (proposal v0.2,
rung A2-L uses L = W = 6). A model step is ``--step-hours`` (24) of store
rows: the AMIP daily-average archives are stored at a 6-hourly cadence
(``data_timedelta_hours: 6``, 1460 rows a year), so one step is 4 rows and
the L-step increment is ``x[t + 4L] - x[t]``. ``--step-rows`` overrides the
derived stride. NOTE: the artifacts shipped on 2026-09-02
(``sigma_c_fancy154.pt``, ``sigma_c_sstpred153.pt``) were built by the
previous version of this script with ``np.diff`` over ROWS, i.e. they are
6-HOUR increment scales of 24-hour running means, not 1-step scales;
``--step-rows 1 --lag 1`` reproduces them (and ``--compare`` checks the
channel order against them).

Relation to ``make_noise_scales.py``: that tool packs through the wrapper
itself (authoritative for any contract), but it cannot fill the DERIVED
``sea_surface_temperature_anomaly`` channel, which exists only after the 12g
rescaler, and it does not read the boundary store for the ocean tail. This
tool hardcodes the v2 pack (verified positionally against the live
RSI_LOSS_DIAG channel decode: surface_pressure ~0.017 at ch 1, v@900 ~0.29 at
ch 43, cloud/precip diagnostics as the healthy top-5).

Plain zarr/numpy -- safe on login nodes (no physicsnemo import).

Usage::

    python tools/data/amip/make_rsi_delta_scales.py \
        --data-root $AI_ROSSBY_DATA \
        --state-store amip_dailyavg_coarse_train7914 \
        --boundary-store amip_dailyavg_boundary_train7914 \
        --years 1985 1995 2005 --contract sst_pred --lag 6 \
        --out $AI_ROSSBY_DATA/norm_stats/sigma_c_sstpred153_lag6.pt

A ``.json`` sidecar next to the output records lag, step, years, contract and
the per-channel names/values.
"""

import argparse
import json
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
SST = "sea_surface_temperature_monthly_interp"
ICE = "sea_ice_cover_monthly_interp"
#: ocean tail per contract: (name, source variable) -- the anomaly channel is
#: derived from sst by the 12g rescaler, so it takes sst's ratio.
OCEAN = {
    "fancy": [(SST, SST), ("sea_surface_temperature_anomaly", SST), (ICE, ICE)],
    "sst_pred": [(SST, SST), (ICE, ICE)],
}
N_CHANNELS = {"fancy": 154, "sst_pred": 153}


def _ratio(a, k=1):
    """k-row increment std / field std, NaN-tolerant (SST/ice are land-masked)."""
    a = np.asarray(a, dtype="float64")
    s = np.nanstd(a)
    if a.shape[0] <= k:
        raise ValueError(f"need more than {k} rows for a {k}-row increment, got {a.shape[0]}")
    return np.nanstd(a[k:] - a[:-k]) / s if s > 0 else np.nan


def _step_rows(stores, step_hours, override):
    """Rows per model step: ``--step-rows`` if given, else derived from the
    store's ``data_timedelta_hours`` (raises if the two do not divide)."""
    if override is not None:
        return int(override)
    dt = None
    for st in stores:
        dt = st.attrs.get("data_timedelta_hours", dt)
        if dt is not None:
            break
    if dt is None:
        raise SystemExit(
            "store carries no data_timedelta_hours attr; pass --step-rows")
    dt = float(dt)
    if step_hours % dt:
        raise SystemExit(
            f"--step-hours {step_hours} is not a multiple of the store cadence "
            f"{dt} h; pass --step-rows")
    return int(round(step_hours / dt))


def compute_scales(stores, bstores, contract, lag, step_rows, log=print):
    """Return ``(values, names)`` for the contract's packed channels.

    Per-year ratios are averaged over the years (finite ones only). Each
    upper-air variable is read ONCE per year and sliced in memory: the
    per-level lazy slice re-reads every level-spanning chunk 26 times.
    """
    k = int(lag) * int(step_rows)
    lv = {lev: i for i, lev in enumerate(LEVELS)}
    ocean_vars = sorted({var for _, var in OCEAN[contract]})
    per_year = []
    for st, b in zip(stores, bstores):
        r = {}
        for v in SURFACE + DIAG:
            r[v] = _ratio(np.asarray(st[v]), k)
        for v in UPPER:
            full = np.asarray(st[v])                    # (T, L, H, W) once
            for lev in LEVELS:
                r[f"{v}@{lev}"] = _ratio(full[:, lv[lev]], k)
            del full
        for var in ocean_vars:
            r[var] = _ratio(np.asarray(b[var]), k)
        per_year.append(r)
        log(f"  year {len(per_year)}/{len(stores)} done")
    names = (list(SURFACE + DIAG)
             + [f"{v}@{lev}" for lev in reversed(LEVELS) for v in UPPER]   # 1000 first
             + [name for name, _ in OCEAN[contract]])
    source = {name: var for name, var in OCEAN[contract]}

    def _mean(name):
        key = source.get(name, name)
        xs = [d[key] for d in per_year if np.isfinite(d[key])]
        return float(np.mean(xs)) if xs else np.nan

    vals = np.asarray([_mean(n) for n in names], dtype="float32")
    if len(vals) != N_CHANNELS[contract]:
        raise SystemExit(
            f"packed {len(vals)} channels, expected {N_CHANNELS[contract]} for {contract}")
    return vals, names


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--state-store", default="amip_dailyavg_coarse_train7914")
    p.add_argument("--boundary-store", default="amip_dailyavg_boundary_train7914")
    p.add_argument("--years", type=int, nargs="+", default=[1985, 1995, 2005],
                   help="years to average over; spread them across the record")
    p.add_argument("--contract", choices=sorted(N_CHANNELS), default="fancy",
                   help="which v2 pack: fancy (154, derived SST anomaly) or "
                        "sst_pred (153)")
    p.add_argument("--lag", type=int, default=1,
                   help="anchor lag L in MODEL steps (RSIScheduler.anchor_lag); "
                        "the increment is x[t + L*step_rows] - x[t]")
    p.add_argument("--step-hours", type=float, default=24.0,
                   help="model step in hours (model.timedelta_hours)")
    p.add_argument("--step-rows", type=int, default=None,
                   help="rows per model step; default derived from the store's "
                        "data_timedelta_hours (4 on the 6-hourly AMIP stores). "
                        "1 reproduces the 2026-09-02 artifacts (6-h increments)")
    p.add_argument("--floor", type=float, default=0.01,
                   help="minimum scale -- a zero would make Gamma singular")
    p.add_argument("--compare", default=None,
                   help="an existing (C,1,1) artifact to print per-channel "
                        "ratios against (channel-order check)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.lag < 1:
        raise SystemExit("--lag must be >= 1")

    root = Path(args.data_root)
    stores = [zarr.open(str(root / args.state_store / f"{y}.zarr"), mode="r")
              for y in args.years]
    bstores = [zarr.open(str(root / args.boundary_store / f"{y}.zarr"), mode="r")
               for y in args.years]
    step_rows = _step_rows(stores, args.step_hours, args.step_rows)
    k = args.lag * step_rows
    print(f"contract={args.contract} lag={args.lag} step(s) x {step_rows} row(s) "
          f"= {k}-row increments over years {args.years}")

    vals, names = compute_scales(stores, bstores, args.contract, args.lag, step_rows)
    bad = [names[i] for i in range(len(vals)) if not np.isfinite(vals[i])]
    if bad:
        raise SystemExit(f"non-finite increment ratio for {bad}")
    raw = vals.copy()
    vals = np.clip(vals, args.floor, None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # bare (C, 1, 1) tensor: what `loss.noise_scale_path` torch.loads
    torch.save(torch.from_numpy(vals)[:, None, None].contiguous(), out)
    sidecar = {
        "contract": args.contract,
        "n_channels": int(len(vals)),
        "lag_steps": int(args.lag),
        "step_rows": int(step_rows),
        "step_hours": float(args.step_hours),
        "increment_rows": int(k),
        "years": [int(y) for y in args.years],
        "floor": float(args.floor),
        "state_store": args.state_store,
        "boundary_store": args.boundary_store,
        "channels": names,
        "values": [float(v) for v in vals],
        "values_unfloored": [float(v) for v in raw],
        "note": "std(x[t+increment_rows] - x[t]) / std(x), normalized units, "
                "averaged over the listed years; channel order = channel_layout v2",
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
    print(f"saved {out} ({len(vals)},1,1) + {out.with_suffix('.json').name}; "
          f"mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f} "
          f"floored={int((raw < args.floor).sum())}")
    show = [1, 9, 14, 21, 43] + list(range(len(vals) - len(OCEAN[args.contract]), len(vals)))
    for i in show:
        print(f"  ch {i:3d} {names[i]:45s} {vals[i]:.4f}")

    if args.compare:
        ref = torch.load(args.compare, map_location="cpu")
        ref = ref.reshape(-1).numpy().astype("float32")
        if ref.shape[0] != vals.shape[0]:
            raise SystemExit(
                f"--compare has {ref.shape[0]} channels, this pack {vals.shape[0]}")
        r = vals / np.maximum(ref, 1e-12)
        print(f"vs {args.compare}: ratio new/ref mean={r.mean():.4f} "
              f"min={r.min():.4f} max={r.max():.4f}; "
              f"max |log ratio| at ch {int(np.abs(np.log(r)).argmax())} "
              f"({names[int(np.abs(np.log(r)).argmax())]})")
        order = np.argsort(-np.abs(np.log(r)))[:8]
        for i in order:
            print(f"  ch {i:3d} {names[i]:45s} new={vals[i]:.4f} ref={ref[i]:.4f} "
                  f"ratio={r[i]:.3f}")


if __name__ == "__main__":
    main()
