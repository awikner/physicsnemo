# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tables for the Polaris RSI-drift test runs (plan: rsi-drift-polaris-test-plan.md).

Reads ``eval_suite.pt`` files produced by ``climate_eval_suite.py`` (the
5-year baselines and the 300-day sweep variants) and the ``summary.json`` /
``*.pt`` files written by ``rsi_drift_probes.py``. Plain torch + json; runs
anywhere the files have been copied to.

    python tools/diagnostics/rsi_drift/polaris_results.py sweep \
        --base eval_rsi_e24/eval_suite.pt --erdm eval_erdm_e24/eval_suite.pt \
        --variant k120=eval_sweep300_k120_e24/eval_suite.pt --variant k145=...
    python tools/diagnostics/rsi_drift/polaris_results.py probes --dir probes_e24
    python tools/diagnostics/rsi_drift/polaris_results.py alpha --run ft25=eval_bias1yr_base_e25/eval_suite.pt
    python tools/diagnostics/rsi_drift/polaris_results.py compare --run shipped=... --run ft28=...
    python tools/diagnostics/rsi_drift/polaris_results.py trace --sigma sigma_c_sstpred153.pt \\
        --names probes_e24/rsi/summary.json --run shipped=.../trace_rank0.pt --run ft28=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

GROUPS = ("surface", "upper_air", "diagnostic")
STEPS = (1, 5, 10, 20, 30, 50, 85, 100, 150, 200, 300)


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def _trace(d, kind, group, steps):
    ra = d["rmse_acc"]
    return [ra.get(f"{kind}_step{s}_{group}") for s in steps]


def _fmt(v, w=8, nd=3):
    return f"{v:>{w}.{nd}f}" if isinstance(v, (int, float)) and v is not None else f"{'-':>{w}}"


def cmd_sweep(args):
    base = _load(args.base)
    erdm = _load(args.erdm) if args.erdm else None
    variants = {}
    for spec in args.variant or []:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            variants[name] = _load(path)
        else:
            print(f"(missing {path})")
    steps = [s for s in STEPS if s <= args.horizon]
    for kind in ("spread", "rmse"):
        for g in GROUPS:
            print(f"\n=== {kind} vs climatology, {g} (8-member; normalized spread / physical RMSE) ===")
            print(f"{'step':>5} {'base':>8} {'ERDM':>8} " + " ".join(f"{n:>8}" for n in variants))
            for i, s in enumerate(steps):
                b = _trace(base, kind, g, [s])[0]
                e = _trace(erdm, kind, g, [s])[0] if erdm else None
                row = f"{s:>5} {_fmt(b)} {_fmt(e)} "
                row += " ".join(_fmt(_trace(v, kind, g, [s])[0]) for v in variants.values())
                print(row)
    # ratios at saturation-ish leads
    print("\n=== variant / base ratios (mean over steps 100-300 where available) ===")
    for kind in ("spread", "rmse"):
        for g in GROUPS:
            def mean_range(d):
                vals = [d["rmse_acc"].get(f"{kind}_step{s}_{g}") for s in range(100, min(args.horizon, 300) + 1)]
                vals = [x for x in vals if x is not None]
                return sum(vals) / len(vals) if vals else None
            b = mean_range(base)
            parts = []
            for n, v in variants.items():
                m = mean_range(v)
                parts.append(f"{n}={m / b:.3f}" if (m is not None and b) else f"{n}=-")
            e = mean_range(erdm) if erdm else None
            print(f"  {kind:>6} {g:>10}: base {b if b is None else round(b, 4)}  ERDM/base {'-' if not (e and b) else round(e / b, 3)}  " + "  ".join(parts))
    # headline bias at the run's end (partial if horizon < 1827)
    print("\n=== headline (bias-map RMSE) at run end ===")
    for name, d in [("base(5yr)", base)] + list(variants.items()):
        h = d.get("headline") or {}
        keys = [k for k in h if isinstance(h[k], dict)] if isinstance(h, dict) else []
        summ = {}
        for k in keys:
            v = h[k]
            for kk in ("rmse", "bias_rmse", "map_rmse"):
                if kk in v:
                    summ[k] = v[kk]
                    break
        if not summ and isinstance(h, dict):
            summ = {k: v for k, v in h.items() if isinstance(v, (int, float))}
        print(f"  {name}: " + ", ".join(f"{k}={v:.4g}" for k, v in list(summ.items())[:8]))


def cmd_probes(args):
    d = Path(args.dir)
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        sj = sub / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        print(f"\n##### {sub.name}: family={s['family']} weights={s['weights']} ICs={s['ic_rows']} ({s.get('seconds')} s)")
        names = s["channel_names"]
        if "readout" in s:
            for tk, r in s["readout"].items():
                print(f"  readout {tk}: slot-mean std_ratio {['%.3f' % x for x in r['std_ratio_per_slot_mean']]}")
                print(f"           slot-mean slope     {['%.3f' % x for x in r['slope_per_slot_mean']]}")
                print(f"           slot-mean level gain{['%.3f' % x for x in r['level_gain_per_slot_mean']]}")
                print(f"           slot-mean shrink ratio (x0.7 in) {['%.3f' % x for x in r['shrink_ratio_per_slot_mean']]}")
                sw = r["slotW_by_channel"]
                for ch in ("surface_pressure", "2m_temperature", "10m_v_component_of_wind", "PRATEsfc_24h",
                           "hcc_24h", "temperature@500", "geopotential@500", "v_component_of_wind@250"):
                    if ch in sw:
                        v = sw[ch]
                        print(f"           slot W {ch:28s} std_ratio {v['std_ratio']:.3f} slope {v['slope']:.3f} level_gain {v['level_gain']:.3f} shrink_ratio {v['shrink_ratio']:.3f}")
        if "cascade" in s:
            c = s["cascade"]
            print(f"  cascade emitted/truth anomaly std (mean over channels): {c['emit_over_truth_astd_mean_over_channels']}")
            if "anchor_over_truth_astd_mean_over_channels" in c:
                print(f"          anchor/truth anomaly std:                   {c['anchor_over_truth_astd_mean_over_channels']}")
                print(f"          anchor per-roll gain:                        {c['anchor_per_roll_gain_mean_over_channels']}")
            fin = c["emit_over_truth_astd_final_by_channel"]
            for ch in ("surface_pressure", "2m_temperature", "10m_v_component_of_wind", "PRATEsfc_24h",
                       "hcc_24h", "temperature@500", "geopotential@500", "v_component_of_wind@250"):
                if ch in fin:
                    print(f"          final emitted/truth {ch:28s} {fin[ch]:.3f}")
        if "flush" in s:
            f = s["flush"]
            for case in ("ic", "free"):
                print(f"  flush [{case}] retained fraction rms(pert-ref)/rms(pert) and two-seed floor:")
                for k in ("shrink", "floor_shrink", "offset", "floor_offset"):
                    row = " | ".join(f"{g}: " + " ".join(f"{r}:{v:.2f}" for r, v in f[case][k][g].items()) for g in f[case][k])
                    print(f"      {k:13s} {row}")



# --------------------------------------------------------------------------- #
# Shrinkage fit (the brief's section 3.5): bias = -alpha (obs - <obs>) + c
# --------------------------------------------------------------------------- #
def shrinkage_alpha(d, groups=("surface", "diagnostic", "upper_air")):
    """Per-channel lat-weighted fit of the time-mean bias map on the obs
    anomaly pattern; returns {group: (alpha[C(,L)], r2[C(,L)])}. Uses the
    ``climatology.{group}_bias`` and ``{group}_truth_mean`` maps written by
    climate_eval_suite (S->N, 180x360)."""
    clim = d["climatology"]
    out = {}
    for g in groups:
        b = clim.get(f"{g}_bias"); t = clim.get(f"{g}_truth_mean")
        if b is None or t is None:
            continue
        b = b.float(); t = t.float()
        H = b.shape[-2]
        lat = torch.linspace(-90 + 90 / H, 90 - 90 / H, H)
        w = torch.cos(torch.deg2rad(lat)).clamp_min(0)[:, None].expand(H, b.shape[-1])
        w = w / w.sum()
        def wmean(x): return (x * w).sum(dim=(-2, -1), keepdim=True)
        ta = t - wmean(t); ba = b - wmean(b)
        cov = (w * ta * ba).sum(dim=(-2, -1)); var = (w * ta * ta).sum(dim=(-2, -1))
        alpha = -cov / var.clamp_min(1e-30)
        resid = ba + alpha[..., None, None] * ta
        r2 = 1 - (w * resid * resid).sum(dim=(-2, -1)) / (w * ba * ba).sum(dim=(-2, -1)).clamp_min(1e-30)
        out[g] = (alpha, r2)
    return out


def cmd_alpha(args):
    names = None
    for spec in args.run:
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"(missing {path})"); continue
        d = _load(path)
        res = shrinkage_alpha(d)
        print(f"\n=== {name}: shrinkage alpha (R2) per channel; mean over channels per group ===")
        for g, (a, r2) in res.items():
            af = a.flatten(); rf = r2.flatten()
            print(f"  {g:>10}: mean alpha {af.mean():.3f}  median {af.median():.3f}  mean R2 {rf.mean():.2f}  n={af.numel()}")
        if "surface" in res:
            a, r2 = res["surface"]
            sfc = ["skin_temperature", "surface_pressure", "2m_temperature", "2m_specific_humidity", "10m_u", "10m_v"]
            print("  surface: " + ", ".join(f"{n} {float(a[i]):.3f} (R2 {float(r2[i]):.2f})" for i, n in enumerate(sfc[: a.numel()])))
        if "upper_air" in res and res["upper_air"][0].dim() == 2:
            a, r2 = res["upper_air"]
            print("  upper air mean alpha per variable (T,u,v,z,q): " + " ".join(f"{float(a[v].mean()):.3f}" for v in range(a.shape[0])))
        h = d.get("headline") or {}
        if isinstance(h, dict):
            flat = {k: v for k, v in h.items() if isinstance(v, (int, float))}
            print("  headline scalars: " + ", ".join(f"{k}={v:.4g}" for k, v in list(flat.items())[:10]) if flat else "  headline: " + str({k: type(v).__name__ for k, v in list(h.items())[:6]}))


# --------------------------------------------------------------------------- #
# Side-by-side table of eval_suite.pt files (Phase 4 fine-tune comparison)
# --------------------------------------------------------------------------- #
def cmd_compare(args):
    D = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            D[name] = _load(path)
        else:
            print(f"(missing {path})")
    names = list(D)
    hi = args.horizon

    def ra(d, k):
        return d["rmse_acc"].get(k)

    def f(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"

    def mean_rng(d, g, lo=100):
        v = [ra(d, f"rmse_step{s}_{g}") for s in range(lo, hi + 1)]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def row(label, fn):
        print(f"| {label} | " + " | ".join(fn(D[k]) for k in names) + " |")

    print("| | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for key, nd in (("z500", 0), ("t2m", 2), ("t850", 2), ("u250", 2), ("u10m", 2), ("v10m", 2)):
        row(f"{key} bias-map RMSE / mean bias",
            lambda d, key=key, nd=nd: f"{f(d['headline'][key]['rmse_bias'], nd)} / {f(d['headline'][key]['mean_bias'], nd)}"
            if key in (d.get("headline") or {}) else "-")
    row(f"surface RMSE-vs-clim mean steps 100-{hi}", lambda d: f(mean_rng(d, "surface"), 0))
    row(f"upper-air RMSE-vs-clim mean steps 100-{hi}", lambda d: f(mean_rng(d, "upper_air"), 0))
    row("surface RMSE-vs-clim at 30/50/85/200/365", lambda d: " / ".join(f(ra(d, f"rmse_step{s}_surface"), 0) for s in (30, 50, 85, 200, 365)))
    row("surface spread at 10/100/365", lambda d: " / ".join(f(ra(d, f"spread_step{s}_surface"), 3) for s in (10, 100, 365)))
    A = {k: shrinkage_alpha(D[k]) for k in names}
    print("| alpha skt/sp/t2m/q2m/u10/v10 | " + " | ".join(" / ".join(f"{float(A[k]['surface'][0][i]):.2f}" for i in range(6)) for k in names) + " |")
    print("| alpha upper-air T/u/v/z/q (mean over levels) | " + " | ".join(" / ".join(f"{float(A[k]['upper_air'][0][v].mean()):.2f}" for v in range(5)) for k in names) + " |")
    print("| alpha diagnostics mean | " + " | ".join(f"{float(A[k]['diagnostic'][0].mean()):.2f}" for k in names) + " |")
    print("\nsurface RMSE-vs-clim trace (steps 25..365 by 25):")
    for k in names:
        print(f"  {k:18s} " + " ".join(f(ra(D[k], f"rmse_step{s}_surface"), 0) for s in range(25, 366, 25)))


# --------------------------------------------------------------------------- #
# Anchor / emitted amplitude traces (RSI_TRACE_PATH files written by rsi.py)
# --------------------------------------------------------------------------- #
STRAT = ("5", "7", "10", "20", "30", "50", "70", "100")
TROP = ("125", "150", "175", "200", "250", "300", "400", "500", "600", "700", "800",
        "850", "875", "900", "925", "950", "975", "1000")
PICK = {"sp": "surface_pressure", "t2m": "2m_temperature", "v10": "10m_v_component_of_wind",
        "prate": "PRATEsfc_24h", "hcc": "hcc_24h", "T850": "temperature@850",
        "T500": "temperature@500", "z500": "geopotential@500", "u250": "u_component_of_wind@250",
        "v250": "v_component_of_wind@250", "q850": "specific_humidity@850",
        "q50": "specific_humidity@50", "T30": "temperature@30"}


def cmd_trace(args):
    T = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            T[name] = _load(path)
        else:
            print(f"(missing {path})")
    runs = list(T)
    rolls = [1, 5, 10, 20, 28, 35, 50, 70, 85, 100, 150, 200, 300, 365]
    lo, hi = args.late

    def rel(r, key="emit_std"):
        return T[r][key] / T[r]["emit_std"][0]

    def line(r, x):
        n = x.shape[0]
        return f"{r:16s} " + " ".join(f"{x[k - 1].mean():5.2f}" if k <= n else "    -" for k in rolls)

    C = T[runs[0]]["emit_std"].shape[1]
    masks = {}
    if args.sigma:
        sig = _load(args.sigma)
        sig = sig["sigma_c"] if isinstance(sig, dict) and "sigma_c" in sig else sig
        sig = torch.as_tensor(sig).flatten().float()[:C]
        q1, q2 = sig.quantile(1 / 3), sig.quantile(2 / 3)
        masks = {"slow S_c": sig < q1, "mid": (sig >= q1) & (sig < q2), "fast S_c": sig >= q2}
    for key in ("emit_std", "anchor_std"):
        print(f"\n=== {key} relative to the roll-1 emitted std; mean over channels ===")
        print(f"{'run':16s} " + " ".join(f"{k:>5}" for k in rolls))
        for r in runs:
            print(line(r, rel(r, key)))
        for tn, m in masks.items():
            print(f"  -- {tn} tercile")
            for r in runs:
                print("  " + line(r, rel(r, key)[:, m]))
    print("\n=== anchor / emitted, same roll, mean over channels ===")
    for r in runs:
        print(line(r, T[r]["anchor_std"] / T[r]["emit_std"]))
    print("\n=== mean over channels of |emitted spatial-mean change from roll 1| (normalized units) ===")
    for r in runs:
        print(line(r, (T[r]["emit_mean"] - T[r]["emit_mean"][0]).abs()))
    if args.names:
        names = json.loads(Path(args.names).read_text())
        names = names["channel_names"] if isinstance(names, dict) else names
        idx = {n: i for i, n in enumerate(names)}
        print(f"\n=== emitted std rel. roll 1 at rolls 28/50/100/200/365 (where available) ===")
        for k, n in PICK.items():
            if n not in idx:
                continue
            i = idx[n]
            print(f"  {k:6s} " + " | ".join(f"{r}: " + "/".join(f"{rel(r)[s - 1, i]:.2f}" for s in (28, 50, 100, 200, 365) if s <= rel(r).shape[0]) for r in runs))
        for label, levs in (("troposphere 125-1000 hPa", TROP), ("stratosphere 5-100 hPa", STRAT)):
            print(f"\n=== late (rolls {lo}-{hi}) emitted std rel. roll 1, {label}, per variable ===")
            for var in ("temperature", "u_component_of_wind", "v_component_of_wind", "geopotential", "specific_humidity"):
                ii = [idx[f"{var}@{p}"] for p in levs if f"{var}@{p}" in idx]
                if ii:
                    print(f"  {var:22s} " + "  ".join(f"{r}: {rel(r)[lo - 1:hi][:, ii].mean():.2f}" for r in runs))
        print(f"\n=== late (rolls {lo}-{hi}) emitted std rel. roll 1, surface + diagnostic channels ===")
        for r in runs:
            print(f"  {r:16s} {rel(r)[lo - 1:hi][:, :21].mean():.2f}")
        ii = [idx[f"v_component_of_wind@{p}"] for p in TROP if f"v_component_of_wind@{p}" in idx]
        print("\n=== first roll where the tropospheric v-wind amplitude drops below 0.7 ===")
        for r in runs:
            x = rel(r)[:, ii].mean(1)
            below = (x < 0.7).nonzero()
            print(f"  {r:16s} roll {int(below[0]) + 1 if len(below) else '-'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("sweep"); a.add_argument("--base", required=True); a.add_argument("--erdm")
    a.add_argument("--variant", action="append"); a.add_argument("--horizon", type=int, default=300)
    b = sp.add_parser("probes"); b.add_argument("--dir", required=True)
    c = sp.add_parser("alpha"); c.add_argument("--run", action="append", required=True,
                                              help="name=path/to/eval_suite.pt (repeatable)")
    d = sp.add_parser("compare"); d.add_argument("--run", action="append", required=True,
                                                help="name=path/to/eval_suite.pt (repeatable)")
    d.add_argument("--horizon", type=int, default=365)
    e = sp.add_parser("trace"); e.add_argument("--run", action="append", required=True,
                                              help="name=path/to/trace_rank0.pt (repeatable)")
    e.add_argument("--sigma", help="sigma_c_sstpred153.pt for the S_c terciles")
    e.add_argument("--names", help="probe summary.json (channel_names) for per-channel rows")
    e.add_argument("--late", type=int, nargs=2, default=(100, 365), metavar=("LO", "HI"))
    args = ap.parse_args()
    {"sweep": cmd_sweep, "probes": cmd_probes, "alpha": cmd_alpha,
     "compare": cmd_compare, "trace": cmd_trace}[args.cmd](args)


if __name__ == "__main__":
    main()
