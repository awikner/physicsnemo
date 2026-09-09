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


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("sweep"); a.add_argument("--base", required=True); a.add_argument("--erdm")
    a.add_argument("--variant", action="append"); a.add_argument("--horizon", type=int, default=300)
    b = sp.add_parser("probes"); b.add_argument("--dir", required=True)
    args = ap.parse_args()
    {"sweep": cmd_sweep, "probes": cmd_probes}[args.cmd](args)


if __name__ == "__main__":
    main()
