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

"""OFFLINE: pin each expert's precip day-alignment, kind, and units.

The archives carry no units/interval attrs, so the ``precip:`` spec of every
expert in ``conf/dataset/*.yaml`` (axis / kind / units / day_offset) is a
guess until verified against data. For a handful of (init, tau) samples per
expert this script assembles the daily precip with ``day_offset`` in
{-1, 0, +1} and reports, against the IMERG record of the adopted convention
(stamped ``date(init) + tau - 1``):

* pattern correlation over the monsoon box (the day_offset with the highest
  correlation is the right alignment);
* magnitude ratio mean(pred)/mean(obs) (a ~1000x ratio means the units are
  m not mm; a ratio growing linearly with tau means the variable is
  cumulative-since-init, not a per-day accumulation).

Record the verified values in the dataset yaml + DATA.md. Login-node safe
(numpy / xarray / zarr / yaml only — no torch, no physicsnemo).

Usage (Derecho, after the 1-degree regrid)::

    python examples/weather/ai_rossbypalooza/tools/verify_precip_alignment.py \\
        --dataset-yaml examples/weather/ai_rossbypalooza/conf/dataset/hindcast_derecho.yaml \\
        --taus 8 11 14 --n-inits 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr

_RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from datapipes.adapters import build_adapter  # noqa: E402
from datapipes.index import build_sample_index  # noqa: E402
from datapipes.precip import PrecipSpec  # noqa: E402
from datapipes.truth import ImergTruth  # noqa: E402
from datapipes.variables import ChannelLayout  # noqa: E402

MONSOON_BOX = (5.0, 35.0, 60.0, 100.0)


def _resolve(reqs, roots) -> list[np.ndarray]:
    out = []
    for r in reqs:
        owner, year, var = r.array_key
        grp = zarr.open_group(str(Path(roots[owner]) / f"{year}.zarr"), mode="r")
        arr = grp[var]
        sel = tuple(np.asarray(i) if isinstance(i, list) else i for i in r.index)
        if any(isinstance(s, np.ndarray) for s in sel):
            out.append(np.asarray(arr.oindex[sel], dtype=np.float32))
        else:
            out.append(np.asarray(arr[sel], dtype=np.float32))
    return out


def _box_mask(lat, lon, box):
    la, lo = np.asarray(lat), np.asarray(lon)
    return ((la >= box[0]) & (la <= box[1]))[:, None] & (
        (lo >= box[2]) & (lo <= box[3])
    )[None, :]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    a, b = a[ok], b[ok]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset-yaml", type=Path, required=True)
    p.add_argument("--expert", action="append", default=None,
                   help="restrict to these expert names (default: all)")
    p.add_argument("--taus", type=int, nargs="+", default=[8, 11, 14])
    p.add_argument("--n-inits", type=int, default=6)
    p.add_argument("--offsets", type=int, nargs="+", default=[-1, 0, 1])
    p.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9],
                   help="init months to sample (default JJAS — the monsoon "
                        "box has little precip variance off-season)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.dataset_yaml.read_text())
    layout = ChannelLayout(list(cfg["master_channels"]))
    truth = ImergTruth(cfg["truth"]["root"], var=cfg["truth"].get(
        "var", "total_precipitation_24hr"))
    truth.discover()
    box = _box_mask(truth.lat, truth.lon, MONSOON_BOX)

    experts = cfg["experts"]
    if args.expert:
        experts = [e for e in experts if e["name"] in set(args.expert)]

    train_block = cfg["train"]
    for e in experts:
        name = e["name"]
        base_spec = dict(e["precip"])
        base_offset = int(base_spec.pop("day_offset", 0))
        print(f"\n=== {name} ({e['schema']}) precip={base_spec} "
              f"config day_offset={base_offset} ===")
        results = {}
        for off in args.offsets:
            spec = PrecipSpec(**base_spec, day_offset=off)
            adapter = build_adapter(
                name, e["schema"], e["root"], layout, spec,
                exclude_variables=tuple(e.get("exclude_variables") or ()),
            )
            try:
                adapter.discover(truth.lat, truth.lon)
            except ValueError as exc:
                print(f"  discover failed: {exc}")
                results = None
                break
            idx = build_sample_index(
                [adapter], truth,
                years=tuple(train_block["years"]),
                init_months=list(args.months),
                lead_days=(min(args.taus), max(args.taus)),
            )
            # Sample PER TAU (a single strided slice can alias onto one tau).
            rows = []
            for t in args.taus:
                t_rows = [r for r in idx.pairs if int(r["tau"]) == t]
                step = max(1, len(t_rows) // args.n_inits)
                rows.extend(t_rows[::step][: args.n_inits])
            corrs, ratios = {t: [] for t in args.taus}, {t: [] for t in args.taus}
            roots = {name: e["root"]}
            for r in rows:
                tau = int(r["tau"])
                year, ii = idx.init_locs[int(r["init_row"]), 0]
                arrays = _resolve(
                    adapter.plan(int(year), int(ii), tau), roots
                )
                pred = adapter.assemble(arrays, tau)[0]  # mm/day channel
                tgt = _resolve(
                    [truth.plan(int(r["imerg_year"]), int(r["imerg_idx"]))],
                    {"__truth__": truth.root},
                )[0]
                pb, tb = pred[box], tgt[box]
                corrs[tau].append(_corr(pb, tb))
                ok = np.isfinite(tb) & np.isfinite(pb)
                if ok.any() and np.nanmean(tb[ok]) > 0:
                    ratios[tau].append(
                        float(np.nanmean(pb[ok]) / np.nanmean(tb[ok]))
                    )
            mean_corr = np.nanmean([c for t in args.taus for c in corrs[t]])
            results[off] = mean_corr
            per_tau = "  ".join(
                f"tau={t}: corr={np.nanmean(corrs[t]):.3f} "
                f"ratio={np.nanmean(ratios[t]) if ratios[t] else float('nan'):.3g}"
                for t in args.taus
            )
            print(f"  day_offset={off:+d}: mean corr={mean_corr:.3f} | {per_tau}")
        if results:
            best = max(results, key=lambda k: (results[k], -abs(k)))
            print(f"  -> best day_offset={best:+d}"
                  + ("  (matches config)" if best == base_offset
                     else f"  (config says {base_offset:+d} — UPDATE THE YAML)"))
            print("  hints: ratio ~1e3 -> units are m not mm; "
                  "ratio growing ~linearly with tau -> kind is cumulative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
