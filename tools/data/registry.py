#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ai-rossby data-location registry tool (Phase 11).

Reads and maintains ``hpc/data_registry.yaml`` — the catalog of which converted
Zarr datasets exist and on which clusters, so a lost copy can be re-transferred
(see ``sync_dataset.py``).

    registry.py show                        # print the catalog + coverage
    registry.py check                       # report gaps + at-risk copies (exit 1 if any)
    registry.py scan <cluster> [--data-root P] [--write]
                                            # scan a cluster's data_root, update its copies

``scan`` runs on the cluster it is scanning (it reads a local filesystem); run
it there directly or over ssh. Year ranges are compact strings like
``"2041-2049"`` / ``"1981"`` / ``"1981,1983-1985"``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "hpc" / "data_registry.yaml"

_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ai-rossby data-location registry (Phase 11). Maintained by
# tools/data/registry.py and consumed by tools/data/sync_dataset.py.
# Year ranges are compact strings: "2041-2049", "1981", or "1981,1983-1985".
"""


# --------------------------------------------------------------------------- #
# Year-range helpers
# --------------------------------------------------------------------------- #
def parse_years(spec) -> set[int]:
    """"2041-2049" / "1981" / "1981,1983-1985" → set of ints."""
    out: set[int] = set()
    if not spec:
        return out
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def compress_years(years) -> str:
    """set/iterable of ints → compact range string (inverse of parse_years)."""
    ys = sorted(set(int(y) for y in years))
    if not ys:
        return ""
    runs: list[tuple[int, int]] = []
    start = prev = ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            runs.append((start, prev))
            start = prev = y
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def load(path: Path = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save(reg: dict, path: Path = REGISTRY_PATH) -> None:
    body = yaml.safe_dump(reg, sort_keys=False, default_flow_style=False, width=100)
    with open(path, "w") as f:
        f.write(_HEADER + "\n" + body)


def resolve_cluster(reg: dict, name: str) -> tuple[str, dict]:
    """Resolve a cluster name, following ``alias_of`` (e.g. deltaai → delta)."""
    c = reg["clusters"].get(name)
    if c is None:
        raise SystemExit(f"unknown cluster: {name!r} (known: {', '.join(reg['clusters'])})")
    if "alias_of" in c:
        canon = c["alias_of"]
        return canon, reg["clusters"][canon]
    return name, c


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
_YEAR_RE = re.compile(r"^(\d+)\.zarr$")  # per-year stores only (skips normalization_*, *_q1, …)


def scan(
    reg: dict,
    cluster: str,
    data_root: str | None = None,
    write: bool = False,
    registry_path: Path = REGISTRY_PATH,
) -> None:
    canon, cinfo = resolve_cluster(reg, cluster)
    root = Path(data_root or cinfo.get("data_root") or "")
    if not root or not root.is_dir():
        raise SystemExit(f"data_root not found for {cluster}: {root!r}")
    print(f"scanning {canon} at {root}")
    for dsname, ds in reg["datasets"].items():
        subdir = root / ds["subdir"]
        years: set[int] = set()
        if subdir.is_dir():
            for p in subdir.iterdir():
                m = _YEAR_RE.match(p.name)
                if m and p.is_dir():
                    years.add(int(m.group(1)))
        rng = compress_years(years)
        copies = [c for c in ds.get("copies", []) if c.get("cluster") != canon]
        if rng:
            copies.append({"cluster": canon, "years": rng})
        ds["copies"] = copies
        print(f"  {dsname:14s} {rng or '(none)'}")
    if write:
        save(reg, registry_path)
        print(f"updated {registry_path}")
    else:
        print("(dry run — pass --write to update the registry)")


# --------------------------------------------------------------------------- #
# check / show
# --------------------------------------------------------------------------- #
def _dataset_status(reg: dict, ds: dict) -> dict:
    target = parse_years(ds.get("target_years"))
    cluster_years = {c["cluster"]: parse_years(c["years"]) for c in ds.get("copies", [])}
    present = set().union(*cluster_years.values()) if cluster_years else set()
    durable = any(reg["clusters"].get(c, {}).get("persistent") for c in cluster_years)
    return {
        "target": target,
        "present": present,
        "missing": target - present,
        "cluster_years": cluster_years,
        "n_copies": len(cluster_years),
        "durable": durable,
    }


def check(reg: dict) -> int:
    problems = 0
    print(f"{'dataset':16s}{'target':13s}{'present':13s}{'status'}")
    for dsname, ds in reg["datasets"].items():
        st = _dataset_status(reg, ds)
        flags = []
        if st["missing"]:
            flags.append(f"MISSING {compress_years(st['missing'])}")
        if st["n_copies"] == 0:
            flags.append("NO ZARR COPIES")
        elif not st["durable"] and st["n_copies"] < 2:
            flags.append("AT-RISK: single volatile copy (raw_source is the only fallback)")
        elif not st["durable"]:
            flags.append("volatile-only (no persistent copy)")
        at_risk = bool(st["missing"]) or st["n_copies"] == 0 or (not st["durable"] and st["n_copies"] < 2)
        problems += at_risk
        print(
            f"{dsname:16s}{ds.get('target_years',''):13s}"
            f"{compress_years(st['present']) or '-':13s}{'  '.join(flags) or 'ok'}"
        )
    print()
    if problems:
        print(f"{problems} dataset(s) need attention.")
    else:
        print("all datasets: complete and durably stored.")
    return problems


def show(reg: dict) -> None:
    print("== clusters ==")
    for name, c in reg["clusters"].items():
        if "alias_of" in c:
            print(f"  {name:11s} → alias of {c['alias_of']}")
            continue
        tags = []
        if c.get("persistent"):
            tags.append("persistent")
        if c.get("volatile"):
            tags.append("volatile")
        if c.get("role"):
            tags.append(c["role"])
        print(f"  {name:11s} {c.get('data_root','?'):45s} [{', '.join(tags)}]")
    print("\n== datasets ==")
    for dsname, ds in reg["datasets"].items():
        st = _dataset_status(reg, ds)
        locs = ", ".join(f"{c}:{compress_years(y)}" for c, y in st["cluster_years"].items()) or "(none)"
        print(f"  {dsname:14s} target {ds.get('target_years',''):12s} copies: {locs}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ai-rossby data-location registry tool.")
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH, help="registry YAML path")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="print the catalog")
    sub.add_parser("check", help="report gaps + at-risk copies (exit 1 if any)")
    sp = sub.add_parser("scan", help="scan a cluster's data_root and update its copies")
    sp.add_argument("cluster")
    sp.add_argument("--data-root", default=None, help="override the cluster's data_root")
    sp.add_argument("--write", action="store_true", help="write the updated registry")
    args = p.parse_args(argv)

    reg = load(args.registry)
    if args.cmd == "show":
        show(reg)
        return 0
    if args.cmd == "check":
        return 1 if check(reg) else 0
    if args.cmd == "scan":
        scan(reg, args.cluster, data_root=args.data_root, write=args.write,
             registry_path=args.registry)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
