#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer ai-rossby datasets between clusters via Globus, driven by the registry.

    sync_dataset.py <dataset> --to <cluster> [--years A-B] [--dry-run]
        Ensure <dataset> (or the given year range) is on <cluster>; if not, pull
        the missing year-stores from a peer that has them (Globus).

    sync_dataset.py --rehydrate <cluster> [--dry-run]
        Restore everything the registry says should live on <cluster> but that a
        scan shows missing (e.g. after a scratch purge), pulling from peers.

Reads ``hpc/data_registry.yaml`` (see ``registry.py``). Emits a Globus CLI
``transfer --batch`` plan; runs it when the ``globus`` CLI is available and
``--dry-run`` is not set. After a real transfer, run
``registry.py scan <cluster> --write`` to record the new copy.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R  # noqa: E402  (load, parse_years, compress_years, resolve_cluster)


def _copies_on(ds: dict, canon: str) -> set[int]:
    out: set[int] = set()
    for c in ds.get("copies", []):
        if c["cluster"] == canon:
            out |= R.parse_years(c["years"])
    return out


def plan_transfer(reg: dict, dataset: str, to_cluster: str, years: str | None) -> dict:
    """Compute which year-stores to pull and from where."""
    if dataset not in reg["datasets"]:
        raise SystemExit(f"unknown dataset {dataset!r} (known: {', '.join(reg['datasets'])})")
    ds = reg["datasets"][dataset]
    tgt_canon, _tgt = R.resolve_cluster(reg, to_cluster)
    want = R.parse_years(years) if years else R.parse_years(ds.get("target_years"))
    needed = want - _copies_on(ds, tgt_canon)

    # candidate sources: other clusters holding some of the needed years
    cands = []
    for c in ds.get("copies", []):
        if c["cluster"] == tgt_canon:
            continue
        cov = needed & R.parse_years(c["years"])
        if cov:
            cands.append((c["cluster"], cov))
    # prefer persistent sources, then largest coverage
    cands.sort(key=lambda s: (not reg["clusters"].get(s[0], {}).get("persistent"), -len(s[1])))

    assign: dict[str, set[int]] = {}
    remaining = set(needed)
    for cl, cov in cands:
        take = remaining & cov
        if take:
            assign[cl] = take
            remaining -= take
        if not remaining:
            break

    return {
        "dataset": dataset,
        "subdir": ds["subdir"],
        "target": tgt_canon,
        "want": want,
        "needed": needed,
        "assign": assign,             # source_cluster -> set(years)
        "unavailable": remaining,     # needed but on no peer (convert from raw_source)
        "raw_source": ds.get("raw_source"),
    }


def _collection(reg: dict, cluster: str) -> str:
    canon, info = R.resolve_cluster(reg, cluster)
    return info.get("globus_collection", f"TODO-{canon}-UUID")


def _root(reg: dict, cluster: str) -> str:
    _canon, info = R.resolve_cluster(reg, cluster)
    return info["data_root"]


def render_and_maybe_run(reg: dict, plan: dict, dry_run: bool) -> None:
    subdir = plan["subdir"]
    tgt = plan["target"]
    if not plan["needed"]:
        print(f"{plan['dataset']}: already on {tgt} ({R.compress_years(plan['want'])}) — nothing to do.")
        return
    print(f"{plan['dataset']} → {tgt}: need {R.compress_years(plan['needed'])}")
    if plan["unavailable"]:
        rs = plan["raw_source"]
        print(f"  ⚠ {R.compress_years(plan['unavailable'])} not on any cluster — "
              f"convert first from raw_source ({rs.get('cluster')}:{rs.get('path')})"
              if rs else f"  ⚠ {R.compress_years(plan['unavailable'])} unavailable and no raw_source recorded.")

    globus = shutil.which("globus")
    for src, yrs in plan["assign"].items():
        src_col, dst_col = _collection(reg, src), _collection(reg, tgt)
        src_root, dst_root = _root(reg, src), _root(reg, tgt)
        batch = "\n".join(f"{subdir}/{y}.zarr {subdir}/{y}.zarr --recursive"
                          for y in sorted(yrs))
        label = f"ai-rossby {plan['dataset']} {src}->{tgt}"
        cmd = ["globus", "transfer", f"{src_col}:{src_root}", f"{dst_col}:{dst_root}",
               "--label", label, "--sync-level", "checksum", "--batch", "-"]
        print(f"\n  # from {src} ({R.compress_years(yrs)}):")
        print("  " + " ".join(cmd))
        lines = batch.splitlines()
        preview = lines[:3] + [f"... ({len(lines)} stores total)"] if len(lines) > 6 else lines
        print("  # batch (stdin):")
        for line in preview:
            print(f"  #   {line}")
        if "TODO-" in src_col or "TODO-" in dst_col:
            print("  ⚠ fill the globus_collection UUIDs in the registry before running.")
            continue
        if dry_run:
            continue
        if not globus:
            print("  ⚠ `globus` CLI not found on PATH — skipping actual transfer.")
            continue
        print("  running globus transfer …")
        subprocess.run(cmd, input=batch, text=True, check=True)
    if not dry_run:
        print(f"\nAfter transfers complete, run: registry.py scan {tgt} --write")


def rehydrate(reg: dict, cluster: str, dry_run: bool) -> None:
    canon, _ = R.resolve_cluster(reg, cluster)
    print(f"rehydrating {canon} …")
    any_work = False
    for dsname, ds in reg["datasets"].items():
        have = _copies_on(ds, canon)
        want = R.parse_years(ds.get("target_years"))
        # only rehydrate datasets that are *supposed* to be on this cluster
        listed = any(c["cluster"] == canon for c in ds.get("copies", []))
        if not listed:
            continue
        if want - have:
            any_work = True
            plan = plan_transfer(reg, dsname, canon, R.compress_years(want - have))
            render_and_maybe_run(reg, plan, dry_run)
    if not any_work:
        print("  nothing missing — all listed datasets are complete on this cluster.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Globus dataset sync driven by the data registry.")
    p.add_argument("dataset", nargs="?", help="dataset name (omit with --rehydrate)")
    p.add_argument("--to", dest="to_cluster", help="target cluster")
    p.add_argument("--years", default=None, help='year range, e.g. "2015-2049"')
    p.add_argument("--rehydrate", metavar="CLUSTER", help="restore missing datasets on CLUSTER")
    p.add_argument("--registry", type=Path, default=R.REGISTRY_PATH)
    p.add_argument("--dry-run", action="store_true", help="print the plan, do not transfer")
    args = p.parse_args(argv)

    reg = R.load(args.registry)
    if args.rehydrate:
        rehydrate(reg, args.rehydrate, args.dry_run)
        return 0
    if not args.dataset or not args.to_cluster:
        p.error("give <dataset> and --to <cluster>, or use --rehydrate <cluster>")
    plan = plan_transfer(reg, args.dataset, args.to_cluster, args.years)
    render_and_maybe_run(reg, plan, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
