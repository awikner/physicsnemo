#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize per-batch training speed from a `bench.per_batch_tsv` file.

The TSV's `wall_s` column is CUMULATIVE wall time since the training loop
started (written by rank 0 only), so per-batch step time is the successive
difference. The first few steps include CUDA/cuDNN autotune, NCCL bucket setup
and dataloader spin-up, so a warmup prefix is discarded before reporting.

Stdlib only — safe to run on a Delta login node (importing physicsnemo there
can core-dump on CUDA/Warp init).

Usage:
    python hpc/scripts/summarize_bench_tsv.py FILE.tsv [--world-size N] [--warmup K]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


def summarize(path: Path, world_size: int, warmup: int) -> dict | None:
    rows = [ln.split("\t") for ln in path.read_text().splitlines()[1:] if ln.strip()]
    walls = [float(r[2]) for r in rows]
    if len(walls) < 2:
        print(f"{path.name}: only {len(walls)} row(s) — no step times", file=sys.stderr)
        return None

    steps = [b - a for a, b in zip(walls, walls[1:])]
    steady = steps[warmup:] if len(steps) > warmup else steps
    med = statistics.median(steady)

    return {
        "file": path.name,
        "total_steps": len(walls),
        "timed_steps": len(steps),
        "warmup_dropped": len(steps) - len(steady),
        "median_s": med,
        "mean_s": statistics.fmean(steady),
        "stdev_s": statistics.stdev(steady) if len(steady) > 1 else 0.0,
        "min_s": min(steady),
        "max_s": max(steady),
        "first_step_s": steps[0],
        "samples_per_s": world_size / med if med > 0 else float("nan"),
        "world_size": world_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+", type=Path)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    for p in args.tsv:
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            continue
        s = summarize(p, args.world_size, args.warmup)
        if s is None:
            continue
        print(f"=== {s['file']}")
        print(f"  world_size        : {s['world_size']}  (1 sample/GPU "
              f"-> global batch {s['world_size']})")
        print(f"  steps logged      : {s['total_steps']} "
              f"({s['timed_steps']} timed, {s['warmup_dropped']} warmup dropped)")
        print(f"  first step        : {s['first_step_s']:.3f} s")
        print(f"  per-batch median  : {s['median_s']:.4f} s")
        print(f"  per-batch mean    : {s['mean_s']:.4f} +/- {s['stdev_s']:.4f} s")
        print(f"  per-batch min/max : {s['min_s']:.4f} / {s['max_s']:.4f} s")
        print(f"  throughput        : {s['samples_per_s']:.2f} samples/s")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
