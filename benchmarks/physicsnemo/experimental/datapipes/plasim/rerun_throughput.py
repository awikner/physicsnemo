#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot loader throughput re-bench used to validate Phase G (cftime everywhere).

Runs the same warm-cache (workers, batch_size) sweep as the ASV benchmark
:mod:`benchmarks.physicsnemo.experimental.datapipes.plasim.loader_throughput`
and prints a markdown table that can be diffed against RESULTS.md.

The full ASV setup is over-engineered for a single-question check; this script
is the headline-number re-run only. The ASV benchmark stays as the system of
record.

Usage::

    python rerun_throughput.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

# Suppress experimental-API warnings — we know.
warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")

# Repo-relative import: the script lives alongside the ASV benchmark, both at
# benchmarks/physicsnemo/experimental/datapipes/plasim/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader_throughput import (  # noqa: E402
    PanguH5Dataset,
    _build_zarr_dataset,
    _h5_dir,
    _iterate_one_epoch,
    _make_loader,
    _zarr_path,
)


CONFIGS = [(0, 1), (0, 4), (2, 1), (2, 4), (4, 1), (4, 4)]


def bench_one(backend: str, num_workers: int, batch_size: int) -> float:
    if backend == "pangu_h5":
        dataset = PanguH5Dataset(_h5_dir())
    elif backend == "zarr":
        dataset = _build_zarr_dataset(_zarr_path(), n_samples=120)
    else:
        raise ValueError(backend)
    loader = _make_loader(dataset, batch_size, num_workers)
    # Warmup epoch for filesystem cache.
    _iterate_one_epoch(loader)
    t0 = time.perf_counter()
    n_samples = 0
    for batch in loader:
        # First-key gives the per-sample item; the batch dim leads.
        first_value = next(iter(batch.values()))
        n_samples += first_value.shape[0]
    dt = time.perf_counter() - t0
    # Persistent workers need explicit shutdown to avoid resource leak.
    del loader
    return n_samples / dt


def main() -> int:
    print("| workers | bs | PanguH5 samples/s | Zarr samples/s | Zarr / PanguH5 |")
    print("|---|---|---|---|---|")
    for w, b in CONFIGS:
        h5_rate = bench_one("pangu_h5", w, b)
        zarr_rate = bench_one("zarr", w, b)
        ratio = zarr_rate / h5_rate if h5_rate else float("inf")
        print(f"| {w} | {b} | {h5_rate:.1f} | {zarr_rate:.1f} | **{ratio:.2f}×** |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
