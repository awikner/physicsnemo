# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12c.12 one-shot loader benchmark: coarse vs full-res daily-avg Zarr.

Times ``ClimateZarrDataset`` sample reads (the training hot path: one
``(start, lead)`` pair -> surface/upper-air/boundary/diagnostic tensors)
against the pre-coarsened 45x90 store and the full-res 180x360 store.

Run on a compute node (physicsnemo import) with both stores present::

    python benchmarks/physicsnemo/experimental/datapipes/climate/\\
        bench_amip_dailyavg_coarse.py \\
        --full   $ZARR_ROOT/amip_dailyavg/1981.zarr \\
        --coarse $ZARR_ROOT/amip_dailyavg_coarse/1981.zarr
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def bench_store(path: str, n_samples: int, seed: int = 0) -> dict:
    from physicsnemo.experimental.datapipes.climate import ClimateZarrDataset

    ds = ClimateZarrDataset(zarr_path=path)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds) - 1, size=n_samples)
    # Warm-up (open lazily, prime metadata caches).
    _ = ds[(int(idx[0]), 1)]
    t0 = time.perf_counter()
    n_bytes = 0
    for i in idx:
        sample = ds[(int(i), 1)]
        n_bytes += sum(
            v.numel() * v.element_size()
            for v in sample.values()
            if hasattr(v, "numel")
        )
    dt = time.perf_counter() - t0
    return {
        "path": path,
        "n": n_samples,
        "samples_per_s": n_samples / dt,
        "ms_per_sample": 1e3 * dt / n_samples,
        "MB_per_sample": n_bytes / n_samples / 1e6,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full", required=True)
    p.add_argument("--coarse", required=True)
    p.add_argument("--n-samples", type=int, default=100)
    args = p.parse_args()

    results = [
        ("full-res", bench_store(args.full, args.n_samples)),
        ("coarse", bench_store(args.coarse, args.n_samples)),
    ]
    print(f"{'store':<10} {'samples/s':>10} {'ms/sample':>10} {'MB/sample':>10}")
    for name, r in results:
        print(
            f"{name:<10} {r['samples_per_s']:>10.2f} "
            f"{r['ms_per_sample']:>10.1f} {r['MB_per_sample']:>10.1f}"
        )
    speedup = results[1][1]["samples_per_s"] / results[0][1]["samples_per_s"]
    print(f"coarse/full speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()