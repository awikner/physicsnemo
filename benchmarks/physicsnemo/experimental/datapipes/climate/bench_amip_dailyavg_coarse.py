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


def _build_dataset(path: str, windowed: int):
    from physicsnemo.experimental.datapipes.climate import ClimateZarrDataset

    ds = ClimateZarrDataset(zarr_path=path)
    if windowed > 1:
        from physicsnemo.experimental.datapipes.climate import SequenceDataset

        # W frames = unroll_steps + 1 — the ERDM rolling-window read the
        # coarse store's --time-chunk 8 was sized for.
        return SequenceDataset(ds, unroll_steps=windowed - 1)
    return ds


def bench_store(
    path: str,
    n_samples: int,
    *,
    seed: int = 0,
    windowed: int = 0,
    num_workers: int = 0,
) -> dict:
    ds = _build_dataset(path, windowed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds) - 1, size=n_samples)

    def _bytes(sample) -> int:
        return sum(
            v.numel() * v.element_size()
            for v in sample.values()
            if hasattr(v, "numel")
        )

    if num_workers > 0:
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            ds,
            batch_size=1,
            sampler=[int(i) if windowed > 1 else (int(i), 1) for i in idx],
            num_workers=num_workers,
            prefetch_factor=2,
        )
        it = iter(loader)
        _ = next(it)  # warm-up: worker spawn + first read
        t0 = time.perf_counter()
        n_bytes = 0
        n_read = 0
        for sample in it:
            n_bytes += _bytes(sample)
            n_read += 1
        dt = time.perf_counter() - t0
    else:
        key0 = int(idx[0]) if windowed > 1 else (int(idx[0]), 1)
        _ = ds[key0]  # warm-up
        t0 = time.perf_counter()
        n_bytes = 0
        n_read = 0
        for i in idx:
            sample = ds[int(i)] if windowed > 1 else ds[(int(i), 1)]
            n_bytes += _bytes(sample)
            n_read += 1
        dt = time.perf_counter() - t0
    return {
        "path": path,
        "n": n_read,
        "samples_per_s": n_read / dt,
        "ms_per_sample": 1e3 * dt / n_read,
        "MB_per_sample": n_bytes / max(1, n_read) / 1e6,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full", default=None, help="Full-res store (optional).")
    p.add_argument("--coarse", required=True)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument(
        "--windowed",
        type=int,
        default=0,
        help="W>1 benches SequenceDataset W-frame window reads (the ERDM "
        "training pattern) instead of single (start, lead=1) samples.",
    )
    p.add_argument(
        "--workers-sweep",
        type=str,
        default="0",
        help="Comma-separated DataLoader num_workers values, e.g. 0,2,4,8.",
    )
    args = p.parse_args()

    workers = [int(w) for w in args.workers_sweep.split(",")]
    targets = ([("full-res", args.full)] if args.full else []) + [
        ("coarse", args.coarse)
    ]
    print(
        f"{'store':<10} {'workers':>7} {'windowed':>8} {'samples/s':>10} "
        f"{'ms/sample':>10} {'MB/sample':>10}"
    )
    rates: dict[str, float] = {}
    for name, path in targets:
        for nw in workers:
            r = bench_store(
                path, args.n_samples, windowed=args.windowed, num_workers=nw
            )
            rates[f"{name}@{nw}"] = r["samples_per_s"]
            print(
                f"{name:<10} {nw:>7} {args.windowed:>8} "
                f"{r['samples_per_s']:>10.2f} {r['ms_per_sample']:>10.1f} "
                f"{r['MB_per_sample']:>10.1f}"
            )
    if args.full and f"coarse@{workers[0]}" in rates:
        speedup = rates[f"coarse@{workers[0]}"] / rates[f"full-res@{workers[0]}"]
        print(f"coarse/full speedup (workers={workers[0]}): {speedup:.1f}x")


if __name__ == "__main__":
    main()