#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async vs sync forecast-writer benchmark for the ai_rossby inference path.

Loads a translated PanguPlasimLegacy ``.mdlus`` checkpoint, builds the
1981 ERA5 S2S Zarr dataset, and runs ``run_inference_streaming_per_ic``
twice from the same warmed-up model + dataset:

  1. **async** mode — ``AsyncForecastWriter(max_in_flight=4, num_workers=2)``
     (the in-prod default).
  2. **sync** mode — same writer wrapped to call ``wait_all()`` after
     every ``submit``; the next IC's GPU rollout only starts after the
     previous IC's file is fully flushed to disk.

The rollout payload itself, the dataset reads, the model forward and
the disk target are bit-identical between the two runs — the only
difference is whether the writer drains synchronously. So the wall-time
delta is the headroom the async writer is hiding.

Reports per-IC and total wall-times for each mode plus the achieved
speedup, and emits a TSV (``timings.tsv``) for downstream comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT / "examples/weather/ai_rossby"))

warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")

from physicsnemo import Module  # noqa: E402
from physicsnemo.experimental.datapipes.plasim import (  # noqa: E402
    NanFillTransform,
    PlasimClimateDataset,
    PlasimNormalizer,
)

from async_writer import AsyncForecastWriter  # noqa: E402
from inference import (  # noqa: E402  pyright: ignore[reportMissingImports]
    Deterministic,
    run_inference_streaming_per_ic,
)


class _BlockingWriter:
    """Drop-in stand-in for :class:`AsyncForecastWriter` that waits after each submit.

    Same interface (``submit`` / ``wait_all`` / ``in_flight`` / context
    manager) so the inference helper doesn't need to know it's running
    synchronously. Implemented on top of the real writer with
    ``max_in_flight=1`` plus a wait_all() after every submit, which
    guarantees no GPU work overlaps disk I/O.
    """

    def __init__(self):
        self._inner = AsyncForecastWriter(max_in_flight=1, num_workers=1)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._inner.__exit__(exc_type, exc_val, exc_tb)

    def submit(self, path, dataset, *, mode="w"):
        fut = self._inner.submit(path, dataset, mode=mode)
        self._inner.wait_all()  # force-drain before returning
        return fut

    def wait_all(self):
        self._inner.wait_all()

    @property
    def in_flight(self) -> int:
        return self._inner.in_flight


@contextmanager
def _time_block(label, log_dict):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log_dict[label] = dt
    print(f"  [{label}] elapsed = {dt:.2f}s")


def build_model(mdlus_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"Loading model from {mdlus_path}")
    model = Module.from_checkpoint(str(mdlus_path)).to(device).eval()
    print(f"  -> {type(model).__name__} on {device}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def build_dataset(
    zarr_path: Path,
    mean_path: Path,
    std_path: Path,
    constant_boundary_vars,
    varying_boundary_vars,
    device: torch.device,
) -> tuple[PlasimClimateDataset, PlasimNormalizer]:
    print(f"Opening dataset {zarr_path}")
    ds = PlasimClimateDataset(str(zarr_path))
    print(f"  -> n_time={ds.n_time}, n_lat={ds.n_lat}, n_lon={ds.n_lon}")
    norm = PlasimNormalizer.from_dataset(
        ds,
        mean_path=str(mean_path),
        std_path=str(std_path),
        normalize_constant_boundary=False,
        normalize_diagnostic=False,
    ).to(device)
    nan_fill = NanFillTransform(
        constant_boundary_variables=list(constant_boundary_vars),
        varying_boundary_variables=list(varying_boundary_vars),
        fill_values={},
        default=0.0,
    )
    ds.transform = nan_fill
    return ds, norm


def run_one(mode: str, writer_ctx, dataset, model, normalizer, *, ic_indices,
            max_step, output_dir, device, logger=None) -> dict:
    print(f"\n=== {mode.upper()} run ===")
    timings: dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop any leftover files from a prior run.
    for p in output_dir.glob("*.zarr"):
        import shutil
        shutil.rmtree(p, ignore_errors=True)
    for p in output_dir.glob("*.nc"):
        p.unlink(missing_ok=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with _time_block(f"{mode}_total", timings), writer_ctx as writer:
        run_inference_streaming_per_ic(
            model,
            dataset,
            normalizer=normalizer,
            device=device,
            ic_indices=ic_indices,
            max_step=max_step,
            writer=writer,
            output_dir=str(output_dir),
            model_name="pangu_plasim_s2s_2000",
            run_name=mode,
            output_format="zarr",
            ensemble_size=1,
            perturber=Deterministic(),
            has_diagnostic=getattr(model, "has_diagnostic", False),
            seed=0,
            logger=logger,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Disk-side bookkeeping.
    total_bytes = 0
    n_files = 0
    for p in output_dir.glob("*.zarr"):
        for root, _, files in os.walk(p):
            for f in files:
                total_bytes += os.path.getsize(os.path.join(root, f))
        n_files += 1
    timings[f"{mode}_files_written"] = n_files
    timings[f"{mode}_total_bytes"] = total_bytes
    print(f"  wrote {n_files} files, {total_bytes/1e6:.1f} MB total")
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlus", required=True, type=Path)
    parser.add_argument("--model-yaml", required=True, type=Path,
                        help="ai-rossby model YAML (for surface/boundary variable names).")
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--mean", required=True, type=Path)
    parser.add_argument("--std", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-ic", type=int, default=4)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--ic-stride", type=int, default=80,
                        help="Stride between consecutive ICs in the dataset's time index.")
    parser.add_argument("--ic-start", type=int, default=0)
    parser.add_argument("--warmup-ics", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # 1. Model.
    model = build_model(args.mdlus, device)

    # Pull variable lists from the YAML — the model's checkpoint stores
    # only counts (num_surface_vars etc.), not the names NanFillTransform
    # needs.
    with open(args.model_yaml) as fh:
        model_cfg = yaml.safe_load(fh)
    cbvs = list(model_cfg["constant_boundary_variables"])
    vbvs = list(model_cfg["varying_boundary_variables"])

    # 2. Dataset + normalizer.
    dataset, normalizer = build_dataset(
        args.zarr, args.mean, args.std,
        constant_boundary_vars=cbvs,
        varying_boundary_vars=vbvs,
        device=device,
    )

    # 3. Pick IC indices.
    n_time = dataset.n_time
    ic_indices = []
    for i in range(args.n_ic):
        ic = args.ic_start + i * args.ic_stride
        if ic + args.max_step >= n_time:
            print(f"  warning: IC {ic} + max_step exceeds n_time={n_time}, truncating ICs")
            break
        ic_indices.append(ic)
    if not ic_indices:
        raise RuntimeError("no valid ICs in dataset for the requested span")
    print(f"ICs (n={len(ic_indices)}): {ic_indices}; max_step={args.max_step}")

    # 4. Warmup pass — exercises CUDA kernels and Zarr index caches once.
    if args.warmup_ics > 0:
        print(f"\n--- WARMUP ({args.warmup_ics} ICs) ---")
        warmup_dir = args.output_dir / "warmup"
        run_one(
            "warmup",
            AsyncForecastWriter(max_in_flight=4, num_workers=2),
            dataset, model, normalizer,
            ic_indices=ic_indices[:args.warmup_ics],
            max_step=args.max_step,
            output_dir=warmup_dir,
            device=device,
        )

    timings: dict[str, float] = {}

    # 5. Sync run.
    sync_dir = args.output_dir / "sync"
    sync_timings = run_one(
        "sync",
        _BlockingWriter(),
        dataset, model, normalizer,
        ic_indices=ic_indices,
        max_step=args.max_step,
        output_dir=sync_dir,
        device=device,
    )
    timings.update(sync_timings)

    # 6. Async run.
    async_dir = args.output_dir / "async"
    async_timings = run_one(
        "async",
        AsyncForecastWriter(max_in_flight=4, num_workers=2),
        dataset, model, normalizer,
        ic_indices=ic_indices,
        max_step=args.max_step,
        output_dir=async_dir,
        device=device,
    )
    timings.update(async_timings)

    # 7. Report.
    sync_t = timings["sync_total"]
    async_t = timings["async_total"]
    saved = sync_t - async_t
    speedup = sync_t / async_t if async_t > 0 else float("nan")

    print("\n" + "=" * 60)
    print("== ASYNC WRITER BENCHMARK")
    print("=" * 60)
    print(f"  ICs              = {len(ic_indices)}")
    print(f"  max_step         = {args.max_step}")
    print(f"  files            = {async_timings['async_files_written']}")
    print(f"  bytes per file   = {async_timings['async_total_bytes']/len(ic_indices)/1e6:.1f} MB")
    print(f"  sync  wall time  = {sync_t:.2f}s ({sync_t/len(ic_indices):.2f}s / IC)")
    print(f"  async wall time  = {async_t:.2f}s ({async_t/len(ic_indices):.2f}s / IC)")
    print(f"  saved by async   = {saved:.2f}s ({100*saved/sync_t:.1f}%)")
    print(f"  speedup          = {speedup:.2f}×")

    report = {
        "ic_indices": list(ic_indices),
        "max_step": args.max_step,
        "device": str(device),
        "model": str(args.mdlus),
        "zarr": str(args.zarr),
        "sync_total_s": sync_t,
        "async_total_s": async_t,
        "saved_s": saved,
        "speedup": speedup,
        "files_per_run": async_timings["async_files_written"],
        "bytes_per_file": async_timings["async_total_bytes"] / max(len(ic_indices), 1),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote report → {args.report}")


if __name__ == "__main__":
    main()
