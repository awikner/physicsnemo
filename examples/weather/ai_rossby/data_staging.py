# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast node-local staging of Zarr stores for training.

Copies the training / validation Zarr stores (and the normalization-stats
store) from a shared parallel filesystem (Delta's Lustre ``/work/hdd``) to
node-local disk (``/tmp``) *once* at the start of a run, then rewrites the
dataset config to read from the local copies. Reading chunks from fast local
NVMe instead of contending on the shared filesystem across every rank and
every concurrent job removes a real I/O tax on data-bound configurations.

Why this is fast where a plain ``cp -r`` was not
-------------------------------------------------
A Zarr store is a directory tree of thousands of independent chunk files
(one E3SM year ≈ 17.5k files / 30 GB). A serial ``cp -r`` pays a full
open/stat/read/close round-trip to the Lustre metadata + object servers for
each file, one at a time — ~17.5k *serialized* latencies, which is what
overran the earlier 25-minute staging budget. Lustre is a *parallel*
filesystem: issuing many copies concurrently hides that per-file latency.

What the bottleneck actually is (measured on Delta, E3SM year → /tmp)
--------------------------------------------------------------------
The read+write copy is **write-bound at ~2.2 GB/s** to node-local ``/tmp``,
independent of the copy method (``shutil.copyfile`` and a raw
``open``+``sendfile`` loop tie) and independent of worker count from 64 to
512 — so there is nothing to gain from a fancier copy primitive. The Lustre
**read** side, by contrast, scales hard with concurrency (~1.6 GB/s at 128
concurrent readers → ~12 GB/s at 512). Concurrency's real job is therefore to
hide cold-read latency; a low worker count leaves the copy read-latency-bound
on a cold cache (64 workers measured ~0.25–0.5 GB/s cold), which is exactly
the fresh-job case that matters. The default (256) stages a full cold year
(30 GB / 17.5k files) in ~47 s (0.64 GB/s), a ~10× speedup over a serial
``cp``; raise it to 512 for even more cold-read concurrency.

Why the standard library is the right tool (no third-party dependency)
----------------------------------------------------------------------
The per-file cost here is I/O latency, not CPU. ``shutil.copyfile`` uses
``os.sendfile`` on Linux (kernel-side zero-copy — no user-space buffer), and
the GIL is released for the duration of that syscall, so a thread pool scales
with concurrency until the write pipe saturates. Heavier machinery
(``mpifileutils`` ``dcp`` — not even installed on Delta — or ``fpsync``) would
add MPI-launch / process-management complexity for no throughput gain on this
write-bound workload, so they are deliberately not used.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

# Keys in ``cfg.dataset`` that name an on-disk Zarr store (or stats store) and
# therefore benefit from node-local staging. Order is stable so logs are
# deterministic. ``mean_path``/``std_path`` frequently point at the *same*
# normalization store — de-duplication by resolved source path (below) copies
# it only once.
_STAGEABLE_KEYS = (
    "zarr_path",
    "val_zarr_path",
    "mean_path",
    "std_path",
    "boundary_zarr_path",
    "leap_boundary_zarr_path",
    "non_leap_boundary_zarr_path",
    "delta_std_path",
)

_MARKER_SUFFIX = ".stage_complete"


def resolve_stage_root(stage_dir: Optional[str]) -> Path:
    """Pick the node-local staging root directory.

    Precedence: explicit ``stage_dir`` config → ``$SLURM_TMPDIR`` →
    ``$TMPDIR`` → ``/tmp``. A per-job subdirectory (``$SLURM_JOB_ID`` when
    present) keeps concurrent jobs on a shared node from colliding while
    staying identical across all ranks *of the same job* on that node (so
    every rank derives the same local path deterministically).
    """
    if stage_dir:
        base = Path(stage_dir)
    else:
        base = Path(
            os.environ.get("SLURM_TMPDIR")
            or os.environ.get("TMPDIR")
            or "/tmp"
        )
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
    root = base / "ai_rossby_stage"
    if job_id:
        root = root / job_id
    return root


def parallel_copy_tree(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    num_workers: int,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Copy a directory tree from ``src`` to ``dst`` with a thread pool.

    Returns ``(n_files, n_bytes)``. Directories are created up front (cheap,
    single-threaded); regular files are copied concurrently via
    ``shutil.copyfile`` (kernel ``sendfile`` zero-copy on Linux). Raises the
    first copy error encountered — a partial stage is never silently accepted.
    """
    src = Path(src)
    dst = Path(dst)

    # Single-threaded walk: enumerate files and pre-create the directory
    # skeleton so the concurrent copies never race on mkdir.
    pairs: list[tuple[str, str]] = []
    for root, _dirs, filenames in os.walk(src):
        rel = os.path.relpath(root, src)
        dst_root = dst if rel == os.curdir else dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            pairs.append((os.path.join(root, fn), str(dst_root / fn)))

    total = len(pairs)
    if total == 0:
        return 0, 0

    counter_lock = threading.Lock()
    state = {"done": 0, "bytes": 0}
    # Log at ~20% granularity without spamming; guard the shared counter.
    milestone = max(1, total // 5)

    def _copy(pair: tuple[str, str]) -> None:
        s, d = pair
        shutil.copyfile(s, d)  # os.sendfile fast-path on Linux
        sz = os.stat(d).st_size
        with counter_lock:
            state["done"] += 1
            state["bytes"] += sz
            done = state["done"]
        if log is not None and (done % milestone == 0 or done == total):
            pct = 100 * done / total
            log(f"    … {done}/{total} files ({pct:.0f}%)")

    # ThreadPoolExecutor.map raises the first worker exception on iteration,
    # so a failed copy aborts staging rather than yielding a partial store.
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for _ in pool.map(_copy, pairs):
            pass

    return total, state["bytes"]


def stage_store(
    src: str,
    stage_root: Path,
    *,
    num_workers: int,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Stage a single Zarr store (directory) into ``stage_root``.

    Idempotent: a successful copy drops a sibling ``<name>.stage_complete``
    marker; a later call (e.g. a requeued job landing on the same node) sees
    the marker and returns the local path without re-copying. A destination
    left over from an *incomplete* prior copy (marker absent) is removed and
    re-staged from scratch, so a killed job never leaves a half-copied store
    that reads as valid.
    """
    src_path = Path(src)
    dst = stage_root / src_path.name
    marker = stage_root / (src_path.name + _MARKER_SUFFIX)

    if marker.exists() and dst.exists():
        if log is not None:
            log(f"  [skip] {src_path.name} already staged at {dst}")
        return str(dst)

    # Clean any partial leftover before a fresh copy.
    if dst.exists():
        shutil.rmtree(dst)
    if marker.exists():
        marker.unlink()

    stage_root.mkdir(parents=True, exist_ok=True)
    if log is not None:
        log(f"  [copy] {src} -> {dst}")
    t0 = time.perf_counter()
    n_files, n_bytes = parallel_copy_tree(
        src_path, dst, num_workers=num_workers, log=log
    )
    dt = time.perf_counter() - t0
    gb = n_bytes / 1e9
    rate = gb / dt if dt > 0 else float("inf")
    if log is not None:
        log(
            f"  [done] {src_path.name}: {n_files} files, {gb:.1f} GB in "
            f"{dt:.1f}s ({rate:.2f} GB/s)"
        )

    marker.touch()
    return str(dst)
