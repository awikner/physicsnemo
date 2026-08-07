#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure NCCL all-reduce bandwidth at the SFNO-E3SM gradient size.

Why this and not nccl-tests: /soft/pbs/all_reduce_perf on Polaris is a 2023
build that wants libcudart.so.11 / libnccl.so.2 from a CUDA 11 stack. Driving
the measurement from the same torch build that would run the training is both
easier and more faithful — it exercises the exact NCCL that DDP would use.

The headline number is the all-reduce of a 1,182,099,456-element fp32 buffer:
that is precisely the gradient payload DDP moves every step for SFNO-E3SM at
embed_dim=512, so `grad_allreduce_s` is a direct lower bound on the per-step
communication cost at this node count.

Launcher-agnostic: reads PMI_* under Polaris/PALS mpiexec and SLURM_* under
Delta/srun, so the identical benchmark runs on both machines. That matters —
the Delta-vs-Polaris fabric comparison was previously synthetic-vs-inferred,
which is not a fair head-to-head.

busbw follows the nccl-tests convention for ring all-reduce:
    algbw = size_bytes / time
    busbw = algbw * 2 * (n - 1) / n
busbw is the number to compare across node counts — it is what the wire sees.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.distributed as dist

# SFNO-E3SM @ embed_dim=512 (counted on Delta: 1,182,099,456 trainable params).
SFNO_E3SM_512_PARAMS = 1_182_099_456

# Sweep sizes in elements (fp32) — powers of two around the gradient size, so
# we can see where the fabric transitions from latency- to bandwidth-bound.
SWEEP_ELEMS = [
    2**20,        # 4 MiB   — near DDP's 25 MB bucket floor
    2**24,        # 64 MiB
    2**26,        # 256 MiB
    2**28,        # 1 GiB
    SFNO_E3SM_512_PARAMS,  # 4.40 GiB — the real gradient payload
]

WARMUP = 3
ITERS = 10


def main() -> int:
    # PALS (Polaris) exports PMI_*; srun (Delta) exports SLURM_*.
    if "PMI_RANK" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        world = int(os.environ["PMI_SIZE"])
        local = int(os.environ.get("PMI_LOCAL_RANK", rank % 4))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ["SLURM_NTASKS"])
        local = int(os.environ.get("SLURM_LOCALID", rank % 4))
    else:
        raise RuntimeError(
            "no launcher rank env found (need PMI_RANK/PMI_SIZE or "
            "SLURM_PROCID/SLURM_NTASKS)"
        )

    torch.cuda.set_device(local)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world,
    )

    nnodes = world // int(os.environ.get("PPN", 4))
    results = []

    for elems in SWEEP_ELEMS:
        buf = torch.ones(elems, dtype=torch.float32, device="cuda")
        nbytes = buf.element_size() * buf.numel()

        for _ in range(WARMUP):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        # Time a back-to-back run of ITERS all-reduces and divide, rather than
        # timing each one individually. A per-iteration dist.barrier() enqueues
        # its own NCCL kernel, and without a synchronize between the barrier and
        # t0 that kernel's completion lands inside the measured window — which
        # inflated small-message times badly in the first version of this probe.
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / ITERS

        del buf
        torch.cuda.empty_cache()
        algbw = nbytes / t / 1e9
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            results.append(
                {
                    "elems": elems,
                    "gib": nbytes / 2**30,
                    "median_s": t,
                    "algbw_GBps": algbw,
                    "busbw_GBps": busbw,
                    "is_grad_size": elems == SFNO_E3SM_512_PARAMS,
                }
            )
            tag = "  <-- SFNO-E3SM-512 gradient" if elems == SFNO_E3SM_512_PARAMS else ""
            print(
                f"[nodes={nnodes} world={world}] {nbytes / 2**30:8.3f} GiB  "
                f"t={t * 1e3:9.2f} ms  algbw={algbw:7.2f} GB/s  "
                f"busbw={busbw:7.2f} GB/s{tag}",
                flush=True,
            )

    if rank == 0:
        grad = next(r for r in results if r["is_grad_size"])
        print(
            f"RESULT_JSON {json.dumps({'nodes': nnodes, 'world': world, 'grad_allreduce_s': grad['median_s'], 'grad_busbw_GBps': grad['busbw_GBps'], 'sweep': results})}",
            flush=True,
        )

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
