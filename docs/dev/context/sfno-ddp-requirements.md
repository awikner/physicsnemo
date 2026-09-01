<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-GPU SFNO training requirements (torch pin, wandb, extras)

> Point-in-time notes (diagnosed 2026-07-07, commits `1a1b843b`, `fc0bb8fd`).
> Verify against current code before relying on file/line specifics.

Multi-GPU (DDP/NCCL) training of the ai-rossby SFNO models (`SfnoPlasim`,
`SfnoE3SM`; `examples/weather/ai_rossby/train.py`) has three non-obvious
requirements:

1. **torch must stay `< 2.11`.** torch 2.11.0 / 2.12.x regressed DDP for these
   models: `DDP.__init__` raises `value cannot be converted to type int without
   overflow` in `_verify_params_across_processes`, plus reducer deadlocks.
   `pyproject.toml` pins `torch>=2.10.0,<2.11.0` (main + cu12/cu13/cu129 extras);
   `uv.lock` is at torch 2.10.0. The old floating `>=2.10.0` pin silently drifted
   to 2.11/2.12 on `uv sync` and broke the working 4×A100 SFNO-PlaSim benchmark.

2. **wandb must be initialized on EVERY rank — and every recipe must actually
   do it.** wandb's background threads (service IPC / console capture / GPU
   monitor) grab the GIL inside CUDA calls; running wandb on rank 0 alone makes
   that jitter asymmetric and stalls that rank's NCCL progress → deadlock (at
   DDP *init* or mid-epoch). Initializing wandb on ALL ranks makes the jitter
   symmetric (this is what the PanguWeather reference trainer does).
   `_maybe_init_wandb` calls `initialize_wandb` on all ranks and returns
   `rank==0`.

   > **2026-08-07 (Phase 12b / wandb_ddp_hang_fix_plan.md):** the hang
   > resurfaced because `train_diffusion.py` called `_maybe_init_wandb` **on
   > rank 0 only** — the exact asymmetric configuration this requirement
   > forbids (`train.py` was always compliant, hence the July validation).
   > With the call-site fixed, wandb-on DDP passed 3/3 short runs + a 93-min
   > full-epoch run (Delta 2×A40, jobs 20921753 / 20921857). A safety guard
   > remains: `wandb.allow_multigpu: false` auto-disables wandb under DDP
   > (console + bench TSV only) if the hang ever resurfaces; default is
   > `true`. **Any NEW recipe must call `_maybe_init_wandb` unconditionally
   > on every rank, never inside an `if rank == 0:` block.** Full history:
   > [`docs/dev/wandb_ddp_hang_fix_plan.md`](../wandb_ddp_hang_fix_plan.md).

3. **The recipe's optional extras must be installed or `uv sync` prunes them.**
   `uv sync --extra cu12 --group dev` alone REMOVES torch-harmonics/tensorly
   (SFNO, `sfno-extras`), wandb/mlflow (`utils-extras`), and
   zarr/xarray/netCDF4/dask (`datapipes-extras`) — silently breaking SFNO after
   any sync. `hpc/scripts/sync-all-clusters.sh` adds
   `--extra sfno-extras --extra utils-extras --extra datapipes-extras`.
   NOTE: DeltaAI (aarch64/GH200) has no torch-harmonics wheel. The `--no-binary`
   recipe that used to be documented here **no longer works**: as of 0.9.x,
   torch-harmonics publishes x86_64-only wheels and **no sdist at all** (verified
   against PyPI 2026-09-01 for 0.9.0/0.9.1/0.9.2; the last version with an sdist
   is 0.8.1, which is below the `>=0.9.0` pin in `sfno-extras`). On aarch64 there
   is therefore nothing on PyPI to install *or* to build from. Install from the
   upstream git tag instead:
   `uv pip install --no-deps --no-build-isolation
    "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@v0.9.1"`
   — `--no-deps` avoids pulling a PyPI torch over DeltaAI's module torch, which
   is what the old `uv pip uninstall torch torchvision triton` step was cleaning
   up after. The container image does exactly this on arm64
   (`hpc/containers/Dockerfile`).

**Diagnosis tip:** on a hang, enable the NCCL flight recorder
(`TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_FR_BUFFER_SIZE`, short
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`) and `pickle.load` the per-rank dump — it shows
each rank's last collective (op/size/state) and stack.

See also: [delta-gpu-partitions](delta-gpu-partitions.md).
