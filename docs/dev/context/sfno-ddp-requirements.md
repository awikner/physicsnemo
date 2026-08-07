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

2. **wandb is AUTO-DISABLED under DDP (`world_size > 1`).**
   > **UPDATE 2026-08-07 (Phase 12b):** the every-rank strategy below is NOT
   > sufficient. wandb's background threads hung `DDP.__init__`'s *first* NCCL
   > collective (watchdog "another thread holding the GIL inside a CUDA api" →
   > SIGABRT; peer rank reports a spurious `int overflow`) on Delta 2×A40 with
   > wandb initialized on every rank — reproducibly, and resolved by disabling
   > wandb (jobs 20918380 vs 20920825). The `1a1b843b` auto-disable guard is
   > restored in `_maybe_init_wandb` with a `wandb.allow_multigpu` escape hatch.
   > Multi-GPU runs log to console + bench TSV. Diagnosis + re-enable plan:
   > [`docs/dev/wandb_ddp_hang_fix_plan.md`](../wandb_ddp_hang_fix_plan.md).

   *Historical rationale for the (insufficient) every-rank strategy:* wandb's
   background threads (service IPC / console capture / GPU monitor) grab the GIL
   inside CUDA calls; running wandb on rank 0 alone makes that jitter asymmetric
   and desyncs DDP's NCCL collectives mid-epoch → deadlock. Initializing wandb on
   ALL ranks makes the jitter symmetric (this is what the PanguWeather reference
   trainer does) — but an *init-time* collective blocks on whichever rank's
   threads stall it, symmetric or not. The every-rank machinery still applies
   when `allow_multigpu=true` overrides the guard.

3. **The recipe's optional extras must be installed or `uv sync` prunes them.**
   `uv sync --extra cu12 --group dev` alone REMOVES torch-harmonics/tensorly
   (SFNO, `sfno-extras`), wandb/mlflow (`utils-extras`), and
   zarr/xarray/netCDF4/dask (`datapipes-extras`) — silently breaking SFNO after
   any sync. `hpc/scripts/sync-all-clusters.sh` adds
   `--extra sfno-extras --extra utils-extras --extra datapipes-extras`.
   NOTE: DeltaAI (aarch64/GH200) has no torch-harmonics wheel — build from source
   (`uv pip install --no-binary torch-harmonics torch-harmonics`), which pulls
   torch as a dep, so `uv pip uninstall torch torchvision triton` afterward to
   fall back to DeltaAI's module torch.

**Diagnosis tip:** on a hang, enable the NCCL flight recorder
(`TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_FR_BUFFER_SIZE`, short
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`) and `pickle.load` the per-rank dump — it shows
each rank's last collective (op/size/state) and stack.

See also: [delta-gpu-partitions](delta-gpu-partitions.md).
