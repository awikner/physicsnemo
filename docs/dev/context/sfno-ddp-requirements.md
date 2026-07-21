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

2. **wandb must be initialized on EVERY rank (log from rank 0 only).** wandb's
   background threads (service IPC / console capture / GPU monitor) grab the GIL
   inside CUDA calls; running wandb on rank 0 alone makes that jitter asymmetric
   and desyncs DDP's NCCL collectives → deadlock. Initializing wandb on ALL ranks
   makes the jitter symmetric → DDP stays in lockstep (this is what the
   PanguWeather reference trainer does). `_maybe_init_wandb` calls
   `initialize_wandb` on all ranks and returns `rank==0`. (An earlier commit that
   auto-DISABLED wandb under DDP is superseded; wandb works multi-GPU now.)

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
