#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optimized SFNO-E3SM training config — the winner of a training-speed
# optimization pass targeting global batch 16 on a single Delta
# gpuA100x4-interactive node (4x A100 40GB).
#
# Winning config vs. the naive fp32 baseline (1.067 s/step, 15.0 samples/sec):
#   bf16 AMP + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   + model.checkpointing=2 + fused AdamW  ->  0.766 s/step, 20.9 samples/sec
#   (~1.39x vs fp32 baseline; ~1.29x vs the earlier bf16+checkpointing=3 best).
#
# Key finding (makani-derived): activation checkpointing is the dominant speed
# tax here — at E3SM's 180x360 grid (~8x more grid points than the PLASIM_5412
# benchmark this architecture was designed for), global batch 16 needs
# checkpointing to fit a 40GB A100. NVIDIA makani's documented allocator fix
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reclaims ~490 MB of
# reserved-but-unallocated fragmentation, which is exactly enough to drop from
# checkpointing=3 (recompute the WHOLE block in backward) to checkpointing=2
# (recompute only the MLP) — a 29% step-time win at identical numerics.
# checkpointing<2 (0 or 1) still OOMs even with expandable_segments; do not
# lower it without re-verifying memory headroom.
#
# Other levers tried and NOT adopted:
#   - CUDA graphs (training.use_static_capture=True): physicsnemo detects
#     SfnoPlasim doesn't support graph capture and auto-disables it (no-op).
#   - NVMe data staging: a naive `cp -r` of the chunked Zarr store (~17.5k
#     chunk files / 30 GB per year) didn't finish within a 25-min budget. This
#     is now a fast, parallel option — `dataset.stage_to_local=True` copies
#     every store to node-local /tmp with a 256-way sendfile thread pool
#     (write-bound at ~2.2 GB/s to /tmp; see data_staging.py). It stays OFF
#     here because nsys profiling showed this model is compute-bound, not
#     I/O-bound, so local staging buys nothing for THIS config. Enable it for
#     data-bound runs.
#   - DDP bucket_cap_mb / static_graph tuning (makani + PanguWeather-e3sm
#     inspired): within measurement noise at this 303.9M-param / batch-16 scale.
#   - torch.compile (training.jit_compile=True): the SHT/complex island is now
#     carved out of the graph via @torch.compiler.disable (makani's strategy),
#     so it compiles cleanly with loss identical to eager — but the steady-state
#     gain is only ~1.3% here (within noise) and it adds ~90 s compile warmup
#     per job, so it's left OFF by default. The infra is kept as an opt-in flag.
#   - channels_last: would need hot-loop tensor-format changes with SHT stride
#     risk; low expected upside given the SHT dominates the spectral path. Future.
#   - Spatial/channel model parallelism (makani's torch_harmonics.distributed):
#     per-rank activation memory is invariant to the data-vs-spatial split at
#     fixed global-batch/GPU-count, and it adds per-SHT all-to-all comm — no
#     win on a single node; an enabler for multi-node / larger-grid scale-out.
#
# Submit with:
#   sbatch hpc/scripts/train_sfno_e3sm.sbatch

#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=train-sfno-e3sm
#SBATCH --output=hpc/scripts/logs/train-sfno-e3sm-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

# makani's documented allocator fix — reclaims reserved-but-unallocated
# fragmentation so global batch 16 fits at checkpointing=2 (see header). MUST
# be exported before CUDA initializes, i.e. before torchrun launches python.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-batch loss/timing output dir on HDD scratch.
OUT_DIR=/work/hdd/bdiu/awikner/sfno_bench
mkdir -p "${OUT_DIR}"

# E3SM year-zarrs (full 18-pressure-level coverage). Train on 2041, validate
# (single-step val_loss only) on 2045 — one archived year each suffices for
# training-speed work; see the optimization plan for why 2015-2040 aren't
# part of this archive.
ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/2041.zarr
VAL_ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/2045.zarr
if [ ! -d "${ZARR_PATH}" ]; then
    echo "ERROR: ${ZARR_PATH} not present." >&2
    exit 1
fi
if [ ! -d "${VAL_ZARR_PATH}" ]; then
    echo "ERROR: ${VAL_ZARR_PATH} not present." >&2
    exit 1
fi
NORM_ZARR=/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/normalization_2015-2050.zarr

nvidia-smi

torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --standalone \
    examples/weather/ai_rossby/train.py \
        --config-dir=examples/weather/ai_rossby/conf \
        --config-name=config \
        run_name=train_sfno_e3sm \
        model=sfno_e3sm \
        loss=raw_l2 \
        training=sfno_plasim \
        validation=off \
        dataset=e3sm \
        seed=0 \
        training.grad_clip_norm=0.0 \
        training.ema.enabled=False \
        training.amp=bf16 \
        model.checkpointing=2 \
        training.optimizer.fused=True \
        dataset.zarr_path=${ZARR_PATH} \
        dataset.val_zarr_path=${VAL_ZARR_PATH} \
        dataset.mean_path=${NORM_ZARR} \
        dataset.std_path=${NORM_ZARR} \
        dataset.batch_size=4 \
        dataset.num_workers=8 \
        bench.per_batch_tsv=${OUT_DIR}/train_sfno_e3sm_${SLURM_JOB_ID}.tsv

echo
echo "SFNO-E3SM training TSV: ${OUT_DIR}/train_sfno_e3sm_${SLURM_JOB_ID}.tsv"
