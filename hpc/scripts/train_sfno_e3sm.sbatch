#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optimized SFNO-E3SM training config — the winner of a training-speed
# optimization pass (see implementation_plan.md / the SFNO-E3SM optimization
# plan) targeting global batch 16 on a single Delta gpuA100x4-interactive
# node (4x A100 40GB).
#
# Winning config vs. the naive fp32 baseline (1.067 s/step, 15.0 samples/sec):
#   bf16 AMP + model.checkpointing=3 + fused AdamW  ->  0.991 s/step,
#   16.15 samples/sec (~1.08x speedup).
#
# Key finding: model.checkpointing=3 (activation checkpointing) is NOT
# optional here — at E3SM's 180x360 grid (~8x more grid points than the
# PLASIM_5412 benchmark this architecture was designed for), batch_size=4/GPU
# (global batch 16) OOMs on a 40GB A100 with checkpointing<3, in BOTH fp32
# and bf16. Do not remove it without re-verifying memory headroom.
#
# Other levers tried and NOT adopted:
#   - CUDA graphs (training.use_static_capture=True): physicsnemo detects
#     SfnoPlasim doesn't support graph capture and auto-disables it (no-op).
#   - NVMe data staging (copy the year-zarr to node-local /tmp before
#     training): a naive `cp -r` of the chunked Zarr store (thousands of
#     small chunk files) didn't finish copying within a 25-minute test
#     budget. Moot anyway — nsys profiling (Phase 2) showed no dataloader
#     I/O-stall evidence; this model is compute-bound, not I/O-bound.
#   - torch.compile / FSDP: not implemented in train.py; would need code
#     changes, out of scope for this pass.
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
        model.checkpointing=3 \
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
