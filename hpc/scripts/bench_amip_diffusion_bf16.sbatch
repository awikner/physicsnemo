#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 8f (F3) fp32-vs-bf16 benchmark for the AMIP SI diffusion recipe.
# Runs two short torchrun jobs back-to-back on the same 2xA40 allocation
# (training=amip_diffusion vs. training=amip_diffusion_bf16, everything
# else identical) against the 1981 AMIP Zarr, and reports throughput +
# peak GPU memory + final loss for both so
# benchmarks/physicsnemo/experimental/models/amip_si/RESULTS.md can be
# filled in from this job's log.
#
# Pairs with:
#   conf/training/amip_diffusion.yaml       — fp32 (amp: none)
#   conf/training/amip_diffusion_bf16.yaml  — bf16-native (amp: bf16)
#
# Prereq: same as smoke_amip_diffusion_2xA40.sbatch — the 1981 AMIP Zarr
# conversion must have completed, and (if not already run)
#   uv pip install git+https://github.com/KellerJordan/Muon
# in the venv, since both training configs default to optimizer.type=Muon.

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=bench-amip-diffusion-bf16
#SBATCH --output=hpc/scripts/logs/bench-amip-diffusion-bf16-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
source .venv/bin/activate

module load aws-ofi-nccl/1.14.2

mkdir -p hpc/scripts/logs

ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
if [[ ! -d "$ZARR_PATH" ]]; then
    echo "ERROR: $ZARR_PATH missing — wait for the AMIP 1981 conversion job"
    exit 1
fi

OUT_DIR=/work/hdd/bdiu/awikner/amip_si_bench
mkdir -p "${OUT_DIR}"

cd examples/weather/ai_rossby

# Shared overrides: shrunk backbone to fit two A40s at batch=2 (matches
# smoke_amip_diffusion_2xA40.sbatch sizing), 3 epochs (enough steps past
# warmup for a stable throughput read), no rollout validation (bench is
# throughput/memory/loss only).
COMMON_ARGS=(
    model=amip_si
    loss=si
    dataset=amip_1981
    validation=off
    training.max_epochs=3
    training.ema.warmup_epochs=0
    'model.dit_kwargs.dim=128'
    'model.dit_kwargs.num_blocks=4'
    'model.dit_kwargs.num_heads=4'
    'model.dit_kwargs.patch_size=4'
    'model.dit_kwargs.c_grid_downsample=1'
    dataset.batch_size=2
    dataset.num_workers=4
    log_every_n_steps=5
)

echo "=== fp32 (training=amip_diffusion) ==="
torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    "${COMMON_ARGS[@]}" \
    training=amip_diffusion \
    run_name="bench-amip-diffusion-fp32-$SLURM_JOB_ID" \
    +checkpoint_dir="./checkpoints_bench_fp32_$SLURM_JOB_ID" \
    bench.per_batch_tsv="${OUT_DIR}/fp32_${SLURM_JOB_ID}.tsv"

echo
echo "=== bf16 (training=amip_diffusion_bf16) ==="
torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    "${COMMON_ARGS[@]}" \
    training=amip_diffusion_bf16 \
    run_name="bench-amip-diffusion-bf16-$SLURM_JOB_ID" \
    +checkpoint_dir="./checkpoints_bench_bf16_$SLURM_JOB_ID" \
    bench.per_batch_tsv="${OUT_DIR}/bf16_${SLURM_JOB_ID}.tsv"

echo
echo "=== bench TSVs ==="
echo "fp32: ${OUT_DIR}/fp32_${SLURM_JOB_ID}.tsv"
echo "bf16: ${OUT_DIR}/bf16_${SLURM_JOB_ID}.tsv"
echo "Peak GPU memory + final loss are logged above per run ('peak GPU memory: ... GB')."
