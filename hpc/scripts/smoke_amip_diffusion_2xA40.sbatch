#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 8c GPU smoke — runs the AMIP SI diffusion recipe through one
# mini-epoch on the 1981 AMIP Zarr archive on a 2xA40 interactive node.
# Verifies the full pack/unpack + diffusion-loss + EMA + multi-stage
# scaffolding end-to-end on real data, real GPU, and DDP launch.
#
# Pairs with:
#   conf/model/amip_si.yaml             — AmipDiTWrapper around AmipDiT
#   conf/loss/si.yaml                   — DriftScheduler (SI baseline)
#   conf/dataset/amip_1981.yaml         — 1981 AMIP Zarr
#   conf/training/amip_diffusion.yaml   — Muon + EMA 0.999 + fp32
#   conf/validation/diffusion_rollout.yaml — every-5-epoch rollout
#
# Prereq: AMIP 1981 conversion job has succeeded and the Zarr at
#   /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
# is present. Tracked in conversation as job 19640283.

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=amip-diff-smoke
#SBATCH --output=hpc/scripts/logs/amip-diff-smoke-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
source .venv/bin/activate

module load aws-ofi-nccl/1.14.2

mkdir -p hpc/scripts/logs

# Confirm the conversion output is on disk before we burn GPU minutes.
ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
if [[ ! -d "$ZARR_PATH" ]]; then
    echo "ERROR: $ZARR_PATH missing — wait for the AMIP 1981 conversion job"
    exit 1
fi

cd examples/weather/ai_rossby

# Knobs:
# - model=amip_si               → AmipDiTWrapper (single-step diffusion)
# - loss=si                     → DriftScheduler (SI baseline)
# - dataset=amip_1981           → 1981 AMIP Zarr
# - training=amip_diffusion     → Muon + EMA + fp32 + cosine
# - validation=diffusion_rollout→ 5-epoch cadence rollout val + spread metric
# - training.max_epochs=1       → cap to one epoch for the smoke
# - validation.every_n_epochs=1 → exercise the validator path inside this 1 epoch
# - training.ema.warmup_epochs=0→ no need to warm EMA in a 1-epoch smoke
# - dit_kwargs.dim=128          → shrink the backbone to fit two A40s with batch=2
# - dit_kwargs.num_blocks=4     → ditto
# - patch_size=4                → coarser patch keeps token count tractable at 180×360
# - c_grid_downsample=1         → wrapper default, matches train.py x_noised resolution
torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    model=amip_si \
    loss=si \
    dataset=amip_1981 \
    training=amip_diffusion \
    validation=diffusion_rollout \
    training.max_epochs=1 \
    training.ema.warmup_epochs=0 \
    training.optimizer.type=AdamW \
    training.optimizer.fused=null \
    validation.every_n_epochs=1 \
    validation.rollout.max_initial_conditions=2 \
    validation.rollout.log_steps='[1]' \
    'model.dit_kwargs.dim=128' \
    'model.dit_kwargs.num_blocks=4' \
    'model.dit_kwargs.num_heads=4' \
    'model.dit_kwargs.patch_size=4' \
    'model.dit_kwargs.c_grid_downsample=1' \
    dataset.batch_size=2 \
    dataset.num_workers=4 \
    log_every_n_steps=5 \
    run_name="amip-diff-smoke-$SLURM_JOB_ID"

# Note: optimizer overridden Muon→AdamW for the smoke. Muon needs param-group
# plumbing (the upstream amip splits ≥2D weights into Muon + 1D into Adam via
# MuonWithAuxAdam — see /work/nvme/bdiu/awikner/amip/modules/train_module.py
# get_dit_muon_param_groups()). Out of scope for the wiring smoke; tracked as
# Phase 8f follow-up alongside the rest of the upstream-Muon port.

echo
echo "=== AMIP diffusion smoke test passed for SLURM job $SLURM_JOB_ID"
