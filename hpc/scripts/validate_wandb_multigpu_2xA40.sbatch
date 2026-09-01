#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# W3 — long-run wandb-ON DDP validation (wandb_ddp_hang_fix_plan.md).
#
# The W2 matrix confirmed the every-rank call-site fix on ~1-min cells;
# this run is the second half of the re-enable gate: >=1 h of 2xA40 DDP
# with wandb ON (allow_multigpu=true), covering the MID-EPOCH desync
# failure mode that motivated the every-rank strategy (fad7c0bb) and
# that short cells cannot see. One full epoch of the v2-layout ERDM
# smoke config = 729 batches x ~7.5 s ~= 91 min, plus the end-of-epoch
# checkpoint path.
#
# Passes => flip conf/config.yaml wandb.allow_multigpu default to True.
#
# Non-interactive gpuA40x4 (the interactive queue's 1 h cap can't fit).
# Submit:  sbatch hpc/scripts/validate_wandb_multigpu_2xA40.sbatch

#SBATCH --partition=gpuA40x4
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=wandb-mgpu-val
#SBATCH --output=/work/nvme/bdiu/awikner/physicsnemo-amip-v2/hpc/scripts/logs/wandb-mgpu-val-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo-amip-v2
VENV_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
# Runs in the ai-rossby container (hpc/containers/) instead of a per-cluster venv.
# AI_ROSSBY_VENV is superseded by AI_ROSSBY_SIF / AI_ROSSBY_CONTAINER_TAG.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

module load aws-ofi-nccl/1.14.2
mkdir -p hpc/scripts/logs

# Flight recorder armed: if the mid-epoch desync exists, we want the
# per-rank last-collective dump, and a 120 s heartbeat fails fast
# instead of burning the allocation.
FR_DIR="$REPO_ROOT/hpc/scripts/logs/nccl_fr_$SLURM_JOB_ID"
mkdir -p "$FR_DIR"
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$FR_DIR/rank_"

cd examples/weather/ai_rossby

# Same tiny v2-layout ERDM config as the W2 matrix, but a FULL epoch
# (no max_iterations cap) with wandb ON.
"$CONTAINER_RUN" torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    model=amip_rfm \
    model.channel_layout=v2 \
    loss=erdm_v2 \
    loss.window_size=3 \
    dataset=amip_1981 \
    training=amip_diffusion \
    validation=off \
    training.max_epochs=1 \
    training.ema.warmup_epochs=0 \
    'model.rolling_dit_kwargs.dim=64' \
    'model.rolling_dit_kwargs.num_blocks=2' \
    'model.rolling_dit_kwargs.num_heads=4' \
    'model.rolling_dit_kwargs.temporal_num_heads=4' \
    dataset.batch_size=1 \
    dataset.num_workers=4 \
    log_every_n_steps=25 \
    wandb.allow_multigpu=true \
    run_name="wandb-mgpu-val-$SLURM_JOB_ID" \
    +checkpoint_dir="./ckpt_wandb_val_$SLURM_JOB_ID"

echo
echo "=== W3 wandb multi-GPU long-run validation passed for SLURM job $SLURM_JOB_ID"