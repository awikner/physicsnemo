#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 6 Track A smoke — runs the new PanguPlasimLegacyNative through one
# mini-epoch on PLASIM sim52 year 12 on a 2xA40 interactive node. Verifies
# the upstream-block migration + native MetaData round-trip cleanly on real
# data + real GPU + DDP launch, end to end. (Moved from gpuA100x4-interactive
# which had long queue waits.)

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=pp-native-smoke
#SBATCH --output=hpc/scripts/logs/pp-native-smoke-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
source .venv/bin/activate

module load aws-ofi-nccl/1.14.2

mkdir -p hpc/scripts/logs
cd examples/weather/ai_rossby

# Single-epoch shake-out on the new PanguPlasimLegacyNative class.
# Uses conf/dataset/plasim_sim52_year12.yaml's default paths so no env
# overrides are needed. Knobs:
# - model=pangu_plasim_legacy_native   → routes build_model to the new class
# - training=pangu_plasim_legacy        → reuses the existing optimizer + sched
# - training.stages.0.num_epochs=1      → cap to one epoch
# - dataset.batch_size=4                → fits comfortably on A100 with checkpointing=0
# - dataset.num_workers=4               → loader parallelism
# - log_every_n_steps=5                 → keep stdout chatty enough to verify
torchrun --standalone --nproc-per-node=2 train.py \
    model=pangu_plasim_legacy_native \
    training=pangu_plasim_legacy \
    dataset=plasim_sim52_year12 \
    'model.levels=[20000,25000,30000,40000,50000,60000,70000,85000,92500,100000]' \
    dataset.batch_size=4 \
    dataset.num_workers=4 \
    log_every_n_steps=5 \
    training.max_epochs=1 \
    run_name="pp-native-smoke-$SLURM_JOB_ID"

# Note: model.levels override is to match the year12 PLASIM zarr's 10 actual
# pressure levels (the committed pangu_plasim_legacy.yaml + the native YAML
# both list 13 levels, including 5000/10000/15000 Pa that aren't in this
# Zarr conversion). Pre-existing config drift — fix is to either re-convert
# with the missing 3 levels or update the model configs; tracked separately.

echo
echo "=== smoke test passed for SLURM job $SLURM_JOB_ID"
