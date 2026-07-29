#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Midway3 FULL MoWE Method-0 training on the gpu partition (A100 node via
# -C a100; for the V100/RTX6000 nodes drop the constraint and add
# EXTRA="training.amp=none"). Data is read from midway2 scratch, which is
# mounted on midway3 compute nodes (dataset=hindcast_midway2).
# Model v1: log-precip space, IMD-masked loss, 2000-2019 / 2020+ Mar-Sep split.
# One-time setup: bash setup_midway3.sh (creates the venv on scratch).
#   sbatch train_mowe_midway3.sbatch
# Variants: sbatch --export=ALL,EXTRA="loss=regional_log_mse run_name=..." ...
# Resume: re-sbatch with the same RUN_NAME.
#SBATCH -J mowe_train_v1
#SBATCH -A ai4s-hackathon
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/midway3/awikner/mowe_runs/train_v1.log

set -uo pipefail
REPO=/scratch/midway3/awikner/physicsnemo
source "$REPO/.venv-mowe/bin/activate"

RUNDIR=/scratch/midway3/awikner/mowe_runs
mkdir -p "$RUNDIR"
cd "$RUNDIR"

NGPU=$(nvidia-smi -L | wc -l)
echo "[$(date)] training on $NGPU GPU(s): $(nvidia-smi -L | head -1)"

torchrun --standalone --nproc-per-node="$NGPU" \
    "$REPO/examples/weather/ai_rossbypalooza/train.py" \
    dataset=hindcast_midway2 \
    run_name="${RUN_NAME:-mowe_v1_prod}" \
    dataset.loader.num_workers=6 \
    wandb.enabled="${WANDB:-false}" \
    ${EXTRA:-}
rc=$?; echo "[$(date)] training exit code $rc"; exit $rc
