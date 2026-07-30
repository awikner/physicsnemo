#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Midway3 FULL MoWE Method-0 training on the pedramh-gpu partition
# (midway3-0423: 4x H100, 32 cores, 512G — dedicated, so bf16 AMP and all
# four GPUs). Data is read from midway2 scratch, which is mounted on the
# compute nodes (dataset=hindcast_midway2).
# Model v1: log-precip space, IMD-masked loss, 2000-2019 / 2020+ Mar-Sep split.
# One-time setup: bash setup_midway3.sh (creates the venv on scratch).
#   sbatch train_mowe_midway3_h100.sbatch
# Variants: sbatch --export=ALL,EXTRA="loss=regional_log_mse",RUN_NAME=... ...
# NOTE: Slurm's --export list is COMMA-separated, so an EXTRA containing a
# comma (e.g. dataset.train.years=[2000,2024]) is silently truncated and Hydra
# then rejects "...years=[2000". For overrides with commas, write them to a
# file and pass EXTRA_FILE=<path> instead (a path has no commas).
# Resume: re-sbatch with the same RUN_NAME (picks up the latest checkpoint).
#SBATCH -J mowe_train_h100
#SBATCH -A pi-pedramh
#SBATCH -p pedramh-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=480G
#SBATCH -t 36:00:00
#SBATCH -o /scratch/midway3/awikner/mowe_runs/train_h100_%j.log

set -uo pipefail
REPO=/scratch/midway3/awikner/physicsnemo
# /scratch/midway2 (where the data lives) is not mounted on every Midway3
# node -- fail fast rather than reporting it as missing data downstream.
if [ ! -d /scratch/midway2/awikner/hindcasts_mowe/pangu_s2s ]; then
    echo "FATAL: /scratch/midway2 not mounted on $(hostname)"; exit 75
fi
source "$REPO/.venv-mowe/bin/activate"

RUNDIR=/scratch/midway3/awikner/mowe_runs
mkdir -p "$RUNDIR"

# Overrides containing commas must arrive via a file, not --export (see above).
if [ -n "${EXTRA_FILE:-}" ]; then
    if [ ! -f "$EXTRA_FILE" ]; then
        echo "FATAL: EXTRA_FILE=$EXTRA_FILE does not exist"; exit 2
    fi
    EXTRA="$(cat "$EXTRA_FILE")"
    echo "overrides from $EXTRA_FILE: $EXTRA"
fi
cd "$RUNDIR"

NGPU=$(nvidia-smi -L | wc -l)
echo "[$(date)] training on $NGPU GPU(s): $(nvidia-smi -L | head -1)"

torchrun --standalone --nproc-per-node="$NGPU" \
    "$REPO/examples/weather/ai_rossbypalooza/train.py" \
    dataset=hindcast_midway2 \
    run_name="${RUN_NAME:-mowe_v1_h100}" \
    dataset.loader.num_workers=7 \
    wandb.enabled="${WANDB:-true}" \
    ${EXTRA:-}
rc=$?; echo "[$(date)] training exit code $rc"; exit $rc
