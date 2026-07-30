#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Stampede3 FULL MoWE Method-0 training on the h100 partition (one node,
# all GPUs on it), model v1 (log-precip space, 2000-2019 / 2020+ JJAS).
# One-time setup first: bash setup_stampede3.sh  (creates .venv-mowe).
#   sbatch train_mowe_stampede3.sbatch
# Variants:
#   sbatch --export=ALL,EXTRA="loss=regional_log_mse run_name=mowe_v1_logmse" ...
#   sbatch --export=ALL,EXTRA="training=all_experts run_name=mowe_v1_allexp" ...
# Resume: re-sbatch with the same run_name (checkpoints under
# $SCRATCH/mowe_runs/outputs/<run_name>/checkpoints).
#SBATCH -J mowe_train_v1
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o /scratch/09979/awikner/mowe_runs/train_v1.log

set -uo pipefail
REPO=/work2/09979/awikner/stampede3/physicsnemo
source "$REPO/.venv-mowe/bin/activate"

RUNDIR=/scratch/09979/awikner/mowe_runs
mkdir -p "$RUNDIR"
cd "$RUNDIR"

NGPU=$(nvidia-smi -L | wc -l)
echo "[$(date)] training on $NGPU GPU(s): $(nvidia-smi -L | head -1)"

torchrun --standalone --nproc-per-node="$NGPU" \
    "$REPO/examples/weather/ai_rossbypalooza/train.py" \
    dataset=hindcast_stampede3 \
    run_name="${RUN_NAME:-mowe_v1}" \
    dataset.loader.num_workers=8 \
    wandb.enabled="${WANDB:-false}" \
    ${EXTRA:-}
rc=$?; echo "[$(date)] training exit code $rc"; exit $rc
