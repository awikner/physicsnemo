#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Week-2 accumulated-precip ACC by month, all sources, on the Midway3 H100 node.
#   sbatch --export=ALL,CKPT=<checkpoints_best>,OUT=<png> tools/plot_week2_acc_midway3.sbatch
#SBATCH -J week2_acc
#SBATCH -A pi-pedramh
#SBATCH -p pedramh-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/midway3/awikner/mowe_runs/week2_acc_%j.log

set -uo pipefail
REPO=/scratch/midway3/awikner/physicsnemo
if [ ! -d /scratch/midway2/awikner/hindcasts_mowe/pangu_s2s ]; then
    echo "FATAL: /scratch/midway2 not mounted on $(hostname)"; exit 75
fi
source "$REPO/.venv-mowe/bin/activate"
cd /scratch/midway3/awikner/mowe_runs
python "$REPO/examples/weather/ai_rossbypalooza/tools/plot_week2_acc.py" \
    dataset=hindcast_midway2 \
    +checkpoint="$CKPT" \
    +out="$OUT" \
    dataset.loader.num_workers=7
rc=$?; echo "[$(date)] exit code $rc"; exit $rc
