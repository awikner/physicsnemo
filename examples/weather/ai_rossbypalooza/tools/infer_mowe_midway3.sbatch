#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Save MoWE gate forecasts on the Midway3 pedramh-gpu H100 node. Inference is
# single-process and I/O bound, so one GPU is enough.
#   sbatch --export=ALL,CKPT=<checkpoints_best dir>,OUT=<zarr>,SPLIT=val \
#          tools/infer_mowe_midway3.sbatch
#SBATCH -J mowe_infer
#SBATCH -A pi-pedramh
#SBATCH -p pedramh-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/midway3/awikner/mowe_runs/infer_%j.log

set -uo pipefail
REPO=/scratch/midway3/awikner/physicsnemo
if [ ! -d /scratch/midway2/awikner/hindcasts_mowe/pangu_s2s ]; then
    echo "FATAL: /scratch/midway2 not mounted on $(hostname)"; exit 75
fi
source "$REPO/.venv-mowe/bin/activate"
cd /scratch/midway3/awikner/mowe_runs

python "$REPO/examples/weather/ai_rossbypalooza/tools/infer_mowe.py" \
    dataset=hindcast_midway2 \
    +checkpoint="$CKPT" \
    +out="$OUT" \
    +split="${SPLIT:-val}" \
    +save_gate="${SAVE_GATE:-true}" \
    dataset.loader.num_workers=7
rc=$?; echo "[$(date)] inference exit code $rc"; exit $rc
