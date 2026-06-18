#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pangu_Plasim (legacy / deterministic) training shake-out on NCSA Delta.
#
# Runs the Hydra entrypoint at examples/weather/ai_rossby/train.py on a full
# A40 node (4 GPUs) via torchrun. This is the *fidelity* run that exercises the
# whole recipe end-to-end on real PLASIM data — substantially longer than the
# smoke tests in test/recipes/ai_rossby/, hence the non-interactive
# `gpuA40x4` partition (2-day walltime).
#
# Submit (defaults shown):
#
#     sbatch hpc/scripts/ai_rossby_shake_out.sbatch
#
# Overrides (Hydra-style — appended verbatim to the train.py command):
#
#     OVERRIDES='data.batch_size=8 max_epochs=2' \
#         sbatch hpc/scripts/ai_rossby_shake_out.sbatch
#
# Required env (set in your shell before sbatch, or hard-coded below):
#
#   AI_ROSSBY_TEST_DATA   — root for the PLASIM fixtures + stats
#                           (default: /work/nvme/bdiu/awikner/physicsnemo_test_data)
#
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=200g
#SBATCH --job-name=pp-legacy-shakeout
#SBATCH --output=hpc/scripts/logs/pp-legacy-shakeout-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
source .venv/bin/activate

# NCCL via AWS-OFI is multi-GPU-only.
module load aws-ofi-nccl/1.14.2

export AI_ROSSBY_TEST_DATA="${AI_ROSSBY_TEST_DATA:-/work/nvme/bdiu/awikner/physicsnemo_test_data}"

# PLASIM Zarr fixture + per-variable mean/std NetCDFs (PanguWeather v2.0 sim52).
ZARR_PATH="${ZARR_PATH:-$AI_ROSSBY_TEST_DATA/plasim/smoke_month.zarr}"
STATS_DIR="${STATS_DIR:-/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data}"
MEAN_PATH="${MEAN_PATH:-$STATS_DIR/data_12-132_mean_sigma.nc}"
STD_PATH="${STD_PATH:-$STATS_DIR/data_12-132_std_sigma.nc}"

for f in "$ZARR_PATH" "$MEAN_PATH" "$STD_PATH"; do
    if [[ ! -e "$f" ]]; then
        echo "[ERROR] missing required input: $f" >&2
        exit 1
    fi
done

OVERRIDES="${OVERRIDES:-}"

mkdir -p hpc/scripts/logs

cd examples/weather/ai_rossby

# torchrun on the local node — `--standalone` skips the rendezvous server.
torchrun --standalone --nproc-per-node=4 train.py \
    data.zarr_path="$ZARR_PATH" \
    data.mean_path="$MEAN_PATH" \
    data.std_path="$STD_PATH" \
    run_name="pp-legacy-shakeout-$SLURM_JOB_ID" \
    $OVERRIDES
