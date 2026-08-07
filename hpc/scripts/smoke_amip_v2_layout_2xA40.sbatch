#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 12b GPU smoke — two cluster-bound follow-ups in one job:
#
#  1. Live translator sweep: pytest re-translates every real v1
#     Lightning ckpt under /work/nvme/bdiu/awikner/amip-checkpoints/
#     with the (now default) --source-contract v1 path and asserts the
#     .mdlus roundtrip carries channel_layout="v1" + finite forward.
#  2. v2-layout mini-epoch: RollingDiTWrapper(channel_layout=v2) +
#     ERDMScheduler with the amip_v2 rebaselined stats (loss=erdm_v2,
#     sigma_data 1.0, churn on) through a capped rolling-window stage
#     on the real 1981 AMIP Zarr, 2xA40 DDP.
#
# Runs from the ai-rossby-amip-v2 WORKTREE (the main Delta checkout is
# on ai-rossbypalooza — do not disturb it). PYTHONPATH pins the
# worktree's physicsnemo over the venv's editable install.
#
# Submit:  sbatch hpc/scripts/smoke_amip_v2_layout_2xA40.sbatch

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=amip-v2-smoke
#SBATCH --output=/work/nvme/bdiu/awikner/physicsnemo-amip-v2/hpc/scripts/logs/amip-v2-smoke-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo-amip-v2
VENV_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
source "$VENV_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

module load aws-ofi-nccl/1.14.2
mkdir -p hpc/scripts/logs

python -c "import physicsnemo, pathlib; p = pathlib.Path(physicsnemo.__file__); \
assert str(p).startswith('$REPO_ROOT'), f'wrong physicsnemo on path: {p}'; \
print('physicsnemo from', p)"

if [[ -z "${SKIP_STEP1:-}" ]]; then
    echo "=== Step 1/2: live translator sweep (v1 contract) ==="
    python -m pytest test/tools/checkpoint_translation/test_amip_si.py \
        -m slow -v --no-header -rA
else
    echo "=== Step 1/2 skipped (SKIP_STEP1 set; passed in job 20916803) ==="
fi

echo
echo "=== Step 2/2: v2-layout rolling mini-stage (ERDM v2 stats) ==="
ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
if [[ ! -d "$ZARR_PATH" ]]; then
    echo "ERROR: $ZARR_PATH missing"
    exit 1
fi

cd examples/weather/ai_rossby

# Knobs:
# - model=amip_rfm               -> RollingDiTWrapper with the AMIP-Zarr
#                                   variable set (the amip_erdm_v2 config
#                                   targets the 12c daily-avg dataset,
#                                   which doesn't exist yet)
# - model.channel_layout=v2      -> the Phase 12b amip_v2 packing under test
# - loss=erdm_v2                 -> rebaselined scheduler stats
#                                   (sigma_data 1.0, S_churn 1.0, alpha 0)
# - loss.window_size=3           -> lighter rolling window for the smoke
#                                   (RollingDiT is unpatched: 64800
#                                   tokens/frame at 180x360)
# - tiny backbone + capped iterations -> fits 2xA40, finishes in minutes
# - wandb.enabled=false          -> jobs 20916803/20918380 hit the NCCL-
#                                   watchdog GIL hang from wandb's background
#                                   threads under DDP (the regression-2
#                                   signature from commit 1a1b843b); the
#                                   smoke needs console logging only
torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    model=amip_rfm \
    wandb.enabled=false \
    model.channel_layout=v2 \
    loss=erdm_v2 \
    loss.window_size=3 \
    dataset=amip_1981 \
    training=amip_diffusion \
    validation=off \
    training.max_epochs=1 \
    training.ema.warmup_epochs=0 \
    'training.stages.0.max_iterations=20' \
    'model.rolling_dit_kwargs.dim=64' \
    'model.rolling_dit_kwargs.num_blocks=2' \
    'model.rolling_dit_kwargs.num_heads=4' \
    'model.rolling_dit_kwargs.temporal_num_heads=4' \
    dataset.batch_size=1 \
    dataset.num_workers=4 \
    log_every_n_steps=5 \
    run_name="amip-v2-smoke-$SLURM_JOB_ID" \
    +checkpoint_dir="./checkpoints_smoke_$SLURM_JOB_ID"

echo
echo "=== Phase 12b smoke passed for SLURM job $SLURM_JOB_ID"
