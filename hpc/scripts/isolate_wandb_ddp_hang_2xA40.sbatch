#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# W2 — wandb x DDP NCCL-hang isolation matrix (wandb_ddp_hang_fix_plan.md).
#
# Sequence of short 2xA40 micro-trainings, each with wandb ON
# (wandb.allow_multigpu=true lifts the W1 guard) and one variable under
# test. The NCCL flight recorder is armed so any hang dumps each rank's
# last collective + stack; a per-cell `timeout` records FAIL and moves on.
#
# The pre-fix baseline (rank-0-only _maybe_init_wandb, the W2 primary
# suspect) is NOT re-run — it is already documented failing twice (jobs
# 20916803 / 20918380). All cells below run the every-rank call-site fix.
#
# Cells (M1 has three repeats — the hang is timing-shaped, single passes
# prove little):
#   M1a/b/c  every-rank init + wandb on (offline)      <- primary validation
#   M2       + GPU/system-stats monitor off (env)
#   M3       + console capture off (env)
#   M4       wandb.init_after_ddp=true (ordering fix)
#   M5       wandb.nonzero_rank_mode=disabled (asymmetry re-test)
#   M6       train.py (PanguPlasimLegacy/PLASIM) + wandb on — the
#            July-validated every-rank path, now on A40
#   (version-drift cell dropped: wandb was already 0.27.0 at 1a1b843b)
#
# Submit:  sbatch hpc/scripts/isolate_wandb_ddp_hang_2xA40.sbatch

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=wandb-ddp-matrix
#SBATCH --output=/work/nvme/bdiu/awikner/physicsnemo-amip-v2/hpc/scripts/logs/wandb-ddp-matrix-%j.out

set -uo pipefail   # NOT -e: failing cells must not kill the matrix

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo-amip-v2
VENV_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
# Runs in the ai-rossby container (hpc/containers/) instead of a per-cluster venv.
# AI_ROSSBY_VENV is superseded by AI_ROSSBY_SIF / AI_ROSSBY_CONTAINER_TAG.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

module load aws-ofi-nccl/1.14.2
mkdir -p hpc/scripts/logs

# --- NCCL flight recorder: fail fast + dump on hang ---------------------
FR_DIR="$REPO_ROOT/hpc/scripts/logs/nccl_fr_$SLURM_JOB_ID"
mkdir -p "$FR_DIR"
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$FR_DIR/rank_"

SUMMARY="$REPO_ROOT/hpc/scripts/logs/wandb-ddp-matrix-$SLURM_JOB_ID.summary"
: > "$SUMMARY"

COMMON_DIFF=(
    model=amip_rfm
    model.channel_layout=v2
    loss=erdm_v2
    loss.window_size=3
    dataset=amip_1981
    training=amip_diffusion
    validation=off
    training.max_epochs=1
    training.ema.warmup_epochs=0
    'training.stages.0.max_iterations=5'
    'model.rolling_dit_kwargs.dim=64'
    'model.rolling_dit_kwargs.num_blocks=2'
    'model.rolling_dit_kwargs.num_heads=4'
    'model.rolling_dit_kwargs.temporal_num_heads=4'
    dataset.batch_size=1
    dataset.num_workers=4
    log_every_n_steps=1
    wandb.allow_multigpu=true
)

run_cell () {
    local name="$1"; shift
    local script="$1"; shift
    # remaining args: NAME=VALUE env pairs first (containing '='+uppercase), then hydra overrides
    local envs=() overrides=()
    for a in "$@"; do
        if [[ "$a" =~ ^WANDB_[A-Z_]+= ]]; then envs+=("$a"); else overrides+=("$a"); fi
    done
    echo
    echo "=== CELL $name: script=$script envs=[${envs[*]:-}] overrides=[${overrides[*]:-}] ==="
    local t0=$SECONDS
    # 420 s cap: heartbeat 120 s + dump + teardown headroom; a clean cell
    # finishes in ~2-3 min.
    env "${envs[@]}" timeout --signal=KILL 420 \
        "$CONTAINER_RUN" torchrun --standalone --nproc-per-node=2 "$script" \
        "${overrides[@]}" \
        run_name="wdm-$name-$SLURM_JOB_ID" \
        +checkpoint_dir="./ckpt_wdm_${name}_$SLURM_JOB_ID"
    local rc=$? dt=$((SECONDS - t0))
    if [[ $rc -eq 0 ]]; then
        echo "CELL $name: PASS (${dt}s)" | tee -a "$SUMMARY"
    else
        echo "CELL $name: FAIL rc=$rc (${dt}s)" | tee -a "$SUMMARY"
    fi
}

cd examples/weather/ai_rossby

# M1 x3 — the every-rank call-site fix, wandb on. Primary validation.
run_cell M1a train_diffusion.py "${COMMON_DIFF[@]}"
run_cell M1b train_diffusion.py "${COMMON_DIFF[@]}"
run_cell M1c train_diffusion.py "${COMMON_DIFF[@]}"

# M2 — stats monitor off (both env spellings; wandb warns on unknown keys
# but proceeds, so passing both covers 0.16-0.27 naming).
run_cell M2 train_diffusion.py WANDB_X_DISABLE_STATS=true WANDB_DISABLE_STATS=true "${COMMON_DIFF[@]}"

# M3 — console capture off.
run_cell M3 train_diffusion.py WANDB_CONSOLE=off "${COMMON_DIFF[@]}"

# M4 — init wandb only after the DDP wrap.
run_cell M4 train_diffusion.py "${COMMON_DIFF[@]}" wandb.init_after_ddp=true

# M5 — non-zero ranks thread-free (asymmetry re-test).
run_cell M5 train_diffusion.py "${COMMON_DIFF[@]}" wandb.nonzero_rank_mode=disabled

# M6 — train.py control (the July-validated path) on A40.
COMMON_DET=(
    model=pangu_plasim_legacy
    dataset=plasim_sim52_year12
    training=pangu_plasim_legacy
    validation=off
    'training.stages.0.num_epochs=1'
    'training.stages.0.max_iterations=5'
    dataset.batch_size=1
    dataset.num_workers=4
    log_every_n_steps=1
    wandb.allow_multigpu=true
)
run_cell M6 train.py "${COMMON_DET[@]}"

echo
echo "=== W2 matrix summary (job $SLURM_JOB_ID) ==="
cat "$SUMMARY"
echo "flight-recorder dumps (FAIL cells only): $FR_DIR"
ls -la "$FR_DIR" 2>/dev/null || true