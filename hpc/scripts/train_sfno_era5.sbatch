#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cluster-agnostic Slurm launcher for SFNO training on the full multi-year ERA5
# archive (conf/dataset/era5_multiyear.yaml + conf/model/sfno_era5.yaml).
#
# Works on any of our Slurm clusters (Delta, DeltaAI, Stampede3, Midway3, DSI).
# NCAR Derecho uses PBS, not Slurm — use a PBS script there instead.
#
# It is deliberately NOT hardwired to one site:
#   * partition / account / walltime come from `sbatch` flags at submit time;
#   * the repo is `$SLURM_SUBMIT_DIR` (submit from the repo root);
#   * per-GPU process count is derived from the allocation;
#   * data + wandb come from env vars (see below);
#   * the venv/module setup keys off $AI_ROSSBY_CLUSTER (only DeltaAI differs).
#
# Required env:
#   AI_ROSSBY_DATA   physicsnemo-zarr root (holds era5/, era5_train/, era5_val/;
#                    see conf/dataset/era5_multiyear.yaml for the symlink-dir setup)
# Optional env:
#   AI_ROSSBY_CLUSTER  deltaai | delta | stampede3 | midway3 | dsi  (default: generic .venv)
#   AI_ROSSBY_VENV     venv dir to activate            (default: .venv, or .venv-deltaai on deltaai)
#   WANDB_PROJECT      wandb project                   (default: ai-rossby-era5)
#   WANDB_ENTITY       wandb team/user                 (default: your default entity)
#   WANDB_MODE         online | offline | disabled     (default: online)
#
# Submit (pick your cluster's partition/account):
#   Delta A100 :  AI_ROSSBY_DATA=/work/hdd/bdiu/awikner/physicsnemo-zarr \
#                 sbatch -p gpuA100x4-interactive -A bdiu-delta-gpu -t 01:00:00 \
#                 hpc/scripts/train_sfno_era5.sbatch
#   DeltaAI    :  AI_ROSSBY_CLUSTER=deltaai AI_ROSSBY_DATA=... \
#                 sbatch -p ghx4 -A bdiu-dtai-gh -t 48:00:00 hpc/scripts/train_sfno_era5.sbatch
#   Stampede3  :  AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr \
#                 sbatch -p h100 -A TG-ATM170020 -t 48:00:00 hpc/scripts/train_sfno_era5.sbatch
#
# Generic defaults below suit a single 4-GPU node; override --nodes/--gpus-per-node
# at submit for other shapes (the script reads the allocation, not these numbers).

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=train-sfno-era5
#SBATCH --output=hpc/scripts/logs/train-sfno-era5-%j.out

set -euo pipefail

# --- Repo: submit from the repo root; scripts run relative to it -------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs

# --- Environment: venv (+ module on DeltaAI) ---------------------------------
CLUSTER="${AI_ROSSBY_CLUSTER:-}"
case "${CLUSTER}" in
  deltaai)
    module load python/miniforge3_pytorch/2.10.0
    VENV="${AI_ROSSBY_VENV:-.venv-deltaai}" ;;
  *)
    VENV="${AI_ROSSBY_VENV:-.venv}" ;;
esac
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# --- Required / optional inputs ----------------------------------------------
: "${AI_ROSSBY_DATA:?set AI_ROSSBY_DATA to the physicsnemo-zarr root (see conf/dataset/era5_multiyear.yaml)}"
export AI_ROSSBY_DATA
WANDB_PROJECT="${WANDB_PROJECT:-ai-rossby-era5}"
WANDB_MODE="${WANDB_MODE:-online}"
# Empty entity => Hydra null => wandb uses your default entity.
if [ -n "${WANDB_ENTITY:-}" ]; then WANDB_ENTITY_ARG="wandb.entity=${WANDB_ENTITY}"; else WANDB_ENTITY_ARG="wandb.entity=null"; fi

# Fail early with a clear message if the train/val symlink dirs are missing.
for d in era5_train era5_val; do
  if [ ! -d "${AI_ROSSBY_DATA}/${d}" ]; then
    echo "ERROR: ${AI_ROSSBY_DATA}/${d} not present — create the per-year symlink dirs" >&2
    echo "       (see the header of conf/dataset/era5_multiyear.yaml)." >&2
    exit 1
  fi
done

# --- Distributed launch shape from the allocation ----------------------------
NPROC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
NNODES="${SLURM_NNODES:-1}"

# makani allocator fix — reclaims fragmentation so batch fits at checkpointing=2.
# Must be exported before CUDA initializes (before python launches).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || true

# Use `python -m torch.distributed.run` rather than the `torchrun` console
# script: on DeltaAI torch comes from the conda module and `torchrun` isn't on
# the venv PATH, but the module form always works when torch is importable.
if [ "${NNODES}" -gt 1 ]; then
  echo "ERROR: multi-node (${NNODES}) not wired here — add rdzv args for >1 node." >&2
  exit 1
fi

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${NPROC}" \
    --standalone \
    examples/weather/ai_rossby/train.py \
        --config-dir=examples/weather/ai_rossby/conf \
        --config-name=config \
        run_name=train_sfno_era5 \
        model=sfno_era5 \
        dataset=era5_multiyear \
        loss=raw_l2 \
        training=sfno_plasim \
        seed=0 \
        training.amp=bf16 \
        training.grad_clip_norm=1.0 \
        training.ema.enabled=True \
        training.optimizer.fused=True \
        model.checkpointing=2 \
        dataset.batch_size=1 \
        wandb.enabled=True \
        wandb.mode="${WANDB_MODE}" \
        wandb.project="${WANDB_PROJECT}" \
        "${WANDB_ENTITY_ARG}"

echo
echo "SFNO-ERA5 training launched on ${NPROC} GPU(s); wandb project=${WANDB_PROJECT} mode=${WANDB_MODE}"
