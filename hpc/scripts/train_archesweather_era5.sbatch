#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cluster-agnostic Slurm launcher for ArchesWeather-M training on the full
# multi-year ERA5 archive (conf/dataset/era5_archesweather.yaml +
# conf/model/archesweather_era5.yaml + conf/loss/archesweather.yaml +
# conf/training/archesweather.yaml).
#
# Mirrors train_sfno_era5.sbatch. NCAR Derecho uses PBS, not Slurm.
#
# Required env:
#   AI_ROSSBY_DATA   physicsnemo-zarr root (holds era5/, era5_train/, era5_val/,
#                    era5_recent/). See conf/dataset/era5_archesweather.yaml.
# Optional env:
#   AI_ROSSBY_CLUSTER   deltaai | delta | stampede3 | ...  (default: generic .venv)
#   AI_ROSSBY_VENV      venv dir                            (default: .venv / .venv-deltaai)
#   AI_ROSSBY_TRAIN_DIR train symlink-dir NAME under AI_ROSSBY_DATA (default: era5_train;
#                       set to era5_recent for the RPFT phase — resume auto-loads the
#                       latest checkpoint, and the cosine schedule continues).
#   WANDB_PROJECT / WANDB_ENTITY / WANDB_MODE   (defaults: ai-rossby-era5 / default / online)
#
# Submit:
#   Stampede3 : AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr \
#               sbatch -p h100 -A TG-ATM170020 -t 48:00:00 \
#               hpc/scripts/train_archesweather_era5.sbatch
#   RPFT phase (resume, 2007-2018):
#               AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr AI_ROSSBY_TRAIN_DIR=era5_recent \
#               sbatch -p h100 -A TG-ATM170020 -t 48:00:00 \
#               hpc/scripts/train_archesweather_era5.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=train-archesweather-era5
#SBATCH --output=hpc/scripts/logs/train-archesweather-era5-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs

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

: "${AI_ROSSBY_DATA:?set AI_ROSSBY_DATA to the physicsnemo-zarr root}"
export AI_ROSSBY_DATA
TRAIN_DIR="${AI_ROSSBY_TRAIN_DIR:-era5_train}"
WANDB_PROJECT="${WANDB_PROJECT:-ai-rossby-era5}"
WANDB_MODE="${WANDB_MODE:-online}"
if [ -n "${WANDB_ENTITY:-}" ]; then WANDB_ENTITY_ARG="wandb.entity=${WANDB_ENTITY}"; else WANDB_ENTITY_ARG="wandb.entity=null"; fi

for d in "${TRAIN_DIR}" era5_val; do
  if [ ! -d "${AI_ROSSBY_DATA}/${d}" ]; then
    echo "ERROR: ${AI_ROSSBY_DATA}/${d} not present — create the per-year symlink dirs" >&2
    echo "       (see the header of conf/dataset/era5_archesweather.yaml)." >&2
    exit 1
  fi
done

NPROC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
NNODES="${SLURM_NNODES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi || true

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
        run_name=train_archesweather_era5 \
        model=archesweather_era5 \
        dataset=era5_archesweather \
        loss=archesweather \
        training=archesweather \
        seed=0 \
        dataset.zarr_path="${AI_ROSSBY_DATA}/${TRAIN_DIR}" \
        dataset.batch_size=1 \
        wandb.enabled=True \
        wandb.mode="${WANDB_MODE}" \
        wandb.project="${WANDB_PROJECT}" \
        "${WANDB_ENTITY_ARG}"

echo
echo "ArchesWeather-ERA5 training launched on ${NPROC} GPU(s) (train dir=${TRAIN_DIR}); wandb project=${WANDB_PROJECT} mode=${WANDB_MODE}"
