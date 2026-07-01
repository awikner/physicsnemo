#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 8f (F7) convergence smoke — 10 epochs of the AMIP SI diffusion
# recipe on the 1981 AMIP Zarr, 2x A40 interactive. The Phase 8c wiring
# smoke (smoke_amip_diffusion_2xA40.sbatch) only ran 1 epoch to verify
# the pack/unpack + loss + EMA scaffolding end-to-end — it observed a
# loss around ~5e6 without checking whether it actually decreases. This
# job runs long enough (10 epochs, Muon optimizer per Phase 8f F1) to
# confirm the loss trends down, and dumps a per-batch TSV +
# matplotlib PNG for benchmarks/physicsnemo/experimental/models/amip_si/CONVERGENCE.md.
#
# Pairs with:
#   conf/model/amip_si.yaml             — AmipDiTWrapper around AmipDiT
#   conf/loss/si.yaml                   — DriftScheduler (SI baseline)
#   conf/dataset/amip_1981.yaml         — 1981 AMIP Zarr
#   conf/training/amip_diffusion.yaml   — Muon + EMA 0.999 + fp32
#
# Prereq: AMIP 1981 conversion job has succeeded, and
#   uv pip install git+https://github.com/KellerJordan/Muon
# has been run in the venv (Phase 8f F1 — this job uses the real Muon
# optimizer, not the AdamW override the wiring smoke used).

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=amip-diff-convergence
#SBATCH --output=hpc/scripts/logs/amip-diff-convergence-%j.out

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
source .venv/bin/activate

module load aws-ofi-nccl/1.14.2

mkdir -p hpc/scripts/logs

ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
if [[ ! -d "$ZARR_PATH" ]]; then
    echo "ERROR: $ZARR_PATH missing — wait for the AMIP 1981 conversion job"
    exit 1
fi

OUT_DIR=/work/hdd/bdiu/awikner/amip_si_convergence
mkdir -p "${OUT_DIR}"

cd examples/weather/ai_rossby

# Same shrunk-backbone sizing as the wiring smoke + bf16 bench (fits
# 2x A40 at batch=2); 10 epochs, no rollout validation (this job is
# purely about the training-loss curve).
torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    model=amip_si \
    loss=si \
    dataset=amip_1981 \
    training=amip_diffusion \
    validation=off \
    training.max_epochs=10 \
    training.ema.warmup_epochs=0 \
    'model.dit_kwargs.dim=128' \
    'model.dit_kwargs.num_blocks=4' \
    'model.dit_kwargs.num_heads=4' \
    'model.dit_kwargs.patch_size=4' \
    'model.dit_kwargs.c_grid_downsample=1' \
    dataset.batch_size=2 \
    dataset.num_workers=4 \
    log_every_n_steps=20 \
    run_name="amip-diff-convergence-$SLURM_JOB_ID" \
    +checkpoint_dir="./checkpoints_convergence_$SLURM_JOB_ID" \
    bench.per_batch_tsv="${OUT_DIR}/loss_${SLURM_JOB_ID}.tsv"

echo
echo "=== convergence TSV: ${OUT_DIR}/loss_${SLURM_JOB_ID}.tsv"

python "${REPO_ROOT}/benchmarks/physicsnemo/experimental/models/amip_si/plot_convergence.py" \
    --tsv "${OUT_DIR}/loss_${SLURM_JOB_ID}.tsv" \
    --output "${OUT_DIR}/convergence_${SLURM_JOB_ID}.png"

echo "=== convergence plot: ${OUT_DIR}/convergence_${SLURM_JOB_ID}.png"
