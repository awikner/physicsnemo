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
#   conf/model/amip_si.yaml             — AmipDiTWrapper around AmipDiT (45x90 coarse state)
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

# NOTE (2026-08-17): dataset moved amip_1981 -> amip_dailyavg_coarse with the
# amip_si config reassignment. amip_si now describes the REAL v1 SI checkpoints
# (151 channels on a 45x90 backbone grid with 1-degree forcings); the previous
# 161-channel/180x360 config it replaced was deleted. Any throughput or loss
# figures recorded from earlier runs of this script are NOT comparable.

set -euo pipefail

REPO_ROOT=/work/nvme/bdiu/awikner/physicsnemo
cd "$REPO_ROOT"
# Runs in the ai-rossby container (hpc/containers/) instead of a per-cluster venv.
# Absolute, captured here where cwd is the repo root — the same assumption the
# replaced `source .venv/bin/activate` already made — so it survives later cd.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"

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
# Two constraints the overrides below encode (these used to sit inline as
# comments between continuation lines, which silently TERMINATED the command —
# bash does not allow a comment inside a `\`-continuation, so torchrun ran
# truncated and the remaining args executed as a bogus command):
#   - num_ca_blocks must be <= num_blocks (AmipDiT asserts it) and the config
#     ships 8 to match the real checkpoints, so it is shrunk alongside.
#   - c_grid_downsample stays at the config's 4: amip_dailyavg_coarse pairs a
#     45x90 state with 1-degree forcings and the backbone's stride-4 conv is
#     what reconciles them. Forcing 1 builds a 1x1 conv and then fails the
#     concat on a 180x360-vs-45x90 spatial mismatch.
"$CONTAINER_RUN" torchrun --standalone --nproc-per-node=2 train_diffusion.py \
    model=amip_si \
    loss=si \
    dataset=amip_dailyavg_coarse \
    training=amip_diffusion \
    validation=off \
    training.max_epochs=10 \
    training.ema.warmup_epochs=0 \
    'model.dit_kwargs.dim=128' \
    'model.dit_kwargs.num_blocks=4' \
    'model.dit_kwargs.num_ca_blocks=2' \
    'model.dit_kwargs.num_output_blocks=1' \
    'model.dit_kwargs.num_heads=4' \
    'model.dit_kwargs.patch_size=4' \
    dataset.batch_size=2 \
    dataset.num_workers=4 \
    log_every_n_steps=20 \
    run_name="amip-diff-convergence-$SLURM_JOB_ID" \
    +checkpoint_dir="./checkpoints_convergence_$SLURM_JOB_ID" \
    bench.per_batch_tsv="${OUT_DIR}/loss_${SLURM_JOB_ID}.tsv"

echo
echo "=== convergence TSV: ${OUT_DIR}/loss_${SLURM_JOB_ID}.tsv"

"$CONTAINER_RUN" python "${REPO_ROOT}/benchmarks/physicsnemo/experimental/models/amip_si/plot_convergence.py" \
    --tsv "${OUT_DIR}/loss_${SLURM_JOB_ID}.tsv" \
    --output "${OUT_DIR}/convergence_${SLURM_JOB_ID}.png"

echo "=== convergence plot: ${OUT_DIR}/convergence_${SLURM_JOB_ID}.png"
