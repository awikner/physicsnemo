#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 8c GPU smoke — runs the AMIP SI diffusion recipe through one
# mini-epoch on the 1981 AMIP Zarr archive on a 2xA40 interactive node.
# Verifies the full pack/unpack + diffusion-loss + EMA + multi-stage
# scaffolding end-to-end on real data, real GPU, and DDP launch.
#
# Pairs with:
#   conf/model/amip_si.yaml             — AmipDiTWrapper around AmipDiT (45x90 coarse state)
#   conf/loss/si.yaml                   — DriftScheduler (SI baseline)
#   conf/dataset/amip_1981.yaml         — 1981 AMIP Zarr
#   conf/training/amip_diffusion.yaml   — Muon + EMA 0.999 + fp32
#   conf/validation/diffusion_rollout.yaml — every-5-epoch rollout
#
# Prereq: AMIP 1981 conversion job has succeeded and the Zarr at
#   /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
# is present. Tracked in conversation as job 19640283.

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100g
#SBATCH --job-name=amip-diff-smoke
#SBATCH --output=hpc/scripts/logs/amip-diff-smoke-%j.out

# NOTE (2026-08-17): dataset moved amip_1981 -> amip_dailyavg_coarse with the
# amip_si config reassignment. amip_si now describes the REAL v1 SI checkpoints
# (151 channels on a 45x90 backbone grid with 1-degree forcings); the previous
# 161-channel/180x360 config it replaced was deleted. Any throughput or loss
# figures recorded from earlier runs of this script are NOT comparable.

set -euo pipefail

REPO_ROOT="${AI_ROSSBY_REPO:-/work/nvme/bdiu/awikner/physicsnemo}"
cd "$REPO_ROOT"

# Runs in the ai-rossby container (hpc/containers/) rather than a per-cluster
# venv; container_run.sh resolves the .sif, bind mounts and env hygiene.
# No aws-ofi-nccl module load: that plugin only matters for multi-NODE NCCL over
# the host fabric, and this is a single-node 2-GPU run using NVLink/PCIe, served
# by the container's own NCCL.
CONTAINER_RUN="${REPO_ROOT}/hpc/scripts/container_run.sh"

mkdir -p hpc/scripts/logs

# Confirm the conversion output is on disk before we burn GPU minutes.
ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
if [[ ! -d "$ZARR_PATH" ]]; then
    echo "ERROR: $ZARR_PATH missing — wait for the AMIP 1981 conversion job"
    exit 1
fi

cd examples/weather/ai_rossby

# Knobs:
# - model=amip_si               → AmipDiTWrapper (single-step diffusion)
# - loss=si                     → DriftScheduler (SI baseline)
# - dataset=amip_dailyavg_coarse           → 1981 AMIP Zarr
# - training=amip_diffusion     → Muon + EMA + fp32 + cosine (Phase 8f F1: real
#                                 MuonWithAuxAdam via AmipDiTWrapper.muon_param_groups())
# - validation=diffusion_rollout→ 5-epoch cadence rollout val + spread metric
# - training.max_epochs=1       → cap to one epoch for the smoke
# - validation.every_n_epochs=1 → exercise the validator path inside this 1 epoch
# - training.ema.warmup_epochs=0→ no need to warm EMA in a 1-epoch smoke
# - dit_kwargs.dim=128          → shrink the backbone to fit two A40s with batch=2
# - dit_kwargs.num_blocks=4     → ditto
# - patch_size=4                → coarser patch keeps token count tractable at 180×360
# - c_grid_downsample stays 4   → the coarse store pairs 45x90 state with 1-degree forcings
#
# Phase 8f F1: this smoke exercises the real Muon optimizer, not an AdamW
# override. Muon no longer needs a manual install — the `muon-optimizers` extra
# is part of the container image's locked closure.
# `python -m torch.distributed.run` rather than the `torchrun` console script,
# matching the rest of hpc/scripts (portable across the DeltaAI layout where
# torchrun is absent from the interpreter's PATH).
#
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
"$CONTAINER_RUN" python -m torch.distributed.run \
    --standalone --nproc-per-node=2 train_diffusion.py \
    model=amip_si \
    loss=si \
    dataset=amip_dailyavg_coarse \
    training=amip_diffusion \
    validation=diffusion_rollout \
    training.max_epochs=1 \
    training.ema.warmup_epochs=0 \
    validation.every_n_epochs=1 \
    validation.rollout.max_initial_conditions=2 \
    validation.rollout.log_steps='[1]' \
    'model.dit_kwargs.dim=128' \
    'model.dit_kwargs.num_blocks=4' \
    'model.dit_kwargs.num_ca_blocks=2' \
    'model.dit_kwargs.num_output_blocks=1' \
    'model.dit_kwargs.num_heads=4' \
    'model.dit_kwargs.patch_size=4' \
    dataset.batch_size=2 \
    dataset.num_workers=4 \
    log_every_n_steps=5 \
    run_name="amip-diff-smoke-$SLURM_JOB_ID" \
    +checkpoint_dir="./checkpoints_smoke_$SLURM_JOB_ID"

echo
echo "=== AMIP diffusion smoke test passed for SLURM job $SLURM_JOB_ID"
