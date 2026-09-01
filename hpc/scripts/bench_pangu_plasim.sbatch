#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optimized PanguPlasim training/bench config — result of transferring the
# SFNO-E3SM speed-optimization levers to the Pangu models. Runs
# PanguPlasimLegacy on PLASIM sim52 year-12 (64x128), global batch 16 (4/GPU),
# on a single Delta gpuA100x4-interactive node.
#
# Winning config vs. the fp32 baseline (0.1235 s/step, 129.6 samples/sec):
#   bf16 AMP + fused AdamW  ->  0.1024 s/step, 156.2 samples/sec (~1.21x).
#   (bf16 alone = 0.1041 s/step, 1.19x; fused adds ~1.6% on top.)
#
# Which SFNO levers transferred, and which did NOT:
#   - bf16 AMP: BIG win here (1.19x) — much larger than SFNO's 1.05%, because
#     Pangu is a GEMM-heavy Swin (Earth-Specific) transformer that maps onto
#     A100 tensor cores far better than SFNO's SHT/FFT-bound compute.
#   - fused AdamW: small win (~1.6%), stacked on bf16. Both were simply OFF in
#     the shipped configs (amp=none, fused=null->False) — free wins left on the
#     table.
#   - expandable_segments -> lower checkpointing (the SFNO "big win", ~1.23x):
#     DOES NOT TRANSFER. Pangu's `checkpointing` flag only gates the tiny
#     patch-recovery heads (the transformer backbone is never checkpointed),
#     levels 2/3 are identical to 1 in code, and the model already fits at
#     checkpointing=0 at PLASIM 64x128. There is no graduated recompute ladder
#     to descend and no fragmentation-forced OOM. So Pangu's speedup is a pure
#     compute-path result (~1.21x), not the memory-path 1.39x SFNO got.
#   - torch.compile: NOT viable on Delta — torch-inductor's C++ codegen fails
#     at link (`ld: undefined reference to 'main'` under gcc-toolset-13); it was
#     only ~1.3% for SFNO anyway. jit_compile stays False.
#
# The PanguPlasim VAE variant (model=pangu_plasim) gets the same bf16+fused
# config (measured on 1 GPU), but note it currently needs
# find_unused_parameters=True to train under multi-GPU DDP (its dual-encoder /
# VAE-head graph leaves some params ungradiented) — a pre-existing gap, not
# addressed here; this benchmark uses the DDP-clean Legacy model.
#
# Submit with:
#   sbatch hpc/scripts/bench_pangu_plasim.sbatch

#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=bench-pangu-plasim
#SBATCH --output=hpc/scripts/logs/bench-pangu-plasim-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
# Runs in the ai-rossby container (hpc/containers/) instead of a per-cluster venv.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"

OUT_DIR=/work/hdd/bdiu/awikner/sfno_bench
mkdir -p "${OUT_DIR}"

ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim/12.zarr
if [ ! -d "${ZARR_PATH}" ]; then
    echo "ERROR: ${ZARR_PATH} not present." >&2
    exit 1
fi
PLASIM_STATS_DIR=/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data
MEAN_PATH=${PLASIM_STATS_DIR}/data_12-132_mean_sigma.nc
STD_PATH=${PLASIM_STATS_DIR}/data_12-132_std_sigma.nc

nvidia-smi

# NOTE: the committed pangu_plasim_legacy.yaml lists 13 pressure levels, but the
# year-12 zarr has only 10 (5000/10000/15000 Pa absent). Override model.levels
# to the 10 present levels (same workaround as smoke_pangu_plasim_native_2xA40).
"$CONTAINER_RUN" torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --standalone \
    examples/weather/ai_rossby/train.py \
        --config-dir=examples/weather/ai_rossby/conf \
        --config-name=config \
        run_name=bench_pangu_plasim \
        model=pangu_plasim_legacy \
        loss=mae \
        training=pangu_plasim_legacy \
        validation=off \
        dataset=plasim_sim52_year12 \
        seed=0 \
        'model.levels=[20000,25000,30000,40000,50000,60000,70000,85000,92500,100000]' \
        model.checkpointing=0 \
        training.ema.enabled=False \
        training.grad_clip_norm=0.0 \
        dataset.zarr_path=${ZARR_PATH} \
        dataset.mean_path=${MEAN_PATH} \
        dataset.std_path=${STD_PATH} \
        dataset.batch_size=4 \
        dataset.num_workers=8 \
        bench.per_batch_tsv=${OUT_DIR}/bench_pangu_plasim_${SLURM_JOB_ID}.tsv

echo
echo "PanguPlasim benchmark TSV: ${OUT_DIR}/bench_pangu_plasim_${SLURM_JOB_ID}.tsv"
