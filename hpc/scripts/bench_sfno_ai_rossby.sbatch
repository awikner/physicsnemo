#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ai-rossby side of the SFNO_PLASIM_5412 vs. PanguWeather v2.0 benchmark.
# Trains 1 epoch on PLASIM sim52 year 12 (1464 6-hourly steps at 64x128,
# legendre-gauss grid), captures per-batch loss + wall time into a TSV
# that the comparison script consumes.
#
# Pairs with hpc/scripts/bench_sfno_panguweather.sbatch (one per code path).
# Both jobs run on Delta gpuA100x4-interactive (4 A100s, 1-hr cap, account
# bdiu-delta-gpu) for apples-to-apples wall-clock comparison.
#
# Submit with:
#   sbatch hpc/scripts/bench_sfno_ai_rossby.sbatch

#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=bench-sfno-ai-rossby
#SBATCH --output=hpc/scripts/logs/bench-sfno-ai-rossby-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

# Per-batch loss output dir on HDD scratch.
OUT_DIR=/work/hdd/bdiu/awikner/sfno_bench
mkdir -p "${OUT_DIR}"

# PLASIM year 12 single-year zarr (sim52, 13 pres + 10 sigma levels at 64x128).
ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim/12.zarr
if [ ! -d "${ZARR_PATH}" ]; then
    echo "ERROR: ${ZARR_PATH} not present." >&2
    exit 1
fi
# Stats live next to the H5 source; PanguWeather + ai-rossby both read these.
PLASIM_STATS_DIR=/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data
MEAN_PATH=${PLASIM_STATS_DIR}/data_12-132_mean_sigma.nc
STD_PATH=${PLASIM_STATS_DIR}/data_12-132_std_sigma.nc

nvidia-smi

# All overrides are passed as Hydra CLI args (no extra config files needed).
# Matches SFNO_PLASIM_H5_DERECHO_5412_test.yaml SFNO block exactly via
# conf/model/sfno_plasim_5412.yaml + conf/training/sfno_plasim.yaml.
# dataset.* overrides point at the prebuilt year-12 zarr.
torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --standalone \
    examples/weather/ai_rossby/train.py \
        --config-dir=examples/weather/ai_rossby/conf \
        --config-name=config \
        run_name=bench_sfno_ai_rossby \
        model=sfno_plasim_5412 \
        loss=raw_l2 \
        training=sfno_plasim \
        validation=off \
        dataset=plasim_sim52_year12 \
        seed=0 \
        training.max_epochs=1 \
        training.stages.0.num_epochs=1 \
        training.amp=none \
        training.grad_clip_norm=0.0 \
        training.ema.enabled=False \
        dataset.zarr_path=${ZARR_PATH} \
        dataset.mean_path=${MEAN_PATH} \
        dataset.std_path=${STD_PATH} \
        dataset.normalize_constant_boundary=True \
        dataset.normalize_diagnostic=True \
        dataset.batch_size=8 \
        dataset.num_workers=8 \
        dataset.prefetch_factor=2 \
        dataset.persistent_workers=True \
        dataset.pin_memory=True \
        dataset.shuffle=True \
        dataset.forecast_lead_times=[1] \
        bench.per_batch_tsv=${OUT_DIR}/ai_rossby_${SLURM_JOB_ID}.tsv

echo
echo "ai-rossby benchmark TSV: ${OUT_DIR}/ai_rossby_${SLURM_JOB_ID}.tsv"
