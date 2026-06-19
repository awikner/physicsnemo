#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ai-rossby side of the SFNO_PLASIM_S2S vs. PanguWeather v2.0 benchmark.
# Trains 1 epoch on ERA5 1981 (single per-year Zarr), captures per-batch
# loss + wall time into a TSV that the comparison script consumes.
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

# Verify the SFNO_S2S re-converted 1981 Zarr exists; if not, halt with a
# clear message — the comparison script downstream can't run without it.
ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/era5_sfno_s2s/1981.zarr
if [ ! -d "${ZARR_PATH}" ]; then
    echo "ERROR: ${ZARR_PATH} not present." >&2
    echo "Run the converter once with:" >&2
    echo "  python tools/data/era5/pangu_h5_to_zarr.py --year 1981 \\" >&2
    echo "    --channel-config benchmarks/physicsnemo/experimental/models/sfno_plasim/era5_sfno_s2s_channels.json \\" >&2
    echo "    --output ${ZARR_PATH}" >&2
    exit 1
fi

nvidia-smi

# All overrides are passed as Hydra CLI args (no extra config files needed).
# Match SFNO_S2S_0003_test.yaml exactly: 17 levels, embed_dim=512, num_layers=12,
# num_blocks=16, etc. (in conf/model/sfno_plasim_s2s.yaml).
# Pair with scheduler=cosine_warmup, loss=raw_l2.
torchrun \
    --nnodes=1 \
    --nproc-per-node=4 \
    --standalone \
    examples/weather/ai_rossby/train.py \
        --config-dir=examples/weather/ai_rossby/conf \
        --config-name=config \
        run_name=bench_sfno_ai_rossby \
        model=sfno_plasim_s2s \
        loss=raw_l2 \
        scheduler=cosine_warmup \
        amp=none \
        seed=0 \
        max_epochs=1 \
        grad_clip_norm=0.0 \
        ema.enabled=False \
        data.zarr_path=${ZARR_PATH} \
        data.mean_path=/work/hdd/bdiu/awikner/physicsnemo-zarr/era5/normalization_pangu_s2s.zarr \
        data.std_path=/work/hdd/bdiu/awikner/physicsnemo-zarr/era5/normalization_pangu_s2s.zarr \
        data.batch_size=4 \
        data.num_workers=4 \
        data.prefetch_factor=2 \
        data.persistent_workers=True \
        data.pin_memory=True \
        data.shuffle=True \
        data.forecast_lead_times=[1] \
        +bench.per_batch_tsv=${OUT_DIR}/ai_rossby_${SLURM_JOB_ID}.tsv

echo
echo "ai-rossby benchmark TSV: ${OUT_DIR}/ai_rossby_${SLURM_JOB_ID}.tsv"
