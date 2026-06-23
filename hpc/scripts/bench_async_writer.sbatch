#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Async-vs-sync forecast writer benchmark on a single A40.
# Runs benchmarks/.../async_writer/run_bench.py against the translated
# Pangu_Plasim S2S 2000 checkpoint + ERA5 1981 S2S Zarr.

#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --job-name=pn-bench-async
#SBATCH --output=hpc/scripts/logs/bench-async-writer-%j.out

set -euo pipefail
cd /work/nvme/bdiu/awikner/physicsnemo

source .venv/bin/activate

MDLUS=/work/hdd/bdiu/awikner/translated_checkpoints/pangu_plasim_s2s_2000.mdlus
MODEL_YAML=examples/weather/ai_rossby/conf/model/pangu_plasim_s2s.yaml
ZARR=/work/hdd/bdiu/awikner/physicsnemo-zarr/era5_sfno_s2s/1981.zarr
MEAN=/work/hdd/bdiu/awikner/physicsnemo-zarr/era5/normalization_pangu_s2s_mean.zarr
STD=/work/hdd/bdiu/awikner/physicsnemo-zarr/era5/normalization_pangu_s2s_std.zarr

OUT=/work/hdd/bdiu/awikner/sfno_bench/async_writer_${SLURM_JOB_ID}
REPORT=hpc/scripts/logs/bench-async-writer-${SLURM_JOB_ID}.json

python benchmarks/physicsnemo/experimental/examples/ai_rossby/async_writer/run_bench.py \
    --mdlus "${MDLUS}" \
    --model-yaml "${MODEL_YAML}" \
    --zarr  "${ZARR}" \
    --mean  "${MEAN}" \
    --std   "${STD}" \
    --output-dir "${OUT}" \
    --n-ic 6 \
    --max-step 20 \
    --ic-stride 80 \
    --warmup-ics 1 \
    --report "${REPORT}"

echo
echo "=== output dir: ${OUT}"
echo "=== report:    ${REPORT}"
