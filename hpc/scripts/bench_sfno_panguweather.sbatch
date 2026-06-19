#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PanguWeather v2.0 side of the SFNO_PLASIM_S2S benchmark. Trains 1 epoch
# on ERA5 1981 using the original PanguWeather code + the SFNO_S2S_0003_test
# config, for head-to-head comparison with ai-rossby's SfnoPlasim.
#
# Captures stdout (which includes per-batch `Loss: X.XXXX` lines from tqdm)
# so the comparison script can parse the loss curve.
#
# Pairs with hpc/scripts/bench_sfno_ai_rossby.sbatch.
#
# Submit with:
#   sbatch hpc/scripts/bench_sfno_panguweather.sbatch

#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=bench-sfno-panguweather
#SBATCH --output=hpc/scripts/logs/bench-sfno-panguweather-%j.out

set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_MODE=offline

# Activate PanguWeather v2.0's conda env (per the upstream submission
# scripts at PanguWeather/v2.0/submit_scripts/deltagpu/).
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh
conda activate /projects/bdiu/awikner/.conda/envs/aires_panguplasim_cu126

# Use the absolute path — srun subprocesses don't inherit conda's PATH mods.
TORCHRUN=/projects/bdiu/awikner/.conda/envs/aires_panguplasim_cu126/bin/torchrun

PW_DIR=/work/nvme/bdiu/awikner/PanguWeather/v2.0
cd "${PW_DIR}"

nvidia-smi

NNODES=${SLURM_JOB_NUM_NODES}
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES=${NNODES} NUM_TASKS_PER_NODE=${NUM_TASKS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12347     # avoid colliding with ai-rossby's job if both run on same node
export WORLD_SIZE
export OMP_NUM_THREADS=1

# Override SFNO_S2S_0003_test.yaml's bf16 default with fp32 (--no_amp).
# Disable ZeRO so both code paths use the same optimizer.
# --epochs=1 → single epoch over 1981.
# --global_seed 0 → matches ai-rossby.
# --debug → disables wandb (sets WANDB_MODE=offline).
# Omit --with_profiling → no nsys overhead.
srun --ntasks=1 --gpus-per-task=${NUM_TASKS_PER_NODE} \
    "${TORCHRUN}" \
        --nproc_per_node=${NUM_TASKS_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        train.py \
            --yaml_config="${PW_DIR}/config/SFNO_S2S_0003_test.yaml" \
            --config=SFNOS2S \
            --epochs=1 \
            --no_amp \
            --use_zero_optimizer false \
            --global_seed 0 \
            --debug

# The SLURM stdout (hpc/scripts/logs/bench-sfno-panguweather-${SLURM_JOB_ID}.out)
# contains the tqdm `Loss: ` lines that benchmarks/.../sfno_plasim/compare.py
# parses for the per-batch loss curve.
echo
echo "PanguWeather stdout: hpc/scripts/logs/bench-sfno-panguweather-${SLURM_JOB_ID}.out"
