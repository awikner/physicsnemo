#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fit the RSI spectral envelope g(l) on the AMIP daily-avg coarse archive.
# CPU-only (a band-averaged spectrum over a few thousand increment pairs); this
# is the prerequisite artifact for ablation rung a4.
#
# g(l) is the per-degree amplitude spectrum of the model's own one-step
# increment in NORMALIZED units, unit band-mean, so it REDISTRIBUTES latent
# amplitude across scales rather than rescaling it (the level stays gamma_0's
# job). Note it is the INCREMENT spectrum, not the conditional spread — at large
# scales it is dominated by deterministic advection. That is still the right
# default (it is the scale on which the entering slot's anchor is wrong), but do
# not read it as uncertainty; a narrower spread-fit envelope is a separate
# ablation and drops into the same --out slot.
#
# Bandwidth is nlat//2, the exactness limit of equiangular quadrature. Do not
# raise it: at lmax = nlat the analysis/synthesis pair is not even a projection,
# and since Gamma and Gamma^{-1} come from that same pair the score identity
# fails silently. The tool and the filter default identically and must agree.
#
# Submit (Stampede3):
#   AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr \
#   sbatch -p spr -A TG-ATM170020 -t 02:00:00 hpc/scripts/fit_rsi_spectrum.sbatch
#
# Then hand the artifact to rung a4:
#   RSI_RUNG=a4 RSI_SPECTRUM=$AI_ROSSBY_DATA/norm_stats/g_l_amip_dailyavg_coarse.pt \
#   sbatch -p h100 -A TG-ATM170020 -t 48:00:00 hpc/scripts/train_rsi_amip.sbatch

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=fit-rsi-spectrum
#SBATCH --output=hpc/scripts/logs/fit-rsi-spectrum-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs

# Runs in the ai-rossby container (hpc/containers/). container_run.sh detects the
# cluster, resolves the .sif and bind mounts, and scrubs the host toolchain env —
# which is what retires the DeltaAI `module load python/miniforge3_pytorch` +
# separate-venv special case that used to live here.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"

: "${AI_ROSSBY_DATA:?set AI_ROSSBY_DATA to the physicsnemo-zarr root}"
STORE="${RSI_STORE:-${AI_ROSSBY_DATA}/amip_dailyavg_coarse}"
STD="${RSI_STD:-${STORE}/normalize_std_dailyavg.nc}"
OUT="${RSI_OUT:-${AI_ROSSBY_DATA}/norm_stats/g_l_amip_dailyavg_coarse.pt}"
# Fit on the TRAINING years only — the envelope is a model hyperparameter, so
# fitting it through the evaluation period would leak.
Y0="${RSI_YEAR_START:-1979}"
Y1="${RSI_YEAR_END:-2015}"

mkdir -p "$(dirname "${OUT}")"

"$CONTAINER_RUN" python tools/data/amip/make_rsi_spectrum.py \
    --zarr "${STORE}" \
    --model-config examples/weather/ai_rossby/conf/model/amip_rsi_v2.yaml \
    --std "${STD}" \
    --year-start "${Y0}" --year-end "${Y1}" \
    --out "${OUT}"

echo
echo "g(l) written to ${OUT}; provenance sidecar alongside it."
echo "Use with: RSI_RUNG=a4 RSI_SPECTRUM=${OUT} sbatch ... hpc/scripts/train_rsi_amip.sbatch"
