#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the E3SM archive (2015-2049) on TACC Stampede3 (`spr` CPU queue) → $SCRATCH.
# Stampede3 sibling of convert_e3sm_full_archive.sbatch (Delta). skip-if-exists,
# so re-submitting fills gaps (2041-2049 already exist elsewhere; this converts
# 2015-2040). Copy the output to Derecho (master) afterward — see phase-11 §11b.
#
# Prereq: raw H5 staged to $RAW (Globus from Delta), preserving the source layout
#   $RAW/h5/sigma_data/{year}_{idx:04d}.h5 , $RAW/sigma_data/climatology.nc , $RAW/bias/
#
# Submit:  sbatch hpc/scripts/convert_e3sm_stampede3.sbatch

#SBATCH -p spr
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH -t 24:00:00
#SBATCH -J pn-conv-e3sm
#SBATCH -o convert-e3sm-%j.out

set -euo pipefail
cd "$WORK/physicsnemo"
source .venv/bin/activate

# Size the per-file H5 reader pool to the whole node (TACC allocates
# whole nodes; without this the pool can bind to a single core).
export SLURM_CPUS_PER_TASK=$(nproc)
echo "[$(date)] $(hostname): $(nproc) cores"

RAW=${RAW:-$SCRATCH/raw/e3sm}
H5_DIR="$RAW/h5/sigma_data"
STATS_DIR="$H5_DIR"
CLIM_PATH="$H5_DIR/climatology.nc"           # climatology.nc lives inside h5/sigma_data/
BIAS_DIR="$RAW/bias"
OUT_ROOT=${OUT_ROOT:-$SCRATCH/physicsnemo-zarr/e3sm}
mkdir -p "$OUT_ROOT"

# 1. Per-year prognostic Zarrs.
for YEAR in $(seq 2015 2049); do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping E3SM year $YEAR (already exists)"
        continue
    fi
    echo "[$(date)] converting E3SM year $YEAR ..."
    python tools/data/e3sm/pangu_h5_to_zarr.py \
        --input-dir "$H5_DIR" \
        --year "$YEAR" \
        --output "$OUT_PATH"
done

# 2. Normalization Zarr (only if the source stats were staged).
if ls "$STATS_DIR"/data_*_mean.nc >/dev/null 2>&1; then
    echo "[$(date)] building E3SM normalization Zarr ..."
    python tools/data/e3sm/build_normalization_zarr.py \
        --source-dir "$STATS_DIR" \
        --output "$OUT_ROOT/normalization_2015-2050.zarr" \
        --overwrite
else
    echo "[$(date)] WARN: no data_*_mean.nc under $STATS_DIR — skipping normalization Zarr"
fi

# 3. Climatology + bias Zarr (only if both were staged).
if [ -f "$CLIM_PATH" ] && [ -d "$BIAS_DIR" ]; then
    echo "[$(date)] building E3SM climatology+bias Zarr ..."
    python tools/data/e3sm/build_climatology_zarr.py \
        --climatology "$CLIM_PATH" \
        --bias-dir    "$BIAS_DIR" \
        --output      "$OUT_ROOT/climatology_bias.zarr" \
        --overwrite
else
    echo "[$(date)] WARN: missing $CLIM_PATH or $BIAS_DIR — skipping climatology Zarr"
fi

echo "[$(date)] E3SM conversion complete."
ls -la "$OUT_ROOT"
