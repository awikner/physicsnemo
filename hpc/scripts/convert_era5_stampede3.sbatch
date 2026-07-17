#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the full ERA5 archive (1979-2024) on TACC Stampede3 (`spr` queue) → $SCRATCH,
# plus the Pangu-S2S normalization Zarrs. Stampede3 sibling of
# convert_era5_full_archive.sbatch (Delta), extended 2018 -> 2024. skip-if-exists.
# Copy the output to Derecho (master) afterward — see phase-11 §11b.
#
# Prereq: raw H5 staged to $RAW ({year}_{idx:04d}.h5) via Globus from Delta.
# Submit:  sbatch hpc/scripts/convert_era5_stampede3.sbatch

#SBATCH -p spr
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH -t 36:00:00
#SBATCH -J pn-conv-era5
#SBATCH -o convert-era5-%j.out

set -euo pipefail
cd "$WORK/physicsnemo"
source .venv/bin/activate

# Size the per-file H5 reader pool to the whole node (TACC allocates
# whole nodes; without this the pool can bind to a single core).
export SLURM_CPUS_PER_TASK=$(nproc)
echo "[$(date)] $(hostname): $(nproc) cores"

RAW=${RAW:-$SCRATCH/raw/era5/h5data}
OUT_ROOT=${OUT_ROOT:-$SCRATCH/physicsnemo-zarr/era5}
mkdir -p "$OUT_ROOT"

# 1. Per-year prognostic Zarrs.
for YEAR in $(seq 1979 2024); do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping ERA5 year $YEAR (already exists)"
        continue
    fi
    echo "[$(date)] converting ERA5 year $YEAR ..."
    python tools/data/era5/pangu_h5_to_zarr.py \
        --input-dir "$RAW" \
        --year "$YEAR" \
        --output "$OUT_PATH"
done

# 2. Normalization Zarr. Use the pangu_s2s_Z200 variant: it carries the COMPLETE
# 18 pressure levels (incl. 200 hPa) so the stats match the 18-level ERA5 Zarr
# dataset. The plain `pangu_s2s` variant drops 200 hPa (17 levels) — that's the
# SFNO_S2S_0003 17-level ablation and would make the by-value ClimateNormalizer
# raise on the full 18-level data.
echo "[$(date)] building ERA5 normalization Zarr (pangu_s2s, 18-level incl 200 hPa) ..."
python tools/data/era5/build_normalization_zarr.py \
    --source-dir "$RAW" \
    --variant pangu_s2s_Z200 \
    --output "$OUT_ROOT/normalization_pangu_s2s.zarr" \
    --overwrite

echo "[$(date)] ERA5 conversion complete."
ls -la "$OUT_ROOT"
