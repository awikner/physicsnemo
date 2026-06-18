#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the full PLASIM sim52 archive (years 12–132 = 121 years) to the
# unified ai-rossby per-year Zarr format. Submits one sbatch job that loops
# the per-year H5→Zarr converter; per-file H5 reads inside the converter
# parallelize via SLURM_CPUS_PER_TASK.
#
# Also runs the normalization and climatology+bias converters at the end so
# the full output dir is self-contained.
#
# Output: /work/hdd/bdiu/awikner/physicsnemo-zarr/plasim/
#   - {12..132}.zarr           — per-year prognostic stores
#   - normalization_12-132.zarr — per-variable mean+std
#   - climatology_bias.zarr     — daily climatology + bias arrays
#
# Submit with:
#   sbatch hpc/scripts/convert_plasim_full_archive.sbatch

#SBATCH --partition=cpu
#SBATCH --account=bdiu-delta-cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --job-name=pn-conv-plasim
#SBATCH --output=hpc/scripts/logs/convert-plasim-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

OUT_ROOT=/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim
mkdir -p "$OUT_ROOT"

SOURCE_DIR=/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52
H5_DIR="$SOURCE_DIR/h5/sigma_data"
STATS_DIR="$SOURCE_DIR/h5/sigma_data"
CLIM_DIR="$SOURCE_DIR/sigma_data"
BIAS_DIR="$SOURCE_DIR/bias"

# 1. Per-year prognostic Zarrs (years 12–132).
for YEAR in $(seq 12 132); do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping year $YEAR (already exists)"
        continue
    fi
    echo "[$(date)] converting PLASIM year $YEAR ..."
    python tools/data/plasim/pangu_h5_to_zarr.py \
        --config tools/data/plasim/configs/sim52_full_coverage.yaml \
        --config-key PLASIM \
        --input-dir "$H5_DIR" \
        --year "$YEAR" \
        --output "$OUT_PATH" \
        --time-chunk 1
done

# 2. Normalization Zarr (per-variable mean+std for years 12–132).
echo "[$(date)] building normalization Zarr ..."
python tools/data/plasim/build_normalization_zarr.py \
    --mean "$STATS_DIR/data_12-132_mean_sigma.nc" \
    --std  "$STATS_DIR/data_12-132_std_sigma.nc" \
    --output "$OUT_ROOT/normalization_12-132.zarr" \
    --overwrite

# 3. Climatology + bias Zarr (no std climatology for PLASIM — NaN-filled).
echo "[$(date)] building climatology+bias Zarr ..."
python tools/data/plasim/build_climatology_zarr.py \
    --climatology "$CLIM_DIR/climatology.nc" \
    --bias-dir    "$BIAS_DIR" \
    --output      "$OUT_ROOT/climatology_bias.zarr" \
    --overwrite

echo "[$(date)] PLASIM full-archive conversion complete."
ls -la "$OUT_ROOT"
