#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the full E3SM archive (2015–2049 = 35 years) to the unified ai-rossby
# per-year Zarr format, plus the normalization Zarr and climatology+bias Zarr.
#
# Output: /work/nvme/bdiu/awikner/physicsnemo-zarr/e3sm/
#   - {2015..2049}.zarr            — per-year prognostic stores
#   - normalization_2015-2050.zarr — per-variable mean+std
#   - climatology_bias.zarr        — daily climatology + bias arrays (soil
#                                     vars decomposed by depth into flat 2D
#                                     channels per user answer #7)
#
# Submit with:
#   sbatch hpc/scripts/convert_e3sm_full_archive.sbatch

#SBATCH --partition=cpu
#SBATCH --account=bdiu-delta-cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --job-name=pn-conv-e3sm
#SBATCH --output=hpc/scripts/logs/convert-e3sm-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

OUT_ROOT=/work/nvme/bdiu/awikner/physicsnemo-zarr/e3sm
mkdir -p "$OUT_ROOT"

SOURCE_DIR=/work/hdd/bdiu/awikner/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101
H5_DIR="$SOURCE_DIR/h5/sigma_data"
STATS_DIR="$SOURCE_DIR/h5/sigma_data"
CLIM_PATH="$SOURCE_DIR/sigma_data/climatology.nc"
BIAS_DIR="$SOURCE_DIR/bias"

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

# 2. Normalization Zarr.
echo "[$(date)] building E3SM normalization Zarr ..."
python tools/data/e3sm/build_normalization_zarr.py \
    --source-dir "$STATS_DIR" \
    --output "$OUT_ROOT/normalization_2015-2050.zarr" \
    --overwrite

# 3. Climatology + bias Zarr.
echo "[$(date)] building E3SM climatology+bias Zarr ..."
python tools/data/e3sm/build_climatology_zarr.py \
    --climatology "$CLIM_PATH" \
    --bias-dir    "$BIAS_DIR" \
    --output      "$OUT_ROOT/climatology_bias.zarr" \
    --overwrite

echo "[$(date)] E3SM full-archive conversion complete."
ls -la "$OUT_ROOT"
