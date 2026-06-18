#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the full ERA5 archive (1979–2018 = 40 years) to the unified ai-rossby
# per-year Zarr format, plus all 5 normalization variants and the climatology
# (mean + std, no bias dir for ERA5).
#
# Output: /work/nvme/bdiu/awikner/physicsnemo-zarr/era5/
#   - {1979..2018}.zarr                    — per-year prognostic stores
#   - normalization_pangu_s2s.zarr         — default Pangu_S2S normalization
#   - normalization_pangu_s2s_withnino.zarr        — with Nino34 lags
#   - normalization_pangu_s2s_log_precip.zarr      — log-precip surface vars
#   - normalization_pangu_s2s_withnino_log_precip.zarr
#   - normalization_pangu_s2s_Z200.zarr            — Z200 ablation
#   - climatology_bias.zarr                — daily mean+std climatology
#
# Submit with:
#   sbatch hpc/scripts/convert_era5_full_archive.sbatch

#SBATCH --partition=cpu
#SBATCH --account=bdiu-delta-cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --job-name=pn-conv-era5
#SBATCH --output=hpc/scripts/logs/convert-era5-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

OUT_ROOT=/work/nvme/bdiu/awikner/physicsnemo-zarr/era5
mkdir -p "$OUT_ROOT"

SOURCE_DIR=/work/hdd/bdiu/bgong1/data/h5data

# 1. Per-year prognostic Zarrs.
for YEAR in $(seq 1979 2018); do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping ERA5 year $YEAR (already exists)"
        continue
    fi
    echo "[$(date)] converting ERA5 year $YEAR ..."
    python tools/data/era5/pangu_h5_to_zarr.py \
        --input-dir "$SOURCE_DIR" \
        --year "$YEAR" \
        --output "$OUT_PATH"
done

# 2. Normalization Zarrs (one per variant).
for VARIANT in pangu_s2s pangu_s2s_withnino pangu_s2s_log_precip pangu_s2s_withnino_log_precip pangu_s2s_Z200; do
    echo "[$(date)] building normalization variant $VARIANT ..."
    python tools/data/era5/build_normalization_zarr.py \
        --source-dir "$SOURCE_DIR" \
        --variant "$VARIANT" \
        --output  "$OUT_ROOT/normalization_${VARIANT}.zarr" \
        --overwrite
done

# 3. Climatology Zarr (mean + std, no bias).
echo "[$(date)] building ERA5 climatology Zarr ..."
python tools/data/era5/build_climatology_zarr.py \
    --source-dir "$SOURCE_DIR" \
    --output     "$OUT_ROOT/climatology_bias.zarr" \
    --overwrite

echo "[$(date)] ERA5 full-archive conversion complete."
ls -la "$OUT_ROOT"
