#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the PLASIM sim52 pressure-level-only archive to the unified
# ai-rossby per-year Zarr format. Sibling to
# convert_plasim_full_archive.sbatch — same per-file H5 loop, but the
# source dir is the new `plev_data/` tree whose H5 files carry the full
# 13-level pressure axis for *all* upper-air variables (ta, ua, va, hus,
# zg), with no sigma data.
#
# Year discovery is dynamic — the script enumerates whichever years are
# present in the source dir at job start (the upstream transfer is still
# in flight), so this sbatch can be re-submitted later to catch up as
# more years finish copying. Already-converted years are skipped.
#
# Output: /work/hdd/bdiu/awikner/physicsnemo-zarr/plasim_plev/
#   - {NN..MM}.zarr  — per-year prognostic stores (pressure-level upper-air)
#
# Normalization + climatology Zarrs intentionally NOT generated here — the
# user can reuse the existing ones from convert_plasim_full_archive.sbatch
# (same per-variable stats), or re-run those converters against the
# matching plev_data stats files if/when they're separated.
#
# Submit with (honor the user's 30-min transfer-in-flight delay):
#
#     sbatch --begin=now+30minutes hpc/scripts/convert_plasim_plev_full_archive.sbatch

#SBATCH --partition=cpu
#SBATCH --account=bdiu-delta-cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --job-name=pn-conv-plasim-plev
#SBATCH --output=hpc/scripts/logs/convert-plasim-plev-%j.out

set -euo pipefail

cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

OUT_ROOT=/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim_plev
mkdir -p "$OUT_ROOT"

SOURCE_DIR=/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52
H5_DIR="/work/hdd/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/plev_data"

# Dynamically discover years present in the source dir at job start.
# Pattern: {year}_{idx:04d}.h5 — we extract distinct {year} prefixes.
mapfile -t YEARS < <(ls "$H5_DIR" 2>/dev/null \
    | awk -F_ 'NF==2{print $1}' \
    | sort -u -n)
if (( ${#YEARS[@]} == 0 )); then
    echo "[ERROR] no {year}_NNNN.h5 files found under $H5_DIR" >&2
    exit 1
fi
echo "[$(date)] discovered ${#YEARS[@]} year(s) in $H5_DIR: ${YEARS[*]}"

# Per-year prognostic Zarrs. Skip already-converted years so we can
# re-submit later to catch up as more years finish transferring.
N_CONVERTED=0
N_SKIPPED=0
N_FAILED=0
for YEAR in "${YEARS[@]}"; do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping year $YEAR (already exists at $OUT_PATH)"
        N_SKIPPED=$((N_SKIPPED+1))
        continue
    fi
    echo "[$(date)] converting PLASIM plev year $YEAR ..."
    if python tools/data/plasim/pangu_h5_to_zarr.py \
            --config tools/data/plasim/configs/sim52_plev_full.yaml \
            --config-key PLASIM \
            --input-dir "$H5_DIR" \
            --year "$YEAR" \
            --output "$OUT_PATH" \
            --time-chunk 1; then
        N_CONVERTED=$((N_CONVERTED+1))
    else
        rc=$?
        echo "[$(date)] year $YEAR FAILED (rc=$rc); continuing"
        N_FAILED=$((N_FAILED+1))
    fi
done

echo
echo "[$(date)] plev archive conversion summary: converted=$N_CONVERTED, skipped=$N_SKIPPED, failed=$N_FAILED"
echo "[$(date)] output dir contents:"
ls -la "$OUT_ROOT"

if (( N_FAILED > 0 )); then
    exit 2
fi
