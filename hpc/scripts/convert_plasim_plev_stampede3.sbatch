#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert PLASIM pressure-level (sim52 plev) H5 → Zarr on TACC Stampede3 (`spr`).
# Stampede3 sibling of convert_plasim_plev_full_archive.sbatch (Delta). Dynamic
# year discovery + skip-if-exists picks up the missing 79-104 (to match sigma).
# The plev .h5 is complete — no sigma merge / no precip computation.
#
# NOTE: the plev raw .h5 lives on **Derecho** (D3). Either stage it to
#   $SCRATCH/raw/plasim_plev/plev_data here, OR (more efficient — no round-trip)
#   convert plev in-place on Derecho and Globus only the Zarr to Stampede3.
#
# Submit:  sbatch hpc/scripts/convert_plasim_plev_stampede3.sbatch

#SBATCH -p spr
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -J pn-conv-plev
#SBATCH -o convert-plasim-plev-%j.out

set -euo pipefail
cd "$WORK/physicsnemo"
source .venv/bin/activate

H5_DIR=${H5_DIR:-$SCRATCH/raw/plasim_plev/plev_data}
OUT_ROOT=${OUT_ROOT:-$SCRATCH/physicsnemo-zarr/plasim_plev}
mkdir -p "$OUT_ROOT"

# Discover {year}_{idx:04d}.h5 → distinct year prefixes.
mapfile -t YEARS < <(ls "$H5_DIR" 2>/dev/null | awk -F_ 'NF==2{print $1}' | sort -u -n)
if (( ${#YEARS[@]} == 0 )); then
    echo "[ERROR] no {year}_NNNN.h5 files under $H5_DIR" >&2
    exit 1
fi
echo "[$(date)] discovered ${#YEARS[@]} year(s): ${YEARS[*]}"

for YEAR in "${YEARS[@]}"; do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping plev year $YEAR (already exists)"
        continue
    fi
    echo "[$(date)] converting PLASIM plev year $YEAR ..."
    python tools/data/plasim/pangu_h5_to_zarr.py \
        --config tools/data/plasim/configs/sim52_plev_full.yaml \
        --config-key PLASIM \
        --input-dir "$H5_DIR" \
        --year "$YEAR" \
        --output "$OUT_PATH" \
        --time-chunk 1
done

echo "[$(date)] PLASIM plev conversion complete."
ls -la "$OUT_ROOT"
