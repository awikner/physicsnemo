#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the full AMIP archive (1978-2024) on TACC Stampede3 (`spr` queue) → $SCRATCH.
# Generalizes convert_amip_1981_test.sbatch (Delta, single year) to a year loop.
# skip-if-exists. Copy the output to Derecho (master) afterward — see phase-11 §11b.
# (AMIP normalization stats stay .nc — the recipe reads them directly; there is no
# AMIP build_normalization_zarr tool yet.)
#
# Prereq: raw H5 staged to $RAW ({year}_{idx:04d}.h5, from 1978) via Globus.
# Submit:  sbatch hpc/scripts/convert_amip_stampede3.sbatch

#SBATCH -p spr
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 36:00:00
#SBATCH -J pn-conv-amip
#SBATCH -o convert-amip-%j.out

set -euo pipefail
cd "$WORK/physicsnemo"
source .venv/bin/activate

RAW=${RAW:-$SCRATCH/raw/amip}
OUT_ROOT=${OUT_ROOT:-$SCRATCH/physicsnemo-zarr/amip}
mkdir -p "$OUT_ROOT"

for YEAR in $(seq 1978 2024); do
    OUT_PATH="$OUT_ROOT/${YEAR}.zarr"
    if [ -d "$OUT_PATH" ]; then
        echo "[$(date)] skipping AMIP year $YEAR (already exists)"
        continue
    fi
    echo "[$(date)] converting AMIP year $YEAR ..."
    python tools/data/amip/amip_h5_to_zarr.py \
        --config tools/data/amip/configs/amip_default.yaml \
        --input-dir "$RAW" \
        --year "$YEAR" \
        --output "$OUT_PATH" \
        --time-chunk 1 \
        --write-batch 50
done

echo "[$(date)] AMIP conversion complete."
ls -la "$OUT_ROOT"
