#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert + coarsen the AMIP daily-avg archive on TACC Stampede3 (Phase
# 12c.12, Stampede3 routing decision 2026-08-08): per year, H5 -> full-res
# Zarr (tools/data/amip/configs/amip_dailyavg.yaml, WITH extras) then ->
# 45x90 coarse Zarr (coarsen_zarr.py, --time-chunk 8). Atomic tmp->rename
# on both steps; skip-if-exists per year.
#
# Target 1978-2022 — the 2023-2024 raw lack global_mean_co2 (probed
# 2026-08-08, same gap as the original amip archive).
#
# Prereq: raw staged via
#   python tools/data/sync_dataset.py amip_dailyavg --to stampede3 --stage-raw
# Runs from the ai-rossby-amip-v2 worktree (PYTHONPATH pins it over the
# venv's editable install of the main checkout).
#
# ~35-65 min/year conversion + ~6 min coarsen => split the range over two
# submissions to fit comfortably, e.g.:
#   sbatch --export=ALL,YEARS_LO=1978,YEARS_HI=2000 hpc/scripts/convert_amip_dailyavg_stampede3.sbatch
#   sbatch --export=ALL,YEARS_LO=2001,YEARS_HI=2022 hpc/scripts/convert_amip_dailyavg_stampede3.sbatch

#SBATCH -p spr
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH -t 40:00:00
#SBATCH -J pn-conv-amipda
#SBATCH -o convert-amipda-%j.out

set -uo pipefail   # NOT -e: a failed year must not kill the loop

REPO=${REPO:-/work2/09979/awikner/stampede3/physicsnemo-amip-v2}
cd "$REPO"

# Runs in the ai-rossby container instead of borrowing the sibling checkout's
# venv (this used to activate physicsnemo/.venv while REPO pointed at
# physicsnemo-amip-v2, so the code and the interpreter came from different
# worktrees). AI_ROSSBY_PYTHONPATH keeps this tree's modules first.
CONTAINER_RUN="$REPO/hpc/scripts/container_run.sh"
export AI_ROSSBY_PYTHONPATH="$REPO"
# CPU partition: no GPU to pass through, so skip apptainer's --nv.
export AI_ROSSBY_NV=0

export SLURM_CPUS_PER_TASK=$(nproc)
echo "[$(date)] $(hostname): $(nproc) cores"

YEARS_LO=${YEARS_LO:-1978}
YEARS_HI=${YEARS_HI:-2022}
RAW=${RAW:-$SCRATCH/raw/amip_dailyavg}
FULL_ROOT=${FULL_ROOT:-$SCRATCH/physicsnemo-zarr/amip_dailyavg}
COARSE_ROOT=${COARSE_ROOT:-$SCRATCH/physicsnemo-zarr/amip_dailyavg_coarse}
mkdir -p "$FULL_ROOT" "$COARSE_ROOT"

# Stage the vendored norm stats next to both store roots (idempotent).
cp -n tools/data/amip/norm_stats/normalize_mean_dailyavg.nc "$FULL_ROOT/" 2>/dev/null || true
cp -n tools/data/amip/norm_stats/normalize_std_dailyavg.nc "$FULL_ROOT/" 2>/dev/null || true
cp -n tools/data/amip/norm_stats/normalize_mean_dailyavg.nc "$COARSE_ROOT/" 2>/dev/null || true
cp -n tools/data/amip/norm_stats/normalize_std_dailyavg.nc "$COARSE_ROOT/" 2>/dev/null || true

n_ok=0; n_skip=0; n_fail=0
for YEAR in $(seq "$YEARS_LO" "$YEARS_HI"); do
    FULL_PATH="$FULL_ROOT/${YEAR}.zarr"
    COARSE_PATH="$COARSE_ROOT/${YEAR}.zarr"

    # --- full-res conversion (atomic) ---
    if [ ! -d "$FULL_PATH" ]; then
        echo "[$(date)] converting year $YEAR ..."
        FULL_TMP="$FULL_ROOT/${YEAR}.zarr.partial"
        rm -rf "$FULL_TMP"
        if "$CONTAINER_RUN" python tools/data/amip/amip_h5_to_zarr.py \
                --config tools/data/amip/configs/amip_dailyavg.yaml \
                --config-key AMIP \
                --input-dir "$RAW" \
                --year "$YEAR" \
                --output "$FULL_TMP" \
                --time-chunk 1 \
                --write-batch 50; then
            mv "$FULL_TMP" "$FULL_PATH"
        else
            echo "[$(date)] YEAR $YEAR convert FAILED"; n_fail=$((n_fail+1)); rm -rf "$FULL_TMP"
            continue
        fi
    else
        echo "[$(date)] skip convert $YEAR (exists)"
    fi

    # --- coarsening (atomic) ---
    if [ ! -d "$COARSE_PATH" ]; then
        echo "[$(date)] coarsening year $YEAR ..."
        COARSE_TMP="$COARSE_ROOT/${YEAR}.zarr.partial"
        rm -rf "$COARSE_TMP"
        if "$CONTAINER_RUN" python tools/data/amip/coarsen_zarr.py \
                --input "$FULL_PATH" \
                --output "$COARSE_TMP" \
                --factor 4 \
                --time-block 64 \
                --time-chunk 8 \
                --smooth-boundaries; then
            mv "$COARSE_TMP" "$COARSE_PATH"
            n_ok=$((n_ok+1))
        else
            echo "[$(date)] YEAR $YEAR coarsen FAILED"; n_fail=$((n_fail+1)); rm -rf "$COARSE_TMP"
        fi
    else
        echo "[$(date)] skip coarsen $YEAR (exists)"; n_skip=$((n_skip+1))
    fi
done
echo "[$(date)] done: ok=$n_ok skip=$n_skip fail=$n_fail"
[ "$n_fail" -eq 0 ]