#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 12d option (b): extract the native-resolution (1 degree) boundary-only
# stores from the full-res amip_dailyavg archive on TACC Stampede3, so training
# can pair coarse state with upstream's native forcings without moving the
# ~2.6 TB full-res archive. ~2.3 GB/year -> ~105 GB for 1978-2022.
#
# Submit:  sbatch hpc/scripts/extract_amip_boundary_stampede3.sbatch

#SBATCH -p spr
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH -t 08:00:00
#SBATCH -J pn-bnd-amipda
#SBATCH -o extract-bnd-%j.out

set -uo pipefail   # NOT -e: a failed year must not kill the loop

REPO=${REPO:-/work2/09979/awikner/stampede3/physicsnemo-amip-v2}
cd "$REPO"
# Runs in the ai-rossby container (hpc/containers/) instead of a per-cluster venv.
# AI_ROSSBY_VENV is superseded by AI_ROSSBY_SIF / AI_ROSSBY_CONTAINER_TAG.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

YEARS_LO=${YEARS_LO:-1978}
YEARS_HI=${YEARS_HI:-2022}
FULL_ROOT=${FULL_ROOT:-$SCRATCH/physicsnemo-zarr/amip_dailyavg}
BND_ROOT=${BND_ROOT:-$SCRATCH/physicsnemo-zarr/amip_dailyavg_boundary}
mkdir -p "$BND_ROOT"

# Norm stats travel with every store root so each is self-contained.
cp -n tools/data/amip/norm_stats/normalize_mean_dailyavg.nc "$BND_ROOT/" 2>/dev/null || true
cp -n tools/data/amip/norm_stats/normalize_std_dailyavg.nc "$BND_ROOT/" 2>/dev/null || true

n_ok=0; n_skip=0; n_fail=0
for YEAR in $(seq "$YEARS_LO" "$YEARS_HI"); do
    IN="$FULL_ROOT/${YEAR}.zarr"
    OUT="$BND_ROOT/${YEAR}.zarr"
    if [ ! -d "$IN" ]; then
        echo "[$(date)] skip $YEAR (no full-res store)"; n_skip=$((n_skip+1)); continue
    fi
    if [ -d "$OUT" ]; then
        echo "[$(date)] skip $YEAR (exists)"; n_skip=$((n_skip+1)); continue
    fi
    echo "[$(date)] extracting boundaries for $YEAR ..."
    TMP="$BND_ROOT/${YEAR}.zarr.partial"
    rm -rf "$TMP"
    if "$CONTAINER_RUN" python tools/data/amip/extract_boundary_store.py \
            --input "$IN" --output "$TMP" --time-chunk 8; then
        mv "$TMP" "$OUT"; n_ok=$((n_ok+1))
    else
        echo "[$(date)] YEAR $YEAR FAILED"; n_fail=$((n_fail+1)); rm -rf "$TMP"
    fi
done
echo "[$(date)] done: ok=$n_ok skip=$n_skip fail=$n_fail"
du -sh "$BND_ROOT" 2>/dev/null || true
[ "$n_fail" -eq 0 ]
