#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot Phase 11 driver (run ON a Stampede3 login node). Stages the four raw
# archives to $SCRATCH via Globus and submits each spr conversion the moment its
# raw lands (globus task wait blocks until SUCCEEDED = checksum-verified, so no
# conversion ever starts on a partial archive). Long-running — launch detached:
#
#   nohup bash hpc/scripts/run_phase11_stampede3.sh > ~/phase11.log 2>&1 &
#
# Prereq: a valid Globus session for NCSA (Delta) + TACC (Stampede3). These are
# high-assurance collections whose session times out, so refresh interactively
# first (browser):   ~/gcli/bin/globus session update
# Derecho (source for plev/E3SM) already works with the cached token.

set -euo pipefail

GLOBUS=${GLOBUS:-$HOME/gcli/bin/globus}
REPO=${REPO:-$WORK/physicsnemo}
DELTA=7e936164-de58-4e3d-85da-21aa23c07169
DERE=d33b3614-6d04-11e5-ba46-22000b92c6ec
TACC=1e9ddd41-fe4b-406f-95ff-f3d79f9cb523

# --- preflight -------------------------------------------------------------
[ -x "$GLOBUS" ] || { echo "ERROR: no globus CLI at $GLOBUS" >&2; exit 2; }
"$GLOBUS" whoami >/dev/null 2>&1 || { echo "ERROR: globus not logged in — run: $GLOBUS login" >&2; exit 2; }

submit() {  # SRC DST LABEL  -> prints the task id
    "$GLOBUS" transfer "$1" "$2" --recursive --sync-level checksum \
        --label "$3" --format unix --jmespath task_id
}

stage_and_convert() {  # SRC DST LABEL SBATCH
    echo "[$(date)] submitting: $3"
    local tid
    tid=$(submit "$1" "$2" "$3") || { echo "  submit FAILED (session expired for a high-assurance collection?)" >&2; return 1; }
    echo "[$(date)] $3 task=$tid — waiting for SUCCEEDED ..."
    "$GLOBUS" task wait "$tid"
    echo "[$(date)] $3 landed — sbatch $4"
    ( cd "$REPO" && sbatch "hpc/scripts/$4" )
}

# plev first — it is the long pole (565 G / 184k files).
stage_and_convert \
    "$DERE:/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/plev_data" \
    "$TACC:/scratch/09979/awikner/raw/plasim_plev/plev_data" \
    "ai-rossby plev raw" convert_plasim_plev_stampede3.sbatch &

# E3SM: the complete archive already lives on Stampede3 (jwan4 scratch, the
# sbatch's default RAW) — no transfer needed, submit the conversion directly.
( echo "[$(date)] e3sm raw is local (jwan4) — sbatch directly"
  cd "$REPO" && sbatch hpc/scripts/convert_e3sm_stampede3.sbatch ) &

stage_and_convert \
    "$DELTA:/work/hdd/bdiu/bgong1/data/h5data" \
    "$TACC:/scratch/09979/awikner/raw/era5/h5data" \
    "ai-rossby era5 raw" convert_era5_stampede3.sbatch &

stage_and_convert \
    "$DELTA:/work/hdd/bdiu/awikner/AMIP/h5" \
    "$TACC:/scratch/09979/awikner/raw/amip" \
    "ai-rossby amip raw" convert_amip_stampede3.sbatch &

wait
echo "[$(date)] all four staged + conversions submitted."
echo "When the spr jobs finish, consolidate:  cd $REPO/tools/data"
echo "  python registry.py scan stampede3 --write"
echo "  for d in era5 amip plasim_plev e3sm; do python sync_dataset.py \$d --to derecho; done"
echo "  python registry.py scan derecho --write && python registry.py check"
