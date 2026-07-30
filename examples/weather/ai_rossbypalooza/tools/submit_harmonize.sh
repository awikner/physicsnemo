#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Submit the full harmonization as SIX parallel develop-queue slices
# (heavy DSI models split by year range; pangu/sfno subsets are light).
#   ./submit_harmonize.sh [COMMIT]
set -euo pipefail
TOOLS="$HOME/mowe_tools/tools"
COMMIT="${1:-unknown}"
LOGDIR=/glade/derecho/scratch/awikner/hindcasts_mowe/logs
mkdir -p "$LOGDIR"

submit() {  # name task years
    qsub -N "$1" -o "$LOGDIR/$1.log" \
        -v "TASK=$2,YEARS=$3,COMMIT=$COMMIT" \
        "$TOOLS/harmonize_derecho.pbs"
}

submit mowe_h_gc1   graphcast      2000-2012
submit mowe_h_gc2   graphcast      2013-2024
submit mowe_h_aifs1 aifs_single_v2 2000-2012
submit mowe_h_aifs2 aifs_single_v2 2013-2024
submit mowe_h_pangu pangu_s2s      2000-2024
submit mowe_h_sfno  sfno_era5      2000-2024
qstat -u "$USER" | tail -10
