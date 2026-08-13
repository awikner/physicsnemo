#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 12h — pull the first real amip_v2-trained checkpoints off Derecho.
#
# Derecho scratch is RETIRING and these are (as far as we know) the only copies
# reachable to us, so this runs before anything that depends on them. Both were
# trained at TACC and rsynced to Derecho; `project: amip_v2` in each config.yml.
#
#   forecaster  ERDM_fancy  5.1 GB  nocean=3 (SST + anomaly + ice), sst append,
#                                   scalar_forcing global_mean_sst, full 12e
#                                   feature set, c_grid_dim 6 / scalar_dim 3
#   downscaler  x_DDC       4.5 GB  dit decoder 302 -> 151, patch 4, 180x360
#
# Plus one prediction/target pair from each for cross-checking (the forecaster
# saves both, which is what makes an end-to-end comparison possible at all).
#
# Runs server-side; no ssh needed. Requires an authenticated globus-cli:
#   uv tool install globus-cli && globus login
# and, because both endpoints are high-assurance, possibly
#   globus session update ucar.edu      # NCAR GLADE
#   globus session update alcf.anl.gov  # ALCF Eagle
#
# Usage:  bash hpc/scripts/fetch_v2_checkpoints_globus.sh [--dry-run]

set -euo pipefail

GLADE=d33b3614-6d04-11e5-ba46-22000b92c6ec      # NCAR GLADE
EAGLE=05d2c76a-e867-4f67-aa57-76edeb0beda0      # ALCF Eagle

# NOTE the Eagle collection is rooted at /eagle, so destination paths OMIT that
# prefix: /lighthouse-uchicago/... is /eagle/lighthouse-uchicago/... on disk.
# Confirm with `globus ls $EAGLE:/lighthouse-uchicago/` before a 9.6 GB run.
DEST=/lighthouse-uchicago/amip-checkpoints
FC_SRC=/glade/derecho/scratch/ayz/AMIP_logs/ERDM_ERDM_fancy_42_2026-08-10T13-21-13
DS_SRC=/glade/derecho/scratch/katyr/ai-models/CMU_model/x_DDC_x_DDC_42_2026-08-07T09-34-49
FC=ERDM_fancy_42_2026-08-10T13-21-13
DS=x_DDC_42_2026-08-07T09-34-49

GLOBUS=${GLOBUS:-$HOME/.local/bin/globus}
DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

BATCH=$(mktemp)
trap 'rm -f "$BATCH"' EXIT
cat >"$BATCH" <<EOF
$FC_SRC/last.ckpt $DEST/$FC/last.ckpt
$FC_SRC/config.yml $DEST/$FC/config.yml
$FC_SRC/predictions_epoch_15.pt $DEST/$FC/predictions_epoch_15.pt
$FC_SRC/targets_epoch_15.pt $DEST/$FC/targets_epoch_15.pt
$DS_SRC/last.ckpt $DEST/$DS/last.ckpt
$DS_SRC/config.yml $DEST/$DS/config.yml
$DS_SRC/predictions_epoch_49.pt $DEST/$DS/predictions_epoch_49.pt
EOF

echo "=== transfer plan (Derecho GLADE -> ALCF Eagle) ==="
cat "$BATCH"
echo

# --sync-level checksum makes re-runs idempotent, which matters for 9.6 GB over
# a link that may need a session refresh mid-flight.
"$GLOBUS" transfer "$GLADE" "$EAGLE" \
    --batch "$BATCH" \
    --sync-level checksum \
    --preserve-mtime \
    --verify-checksum \
    --label "amip_v2 ckpts -> eagle (phase 12h)" \
    $DRY

echo
echo "Watch with:  $GLOBUS task list        (or: $GLOBUS task wait <TASK_ID>)"
echo "On Polaris:  ls -la /eagle$DEST/{$FC,$DS}"
