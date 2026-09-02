#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ship a pulled .sif from Delta to another cluster over Globus.
#
#   hpc/containers/replicate_sif.sh <tag> <dst-cluster> [arch]
#   hpc/containers/replicate_sif.sh latest stampede3
#
# Run from the Delta login node after hpc/containers/pull_sif.sh.
#
# Unlike hpc/scripts/replicate_tar.sh, there is no tar-bundling step: that
# script exists because a converted Zarr store is ~17k tiny chunk files and
# Globus pays ~20 ms of per-file overhead. A .sif is one large file, so plain
# Globus already runs at link rate.
#
# Requires a live Globus session for both endpoints' domains — high-assurance
# sessions time out:
#   globus session update access-ci.org    # NCSA Delta
#   globus session update uchicago.edu     # TACC Stampede3
set -euo pipefail

TAG="${1:?usage: replicate_sif.sh <tag> <dst-cluster> [arch]}"
DST="${2:?destination cluster: stampede3|midway3|polaris|derecho}"
ARCH="${3:-x86_64}"

GLOBUS="${GLOBUS:-$HOME/gcli/bin/globus}"
command -v "$GLOBUS" >/dev/null 2>&1 || GLOBUS=globus

# Collection UUIDs are the same ones hpc/data_registry.yaml already records.
SRC_UUID="${SRC_UUID:-7e936164-de58-4e3d-85da-21aa23c07169}"   # NCSA Delta
SRC_DIR="${AI_ROSSBY_CONTAINER_DIR:-/work/nvme/bdiu/awikner/containers}"

case "$DST" in
    stampede3)
        DST_UUID=1e9ddd41-fe4b-406f-95ff-f3d79f9cb523           # TACC Stampede3
        DST_DIR=/work2/09979/awikner/stampede3/containers ;;
    polaris)
        DST_UUID=05d2c76a-e867-4f67-aa57-76edeb0beda0           # ALCF Eagle
        DST_DIR=/eagle/lighthouse-uchicago/members/awikner/containers ;;
    derecho)
        DST_UUID=d33b3614-6d04-11e5-ba46-22000b92c6ec           # NCAR GLADE
        DST_DIR=/glade/work/awikner/containers ;;
    midway3)
        # Midway3 has no Globus collection registered (hpc/data_registry.yaml
        # leaves midway3/dsi blank) and no globus-cli installed, so it is
        # scp-only for now. One ~20 GB file makes that acceptable.
        echo "Midway3 has no registered Globus collection — using scp instead." >&2
        SIF="${SRC_DIR}/ai-rossby-${TAG}-${ARCH}.sif"
        [ -s "$SIF" ] || { echo "error: missing $SIF — run pull_sif.sh first" >&2; exit 1; }
        ssh midway3 "mkdir -p /project/pedramh/awikner/containers"
        exec scp "$SIF" "midway3:/project/pedramh/awikner/containers/"
        ;;
    *)
        echo "error: unknown destination '$DST'" >&2
        exit 2 ;;
esac

SIF="ai-rossby-${TAG}-${ARCH}.sif"

# Verify the source through Globus rather than with a local `-s` test: this is a
# server-to-server transfer, so the script does not have to run on the source
# cluster (driving it from a workstation is the common case) and the file will
# usually not be on the local filesystem at all.
if ! "$GLOBUS" ls "${SRC_UUID}:${SRC_DIR}/" 2>/dev/null | grep -qx "${SIF}"; then
    echo "error: ${SIF} not found on the source collection at ${SRC_DIR}/" >&2
    echo "       Run hpc/containers/pull_sif.sh on a DeltaAI login node first," >&2
    echo "       or check the session: globus session show" >&2
    exit 1
fi

echo "[$(date)] transfer ${SIF}"
echo "          ${SRC_UUID}:${SRC_DIR} -> ${DST_UUID}:${DST_DIR}"

# --sync-level mtime: one big file, and Globus integrity-checks the transfer
# regardless. Matches replicate_tar.sh's reasoning.
tid=$("$GLOBUS" transfer \
        "${SRC_UUID}:${SRC_DIR}/${SIF}" \
        "${DST_UUID}:${DST_DIR}/${SIF}" \
        --sync-level mtime \
        --label "ai-rossby container ${TAG} ${ARCH}" \
        --format unix --jmespath task_id)

echo "[$(date)] task: ${tid}"
echo "Watch with: ${GLOBUS} task wait ${tid}"
