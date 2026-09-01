#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pull the ai-rossby image from GHCR and convert it to .sif. Run this ON THE
# DELTA LOGIN NODE.
#
#   hpc/containers/pull_sif.sh [tag]        # default tag: latest
#
# Delta is the only cluster that can do this: it has apptainer natively (1.5.1,
# login nodes included) plus outbound internet. Stampede3 and Polaris have
# apptainer only on compute nodes, and those compute nodes have no direct
# internet — so nothing there can reach GHCR. Both arch variants land on
# /work/nvme, which DeltaAI shares, so DeltaAI needs no transfer at all; the
# rest are fed by hpc/containers/replicate_sif.sh.
set -euo pipefail

TAG="${1:-latest}"
IMAGE="${AI_ROSSBY_IMAGE:-ghcr.io/awikner/physicsnemo-ai-rossby}"
DEST="${AI_ROSSBY_CONTAINER_DIR:-/work/nvme/bdiu/awikner/containers}"

# Cache and unpack go on NODE-LOCAL disk, not /work.
#
# `apptainer pull` needs roughly 3x the final .sif: the compressed layer blobs in
# the cache (~10 GB) plus a fully-expanded rootfs (~25 GB) before it squashes the
# image. Putting that on /work consumed enough of the bdiu project quota to fail
# mid-unpack with EDQUOT ("disk quota exceeded" on a gconv .so), because the
# project quota is shared across /work/nvme and /work/hdd and /work/hdd is
# already over its own soft limit (20.79T of 19.53T). See
# docs/dev/context/lat-orientation-audit.md for the standing storage squeeze.
#
# Delta's login nodes have ~1.1 TB of local /tmp, which is not quota'd, so only
# the finished .sif (~10-15 GB) ever lands on shared storage. $HOME is never used
# either — the same small-quota trap that bit UV_CACHE_DIR on Midway3.
pick_scratch() {
    local cand avail
    for cand in "${AI_ROSSBY_LOCAL_SCRATCH:-}" /tmp /var/tmp; do
        [ -n "$cand" ] && [ -d "$cand" ] || continue
        avail=$(df -BG --output=avail "$cand" 2>/dev/null | tail -1 | tr -dc '0-9')
        if [ -n "$avail" ] && [ "$avail" -ge 80 ]; then echo "$cand"; return; fi
    done
    # Nothing local and roomy; fall back to /work and hope the quota allows it.
    echo "${DEST%/*}"
}
LOCAL_SCRATCH="$(pick_scratch)/${USER}-apptainer"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${LOCAL_SCRATCH}/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${LOCAL_SCRATCH}/tmp}"

command -v apptainer >/dev/null 2>&1 || {
    echo "error: apptainer not on PATH. Run this on a Delta login node." >&2
    exit 1
}

mkdir -p "$DEST" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "image : ${IMAGE}:${TAG}"
echo "dest  : ${DEST}"
echo "cache : ${APPTAINER_CACHEDIR}  (local, not quota'd)"
echo

# Name the .sif by `uname -m` values, not Docker's, so container_run.sh can
# select one with a bare $(uname -m).
for pair in "amd64:x86_64" "arm64:aarch64"; do
    docker_arch="${pair%%:*}"
    uname_arch="${pair##*:}"
    out="${DEST}/ai-rossby-${TAG}-${uname_arch}.sif"

    if [ -s "$out" ] && [ "${AI_ROSSBY_FORCE_PULL:-0}" != 1 ]; then
        echo "[skip] $out already exists (AI_ROSSBY_FORCE_PULL=1 to replace)"
        continue
    fi

    echo "[pull] ${docker_arch} -> ${out}"
    # --arch selects from the multi-arch manifest list; the arm64 variant is
    # pulled on x86 hardware here purely as a file to ship onward, so it is
    # never executed on Delta.
    apptainer pull --force --arch "${docker_arch}" "${out}.part" \
        "docker://${IMAGE}:${TAG}"
    mv "${out}.part" "$out"
    echo "[ok]   $(du -h "$out" | cut -f1)  $out"
done

echo
echo "Sanity-check the native image:"
echo "  apptainer exec --nv ${DEST}/ai-rossby-${TAG}-x86_64.sif python -c \\"
echo "    'import torch; print(torch.__version__, torch.version.cuda)'"
echo
echo "Then replicate to the other clusters:"
echo "  hpc/containers/replicate_sif.sh ${TAG} stampede3"
