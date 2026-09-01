#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pull the ai-rossby image from GHCR and convert it to .sif for both cluster
# architectures. Run this ON A DELTAAI LOGIN NODE.
#
#   hpc/containers/pull_sif.sh [tag]        # default tag: latest
#
# Why DeltaAI and not Delta, which is the machine with the data and the bigger
# user base:
#
#   * Delta's apptainer 1.5.1 bundles mksquashfs 4.7.5 (2026/03/01), which is
#     BROKEN for an image this size — it fails either with
#     "FATAL ERROR: Bug in orderer" (default 128 procs) or SIGSEGV (exit 139,
#     with --mksquashfs-args "-processors 8"). The OCI download and rootfs
#     extraction both succeed; only the squashfs step dies. Delta does ship a
#     working system mksquashfs 4.4 in /usr/sbin, but apptainer ignores $PATH
#     for its bundled helpers, so it cannot be redirected there without root.
#   * DeltaAI's apptainer 1.4.2 converts the same image without complaint.
#   * DeltaAI shares the /work/nvme Lustre filesystem with Delta, so images
#     written here are immediately visible to Delta with no transfer at all.
#   * `apptainer pull --arch` only downloads and squashes layers — it never
#     executes them — so an aarch64 host can produce the x86_64 image too.
#     Verified: 8.8 GB x86_64 and 9.1 GB aarch64, both built on gh-login01.
#
# If DeltaAI is unavailable, hpc/containers/build_sif_polaris.pbs builds x86_64
# with --fakeroot on a Polaris compute node.
set -euo pipefail

TAG="${1:-latest}"
IMAGE="${AI_ROSSBY_IMAGE:-ghcr.io/awikner/physicsnemo-ai-rossby}"
DEST="${AI_ROSSBY_CONTAINER_DIR:-/work/nvme/bdiu/awikner/containers}"

command -v apptainer >/dev/null 2>&1 || {
    echo "error: apptainer not on PATH. Run this on a DeltaAI login node." >&2
    exit 1
}

# Refuse to run where the conversion is known to fail, rather than burning 20
# minutes of download to die in mksquashfs.
if [ "${AI_ROSSBY_FORCE_HOST:-0}" != 1 ]; then
    mks=/usr/libexec/apptainer/bin/mksquashfs
    if [ -x "$mks" ] && "$mks" -version 2>/dev/null | head -1 | grep -q '4\.7\.'; then
        echo "error: this host's apptainer bundles mksquashfs $("$mks" -version 2>/dev/null | head -1 | awk '{print $3}')," >&2
        echo "       which segfaults converting this image (see this script's header)." >&2
        echo "       Run on a DeltaAI login node instead, or set" >&2
        echo "       AI_ROSSBY_FORCE_HOST=1 to try anyway." >&2
        exit 1
    fi
fi

# Cache and unpack go on NODE-LOCAL disk, not /work.
#
# `apptainer pull` needs roughly 3x the final .sif: compressed layer blobs in the
# cache (~10 GB) plus a fully-expanded rootfs (~25 GB) before it squashes the
# image. Running that under /work failed mid-unpack with EDQUOT, because the bdiu
# project quota is shared across /work/nvme and /work/hdd and /work/hdd is
# already over its own soft limit (20.79T of 19.53T) — see
# docs/dev/context/lat-orientation-audit.md. DeltaAI login nodes have ~1.6 TB of
# local /tmp, so only the finished ~9 GB .sif lands on shared storage. $HOME is
# never used either: same small-quota trap that bit UV_CACHE_DIR on Midway3.
pick_scratch() {
    local cand avail
    for cand in "${AI_ROSSBY_LOCAL_SCRATCH:-}" /tmp /var/tmp; do
        [ -n "$cand" ] && [ -d "$cand" ] || continue
        avail=$(df -BG --output=avail "$cand" 2>/dev/null | tail -1 | tr -dc '0-9')
        if [ -n "$avail" ] && [ "$avail" -ge 80 ]; then echo "$cand"; return; fi
    done
    echo "${DEST%/*}"
}
LOCAL_SCRATCH="$(pick_scratch)/${USER}-apptainer"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${LOCAL_SCRATCH}/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${LOCAL_SCRATCH}/tmp}"

mkdir -p "$DEST" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "host  : $(hostname) ($(uname -m)), $(apptainer --version)"
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
    # Stage on local disk, then move the finished image onto Lustre, so a failed
    # conversion never leaves a partial multi-GB file against the shared quota.
    staged="${LOCAL_SCRATCH}/ai-rossby-${TAG}-${uname_arch}.sif"
    apptainer pull --force --arch "${docker_arch}" "$staged" \
        "docker://${IMAGE}:${TAG}"
    cp "$staged" "${out}.part" && mv "${out}.part" "$out"
    rm -f "$staged"
    echo "[ok]   $(du -h "$out" | cut -f1)  $out"
done

echo
echo "Delta and DeltaAI both read this directly (shared /work/nvme)."
echo "Sanity-check the native image:"
echo "  apptainer exec --nv ${DEST}/ai-rossby-${TAG}-$(uname -m).sif \\"
echo "    python -c 'import torch; print(torch.__version__, torch.version.cuda)'"
echo
echo "Then replicate to the clusters that do not share this filesystem:"
echo "  hpc/containers/replicate_sif.sh ${TAG} stampede3"
