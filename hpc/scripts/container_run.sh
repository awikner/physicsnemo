#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a command inside the ai-rossby container on any cluster.
#
#   hpc/scripts/container_run.sh python -m torch.distributed.run --standalone ...
#   hpc/scripts/container_run.sh pytest -m "smoke and cuda" -x -q test/
#
# Replaces the per-cluster venv-activation blocks (the `AI_ROSSBY_CLUSTER` case
# statements in train_*.sbatch / eval_*.sbatch and the `uname -m` switch in
# smoke_amip_si_multiyear_delta.sbatch). The AI_ROSSBY_* env contract is
# unchanged, so Hydra's ${oc.env:...} interpolation keeps working as-is; only
# AI_ROSSBY_VENV is superseded, by AI_ROSSBY_SIF.
#
# Knobs:
#   AI_ROSSBY_CLUSTER        delta|deltaai|stampede3|midway3|polaris|derecho
#                            (auto-detected if unset)
#   AI_ROSSBY_SIF            explicit .sif path (overrides the default location)
#   AI_ROSSBY_CONTAINER_TAG  image tag to select, default "latest"
#   AI_ROSSBY_CONTAINER_DIR  directory holding the .sif files
#   AI_ROSSBY_NV             1|0 to force GPU passthrough on/off (auto-detected)
#   AI_ROSSBY_BIND_EXTRA     extra comma-separated bind paths
#   AI_ROSSBY_PYTHONPATH     PYTHONPATH inside the container (default: repo root)
#
# The repo checkout is bind-mounted at its host path rather than copied into the
# image, and the host filesystems are bound with identical paths inside, so no
# path translation is needed and multiple worktrees keep working.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    sed -n '8,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//' >&2
    echo "error: no command given" >&2
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="$(uname -m)"

# ---------------------------------------------------------------- cluster ----
detect_cluster() {
    local h; h="$(hostname -f 2>/dev/null || hostname)"
    case "$h" in
        *midway3*)        echo midway3;  return ;;
        *stampede3*)      echo stampede3; return ;;
        *delta*|dt-*|gh-*) [ "$ARCH" = aarch64 ] && echo deltaai || echo delta; return ;;
    esac
    # Compute nodes often have opaque names; fall back to filesystem signatures.
    if   [ -d /eagle ] || [ -d /lus/eagle ];      then echo polaris
    elif [ -d /glade ];                           then echo derecho
    elif [ -d /work/nvme/bdiu ]; then [ "$ARCH" = aarch64 ] && echo deltaai || echo delta
    elif [ -d /work2 ];                           then echo stampede3
    elif [ -d /project/pedramh ];                 then echo midway3
    else echo unknown
    fi
}
CLUSTER="${AI_ROSSBY_CLUSTER:-$(detect_cluster)}"

# ------------------------------------------------- per-cluster resolution ----
# MODULE_CMD is run before apptainer: on Stampede3, Midway3, Polaris and Derecho
# apptainer is a module (and on Stampede3/Polaris it exists only on compute
# nodes). Delta and DeltaAI ship it natively in /usr/bin.
MODULE_CMD=":"
case "$CLUSTER" in
    delta)
        DEF_DIR=/work/nvme/bdiu/awikner/containers
        BINDS="/work/nvme,/work/hdd,/scratch"
        ;;
    deltaai)
        DEF_DIR=/work/nvme/bdiu/awikner/containers   # same Lustre as Delta
        BINDS="/work/nvme,/scratch"
        ;;
    stampede3)
        DEF_DIR="${WORK:-/work2/09979/awikner/stampede3}/containers"
        BINDS="/work2,/scratch,${HOME}"
        MODULE_CMD="module load tacc-apptainer"
        ;;
    midway3)
        DEF_DIR=/project/pedramh/awikner/containers
        BINDS="/project/pedramh,/scratch/midway3"
        MODULE_CMD="module load apptainer"
        ;;
    polaris)
        DEF_DIR=/eagle/lighthouse-uchicago/members/awikner/containers
        # /eagle really resolves to /lus/eagle/projects, so bind both spellings.
        # /opt/cray + /soft/libraries carry the CXI/libfabric stack that NCCL
        # needs for multi-node; harmless on single-node runs.
        BINDS="/eagle,/lus/eagle,/opt/cray,/soft/libraries,/var/spool/pbs"
        MODULE_CMD="module use /soft/modulefiles && module load spack-pe-base && module load apptainer"
        ;;
    derecho)
        DEF_DIR=/glade/work/awikner/containers
        BINDS="/glade/work,/glade/derecho/scratch,/glade/campaign"
        MODULE_CMD="module load apptainer"
        ;;
    *)
        echo "error: could not determine cluster (hostname $(hostname), arch ${ARCH})." >&2
        echo "       Set AI_ROSSBY_CLUSTER to one of:" >&2
        echo "       delta deltaai stampede3 midway3 polaris derecho" >&2
        exit 2
        ;;
esac

CONTAINER_DIR="${AI_ROSSBY_CONTAINER_DIR:-$DEF_DIR}"
TAG="${AI_ROSSBY_CONTAINER_TAG:-latest}"
SIF="${AI_ROSSBY_SIF:-${CONTAINER_DIR}/ai-rossby-${TAG}-${ARCH}.sif}"

if [ ! -f "$SIF" ]; then
    echo "error: container image not found: $SIF" >&2
    echo "       Pull it with hpc/containers/pull_sif.sh (Delta), or replicate" >&2
    echo "       it with hpc/containers/replicate_sif.sh, or set AI_ROSSBY_SIF." >&2
    exit 1
fi

# Only bind paths that exist — apptainer errors out on a missing source, and
# not every cluster has every path on every node (e.g. /scratch/midway2 is
# absent on some Midway3 nodes, and /glade/campaign is not always mounted).
BIND_LIST=""
for p in $(echo "${BINDS},${AI_ROSSBY_BIND_EXTRA:-}" | tr ',' ' '); do
    [ -n "$p" ] && [ -e "$p" ] && BIND_LIST="${BIND_LIST:+$BIND_LIST,}$p"
done
# The repo itself may live outside the bound roots (e.g. a worktree elsewhere).
repo_covered=0
IFS=',' read -r -a _bound <<< "$BIND_LIST"
for b in "${_bound[@]}"; do
    case "$REPO/" in "$b"/*) repo_covered=1; break ;; esac
done
[ "$repo_covered" = 1 ] || BIND_LIST="${BIND_LIST:+$BIND_LIST,}$REPO"

# ------------------------------------------------------------ GPU passthrough --
# --nv on a CPU-only node makes apptainer warn and can fail; decide by whether
# a GPU is actually visible, honouring an explicit override.
if [ -n "${AI_ROSSBY_NV:-}" ]; then
    NV_FLAG=$([ "$AI_ROSSBY_NV" = 1 ] && echo "--nv" || echo "")
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG="--nv"
else
    NV_FLAG=""
fi

# ------------------------------------------------------------- env hygiene ----
# apptainer inherits the host environment, and --cleanenv is not an option here
# because SLURM_* / PMI_* / PBS_* must reach DistributedManager. So scrub only
# what actively breaks the container:
#
#   CC/CXX  - DeltaAI's Cray PE sets CXX=CC, which breaks TorchInductor's C++
#             build inside the container exactly as it does outside
#             (hpc/deltaai.md: 40 failures -> 0 once overridden).
#   LD_LIBRARY_PATH - host module paths point at host glibc/CUDA builds that
#             must not shadow the image's own.
#   PYTHONPATH - a host venv on PYTHONPATH would shadow the bind-mounted repo.
#   VIRTUAL_ENV - stale venv activation confuses uv/pip inside the container.
unset CC CXX FC F77 F90 LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME VIRTUAL_ENV
# PYTHONPATH is set, not inherited: a host venv on it would shadow the
# bind-mounted checkout. AI_ROSSBY_PYTHONPATH lets a caller point at a different
# worktree (e.g. the amip-v2 tree, whose jobs used to borrow another tree's venv).
export APPTAINERENV_PYTHONPATH="${AI_ROSSBY_PYTHONPATH:-$REPO}"
export APPTAINERENV_PYTHONNOUSERSITE=1

# Polaris multi-node NCCL needs the host CXI/libfabric stack visible inside.
if [ "$CLUSTER" = polaris ] && [ "${AI_ROSSBY_POLARIS_FABRIC:-1}" = 1 ]; then
    export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/libfabric/2.2.0rc1/lib64:/soft/libraries/hwloc/lib:/opt/cray/pe/lib64"
fi

eval "$MODULE_CMD" >/dev/null 2>&1 || {
    echo "error: could not load the apptainer module on ${CLUSTER}:" >&2
    echo "       ${MODULE_CMD}" >&2
    echo "       On Stampede3 and Polaris apptainer exists only on compute nodes." >&2
    exit 1
}

command -v apptainer >/dev/null 2>&1 || {
    echo "error: apptainer not on PATH after '${MODULE_CMD}' (cluster ${CLUSTER})." >&2
    exit 1
}

# Diagnostics go to stderr so that `VAR=$(container_run.sh python -c ...)`
# captures only the payload's stdout.
echo "container_run: cluster=${CLUSTER} arch=${ARCH} nv=${NV_FLAG:-none}" >&2
echo "container_run: sif=${SIF}" >&2
echo "container_run: binds=${BIND_LIST}" >&2

exec apptainer exec ${NV_FLAG} \
    --bind "${BIND_LIST}" \
    --pwd "${PWD}" \
    "${SIF}" "$@"
