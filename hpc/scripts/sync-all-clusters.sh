#!/usr/bin/env bash
# hpc/scripts/sync-all-clusters.sh
# Pull the ai-rossby branch and re-run `uv sync` on all configured clusters.
# Requires active ControlMaster connections (run `morning-login` first) — each
# `ssh <alias>` below reuses the socket opened in the morning, so no MFA here.
#
# Usage:  sync-all-clusters.sh [branch] [cluster …]
#   branch    default: ai-rossby
#   cluster … default: all configured clusters; pass names to limit
#             e.g.  sync-all-clusters.sh ai-rossby deltaai midway3
#
# Only touches each cluster's persistent code filesystem (repo + venv). Training
# data (Zarr archives) live on scratch and are NOT managed here.
#
# Portable to macOS's stock bash 3.2 (no associative arrays — per-cluster config
# lives in the `cluster_cfg` case below).
#
# NB: a plain `git pull --ff-only` fails on a dirty tree. If a clone has local
# edits (e.g. a hand-applied pyproject fix), reconcile once before running this:
#     ssh <cluster> 'cd <repo> && git checkout -- pyproject.toml uv.lock'

set -uo pipefail   # not -e: one cluster failing must not abort the rest
BRANCH="${1:-ai-rossby}"
[ $# -gt 0 ] && shift

ALL_CLUSTERS="delta deltaai stampede3 derecho midway3 dsi"
targets="${*:-$ALL_CLUSTERS}"

DEFAULT_SYNC="uv sync --extra cu12 --group dev --python 3.12"

# Per-cluster config. Sets REPO / VENV / SYNC / CACHE / EXTRA_ENV for cluster $1.
#   REPO   — persistent code filesystem (never scratch); \$WORK etc. expand remotely
#   VENV   — separate per cluster (different GPU / CUDA build)
#   SYNC   — CUDA extra chosen to match the site's system Nsight (plan § 9f):
#            cu12=CUDA 12.8 (Delta, Stampede3); cu129=CUDA 12.9 (Derecho, Midway3,
#            DSI); DeltaAI uses Option A (reuse its torch 2.10+cu129 module).
#   CACHE  — uv cache off the small $HOME (multi-GB cu-wheel tree); uv creates it
#   EXTRA_ENV — extra exports before uv (Stampede3's 8 GB login vmem cap → throttle)
cluster_cfg() {
    REPO=""; VENV=".venv"; SYNC="$DEFAULT_SYNC"; CACHE=""; EXTRA_ENV=":"
    case "$1" in
      delta)
        REPO="/work/nvme/bdiu/awikner/physicsnemo"
        SYNC="uv sync --extra cu12 --group dev --python 3.12"          # Nsight 12.8
        CACHE="/work/nvme/bdiu/awikner/.uv-cache" ;;
      deltaai)
        REPO="/work/nvme/bdiu/awikner/physicsnemo"; VENV=".venv-deltaai"  # shared /work with Delta
        SYNC="module load python/miniforge3_pytorch/2.10.0 && source .venv-deltaai/bin/activate && uv pip install -e . && uv pip install --group dev && uv pip uninstall torch torchvision triton"
        CACHE="/work/nvme/bdiu/awikner/.uv-cache" ;;
      stampede3)
        REPO="\$WORK/physicsnemo"                                       # $WORK persistent; $SCRATCH is not
        SYNC="uv sync --extra cu12 --group dev --python 3.12"          # Nsight 12.8
        CACHE="\$SCRATCH/.uv-cache"
        EXTRA_ENV="export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1" ;;
      derecho)
        REPO="/glade/work/awikner/physicsnemo"
        SYNC="uv sync --extra cu129 --group dev --python 3.12"         # Nsight 12.9
        CACHE="/glade/derecho/scratch/awikner/.uv-cache" ;;
      midway3)
        REPO="/project/pedramh/awikner/physicsnemo"
        SYNC="uv sync --extra cu129 --group dev --python 3.12"         # Nsight 12.9
        CACHE="/scratch/midway3/awikner/.uv-cache" ;;
      dsi)
        REPO="/net/projects2/laude/awikner/physicsnemo"                # general_group = SLURM account; storage is the laude group
        SYNC="uv sync --extra cu129 --group dev --python 3.12"         # general partition: driver 595 / nsys 2026.1.3
        CACHE="/net/scratch/awikner/.uv-cache" ;;
      *) REPO="" ;;
    esac
}

rc=0
for cluster in $targets; do
    cluster_cfg "$cluster"
    echo "── $cluster ─────────────────────────────────────────────"
    if [ -z "$REPO" ]; then
        echo "  → SKIP (unknown cluster '$cluster')"; rc=1; continue
    fi
    # Fail fast with a clear message instead of hanging on an interactive prompt.
    if ! ssh -O check "$cluster" &>/dev/null; then
        echo "  → SKIP (no live connection — run: morning-login $cluster)"; rc=1; continue
    fi
    if ssh "$cluster" bash -lc "
        set -euo pipefail
        export PATH=\$HOME/.local/bin:\$PATH   # ensure uv is found (not all ~/.bashrc add it)
        cd $REPO
        git fetch origin
        git checkout $BRANCH
        git pull --ff-only origin $BRANCH
        unset VIRTUAL_ENV
        export UV_PROJECT_ENVIRONMENT=$VENV
        ${CACHE:+export UV_CACHE_DIR=$CACHE}
        $EXTRA_ENV
        $SYNC
        echo 'uv sync: OK'
    "; then
        echo "  → OK"
    else
        echo "  → FAILED (see above; if it's a dirty tree: git checkout -- pyproject.toml uv.lock)"; rc=1
    fi
done
echo "Done."
exit $rc
