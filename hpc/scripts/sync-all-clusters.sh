#!/usr/bin/env bash
# hpc/scripts/sync-all-clusters.sh
# Pull the ai-rossby branch and re-run `uv sync` on the clusters that still need
# a venv.
#
# ==> SUPERSEDED FOR MOST CLUSTERS BY THE CONTAINER. <==
#
# Delta, DeltaAI, Stampede3, Midway3, Polaris and Derecho now run out of the
# ai-rossby container image (see hpc/containers.md). To update them, rebuild and
# re-pull the image instead of syncing a venv:
#     hpc/containers/pull_sif.sh <tag>          # on a DeltaAI login node
#     hpc/containers/replicate_sif.sh <tag> stampede3
#
# DSI is the one remaining venv cluster: it has no container runtime and no
# module system, so it stays on `uv sync`. Keeping the machinery here also keeps
# the documented fallback in hpc/install.md executable if a cluster's apptainer
# breaks.
#
# Pass cluster names explicitly to sync a non-default cluster anyway.
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

# Default target is DSI only — everything else is containerized. The other
# cluster configs below are retained for the fallback path and can still be
# requested by name.
ALL_CLUSTERS="dsi"
targets="${*:-$ALL_CLUSTERS}"

# Optional extras the ai-rossby recipe needs on top of the CUDA extra. Without
# them `uv sync` prunes their packages and SFNO training breaks:
#   sfno-extras      — torch-harmonics / tensorly (required by the SFNO models)
#   utils-extras     — wandb / mlflow (logging backends)
#   datapipes-extras — zarr / xarray / netCDF4 / dask (PLASIM/ERA5 data I/O)
RECIPE_EXTRAS="--extra sfno-extras --extra utils-extras --extra datapipes-extras"

DEFAULT_SYNC="uv sync --extra cu12 $RECIPE_EXTRAS --group dev --python 3.12"

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
        SYNC="uv sync --extra cu12 $RECIPE_EXTRAS --group dev --python 3.12"  # Nsight 12.8
        CACHE="/work/nvme/bdiu/awikner/.uv-cache" ;;
      deltaai)
        REPO="/work/nvme/bdiu/awikner/physicsnemo"; VENV=".venv-deltaai"  # shared /work with Delta
        # The inherited conda module's wandb is broken (protobuf-generated-code
        # mismatch) and, via --system-site-packages, uv treats utils-extras'
        # wandb as already-satisfied — so it never lands a venv-local copy and
        # `import physicsnemo...logging` crashes. Force a fresh wandb into the
        # venv LAST (after the torch uninstall; wandb pulls no torch) so it
        # shadows the conda copy on sys.path.
        SYNC="module load python/miniforge3_pytorch/2.10.0 && source .venv-deltaai/bin/activate && uv pip install -e \".[sfno-extras,utils-extras,datapipes-extras]\" && uv pip install --group dev && uv pip uninstall torch torchvision triton && uv pip install -U --force-reinstall wandb"
        CACHE="/work/nvme/bdiu/awikner/.uv-cache" ;;
      stampede3)
        REPO="\$WORK/physicsnemo"                                       # $WORK persistent; $SCRATCH is not
        SYNC="uv sync --extra cu12 $RECIPE_EXTRAS --group dev --python 3.12"  # Nsight 12.8
        CACHE="\$SCRATCH/.uv-cache"
        EXTRA_ENV="export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1" ;;
      derecho)
        REPO="/glade/work/awikner/physicsnemo"
        SYNC="uv sync --extra cu129 $RECIPE_EXTRAS --group dev --python 3.12"  # Nsight 12.9
        CACHE="/glade/derecho/scratch/awikner/.uv-cache" ;;
      midway3)
        REPO="/project/pedramh/awikner/physicsnemo"
        SYNC="uv sync --extra cu129 $RECIPE_EXTRAS --group dev --python 3.12"  # Nsight 12.9
        CACHE="/scratch/midway3/awikner/.uv-cache" ;;
      dsi)
        REPO="/net/projects2/laude/awikner/physicsnemo"                # general_group = SLURM account; storage is the laude group
        SYNC="uv sync --extra cu129 $RECIPE_EXTRAS --group dev --python 3.12"  # general partition: driver 595 / nsys 2026.1.3
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
    # Build the remote script with the local per-cluster values baked in; \$HOME
    # etc. stay literal for remote expansion.
    remote_script="set -euo pipefail
export PATH=\$HOME/.local/bin:\$PATH
cd $REPO
git fetch origin
git checkout $BRANCH
git pull --ff-only origin $BRANCH
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT=$VENV
${CACHE:+export UV_CACHE_DIR=$CACHE}
$EXTRA_ENV
$SYNC
echo 'uv sync: OK'"
    # Feed it to a remote LOGIN bash via stdin. (Running `bash -lc "<multiline>"`
    # over ssh mis-parses: ssh re-joins argv, and the newline after -lc leaves -c
    # without an argument — a harmless but noisy `bash: -c` error — while the body
    # runs in the outer shell anyway. Piping to `bash -l` avoids both.)
    if ssh "$cluster" bash -l <<<"$remote_script"; then
        echo "  → OK"
    else
        echo "  → FAILED (see above; if it's a dirty tree: git checkout -- pyproject.toml uv.lock)"; rc=1
    fi
done
echo "Done."
exit $rc
