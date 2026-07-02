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

set -uo pipefail   # not -e: one cluster failing must not abort the rest
BRANCH="${1:-ai-rossby}"
[ $# -gt 0 ] && shift

# Ordered so output is deterministic (bash assoc arrays are unordered).
ALL_CLUSTERS=(delta deltaai stampede3 derecho midway3 dsi)
targets=("${@:-${ALL_CLUSTERS[@]}}")

# Persistent work/project filesystems only — never scratch. Paths marked
# "verify" are TBD until first login on that cluster (see hpc/<cluster>.md).
declare -A REPO_DIRS=(
    [delta]="/work/nvme/bdiu/awikner/physicsnemo"
    [deltaai]="/work/nvme/bdiu/awikner/physicsnemo"          # shared /work with Delta? verify
    [stampede3]="\$WORK/physicsnemo"                          # $WORK persistent; $SCRATCH is not
    [derecho]="/glade/work/awikner/physicsnemo"              # /glade/work persistent; scratch is not
    [midway3]="/project/pedramh/awikner/physicsnemo"         # verify pedramh project path
    [dsi]="/net/projects/general_group/awikner/physicsnemo"  # verify path on first login
)

# Separate venv per cluster even when the repo path is shared (different GPU
# hardware / CUDA build). DeltaAI rides Delta's clone but its own venv.
declare -A VENV_NAMES=(
    [delta]=".venv"
    [deltaai]=".venv-deltaai"
    [stampede3]=".venv"
    [derecho]=".venv"
    [midway3]=".venv"
    [dsi]=".venv"
)

# Per-cluster sync command, chosen so torch's CUDA matches each site's system
# Nsight profiler (see the Phase 9 plan § 9f and hpc/<cluster>.md § Profiling):
#   cu12  = CUDA 12.8 wheels (Delta, Stampede3 — system Nsight 12.8)
#   cu129 = CUDA 12.9 wheels (Derecho, Midway3 — system Nsight 12.9)
#   DeltaAI uses Option A: reuse the site's torch 2.10+cu129 module, no cu-wheel.
DEFAULT_SYNC="uv sync --extra cu12 --group dev --python 3.12"
declare -A SYNC_CMD=(
    [delta]="uv sync --extra cu12 --group dev --python 3.12"       # Nsight 12.8
    [stampede3]="uv sync --extra cu12 --group dev --python 3.12"   # Nsight 12.8
    [derecho]="uv sync --extra cu129 --group dev --python 3.12"    # Nsight 12.9 (cu129 = new extra)
    [midway3]="uv sync --extra cu129 --group dev --python 3.12"    # Nsight 12.9 (Option A if module torch>=2.10 — verify)
    [dsi]="$DEFAULT_SYNC"                                          # TBD: match to DSI system Nsight
    # DeltaAI Option A — reuse the miniforge torch 2.10+cu129 module instead of a cu-wheel:
    [deltaai]="module load python/miniforge3_pytorch/2.10.0 && source .venv-deltaai/bin/activate && uv pip install -e . && uv pip install --group dev"
)

rc=0
for cluster in "${targets[@]}"; do
    repo="${REPO_DIRS[$cluster]:-}"
    venv="${VENV_NAMES[$cluster]:-.venv}"
    sync_cmd="${SYNC_CMD[$cluster]:-$DEFAULT_SYNC}"
    echo "── $cluster ─────────────────────────────────────────────"
    if [ -z "$repo" ]; then
        echo "  → SKIP (no repo path configured)"; rc=1; continue
    fi
    # First check the ControlMaster socket is live so we fail fast with a clear
    # message instead of hanging on an interactive auth prompt.
    if ! ssh -O check "$cluster" &>/dev/null; then
        echo "  → SKIP (no live connection — run: morning-login $cluster)"; rc=1; continue
    fi
    if ssh "$cluster" bash -lc "
        set -euo pipefail
        cd $repo
        git fetch origin
        git checkout $BRANCH
        git pull --ff-only origin $BRANCH
        unset VIRTUAL_ENV
        export UV_PROJECT_ENVIRONMENT=$venv
        $sync_cmd
        echo 'uv sync: OK'
    "; then
        echo "  → OK"
    else
        echo "  → FAILED (see above; SSH in and resolve manually)"; rc=1
    fi
done
echo "Done."
exit $rc
