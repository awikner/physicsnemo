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
# NB: a plain `git pull --ff-only` fails on a dirty tree. If a clone has local
# edits (e.g. a hand-applied pyproject fix), reconcile once before running this:
#     ssh <cluster> 'cd <repo> && git checkout -- pyproject.toml uv.lock'

set -uo pipefail   # not -e: one cluster failing must not abort the rest
BRANCH="${1:-ai-rossby}"
[ $# -gt 0 ] && shift

# Ordered so output is deterministic (bash assoc arrays are unordered).
ALL_CLUSTERS=(delta deltaai stampede3 derecho midway3 dsi)
targets=("${@:-${ALL_CLUSTERS[@]}}")

# Persistent work/project filesystems only — never scratch. Verified during Phase 9 setup.
declare -A REPO_DIRS=(
    [delta]="/work/nvme/bdiu/awikner/physicsnemo"
    [deltaai]="/work/nvme/bdiu/awikner/physicsnemo"          # shared /work (Lustre) with Delta
    [stampede3]="\$WORK/physicsnemo"                          # $WORK persistent; $SCRATCH is not
    [derecho]="/glade/work/awikner/physicsnemo"              # /glade/work persistent; scratch is not
    [midway3]="/project/pedramh/awikner/physicsnemo"
    [dsi]="/net/projects2/laude/awikner/physicsnemo"         # general_group is only the SLURM account; storage is the laude group
)

# Separate venv per cluster even when the repo path is shared (different GPU
# hardware / CUDA build). DeltaAI rides Delta's clone but its own aarch64 venv.
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
#   cu129 = CUDA 12.9 wheels (Derecho, Midway3, DSI — system Nsight 12.9+)
#   DeltaAI uses Option A: reuse the site's torch 2.10+cu129 module, no cu-wheel.
DEFAULT_SYNC="uv sync --extra cu12 --group dev --python 3.12"
declare -A SYNC_CMD=(
    [delta]="uv sync --extra cu12 --group dev --python 3.12"       # Nsight 12.8
    [stampede3]="uv sync --extra cu12 --group dev --python 3.12"   # Nsight 12.8
    [derecho]="uv sync --extra cu129 --group dev --python 3.12"    # Nsight 12.9
    [midway3]="uv sync --extra cu129 --group dev --python 3.12"    # Nsight 12.9
    [dsi]="uv sync --extra cu129 --group dev --python 3.12"        # general partition: driver 595 / nsys 2026.1.3
    # DeltaAI Option A — reuse the miniforge torch 2.10+cu129 module instead of a cu-wheel:
    [deltaai]="module load python/miniforge3_pytorch/2.10.0 && source .venv-deltaai/bin/activate && uv pip install -e . && uv pip install --group dev"
)

# Per-cluster uv cache dir — a small $HOME overflows on the multi-GB cu-wheel
# tree, so cache on the roomy work/scratch filesystem. uv creates it if missing.
declare -A CACHE_DIRS=(
    [delta]="/work/nvme/bdiu/awikner/.uv-cache"
    [deltaai]="/work/nvme/bdiu/awikner/.uv-cache"
    [stampede3]="\$SCRATCH/.uv-cache"
    [derecho]="/glade/derecho/scratch/awikner/.uv-cache"
    [midway3]="/scratch/midway3/awikner/.uv-cache"
    [dsi]="/net/scratch/awikner/.uv-cache"
)

# Extra per-cluster env before uv. Stampede3's login node caps vmem at 8 GB, so
# throttle uv's concurrency there or it OOMs mid-resolve (":" = no-op elsewhere).
declare -A EXTRA_ENV=(
    [stampede3]="export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1"
)

rc=0
for cluster in "${targets[@]}"; do
    repo="${REPO_DIRS[$cluster]:-}"
    venv="${VENV_NAMES[$cluster]:-.venv}"
    sync_cmd="${SYNC_CMD[$cluster]:-$DEFAULT_SYNC}"
    cache="${CACHE_DIRS[$cluster]:-}"
    extra_env="${EXTRA_ENV[$cluster]:-:}"
    echo "── $cluster ─────────────────────────────────────────────"
    if [ -z "$repo" ]; then
        echo "  → SKIP (no repo path configured)"; rc=1; continue
    fi
    # Fail fast with a clear message instead of hanging on an interactive prompt.
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
        ${cache:+export UV_CACHE_DIR=$cache}
        $extra_env
        $sync_cmd
        echo 'uv sync: OK'
    "; then
        echo "  → OK"
    else
        echo "  → FAILED (see above; if it's a dirty tree: git checkout -- pyproject.toml uv.lock)"; rc=1
    fi
done
echo "Done."
exit $rc
