#!/usr/bin/env bash
# Recreate the ai-rossby physicsnemo uv environment on Derecho.
#
# Portable: defaults to the RUNNING user's own /glade/work + scratch. Override any
# of REPO / BRANCH / VENV / REMOTE / UV_CACHE_DIR via the environment, e.g.
#   REPO=/glade/work/$USER/code/physicsnemo BRANCH=ai-rossbypalooza ./recreate_airossby_env.sh
#
# Canonical recipe = the Derecho case of hpc/scripts/sync-all-clusters.sh
# (cu129 + sfno/utils/datapipes extras + python 3.12 + torch<2.11).
set -euo pipefail

REPO="${REPO:-/glade/work/$USER/physicsnemo}"
BRANCH="${BRANCH:-ai-rossby}"
VENV="${VENV:-.venv}"
REMOTE="${REMOTE:-/glade/work/awikner/physicsnemo}"   # default: local Derecho repo (world-readable, no GitHub needed); override w/ https://github.com/awikner/physicsnemo.git off-Derecho
CACHE="${UV_CACHE_DIR:-/glade/derecho/scratch/$USER/.uv-cache}"

echo "Building ai-rossby env for user '$USER':"
echo "  repo   = $REPO   (branch $BRANCH)"
echo "  venv   = $REPO/$VENV"
echo "  cache  = $CACHE"
echo "  remote = $REMOTE"

# 0. uv (install if missing)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. repo on the requested branch (clone into the user's space if absent)
[ -d "$REPO/.git" ] || git clone "$REMOTE" "$REPO"
cd "$REPO"
git fetch origin && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"

# 2. uv sync  -- the three extras are MANDATORY (uv prunes them otherwise):
#    sfno-extras  = torch-harmonics/tensorly (SFNO models)
#    utils-extras = wandb/mlflow (logging)
#    datapipes-extras = zarr/xarray/netCDF4/dask (PLASIM/ERA5 I/O)
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_CACHE_DIR="$CACHE"          # keep the multi-GB wheel cache off small $HOME
uv sync --extra cu129 \
        --extra sfno-extras --extra utils-extras --extra datapipes-extras \
        --group dev --python 3.12

# 3. (optional) needed only to load PanguWeather .tar checkpoints for translation
uv pip install ruamel.yaml

echo "ai-rossby env ready: $REPO/$VENV  (python 3.12, torch 2.10+cu129)"
