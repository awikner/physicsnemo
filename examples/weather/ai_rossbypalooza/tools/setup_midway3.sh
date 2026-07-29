#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# One-time Midway3 env bootstrap for MoWE training. Repo working tree
# expected at /scratch/midway3/$USER/physicsnemo (rsync'd).
#   bash setup_midway3.sh
set -euo pipefail
REPO="${REPO:-/scratch/midway3/$USER/physicsnemo}"
cd "$REPO"
module load python 2>/dev/null || true

python3 -m venv .venv-mowe
source .venv-mowe/bin/activate
pip install --upgrade pip --quiet
pip install --quiet "torch==2.10.0" torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --quiet \
    zarr xarray cftime numcodecs \
    hydra-core omegaconf wandb \
    timm jaxtyping einops tensordict s3fs nvtx \
    treelib termcolor gitpython warp-lang psutil pytest
pip install --quiet -e . --no-deps
# Dep chain may have upgraded torch; re-pin LAST (fork: torch<2.11 for DDP).
pip install --quiet --force-reinstall "torch==2.10.0" "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu126
pip install --quiet "fsspec>=2026.7.0"

python - <<'EOF'
import torch
from physicsnemo.models.dit import DiT
print("setup ok: torch", torch.__version__, "| DiT importable")
EOF
echo "Done: source $REPO/.venv-mowe/bin/activate"
