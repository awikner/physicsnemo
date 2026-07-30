#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# One-time Stampede3 environment bootstrap for MoWE training.
# Expects the repo working tree at $WORK/physicsnemo (rsync'd; the fork is
# not on a public remote). Creates $WORK/physicsnemo/.venv-mowe with the
# minimal dependency set for the ai_rossbypalooza recipe (DiT gate; no
# SFNO/DALI extras needed).
#   bash setup_stampede3.sh
set -euo pipefail

REPO="${REPO:-$WORK/physicsnemo}"
cd "$REPO"
module load python 2>/dev/null || true

python3 -m venv .venv-mowe
source .venv-mowe/bin/activate
pip install --upgrade pip --quiet
# torch pinned <2.11 per the fork's DDP guidance; cu126 wheels run on H100.
pip install --quiet "torch==2.10.*" --index-url https://download.pytorch.org/whl/cu126
pip install --quiet \
    zarr xarray cftime numcodecs \
    hydra-core omegaconf wandb \
    timm jaxtyping einops tensordict s3fs nvtx \
    treelib termcolor gitpython warp-lang pytest
pip install --quiet -e . --no-deps
# tensordict et al. may have dragged in a newer torch; re-pin LAST
# (fork guidance: torch<2.11 — 2.11/2.12 regress DDP).
pip install --quiet --force-reinstall "torch==2.10.1" \
    --index-url https://download.pytorch.org/whl/cu126

python - <<'EOF'
import torch
from physicsnemo.models.dit import DiT
print("setup ok: torch", torch.__version__, "| DiT importable")
EOF
echo "Done. Activate with: source $REPO/.venv-mowe/bin/activate"
