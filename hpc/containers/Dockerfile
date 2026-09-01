# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ai-rossby HPC image: dependencies only, code bind-mounted at run time.
#
# Deliberately separate from the repo-root Dockerfile. That one serves upstream
# PhysicsNeMo CI and builds the GNN/mesh stack (torch_scatter, pyg-lib, natten,
# torch_sparse) while omitting `sfno-extras` entirely, so it cannot run SFNO;
# it also bakes physicsnemo in non-editable and its `deploy` stage uninstalls
# wandb, which every ai-rossby training run needs.
#
# Base-image choice: 26.01-py3 ships torch 2.10.0a0 — the only NGC line that
# lands inside the `torch>=2.10.0,<2.11.0` window that pyproject.toml pins
# because torch 2.11 deadlocks SFNO DDP. It is a genuine multi-arch manifest
# (linux/amd64 + linux/arm64), so the same recipe serves DeltaAI's GH200.
# CUDA 13.1.1 requires driver r580+; Delta/DeltaAI are on 595.71.05 and
# Polaris on 580.65.06.
ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_CONTAINER}

ARG TARGETPLATFORM

# uv pinned to match the repo-root Dockerfile (bundles rustls-webpki >= 0.103.13,
# fixing GHSA-82j2-j2ch-gfr8).
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1
# The base image's Python is PEP 668 externally-managed; allow system installs.
ENV UV_BREAK_SYSTEM_PACKAGES=1

RUN uv pip install "pip>=23.2.1" "setuptools>=77.0.3"

# The NGC image ships /etc/pip/constraint.txt, which pins packaging==23.2 and
# the frameworks it bundles. Drop the packaging line so a later interactive
# `pip install` inside the container does not fight the lock's packaging==26.0.
#
# Deliberately NOT exported as UV_CONSTRAINT (which is what the repo-root
# Dockerfile does): that file is only needed to stop a resolver from replacing
# NGC's torch, and this image achieves that structurally instead — the closure is
# fully pinned and installed with --no-deps, with torch stripped out. Pointing uv
# at the constraint file would only add a second, opaque set of pins that can
# conflict with the lock.
RUN FILE="/etc/pip/constraint.txt"; \
    if [ -f "$FILE" ]; then sed -i '/packaging/d' "$FILE"; fi

# System (non-Python) prerequisites available on both arches:
#   libgeos/libproj        - cartopy
#   git-lfs, zip, rsync    - hpc/scripts data-movement helpers
#   libgl1, graphviz       - vtk / pyvista
# cdo is handled separately below (amd64 only). burn7/pumaburner is NOT
# installed at all: it is an out-of-tree compiled binary whose path comes from
# the postprocessor config's `pumaburner_script` key, so it is bind-mounted from
# the host instead. See tools/data/plasim/postprocessor/README.md.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev libproj-dev proj-bin \
        git-lfs graphviz libgl1 zip unzip rsync && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# cdo is amd64-only: the NGC arm64 image's apt sources have no installation
# candidate for it. That is fine functionally — cdo is used only by the PLASIM
# postprocessor chain (tools/data/plasim/postprocessor), which runs as a CPU job
# on Stampede3/Delta and never on DeltaAI's GH200. The build fails loudly on
# amd64 if it ever disappears there.
RUN if [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends cdo && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "skipping cdo on $TARGETPLATFORM (no arm64 candidate; PLASIM chain is x86_64-only)"; \
    fi

# Dependency closure, rendered from uv.lock so the image matches what
# `uv sync` resolves on the clusters today. See make_requirements.sh for why
# torch and its CUDA wheels are excluded.
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY hpc/containers/make_requirements.sh hpc/containers/
RUN hpc/containers/make_requirements.sh /build/requirements.txt

# torch-harmonics is handled separately: 0.9.x is published as x86_64-only
# wheels with NO sdist, so on aarch64 there is nothing on PyPI to install and
# nothing to build from either (which also means hpc/deltaai.md's
# `uv pip install --no-binary torch-harmonics` recipe cannot work on 0.9.x).
# Take it from the upstream git tag there instead, at the version the lock pins.
#
# --no-deps: the export is already a complete, fully-pinned closure, so
# resolution is neither needed nor wanted — it would let torch back in via a
# transitive edge.
RUN TH_VER="$(grep -E '^torch-harmonics==' /build/requirements.txt | cut -d'=' -f3)" && \
    echo "torch-harmonics pinned at ${TH_VER}" && \
    grep -v -E '^torch-harmonics==' /build/requirements.txt > /build/req-main.txt && \
    uv pip install --no-deps -r /build/req-main.txt && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --no-deps --no-build-isolation \
            "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@v${TH_VER}"; \
    else \
        uv pip install --no-deps "torch-harmonics==${TH_VER}"; \
    fi

# Reachable from the recipes but present in no pyproject extra:
#   cartopy      - examples/weather/ai_rossbypalooza/physmetrics/plot_tcwv_bias.py
#   natsort      - tools/data/plasim/postprocessor/postprocess_data.py
#   ruamel.yaml  - tools/data/plasim/postprocessor/src/utils.py
# Pinned even though they are outside uv.lock, so the image stays reproducible.
# Folding them into a pyproject extra is the proper fix, but that regenerates the
# lock and would perturb every cluster venv mid-migration.
# Note cartopy publishes no linux-aarch64 wheel, so on arm64 it builds from the
# sdist against the libgeos/libproj headers installed above.
RUN uv pip install "cartopy==0.25.0" "natsort==8.4.0" "ruamel.yaml==0.19.1"

# PhysMetrics.Weather (ai-rossbypalooza physics diagnostics). Upstream's
# pyproject.toml names the package `physeval*` while the source tree is
# src/physmetrics_weather/. docs/dev/context/PhysMetrics.md also has us add a
# missing requires-python, but upstream has since added it, so inserting it
# unconditionally now produces a duplicate TOML key — hence the grep guard.
RUN git clone --depth 1 https://github.com/Emmakast/PhysMetrics.Weather.git /tmp/physmetrics && \
    cd /tmp/physmetrics && \
    sed -i 's/physeval/physmetrics/g' pyproject.toml && \
    if ! grep -q '^requires-python' pyproject.toml; then \
        sed -i '/^readme = "README.md"$/a requires-python = ">=3.10"' pyproject.toml; \
    fi && \
    uv pip install --no-deps . && \
    rm -rf /tmp/physmetrics

RUN uv cache clean && rm -rf /build

# HDF5 file locking breaks on the parallel filesystems these jobs bind-mount;
# test/conftest.py:20-22 forces this for exactly that reason, so bake it in
# rather than relying on every launcher to remember.
ENV HDF5_USE_FILE_LOCKING=FALSE
# The makani allocator fix, currently re-exported by every training sbatch.
# Must be set before CUDA initializes.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARG PHYSICSNEMO_GIT_HASH
ENV PHYSICSNEMO_GIT_HASH=${PHYSICSNEMO_GIT_HASH:-unknown}

WORKDIR /workspace
