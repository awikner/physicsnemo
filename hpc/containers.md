<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The ai-rossby container

One image, built on NVIDIA's optimized PyTorch NGC container, replaces the
per-cluster `uv` environments that `hpc/scripts/sync-all-clusters.sh` used to
maintain. It carries **dependencies only** — the repo checkout is bind-mounted at
run time, so worktrees and the edit-without-reinstall loop are unchanged.

## Why

Six divergent `uv sync` invocations across three CUDA extras kept drifting off
the `torch>=2.10,<2.11` pin that `pyproject.toml` sets because torch 2.11
deadlocks SFNO DDP: the same `--extra cu129` line resolved **2.12.1** on
Derecho/Midway3/DSI and **2.11.0** then **2.10.0** on Stampede3. Most of the
per-cluster landmines in `docs/dev/context/` are environment artifacts rather
than hardware facts, and the image removes that whole class.

## What is in it

| | |
|---|---|
| Base | `nvcr.io/nvidia/pytorch:26.01-py3` — torch **2.10.0a0**, CUDA 13.1.1, Python 3.12, NCCL 2.29.2, cuDNN 9.17.1, apex, TE 2.11 |
| Arches | `linux/amd64` + `linux/arm64` from one manifest (arm64 serves DeltaAI's GH200) |
| Python deps | 211 packages, exported from `uv.lock` — identical pins to what the venvs resolve |
| Extras | `sfno-extras`, `utils-extras`, `datapipes-extras`, `muon-optimizers`, `dev` group |
| System | `cdo` (PLASIM extremes chain), GEOS/PROJ (cartopy), git-lfs, graphviz, libgl1 |
| Also | `cartopy`, `natsort`, `ruamel.yaml`, PhysMetrics.Weather — reachable from the recipes but in no pyproject extra |

**Not** in it, deliberately:

- **torch / torchvision / triton and torch's `nvidia-*` CUDA wheels.** The base
  image already provides them, and its apex / transformer-engine C extensions are
  compiled against that exact torch build. Note `2.10.0a0 < 2.10.0` under PEP 440,
  so the pyproject pin does *not* accept NGC's torch and a plain
  `uv pip install .` silently replaces it — the shadowing failure already
  documented in `hpc/deltaai.md`. `hpc/containers/make_requirements.sh` strips
  them, with a build-time guard that fails if the lock ever introduces an
  `nvidia-*` wheel that is not torch-owned.
- **The `cu12`/`cu129`/`cu13` extras.** Beyond torch they add only cuml,
  pylibraft, cupy and DALI. cuml/pylibraft/cupy have zero imports in
  `physicsnemo/` or `test/`, and every DALI import is `OptionalImport`-guarded.
  Dropping the extra is what eliminates the per-cluster CUDA-variant split.
- **`earth2grid`, `makani`** — documented-fragile upstream, already
  `@requires_module`-skipped.

## Running things

```bash
hpc/scripts/container_run.sh pytest -m "smoke and cuda" -x -q test/
hpc/scripts/container_run.sh python -m torch.distributed.run \
    --standalone --nproc-per-node=2 train_diffusion.py [overrides]
```

`container_run.sh` detects the cluster, resolves the `.sif` and bind list, and
scrubs the host environment. Knobs:

| Variable | Meaning |
|---|---|
| `AI_ROSSBY_CLUSTER` | `delta`\|`deltaai`\|`stampede3`\|`midway3`\|`polaris`\|`derecho` (auto-detected) |
| `AI_ROSSBY_SIF` | explicit `.sif` path |
| `AI_ROSSBY_CONTAINER_TAG` | image tag, default `latest` |
| `AI_ROSSBY_CONTAINER_DIR` | directory holding the `.sif` files |
| `AI_ROSSBY_NV` | `1`/`0` to force GPU passthrough (auto-detected via `nvidia-smi`) |
| `AI_ROSSBY_PYTHONPATH` | in-container `PYTHONPATH`, default the repo root |
| `AI_ROSSBY_BIND_EXTRA` | extra comma-separated bind paths |

Every other `AI_ROSSBY_*` / `WANDB_*` / `SLURM_*` / `PMI_*` variable passes
through untouched, so Hydra's `${oc.env:...}` interpolation keeps working.

### Env hygiene, and why `--cleanenv` is not used

`SLURM_*` / `PMI_*` / `PBS_*` must reach `physicsnemo/distributed/manager.py`,
so the environment is inherited and only the actively harmful parts are removed:

- `CC` / `CXX` / `FC` — DeltaAI's Cray PE sets `CXX=CC`, which breaks
  TorchInductor's C++ build (40 test failures → 0 once overridden). Unsetting
  them means the image's own toolchain is used on every cluster, so the
  per-cluster `export CXX=g++` workaround is gone.
- `LD_LIBRARY_PATH` / `LD_PRELOAD` — host module paths would shadow the image's
  CUDA/glibc. Re-populated only on Polaris, for the CXI/libfabric stack.
- `PYTHONPATH` / `VIRTUAL_ENV` — a host venv would shadow the bind-mounted repo.

## Per-cluster facts

| Cluster | Arch | Runtime | Where | Notes |
|---|---|---|---|---|
| Delta | x86_64 | apptainer 1.5.1, native | login + compute | Runs images fine, but **cannot build them** — its bundled mksquashfs 4.7.5 dies on this image |
| DeltaAI | **aarch64** | apptainer 1.4.2, native | login + compute | **The conversion hub** for both arches; shares `/work/nvme` with Delta so Delta needs no transfer |
| Stampede3 | x86_64 | `module load tacc-apptainer` | **compute only** | Compute nodes have no internet, so the `.sif` must be shipped in |
| Midway3 | x86_64 | `module load apptainer` | **compute only** | Every version fails to load on the login node (`Module ERROR: wrong # args`) |
| Polaris | x86_64 | `ml use /soft/modulefiles; ml spack-pe-base; ml apptainer` | **compute only** | Only fakeroot-capable host → builds its own, see `build_sif_polaris.pbs` |
| Derecho | x86_64 | `module load apptainer` | — | Retiring; all its jobs are single-node so no Cray-MPICH bind model needed |

GPU drivers must be r580+ for CUDA 13.1. Measured: Delta, DeltaAI **595.71.05**;
Polaris **580.65.06**. Stampede3 `h100` and Midway3 `pedramh-gpu` are pending.

## Distribution

```bash
# On a DeltaAI login node — see "Why DeltaAI converts, not Delta" below:
hpc/containers/pull_sif.sh latest              # writes both arches to /work/nvme/.../containers

# Then fan out (needs a live Globus session; see Step 0 of the migration plan):
hpc/containers/replicate_sif.sh latest stampede3
hpc/containers/replicate_sif.sh latest midway3     # scp: Midway3 has no Globus collection
hpc/containers/replicate_sif.sh latest derecho

# Polaris builds its own instead, straight from GHCR through the ALCF proxy:
qsub -v TAG=latest hpc/containers/build_sif_polaris.pbs
```

Measured image sizes: **8.8 GB** (x86_64), **9.1 GB** (aarch64).

Replication to Stampede3 over Globus is verified end-to-end (9.43 GB, ~1 min).
`replicate_sif.sh` drives a server-to-server transfer, so it does **not** have to
run on the source cluster — a workstation with the Globus CLI and a live session
is the usual case; it verifies the source with `globus ls`, not a local file test.

### Why DeltaAI converts, not Delta

Delta looks like the obvious hub — it has the data, native apptainer on login
nodes, and outbound internet. But its apptainer 1.5.1 bundles **mksquashfs 4.7.5
(2026/03/01), which is broken for an image this size**: `FATAL ERROR: Bug in
orderer` at the default 128 processors, and SIGSEGV (exit 139) at
`-processors 8`. The OCI download and rootfs extraction both succeed; only the
squashfs step dies. Delta *does* ship a working system mksquashfs 4.4 in
`/usr/sbin`, but apptainer ignores `$PATH` for its bundled helpers, so it cannot
be redirected there without root (tested). `pull_sif.sh` refuses to run on a host
whose bundled mksquashfs is 4.7.x rather than waste the download.

DeltaAI's apptainer 1.4.2 converts the same image cleanly, shares `/work/nvme`
with Delta so the output needs no transfer, and — because `apptainer pull --arch`
only downloads and squashes layers without ever executing them — produces the
**x86_64** image from aarch64 hardware too.

This is worth an NCSA ticket: a broken mksquashfs on Delta will bite anyone
building containers there, not only this project.

### Disk and quota

`apptainer pull` needs roughly **3x** the final image on disk (layer cache plus a
fully-expanded rootfs) before squashing. Running that under `/work` fails with
`EDQUOT`, because the `bdiu` project quota is shared across `/work/nvme` and
`/work/hdd` and `/work/hdd` is already over its own soft limit (20.79T of
19.53T) — see `docs/dev/context/lat-orientation-audit.md`. `pull_sif.sh` stages
everything on node-local `/tmp` (~1.6 TB on DeltaAI login nodes) and moves only
the finished image onto Lustre. Never let apptainer cache into `$HOME` — the same
small-quota trap that bit `UV_CACHE_DIR` on Midway3 and Stampede3.

## Rebuilding

`.github/workflows/container-ai-rossby.yml` builds both arches on GitHub-hosted
runners (`ubuntu-24.04` and `ubuntu-24.04-arm`, native — not QEMU, which matters
because torch-harmonics has no aarch64 wheel and must compile) and publishes a
multi-arch manifest to `ghcr.io/awikner/physicsnemo-ai-rossby`. It triggers on
changes to `pyproject.toml`, `uv.lock` or `hpc/containers/**`, and manually via
`workflow_dispatch`. Jobs should pin the immutable `sha-<short>` tag; `latest` is
a convenience pointer.

If the runner ever exhausts disk on the ~25 GB base image, fall back to
`hpc/containers/build_sif_polaris.pbs`, which builds with `--fakeroot` on a
Polaris compute node.

## The venv fallback

DSI has no container runtime and no module system, so it stays on
`uv sync` — see `hpc/install.md`. That is the only supported exception.
