# Installing the ai-rossby fork on an HPC cluster

A portable recipe for setting up the `awikner/physicsnemo` fork (branch `ai-rossby`) for
development and smoke-testing on any cluster. The goal is **reuse system-installed PyTorch
and CUDA** whenever the cluster provides them, and use [`uv`](https://docs.astral.sh/uv/) as
the package manager for everything else.

Cluster-specific recipes live alongside this file (e.g. `hpc/delta.md`). This document is the
template they follow.

---

## Strategy

| Layer | Source | Why |
|---|---|---|
| Python interpreter | Cluster-provided (module) | Matches the cluster's MPI/NCCL build, NVIDIA drivers, glibc |
| PyTorch + CUDA + cuDNN + NCCL | Cluster-provided (module) | These are the painful ones to build correctly against the system stack |
| `physicsnemo` + everything else | uv + this repo | Fast resolution, lockfile, editable installs for porting work |

A `uv` venv with `--system-site-packages` lets the venv *see* the cluster's PyTorch without
re-downloading it, while `uv pip install -e .` installs the rest of the dependency tree on top.

---

## Step 0 — Identify cluster-provided pieces

Each cluster differs. Find:

1. The module (or path) that puts a Python interpreter with **PyTorch + CUDA** on `PATH`. On
   most NCSA/HPC sites this is a `pytorch`/`pytorch-conda`/`anaconda` module. Verify with:
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```
2. The CUDA toolkit module (often loaded transitively, sometimes not). Needed for builds of
   any package that compiles CUDA kernels (e.g., `apex`, `flash-attn`).
3. Distributed-training pieces (`nccl`, `aws-ofi-nccl`, `libfabric`) if multi-GPU is in scope.
4. Where to put the venv. `$HOME` is usually small and slow; prefer a scratch/project
   filesystem (`/scratch/...`, `/work/...`, `/projects/...`).

Document these in the cluster's own file (e.g. `hpc/<cluster>.md`).

## Step 1 — Install `uv` (one-time, per-user)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Adds ~/.local/bin/uv. Make sure that's on PATH.
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc
uv --version
```

No root required; runs entirely from `$HOME`. If outbound network is restricted on the cluster,
download the `uv` static binary on a node with network, then copy to `~/.local/bin/uv`.

## Step 2 — Load the system stack

Whatever you found in step 0. Example pattern:

```bash
module load <pytorch-module>     # gives `python` with torch+CUDA
module load <cuda-module>        # if not auto-loaded
module load <distributed-modules>  # NCCL/OFI if needed
```

Confirm:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# torch 2.X.Y+cuZZZ cuda Z.Z
```

The version you get back drives Step 4 below.

## Step 3 — Create the venv

From the repo root:

```bash
cd /path/to/physicsnemo                                    # the ai-rossby fork
uv venv --system-site-packages --python "$(which python)" .venv
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"          # should still see system torch
```

`--system-site-packages` lets the venv inherit the cluster's torch + CUDA + NCCL.
`--python "$(which python)"` pins the venv to the same interpreter the module just put on
`PATH` (critical — must match the ABI of the inherited site-packages).

Use a **separate venv per cluster** even if the repo path is shared (different system stacks).

## Step 4 — Install `physicsnemo` and dev dependencies

This is the step that handles the version-pin question. `physicsnemo`'s `pyproject.toml`
pins `torch>=2.10.0`. If the cluster's PyTorch is **≥ 2.10**:

```bash
uv pip install -e .                       # respects the existing pin, no torch reinstall
uv pip install --group dev                # pytest, ruff, etc.
```

If the cluster's PyTorch is **< 2.10**, pick one:

**Option A — relax the pin locally** (preferred for fork-local development; less network IO,
exact match to the cluster's NCCL/CUDA build):

1. Edit `pyproject.toml` on the `ai-rossby` branch to `torch>=2.8,<2.11` (or whatever range
   covers the cluster's version).
2. Document the relaxation in `hpc/<cluster>.md` so future readers know why.
3. `uv pip install -e .` then `uv pip install --group dev`.

**Option B — let uv pull a fresh torch from the upstream index** (preferred when the
cluster's torch is too far behind, or when you need an upstream-pure environment):

1. **Skip the cluster's PyTorch module** from Step 2 — load only the CUDA toolkit module.
   (You don't want two torch installations colliding.)
2. Don't create the venv with `--system-site-packages` and don't pin to the cluster's Python.
   Let uv manage its own interpreter: `uv sync --extra cu12 --python 3.12` (or `--extra cu13`
   for CUDA 13 clusters). uv will install Python and fetch torch + CUDA wheels from the
   index configured in `pyproject.toml`.
3. You still need the cluster's CUDA driver and matching toolkit on `LD_LIBRARY_PATH`. Load
   the right `cuda*` module; do NOT load the `pytorch-conda`/`pytorch` one.
4. **Pin the Python version explicitly.** physicsnemo's `requires-python = ">=3.11,<=3.14"`
   but several CUDA-extension deps (e.g., `cuml-cu12`) only ship wheels for 3.11–3.13. uv
   defaults to the highest allowed Python and will fail to resolve. **Use `--python 3.12`**
   unless you have a specific reason to deviate.

Option A keeps tight integration with the cluster's NCCL/MPI. Option B is portable across
clusters but slower and may miss optimizations.

**Match the CUDA version to the system Nsight profiler.** If you profile with the cluster's
Nsight Systems / Compute (`nsys` / `ncu`, see Step 7), the venv's torch CUDA build must be **≤**
the system Nsight's CUDA version — Nsight is backward-compatible (a newer `ncu` profiles an app
built with an equal-or-older CUDA, never a newer one). So: find the system Nsight/CUDA first, then
pick the CUDA extra whose toolkit matches it. physicsnemo ships `cu12` (CUDA 12.8), `cu129`
(CUDA 12.9), and `cu13` (CUDA 13.0) — choose the one **equal** to the system Nsight's CUDA where
possible (exact match is ideal for `ncu` kernel profiling). When a cluster's module already
provides torch ≥ 2.10 at the right CUDA (Option A), reusing it is the tightest Nsight match of
all: toolkit, driver, and profiler are the same build the site ships.

## Step 5 — Smoke check

A trivial check on a login node (CPU-only):

```bash
python -c "import physicsnemo; print(physicsnemo.__version__)"
python -m pytest test/common -x -q                 # exercises validators, no GPU needed
```

A real check on a GPU node (use the cluster's interactive queue — see `hpc/<cluster>.md`):

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
pytest -m "smoke and cuda" -x -q test/             # all smoke tests
```

## Step 6 — Document the cluster

Create `hpc/<cluster>.md` capturing:

- Cluster name and scheduler (SLURM / PBS).
- Modules used in Step 0 (with exact versions).
- Whether Option A or Option B was used in Step 4 (with rationale).
- Default interactive queue, walltime cap, account/project code.
- sbatch / qsub job-script templates for smoke tests and interactive sessions.
- Test-data path conventions (scratch vs. in-repo).
- Known oddities (HDF5 locking, NCCL fabric tuning, container vs. native, etc.).

The Delta recipe at `hpc/delta.md` is a worked example.

## Step 7 — Profiling with Nsight (Systems & Compute)

ai-rossby performance work uses NVIDIA **Nsight Systems** (`nsys`, timeline / system profiling)
and **Nsight Compute** (`ncu`, kernel profiling). These ship with the cluster's CUDA toolkit /
NVHPC SDK — **not** with the venv — so you run them *around* the venv's Python.

1. **Find the system Nsight and its CUDA version.** Usually arrives with a `cuda` / `nvhpc`
   module (sometimes already on the default PATH):
   ```bash
   module load <cuda-or-nvhpc-module>     # cluster-specific; see hpc/<cluster>.md
   nsys --version        # e.g. "2025.1.3" (CUDA 12.9-era)
   ncu  --version        # e.g. "2025.2.0.0"
   nvcc --version        # toolkit version → the CUDA the profilers target
   ```
2. **Match the venv's torch CUDA to it** (Step 4). `ncu` is happiest when the app's CUDA equals
   the profiler's, and refuses an app built with a *newer* CUDA than it supports.
3. **Profile the venv under the loaded module**, from inside a GPU job (the profilers need a GPU):
   ```bash
   module load <cuda-or-nvhpc-module>
   source .venv/bin/activate
   nsys profile -o /scratch/.../run_%p       python -m my_script ...
   ncu   --set full -o /scratch/.../kernel_%p python -m my_script ...
   ```
   Profiler output is large — write it to scratch. Record the per-cluster profiling module +
   `nsys`/`ncu` versions in `hpc/<cluster>.md` under a **Profiling** section.

---

## Updating the install when the cluster's PyTorch changes

When the cluster upgrades its PyTorch module:

1. Re-run Step 3 (recreate venv against the new interpreter).
2. Re-run Step 4. If you used Option A, revisit whether the pin still needs relaxing.
3. Re-run Step 5. If anything regresses, file an issue and pin the old module path in the
   cluster doc for the rollback window.
