# NCSA Delta — install & smoke-test recipe

The realization of `hpc/install.md` for **NCSA Delta** (SLURM, Cray PE on RHEL 9). All ai-rossby
smoke tests run here unless explicitly redirected to a different cluster.

---

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | SLURM |
| Default smoke-test partition (GPU) | `gpuA40x4-interactive` (account `bdiu-delta-gpu`) |
| Default data-conversion partition (CPU) | `cpu` / `cpu-interactive` (account `bdiu-delta-cpu`) |
| Walltime cap (interactive GPU / CPU) | **1 hour** / **1 hour** |
| Walltime cap (non-interactive GPU / CPU) | **2 days** / **2 days** |
| GPU node geometry | 4× NVIDIA A40 (48 GB), 64 CPU, ~258 GB RAM |
| CPU node geometry | 128 CPU cores, ~256 GB RAM (no GPU) |
| Single-node constraint | ✅ all smoke tests + data-conversion jobs run on 1 node |
| Repo path | `/work/nvme/bdiu/awikner/physicsnemo` |
| Test-data path (gitignored, large fixtures) | `/work/nvme/bdiu/awikner/physicsnemo_test_data` (symlinked at `test/_data`) |

The non-interactive `gpuA40x4` partition (2-day walltime) exists for longer fidelity tests
(Phase 5, full training-recipe shake-out), but **smoke tests must use the interactive queue**.

The CPU partitions (`cpu` for batch, `cpu-interactive` for ≤ 1-hour jobs) under account
`bdiu-delta-cpu` are the home for data-conversion / preprocessing work (HDF5→Zarr
converters, climatology + bias aggregations, normalization-stat computations). See the
**Data-conversion CPU jobs** section below and the `delta-cpu-job` Claude skill.

## System stack we reuse

| Layer | Module / source |
|---|---|
| CUDA toolkit | `cudatoolkit/25.3_12.8` (auto-loaded with the default environment) |
| Distributed fabric | `libfabric/1.22.0` (auto-loaded) |
| Distributed NCCL transport | `aws-ofi-nccl/1.14.2` (load only for multi-GPU runs) |
| Python interpreter | **uv-managed** (CPython 3.12 via `uv python install 3.12`) |
| PyTorch | **fresh, from `pytorch-cu128` index** — torch 2.10.x against system CUDA 12.8 |

We deliberately do **not** load `pytorch-conda/2.8` — it ships torch 2.8.0+cu128, below
physicsnemo's `torch>=2.10.0` pin in `pyproject.toml`. Per `hpc/install.md` step 4 we use
**Option B**: keep the pin intact, let uv pull torch 2.10 from the `pytorch-cu128` index, and
ride the system CUDA toolkit (12.8) underneath. Cost: one-time multi-GB wheel download.
Benefit: tracks upstream pinned versions; no fork-local pyproject divergence.

## One-time setup

```bash
# 1. uv (one-time, per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# 2. Confirm the system CUDA module is loaded (default env already loads cudatoolkit/25.3_12.8)
module list | grep cudatoolkit
# (do NOT load pytorch-conda/2.8)

# 3. Sync the project: uv installs Python 3.12, creates .venv, and resolves all deps.
#    The `cu12` extra activates the pytorch-cu128 index for torch + torchvision.
#    The `dev` group brings in pytest, ruff, etc.
#    --python 3.12 is required: physicsnemo's pyproject allows 3.11–3.14, but cuml-cu12
#    only ships wheels for 3.11–3.13 and uv defaults to the highest allowed (3.14),
#    which fails to resolve. 3.12 is the safe ML choice.
cd /work/nvme/bdiu/awikner/physicsnemo
unset VIRTUAL_ENV   # ignore any stale VIRTUAL_ENV from the parent shell (IDE etc.)
uv sync --extra cu12 --group dev --python 3.12
source .venv/bin/activate

# 4. Set up the gitignored test-data area
#    (root path is committed to repo as a `test/_data` symlink for IDE convenience —
#     symlink itself is gitignored)
mkdir -p /work/nvme/bdiu/awikner/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=/work/nvme/bdiu/awikner/physicsnemo_test_data  # add to ~/.bashrc
```

For every subsequent shell session you only need:

```bash
cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate
```

Verify the install (CPU-only check on the login node; CUDA visibility requires a GPU node):

```bash
python -c "import torch, physicsnemo; print('torch', torch.__version__, 'cuda', torch.version.cuda, '/', 'physicsnemo', physicsnemo.__version__)"
# expected something like: torch 2.11.0+cu128 cuda 12.8 / physicsnemo 2.2.0a0
```

A subset of `test/models/mlp` can be run on a login node as a quick infra check, but a few
`_optims` tests using `torch.compile` will fail there because the login node's NVHPC SDK
linker conflicts with Inductor's C++ build path. **This is a CPU-on-login-node quirk only**;
those tests pass on GPU compute nodes. Use the `delta-smoke-test` skill (or pattern A in this
doc) to run anything that hits `torch.compile` or CUDA.

## Smoke-test contract

A **smoke test** is a small, GPU-required end-to-end check that proves a newly ported feature
*wires together*. Each is:

- Marked `@pytest.mark.smoke` **and** `@pytest.mark.cuda`.
- Runs on **1 node, 1 or 2 A40 GPUs**, finishes in ≤ 5 minutes wall.
- Lives in the same `test/` file as the feature's unit tests (not a separate tree).
- Uses synthetic tiny tensors **except** for data-loading code, which must read at least
  one real fixture from `$AI_ROSSBY_TEST_DATA` (see below).

| Feature category | What the smoke test does |
|---|---|
| **Model** | Instantiate on CUDA → forward on synthetic tiny tensors → backward → 1 AdamW step → `save_checkpoint`/`from_checkpoint` roundtrip → re-forward matches |
| **Datapipe** | Read a real fixture from `$AI_ROSSBY_TEST_DATA` → iterate N batches → shape/dtype/device + channel-routing assertions |
| **Validation metric** | Synthetic ground-truth + predictions → metric value within tolerance of analytic/reference (lat-weighting, dayofyear ACC, CRPS, power spectra) |
| **Training recipe** | 1–2 train steps on synthetic data, single GPU **+** 2-GPU DDP via `torchrun --nproc-per-node=2`; checkpoint at step 1 reloads |
| **Checkpoint translation** | Fabricate a small source-format ckpt → translate → load into faithful model → forward matches |
| **Interpolant solver** | Synthetic state → 1 step **+** full 5-step rollout, output finite, shapes preserved |

### Test-data fixtures

Tiny reference tensors (≤ ~1 MB) used by `validate_forward_accuracy` live **in-repo** at
`test/models/<name>/data/*.pth` and are committed.

Larger fixtures — real HDF5/NetCDF samples for datapipe smoke tests, full reference rollouts
for fidelity checks — live **out-of-repo** at `$AI_ROSSBY_TEST_DATA` and are gitignored. For
IDE convenience the repo contains a `test/_data` symlink pointing at that path; the symlink
itself is gitignored, so other hosts can recreate it pointing wherever their fixtures live.
Tests that depend on them must `pytest.skip(...)` if the path is unset or the file is missing,
with a message pointing at the canonical fixture-generation script under `hpc/test_data/`.

## Job-script templates

### Pattern A — `srun` (streams output, blocks until done)

The default for smoke tests. Output streams to your terminal; the job ends automatically when
pytest exits.

```bash
srun \
  --partition=gpuA40x4-interactive \
  --account=bdiu-delta-gpu \
  --time=00:30:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
  --gpus-per-node=1 --mem=64g \
  --job-name=pn-smoke \
  bash -lc 'module load pytorch-conda/2.8 && \
           cd /work/nvme/bdiu/awikner/physicsnemo && \
           source .venv/bin/activate && \
           pytest -m "smoke and cuda" -x -q <TARGET>'
```

Replace `<TARGET>` with `test/models/pangu_plasim/` (or any pytest path/`-k` expression). Bump
`--gpus-per-node=2` for DDP smoke tests.

### Pattern B — `sbatch` (queued, output to file)

When you don't want to hold a terminal open or you're queueing several in a row.

Save as `hpc/scripts/smoke.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --job-name=pn-smoke
#SBATCH --output=hpc/scripts/logs/smoke-%j.out

set -euo pipefail
module load pytorch-conda/2.8
cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

pytest -m "smoke and cuda" -x -q "${TARGET:-test/}"
```

Submit with `TARGET=test/models/pangu_plasim/ sbatch hpc/scripts/smoke.sbatch`.

### Pattern C — interactive shell

For debugging a failing smoke test on an actual A40 (file-edit, re-run, repeat):

```bash
srun \
  --partition=gpuA40x4-interactive \
  --account=bdiu-delta-gpu \
  --time=01:00:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
  --gpus-per-node=1 --mem=64g \
  --pty bash
# Once inside:
module load pytorch-conda/2.8
cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate
pytest -m "smoke and cuda" -x test/models/pangu_plasim/test_pangu_plasim.py::test_forward
```

The `delta-shell` and `delta-smoke-test` Claude skills wrap patterns C and A respectively.

## Multi-GPU (DDP) smoke tests

Inside a 2-GPU srun, drive DDP with `torchrun`:

```bash
srun ... --gpus-per-node=2 --ntasks-per-node=1 ... \
  bash -lc 'module load pytorch-conda/2.8 aws-ofi-nccl/1.14.2 && \
            cd /work/nvme/bdiu/awikner/physicsnemo && source .venv/bin/activate && \
            torchrun --standalone --nproc-per-node=2 \
              -m pytest -m "smoke and cuda" -x -q test/models/.../test_ddp.py'
```

Cap at `--gpus-per-node=4`. NCCL on Delta benefits from `aws-ofi-nccl`; only load it for
multi-GPU runs.

## Conventions to keep this maintainable

- Never hard-code `--account` or `--partition` outside `hpc/`. Tests stay portable; the
  cluster info lives here.
- Per-test wall time: aim for ≤ 5 minutes. If it grows past that, it's a fidelity test, not a
  smoke test — move it to `gpuA40x4` (non-interactive) with its own job script.
- DDP smoke tests run with **exactly 2** GPUs (enough to exercise the all-reduce path without
  burning the queue).
- All scratch outputs from smoke tests go to `$SLURM_JOB_ID`-scoped paths under
  `$AI_ROSSBY_TEST_DATA/scratch/`. Tests are responsible for cleanup on success.

## When you outgrow the interactive queue

`gpuA40x4-interactive` is the wrong tool for:

- Phase 5 fidelity gate (load translated checkpoint, run real rollout, compare against
  PanguWeather reference). Run on `gpuA40x4` (2-day walltime).
- Phase 3 training recipe shake-out beyond a 1–2 step smoke. Run on `gpuA40x4`.
- Anything needing > 4 GPUs. Move to a multi-node partition (TBD).

Those non-smoke job scripts get their own files under `hpc/scripts/` and reference this doc
for environment setup.

## Data-conversion CPU jobs

CPU-only work (HDF5→Zarr converters, climatology + bias aggregations, normalization-stat
computations, multiprocessing batches) runs under account `bdiu-delta-cpu` on either:

- `cpu` — **2-day** walltime, the default for full-dataset conversions.
- `cpu-interactive` — **1-hour** walltime, for fixture-sized smoke conversions and
  interactive debugging.

The `delta-cpu-job` Claude skill wraps the common srun pattern; see that skill for
defaults and example commands. Manual srun pattern:

```bash
srun \
  --partition=cpu \
  --account=bdiu-delta-cpu \
  --time=04:00:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
  --mem=128g \
  --job-name=pn-conversion \
  bash -lc 'cd /work/nvme/bdiu/awikner/physicsnemo && \
            source .venv/bin/activate && \
            python tools/data/<dataset>/<script>.py [args]'
```

Conventions:

- Data-conversion CLIs live in `tools/data/<dataset>/`. Each is a runnable Python script
  that reads `SLURM_CPUS_PER_TASK` (falling back to `os.cpu_count()`) to size its
  `multiprocessing.Pool` / `concurrent.futures` worker count.
- Conversions write to gitignored paths — typically under
  `/work/nvme/bdiu/awikner/physicsnemo_test_data/<dataset>/` for fixture-sized outputs,
  or the dataset's source-collocated path for the full archive.
- Long-running conversions (> 1 hr) submit to `cpu` (non-interactive) via the same srun
  pattern; queue and forget. Use `sbatch` only when wrapping a multi-step pipeline that
  the user doesn't want to babysit.
