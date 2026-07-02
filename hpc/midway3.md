# UChicago RCC Midway3 — install & smoke-test recipe

The realization of `hpc/install.md` for **UChicago RCC Midway3** (SLURM, Lmod, x86_64). Sister
document to `hpc/delta.md`. **Set up 2026-07-01.**

---

## Cluster facts

| Item | Value |
|---|---|
| Login node | `midway3-login3.rcc.local` (SSH alias `midway3` → `midway3.rcc.uchicago.edu`) |
| Architecture | x86_64 |
| Scheduler | SLURM (Lmod modules) |
| GPU account | `pi-pedramh` |
| GPU partition (smoke) | **`pedramh-gpu`** (H100, `gpu:4`/node) — the group's dedicated partition (`AllowAccounts=pi-pedramh`) |
| Alternate GPU option | **`schmidt-gpu`** (generally **A100**; mixed A100/H100) via `--account=pi-dfreedman --qos=schmidt` — use when `pedramh-gpu` is busy |
| Other GPU partitions | `gpu` (V100, open to all) — **incompatible** with cu129 torch (⚠️); `beagle3` (A100) is a condo `pi-pedramh` cannot use |
| CPU partition | `caslake` (general CPU) |
| Walltime caps | per-QoS (TBD — RCC GPU jobs are typically ≤ 36 h) |
| Repo path | `/project/pedramh/awikner/physicsnemo` |
| venv | `.venv` (Option B, x86_64) |
| Test-data path | `/project/pedramh/awikner/physicsnemo_test_data` |

> **Project path is `/project/pedramh`** (the PI group `pedramh`), not `pedrahm`.

> **⚠️ Run GPU work on `pedramh-gpu` (H100), not `gpu` (V100).** torch ≥ 2.10 on cu129 is built
> for compute capability ≥ 7.5; the open `gpu` partition's Tesla **V100 is CC 7.0 (Volta) and is
> unsupported** — smoke tests fail with *"GPU0 … which is of compute capability 7.0"*. The
> group's dedicated **`pedramh-gpu` H100** nodes (CC 9.0) work and exact-match the 12.9 Nsight.
> (`beagle3`'s A100s would also work, but its `AllowAccounts` excludes `pi-pedramh`.)

## Filesystem conventions (RCC)

| Filesystem | Policy | Use for |
|---|---|---|
| `/home/awikner/` | ~30 GB, backed up | dot-files only |
| `/project/pedramh/awikner/` | project quota, persistent (shared with the pedramh group) | repo clone, venv, test fixtures |
| `/scratch/midway3/awikner/` | large, purge policy TBD | training data, Zarr archives, job outputs |

**Existing data:** collaborator amip checkpoints at `/project/pedramh/ayz/AMIP_logs/` (used in
Phase 8e live tests; unlocks x_DDC translator validation here). Shared group env at
`/project/pedramh/shared/conda/envs/py311_pip_sfno_cu129` (torch 2.9.1+cu129 — see below).

## System stack — Option B with cu129

The shared group env (`py311_pip_sfno_cu129`) ships **torch 2.9.1+cu129**, which is **below**
physicsnemo's `torch>=2.10.0` pin, so Option A (reuse it) is not viable. Instead we use
**Option B with the `cu129` extra** — uv pulls torch 2.10+cu129 (x86_64), matching the system
Nsight (12.9). RAPIDS deps (cuml-cu12 etc.) have x86_64 wheels, so cu129 resolves here.

| Layer | Source |
|---|---|
| Python | uv-managed **3.12** |
| CUDA toolkit | `cuda/12.9` module (`CUDA_HOME=/software/cuda-12.9-el8-x86_64`) |
| torch / torchvision | **cu129** (latest ≥ 2.10 on the `pytorch-cu129` index — resolved to **2.12.1+cu129**; CUDA 12.9 either way) |
| physicsnemo + deps | uv |

> **`pyproject.toml` on this clone carries the `cu129` extra + `[tool.hatch.metadata]
> allow-direct-references = true` fix** (copied from the Mac ahead of those changes being
> committed to `ai-rossby`). Midway3 is the **first cluster to resolve `--extra cu129`** — commit
> the resulting `uv.lock` once validated.

## One-time setup

```bash
# 1. uv (x86_64, per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # add to ~/.bashrc
# ⚠️ Point uv's cache at scratch — /home quota is small (~30 GB) and the cu129
#    extra (torch + RAPIDS + DALI, ~18 GB cached) overflows it, failing mid-sync.
mkdir -p /scratch/midway3/awikner/.uv-cache
export UV_CACHE_DIR=/scratch/midway3/awikner/.uv-cache   # add to ~/.bashrc

# 2. Clone into /project/pedramh (persistent), NOT /home or /scratch.
mkdir -p /project/pedramh/awikner && cd /project/pedramh/awikner
git clone git@github.com:awikner/physicsnemo.git      # ForwardAgent serves the Mac's GitHub key
cd physicsnemo && git checkout ai-rossby

# 3. Option B with cu129 (matches the system Nsight 12.9):
unset VIRTUAL_ENV
uv sync --extra cu129 --group dev --python 3.12

# 4. Test-data area on project storage
mkdir -p /project/pedramh/awikner/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data   # add to ~/.bashrc
```

Verify (login node, CPU-only):

```bash
source .venv/bin/activate
python -c "import torch, physicsnemo; print('torch', torch.__version__, torch.version.cuda, '/ physicsnemo', physicsnemo.__version__)"
# expect: torch 2.10.x+cu129 12.9 / physicsnemo 2.2.0a0
```

## Profiling with Nsight

`nsys`/`ncu` come from the `cuda/12.9` module (`/software/cuda-12.9-el8-x86_64/bin`). Verified:

| Tool | Version | CUDA |
|---|---|---|
| Nsight Systems (`nsys`) | 2025.1.3 | 12.9 |
| Nsight Compute (`ncu`) | 2025.2.0 | 12.9 |

ai-rossby's torch is **cu129 (12.9)** — an *exact* match, so `ncu` attaches cleanly (see
`hpc/install.md` § Step 7). Profile from inside a GPU job, output to scratch:

```bash
module load cuda/12.9
cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate
nsys profile   -o /scratch/midway3/awikner/nsys_%p python -m <target> ...
ncu   --set full -o /scratch/midway3/awikner/ncu_%p python -m <target> ...
```

## Smoke-test contract

Identical to `hpc/delta.md` — `@pytest.mark.smoke` **and** `@pytest.mark.cuda`, single-node,
≤ 5 min wall. Target: `pytest -m "smoke and cuda" -x -q test/`.

## Job-script templates

### Streaming smoke via `srun` (blocks until pytest exits)

```bash
srun --partition=pedramh-gpu --account=pi-pedramh --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64g \
  --job-name=pn-smoke \
  bash -lc 'cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate && \
            export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data && \
            pytest -m "smoke and cuda" -x -q <TARGET>'
```

Always use an A100/H100 partition — the open `gpu` (V100, CC 7.0) is unsupported by cu129 torch
(needs CC ≥ 7.5). Primary is **`pedramh-gpu`** (H100, `--account=pi-pedramh`); when it is busy,
fall back to **`schmidt-gpu`** with `--account=pi-dfreedman --qos=schmidt` (generally A100 — also
cu129-capable):

```bash
srun --partition=schmidt-gpu --account=pi-dfreedman --qos=schmidt --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64g --job-name=pn-smoke \
  bash -lc 'cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate && \
            export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data && \
            pytest -m "smoke and cuda" -x -q <TARGET>'
```

Bump `--gres=gpu:2` for DDP. The `midway3-smoke-test` / `midway3-shell` / `midway3-cpu-job` skills
wrap these.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| 2026-07-02 | 2.12.1+cu129 | A100 (`schmidt-gpu`) | **PASS** | `pangu_plasim` smoke: 2 passed, 34 deselected, 112 s |

## TBD (low priority)

- Exact GPU-partition walltime caps / required QoS for `pi-pedramh`.
- `/scratch/midway3` purge policy.
- Confirm `beagle3` (A100) access for `pi-pedramh`.
