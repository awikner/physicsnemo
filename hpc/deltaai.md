# NCSA DeltaAI — install & smoke-test recipe

The realization of `hpc/install.md` for **NCSA DeltaAI** (SLURM, **aarch64 / GH200 Grace-Hopper**).
Sister document to `hpc/delta.md`. **Verified end-to-end on 2026-07-01** (pangu_plasim smoke:
2 passed on a GH200 node).

> **DeltaAI is aarch64 (ARM), not x86_64.** Each node is an NVIDIA GH200 Grace-Hopper superchip
> (Grace ARM CPU + H100-class GPU, 120 GB). Wheels, the venv, and `uv` must all be aarch64 — they
> are **not** interchangeable with Delta's x86_64 build, even though `/work` is shared.

---

## Cluster facts

| Item | Value |
|---|---|
| Login node | `gh-login01.delta.ncsa.illinois.edu` (SSH alias `deltaai` → `dtai-login.delta.ncsa.illinois.edu`) |
| Architecture | **aarch64** (GH200 Grace-Hopper) |
| Scheduler | SLURM |
| GPU hardware | NVIDIA **GH200 120 GB**, 4 per node, nodes `gh[001-152]` |
| GPU account | `bdiu-dtai-gh` |
| Default smoke-test partition (GPU) | `ghx4-interactive` (**2 h** cap) |
| Non-interactive partition | `ghx4` (**2-day** cap, default `*`); also `full` (1-day), `test` (2 h, `gh[001-002]`) |
| Single-node constraint | ✅ all smoke tests run on 1 node |
| Repo path | `/work/nvme/bdiu/awikner/physicsnemo` (**shared Lustre with Delta**) |
| venv | `.venv-deltaai` (aarch64; separate from Delta's `.venv`) |
| Test-data path | `/work/nvme/bdiu/awikner/physicsnemo_test_data` (shared with Delta) |

## Filesystem — shared `/work`, separate `/home`

`/work/nvme` is the **same Lustre filesystem as Delta** (`…/dltawork`, 8.5 PB), so the repo clone,
test fixtures, and Zarr archives are visible from both clusters — **one clone serves both**. But
`/home` (`/u/awikner`) is **separate** per cluster, so `uv` and any dot-files must be installed
independently on DeltaAI (Delta's x86_64 `uv` is neither present nor usable here).

**Separate venv is mandatory:** `.venv-deltaai` holds aarch64 wheels and inherits an aarch64
torch; Delta's `.venv` holds x86_64. They coexist in the shared repo dir. Scratch for training
data is `/scratch/bdiu/awikner/physicsnemo-zarr/` (TBD purge policy).

## System stack — Option A (reuse the site's torch)

DeltaAI ships a conda PyTorch module that already matches the system Nsight (CUDA 12.9), so we
**reuse it** (Option A) rather than pulling cu-wheels — this also sidesteps RAPIDS aarch64 wheel
gaps that a full `cu129` Option B would hit.

| Layer | Source |
|---|---|
| Module | `python/miniforge3_pytorch/2.10.0` → conda env `/sw/user/python/miniforge3-pytorch-2.10.0` |
| Python | 3.12.9 (from the module) |
| torch / torchvision | **2.10.0+cu129 / 0.25.0+cu129** (inherited from the module; triton 3.6.0) |
| physicsnemo + deps | uv, installed on top into `.venv-deltaai` |

## One-time setup (the exact verified sequence)

```bash
# 1. uv — aarch64, per-user (DeltaAI home is separate from Delta):
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs aarch64 uv to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"                      # add to ~/.bashrc

# 2. The repo is ALREADY on the shared /work (cloned from Delta) — do NOT re-clone.
cd /work/nvme/bdiu/awikner/physicsnemo                    # on branch ai-rossby

# 3. Option A venv inheriting the module's torch 2.10+cu129:
module load python/miniforge3_pytorch/2.10.0
unset VIRTUAL_ENV
uv venv --system-site-packages \
   --python /sw/user/python/miniforge3-pytorch-2.10.0/bin/python .venv-deltaai
source .venv-deltaai/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.10.0+cu129 12.9

# 4. Install physicsnemo + dev group, THEN reveal the inherited torch (see gotcha):
uv pip install -e .                 # NOTE: this pulls torch 2.12.1 from PyPI into the venv…
uv pip install --group dev
uv pip uninstall torch torchvision triton   # …remove the shadow so the conda 2.10+cu129 shows through
python -c "import torch, physicsnemo; print(torch.__version__, torch.version.cuda, physicsnemo.__version__)"
#   → 2.10.0+cu129 12.9 2.2.0a0

# 5. Test-data area is shared with Delta (already present):
export AI_ROSSBY_TEST_DATA=/work/nvme/bdiu/awikner/physicsnemo_test_data   # add to ~/.bashrc
```

### ⚠️ Gotchas (both hit during first install)

1. **`uv pip install -e .` shadows the inherited torch.** In a `--system-site-packages` venv, uv
   does *not* treat the conda env's torch as satisfying `torch>=2.10`, so it installs the newest
   from PyPI (torch 2.12.1, **wrong CUDA**). Fix: after installing, `uv pip uninstall torch
   torchvision triton` — the venv then falls back to the conda 2.10.0+cu129 build (verified).
2. **`muon-optimizers` extra needs `allow-direct-references`.** The editable build fails at
   metadata construction until `pyproject.toml` has `[tool.hatch.metadata] allow-direct-references
   = true` (the extra installs from a git direct reference). This is a committed repo fix.

## Profiling with Nsight

`nsys`/`ncu` come from the NVHPC SDK 25.5, already on the default PATH (aarch64). Verified:

| Tool | Version | CUDA |
|---|---|---|
| Nsight Systems (`nsys`) | 2025.3.1 | 12.9 |
| Nsight Compute (`ncu`) | 2025.2.0 | 12.9 |

ai-rossby's torch is **cu129 (12.9)** — an *exact* match, so `ncu` attaches cleanly. Profile from
inside a GPU job (see `hpc/install.md` § Step 7), writing output to scratch:

```bash
cd /work/nvme/bdiu/awikner/physicsnemo && source .venv-deltaai/bin/activate
nsys profile   -o /scratch/bdiu/awikner/nsys_%p python -m <target> ...
ncu   --set full -o /scratch/bdiu/awikner/ncu_%p python -m <target> ...
```

## Smoke-test contract

Identical to `hpc/delta.md` — `@pytest.mark.smoke` **and** `@pytest.mark.cuda`, single-node,
≤ 5 min wall. Target: `pytest -m "smoke and cuda" -x -q test/`.

## Job-script templates

### Streaming smoke via `srun` (blocks until pytest exits)

```bash
srun --partition=ghx4-interactive --account=bdiu-dtai-gh --time=00:30:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gpus-per-node=1 --mem=64g \
  --job-name=pn-smoke \
  bash -lc 'module load python/miniforge3_pytorch/2.10.0 && \
            cd /work/nvme/bdiu/awikner/physicsnemo && source .venv-deltaai/bin/activate && \
            export AI_ROSSBY_TEST_DATA=/work/nvme/bdiu/awikner/physicsnemo_test_data && \
            pytest -m "smoke and cuda" -x -q <TARGET>'
```

Bump `--gpus-per-node=2` for DDP (module `nccl-ofi-plugin/1.18.0-cuda129` for NCCL fabric). The
`deltaai-smoke-test` / `deltaai-shell` skills wrap the streaming and interactive patterns.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| 2026-07-01 | 2.10.0+cu129 | GH200 120 GB | **PASS** | `pangu_plasim` smoke: 2 passed, 34 deselected, 16.7 s |

## TBD (low priority)

- Scratch purge policy for `/scratch/bdiu/awikner/`.
- Multi-GPU DDP smoke on 2×GH200 with `nccl-ofi-plugin/1.18.0-cuda129`.
