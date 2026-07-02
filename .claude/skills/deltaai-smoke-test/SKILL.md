---
name: deltaai-smoke-test
description: Submit a pytest target as a GPU smoke test on NCSA DeltaAI's ghx4-interactive queue (GH200, aarch64) under account bdiu-dtai-gh. Use whenever the user asks to run a smoke test, run GPU tests, or verify a ported physicsnemo feature on DeltaAI / GH200. Streams output back; blocks until pytest exits.
---

# deltaai-smoke-test

Submits a pytest target to NCSA DeltaAI's `ghx4-interactive` SLURM partition (GH200 Grace-Hopper,
**aarch64**) under account `bdiu-dtai-gh`, in the ai-rossby workflow defined by `hpc/deltaai.md`.
The job blocks until pytest exits and streams output. Pairs with `deltaai-shell`.

Verified working 2026-07-01 (torch 2.10.0+cu129, GH200). **Never exceed the 2-hour interactive
cap** — longer runs use `ghx4` (2-day) with their own job script.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `ghx4-interactive` | Only deviation: user names another partition |
| `--account` | `bdiu-dtai-gh` | Only deviation: user names another account |
| `--time` | `00:30:00` | Bump to `01:00:00` for DDP; never exceed the 2 h cap |
| `--nodes` | `1` | Single-node by contract |
| `--ntasks-per-node` | `1` | Pytest runs once; DDP driven by `torchrun` inside |
| `--cpus-per-task` | `8` | Grace CPU dataloader workers |
| `--gpus-per-node` | `1` | Bump to `2` for DDP; GH200 nodes have 4 |
| `--mem` | `64g` | |
| `--job-name` | `pn-smoke` | |

## What this skill does, step by step

1. **Confirm the target.** A pytest path, `-k` expression, or node-id. If missing, ask.
2. **Choose flags.** For "DDP"/"multi-GPU", set `--gpus-per-node=2` and `--time=01:00:00`. Never
   above 4 GPUs, 1 node, or 2 h.
3. **Build the command.** DeltaAI uses **Option A** — the venv inherits torch 2.10+cu129 from the
   `python/miniforge3_pytorch/2.10.0` module, so that module MUST be loaded:
   ```
   module load python/miniforge3_pytorch/2.10.0 && \
   cd /work/nvme/bdiu/awikner/physicsnemo && source .venv-deltaai/bin/activate && \
   export AI_ROSSBY_TEST_DATA=/work/nvme/bdiu/awikner/physicsnemo_test_data && \
   pytest -m "smoke and cuda" -x -q <TARGET>
   ```
   Note the venv is **`.venv-deltaai`** (not `.venv`, which is Delta's x86_64). For DDP, prepend
   `torchrun --standalone --nproc-per-node=2 -m ` before `pytest` and load
   `nccl-ofi-plugin/1.18.0-cuda129`.
4. **Submit via `srun`.** Stream output; job ends when pytest does. Do NOT use `--pty` (that's
   `deltaai-shell`) and do NOT `sbatch` (the user wants to see the result).
5. **Report.** On pass, state the test name + wall time. On failure, show the pytest tail and stop.

## Example (Bash tool)

```bash
srun --partition=ghx4-interactive --account=bdiu-dtai-gh --time=00:30:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gpus-per-node=1 --mem=64g \
  --job-name=pn-smoke \
  bash -lc 'module load python/miniforge3_pytorch/2.10.0 && \
            cd /work/nvme/bdiu/awikner/physicsnemo && source .venv-deltaai/bin/activate && \
            export AI_ROSSBY_TEST_DATA=/work/nvme/bdiu/awikner/physicsnemo_test_data && \
            pytest -m "smoke and cuda" -x -q test/models/pangu_plasim/'
```

## Refuse / push back when

- User requests a non-`ghx4-interactive` partition *for a smoke test* — ask whether they want a
  fidelity/recipe job (which belongs on `ghx4` with its own script).
- User asks for `--time` > 2 h, > 4 GPUs, or multi-node — stop and surface the contract violation.
- User names an account other than `bdiu-dtai-gh` without explanation — confirm first.
- The venv named is `.venv` rather than `.venv-deltaai` — that's Delta's x86_64 build and will not
  work on GH200; correct it.
- The repo is not on `ai-rossby` — note it (smoke tests likely don't exist on `main`).

## Out-of-scope

- Long-running fidelity tests — `ghx4` (2-day) with its own job script.
- Multi-node DDP — out of scope for smoke.
- Interactive debugging — use `deltaai-shell`.
- Any other cluster — its own skill + doc.
