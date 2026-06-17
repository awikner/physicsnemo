---
name: delta-smoke-test
description: Submit a pytest target as a GPU smoke test on NCSA Delta's gpuA40x4-interactive queue under account bdiu-delta-gpu. Use whenever the user asks to run a smoke test, run GPU tests, run pytest on Delta/A40, or verify a newly ported physicsnemo feature on a real GPU. Streams output back; non-interactive.
---

# delta-smoke-test

Submits a pytest target to NCSA Delta's `gpuA40x4-interactive` SLURM partition under account
`bdiu-delta-gpu`, in the ai-rossby workflow defined by `hpc/delta.md`. The job blocks until pytest
exits and streams stdout/stderr back. **Never use this for runs longer than the 1-hour interactive cap** —
fidelity tests / full training-recipe shake-outs need `gpuA40x4` (non-interactive) with their own job script.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `gpuA40x4-interactive` | Only deviation: user explicitly names another partition |
| `--account` | `bdiu-delta-gpu` | Only deviation: user explicitly names another account |
| `--time` | `00:30:00` | Bump to `01:00:00` for DDP smoke; never exceed the 1-hour cap |
| `--nodes` | `1` | Smoke tests are single-node by contract |
| `--ntasks-per-node` | `1` | Pytest runs once; DDP is driven by `torchrun` inside |
| `--cpus-per-task` | `8` | Plenty for dataloader workers |
| `--gpus-per-node` | `1` | Bump to `2` for DDP smoke tests; never exceed `4` |
| `--mem` | `64g` | Comfortable for tiny models + a real fixture |
| `--job-name` | `pn-smoke` | Override if running several in parallel |

## What this skill does, step by step

1. **Confirm the target.** The user gives a pytest target (a path, a `-k` expression, or a node-id like
   `test/models/pangu_plasim/test_pangu_plasim.py::test_forward`). If they don't, ask before submitting.
2. **Choose flags.** Start from the defaults above. If the user said "DDP" or "multi-GPU," set
   `--gpus-per-node=2` and `--time=01:00:00`. Never go above 4 GPUs or 1 hour — explain and stop instead.
3. **Build the command.** Use the wrapper `bash -lc` form so module commands resolve. The body is:
   ```
   module load pytorch-conda/2.8 && \
   cd /work/nvme/bdiu/awikner/physicsnemo && \
   source .venv/bin/activate && \
   pytest -m "smoke and cuda" -x -q <TARGET>
   ```
   For DDP, prepend `torchrun --standalone --nproc-per-node=2 -m ` before `pytest` and append `&& \`
   handling per the user's request. The `aws-ofi-nccl/1.14.2` module gets loaded too for DDP only.
4. **Submit via `srun`.** Stream the output; the job ends when pytest does. Do NOT use `--pty` — that's
   for `delta-shell`. Do NOT use `sbatch` — the user invoked this skill to *see* the result, not queue and
   forget.
5. **Report.** If pytest passed, say so briefly with the test name and wall time. If it failed, show the
   pytest tail (the FAILED line + traceback) and stop — do NOT silently rerun.

## Example commands (Bash tool)

**Single-GPU smoke** (the common case):

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
            pytest -m "smoke and cuda" -x -q test/models/pangu_plasim/'
```

**2-GPU DDP smoke**:

```bash
srun \
  --partition=gpuA40x4-interactive \
  --account=bdiu-delta-gpu \
  --time=01:00:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
  --gpus-per-node=2 --mem=64g \
  --job-name=pn-smoke-ddp \
  bash -lc 'module load pytorch-conda/2.8 aws-ofi-nccl/1.14.2 && \
            cd /work/nvme/bdiu/awikner/physicsnemo && \
            source .venv/bin/activate && \
            torchrun --standalone --nproc-per-node=2 \
              -m pytest -m "smoke and cuda" -x -q <DDP_TARGET>'
```

## Refuse / push back when

- User asks to run on a partition other than `gpuA40x4-interactive` *for a smoke test* — explain that
  smoke tests are scoped to that partition by `hpc/delta.md` and ask whether they really want a
  fidelity/recipe job (which belongs in `hpc/scripts/` instead).
- User asks for `--time` > 1 hour, > 4 GPUs, or multi-node — stop and surface the contract violation.
- User asks for an account other than `bdiu-delta-gpu` without explanation — confirm before proceeding.
- The repo is not on the `ai-rossby` branch — note it (the smoke tests probably don't exist on `main`).

## Out-of-scope

- Long-running fidelity tests (Phase 5) — point at `gpuA40x4` non-interactive instead.
- Multi-node DDP — out of scope for smoke.
- Interactive debugging — point at the `delta-shell` skill.
- Running smoke tests on any other cluster — that cluster needs its own skill + doc.
