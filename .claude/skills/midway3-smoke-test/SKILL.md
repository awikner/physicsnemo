---
name: midway3-smoke-test
description: Submit a pytest target as a GPU smoke test on UChicago RCC Midway3's pedramh-gpu (H100) partition under account pi-pedramh. Use whenever the user asks to run a smoke test, run GPU tests, or verify a ported physicsnemo feature on Midway3. Streams output back; blocks until pytest exits.
---

# midway3-smoke-test

Submits a pytest target to UChicago RCC Midway3's `pedramh-gpu` (**H100**) SLURM partition — the
pedramh group's dedicated GPU nodes — under account `pi-pedramh`, in the ai-rossby workflow defined
by `hpc/midway3.md`. The job blocks until pytest exits and streams output. Pairs with
`midway3-shell` and `midway3-cpu-job`.

> **Do not use the open `gpu` partition:** its Tesla V100 (CC 7.0) is unsupported by the venv's
> cu129 torch (needs CC ≥ 7.5). Only `pedramh-gpu` (H100) works.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `pedramh-gpu` (H100) | Never the open `gpu` (V100 — unsupported). Alt: `schmidt-gpu` (A100) |
| `--account` | `pi-pedramh` | Alt for `schmidt-gpu`: `pi-dfreedman` with `--qos=schmidt` |
| `--time` | `00:30:00` | Bump to `01:00:00` for DDP; keep within the QoS walltime cap |
| `--nodes` | `1` | Single-node by contract |
| `--ntasks` | `1` | Pytest runs once; DDP driven by `torchrun` inside |
| `--cpus-per-task` | `8` | Dataloader workers |
| `--gres` | `gpu:1` | Bump to `gpu:2` for DDP; nodes have 4 |
| `--mem` | `64g` | |
| `--job-name` | `pn-smoke` | |

## What this skill does, step by step

1. **Confirm the target.** A pytest path, `-k` expression, or node-id. If missing, ask.
2. **Choose GPUs / partition.** Default `pedramh-gpu` (H100), 1 GPU. **Never use the open `gpu`
   partition** — its V100 (CC 7.0) is unsupported by cu129 torch. When `pedramh-gpu` is busy, fall
   back to **`schmidt-gpu`** (generally A100, also cu129-capable) with `--account=pi-dfreedman
   --qos=schmidt`. For "DDP"/"multi-GPU", set `--gres=gpu:2` and `--time=01:00:00`.
3. **Build the command** (Option B venv — no module load needed for the venv's own cu129 torch):
   ```
   cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate && \
   export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data && \
   pytest -m "smoke and cuda" -x -q <TARGET>
   ```
   For DDP, prepend `torchrun --standalone --nproc-per-node=2 -m ` before `pytest`.
4. **Submit via `srun`.** Stream output; job ends when pytest does. Do NOT use `--pty` (that's
   `midway3-shell`) and do NOT `sbatch` (the user wants to see the result).
5. **Report.** On pass, state the test name + wall time. On failure, show the pytest tail and stop.

## Example (Bash tool)

```bash
srun --partition=pedramh-gpu --account=pi-pedramh --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64g \
  --job-name=pn-smoke \
  bash -lc 'cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate && \
            export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data && \
            pytest -m "smoke and cuda" -x -q test/models/pangu_plasim/'
```

**Alternate (A100, when `pedramh-gpu` is busy):** swap the first line for
`srun --partition=schmidt-gpu --account=pi-dfreedman --qos=schmidt --time=00:30:00 \`.

## Refuse / push back when

- User requests a non-GPU partition for a smoke test — ask whether they want CPU
  (`midway3-cpu-job`) or a fidelity job (its own script).
- User asks for `--time` beyond the QoS walltime cap, > the node's GPU count, or multi-node — stop
  and surface it.
- User names an account other than `pi-pedramh` without explanation — confirm first.
- The repo is not on `ai-rossby` — note it (smoke tests likely don't exist on `main`).

## Out-of-scope

- Long-running fidelity tests — separate job script.
- Multi-node DDP — out of scope for smoke.
- Interactive debugging — use `midway3-shell`.
- CPU preprocessing — use `midway3-cpu-job`.
- Any other cluster — its own skill + doc.
