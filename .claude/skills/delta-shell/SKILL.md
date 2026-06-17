---
name: delta-shell
description: Open an interactive shell on an NCSA Delta A40 GPU node via `srun --pty` on the gpuA40x4-interactive partition under bdiu-delta-gpu. Use when the user wants to debug a failing smoke test on a real GPU, run pytest by hand on a node, or otherwise needs a GPU-attached bash session on Delta.
---

# delta-shell

Opens an interactive shell on a Delta A40 GPU node — `srun --pty bash` against
`gpuA40x4-interactive` under `bdiu-delta-gpu`. The user lands in the activated ai-rossby venv with
`pytorch-conda/2.8` loaded, ready to run pytest by hand.

Use this for *debugging*, not *batch verification* — that's what `delta-smoke-test` is for.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `gpuA40x4-interactive` | Only deviation: user explicitly names another partition |
| `--account` | `bdiu-delta-gpu` | Only deviation: user explicitly names another account |
| `--time` | `01:00:00` | The interactive cap. Lower for short debugging sessions |
| `--nodes` | `1` | Single-node by contract |
| `--ntasks-per-node` | `1` | One shell |
| `--cpus-per-task` | `8` | Comfortable for an interactive session |
| `--gpus-per-node` | `1` | Bump to `2` if the user wants to drive `torchrun` by hand |
| `--mem` | `64g` | Same as smoke tests |
| `--job-name` | `pn-shell` | |

## What this skill does

1. **Confirm gpu count and walltime.** Default 1 GPU, 1 hour. If the user said "DDP debugging" or
   "torchrun," set `--gpus-per-node=2`. Never above 4 or above 1 hour.
2. **Run `srun --pty`** with the env loaded inline so the user lands in the activated venv:
   ```bash
   srun \
     --partition=gpuA40x4-interactive \
     --account=bdiu-delta-gpu \
     --time=01:00:00 \
     --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
     --gpus-per-node=1 --mem=64g \
     --job-name=pn-shell \
     --pty bash -lc 'module load pytorch-conda/2.8; \
                    cd /work/nvme/bdiu/awikner/physicsnemo; \
                    source .venv/bin/activate; \
                    exec bash'
   ```
3. **Don't try to do anything inside the shell** — this skill just opens it. The user takes it from
   there. If they want you to run something specific on the node, they should use `delta-smoke-test`
   for non-interactive cases.

## Refuse / push back when

- User asks for `--time` > 1 hour, > 4 GPUs, or multi-node.
- The shell is being requested to do something `delta-smoke-test` already does — recommend that skill
  instead so output is captured.

## Out-of-scope

- Long-lived debugging sessions (queue is interactive-only, 1-hour cap).
- Anything that should be reproducible — wrap it as a pytest smoke test instead and use
  `delta-smoke-test`.
