---
name: deltaai-shell
description: Open an interactive shell on an NCSA DeltaAI GH200 GPU node via `srun --pty` on the ghx4-interactive partition (aarch64) under account bdiu-dtai-gh. Use when the user wants to debug a failing smoke test on a real GH200, run pytest by hand on a node, or otherwise needs a GPU-attached bash session on DeltaAI.
---

# deltaai-shell

Opens an interactive shell on a DeltaAI **GH200** GPU node — `srun --pty bash` against
`ghx4-interactive` (aarch64) under `bdiu-dtai-gh`. The user lands with the
`python/miniforge3_pytorch/2.10.0` module loaded and the `.venv-deltaai` venv activated, ready to
run pytest by hand. Use for *debugging*, not batch verification — that's `deltaai-smoke-test`.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `ghx4-interactive` | Only deviation: user names another partition |
| `--account` | `bdiu-dtai-gh` | Only deviation: user names another account |
| `--time` | `01:00:00` | Lower for short sessions; never exceed the 2 h interactive cap |
| `--nodes` | `1` | Single-node |
| `--ntasks-per-node` | `1` | One shell |
| `--cpus-per-task` | `8` | |
| `--gpus-per-node` | `1` | Bump to `2` for DDP-by-hand; GH200 nodes have 4 |
| `--mem` | `64g` | |
| `--job-name` | `pn-shell` | |

## What this skill does

1. **Confirm gpu count and walltime.** Default 1 GPU, 1 h. For DDP set `--gpus-per-node=2`. Never
   above 4 or above 2 h.
2. **Run `srun --pty`** with the env staged so the user lands ready (note `.venv-deltaai`, not
   `.venv`):
   ```bash
   srun --partition=ghx4-interactive --account=bdiu-dtai-gh --time=01:00:00 \
     --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gpus-per-node=1 --mem=64g \
     --job-name=pn-shell \
     --pty bash -lc 'module load python/miniforge3_pytorch/2.10.0; \
                    cd /work/nvme/bdiu/awikner/physicsnemo; \
                    source .venv-deltaai/bin/activate; exec bash'
   ```
3. **Don't drive work inside the shell** — this skill just opens it. For non-interactive runs the
   user should use `deltaai-smoke-test` so output is captured.

## Refuse / push back when

- User asks for `--time` > 2 h, > 4 GPUs, or multi-node.
- The shell is being used to do what `deltaai-smoke-test` already does — recommend that skill.

## Out-of-scope

- Long-lived sessions (2 h interactive cap).
- Reproducible runs — wrap as a pytest smoke test and use `deltaai-smoke-test`.
- Any other cluster — its own skill + doc.
