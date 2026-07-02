---
name: dsi-shell
description: Open an interactive shell on a UChicago DSI GPU node via `srun --pty` under account general_group, QoS interactive (SLURM, --gres=gpu). Use when the user wants to debug a failing smoke test on a real DSI GPU, run pytest by hand on a node, or otherwise needs a GPU-attached bash session on DSI.
---

# dsi-shell

Opens an interactive shell on a UChicago DSI GPU node — `srun --pty bash` under account
`general_group`, QoS `interactive`. The user lands ready to `source .venv/bin/activate` and run
pytest by hand. Use for *debugging*, not batch verification — that's `dsi-smoke-test`.

> **⚠️ SKELETON.** Storage paths and compute-node CUDA driver version are **TBD until verified on
> DSI** (`hpc/dsi.md`).

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--account` | `general_group` | Only deviation: user names another account |
| `--qos` | `interactive` | 4 h, non-preemptable, 1 session |
| `--time` | `01:00:00` | Lower for short sessions; keep ≤ the 4 h cap |
| `--nodes` | `1` | Single-node |
| `--ntasks` | `1` | One shell |
| `--cpus-per-task` | `8` | |
| `--gres` | `gpu:1` | `gpu:2` for DDP-by-hand; `gpu:<type>:1` to pin a type |
| `--mem` | `64G` | |

## What this skill does

1. **Confirm walltime / GPU needs.** Default 1 GPU, 1 h. For DDP-by-hand set `--gres=gpu:2`; to
   pin hardware use `--gres=gpu:H100:1` etc.
2. **Run `srun --pty`** with the env staged so the user lands in the venv:
   ```bash
   srun --account=general_group --qos=interactive --time=01:00:00 \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 \
     --pty bash -lc 'cd /net/projects/general_group/awikner/physicsnemo; \
                    source .venv/bin/activate; exec bash'
   ```
3. **Don't drive work inside the shell** — this skill just opens it. For non-interactive runs
   the user should use `dsi-smoke-test` so output is captured.

## Refuse / push back when

- User asks for `--time` beyond the 4 h interactive cap, > the node's GPU count, or multi-node.
- The shell is being used to do what `dsi-smoke-test` already does — recommend that skill.

## Out-of-scope

- Long-lived sessions (4 h interactive cap; use `--qos=general` in a batch script for longer).
- Reproducible runs — wrap as a pytest smoke test and use `dsi-smoke-test`.
- Any other cluster — its own skill + doc.
