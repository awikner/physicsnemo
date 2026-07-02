---
name: stampede3-shell
description: Open an interactive shell on a TACC Stampede3 H100 GPU node via idev under allocation tg-atm170020. Use when the user wants to debug a failing smoke test on a real H100, run pytest by hand on a node, or otherwise needs a GPU-attached bash session on Stampede3.
---

# stampede3-shell

Opens an interactive shell on a Stampede3 **H100** node using TACC's `idev` node-grabber under
allocation `tg-atm170020`. The user lands ready to `source .venv/bin/activate` and run pytest by
hand. Use for *debugging*, not batch verification — that's `stampede3-smoke-test`.

> **⚠️ SKELETON.** Partition name and walltime cap are **TBD until verified on Stampede3**
> (`hpc/stampede3.md`). `idev` is TACC's preferred interactive allocator (over bare `srun --pty`).

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `-p` (partition) | **TBD** (`gpu-h100` / `h100`) | `sinfo -s \| grep -i h100` |
| `-A` (account) | `tg-atm170020` | Only deviation: user names another allocation |
| `-t` (time) | `01:00:00` | Lower for short sessions; keep ≤ the interactive cap |
| `-N` (nodes) | `1` | Single-node |
| `-n` (tasks) | `1` | One shell |

## What this skill does

1. **Confirm walltime / GPU needs.** Default 1 node, 1 h. For DDP-by-hand the user may want the
   full node's GPUs.
2. **Run `idev`** to grab the node:
   ```bash
   idev -p <GPU_PARTITION> -A tg-atm170020 -N 1 -n 1 -t 01:00:00
   ```
   `idev` drops the user onto the compute node. Then (they run, or you pre-stage the reminder):
   ```bash
   cd $WORK/physicsnemo && source .venv/bin/activate
   ```
3. **Don't drive work inside the shell** — this skill just opens it. For non-interactive runs
   the user should use `stampede3-smoke-test` so output is captured.

## Refuse / push back when

- User asks for walltime beyond the interactive cap, more GPUs than the node has, or multi-node.
- The shell is being used to do what `stampede3-smoke-test` already does — recommend that skill
  so output is captured.

## Out-of-scope

- Long-lived sessions (interactive walltime cap).
- Reproducible runs — wrap as a pytest smoke test and use `stampede3-smoke-test`.
- CPU preprocessing — `stampede3-cpu-job`.
- Any other cluster — its own skill + doc.
