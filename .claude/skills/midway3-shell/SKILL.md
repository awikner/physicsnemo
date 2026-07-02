---
name: midway3-shell
description: Open an interactive shell on a UChicago RCC Midway3 GPU node via `srun --pty` on the pedramh-gpu (H100) partition under account pi-pedramh. Use when the user wants to debug a failing smoke test on a real Midway3 GPU, run pytest by hand on a node, or otherwise needs a GPU-attached bash session on Midway3.
---

# midway3-shell

Opens an interactive shell on a Midway3 GPU node — `srun --pty bash` against `pedramh-gpu`
(**H100**, the pedramh group's partition) under `pi-pedramh`. The user lands in the activated
ai-rossby `.venv`, ready to run pytest by hand. Use for *debugging*, not batch verification —
that's `midway3-smoke-test`. (Not the open `gpu` partition: its V100 is unsupported by cu129 torch.)

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `pedramh-gpu` (H100) | Never the open `gpu` (V100 — unsupported). Alt: `schmidt-gpu` (A100) |
| `--account` | `pi-pedramh` | Alt for `schmidt-gpu`: `pi-dfreedman` with `--qos=schmidt` |
| `--time` | `01:00:00` | Lower for short sessions; keep within the QoS cap |
| `--nodes` | `1` | Single-node |
| `--ntasks` | `1` | One shell |
| `--cpus-per-task` | `8` | |
| `--gres` | `gpu:1` | Bump to `gpu:2` for DDP-by-hand; nodes have 4 |
| `--mem` | `64g` | |

## What this skill does

1. **Confirm gpu count / walltime.** Default `pedramh-gpu` (H100), 1 GPU, 1 h. **Never the open
   `gpu` partition** (V100, unsupported by cu129 torch). When `pedramh-gpu` is busy, use
   `schmidt-gpu` with `--account=pi-dfreedman --qos=schmidt`.
2. **Run `srun --pty`** with the env staged so the user lands in the venv:
   ```bash
   srun --partition=pedramh-gpu --account=pi-pedramh --time=01:00:00 \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64g \
     --job-name=pn-shell \
     --pty bash -lc 'cd /project/pedramh/awikner/physicsnemo; \
                    source .venv/bin/activate; exec bash'
   ```
   Alternate (A100) when `pedramh-gpu` is busy: `--partition=schmidt-gpu --account=pi-dfreedman --qos=schmidt`.
3. **Don't drive work inside the shell** — this skill just opens it. For non-interactive runs the
   user should use `midway3-smoke-test` so output is captured.

## Refuse / push back when

- User asks for `--time` beyond the QoS cap, > the node's GPU count, or multi-node.
- The shell is being used to do what `midway3-smoke-test` already does — recommend that skill.

## Out-of-scope

- Long-lived sessions beyond the QoS cap.
- Reproducible runs — wrap as a pytest smoke test and use `midway3-smoke-test`.
- CPU preprocessing — `midway3-cpu-job`.
- Any other cluster — its own skill + doc.
