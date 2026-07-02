---
name: derecho-shell
description: Open an interactive shell on an NCAR Derecho A100 GPU node via `qsub -I` under project UCHI0018 (PBS). Use when the user wants to debug a failing smoke test on a real A100, run pytest by hand on a node, or otherwise needs a GPU-attached bash session on Derecho.
---

# derecho-shell

Opens an interactive shell on a Derecho **A100** node with PBS's `qsub -I` under project
`UCHI0018`. **Derecho is PBS, not SLURM** — this is `qsub -I`, not `srun --pty`. Use for
*debugging*, not batch verification — that's `derecho-smoke-test`.

> **⚠️ SKELETON.** Queue name (plan guess `develop`) and walltime cap are **TBD until verified
> on Derecho** (`hpc/derecho.md`).

## Defaults (override only when the user asks)

| PBS directive | Default | Notes |
|---|---|---|
| `-A` (project) | `UCHI0018` | Only deviation: user names another project |
| `-q` (queue) | **TBD** `develop` | `qstat -Q` |
| `-l walltime` | `01:00:00` | Lower for short sessions; keep ≤ the interactive cap |
| `-l select` | `1:ncpus=8:ngpus=1:mem=64gb` | `ngpus=2` for DDP-by-hand; A100 nodes have 4 |

## What this skill does

1. **Confirm walltime / GPU needs.** Default 1 GPU, 1 h. For DDP-by-hand set `ngpus=2`.
2. **Run `qsub -I`** to grab the node:
   ```bash
   qsub -I -A UCHI0018 -q develop \
     -l walltime=01:00:00 -l select=1:ncpus=8:ngpus=1:mem=64gb
   ```
   Once on the node, the user runs:
   ```bash
   module load ncarenv/24.12 cuda/12.5 python/3.12.5    # versions TBD
   cd /glade/work/awikner/physicsnemo && source .venv/bin/activate
   ```
3. **Don't drive work inside the shell** — this skill just opens it. For non-interactive runs
   the user should use `derecho-smoke-test` so output is captured.

## Refuse / push back when

- User asks for walltime beyond the interactive cap, > 4 GPUs, or multi-node.
- The shell is being used to do what `derecho-smoke-test` already does — recommend that skill.

## Out-of-scope

- Long-lived sessions (interactive walltime cap).
- Reproducible runs — wrap as a pytest smoke test and use `derecho-smoke-test`.
- CPU preprocessing — `derecho-cpu-job`.
- Any other cluster — its own skill + doc.
