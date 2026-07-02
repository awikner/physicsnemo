---
name: derecho-smoke-test
description: Submit a pytest target as a GPU smoke test on NCAR Derecho's GPU dev queue under project UCHI0018, using PBS (qsub). Use whenever the user asks to run a smoke test, run GPU tests, or verify a ported physicsnemo feature on Derecho/A100. PBS-aware; blocks until the job finishes and reports the result.
---

# derecho-smoke-test

Runs a pytest smoke target on an NCAR Derecho **A100** node under project `UCHI0018`, in the
ai-rossby workflow defined by `hpc/derecho.md`. **Derecho uses PBS Pro, not SLURM** — this skill
submits with `qsub`, not `sbatch`/`srun`. Pairs with `derecho-shell` and `derecho-cpu-job`.

> **⚠️ SKELETON.** GPU queue name (plan guess `develop`), its walltime cap, and the module
> versions are **TBD until verified on Derecho** (`qstat -Q`, `module avail`). Confirm against
> `hpc/derecho.md` before relying on this.

## Defaults (override only when the user asks)

| PBS directive | Default | Notes |
|---|---|---|
| `-A` (project) | `UCHI0018` | GPU project; only deviation: user names another |
| `-q` (queue) | **TBD** `develop` | Verify it's GPU-capable with `qstat -Q` |
| `-l walltime` | `00:30:00` | Bump to `01:00:00` for DDP; keep ≤ the queue cap |
| `-l select` | `1:ncpus=8:ngpus=1:mem=64gb` | Bump `ngpus=2` for DDP; A100 nodes have 4 |
| `-N` (name) | `pn-smoke` | |

## What this skill does, step by step

1. **Confirm the target.** A pytest path, `-k` expression, or node-id. If missing, ask.
2. **Choose resources.** Default 1 GPU / 30 min. For "DDP"/"multi-GPU", set `ngpus=2` and
   `walltime=01:00:00`. Never exceed 4 GPUs, 1 node, or the queue's walltime cap.
3. **Submit via `qsub`.** Use the committed script and pass the target through PBS `-v`:
   ```bash
   TARGET=<TARGET> qsub -v TARGET hpc/scripts/smoke_derecho.pbs
   ```
   For ad-hoc resource changes, add `-l select=...`/`-l walltime=...`/`-q ...` on the qsub line
   (they override the script's directives).
4. **Block on completion.** `qsub` returns a job id immediately, so poll `qstat <id>` until the
   job leaves the queue, then read `hpc/scripts/logs/pn-smoke.o<jobid>`. (Do not queue-and-walk-
   away — the user invoked this to see the result.)
5. **Report.** On pass, state the test name + wall time. On failure, show the log tail (FAILED
   line + traceback) and stop — do not silently resubmit.

## PBS cheatsheet (differs from SLURM)

- Submit: `qsub script.pbs` · Status: `qstat <id>` / `qstat -u awikner` · Cancel: `qdel <id>`
- Job id: `$PBS_JOBID` · CPU count: `$NCPUS` · Directives can't expand shell vars.

## Refuse / push back when

- User requests a non-GPU or non-dev queue for a smoke test — ask whether they actually want a
  fidelity/recipe job (its own PBS script under `hpc/scripts/`).
- User asks for walltime beyond the dev-queue cap, > 4 GPUs, or multi-node — stop and surface it.
- User names a project other than `UCHI0018` for GPU work without explanation — confirm first.
- The repo is not on `ai-rossby` — note it (smoke tests likely don't exist on `main`).

## Out-of-scope

- Long-running fidelity tests — separate PBS script on a batch GPU queue.
- Multi-node DDP — out of scope for smoke.
- Interactive debugging — use `derecho-shell`.
- CPU preprocessing — use `derecho-cpu-job`.
- Any other cluster — its own skill + doc.
