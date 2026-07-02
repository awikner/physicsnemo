---
name: stampede3-smoke-test
description: Submit a pytest target as a GPU smoke test on TACC Stampede3's H100 partition under allocation tg-atm170020. Use whenever the user asks to run a smoke test, run GPU tests, or verify a ported physicsnemo feature on Stampede3/H100. Blocks until pytest exits and reports the result.
---

# stampede3-smoke-test

Runs a pytest smoke target on a TACC Stampede3 **H100** node under allocation `tg-atm170020`,
in the ai-rossby workflow defined by `hpc/stampede3.md`. Pairs with `stampede3-shell`
(interactive) and `stampede3-cpu-job` (CPU preprocessing).

> **⚠️ SKELETON.** Written from the Phase 9 plan; the partition name and the run mechanism
> (`sbatch --wait` vs. direct `srun`) are **TBD until verified on Stampede3**. Confirm against
> `hpc/stampede3.md` before relying on this. Never exceed the interactive/smoke walltime once
> the real caps are known.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `-p` / `--partition` | **TBD** (`gpu-h100` / `h100`) | Verify with `sinfo -s \| grep -i h100` |
| `-A` / `--account` | `tg-atm170020` | Only deviation: user names another allocation |
| `-t` / `--time` | `00:30:00` | Bump to `01:00:00` for DDP; keep ≤ the smoke cap |
| `-N` (nodes) | `1` | Smoke tests are single-node by contract |
| `-n` (tasks) | `1` | Pytest runs once; DDP driven by `torchrun` inside |
| GPUs | `1` | Bump to `2` for DDP; TACC H100 nodes — verify per-node GPU count |
| `-J` (job name) | `pn-smoke` | |

## What this skill does, step by step

1. **Confirm the target.** A pytest path, `-k` expression, or node-id. If missing, ask.
2. **Choose the run mechanism.** TACC discourages bare `srun` from login nodes, so default to
   **`sbatch --wait`** (submits, blocks until pytest exits, output to a log file which is then
   tailed). If first-login verification (see `hpc/stampede3.md`) established that direct `srun`
   works, the Delta-style streaming one-liner may be used instead.
3. **Build the job body** (venv activation via login shell):
   ```
   cd $WORK/physicsnemo && \
   source .venv/bin/activate && \
   pytest -m "smoke and cuda" -x -q <TARGET>
   ```
   For DDP, prepend `torchrun --standalone --nproc-per-node=2 -m ` before `pytest`.
4. **Submit and block.** `sbatch --wait hpc/scripts/smoke_stampede3.sbatch` with `TARGET`
   exported. Do NOT queue-and-forget — the user invoked this to see the result.
5. **Report.** On pass, state the test name + wall time. On failure, show the pytest tail
   (FAILED line + traceback) and stop — do not silently rerun.

## Example (Bash tool)

```bash
TARGET=test/models/pangu_plasim/ sbatch --wait \
  -p <GPU_PARTITION> -A tg-atm170020 -t 00:30:00 -N 1 -n 1 -J pn-smoke \
  -o hpc/scripts/logs/smoke-%j.out \
  hpc/scripts/smoke_stampede3.sbatch
# then tail the newest hpc/scripts/logs/smoke-*.out and report the tail
```

## Refuse / push back when

- User requests a non-H100 / non-smoke partition for a smoke test — ask whether they actually
  want a fidelity/recipe job (which belongs in `hpc/scripts/` with its own script).
- User asks for walltime beyond the smoke cap, > the node's GPU count, or multi-node — stop and
  surface the contract violation.
- User names an allocation other than `tg-atm170020` without explanation — confirm first.
- The repo is not on `ai-rossby` — note it (smoke tests likely don't exist on `main`).

## Out-of-scope

- Long-running fidelity tests — separate non-interactive job script.
- Multi-node DDP — out of scope for smoke.
- Interactive debugging — use `stampede3-shell`.
- CPU preprocessing — use `stampede3-cpu-job`.
- Any other cluster — that cluster has its own skill + doc.
