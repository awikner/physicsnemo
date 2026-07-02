---
name: derecho-cpu-job
description: Submit a CPU job (data conversion, climatology computation, normalization stats, .nc-to-Zarr conversion, multiprocessing batch, etc.) on NCAR Derecho's CPU queue under project UCHI0014, using PBS (qsub). Use whenever the user asks to run a data-conversion or non-GPU compute job on Derecho. Pairs with derecho-smoke-test (GPU) and derecho-shell (interactive GPU).
---

# derecho-cpu-job

Submits CPU-only preprocessing to an NCAR Derecho CPU queue under project `UCHI0014`, in the
ai-rossby workflow defined by `hpc/derecho.md`. **Derecho is PBS, not SLURM** — `qsub`, not
`sbatch`. Target: HDF5→Zarr converters, climatology/bias aggregations, normalization-stat
computations, and other `multiprocessing` batches under `tools/data/`.

> **⚠️ SKELETON.** CPU queue name (`main`/`cpu`?) and walltime caps are **TBD until verified on
> Derecho** (`qstat -Q`). Confirm against `hpc/derecho.md`.

## Defaults (override only when the user asks)

| PBS directive | Default | Notes |
|---|---|---|
| `-A` (project) | `UCHI0014` | CPU project; only deviation: user names another |
| `-q` (queue) | **TBD** `main`/`cpu` | `qstat -Q` |
| `-l walltime` | `04:00:00` | Bump for large conversions; keep ≤ the queue cap |
| `-l select` | `1:ncpus=<node>:mem=128gb` | `$NCPUS` sizes the multiprocessing pool |
| `-N` (name) | `pn-cpu` | |

## What this skill does, step by step

1. **Confirm the target.** A Python command (typically a `tools/data/` CLI), script path, or
   `bash -lc '...'` body. If missing, ask.
2. **Choose interactive vs batch.** `qsub -I` on the CPU queue for quick/"interactive"
   conversions; `qsub script.pbs` for full-scale ones.
3. **Build the body** (venv + test-data env):
   ```
   cd /glade/work/awikner/physicsnemo && \
   source .venv/bin/activate && \
   export AI_ROSSBY_TEST_DATA=/glade/work/awikner/physicsnemo_test_data && \
   <USER_COMMAND>
   ```
   The script should read **`$NCPUS`** (PBS's per-job CPU count; fall back to `os.cpu_count()`)
   to size its pool — note this differs from SLURM's `$SLURM_CPUS_PER_TASK`.
4. **Submit and block.** `qsub`, then poll `qstat <id>` until done and read the log. Report what
   was written (paths, file count, bytes) from the script's own log lines; on failure show the
   tail and stop.

## Refuse / push back when

- User requests a GPU queue for CPU work — point at `derecho-smoke-test` (GPU) instead.
- User asks for walltime beyond the queue cap — stop and surface it.
- User names a project other than `UCHI0014` for CPU work without explanation — confirm first.
- The conversion script reads `$SLURM_CPUS_PER_TASK` but not `$NCPUS` — note it; on PBS the pool
  won't be sized correctly (the script should accept both, or `$NCPUS` on Derecho).

## Out-of-scope

- CUDA work — `derecho-smoke-test` / `derecho-shell`.
- Multi-node CPU jobs — single-node by contract.
- Any other cluster — its own skill + doc.
