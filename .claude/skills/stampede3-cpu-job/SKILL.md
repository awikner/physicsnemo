---
name: stampede3-cpu-job
description: Submit a CPU job (data conversion, climatology computation, normalization stats, .nc-to-Zarr conversion, multiprocessing batch, etc.) on TACC Stampede3's CPU partition under allocation tg-atm170020. Use whenever the user asks to run a data-conversion or non-GPU compute job on Stampede3. Pairs with stampede3-smoke-test (GPU) and stampede3-shell (interactive GPU).
---

# stampede3-cpu-job

Submits CPU-only preprocessing work to a TACC Stampede3 CPU partition under allocation
`tg-atm170020`, in the ai-rossby workflow defined by `hpc/stampede3.md`. Target: HDF5→Zarr
converters, climatology/bias aggregations, normalization-stat computations, and other
`multiprocessing` batches under `tools/data/`.

> **⚠️ SKELETON.** CPU partition name (`skx` / `normal`?) and walltime caps are **TBD until
> verified on Stampede3** (`sinfo -s`, `sinfo -o "%P %l"`). Confirm against `hpc/stampede3.md`.

## Partition + walltime

| Partition | Walltime cap | Use for |
|---|---|---|
| **TBD** (`skx` / `normal`) | **TBD** | Full-dataset conversions, multi-year batches |
| via `idev` on a CPU partition | interactive cap **TBD** | Quick fixture-sized conversions, debugging |

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `-p` (partition) | **TBD** CPU partition | `sinfo -s` on first login |
| `-A` (account) | `tg-atm170020` | Only deviation: user names another allocation |
| `-t` (time) | `04:00:00` | Bump for very large conversions; keep ≤ the cap |
| `-N` (nodes) | `1` | Single-node by contract |
| `-n` (tasks) | `1` | One process; `multiprocessing` spawns the pool internally |
| cpus | full node | Sets `SLURM_CPUS_PER_TASK`, which sizes the pool |
| `-J` (job name) | `pn-cpu` | |

## What this skill does, step by step

1. **Confirm the target.** A Python command (typically a `tools/data/` CLI), script path, or
   `bash -lc '...'` body. If missing, ask.
2. **Choose interactive vs batch.** `idev` for quick/"smoke"/"interactive" conversions;
   `sbatch` for full-scale ones.
3. **Build the body** (venv + test-data env):
   ```
   cd $WORK/physicsnemo && \
   source .venv/bin/activate && \
   export AI_ROSSBY_TEST_DATA=$WORK/physicsnemo_test_data && \
   <USER_COMMAND>
   ```
   The script should read `SLURM_CPUS_PER_TASK` (fallback `os.cpu_count()`) to size its pool.
4. **Submit.** `sbatch --wait` (blocks, output to file) for batch; `idev` then run for
   interactive. Report what was written (paths, file count, bytes) from the script's own log
   lines; on failure show the tail and stop.

## Refuse / push back when

- User requests a GPU partition for CPU work — point at `stampede3-smoke-test` (GPU) instead.
- User asks for walltime beyond the partition cap — stop and surface it.
- User names an allocation other than `tg-atm170020` without explanation — confirm first.
- The conversion script ignores `SLURM_CPUS_PER_TASK` — note it; parallelism won't match the
  allocation.

## Out-of-scope

- CUDA work — `stampede3-smoke-test` / `stampede3-shell`.
- Multi-node CPU jobs — single-node by contract (multiprocessing scales to one node).
- Any other cluster — its own skill + doc.
