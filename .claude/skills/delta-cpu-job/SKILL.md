---
name: delta-cpu-job
description: Submit a CPU job (data conversion, climatology computation, normalization stats, .nc-to-Zarr conversion, multiprocessing batch job, etc.) on NCSA Delta's `cpu` or `cpu-interactive` partition under account `bdiu-delta-cpu`. Use whenever the user asks to run a data-conversion script, a climatology/bias computation, a multiprocessing batch on Delta CPUs, or any non-GPU compute job on Delta. Streams output back; non-interactive. Pairs with delta-smoke-test (GPU smoke) and delta-shell (interactive GPU).
---

# delta-cpu-job

Submits a CPU command to NCSA Delta's `cpu` or `cpu-interactive` SLURM partition under
account `bdiu-delta-cpu`, in the ai-rossby workflow defined by `hpc/delta.md`. The job
blocks until the command exits and streams stdout/stderr back. Pair with
`delta-smoke-test` (GPU) and `delta-shell` (interactive GPU).

The default target is **data-conversion / preprocessing work** that doesn't need a GPU:
HDF5→Zarr converters, climatology / bias aggregations, normalization-stat computations,
and other multiprocessing batches. Most ai-rossby data-conversion CLIs live in
`tools/data/` and use `multiprocessing` (or `concurrent.futures`) to parallelize across
years / files / variables — this skill exists so those jobs get the right partition + the
right CPU count without the user having to remember the SLURM flags every time.

## Partition + walltime

| Partition | Walltime cap | Use for | Defaults |
|---|---|---|---|
| `cpu-interactive` | **1 h** | Quick conversions on a small fixture (smoke-style), interactive debugging | `--time=00:30:00`, `--cpus-per-task=32` |
| `cpu` | **2 days** | Full-dataset conversions, multi-year batches, large climatology aggregates | `--time=04:00:00`, `--cpus-per-task=64` |

Pick `cpu-interactive` when the work fits in under an hour (a few years of PLASIM, the
fixture-only PLASIM conversion, the smoke fixture for a new converter). Pick `cpu` for
anything full-scale.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `cpu` | Use `cpu-interactive` for ≤ 30-min jobs |
| `--account` | `bdiu-delta-cpu` | Only deviation: user explicitly names another account |
| `--time` | `04:00:00` (cpu), `00:30:00` (cpu-interactive) | Bump up for very large conversions; cap is 2 days for `cpu`, 1 hr for `cpu-interactive` |
| `--nodes` | `1` | All ai-rossby CPU jobs are single-node by contract |
| `--ntasks-per-node` | `1` | The script is one process; `multiprocessing` spawns the pool internally |
| `--cpus-per-task` | `64` (cpu), `32` (cpu-interactive) | Sets the `multiprocessing.Pool` size via `SLURM_CPUS_PER_TASK` |
| `--mem` | `128g` (cpu), `64g` (cpu-interactive) | Most data-conversion jobs are memory-bound |
| `--job-name` | `pn-cpu` | Override if running several in parallel |

## What this skill does, step by step

1. **Confirm the target.** The user gives a Python command (typically a CLI under
   `tools/data/`), a script path, or a `bash -lc '...'` body. If they don't, ask before
   submitting.
2. **Choose partition.** Default to `cpu` (2-day cap). Switch to `cpu-interactive` when
   the user says "interactive", "quick", "smoke", or names a target known to finish in
   under an hour.
3. **Build the command.** Use the wrapper `bash -lc` form so the venv activation works:
   ```
   cd /work/nvme/bdiu/awikner/physicsnemo && \
   source .venv/bin/activate && \
   export AI_ROSSBY_TEST_DATA=/work/nvme/bdiu/awikner/physicsnemo_test_data && \
   <USER_COMMAND>
   ```
   For data conversions that spawn a `multiprocessing.Pool`, the script should read
   `SLURM_CPUS_PER_TASK` (or default to `os.cpu_count()`) to size the pool.
4. **Submit via `srun`.** Stream the output; the job ends when the command does.
   Do NOT use `--pty` (that's `delta-shell`-style and incompatible with `bash -lc`).
   Do NOT use `sbatch` — the user invoked this skill to *see* the result.
5. **Report.** If the command exited 0, summarize what was written (output paths, file
   count, total bytes) — pull this from the script's own log lines, don't fabricate.
   If it failed, show the tail (stderr + last ~30 stdout lines) and stop.

## Example commands (Bash tool)

**Full PLASIM normalization conversion** (large batch):

```bash
srun \
  --partition=cpu \
  --account=bdiu-delta-cpu \
  --time=02:00:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
  --mem=128g \
  --job-name=pn-plasim-norm \
  bash -lc 'cd /work/nvme/bdiu/awikner/physicsnemo && \
            source .venv/bin/activate && \
            python tools/data/plasim/build_normalization_zarr.py \
              --mean /work/nvme/.../data_12-132_mean_sigma.nc \
              --std  /work/nvme/.../data_12-132_std_sigma.nc \
              --output /work/nvme/.../plasim/normalization_12-132.zarr'
```

**Fixture-sized smoke conversion** (interactive):

```bash
srun \
  --partition=cpu-interactive \
  --account=bdiu-delta-cpu \
  --time=00:30:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 \
  --mem=64g \
  --job-name=pn-fixture-norm \
  bash -lc 'cd /work/nvme/bdiu/awikner/physicsnemo && \
            source .venv/bin/activate && \
            python tools/data/plasim/build_normalization_zarr.py [...] --output [...]'
```

## Refuse / push back when

- User asks to run on a partition other than `cpu` / `cpu-interactive` for a CPU job —
  ask whether they really want GPU (`delta-smoke-test`) or a non-Delta cluster.
- User asks for `--time` > 2 days on `cpu` (hard partition cap) or > 1 hour on
  `cpu-interactive` — stop and surface the contract violation.
- User asks for an account other than `bdiu-delta-cpu` for a CPU job without
  explanation — confirm before proceeding.
- The conversion script doesn't honor `SLURM_CPUS_PER_TASK` — note it; the parallelism
  will not match the allocation.

## Out-of-scope

- Anything that needs CUDA — `delta-smoke-test` (smoke GPU) or `delta-shell` (interactive
  GPU) instead.
- Multi-node CPU jobs — out of scope; the ai-rossby data-conversion jobs are single-node
  by contract (multiprocessing scales to one node fine).
- Interactive CPU debugging shells — out of scope here; use `--pty` via the user
  directly or extend `delta-shell` if it becomes a common pattern.
- Submitting CPU jobs on any other cluster — that cluster needs its own skill + doc.
