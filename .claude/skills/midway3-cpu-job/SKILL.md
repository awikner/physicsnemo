---
name: midway3-cpu-job
description: Submit a CPU job (data conversion, climatology computation, normalization stats, .nc-to-Zarr conversion, multiprocessing batch, etc.) on UChicago RCC Midway3's caslake CPU partition under account pi-pedramh. Use whenever the user asks to run a data-conversion or non-GPU compute job on Midway3. Pairs with midway3-smoke-test (GPU) and midway3-shell (interactive GPU).
---

# midway3-cpu-job

Submits CPU-only preprocessing to UChicago RCC Midway3's `caslake` SLURM partition under account
`pi-pedramh`, in the ai-rossby workflow defined by `hpc/midway3.md`. Target: HDF5→Zarr converters,
climatology/bias aggregations, normalization-stat computations, and other `multiprocessing`
batches under `tools/data/`.

## Defaults (override only when the user asks)

| Flag | Default | Notes |
|---|---|---|
| `--partition` | `caslake` | General Midway3 CPU partition |
| `--account` | `pi-pedramh` | Only deviation: user names another account |
| `--time` | `04:00:00` | Bump for large conversions; keep within the QoS cap |
| `--nodes` | `1` | Single-node by contract |
| `--ntasks` | `1` | One process; `multiprocessing` spawns the pool internally |
| `--cpus-per-task` | `32` | Sets `SLURM_CPUS_PER_TASK`, which sizes the pool |
| `--mem` | `64g` | Most conversions are memory-bound |
| `--job-name` | `pn-cpu` | |

## What this skill does, step by step

1. **Confirm the target.** A Python command (typically a `tools/data/` CLI), script path, or
   `bash -lc '...'` body. If missing, ask.
2. **Build the body** (venv + test-data env):
   ```
   cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate && \
   export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data && \
   <USER_COMMAND>
   ```
   The script should read `SLURM_CPUS_PER_TASK` (fallback `os.cpu_count()`) to size its pool.
3. **Submit via `srun`.** Stream output; job ends when the command does. Do NOT use `--pty`. Do NOT
   `sbatch` — the user invoked this to see the result. If a job is rejected for a missing QoS, add
   `--qos=<pi-pedramh QoS>` (see `hpc/midway3.md`).
4. **Report.** On exit 0, summarize what was written (paths, file count, bytes) from the script's
   own log lines. On failure, show the tail and stop.

## Example (Bash tool)

```bash
srun --partition=caslake --account=pi-pedramh --time=02:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64g \
  --job-name=pn-cpu \
  bash -lc 'cd /project/pedramh/awikner/physicsnemo && source .venv/bin/activate && \
            export AI_ROSSBY_TEST_DATA=/project/pedramh/awikner/physicsnemo_test_data && \
            python tools/data/<dataset>/<script>.py [args]'
```

## Refuse / push back when

- User requests a GPU partition for CPU work — point at `midway3-smoke-test` (GPU) instead.
- User asks for walltime beyond the QoS cap — stop and surface it.
- User names an account other than `pi-pedramh` without explanation — confirm first.
- The conversion script ignores `SLURM_CPUS_PER_TASK` — note it; parallelism won't match the
  allocation.

## Out-of-scope

- CUDA work — `midway3-smoke-test` / `midway3-shell`.
- Multi-node CPU jobs — single-node by contract (multiprocessing scales to one node).
- Any other cluster — its own skill + doc.
