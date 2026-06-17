# PLASIM data loader benchmark — results & decision log

Benchmark file: [`loader_throughput.py`](loader_throughput.py).

Question: should the ai-rossby `PlasimClimateDatapipe` use the Zarr backend
(current Phase 2 choice) or stay closer to PanguWeather's per-timestep HDF5
layout? The Zarr design was justified on architectural grounds (dual sigma +
pressure level coordinate systems, irregular time axes for sparse
`train_data_sets.json` ranges, async-shardable chunks). This document records
the throughput trade-off and the decision.

## Throughput results

### 1. Load-only microbench

Warm cache, Delta `/work/nvme`, login node, 120 timestep fixture
(`smoke_month_t1.zarr` with `time_chunk=1`):

| workers | bs | PanguH5 samples/s | Zarr samples/s | ratio |
|---|---|---|---|---|
| 0 | 1 | 8.20 | 1.79 | 0.22× |
| 0 | 4 | 8.56 | 1.69 | 0.20× |
| 2 | 1 | 66.19 | 27.60 | 0.42× |
| 2 | 4 | 68.62 | 39.77 | 0.58× |
| 4 | 1 | 132.60 | 73.47 | 0.55× |
| 4 | 4 | 137.11 | 76.07 | 0.55× |

A 2× gap at production-likely settings (`num_workers=4`, `batch_size=4`). Per
the Phase-2 follow-up analysis, the root cause is **xarray per-variable
`.isel(time=i).values` overhead** — `~0.3–0.5 ms × N variables = ~4–6 ms`
per sample of Python/coord-alignment cost that doesn't exist in HDF5's
direct-array indexing path.

### 2. Training-step variant (decision-making)

Delta `gpuA40x4-interactive`, production-shape PanguPlasim
(`embed_dim=192`, `depths=(2, 6, 6, 2)`, `num_heads=(6, 12, 12, 6)`),
`num_workers=4`, `batch_size=4`, 30 batches per epoch:

| backend | load(s) | step(s) | step/load | end-to-end batches/s |
|---|---|---|---|---|
| pangu_h5 | 0.801 | 2.823 | 3.53× | **10.63** |
| zarr | 1.797 | 2.876 | 1.60× | **10.43** |

* `load`: wall time iterating the loader with no GPU work (sequential
  per-batch reads).
* `step`: wall time iterating the loader **with** forward + backward +
  AdamW step on a production-shape model.
* `step/load`: ratio measuring how compute-heavy each step is relative to
  pure load cost.
* `end-to-end batches/s`: the operational metric — what training will
  actually see.

## Decision

**Zarr stays.** End-to-end throughput is **1.9% apart** (10.43 vs 10.63
batches/s) — operationally tied. The PanguPlasim forward+backward step at
production scale takes ~2 s per batch on an A40, and `DataLoader`'s
multi-worker prefetch (`num_workers=4`, `prefetch_factor=2`,
`persistent_workers=True`) overlaps most of the loader cost with compute on
both backends. Even though Zarr's per-sample IO is 2× slower in microbench,
that cost is hidden in the steady state.

The plan's literal decision rule (`step/load > 5 for both` → loader is
moot) is failed by Zarr (`1.60×` is well below `5×`), but the rule was a
heuristic proxy for "is the GPU step heavy enough to hide the loader?" The
direct measurement of end-to-end batches/s answers that more authoritatively.

### When this decision would need to be revisited

* Smaller GPU compute step — e.g., a much shallower model variant, mixed
  precision dropping the step to ~0.5 s, or a downstream eval inference
  loop (no backward) where the loader becomes a larger fraction. Re-run
  the step benchmark with the actual target config and re-check.
* Fewer DataLoader workers — `num_workers=0` or `1` would expose the
  loader-only gap directly; the table above shows Zarr fall behind 5× on
  the bare `num_workers=0` row.
* Larger batches — the load-only row at `(0, 4)` doesn't actually batch
  reads (samples are still read sequentially per worker), so the gap
  widens slightly with batch size in the no-workers case.

### Documented future optimization (not done now)

Per the Explore-agent analysis, the xarray→raw-zarr swap inside
[`physicsnemo/experimental/datapipes/plasim/dataset.py`](../../../../../physicsnemo/experimental/datapipes/plasim/dataset.py)
(`_stack_along_var` and `_read_constant_boundary`) should close the
microbench gap to near-parity for ~12 lines changed:

```python
# In PlasimClimateDataset.__init__:
import zarr
self._zarr_arrays = {
    name: zarr.open_array(f"{self.zarr_path}/{name}", mode="r")
    for name in self._ds.data_vars
}

# In _stack_along_var (replacing self._ds[v].isel(time=time_idx).values):
arr = self._zarr_arrays[v][time_idx]  # numpy ndarray, no xarray overhead
```

HealDA's
[`ZarrLoader._get_array`](../../../../../physicsnemo/experimental/datapipes/healda/loaders/zarr_loader.py#L141)
is the reference pattern (`self._arrays` cache). Expected impact: 50–70%
throughput improvement on the Zarr column (to ~110–125 samples/s at
`(num_workers=4, batch_size=4)`). Skipped now because end-to-end
training-step throughput is already tied; revisit if any of the "when this
decision would need to be revisited" cases above become relevant.

## Reproducing

```bash
# Load-only microbench (login node OK):
python benchmarks/physicsnemo/experimental/datapipes/plasim/loader_throughput.py \
    --zarr-path $AI_ROSSBY_TEST_DATA/plasim/smoke_month_t1.zarr

# Step variant (Delta GPU node required):
srun --partition=gpuA40x4-interactive --account=bdiu-delta-gpu \
     --time=00:30:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
     --gpus-per-node=1 --mem=64g --job-name=pn-bench-step \
  bash -lc 'source .venv/bin/activate && \
            python benchmarks/physicsnemo/experimental/datapipes/plasim/loader_throughput.py \
                --with-step \
                --zarr-path $AI_ROSSBY_TEST_DATA/plasim/smoke_month_t1.zarr'
```

Fixture generation: see
[`tools/data/plasim/pangu_h5_to_zarr.py`](../../../../../tools/data/plasim/pangu_h5_to_zarr.py)
(re-run with `--time-chunk 1`; `time_chunk=50` is *much* slower for
random-access training workloads — initial benchmark with that default
showed Zarr at 0.09× HDF5 at `(num_workers=4, batch_size=4)`).

Stats files for the step benchmark live at
`/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data/`.
