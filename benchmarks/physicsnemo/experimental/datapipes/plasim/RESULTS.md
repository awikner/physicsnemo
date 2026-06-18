# PLASIM data loader benchmark — results & decision log

Benchmark file: [`loader_throughput.py`](loader_throughput.py).

Question: should the ai-rossby `PlasimClimateDatapipe` use the Zarr backend
(current Phase 2 choice) or stay closer to PanguWeather's per-timestep HDF5
layout? The Zarr design was justified on architectural grounds (dual sigma +
pressure level coordinate systems, irregular time axes for sparse
`train_data_sets.json` ranges, async-shardable chunks). This document records
the throughput trade-off and the decision.

## Throughput results (current, post-optimization)

### 1. Load-only microbench (Delta `gpuA40x4-interactive`)

Warm cache, 120-timestep fixture (`smoke_month_t1.zarr` with `time_chunk=1`):

| workers | bs | PanguH5 samples/s | Zarr samples/s | Zarr / PanguH5 |
|---|---|---|---|---|
| 0 | 1 | 55.3 | 77.2 | **1.40×** |
| 0 | 4 | 55.4 | 94.2 | **1.70×** |
| 2 | 1 | 96.9 | 150.6 | **1.55×** |
| 2 | 4 | 100.9 | 167.4 | **1.66×** |
| 4 | 1 | 185.4 | 219.0 | **1.18×** |
| 4 | 4 | 187.2 | 233.4 | **1.25×** |

Zarr **outperforms** PanguH5 at every configuration after the optimization
described in the next section. Earlier numbers (pre-optimization) had Zarr
at 0.22×–0.55× of PanguH5; the swing was a ~3× improvement on the Zarr column.

### 2. Training-step variant (production-shape PanguPlasim)

`num_workers=4`, `batch_size=4`, 30 batches, production-shape PanguPlasim
(`embed_dim=192`, `depths=(2, 6, 6, 2)`, `num_heads=(6, 12, 12, 6)`):

| backend | load(s) | step(s) | step/load | end-to-end batches/s |
|---|---|---|---|---|
| pangu_h5 | 0.845 | 2.464 | 2.92× | 12.17 |
| zarr     | 0.707 | 2.373 | 3.36× | **12.64** |

End-to-end throughput is now ~4% better on Zarr, with Zarr's load time
60% lower than before optimization (1.797s → 0.707s for the same 30 batches).

## The diagnosis (why Zarr was slow pre-optimization)

A cProfile run on the un-optimized dataset (20 samples, login node)
attributed **78% of `__getitem__` time** to zarr's asyncio bookkeeping:

| component | cumtime | % |
|---|---|---|
| `zarr.core.sync.sync` | 2.609 s | 31% |
| `asyncio.base_events._run_once` | 2.284 s | 27% |
| `selectors.epoll.select` | 1.789 s | 21% |
| `xarray.values` path | 2.944 s | 35% (overlapping with sync) |

Raw measurement of `zarr.open(arr)[i]` latency (no xarray): **2.5 ms per read**.
With ~10 reads per `__getitem__`, that's **~25 ms of asyncio overhead per
sample** baked into the synchronous code path.

The root cause is upstream: zarr-python 3 rewrote its core to be async-first,
and the synchronous read API now wraps every read in a `zarr.core.sync.sync(...)`
call that spins the asyncio event loop. Confirmed upstream:

- [zarr-developers/zarr-python#3524](https://github.com/zarr-developers/zarr-python/issues/3524)
  — "Array indexing with Zarr 3 is noticeably slower than with Zarr 2".
  Reporter measured `data[::step]` at **7.6× slower** in v3 (0.51 s vs 3.91 s);
  attributes it to "switching between synchronous and asynchronous code." Open
  as of Oct 2025.
- [zarr-developers/zarr-python#2084](https://github.com/zarr-developers/zarr-python/issues/2084)
  — "Inconsistent reading performance with multiple cpu threads". Chunked
  reads don't parallelize with threads (~2.8 s regardless of thread count),
  unchunked reads do. Confirms the sync-API serialization point.
- [Earthmover blog: Accelerating Xarray with Zarr-Python 3](https://www.earthmover.io/blog/xarray-open-zarr-improvements/)
  — concurrency tuning helps. They saw 10 s wall-time improvement just
  raising `async.concurrency` from 10 → 100.
- [zarr-developers/zarr-python#3757](https://github.com/zarr-developers/zarr-python/issues/3757)
  — open bug noting upstream benchmarks don't test v2-format reads, which
  is why this regression slipped through their CI.

## The optimization (what we did)

Three changes in
[`physicsnemo/experimental/datapipes/plasim/dataset.py`](../../../../../physicsnemo/experimental/datapipes/plasim/dataset.py):

1. **Module-level concurrency bump**: `zarr.config.set({"async.concurrency": 100})`
   at import (dotted-key form — full-dict form replaces and strips
   `async.timeout`).

2. **Cache raw `AsyncArray` handles at `__init__`**:
   ```python
   self._async_group = _zarr_sync(_zarr_open_group_async(self.zarr_path, mode="r"))
   self._async_arrays = {v: _zarr_sync(self._async_group.get(v)) for v in all_time_varying_vars}
   ```
   Pattern lifted from
   [`physicsnemo.experimental.datapipes.healda.loaders.zarr_loader.ZarrLoader`](../../../../../physicsnemo/experimental/datapipes/healda/loaders/zarr_loader.py).

3. **Single `asyncio.gather` per `__getitem__`** covering BOTH start and target
   sample reads (a tuple-indexed dataset returns `(start, target)` so we coalesce
   ~22 reads into 1 sync call). The previous implementation had 22 separate
   `sync()` calls per `__getitem__` (xarray's `.isel(time=i).values` invokes
   `sync()` for each variable).

4. **Eager-load constant boundaries**: they don't vary with time, so reading
   them once at `__init__` removes 3 reads from the per-sample batch.

Eager constants + 1 sync call replacing 22 = ~95% reduction in asyncio
bookkeeping per sample. Confirmed by the load-only Zarr column climbing
from 76 samples/s to 233 samples/s (3.1×) at `(num_workers=4, batch_size=4)`.

## cftime parity check (2026-06-18)

Phase G forces `decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)` on
all xarray opens (dataset + converters) so the in-memory time coord is uniform
across PLASIM (pre-1582 year 1, already cftime by default) and ERA5/E3SM
(post-1582 dates, otherwise decoded to `numpy.datetime64`). Re-ran the
load-only microbench on Delta after the change:

| workers | bs | PanguH5 samples/s | Zarr samples/s | Zarr / PanguH5 |
|---|---|---|---|---|
| 0 | 1 | 55.1 | 76.3 | **1.38×** |
| 0 | 4 | 54.5 | 90.9 | **1.67×** |
| 2 | 1 | 97.9 | 154.0 | **1.57×** |
| 2 | 4 | 101.5 | 171.5 | **1.69×** |
| 4 | 1 | 183.1 | 218.8 | **1.19×** |
| 4 | 4 | 187.4 | 235.6 | **1.26×** |

All cells within ±1% of the pre-cftime numbers above — cftime has **no
measurable impact** on the loader hot path because the dataset's per-sample
reads go through the cached async-zarr handles, not xarray. The time-coord
decode is a one-time cost at `__init__` and the per-sample reads bypass
xarray entirely (the optimized path documented above).

Footnote: an earlier re-bench against `smoke_month.zarr` (50-step time
chunks) initially showed a ~10× regression that was actually unrelated to
cftime — it was the fixture chunk-size mismatch from the previous section.
The benchmark now hard-codes `smoke_month_t1.zarr` (time_chunk=1) so the
correct fixture is always used.

## Future maintenance

A diagnostic test (`test_dataset_uses_batched_async_zarr_reads` in
[`test/datapipes/plasim/test_plasim_dataset.py`](../../../../../test/datapipes/plasim/test_plasim_dataset.py))
asserts that the cached-handle + batched-gather optimization is in place.
If somebody reverts the optimization without re-benchmarking, the test
fails — at which point they'll find this RESULTS.md.

When upstream fixes zarr-python's sync-API per-call overhead
(track [zarr-python#3524](https://github.com/zarr-developers/zarr-python/issues/3524)),
the hand-rolled batching becomes unnecessary. Recommended workflow at that
point:

1. Update the zarr pin in `pyproject.toml` to the fixed version.
2. Run this benchmark to confirm raw `zarr.open(arr)[i]` per-read latency
   has dropped to sub-millisecond (vs the 2.5 ms we measured against
   zarr 3.2.x).
3. If yes: revert the `_async_*` caching + `_read_many_async` + `_build_sample`
   structure in `dataset.py` to a simpler xarray-based or naive-sync-zarr
   per-variable read. Re-run this benchmark to confirm we haven't regressed
   end-to-end batches/s. The optimization complexity goes away with the
   upstream fix.
4. Delete the diagnostic test
   `test_dataset_uses_batched_async_zarr_reads` and update this RESULTS.md.

The `async.concurrency = 100` setting is harmless either way and worth keeping.

## When this decision would need to be revisited

* Even smaller GPU compute step than the production PanguPlasim. If a fast
  inference loop drops the GPU step under ~100 ms, the loader could become
  visible again.
* Cold-cache / first-epoch performance — these numbers are all warm-cache
  on Delta `/work/nvme` NVMe. Cold reads on Lustre are a different story
  (PanguH5's per-node cache helps it more than Zarr's would).

## Reproducing

```bash
# Load-only microbench (login node OK, but the GPU node has less CPU
# contention and is preferred):
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
(`--time-chunk 1` is the smart default; `--time-chunk 50` was the original
default and produced a 6× throughput regression because each per-sample
read had to decompress 50 timesteps to extract 1).

Stats files for the step benchmark live at
`/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data/`.
