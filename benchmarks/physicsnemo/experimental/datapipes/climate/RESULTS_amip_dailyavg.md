<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# AMIP daily-avg loader benchmark (Phase 12c.12)

Single-threaded `ClimateZarrDataset` sample reads, 100 random `(start, lead=1)`
pairs, Derecho develop node (8 cpus), Lustre scratch, 1981 stores.
Reproduce: `bench_amip_dailyavg_coarse.py` (job 7046751, 2026-08-07).

| store | grid | time-chunk | samples/s | ms/sample | MB/sample |
|---|---|---|---|---|---|
| amip_dailyavg (full-res) | 180×360 | 1 | 7.98 | 125.3 | 143.3 |
| amip_dailyavg_coarse | 45×90 | **8** | **14.65** | **68.3** | 9.0 |

- **Chunking finding**: the first coarse build used 64-step time chunks
  (= the writer's processing block) — every random sample read pulled 64×
  the needed data and throughput landed exactly at full-res parity
  (8.03 samples/s, job 7046709). Rebuilt with `--time-chunk 8` (aligned
  with the ERDM W+1=7 rolling-window read): 1.8× single-thread speedup,
  16× less bandwidth. Pinned by `test_coarsen_store_time_chunking`.
- Residual cost is per-chunk open latency (~37 chunk files per sample on
  Lustre), not bandwidth — parallel DataLoader workers scale it out
  (4–8 workers ≈ 60–120 samples/s, well above the ~4 windows/s a
  dim-1024 W=6 training step consumes).
- Store sizes (1981): full-res 55 GB (~72k files), coarse 3.0 GB
  (~6k files at chunk 8).

## Polaris/eagle follow-up (Phase 12c.12 gaps — job 7409438, 2026-08-10)

Coarse store on its training filesystem
(`/eagle/lighthouse-uchicago/physicsnemo-zarr/amip_dailyavg_coarse`),
Polaris debug node, 100 samples per cell.
Reproduce: `hpc/scripts/bench_amip_dailyavg_polaris.pbs`.

**Single-frame samples, DataLoader worker sweep:**

| workers | samples/s | ms/sample |
|---|---|---|
| 0 | 18.07 | 55.3 |
| 2 | 37.06 | 27.0 |
| 4 | 94.04 | 10.6 |
| 8 | **145.98** | 6.9 |

**Windowed W+1=7 reads (`SequenceDataset` — the ERDM training pattern):**

| workers | windows/s | ms/window | frame-reads/s |
|---|---|---|---|
| 0 | 4.23 | 236.6 | 30 |
| 2 | 7.69 | 130.0 | 54 |
| 4 | 14.69 | 68.1 | 103 |
| 8 | **25.14** | 39.8 | 176 |

- **Worker scaling is near-linear to 8** (8.1× single-frame, 5.9×
  windowed) — the "4–8 workers ≈ 60–120 samples/s" extrapolation from
  the Derecho run was conservative; eagle delivers 146.
- **Chunk-8 alignment validated on the real pattern**: a 7-frame window
  costs 236.6 ms single-threaded vs 7 × 55.3 = 387 ms if frames were
  independent reads — ~1.6× from consecutive frames sharing chunks.
- **Training headroom**: a dim-1024 W=6 ERDM step consumes ~4 windows/s
  per rank; 4 workers/rank (16/node across 4 ranks, ALCF's useful-worker
  ceiling) supplies ~14.7 windows/s per rank — ~3.7× the need. The
  loader is not the bottleneck at production scale.
- eagle single-threaded beats Derecho Lustre (18.07 vs 14.65 samples/s).
