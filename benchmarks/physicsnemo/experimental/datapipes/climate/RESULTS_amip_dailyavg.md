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
