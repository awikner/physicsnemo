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

## Re-measured against the strided read (2026-08-13, Polaris job 7448391)

The choice above was made against a W=7 window of **consecutive** rows, so one
8-row chunk served a whole window. Training actually steps 24 h over 6-hourly
rows (see the stride audit in `docs/dev/phase12_implementation_plan.md`), so a
W=7 window spans `1 + 6x4 = 25` rows and straddles ~4 chunks. 1981 rechunked to
1 and 32 rows, same reads, eagle Lustre, one Polaris node:

| time-chunk | stride | 0 workers | 4 workers | files (1981) | du |
|---|---|---|---|---|---|
| 1 | 1 | 4.73 | **17.35** | 43,843 | 3.5 G |
| 1 | **4** | 4.82 | **15.51** | 43,843 | 3.5 G |
| **8** (shipped) | 1 | 4.38 | 14.32 | 5,533 | 3.1 G |
| **8** (shipped) | **4** | 4.46 | 14.59 | 5,533 | 3.1 G |
| 32 | 1 | 3.05 | 9.07 | 1,423 | 3.0 G |
| 32 | **4** | 3.04 | 9.58 | 1,423 | 3.0 G |

single-frame baseline (chunk 8, 9.0 MB/sample): 31.36 / 97.21 samples/s.
Windowed reads are 32.4 MB/sample throughout.

**Striding is free.** Every chunking gives the same throughput at stride 4 as at
stride 1 (chunk 8: 4.46 vs 4.38 single-threaded, 14.59 vs 14.32 at 4 workers),
even though a strided window decompresses ~7x more rows. So the cost is
per-request latency, not chunk bytes — which is also why the original 64-row
build was slow and why 32 is the worst option here.

**The window-alignment rationale was wrong; "don't over-read" was right.**
Sizing the chunk to cover a window (32) is the *slowest* configuration, 35%
below chunk 8. Smaller is better: chunk 1 is ~21% faster at 4 workers (17.35 vs
14.32) — at **8x the file count** (43.8k vs 5.5k per year, i.e. ~2.0M vs 250k
files for 45 years).

**Recommendation: keep `--time-chunk 8`.** The 21% is on a read path that is
already ~3.5x faster than a `dim=1024`, W=6 training step consumes
(14.6 windows/s vs ~4), so it buys nothing today, while 2.0M files would be a
real problem on an inode-capped filesystem — which is exactly what pushed the
AMIP conversion off Derecho scratch in the first place
(`docs/dev/context/phase11-data-consolidation.md`). Revisit only if the loader
becomes the bottleneck on a filesystem with inodes to spare.

**Caveats.** One node, one year, page cache in play (the rechunked copies were
freshly written, so if anything they are flattered — and chunk 32 was still the
slowest). Absolute numbers are not comparable with the Derecho table above; the
ranking held across both strides and both worker counts.

Reproduce: `hpc/scripts/bench_amip_chunking_polaris.pbs`.

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
