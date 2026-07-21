<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SFNO-E3SM is compute-bound on GH200

> Point-in-time benchmark result (2026-07-09).

Benchmarked the optimized SFNO-E3SM config (bf16 + `expandable_segments` +
`checkpointing=2` + fused AdamW) on a single NCSA DeltaAI **GH200** (job 2634504,
single GPU, batch 4, 120 steps) to confirm it is **compute-bound** — the question
mattered because GH200 is faster than the A100 the original optimization ran on,
and a faster GPU can tip a workload toward I/O-bound.

**Result: compute-bound, three independent lines of evidence:**
- Steady-state step time **272.9 ms** (14.7 samples/s/GPU) — vs A100 optimized
  0.766 s/step at batch-4/GPU → GH200 ~2.8× faster/GPU (sane Hopper vs Ampere).
- **GPU utilization median 99% / mean 94%** during steady state (94% of samples
  ≥90%); no mid-training starvation dips. Power ~519 W, mem-bw util 68%, only
  27.6 GB of 120 GB used (checkpointing keeps memory low).
- **Step time invariant to data-loader workers**: `num_workers` 8 → 1 changed
  nothing (272.9 vs 272.3 ms), so a single loader keeps the GPU fully fed → I/O is
  not on the critical path.

**Implication:** node-local data staging (`dataset.stage_to_local`) correctly
stays OFF by default for this config — it buys nothing when GPU-bound. Enable it
only for data-bound runs. (The stager is a thread-pool `sendfile` copier, not
MPI; see `examples/weather/ai_rossby/data_staging.py`.)

Env caveats: [deltaai-venv-wandb-fix](deltaai-venv-wandb-fix.md).
