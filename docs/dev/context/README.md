<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ai-rossby context notes

These are durable engineering-context notes for the `ai-rossby` fork —
cross-session knowledge that isn't obvious from the code or git history:
cluster quirks, environment gotchas, data-pipeline decisions, and deferred work.
They were exported from the assistant's project memory into the repo so they
travel with `git` (independent of any machine or Claude account).

For a top-level orientation see the repo-root **`CLAUDE.md`**.

| Note | What it covers |
|---|---|
| [sfno-ddp-requirements](sfno-ddp-requirements.md) | Multi-GPU SFNO training: `torch<2.11` pin, wandb-on-all-ranks, recipe extras |
| [delta-gpu-partitions](delta-gpu-partitions.md) | Delta A40 vs A100 partitions; the SFNO benchmark uses A100 |
| [deltaai-venv-wandb-fix](deltaai-venv-wandb-fix.md) | DeltaAI (GH200/aarch64) broken conda wandb + torchrun/launch caveats |
| [sfno-e3sm-compute-bound-gh200](sfno-e3sm-compute-bound-gh200.md) | SFNO-E3SM is compute-bound on GH200; node-local staging off by default |
| [phase11-data-consolidation](phase11-data-consolidation.md) | Phase 11: convert all datasets, consolidate; Globus/tar/inode gotchas; ERA5 norm fix |
| [derecho-retire-rehome-to-delta](derecho-retire-rehome-to-delta.md) | **DEFERRED** work: retire inode-limited Derecho scratch, re-home to Delta |
| [known-test-failures](known-test-failures.md) | **Both fixed 2026-08-14.** Two recipe dirs both export `train` *and* `ema` (import-collision trap any new recipe dir can hit); a stale ArchesWeather guard assertion |
| [PhysMetrics](PhysMetrics.md) | PhysMetrics.Weather install fix + custom mass-drift/TCWV-bias scripts for the hackathon hindcasts (`pangu_s2s`/`sfno_s2s` vs ERA5) |
| [lat-orientation-audit](lat-orientation-audit.md) | **Every registry store audited + 346 repaired 2026-08-14.** True row order of each raw archive; the whole AMIP family was upside-down (relabelled S→N, 322 stores); ERA5 half-flips + Derecho 1979–1999 fixed; **Delta's 4 ERA5 stores still outstanding** (filesystem full) |
| [rsi-h1-precond-instability](rsi-h1-precond-instability.md) | **Root-caused + fixed 2026-08-22.** RSI A2 divergence at ~11.7k steps = raw H1 state readout (missing EDM skip/out); `loss.h1_precond: edm` is the fix and default. Confirmed by 2x-lr three-way: raw trips at b5795, edm at b7092, ERDM itself at b7145 — with the fix RSI matches ERDM's stability; never raise base lr to 1e-4 (Muon 10x) on this trunk |
