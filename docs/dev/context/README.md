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
| [rsi-h1-precond-instability](rsi-h1-precond-instability.md) | **The full RSI A2 stability campaign (2026-08-22..24).** Three implementation-vs-proposal deviations found and fixed (raw H1 readout → `h1_precond: edm`; state-unit SNR → measured `delta_std`; white unscaled Γ → `noise_scale_path` increment units). Residual property: the objective sharpens with fit quality, so EVERY constant lr (Muon or AdamW) holds its floor ~4-6k steps then excurses; clip/governor/rewind mechanisms tried and documented. **Production recipe: decay-and-harvest** (CosineToFloor, take the last pre-excursion epoch checkpoint). Never raise base lr to 1e-4 (Muon 10x) on this trunk |
| [rsi-drift-diagnosis-brief](rsi-drift-diagnosis-brief.md) | **Measurements (2026-09-08):** RSI 5-year rollouts collapse ~70% toward each channel's global mean while the ERDM control through the identical pipeline is clean; what is ruled out; hypotheses H1-H9 |
| [rsi-drift-diagnosis-report](rsi-drift-diagnosis-report.md) | **Diagnosis (2026-09-08):** two layers - a proven fresh-slot variance deficit (`_fresh_slot` seeds a conditional mean + 1.0 S_c; under-dispersion, member locking; inference fix ~1.45 S_c) and a leverage pathway for a head level bias (scalar-gamma EDM skip, c_skip 0.631 at the anchor readout); brief corrections (tau_W = 1/12 not 11/72), confirmed code/config bugs, ranked Polaris tests; probes in `tools/diagnostics/rsi_drift/` |
| [rsi-anchor-fix-report-for-proposal-revision](rsi-anchor-fix-report-for-proposal-revision.md) | **For the proposal's author (2026-09-10):** what the Polaris campaign established (RSI's entering-slot anchor is a conditional mean iterated without innovation; perturbation cannot repair it; self-generated anchors delay the collapse ~2 years), the proposed lag-W sample anchor (anchor slot w on the emitted frame w-W; W-day increment noise), its implementation and cost, and suggested revisions to sections 3.2, 3.6, 5 and 7 and to the A0-A5 ablation ladder |
| [rsi-ablation-ladder-status](rsi-ablation-ladder-status.md) | **Ladder audit (2026-09-10):** every trained RSI run on both contracts is A2 (with the campaign-1 edm readout and A3's increment-scaled Gamma); A1, A3, A4, A5 never trained/run; both ERDM controls were trained upstream, none in-harness -- A1 is the fallback and the missing in-harness A0 |
| [rsi-a0-a1-batch40-plan](rsi-a0-a1-batch40-plan.md) | **Plan (2026-09-10; A0 launched, A1 superseded by A2-L):** in-harness A0 (ERDM v2) and A1 (RSI in the exact ERDM reduction) on the sst_pred contract at batch 40; upstream sst_pred ERDM training config decoded from its Lightning checkpoint (the A0 recipe reference); `rsi_a1.yaml` instantiation fix |
| [rsi-v0.2-lag-w-implementation](rsi-v0.2-lag-w-implementation.md) | **Proposal v0.2 implemented (2026-09-10):** `anchor_lag` L in `RSIScheduler` (slot w anchored on frame w-L; L = W anchors the fresh slot on the emitted sample), 2W-frame training samples, anchor-time ocean imposition, pre-IC boundary frames in every driver, `loss/rsi_a2l.yaml`; **finding:** the shipped `sigma_c_*.pt` were 6-hour, not one-step, increment scales (one-step is ~3x larger); A0 and A2-L batch-40 baselines running on Polaris with the evaluation protocol |
