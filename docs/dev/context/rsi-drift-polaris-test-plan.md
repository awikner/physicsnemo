<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI drift: Polaris test plan and results log (2026-09-08)

Executes section 4 of [rsi-drift-diagnosis-report](rsi-drift-diagnosis-report.md)
on ALCF Polaris. Working tree `$R = /eagle/lighthouse-uchicago/members/awikner/physicsnemo-rsi`
(not a git checkout; files are shipped by `scp` from branch `rsi-drift-diagnosis`,
whose base is byte-identical to `$R` for every file touched). Data root
`$AI_ROSSBY_DATA = /eagle/lighthouse-uchicago/physicsnemo-zarr`. Environment
`$R/polaris_env.sh`. Checkpoint under test: `checkpoints_prod24_b40/RollingDiTWrapper.0.24.mdlus`
(+ EMA from `checkpoint.0.24.pt`); control: `amip-checkpoints/sstpred_epochs/erdm_sstpred_ep24.mdlus`.

Code shipped for this plan (all on the branch): `rsi.py` gains three
non-default knobs — `fresh_noise_scale` (inference, multiplies the fresh-slot
latent), `anchor_shrink` (training, pattern-shrink anchor augmentation) and an
env-gated per-roll trace (`RSI_TRACE_PATH`, records the emitted frame's, the
fresh-slot anchor's and every window slot's per-channel spatial-anomaly std
and mean); sampler variants `conf/sampler/rsi_sstpred_e1_{fd,k120,k145,k180}.yaml`;
the probe driver `examples/weather/ai_rossby/rsi_drift_probes.py`; job scripts
`hpc/scripts/polaris_rsi_drift_{probes_phase1,sweep300_phase2,eval5yr_phase3}.pbs`.
`fresh_noise_scale = 1.0` and `anchor_shrink = 0` reproduce the shipped code
bit for bit (88 scheduler tests pass).

## Queues used

| phase | queue | nodes | wall | why |
|---|---|---|---|---|
| 0 (zero compute) | none | - | - | reads existing `eval_suite.pt` and `sigma_c` on this machine |
| 1 probes | `debug` | 1 | 1 h | one process per GPU; 1 running job per user |
| 2 sweep | `debug-scaling` | 10 | 1 h | five 2-node member-split configs; 1 queued per user |
| 3 five-year | `preemptable` | 2 | 3.5 h | `prod` routes 10-24 nodes only; `capacity` is held by the bundle chain |
| 4 fine-tunes | `prod` -> `small` or `preemptable` | 10 | 3 h links | conditional on Phase 1 |

The bundle-recipe retrain chain (jobs 7599770-7599776, `small`, 10 nodes) and
its `capacity` sibling (7599793) were running/held when this started; nothing
here touches them. Allocation balance at start: 33,329 node-hours.

## Phase 0: zero-compute tests (done locally, 2026-09-08)

**Test 6, dispersion already on disk.** `rmse_acc['spread_step{S}_{group}']`
from `eval_bias5yr_e24/eval_suite.pt` and `eval_erdm_bias5yr_e24/eval_suite.pt`
(8 members, lat-weighted, normalized units):

| step | surface RSI / ERDM / ratio | upper air RSI / ERDM / ratio | diagnostic RSI / ERDM / ratio |
|---|---|---|---|
| 1 | 0.016 / 0.017 / 0.97 | 0.019 / 0.019 / 1.00 | 0.061 / 0.061 / 1.00 |
| 5 | 0.042 / 0.051 / 0.84 | 0.067 / 0.073 / 0.91 | 0.137 / 0.147 / 0.93 |
| 10 | 0.110 / 0.136 / 0.81 | 0.156 / 0.178 / 0.88 | 0.221 / 0.251 / 0.88 |
| 20 | 0.232 / 0.286 / 0.81 | 0.392 / 0.398 / 0.99 | 0.360 / 0.420 / 0.86 |
| 50 | 0.291 / 0.317 / 0.92 | 0.720 / 0.484 / 1.49 | 0.457 / 0.447 / 1.02 |
| 365 | 0.241 / 0.312 / 0.77 | 0.689 / 0.458 / 1.50 | 0.407 / 0.460 / 0.89 |
| mean 1097-1827 | 0.244 / 0.303 / 0.81 | 0.695 / 0.455 / 1.53 | 0.402 / 0.432 / 0.93 |

Reading: the real RSI is under-dispersed relative to ERDM by 12-19% at leads
5-20 (surface, diagnostics), not by the 30-37% the linear oracle predicts, and
at long lead its surface spread is 0.81x ERDM's while the upper-air spread is
1.5x *larger* (the collapsed upper-air state wanders more). Chaotic error
growth, absent from both toys, evidently compensates much of the fresh-slot
deficit; the prediction for Test 3 is therefore a modest spread gain (10-20%)
at kappa ~1.45, not a 40% one.

**Test 7, alpha vs S_c.** The 153-channel scale artifact
`norm_stats/sigma_c_sstpred153.pt` maps onto the model's `channel_layout: v2`
pack order (surface | diagnostics | upper air level-major from 1000 hPa,
[T, u, v, z, q] per level | ocean) and is physically sensible there: surface
pressure 0.017, temperature 0.04-0.06, geopotential 0.02-0.12, u 0.06-0.20,
v 0.14-0.29, q 0.05-0.17, precipitation 0.37, high cloud 0.33, mxtpr 0.40,
ocean at the 0.01 floor. **The real range is 0.017-0.40** (median 0.096); the
brief's and the toys' "S_c up to 0.9" does not occur. Regressing the brief's
per-channel shrinkage alpha (epoch 24, 81 channels) on log S_c:

    Pearson 0.72, Spearman 0.78; alpha = 1.00 + 0.086 log S_c
    -> alpha 0.65 at S 0.017, 0.80 at S 0.10, 0.92 at S 0.40
    within-family: surface 0.92, u 0.87, geopotential 0.46, q 0.44, T 0.33, v -0.18 (saturated ~0.9)

The channels that collapse most are the fast ones. This is the ordering Layer
A predicts (anchor-chain deficit grows with S_c) and the opposite of the
skip-leverage mechanism (largest lever at small S_c), so Test 7 sides with the
report's revised verdict without any GPU time.

Corrected artifact facts for the report: the per-channel scales are ordered
correctly and were produced by the logic of `make_rsi_delta_scales.py` for the
153-channel pack (the tool's shipped copy hard-codes the 154-channel fancy
pack; the 153 variant was an ad-hoc edit, file dated 2026-09-02 16:03:37).

## Phase 1: probes on the epoch-24 checkpoints (`polaris_rsi_drift_probes_phase1.pbs`)

Four processes on one node, ICs 1996-01-01/04-01/07-01/10-01, batch 2:

| GPU | model | probes | what it decides |
|---|---|---|---|
| 0 | RSI e24 (EMA) | readout, cascade 120 rolls (+ anchor trace) | Test 1: slot-6 slope vs the S_c-implied Bayes floor (~1 - 1.2 S_c^2); level gain and shrink response of the readout. Test 2: anchor amplitude per roll, its per-roll gain, emitted vs truth anomaly std; latency and per-channel onset ordering |
| 1 | RSI e24 | flush (36 rolls from the IC; 36 rolls from the free-run window at roll 30) | Test 4: fraction of a x0.7 pattern shrink / -0.3 offset retained above the model's own two-seed floor, on- vs off-manifold |
| 2 | ERDM e24 | readout, cascade | control for Tests 1-2 |
| 3 | ERDM e24 | flush | control for Test 4 |

Predictions written down before running:
- readout, RSI slot 6 at t = 0.5: `std_ratio`/slope near `1 - 1.2 S_c^2`
  (0.999 for surface pressure, ~0.95 at S 0.2, ~0.83 at S 0.37) if the head is
  at its Bayes floor on-manifold; a slope well below that for slow channels is
  Layer B on-manifold. Slot 1: ~1.00 for both models. `level_gain` ~1 for both
  (no restoring force within one roll); `shrink_ratio` ~0.7 (pass-through).
- cascade: RSI anchor amplitude decays from roll 1 and the emitted frame
  follows it by W-1 = 5 rolls; anchor per-roll gain ~1 for ~20 rolls then
  < 1; fast channels first. ERDM emitted amplitude ~1 throughout.
- flush: ERDM residual falls to its floor within ~W rolls; RSI keeps a large
  fraction; the free-run case retains more than the IC case.

## Phase 2: inference-only sweep, 300 days (`polaris_rsi_drift_sweep300_phase2.pbs`)

Five 8-member configs, one per 2-node group, scored like the 5-year protocol
(combined RSI -> x_DDC at 180x360, obs-climatology truth), horizon 300,
partial saves every 100 frames, per-rank anchor traces every 10 rolls:

| cfg | sampler | prediction (vs the first 300 steps of `eval_bias5yr_e24`) |
|---|---|---|
| k120 | `fresh_noise_scale 1.2` | spread +5-10%; RMSE-vs-climatology trace unchanged in shape |
| k145 | `fresh_noise_scale 1.45` | spread +10-20%, members unlock; if the onset at step 28-30 moves or the plateau falls, variance-to-mean rectification is real |
| k180 | `fresh_noise_scale 1.8` | over-injection: spread above ERDM's, short-range RMSE up |
| fd | `final_denoise true` | +5-9% amplitude, alpha at most ~15% lower; +33% cost |
| ns4 | `num_steps 4` | little change (oracle: +3-8 pp spread) |

## Phase 3: five-year run of the best variant (`polaris_rsi_drift_eval5yr_phase3.pbs`, `EVAL_CFG`)

Full protocol; scored with `$R/drift_shrinkage.py` against `eval_bias5yr_e24`.
Prediction under the report's verdict: alpha essentially unchanged for k145
(Layer A alone is not the drift); a large drop would mean rectification.

## Phase 4: training-side (conditional on Phase 1)

Only if Test 1 shows the head at its Bayes floor on-manifold and Test 2/4 show
the off-manifold loss of restoring force: fine-tune from epoch 24 with
`loss.anchor_shrink=0.3` (pattern-shrink anchors) paired with
`fresh_noise_scale 1.45` at inference; 2-epoch links on 10 nodes
(`polaris_rsi_prod24_b40.pbs` pattern), then a 1-year eval. If Test 1 shows a
slow-channel level bias on-manifold instead: retrain with per-channel EDM
coefficients (not yet implemented; a retrain, never an inference swap).

## Results log

**2026-09-09, Phase 1 job 7600866 (debug).** First submission (7600839) died at
config parsing (Hydra reads a comma list as a sweep; now passed as
`[a,b]`). Second submission: readouts done; the cascade crashed on a wrapper
method name (`pack_state` is not on the rolling wrapper; fixed to a one-frame
`pack_window_state`) and was resubmitted as job 7600870
(`polaris_rsi_drift_cascade_phase1b.pbs`); the flush probes were unaffected.

**Test 1 (teacher-forced readout at global t = 0.5, 4 ICs, mean over 153
channels), slot 1 .. 6:**

| model | std(y_hat)/std(y) per slot |
|---|---|
| RSI e24 (EMA) | 1.0000, 0.9999, 0.9996, 0.9989, 0.9969, **0.9958** |
| ERDM e24 | 1.0000, 1.0000, 0.9996, 0.9978, 0.9956, 0.9910 |

The RSI anchor readout (slot 6, where `c_skip = 0.631`) passes 99.6% of the
truth's anomaly amplitude through on-manifold, slightly *more* than ERDM's
denoiser at its back slot and above the oracle's Bayes-floor expectation for
the median channel (~0.99 at S_c ~0.1). There is no on-manifold amplitude
contraction in the quantity RSI copies forward. Consequences: the mis-scaled
skip is not producing a level bias on the training manifold (report Layer B,
"on-manifold" branch: rejected); whatever drives the time-mean collapse
switches on off-manifold (Tests 2 and 4 decide). Per-channel slot-6 values,
level gain and shrink response follow when job 7600870 writes `summary.json`.
