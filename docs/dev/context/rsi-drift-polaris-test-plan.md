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
| 4 fine-tunes | `prod` -> `small` | 10 | 3 h links (2 epochs each) | Test 1 showed no on-manifold bias, so the off-manifold remedy was launched |

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

## Phase 3: five-year runs (`polaris_rsi_drift_eval5yr_fanout_phase3.pbs`)

The 2-node `preemptable` submission (job 7600928) never scheduled ("Insufficient
amount of resource: queue_tags": the preemptable node pool was fully allocated),
so Phase 3 became a five-config fan-out on `prod` -> `small` (10 nodes, 2:59,
job 7601013): k120 / k145 / k180 / base (shipped sampler, with the anchor
trace, i.e. a 5-year baseline trace) / k145s1 (seed 1, to size the latent-noise
scatter of a 5-year alpha). `final_denoise` and `num_steps 4` are 1.3-2x slower
and would not finish 1827 frames in 3 h. Scored with `$R/drift_shrinkage.py`
against `eval_bias5yr_e24`. Prediction under the report's verdict: alpha
essentially unchanged for every kappa (Layer A alone is not the drift); a large
drop would mean rectification. The single-variant script
(`polaris_rsi_drift_eval5yr_phase3.pbs`, `EVAL_CFG`/`EVAL_CKPT_DIR`/`EVAL_EPOCH`/
`EVAL_HORIZON`/`EVAL_TAG`) is kept for the fine-tune's 1-year scoring.

## Phase 3 results (2026-09-09, job 7601013, five 5-year 8-member runs; killed at the wall 80 s after the last variant had written its results)

Shrinkage alpha of the 5-year time-mean bias map (`polaris_results.py alpha`),
surface spread at step 10 / mean over steps 1097-1827, and the surface
RMSE-vs-climatology plateau (mean over steps 100-1827):

| run | alpha surface mean (skt / sp / t2m / q2m / u10 / v10) | alpha upper-air T / u / v / z / q | alpha diag mean | spread 10 / late | plateau |
|---|---|---|---|---|---|
| shipped e24 (2026-09-07 file) | 0.718 (0.681 / 0.490 / 0.683 / 0.718 / 0.864 / 0.871) | 0.751 / 0.845 / 0.901 / 0.757 / 0.807 | 0.758 | 0.110 / 0.244 | 612.5 |
| base, re-run with the anchor trace | identical to 3 decimals | identical | 0.758 | 0.110 / 0.244 | 612.5 |
| k120 (`fresh_noise_scale` 1.2) | 0.719 (0.682 / 0.490 / 0.684 / 0.720 / 0.865 / 0.874) | 0.751 / 0.845 / 0.901 / 0.757 / 0.809 | 0.757 | 0.115 / 0.246 | 611.7 |
| k145 (1.45) | 0.717 (0.680 / 0.487 / 0.682 / 0.719 / 0.863 / 0.872) | 0.750 / 0.845 / 0.901 / 0.757 / 0.807 | 0.756 | 0.121 / 0.233 | 615.1 |
| k145, seed 1 | 0.718 (0.680 / 0.487 / 0.683 / 0.719 / 0.864 / 0.873) | 0.751 / 0.845 / 0.901 / 0.757 / 0.807 | 0.756 | 0.121 / 0.228 | 616.8 |
| k180 (1.8) | 0.713 (0.675 / 0.483 / 0.678 / 0.713 / 0.861 / 0.870) | 0.743 / 0.842 / 0.896 / 0.753 / 0.804 | 0.753 | 0.130 / 0.231 | 610.2 |
| ERDM e24 | ~0.000 | ~0.01 | -0.005 | 0.136 / 0.303 | 64.5 |

Five-year headline for k120 from its log: z500 bias-map RMSE 2061, t2m
10.51 K, t2m mean bias -3.71 K (shipped: 2059 / 10.42 / -3.47).

Readings. (i) The fresh-slot latent inflation changes the five-year
shrinkage alpha by at most 0.006 on any channel and the drift plateau by
< 1%: Layer A is fully eliminated as a cause of the time-mean collapse, with
the cleanest possible control (the shipped-sampler re-run reproduces the
2026-09-07 file to 3 decimals, and seed 1 agrees with seed 0 to 0.001).
(ii) What it does change is the short-lead dispersion, monotonically in
kappa (day-10 surface spread 0.110 -> 0.115 / 0.121 / 0.130, ERDM 0.136),
with the late spread unchanged or slightly lower. So the report's Layer A
fix is worth keeping for calibration at leads 5-20 and is irrelevant to the
drift. (iii) The five-year runs carry per-rank anchor traces every 50 rolls
(`trace_rank*.pt`) for any later look at the anchor chain over the full span.

## Phase 4 results (2026-09-09)

Fine-tune job 7600874 (`polaris_rsi_ft_shrink_phase4.pbs`, `loss.anchor_shrink 0.3`,
resume from epoch 24, Muon lr fast-forwarded to 4.6e-4 Muon-group / 4.6e-5
base): the first augmented batch has loss 2484, back to ~215 by batch 500
(the shipped run's epoch-24 level is 220). Epoch-25 10-day validation:
surface RMSE step 1 = 9.57, step 10 = 94.8, upper air step 10 = 108.9
(shipped epoch 20: 2.28 / 80.7 / 114.4) -- the augmentation costs short-range
surface skill, mostly at step 1, as the trained toy predicted.

**One-year evaluation of the epoch-25 weights** (job 7601120; 8 members,
IC 1996-01-01, obs-climatology truth; `eval_bias1yr_{base,k145}_e25`):

| | shipped e24 (5-yr / 300-d) | ft25, shipped sampler | ft25 + `fresh_noise_scale` 1.45 | ERDM e24 |
|---|---|---|---|---|
| z500 bias-map RMSE (m2/s2), 1 yr | 2059 (5 yr) / ~1850 (300 d) | **1029** | 1037 | 46 (5 yr) |
| t2m bias-map RMSE (K) | 10.4 / ~9.2 | **4.98** | 5.00 | 0.22 |
| t2m global-mean bias (K) | -3.5 / -3.8 | -2.47 | -2.47 | -0.01 |
| surface RMSE-vs-clim, steps 100-365 mean | 611 | **397** | 393 | 63 |
| surface RMSE-vs-clim at step 30 / 50 / 85 / 200 / 365 | 219 / 487 / 619 / 614 / 609 | 139 / 270 / 337 / 408 / 413 | 132 / 266 / 328 / 402 / 408 | 75 / 70 / 53 / 76 / 86 |
| surface spread step 10 / 100 / 365 | 0.110 / 0.255 / 0.241 | 0.127 / 0.236 / 0.209 | 0.134 / 0.255 / 0.212 | 0.136 / 0.310 / 0.312 |

Shrinkage alpha of the one-year time-mean bias map (`polaris_results.py alpha`,
which reproduces the brief's 5-year numbers on the epoch-24 file to 3 decimals:
skt 0.681, sp 0.490, t2m 0.683, q2m 0.718, u10 0.864, v10 0.871; ERDM 0.000):

| channel / group | shipped e24 (5 yr) | ft25 (1 yr) | ft25 + 1.45 (1 yr) |
|---|---|---|---|
| skin temperature | 0.681 (R2 0.95) | 0.214 (0.41) | 0.221 |
| surface pressure | 0.490 (0.87) | -0.081 (0.08) | -0.078 |
| 2m temperature | 0.683 (0.95) | 0.201 (0.43) | 0.211 |
| 2m specific humidity | 0.718 (0.97) | 0.345 (0.75) | 0.349 |
| 10m u / v | 0.864 / 0.871 | 0.399 / 0.605 | 0.406 / 0.599 |
| diagnostics, mean | 0.758 | 0.406 | 0.407 |
| upper air T / u / v / z / q, mean over levels | 0.751 / 0.845 / 0.901 / 0.757 / 0.807 | 0.434 / 0.483 / 0.679 / 0.309 / 0.639 | 0.463 / 0.487 / 0.678 / 0.310 / 0.618 |

Readings. (i) One epoch of pattern-shrink anchor augmentation removes ~40%
of the drift plateau's excess over ERDM ((611 - 397) / (611 - 63)) and halves
the one-year bias-map RMSE of z500 and t2m; the run-away is delayed and
slowed (step 50: 270 vs 487) but the fine-tuned trace keeps creeping upward
after day 100 (333 -> 413), so this is a partial fix that may continue to
drift over 5 years. (ii) The fresh-slot inflation on top adds nothing to the
mean and only raises the spread, exactly as on the epoch-24 weights. (iii)
The day-10 spread rose (0.110 -> 0.127) without any sampler change: a head
that trusts its anchor less disperses more. (iv) The alpha table shows the relief is largest for the slow channels
(surface pressure's collapse is gone; T, z, skt down to 0.2-0.4) and smallest
for the fast v-winds (0.6-0.7) whose anchor chains go off-manifold first, which
is where the wider shrink range should help. (v) This is the report's
training-side remedy confirmed in direction on the real model; the obvious
next steps are more epochs, a wider shrink range (the real collapse reaches
0.3-0.4 for the fast channels, while the augmentation only shows the network
0.7-1.0), and pairing with self-generated anchors.

**Epoch 26 (second fine-tune epoch), one year, seeds 0 and 1 (job 7601186):**

| | shipped e24 | ft25 | ft26 seed 0 | ft26 seed 1 | ERDM |
|---|---|---|---|---|---|
| z500 bias-map RMSE / mean bias | 2059 / -923 (5 yr) | 1029 / -668 | 1030 / -568 | 1026 / -564 | 46 / -5 |
| t2m bias-map RMSE / mean bias (K) | 10.4 / -3.5 | 4.98 / -2.47 | 4.81 / **-1.40** | 4.79 / -1.39 | 0.22 / -0.01 |
| surface RMSE-vs-clim, mean steps 100-365 | 611 | 397 | **350** | 350 | 63 |
| surface RMSE-vs-clim at 30 / 50 / 85 / 200 / 365 | 219 / 487 / 619 / 614 / 609 | 139 / 270 / 337 / 408 / 413 | 128 / 326 / 328 / 347 / 358 | 128 / 320 / 328 / 347 / 357 | 75 / 70 / 53 / 76 / 86 |
| alpha: skt / sp / t2m / q2m / u10 / v10 | 0.68 / 0.49 / 0.68 / 0.72 / 0.86 / 0.87 | 0.21 / -0.08 / 0.20 / 0.35 / 0.40 / 0.61 | 0.27 / -0.02 / 0.26 / 0.36 / 0.42 / 0.65 | 0.27 / -0.02 / 0.26 / 0.36 / 0.42 / 0.64 | ~0 |
| alpha: upper-air T / u / v / z / q (mean) | 0.75 / 0.85 / 0.90 / 0.76 / 0.81 | 0.43 / 0.48 / 0.68 / 0.31 / 0.64 | 0.48 / 0.51 / 0.69 / 0.34 / 0.49 | 0.48 / 0.51 / 0.69 / 0.34 / 0.49 | ~0 |
| 10-day validation, surface RMSE step 1 / 10 | 2.3 / 80.7 (e20) | 9.6 / 94.8 | 10.7 / 89.0 | | |

Readings. (i) Two seeds agree to +-0.005 in alpha and +-4 in z500 RMSE: the
latent-noise scatter of these one-year statistics is negligible, so
epoch-to-epoch differences are real. (ii) The second epoch removes more of
the level drift (t2m mean bias -2.47 -> -1.40 K; plateau 397 -> 350, i.e. 48%
of the excess over ERDM gone) and flattens the late trace (347 -> 358 over
days 200-365 vs 408 -> 413 for epoch 25), but the pattern-shrinkage alpha
does not improve further (T 0.43 -> 0.48, z 0.31 -> 0.34, v 0.68 -> 0.69,
q 0.64 -> 0.49). The augmentation as configured (shrink 0.7-1.0) appears to
have bought what it can for the pattern; the wider-shrink variant (0.6, job
7601155) and more epochs (link 2, job 7601154) test whether the remaining
alpha is reachable this way. (iii) The short-range cost grows slightly with
epochs (step-1 surface RMSE 9.6 -> 10.7).

**Epoch 28 (four shrink-0.3 epochs, job 7601154 -> eval 7601399) and the
shrink-0.6 variant at epoch 26 (job 7601155 -> eval 7601400), one year, 8
members** (`polaris_results.py compare`; all fine-tunes resume from the shipped
epoch-24 pair; "ep" counts fine-tune epochs after 24):

| | shipped e24 (5 yr) | ft25 (0.3) | ft26 (0.3) | ft28 (0.3) | ft26 (0.6) | ERDM e24 (5 yr) |
|---|---|---|---|---|---|---|
| z500 bias-map RMSE / mean bias (m2/s2) | 2059 / -973 | 1029 / -668 | **1030 / -568** | 1105 / **-478** | 1134 / -579 | 42 / -4 |
| t2m bias-map RMSE / mean bias (K) | 10.47 / -3.65 | 4.98 / -2.47 | **4.81** / -1.40 | 5.39 / **-0.99** | 5.66 / -2.04 | 0.21 / -0.01 |
| t850 bias-map RMSE / mean bias (K) | 8.88 / -3.36 | 5.01 / -3.05 | 4.90 / -2.14 | 5.20 / -1.73 | 5.50 / -2.50 | 0.18 / -0.03 |
| u250 bias-map RMSE / mean bias (m/s) | 10.47 / -0.78 | 7.67 / -2.15 | 8.06 / -2.23 | 8.12 / -1.98 | 8.89 / -3.94 | 0.70 / 0.21 |
| surface RMSE-vs-clim, mean steps 100-365 | 611 | 397 | 350 | **336** | 345 | 63 |
| upper-air RMSE-vs-clim, mean steps 100-365 | 519 | 478 | 395 | **370** | 389 | 453 |
| surface RMSE-vs-clim at 30 / 50 / 85 / 200 / 365 | 219 / 487 / 619 / 614 / 609 | 139 / 270 / 337 / 408 / 413 | 128 / 326 / 328 / 347 / 358 | 136 / 373 / 344 / 333 / 339 | 242 / 384 / 346 / 347 / 347 | 75 / 70 / 53 / 76 / 86 |
| surface spread at 10 / 100 / 365 | 0.110 / 0.255 / 0.241 | 0.127 / 0.236 / 0.209 | 0.126 / 0.249 / 0.208 | 0.121 / 0.240 / 0.210 | 0.115 / 0.227 / 0.209 | 0.136 / 0.310 / 0.312 |
| alpha skt / sp / t2m / q2m / u10 / v10 | 0.68 / 0.49 / 0.68 / 0.72 / 0.86 / 0.87 | **0.21 / -0.08 / 0.20 / 0.34 / 0.40 / 0.60** | 0.27 / -0.02 / 0.26 / 0.36 / 0.42 / 0.65 | 0.32 / 0.05 / 0.32 / 0.39 / 0.49 / 0.69 | 0.32 / 0.06 / 0.31 / 0.44 / 0.51 / 0.72 | ~0 |
| alpha upper-air T / u / v / z / q (mean over levels) | 0.75 / 0.84 / 0.90 / 0.76 / 0.81 | **0.43 / 0.48 / 0.68 / 0.31** / 0.64 | 0.48 / 0.51 / 0.69 / 0.34 / 0.49 | 0.48 / 0.55 / 0.70 / 0.41 / 0.52 | 0.47 / 0.55 / 0.75 / 0.42 / 0.31 | ~0 |
| alpha diagnostics, mean | 0.76 | 0.41 | 0.44 | 0.48 | 0.49 | ~0 |
| 10-day validation, surface RMSE step 1 / step 10 | 2.3 / 80.7 (e20) | 9.6 / 94.8 | 10.7 / 89.0 | 5.9 / 86.9 (ep27), 8.1 / 87.1 (ep28) | 11.7 / 148.7 (ep25), 10.5 / 138.8 (ep26) | |

The 300-frame surface trace (steps 25..365 by 25): ft28 `115 373 349 335 333
333 334 333 334 337 337 337 336 337`; ft26 (0.6) `150 384 358 344 345 347 349
347 344 343 343 343 344 348`; ft26 (0.3) `111 326 344 319 346 348 350 347 349
347 352 355 356 358`; shipped `173 487 604 620 621 626 627 614 608 614 605 607
606 617`; ERDM `80 70 59 45 59 55 64 76 69 62 61 55 60 73`.

**Anchor traces of the fine-tuned runs** (`RSI_TRACE_PATH`, member 0 of each
one-year run and of the shipped five-year re-run; `polaris_results.py trace`;
emitted-frame spatial-anomaly std relative to roll 1, mean over rolls
100-365):

| channel set | shipped e24 | ft25 (0.3) | ft26 (0.3) | ft28 (0.3) | ft26 (0.6) |
|---|---|---|---|---|---|
| surface + diagnostics (21) | 0.43 | 0.68 | 0.63 | 0.61 | 0.66 |
| tropospheric T (125-1000 hPa) | 0.37 | 0.60 | 0.51 | 0.49 | 0.55 |
| tropospheric z | 0.32 | 0.62 | 0.55 | 0.50 | 0.53 |
| tropospheric q | 0.43 | 0.64 | 0.65 | 0.65 | 0.69 |
| tropospheric u | 0.27 | 0.42 | 0.39 | 0.38 | 0.39 |
| tropospheric v | 0.28 | 0.34 | 0.34 | 0.33 | 0.35 |
| stratospheric T (5-100 hPa) | 3.40 | 0.63 | 0.46 | 0.40 | 0.42 |
| stratospheric v | 1.44 | 0.54 | 0.47 | 0.48 | 0.48 |
| stratospheric q | 1.81 | 13.2 | 13.8 | 14.4 | 16.8 |
| first roll with tropospheric v below 0.7 | 34 | 44 | 46 | 38 | 37 |
| anchor / emitted, same roll, rolls 100-365 | 0.95 | 0.88 | 0.89 | 0.87 | 0.88 |

Per channel (emitted, rolls 28 / 50 / 100 / 200), shipped vs ft28: 2m
temperature 0.95 / 0.75 / 0.40 / 0.37 vs 1.13 / 1.04 / 0.71 / 0.67; z@500 0.83 /
0.59 / 0.37 / 0.31 vs 1.00 / 1.22 / 0.55 / 0.48; surface pressure 0.96 / 0.93 /
0.71 / 0.71 vs 1.05 / 1.12 / 1.11 / 1.13; v@250 0.75 / 0.27 / 0.25 / 0.22 vs
0.90 / 0.42 / 0.27 / 0.24; precipitation 1.06 / 0.28 / 0.25 / 0.30 vs 0.95 /
0.40 / 0.29 / 0.36. The channel-mean |level change| of the emitted frame is
0.06 (normalized units) at rolls 28-35 for ft28 against 0.16-0.19 for the
shipped run, and the two meet at 0.20-0.22 after roll 100.

Readings. (i) Four epochs of shrink 0.3 leave a plateau that is exactly flat
(333-337 from day 100 to day 365) at 336, i.e. 50% of the excess over ERDM
removed; the wider shrink range (0.4-1.0) at the same epoch count gives 345
against 350, no gain, at a large short-range cost (surface RMSE-vs-clim 242
vs 128 at day 30; 10-day validation step 10 of 139 vs 87-89). (ii) The level
and the pattern respond differently to more epochs. The global-mean bias
improves monotonically (t2m -3.65 -> -2.47 -> -1.40 -> -0.99 K; z500 -973 ->
-668 -> -568 -> -478), but the pattern-shrinkage alpha bottomed after the
first epoch and has crept back since (t2m 0.20 -> 0.26 -> 0.32, z 0.31 ->
0.34 -> 0.41, u 0.48 -> 0.55; v flat at 0.68-0.70), so the bias-map RMSE,
which combines both, is minimal at epoch 26 (z500 1030, t2m 4.81 K) and rises
again at epoch 28 (1105, 5.39 K). The shrink-0.6 variant does not lower alpha
either (t2m 0.31, z 0.42, v 0.75). The augmentation as designed has bought
what it can for the pattern; the remaining alpha (~0.3 for the slow channels,
~0.7 for the fast winds) is not reachable by more of it. (iii) The traces
show why. The fine-tuned readout is now an amplifier (anchor/emitted 0.87-0.89
against 0.95): the emitted amplitude holds at or above 1 to roll ~50 instead
of ~28, and the late amplitude of the surface and slow upper-air channels is
0.5-0.7 instead of 0.3-0.4 (surface pressure is now over-amplified at
1.1-1.2). But the fast tropospheric winds still collapse to 0.33-0.39 (shipped
0.27-0.28) and their onset moves only from roll 34 to 37-46: the fast-channel
collapse is essentially untouched, which is the residual v alpha of ~0.7. The
augmentation teaches the head to undo a uniform pattern shrink; the fast
channels leave the manifold by decorrelating and smoothing, not by a uniform
shrink, so the augmentation does not cover them. That is the case for the
report's second training-side remedy, self-generated (rolled-out) anchors,
which show the network the actual off-manifold structure. (iv) Side effect:
stratospheric specific humidity (50-125 hPa, negligible physical variance) is
over-amplified 5-70x by every fine-tuned head (shipped 2-6x), while the
shipped run's stratospheric T and v blow-ups (3-7x) are cured. This is
irrelevant to the tropospheric climate but shows that an un-shrinking readout
amplifies wherever it cannot judge the amplitude; exclude channels with
negligible increment scale from the augmentation (or cap the learned gain)
before using it in production. The shrink-0.6 upper-air q alpha (0.31) is
contaminated by this. (v) Best pattern-shrink checkpoint: epoch 25-26 at shrink
0.3. The next lever is not more shrink epochs but self-generated anchors
paired with the Layer A injection, plus a level-restoring term, as ranked in
the report's section 5.

**Status (2026-09-09 14:10 UTC): every run in this plan has been executed and
logged** (Phase 0: Tests 6, 7; Phase 1: Tests 1, 2, 4; Phase 2: the 300-day
sampler sweep; Phase 3: the five-year fan-out with a second seed; Phase 4:
shrink-0.3 fine-tune epochs 25-28 and the shrink-0.6 variant to epoch 26,
each scored over one year). Fetched artifacts live under the session
scratchpad only; the `eval_suite.pt`, `trace_rank*.pt` and checkpoint files
remain on Polaris under `$R/eval_bias*` and `$R/checkpoints_ft_shrink*_b40`.

## Phase 5: new base (bundle run) and self-generated anchors (2026-09-09)

**Base change.** The batch-40 *bundle* production run
(`checkpoints_prod24_bundle_b40`, `polaris_rsi_prod24_bundle_b40.pbs`: lr 5e-4
linear scaling with 1-epoch warmup and cosine to 5e-5 over 24 epochs, Muon
momentum 0.85, partial activation checkpointing; chain jobs 7599767-7599776,
2 epochs per 3 h link) had reached epoch 19 (epoch 20 in progress) when it was
checked, against the previous version `checkpoints_prod24_b40` (sqrt-scaled lr
1.58e-4, StepLR 0.95) whose epoch-24 weights every earlier phase used.
Per-epoch mean training loss (all 1315 batches):

| epoch | 12 | 14 | 16 | 17 | 18 | 19 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| previous (prod24_b40) | 248.0 | 243.6 | 233.6 | 231.5 | 229.6 | 228.2 | 226.5 | 220.4 |
| bundle | 220.1 | 213.4 | 207.6 | 204.7 | 202.7 | 200.9 | | |

The bundle run is 11-12% lower at equal epochs and at epoch 19 already below
the previous run's final epoch. Ten-day validation (4 ICs x 10 members, every
5 epochs; surface RMSE at steps 1 / 3 / 6 / 10 and ACC at step 10): bundle
e15 2.10 / 3.28 / 18.7 / **188** / 0.67 vs previous e15 2.45 / 3.74 / 21.9 /
86 / 0.83 and e20 2.28 / 3.58 / 19.5 / 81 / 0.85. The bundle model is better
through day 6 and about **2x worse at day 10** on every group (upper air 206 vs
119, diagnostics 20.3 vs 12.6), consistently at epochs 5, 10 and 15, i.e. a
faster error growth between days 6 and 10 that the one-year baseline below has
to place relative to the drift. The training-loss criterion is met, so the
bundle checkpoint is the base for everything in this phase; epoch 20 was
chosen (first epoch >= 16 that also carries a validation line, so its 10-day
skill is directly comparable to the previous run's epoch 20). That line
arrived at 15:50 UTC: bundle e20 surface RMSE 1.95 / 3.13 / 15.1 / **116**
at steps 1 / 3 / 6 / 10 with step-10 ACC 0.82 (previous e20: 2.28 / 3.58 /
19.5 / 81, ACC 0.85; upper air step 10: 137 vs 114; diagnostics 14.0 vs
11.9). The day-10 deficit has narrowed from 2.2x at epoch 15 to 1.4x at
epoch 20 as the cosine schedule anneals, while the short-range advantage
grew; the one-year baseline (B20) settles what remains of it.

**Design.** All fine-tunes resume from the bundle epoch-20 pair with the
bundle recipe's own optimizer and (continued cosine) schedule, so a fine-tune
from epoch 20 to 22 is "the same run with the augmentation switched on" and
the chain's own epoch 22 is the control (`polaris_rsi_ft_phase5.pbs`).
Variants, each scored over one year (8 members, IC 1996-01-01,
obs-climatology truth, per-rank anchor trace; `polaris_rsi_drift_eval_multi_phase5.pbs`):

| id | checkpoint dir | loss knobs | sampler at eval | purpose |
|---|---|---|---|---|
| B20 | bundle e20 | (base) | base, k145 | new baseline; is the day-10 deficit a drift-onset difference? |
| C22 | bundle e22 (chain) | (none) | base | control for the fine-tunes |
| S22 | `checkpoints_ft5_shrink_b40` e22 | `anchor_shrink 0.3` | base | Phase 4 remedy re-done on the new base |
| P22 | `checkpoints_ft5_pf_b40` e22 | `pushforward_rolls 2`, `fresh_noise_scale 1.45` | k145 (and base) | self-generated anchors paired with the Layer A injection |
| PS22 | `checkpoints_ft5_pfshrink_b40` e22 | pushforward 2 + fresh 1.45 + shrink 0.3 with the 9 stratospheric-q channels (<= 125 hPa) excluded | k145 | both remedies; tests the exclusion fix for the q blow-up |

**Self-generated anchors (`pushforward_rolls = K`, rsi.py).** The loader
reads K extra history frames ahead of the anchor (`RSIScheduler.history_frames`
-> `SequenceDataset(unroll_steps = W-1+K)`); per step the scheduler draws
k ~ U{0..K} from a private generator (identical on every DDP rank, so no
stragglers), initializes the window from truth k frames back
(`warmup_window`), rolls its own sampler k times without grad
(`sample_window` + `_fresh_slot`, forcings and own-time ocean boundary sliced
from the W+K-slot stack, `fresh_noise_scale` applied exactly as at
inference), and anchors the last k slots of the loss window on the sampler's
slot-W readouts `y_hat[:, -1]` instead of on truth; the first W-k anchors and
all targets stay truth, the anchor-time ocean block is kept from truth (it is
imposed from truth at every roll top at inference). K = 0 is bit-identical to
the shipped loss (109 scheduler/recipe tests pass, including a k = 0 equality
test and an end-to-end train step with K = 2). This is the toy's
`rsi_ft_selfanchor` (one preceding roll, slot W only) generalized to a chain
of up to K links. Cost: one no-grad roll is ~3 head evaluations; the
one-node smoke run (job 7601494, epoch-19 pair, K = 2 + shrink 0.3 +
exclusion, 4 ranks) measured 5.5 s/step against the chain's 3.3 s, i.e.
1.67x, so about 125 min per epoch on 10 nodes: one epoch per 3 h link. Its
first batches behave like the Phase 4 fine-tune's (loss 4446 -> ~500 by batch
40 with the ocean term at ~2, gnorm 5e4 -> 1e3; Phase 4: 2484 -> 399,
ocean 2.4). Gotcha found on the way: the forkserver DataLoader binds an
AF_UNIX socket under TMPDIR, and the debug node's PBS TMPDIR
(`/var/tmp/pbs.<jobid>.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/...`) pushed
that path past the 108-byte limit; the Phase 5 fine-tune script pins
`TMPDIR=/tmp`.

**Submitted 2026-09-09 15:50 UTC** (all resume from the bundle epoch-20
pair copied into `checkpoints_ft5_{shrink,pf,pfshrink}_b40`): S22 job
7601531 (one 2-epoch link); P22 jobs 7601532 -> 7601533 (one epoch per link,
`depend=afterany`); PS22 jobs 7601534 -> 7601535; B20 baseline eval job
7601536 (debug-scaling, `b20_base` + `b20_k145`). The chain's own epoch 22
(control C22) is due from link 7599775. Evaluations of C22 / S22 / P22 /
PS22 follow with `polaris_rsi_drift_eval_multi_phase5.pbs` on 10 nodes once
the checkpoints exist.

*17:50 UTC:* the first links of P22 (7601532) and PS22 (7601534) both died
before their first batch with `CUDA error: an illegal instruction was
encountered` -- rank 18 of one and rank 2 of the other, i.e. the same
device, **x3003c0s25b0n0 GPU 2** (recorded in `$R/bad_nodes_20260909.txt`;
worth reporting to ALCF). S22 on x3014 nodes was unaffected. The fine-tune
script now runs a GPU preflight (one matmul per rank) and resubmits itself on
failure; the chains were extended so each variant still reaches epoch 22:
P22 7601533 -> 7601656 -> 7601657, PS22 7601535 -> 7601658 -> 7601659 (every
link resumes from whatever epoch exists and exits early at the target).

### Phase 5 results

**B20, the bundle epoch-20 baseline (job 7601536, one year, 8 members,
shipped sampler and `fresh_noise_scale` 1.45):**

| | shipped e24 (old base, 5 yr) | **bundle e20, base** | bundle e20, k145 | ft26 (0.3, old base) | ERDM e24 |
|---|---|---|---|---|---|
| z500 bias-map RMSE / mean bias | 2059 / -973 | 2098 / -1127 | 2070 / -1134 | 1030 / -568 | 42 / -4 |
| t2m bias-map RMSE / mean bias (K) | 10.47 / -3.65 | 11.59 / -5.47 | 11.36 / -5.47 | 4.81 / -1.40 | 0.21 / -0.01 |
| t850 bias-map RMSE / mean bias (K) | 8.88 / -3.36 | 9.19 / -3.66 | 9.03 / -3.75 | 4.90 / -2.14 | 0.18 / -0.03 |
| surface RMSE-vs-clim, mean steps 100-365 | 611 | **794** | 792 | 350 | 63 |
| upper-air RMSE-vs-clim, mean steps 100-365 | 519 | 552 | 555 | 395 | 453 |
| surface RMSE-vs-clim at 30 / 50 / 85 / 200 / 365 | 219 / 487 / 619 / 614 / 609 | 214 / 611 / 776 / 778 / 796 | 255 / 454 / 713 / 789 / 816 | 128 / 326 / 328 / 347 / 358 | 75 / 70 / 53 / 76 / 86 |
| surface spread at 10 / 100 / 365 | 0.110 / 0.255 / 0.241 | 0.106 / 0.254 / 0.244 | 0.115 / 0.265 / 0.248 | 0.126 / 0.249 / 0.208 | 0.136 / 0.310 / 0.312 |
| alpha skt / sp / t2m / q2m / u10 / v10 | 0.68 / 0.49 / 0.68 / 0.72 / 0.86 / 0.87 | 0.71 / 0.60 / 0.72 / 0.75 / 0.83 / 0.85 | 0.69 / 0.58 / 0.70 / 0.74 / 0.83 / 0.85 | 0.27 / -0.02 / 0.26 / 0.36 / 0.42 / 0.65 | ~0 |
| alpha upper-air T / u / v / z / q | 0.75 / 0.84 / 0.90 / 0.76 / 0.81 | 0.79 / 0.84 / 0.86 / 0.76 / 0.83 | 0.78 / 0.83 / 0.85 / 0.75 / 0.82 | 0.48 / 0.51 / 0.69 / 0.34 / 0.49 | ~0 |
| anchor trace: first roll with tropospheric v below 0.7 | 34 | 32 | 32 | 46 | |
| anchor trace: late amplitude, surface+diag / trop. T / trop. z / sp | 0.43 / 0.37 / 0.32 / 0.71 | 0.36 / 0.37 / 0.35 / 0.46 | 0.38 / 0.33 / 0.30 / 0.48 | 0.63 / 0.51 / 0.55 / 1.10 | |

Reading. The bundle model, with 11% lower training loss and better 1-6 day
skill, drifts **harder** than the old base: the plateau is 794 against 611,
the one-year t2m level bias -5.5 K against -3.65 K, the onset two rolls
earlier (roll 32) and surface pressure now collapses to 0.46 of its pattern
amplitude (old base 0.71) -- the pattern alpha is the same 0.7-0.85 on every
channel. The fresh-slot inflation again moves the mean by nothing (794 vs
792) while adding day-10 spread. This is the strongest evidence so far that
the drift is a property of the formulation and not of a particular training
run's optimizer or schedule: a better-fit head trusts its anchor at least as
much, and the day-10 validation deficit of the bundle (116 vs 81) was the
first 10 rolls of this earlier onset. It also sets the bar for the Phase 5
fine-tunes: the Phase 4 remedy has to be re-measured on this base (S22), and
the control is the chain's own epoch 22 (C22), not the old base.

**C22, the bundle epoch-22 control (job 7601774):** two more plain epochs
change nothing -- plateau 792 (B20 794), z500 2094 / -1140, t2m 11.50 /
-5.41 K, alpha skt / sp / t2m 0.71 / 0.59 / 0.71, upper-air T / z 0.78 / 0.75,
onset roll 32, late tropospheric amplitude 0.35-0.38. The fine-tunes below
are measured against 792.

**S22, pattern-shrink 0.3 for two epochs from the bundle epoch 20 (job
7601531 -> eval 7601830):**

| | C22 control | **S22** | ft26 (0.3, old base, for reference) | ERDM |
|---|---|---|---|---|
| z500 bias-map RMSE / mean bias | 2094 / -1140 | 1805 / **-1483** | 1030 / -568 | 42 / -4 |
| t2m bias-map RMSE / mean bias (K) | 11.50 / -5.41 | 7.99 / **-6.06** | 4.81 / -1.40 | 0.21 / -0.01 |
| t850 bias-map RMSE / mean bias (K) | 9.12 / -3.67 | 7.21 / -5.18 | 4.90 / -2.14 | 0.18 / -0.03 |
| u250 bias-map RMSE / mean bias | 10.45 / -2.74 | 9.35 / +2.54 | 8.06 / -2.23 | 0.70 / 0.21 |
| surface RMSE-vs-clim, mean steps 100-365 | 792 | **455** | 350 | 63 |
| upper-air RMSE-vs-clim, mean steps 100-365 | 550 | **693** | 395 | 453 |
| surface RMSE-vs-clim at 30 / 50 / 85 / 200 / 365 | 201 / 549 / 764 / 804 / 808 | 146 / 166 / 349 / 453 / 470 | 128 / 326 / 328 / 347 / 358 | 75 / 70 / 53 / 76 / 86 |
| surface spread at 10 / 100 / 365 | 0.101 / 0.268 / 0.256 | 0.099 / 0.301 / 0.255 | 0.126 / 0.249 / 0.208 | 0.136 / 0.310 / 0.312 |
| alpha skt / sp / t2m / q2m / u10 / v10 | 0.71 / 0.59 / 0.71 / 0.75 / 0.83 / 0.85 | 0.31 / 0.23 / 0.32 / 0.47 / 0.62 / 0.71 | 0.27 / -0.02 / 0.26 / 0.36 / 0.42 / 0.65 | ~0 |
| alpha upper-air T / u / v / z / q | 0.78 / 0.84 / 0.85 / 0.75 / 0.83 | 0.63 / 0.68 / 0.68 / 0.49 / 0.41 | 0.48 / 0.51 / 0.69 / 0.34 / 0.49 | ~0 |
| anchor trace: onset (trop. v below 0.7) / late amplitude sfc+diag, trop. T, trop. z, sp | 32 / 0.38, 0.36, 0.35, 0.47 | **62** / 0.53, 0.50, 0.53, 0.92 | 46 / 0.63, 0.51, 0.55, 1.10 | |
| 10-day validation, surface RMSE step 1 / 10 | 1.95 / 116 (e20) | 11.1 / 171 (ep21), 9.2 / 161 (ep22) | 10.7 / 89 | |

Reading. (i) The pattern remedy transfers in direction and size: the plateau
excess over ERDM drops by 46% ((792 - 455) / (792 - 63); Phase 4 at two
epochs: 48%), the onset moves from roll 32 to 62, the late pattern amplitude
of the slow channels rises from 0.35 to 0.5 (surface pressure 0.47 -> 0.92)
and the surface alpha falls to 0.2-0.3 (winds 0.6-0.7 as before). (ii) The
**level** goes the other way on this base: the one-year t2m mean bias is
-6.06 K against the control's -5.41 K, z500 -1483 against -1140, T850 -5.18
against -3.67, and the upper-air RMSE-vs-climatology plateau is *worse* (693
vs 550) because it is dominated by the falling geopotential level. On the
old base the same augmentation had improved the level with every epoch
(t2m -3.65 -> -1.40 K), so the level response of the pattern shrink is not a
robust effect of the augmentation -- it acts on the pattern only (by
construction it preserves the spatial mean) and the level drift, which the
bundle base has more of, is a separate failure that needs its own term. The
report's Test 1 result (RSI passes 77% of a uniform window offset where ERDM
re-derives the level) is the mechanism. (iii) The short-range cost is the
same as before (step-1 surface RMSE 9-11 vs 2). (iv) The stratospheric
humidity blow-up (q@50 up to 40x) is present again; PS22 carries the
exclusion.

Consequence: a level-offset augmentation is added as a fourth variant, L22
(`anchor_level_noise`, a spatially uniform per-channel offset of the anchor
of s ~ N(0, 1) increment-std units, on top of shrink 0.3, so the difference
to S22 is the level term alone). Submitted 20:50 UTC as jobs 7601898 ->
7601899 (`checkpoints_ft5_shrinklvl_b40`, two epochs per link). An
epoch-21 three-way comparison (S21 / P21 / PS21, one augmented epoch each)
is scored as soon as the pushforward chains write their first epoch, so the
pushforward effect is visible before their second links get through the
queue.

**Ten-day validation after one augmented epoch (epoch 21; 4 ICs x 10
members; surface RMSE at steps 1 / 3 / 6 / 10, upper-air RMSE at step 10,
surface ACC at step 10):**

| model | 1 | 3 | 6 | 10 | UA 10 | ACC 10 |
|---|---|---|---|---|---|---|
| bundle e20 (base) | 1.95 | 3.13 | 15.1 | 116 | 137 | 0.82 |
| previous run e20 | 2.28 | 3.58 | 19.5 | 81 | 114 | 0.85 |
| S21 shrink 0.3 | 11.1 | 17.0 | 35.2 | 171 | | 0.82 |
| **P21 pushforward 2 + fresh 1.45** | **1.99** | **3.17** | **13.4** | **58** | **67** | **0.90** |
| PS21 pushforward + shrink (q excluded) | 12.0 | 18.5 | 34.3 | 105 | 81 | 0.87 |

One epoch of self-generated anchors leaves the day-1 skill untouched and
halves the day-10 error of the base (58 vs 116; the previous production
model: 81), the best 10-day RSI score of this project; the shrink costs a
factor 5 at day 1 whether or not the pushforward is on, and the pushforward
recovers most of the shrink's day-10 damage (171 -> 105). The one-year
consequences are the E21 evaluation (job 7601945).

**E21: one year after ONE augmented epoch (job 7601945; S21 with the shipped
sampler, P21 and PS21 with `fresh_noise_scale` 1.45 as trained):**

| | C22 control | S21 shrink | **P21 pushforward** | PS21 both | S22 shrink (2 ep) | ERDM e24 |
|---|---|---|---|---|---|---|
| z500 bias-map RMSE / mean bias | 2094 / -1140 | 1888 / -1590 | **542 / -230** | 999 / -951 | 1805 / -1483 | 42 / -4 |
| t2m bias-map RMSE / mean bias (K) | 11.50 / -5.41 | 8.15 / -6.39 | **1.59 / -1.19** | 4.18 / -3.51 | 7.99 / -6.06 | 0.21 / -0.01 |
| t850 bias-map RMSE / mean bias (K) | 9.12 / -3.67 | 7.55 / -5.73 | **1.52 / -1.02** | 4.23 / -3.81 | 7.21 / -5.18 | 0.18 / -0.03 |
| u250 bias-map RMSE / mean bias | 10.45 / -2.74 | 9.73 / +2.85 | 9.53 / +3.35 | 7.87 / +4.50 | 9.35 / +2.54 | 0.70 / 0.21 |
| v10m bias-map RMSE | 1.58 | 1.41 | **0.86** | 0.97 | 1.40 | 0.14 |
| surface RMSE-vs-clim, mean steps 100-365 | 792 | 416 | **152** | 290 | 455 | 63 |
| upper-air RMSE-vs-clim, mean steps 100-365 | 550 | 735 | **356** | 549 | 693 | 453 |
| surface RMSE-vs-clim at 30 / 50 / 85 / 200 / 365 | 201 / 549 / 764 / 804 / 808 | 155 / 261 / 341 / 412 / 415 | 113 / 128 / 113 / 184 / 164 | 145 / 155 / 202 / 276 / 322 | 146 / 166 / 349 / 453 / 470 | 75 / 70 / 53 / 76 / 86 |
| surface spread at 10 / 100 / 365 | 0.101 / 0.268 / 0.256 | 0.095 / 0.299 / 0.258 | **0.108 / 0.307 / 0.315** | 0.111 / 0.324 / 0.358 | 0.099 / 0.301 / 0.255 | 0.136 / 0.310 / 0.312 |
| alpha skt / sp / t2m / q2m / u10 / v10 | 0.71 / 0.59 / 0.71 / 0.75 / 0.83 / 0.85 | 0.28 / 0.18 / 0.29 / 0.48 / 0.62 / 0.72 | **-0.02 / 0.03 / -0.03 / 0.01 / 0.39 / 0.29** | -0.06 / -0.03 / -0.06 / 0.14 / 0.35 / 0.43 | 0.31 / 0.23 / 0.32 / 0.47 / 0.62 / 0.71 | ~0 |
| alpha upper-air T / u / v / z / q | 0.78 / 0.84 / 0.85 / 0.75 / 0.83 | 0.66 / 0.69 / 0.66 / 0.48 / 0.47 | **0.32 / 0.50 / 0.22 / 0.25 / 0.30** | 0.52 / 0.47 / 0.49 / 0.17 / 0.07 | 0.63 / 0.68 / 0.68 / 0.49 / 0.41 | ~0 |
| alpha diagnostics, mean | 0.75 | 0.49 | **0.13** | 0.25 | 0.49 | ~0 |
| anchor trace: first roll with trop. v below 0.7 | 32 | 59 | **335** | 90 | 62 | |
| late (rolls 100-365) amplitude: sfc+diag / trop. T / z / u / v / q | 0.38 / 0.36 / 0.35 / 0.38 / 0.35 / 0.41 | 0.57 / 0.59 / 0.57 / 0.37 / 0.41 / 0.60 | **0.91 / 1.07 / 0.85 / 0.83 / 0.83 / 1.14** | 0.76 / 0.91 / 1.04 / 0.62 / 0.49 / 0.78 | 0.53 / 0.50 / 0.53 / 0.37 / 0.39 / 0.61 | |
| stratospheric T / q late amplitude | 2.56 / 1.93 | 3.53 / 8.11 | **0.60 / 3.23** | 2.73 / 5.98 | 3.60 / 9.03 | |

The 300-frame surface trace (steps 25..365 by 25): P21 `110 128 114 116 118
145 187 184 168 153 126 128 155 150`; PS21 `138 155 188 215 239 269 283 276
293 314 306 311 319 322`; S21 `169 261 320 366 438 441 438 412 412 428 392 393
423 414`; control `204 549 756 763 764 791 777 804 799 813 793 794 805 795`;
ERDM `80 70 59 45 59 55 64 76 69 62 61 55 60 73`.

Readings. (i) **One epoch of self-generated anchors removes 88% of the
drift excess over ERDM** ((792 - 152) / (792 - 63)) and takes the one-year
bias-map RMSE of t2m from 11.5 K to 1.6 K and of z500 from 2094 to 542; the
shrinkage alpha of every slow channel is zero to within 0.03 (surface
pressure 0.03, 2m temperature -0.03, skin temperature -0.02), the diagnostics
are at 0.13 and the winds at 0.2-0.5 -- the pattern collapse is gone. The
anchor chain holds the fast tropospheric winds above 0.7 of their amplitude
for 335 rolls instead of 32, the late tropospheric amplitudes sit at
0.83-1.07 instead of 0.35-0.4, and the long-lead surface spread (0.315)
matches ERDM's (0.312), i.e. the Layer A dispersion deficit is closed at the
same time by the paired injection. The upper-air RMSE-vs-climatology (356)
is below ERDM's own (453), and the day-10 skill improved (58 vs 116). This
is the report's remedy (b) confirmed on the real model at full strength;
it does what the pattern shrink could not because the head is trained on the
actual off-manifold structure of its own anchor chain (smoothed AND
decorrelated fields with their own level errors), not on a uniform pattern
shrink. (ii) The shrink is now harmful: on top of the pushforward it costs a
factor two in plateau (290 vs 152) and re-introduces level bias (t2m -3.5 K
vs -1.2 K), and alone it leaves 416; the reason is visible in the traces --
the shrink trains an amplifier (PS21 late amplitude of z 1.04 with the
surface at 0.76, an inconsistent state), whereas the pushforward trains a
restorer. The stratospheric-q exclusion did its job (PS21 q 6.0x vs S21
8.1x) but is moot. (iii) What remains in P21: a residual cold level (t2m
-1.2 K, z500 -230 m2/s2), a zonal-wind level (u250 +3.4 m/s, the sign flipped
from the control's -2.7), wind alphas of 0.2-0.5 and a slight
over-amplification of tropospheric T and q (1.07, 1.14). Two epochs (P22)
and the paired-versus-unpaired sampler (P21 with `fresh_noise_scale` 1.0)
are the next measurements, plus a five-year run to see whether 152 is a
plateau or a slow creep. (iv) The level-offset variant L22 becomes a
secondary check (does an explicit level term remove P21's residual -1.2 K?)
rather than the fix.

**P21 under the plain sampler (job 7602000, `fresh_noise_scale` 1.0):**
plateau 141 (paired 152), z500 541 / -287, t2m 1.84 / -1.42 K, alpha
identical to three decimals on every group, late tropospheric amplitude
1.07 / 0.79 (T / v; paired 1.07 / 0.83); the only difference is the
short-lead spread -- surface / upper-air / diagnostic 0.098 / 0.122 / 0.186
at day 10 unpaired against 0.108 / 0.137 / 0.213 paired (ERDM 0.136 /
0.146 / 0.209) -- and at one year the two agree (0.314 vs 0.315). So the
mean-state repair comes entirely from the training-side self-generated
anchors; the fresh-slot injection remains what Phase 2/3 found it to be, a
dispersion knob that brings the day-10 spread to ERDM's, and the two are
independent. Recommended pairing for production: train with
`pushforward_rolls` and sample with `fresh_noise_scale` 1.45 (or whatever
matches the spread-skill target), knowing the climate does not depend on the
latter.

*22:20 UTC:* the bundle production chain reached its target (epoch 24
written 22:0x UTC). Because P21/P22 start from a mid-schedule epoch, the
deployable candidate is the same fine-tune applied to the finished model:
**P24 = `checkpoints_ft5_pf24_b40`**, `pushforward_rolls 2` +
`fresh_noise_scale 1.45` for two epochs from epoch 24 (the cosine schedule
is at its floor, base lr 5e-5 / Muon 5e-4), jobs 7602052 -> 7602053 ->
7602054, to be scored over one and five years against the plain epoch 24
(B24). Also queued: the first five-year wave (job 7602002: P21 paired and
plain, C22, S22, PS21).

**Wave 2a (job 7602113): PS22, L22 and the finished plain model B24, one year:**

| | B24 plain e24 | S22 shrink | **L22 shrink + level** | PS21 pf+shrink (1 ep) | PS22 pf+shrink (2 ep) | P21 pf (1 ep) | ERDM |
|---|---|---|---|---|---|---|---|
| z500 bias-map RMSE / mean bias | 2091 / -1137 | 1805 / -1483 | 994 / -497 | 999 / -951 | 1268 / -1147 | **542 / -230** | 42 / -4 |
| t2m bias-map RMSE / mean bias (K) | 11.49 / -5.43 | 7.99 / -6.06 | 6.08 / -3.82 | 4.18 / -3.51 | 4.78 / -4.11 | **1.59 / -1.19** | 0.21 / -0.01 |
| t850 bias-map RMSE / mean bias (K) | 9.10 / -3.67 | 7.21 / -5.18 | 5.19 / -1.60 | 4.23 / -3.81 | 4.61 / -3.92 | **1.52 / -1.02** | 0.18 / -0.03 |
| surface RMSE-vs-clim, mean steps 100-365 | 795 | 455 | 422 | 290 | 329 | **152** | 63 |
| upper-air RMSE-vs-clim, mean steps 100-365 | 553 | 693 | 369 | 549 | 511 | **356** | 453 |
| alpha skt / sp / t2m / q2m / u10 / v10 | 0.70 / 0.59 / 0.71 / 0.74 / 0.83 / 0.85 | 0.31 / 0.23 / 0.32 / 0.47 / 0.62 / 0.71 | 0.28 / 0.15 / 0.29 / 0.42 / 0.52 / 0.61 | -0.06 / -0.03 / -0.06 / 0.14 / 0.35 / 0.43 | 0.09 / 0.09 / 0.08 / 0.24 / 0.48 / 0.53 | **-0.02 / 0.03 / -0.03 / 0.01 / 0.39 / 0.29** | ~0 |
| alpha upper-air T / u / v / z / q | 0.79 / 0.84 / 0.85 / 0.75 / 0.82 | 0.63 / 0.68 / 0.68 / 0.49 / 0.41 | 0.58 / 0.57 / 0.68 / 0.40 / 0.59 | 0.52 / 0.47 / 0.49 / 0.17 / 0.07 | 0.41 / 0.58 / 0.58 / 0.33 / 0.22 | **0.32 / 0.50 / 0.22 / 0.25 / 0.30** | ~0 |
| anchor trace: onset roll / late amplitude sfc+diag, trop. T, z | 34 / 0.37, 0.34, 0.34 | 62 / 0.53, 0.50, 0.53 | 99 / 0.69, 0.59, 0.82 | 90 / 0.76, 0.91, 1.04 | 126 / 0.72, 0.86, 0.77 | **335 / 0.91, 1.07, 0.85** | |
| 10-day validation, surface RMSE step 1 / 10 | 1.95 / 116 (e20) | 9.2 / 161 | 11.8 / 189 | 12.0 / 105 | 8.7 / 89 | **2.0 / 58** | |

Readings. (i) The finished plain model (B24) drifts exactly like epochs 20
and 22 (795 / 792 / 794): the production run's last four epochs changed
nothing about the drift, so the P24 fine-tune below starts from the same
place as P21 did. (ii) The level term does what it was added for: on top of
the shrink it brings the one-year t2m level from -6.06 K to -3.82 K, T850
from -5.18 to -1.60 K and z500 from -1483 to -497, and the upper-air
RMSE-vs-climatology from 693 to 369 -- so the shrink's level damage was a
level problem the network could unlearn once shown wrong-level anchors --
but the pattern alpha barely moves (t2m 0.32 -> 0.29, sp 0.23 -> 0.15) and
the plateau only from 455 to 422. (iii) A second epoch of pushforward +
shrink is worse than the first (329 vs 290, level -4.1 vs -3.5 K): the two
augmentations fight, and the shrink wins over time. (iv) Ranking after 1-2
augmented epochs, by plateau: pushforward alone 152 << pushforward + shrink
290-329 < shrink + level 422 < shrink 455 << plain 792-795, with ERDM at 63.
Self-generated anchors are the fix; the shrink should be dropped from the
recipe, and the level term is only worth re-testing on top of the
pushforward if P22/P24 keep the residual -1.2 K level.

**Queued at hand-off (2026-09-10 00:30 UTC; the `small` queue is blocked by
a 10-hour reservation until about 05:35 UTC):** P22 second epoch (jobs
7601656 -> 7601657, `checkpoints_ft5_pf_b40` epoch 22); P24 (7602052 ->
7602053 -> 7602054, `checkpoints_ft5_pf24_b40` epochs 25-26); the five-year
wave (7602002: `eval_bias5yr_{p21_k145,p21_base,c22_base,s22_base,ps21_k145}`).
To finish: once the P22 / P24 checkpoints exist, score them with

    qsub -q debug-scaling -l select=8:system=polaris -l walltime=01:00:00 \
      -v EVAL_TAG=bias1yr,EVAL_JOBS=p22_k145:checkpoints_ft5_pf_b40:22:k145+p22_base:checkpoints_ft5_pf_b40:22:base+p26_k145:checkpoints_ft5_pf24_b40:26:k145+p26_base:checkpoints_ft5_pf24_b40:26:base \
      polaris_rsi_drift_eval_multi_phase5.pbs

(and a five-year wave for P26 with `EVAL_TAG=bias5yr EVAL_HORIZON=1827` on
10 nodes), then fetch each `eval_suite.pt` and `trace_rank0.pt` and run
`polaris_results.py compare` / `trace` as in the tables above. If the
five-year P21 plateau stays near its one-year value (152), the recipe change
for production is: keep `loss.anchor_shrink 0`, set `loss.pushforward_rolls
2` (with `fresh_noise_scale 1.45` in the loss config so the training rolls
match the sampler), and sample with `rsi_sstpred_e1_k145`.

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
switches on off-manifold (Tests 2 and 4 decide).

**Test 1, full (job 7600870; teacher-forced true windows, 4 ICs, global
t = 0.5, slot 6 = the anchor-producing readout; 151 state channels):**

| quantity, slot 6 | RSI mean (lo / mid / hi S_c tercile) | ERDM mean (lo / mid / hi) |
|---|---|---|
| std(y_hat)/std(y) | 0.996 (0.999 / 0.996 / 0.992) | 0.991 (1.000 / 0.993 / 0.980) |
| slope of y_hat on y | 0.991 (0.999 / 0.992 / 0.982) | 0.982 (0.998 / 0.986 / 0.962) |
| level gain (readout mean response to a uniform +0.3 window offset) | **0.77** (0.84 / 0.84 / 0.62) | **0.48** (0.62 / 0.60 / 0.23) |
| shrink response (readout anomaly std after x0.7 window shrink; 0.70 = pass-through, 1.0 = full restore) | 0.82 (0.80 / 0.81 / 0.83) | 0.82 (0.84 / 0.81 / 0.82) |

Per channel, level gain at slot 6 (RSI / ERDM): surface pressure 0.66 /
**0.06**, 2m temperature 0.74 / 0.38, T@500 0.79 / 0.58, z@500 0.82 / 0.55,
q@850 0.86 / 0.63, v@250 0.65 / 0.04, 10m v 0.55 / 0.05, precipitation 0.42 /
0.08, high cloud 0.47 / 0.17. Slot 1 (the emitted readout) has level gain
1.00 and shrink response 0.70 for both models (pure pass-through, as it
should be at tau = 11/12). Same picture at t = 0 (RSI slot-6 level gain 0.70,
ERDM 0.46).

Readings. (i) Amplitude: both heads are *above* the Bayes-floor expectation
for the fast channels (precipitation std ratio 0.98 RSI / 0.94 ERDM against
an oracle ~0.83): the trained readouts pass the window's pattern through
rather than shrinking it. No contraction anywhere on-manifold. (ii) Level:
this is the formulation asymmetry, measured. ERDM's back slot is sigma_max
noise, so its denoiser must and does re-derive the absolute level from the
forcings and the cleaner slots, and a uniform offset in the window is largely
ignored (5% passed for surface pressure). RSI's anchor readout sees an input
that *is* the previous state to within one increment, so an offset in the
window is information and is passed on at 0.66-0.86 for the slow channels.
That is Bayes-correct under RSI's training law (true anchors) and is exactly
why an off-manifold level error has nothing to oppose it in RSI while ERDM
resets it every roll. (iii) The x0.7 shrink is partly restored by both back
slots (0.70 -> 0.82), ERDM a little more for surface pressure (0.985 vs
0.814).

**Test 2, cascade with truth normalization (job 7600870; 4 ICs, batch 2,
120 rolls; emitted and anchor spatial-anomaly std / truth's, mean over 151
channels; lo / mid / hi S_c terciles in parentheses):**

| roll | RSI emitted | RSI anchor | RSI anchor, hi tercile | ERDM emitted |
|---|---|---|---|---|
| 1 | 1.000 | 0.995 | 0.973 | 1.000 |
| 10 | 1.012 (1.00 / 1.02 / 1.02) | 0.968 | 0.958 | 0.997 |
| 20 | 0.991 (0.98 / 1.03 / 0.96) | 0.958 | 0.917 | 0.992 |
| 28 | 0.999 (1.01 / 1.06 / 0.93) | 0.937 | **0.778** | 1.003 |
| 35 | 0.986 (1.09 / 1.05 / 0.81) | 0.917 | 0.680 | 0.995 |
| 50 | 0.838 (1.13 / 0.94 / **0.45**) | 0.732 | 0.384 | 1.010 |
| 70 | 0.731 (1.02 / 0.77 / 0.42) | 0.678 | 0.374 | 0.995 |
| 100 | 0.768 (1.07 / 0.78 / 0.46) | 0.701 | 0.418 | 1.005 |
| 120 | 0.725 (1.09 / 0.69 / 0.40) | 0.666 | 0.366 | 0.986 (1.00 / 0.98 / 0.98) |

Per channel, RSI emitted / truth at rolls 28 / 50 / 100 / 120: precipitation
0.99 / 0.27 / 0.28 / 0.25; v@250 0.80 / 0.28 / 0.28 / 0.25; 10m v 0.92 /
0.40 / 0.35 / 0.33; 2m temperature 0.96 / 0.75 / 0.37 / 0.37; T@500 1.01 /
0.69 / 0.38 / 0.39; z@500 0.84 / 0.59 / 0.32 / 0.33; q@850 0.94 / 0.53 /
0.35 / 0.36; surface pressure 0.98 / 0.91 / 0.71 / 0.71. ERDM: every channel
0.87-1.05 at every roll. The RSI window slots at roll 120 sit at 0.67-0.73
of truth. Confirms the sweep-trace reading with an independent truth
normalization and the ERDM control: latency to roll ~28, fast-channel
anchors lead (0.78 at roll 28 while their emitted frames are still 0.93),
the fast channels lose 55-75% of their amplitude by roll 50, the slow
channels follow by roll 70-100, and the slow tercile's *instantaneous*
anomaly std ends *above* truth (1.07-1.13) because the collapsed state has a
different spatial structure, not a uniformly damped one.

**Test 4 (multi-roll flush, 4 ICs x 2, 36 rolls; job 7600866).** Metric:
`rms(perturbed - reference) / rms(perturbation)` per group, divided by the
model's own two-seed floor `rms(ref2 - ref) / rms(perturbation)`; a ratio of
1 means the perturbation has dissolved into weather chaos, > 1 means part of
it persists coherently.

*From the true IC window* (anomaly x0.7 / uniform -0.3, surface group, ratio
to floor at rolls 6 / 12 / 18 / 24 / 36):

| model | shrink | offset |
|---|---|---|
| RSI | 2.2 / 1.6 / 1.2 / 1.1 / 0.96 | 2.2 / 1.4 / 1.2 / 1.1 / 1.0 |
| ERDM | 2.0 / 1.2 / 1.0 / 0.9 / 1.0 | 1.5 / 1.1 / 1.0 / 1.0 / 1.1 |

Both models flush an on-manifold perturbation within ~24 rolls; RSI holds it
about 1.5x longer at rolls 6-18.

*From the free-run window at roll 30* (the off-manifold case; the RSI window's
anomaly amplitude there is still 1.00-1.03 of truth, so the drift has not yet
set in):

| model | offset, surface (ratio at 12 / 18 / 24 / 30 / 36) | offset, diagnostic | shrink, surface |
|---|---|---|---|
| RSI | 1.74 / 1.55 / 1.35 / 1.22 / 1.20 | 1.22 / 1.23 / 1.26 / 1.23 / 1.18 | 2.31 / 1.84 / 1.56 / 1.34 / 1.28 |
| ERDM | 1.47 / 1.02 / 1.04 / 1.03 / 1.01 | 1.29 / 1.03 / 0.99 / 0.97 / 1.00 | (denominator invalid: the back slot is sigma_max noise; fixed for the next run) |

ERDM erases a uniform offset applied to its rolling window by roll 18. RSI
still carries ~1/3 of the offset (surface: excess `sqrt(0.60^2 - 0.50^2) =
0.33` of the applied -0.3) and ~45% of the pattern shrink at roll 36, and for
the upper air the perturbed trajectory departs from the reference far beyond
the chaotic floor (4.7 vs 2.7 at roll 36). This is the "no restoring force
once the window is the model's own" signature on the real model, and it is
RSI-specific. Caveat: 4 ICs, floors are noisy at the +-10% level; the
upper-air excess deserves a per-channel look in the cascade outputs.

**Phase 2 sweep, 300 days (job 7600840), headline bias-map RMSE
(180x360, 8 members, obs-climatology truth):**

| variant | z500 RMSE (m2/s2) | t2m RMSE (K) | t2m mean bias (K) |
|---|---|---|---|
| k120 (`fresh_noise_scale` 1.2) | 1858 | 9.28 | -3.79 |
| k145 (1.45) | 1850 | 9.18 | -3.81 |
| k180 (1.8) | 1824 | 8.97 | -3.71 |
| fd (`final_denoise` true) | 1771 | 9.05 | -3.43 |
| ns4 (`num_steps` 4) | killed at the 1 h wall after 200 scored frames (partial save valid to frame 200); trace below | | |
| baseline, 5-year mean for scale | 2059 | 10.42 | -3.47 |

ns4 trace (surface RMSE vs climatology / surface spread), base in
parentheses: step 10: 162 (165) / 0.114 (0.110); step 20: 163 (154) / 0.237
(0.232); step 30: **383 (219)** / 0.299 (0.313); step 50: 565 (487) / 0.271
(0.291); step 85: 646 (619); step 100: 639 (620); step 200: 624 (614). Four
solver steps accelerate the run-away like `final_denoise` does and change the
spread by nothing measurable. Both variants make the head evaluations more
numerous or cleaner; both speed the collapse up. Lesson for the job script:
a 300-frame 4-step run needs ~1.3 h, more than debug-scaling allows.

Inflating the fresh-slot latent by 1.2-1.8x leaves the 300-day collapse
essentially unchanged (the trend with kappa is a few percent, within what
member noise can do); `final_denoise` buys ~5% on the headline. Exactly the
report's prediction for Layer A alone: the dispersion defect is not the
time-mean drift.

Trace-level comparison with the baseline's first 300 steps (`rmse_acc`,
8 members; surface group, physical units, dominated by surface pressure):

| step | base | ERDM | k120 | k145 | k180 | fd |
|---|---|---|---|---|---|---|
| spread 10 (surface / diagnostic) | 0.110 / 0.221 | 0.136 / 0.251 | 0.115 / 0.235 | 0.121 / 0.251 | 0.130 / 0.276 | 0.118 / 0.236 |
| spread mean 100-300, ratio to base (sfc / ua / diag) | 1 | 1.22 / 0.65 / 1.04 | 1.00 / 0.97 / 0.99 | 1.00 / 0.97 / 0.99 | 1.04 / 1.03 / 1.04 | 1.02 / 0.94 / 0.98 |
| RMSE-vs-clim, surface, step 30 | 219 | 75 | 202 | 190 | 149 | **402** |
| step 50 | 487 | 70 | 479 | 471 | 426 | 517 |
| step 85 | 619 | 53 | 621 | 611 | 596 | 616 |
| step 100-300 mean, ratio to base | 1 | 0.10 | 1.00 | 1.00 | 0.99 | 1.04 |

Readings. (i) The fresh-slot inflation does what Layer A says it does and
nothing more: the day-10 spread deficit closes (at 1.45 the diagnostic
spread equals ERDM's 0.251; at 1.8 the surface spread reaches ERDM's 0.13),
while the long-lead spread (100-300) is unchanged, because the collapsed
state sets it. (ii) The drift trace is untouched by kappa 1.2-1.45; kappa
1.8 delays the run-away by ~5 days (day-30 149 vs 219) but reaches the same
plateau (~600) by day 85. (iii) `final_denoise` makes the run-away arrive
*earlier* (day-30 402 vs 219) and ends at a slightly higher plateau: reading
the anchor at tau = 1/6, i.e. a cleaner conditional mean, accelerates the
collapse, which is the direction the "chain of conditional means" mechanism
predicts and the opposite of what would help. Layer A is confirmed as the
dispersion defect and eliminated as the drift; the drift needs the
training-side fix (Phase 4).

**Test 2 from the sweep's anchor traces** (`RSI_TRACE_PATH`, 4 of the 8
members of k120, 300 rolls; per-channel spatial-anomaly std relative to the
roll-1 emitted frame, mean over the 151 state channels; k145/k180 agree to
within 0.03):

| roll | 1 | 5 | 10 | 20 | 28 | 35 | 50 | 70 | 85 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emitted | 1.00 | 0.97 | 1.00 | 0.96 | 0.97 | 0.94 | 0.82 | 0.69 | 0.69 | 0.70 | 0.69 | 0.70 |
| fresh-slot anchor | 0.98 | 0.97 | 0.93 | 0.92 | 0.91 | 0.86 | 0.69 | 0.64 | 0.64 | 0.63 | 0.64 | 0.64 |
| anchor / emitted, same roll | 0.98 | 1.00 | 0.93 | 0.96 | 0.92 | 0.86 | 0.82 | 0.96 | 0.94 | 0.92 | 0.93 | 0.93 |

By S_c tercile (low S < 0.058 / mid / high S > 0.15), emitted amplitude:
roll 28: 0.95 / 1.07 / 0.90; roll 50: 0.99 / 1.03 / **0.44**; roll 100:
0.87 / 0.83 / 0.41; roll 300: 0.88 / 0.80 / 0.41. Anchor at roll 28: 0.94 /
1.02 / **0.77**. Per channel (emitted, rolls 28 / 50 / 100 / 300):
precipitation 1.06 / 0.30 / 0.31 / 0.32; v@250 0.75 / 0.26 / 0.23 / 0.21;
2m temperature 0.97 / 0.76 / 0.41 / 0.37; T@500 1.05 / 0.71 / 0.41 / 0.39;
z@500 0.85 / 0.59 / 0.33 / 0.33; surface pressure 0.97 / 0.92 / 0.73 / 0.73.

Readings. (i) The anchor chain does not decay geometrically from roll 1: it
holds ~0.92 through roll 28 and then transitions in ~20 rolls, i.e. the
latency-then-transition shape of the brief's trace, and the anchor leads the
emitted frame by 10-15 rolls as predicted. (ii) The anchor/emitted ratio is
flat at 0.92-0.96 on- and off-manifold: the readout does not shrink more per
roll as the state collapses, so a compounding readout contraction (the skip
mechanism) is ruled out directly. (iii) The transition starts in the fast
channels: at roll 28 the anchors of v-wind, precipitation and cloud are
already at 0.6-0.8 while their emitted frames are intact, and by roll 50
those channels have lost 60-75% of their amplitude while the slow channels
have lost nothing; the slow channels (T, z, surface pressure) follow between
rolls 50 and 100 through the network's cross-channel response. This is the
composite chain of the report: the Layer A anchor-chain deficit (timescale
2/S_c^2 ~ 15-25 rolls for S_c 0.3-0.4) takes the fast channels off-manifold
first, and off-manifold the network has no restoring force (Test 4), so the
whole coupled state collapses. (iv) The instantaneous pattern amplitude
plateaus at 0.70 (0.88 slow / 0.42 fast); the brief's time-mean amplitude of
0.15-0.45 is lower because the time mean also loses the part of the pattern
that the weather variance carries.
