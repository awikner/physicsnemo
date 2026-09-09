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
