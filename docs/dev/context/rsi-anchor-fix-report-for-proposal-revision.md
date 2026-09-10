<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI: what the drift campaign established, and a proposed change to the anchor

**Audience.** The author of `rolling_stochastic_interpolants_proposal.md` (the
RSI design proposal, Aug 2026). This report is written so that the proposal
and its ablation ladder (A0-A5, section 5) can be revised against what has
since been measured on the real model. Everything quantitative below was run
on ALCF Polaris between 2026-09-08 and 2026-09-10 on the AMIP daily-mean
`sst_pred` configuration (45x90, 153 channels, W = 6, RollingDiT 561M); the
full tables are in [rsi-drift-polaris-test-plan](rsi-drift-polaris-test-plan.md)
and the diagnosis in [rsi-drift-diagnosis-report](rsi-drift-diagnosis-report.md).
Numbers quoted here are copied from those logs.

**One-paragraph summary.** RSI as proposed (A2: clean-data anchor
`a_w = y_{w-1}`, scalar white `Gamma`, `Gamma_0` = one increment standard
deviation) beats ERDM at one to six days and then loses its climate: within
30 to 100 rolls the free run collapses to a state whose anomaly pattern is
about 0.35 to 0.4 of the truth's, and its multi-year time mean is a shrunk
copy of the observed anomaly field plus a cold level. The cause is in the
entering-slot rule of section 3.2: the fresh slot's base is the state head's
readout of slot W, which at local time 1/12 is the conditional mean of the
next frame given the previous anchor, so the anchor sequence is the
conditional-expectation operator iterated without innovation. This violates
the proposal's own boundary-consistency condition ("provided the anchor
entering the construction is distributed as in training"), and the two
mitigations the proposal names cannot repair it: anchor perturbation adds
white noise the latent head is trained to remove, and self-generated-anchor
fine-tuning removes the drift for about two years and then collapses because
it only shows the head the first links of the chain. The fix proposed here
keeps option A and changes the lag: anchor slot w on the frame W steps back,
which is the frame being emitted at the roll where the fresh slot is created,
i.e. a completed conditional sample rather than a mean. This is a new rung on
the coupling axis of section 3.6, between A1 and A2, and it is not in the
current ladder.

---

## 1. Measured facts the revision should be consistent with

**1.1 The collapse and its anatomy.** Metrics: surface RMSE of the 8-member
ensemble mean against the observed climatology, averaged over days 100-365
("plateau"; ERDM sits at 63-67); the shrinkage alpha of the time-mean bias map
(lat-weighted regression of bias on the observed anomaly pattern; 0 is no
shrink, 1 is a flat climate); the per-roll anchor trace (spatial-anomaly std
of the emitted frame and of the fresh-slot anchor relative to roll 1).

| model | 1-yr plateau | 5-yr plateau | alpha, slow / fast surface channels | t2m level, 5 yr |
|---|---|---|---|---|
| ERDM epoch 24 | 63 | 65 | 0.00 / 0.00 | -0.01 K |
| RSI previous production run, epoch 24 | ~609 | 611 | 0.68 / 0.87 | -3.65 K |
| RSI bundle run, epochs 20, 22, 24 | 794, 792, 795 | 783 | 0.71 / 0.85 | -5.56 K |

The bundle run has 11-12% lower training loss than the previous run and
better 1-6 day skill, and it drifts harder. The defect scales with fit, not
with the recipe.

Anchor traces (one member, 300-1800 rolls): the anchor chain holds ~0.93 of
the truth's anomaly amplitude through roll 28 and then transitions in ~20
rolls, fast channels first (S_c > 0.15: anchors 0.78 at roll 28, 0.38 at roll
50), slow channels following through cross-channel coupling by roll 70-100.
The anchor is smoother than the emitted frame at every roll (anchor/emitted
amplitude ratio 0.92-0.96, flat on and off manifold), so there is no
compounding readout contraction; the contraction is at the anchor. The
onset roll and the per-channel alpha correlate with the increment scale
(Spearman 0.78 over 81 channels), as expected for a per-link loss of the
one-day posterior variance.

**1.2 Teacher-forced probes.** The slot-W readout passes 99.6% of the truth's
anomaly amplitude when its window is truth (ERDM's back-slot denoiser: 99.1%):
the per-link contraction is tiny on-manifold. But RSI's readout passes 77% of
a uniform level offset added to the window (ERDM 48%; surface pressure 6%),
and from a free-run window at roll 30 RSI still carries a third of an
injected offset 36 rolls later where ERDM has erased it. ERDM re-derives level
and pattern from context every roll; RSI trusts its anchor, because in
training the anchor was always exactly a true frame.

**1.3 The inference-side knobs.** Inflating the fresh-slot latent by 1.2, 1.45
or 1.8 closes the day-10 spread gap to ERDM (1.45 matches) and leaves the
five-year alpha unchanged to within 0.006 on every channel and the plateau to
within 1%. `final_denoise` and four sampler steps both make the collapse
faster (a better estimate of a mean is a smoother field). The `eps`-family
noise (A5) moves spread, not the mean.

**1.4 Training-side remedies, all as two-epoch fine-tunes from the bundle
epoch 20, scored over one year (control = the chain's own epoch 22):**

| remedy | 1-yr plateau | 5-yr plateau | t2m level | alpha slow / winds | note |
|---|---|---|---|---|---|
| none (control) | 792 | 783 | -5.4 K | 0.71 / 0.85 | |
| anchor pattern shrink 0.3 (`a <- <a> + s (a - <a>)`, s ~ U(0.7, 1)) | 455 | 450, flat | -6.1 K | 0.32 / 0.71 | trains an amplifier; slow channels held at 0.5-0.7, fast winds still collapse; over-amplifies stratospheric q 5-70x |
| shrink + uniform level offset `a <- a + s S_c e` | 422 | | -3.8 K | 0.29 / 0.61 | repairs the shrink's level damage only |
| self-generated anchors, 1 epoch (`pushforward_rolls 2`: roll own sampler k ~ U{0..2} times from truth, anchor last k slots on own slot-W readouts, targets truth, injection 1.45 as at inference) | **152** | 143, 211, 419, 593, 606 by year | -1.2 K | **-0.03 / 0.3-0.4** | anchor chain intact 335 rolls, then collapses in year 2-3 to the base state |
| self-generated anchors, 2 epochs | 159 | queued, then cancelled | **-0.24 K** (z500 +13) | 0.00 / 0.3-0.4 | plain sampler creeps late in year 1: the head adapts to the trained anchor law |
| self-generated + shrink | 290 | 320, flat | -5.9 K | 0.02 / 0.4-0.5 | the two fight; second epoch worse (329) |

Short-range cost: the shrink multiplies the step-1 surface RMSE by five; the
self-generated anchors cost nothing at day 1 and halve the day-10 error
(58 vs 116; the previous production model: 81).

**1.5 The toy models** (tools/diagnostics/rsi_drift) reproduced the ordering:
anchor noise lowers dispersion monotonically and does not move the mean;
the pattern shrink has a positive one-roll restoring force; self-generated
anchors give the best pattern but introduced a level bias.

---

## 2. Where the conditional mean enters, mechanically

Both models' heads are squared-loss regressions and both return conditional
means; both samplers use them only as the drift of an ODE that preserves the
noise realization carried in the state. The three settings that take the
code from ERDM (A1) to RSI (A2) are: `beta` from identically one to `tau`
(the base end of each slot becomes the anchor instead of pure noise);
`Gamma(0)` from the state scale (sigma_max 500) to one increment standard
deviation (the anchor dominates the base state); and the fresh slot from
`Gamma(0) z` to `a + Gamma(0) z` with `a = y_hat[:, -1]`, the state head's
readout of slot W. The mean enters at the third setting, made consequential
by the first two: in ERDM no value of the denoiser is ever assigned to the
state, in RSI the readout is copied out of the ODE into the base of a new
interpolant.

At that moment slot W is at local time 1/12:

```
x_W = a + (1/12) (y_W - a) + Gamma(1/12) z
```

The only term carrying information about y_W beyond the anchor is the
increment scaled by 1/12 under noise of about one increment standard
deviation; its signal-to-noise ratio is of order 1/150, so the readout is
`E[y_W | a, context]` to that accuracy, independent of the latent that would
later be transported into the conditional spread of y_W. The anchor sequence
therefore obeys `a_{n+1} = E[y_{n+1} | a_n, context_n]`, the
conditional-expectation operator iterated with no innovation, which for a
stationary Markov process converges to the climatological mean and by the law
of total variance removes the one-day posterior variance per link. The
emitted frames are genuine conditional samples (the ODE adds the innovation
to them over the following five rolls), but they are conditioned on the
already-deficient chain and are never fed back. The window's other slots
inherit the deficit through their `(1 - beta)` anchor weight, so the head,
trained on full-variance context, continues a smoother state as if it were
real. This is why a stronger fit drifts harder and why the fresh-slot noise
cannot help: the missing quantity is structured posterior variance, and
white noise of the right amplitude is exactly what the latent head is trained
to remove.

In ERDM the new frame's information is regenerated by the ODE from pure noise
conditioned on the context, and that noise is converted into the frame's
conditional spread before anything downstream uses the frame. The states
carried between rolls always have the forward-process variance. There is a
mean in ERDM at every step, but only as a drift.

**Relation to the proposal's text.** Section 3.2's boundary/marginal
consistency paragraph states the correct condition and then, in the
parenthetical about the entering slot ("the natural anchor for the entering
slot is the current partially-resolved estimate of slot W; its residual noise
is absorbed into Gamma_0 by the anchor-perturbation training scheme"),
assumes the estimate differs from a sample by noise. It differs by a missing
structured variance and by being an expectation. Section 7 names anchor
mismatch under rollout as the main risk and proposes anchor perturbation and
a self-generated-anchor fine-tune as mitigations; both were run, the first is
neutral to harmful, the second removes the drift for about two years. The
shipped model is in effect option-A training paired with an option-B-like
inference anchor (a denoiser estimate of a neighbouring slot) without
option-B training, which is the mismatch in its purest form.

---

## 3. The proposed change: anchor on the frame that has completed its transport

**3.1 Rule.** In a rolling window of length W exactly one frame finishes its
transport at every roll, the one being emitted; its readout at local time
11/12 is a conditional sample up to the `Gamma_1` floor. Anchor the fresh slot
on it:

```
today:     x_fresh(0) = y_hat[:, -1] + Gamma(0) z      anchor = mean of frame n+W     (lag 1)
proposed:  x_fresh(0) = y_hat[:,  0] + Gamma(0) z      anchor = sample of frame n+1   (lag W)
```

and, consistently, every slot's interpolant runs from the frame W steps
earlier to its target:

```
x_w(tau) = y_{w-W} + beta(tau) (y_w - y_{w-W}) + Gamma(tau) z,     Gamma(0) = 1 x std(y_w - y_{w-W}) per channel
```

Every anchor in the window is then an emitted sample; no state variable is
ever set equal to a head output; the base of every slot has the data's
variance plus the noise, exactly the property that protects ERDM. The
consistency condition of section 3.2 is met by construction rather than
enforced approximately.

**3.2 Where it sits.** The anchor lag L is a dial along the coupling axis of
section 3.6. L = 1 is A2; its anchor is necessarily a mean because no lag-one
frame is complete when it is needed. L = W is the smallest lag whose anchor
is a complete sample in a window of length W. As `Gamma(0)` approaches the
state scale the design becomes A1/ERDM. At L = W the base-end noise is the
W-day increment scale: still well below the anomaly scale for surface
pressure, temperature and geopotential (which keep the RSI advantage), and
close to the full anomaly for the fast winds and precipitation (whose slots
start nearly from noise, ERDM-like, which is where their anchor chain
collapsed first). This is the "forecast stride" prediction of section 5 run
as a controlled experiment: the anchor stride is lengthened while the
emission stride stays daily.

**3.3 Implementation in `physicsnemo/experimental/diffusion/rsi.py`.**

- `_fresh_slot`: use `y_hat[:, 0:1]` instead of `y_hat[:, -1:]`.
- `compute_loss`: the state stack carries 2W frames; anchors are `y[:, :W]`,
  targets `y[:, W:]`. `anchor_frames` becomes W and `init_frames` 2W; the
  recipe's loader already reads extra history frames (`history_frames`,
  added for the pushforward work) and `_pack_window` is generic in the frame
  count.
- `warmup_window` builds slot w from `y_{w-W}` and `y_w` at the t = 0
  staircase (2W oracle frames).
- Noise scale: recompute `sigma_c_sstpred153.pt` for the W-day increment
  (`tools/data/amip/make_rsi_delta_scales.py` with lag W); keep `gamma_0 = 1`
  in those units, `gamma_1` as is.
- Ocean imposition (`impose_ocean`): the anchor-time boundary is the
  forcing W steps back, not one step back; the drivers pass `c_grid` windows,
  so this is a slicing change.
- Unchanged: heads, `c_in`/`z_precond`/H1 skip, coefficient integrator,
  weighting, readout, trace. The ERDM reduction (`reduce_to_erdm`) still
  holds since `beta = 1` removes the anchor regardless of lag, so A1 parity
  tests remain the guard.
- The backbone is unchanged, so a fine-tune from current weights is possible
  as a first look; a training run from scratch is the clean test because the
  base of every slot changes.

**3.4 Cost and expected trade.** No extra head evaluations per roll (the
emitted readout exists anyway); training reads W more frames per sample.
The short-range skill of lag 1 is given up: the fresh slot starts from the
state six days back plus the partially resolved context, which is ERDM's
information set plus one frame, so day-1 skill should land near ERDM's, with
the slow channels keeping some advantage. What it buys is a memory made only
of samples. If the emitted readout's residual posterior variance at 11/12
matters, `final_denoise` or a slightly larger `Gamma_1` are the knobs; it
should be negligible.

**3.5 The alternative that was not chosen.** Making the newest frame's anchor
a sample instead of a mean requires finishing that frame's transport at the
moment the fresh slot is created: about fifteen extra head evaluations per
roll, and a realization different from the one the window later emits for
the same frame. Adding posterior variance to the mean as noise fails for the
reason in section 2 unless the noise is structured, and structured posterior
samples require a generative step. The lag-W anchor gets that step for free.

**3.6 What to measure first.** One-year run with the existing protocol
(`climate_eval_suite.py`, 8 members, obs-climatology truth, `RSI_TRACE_PATH`
on): the anchor/emitted amplitude ratio should sit at 1.0 from the first roll
(today 0.92-0.96), the fast-wind amplitude should not decay, alpha should be
near zero on every channel. Then five years, since the self-generated-anchor
result showed a one-year plateau can hide a year-3 collapse. The ten-day
validation quantifies the short-range cost.

---

## 4. Suggested revisions to the proposal

1. **Section 3.2, entering-slot initialization.** Replace the claim that the
   partially resolved estimate's residual noise is absorbed into `Gamma_0`
   with the statement that this estimate is a conditional mean whose missing
   variance is structured and cannot be restored by perturbation, and that
   the consistency condition requires the entering anchor to be a completed
   sample. Present option A with anchor lag L as the design axis; lag 1 is
   only consistent if the fresh slot is transported before use.
2. **Section 3.6.** The coupling axis has an intermediate point: lag-W
   anchor (sample anchor, W-day increment noise) between A1 (anchor zero) and
   A2 (lag-1 mean anchor). The special-case statement is unchanged.
3. **Section 5, ladder.** Insert the lag as a rung: A2 (lag 1) and A2-L (lag
   W), so the "gap shrinks with stride" prediction becomes a direct test on
   ERA5 as well as on Navier-Stokes. Drop anchor-perturbation strength from
   the mitigations (measured neutral to harmful) and keep it, if at all, as a
   negative control. Add self-generated anchors (`pushforward_rolls`) as a
   training-side rung for the lag-1 model, with the caveat that its
   restoring force is local to the chain depth it sees. Consider whether A3
   (residual parameterization) and A4 (spectral Gamma) should be run on the
   lag-W base rather than on lag 1, since the lag-1 base does not have a
   climate to calibrate against.
4. **Section 5, metrics.** The proposal's metrics are forecast metrics; the
   defect is invisible in them and in the training loss until roll 30. Add
   the long-rollout climate protocol as a primary metric: the RMSE-vs-
   climatology plateau, the shrinkage alpha of the time-mean bias map, the
   level bias, and the per-roll anchor trace (anchor/emitted amplitude
   ratio, onset roll of the fast channels), at one and five years.
5. **Section 7, risks.** Restate the main risk as structural: the lag-1
   entering slot is a mean by construction, not a noisy sample; the
   mitigations that were proposed do not remove it. Note the observed
   train/inference dependence on the sampler: once a head is trained on its
   own anchor law with a given injection, the inference sampler must use the
   same injection.
6. **Option B.** The shipped inference rule is already an option-B-style
   anchor without option-B training. Either commit to B with matching
   training (the exploratory analysis the proposal calls for) or use A with
   lag W; the current hybrid is the worst of both.

---

## 5. Open questions for the revision

- How much of RSI's short-range advantage survives at lag W, per channel?
  The proposal's M1 argument applies most strongly to the slow channels; a
  per-channel accounting of the W-day increment scale against the anomaly
  scale would say where the coupling still buys anything.
- Is there a consistent intermediate lag? Lags 1 < L < W have anchors that
  are partially transported states, not samples; anchoring on the noisy
  state itself (rather than the readout) preserves variance but drags an
  old frame forward. The toy's `anchor_xlast` variant was predicted worse;
  a proper analysis is missing.
- Should `Gamma(0)` at lag W be the W-day increment scale per channel, or
  larger for the fast channels to make them fully ERDM-like?
- Does the residual `Gamma_1` floor on the emitted readout matter as an
  anchor, or should the fresh slot use the `final_denoise` readout?
- The self-generated-anchor training removed the slow-channel collapse for
  two years on the lag-1 model at no short-range cost. Is it worth keeping on
  top of lag W as insurance, or does a sample anchor make it unnecessary?

---

## 6. Pointers

- Code (branch `rsi-drift-diagnosis`): `physicsnemo/experimental/diffusion/rsi.py`
  (`_fresh_slot`, `compute_loss`, `_pushforward_anchors`, `perturb_anchor`,
  `warmup_window`, `_trace`); `examples/weather/ai_rossby/train_diffusion.py`
  (`history_frames` loader contract); `examples/weather/ai_rossby/rsi_drift_probes.py`
  (readout / cascade / flush probes); `tools/diagnostics/rsi_drift/`
  (oracle and trained toys, `polaris_results.py compare|trace`).
- Runs and artifacts on Polaris under `/eagle/lighthouse-uchicago/members/awikner/physicsnemo-rsi`:
  `eval_bias1yr_*`, `eval_bias5yr_*` (each with `eval_suite.pt` and
  `trace_rank*.pt`), checkpoints `checkpoints_prod24_bundle_b40` (plain
  epochs 20-24), `checkpoints_ft5_{shrink,shrinklvl,pf,pfshrink}_b40`.
- Documents: [rsi-drift-diagnosis-brief](rsi-drift-diagnosis-brief.md) (the
  user's original brief), [rsi-drift-diagnosis-report](rsi-drift-diagnosis-report.md)
  (diagnosis, corrections to the brief, confirmed code findings A1-A12,
  literature), [rsi-drift-polaris-test-plan](rsi-drift-polaris-test-plan.md)
  (every run, table and reading, Phases 0-5).
