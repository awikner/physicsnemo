<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The batch-40 correction campaign: A0-HN / A2-L-HN, the SDE fix, and the first real calibration metrics (2026-09-15)

Branch `worktree-rsi-ablation-review`. Eight code commits plus this note;
everything is default-off or additive, so the uncorrected A0 and A2-L chains
keep running as controls.

## 1. What the A0-vs-upstream comparison actually showed

The question was whether the in-harness ERDM baseline (A0, global batch 40)
was training correctly. It is — and that turned out to be the interesting
part.

Scoring the **upstream batch-4 epoch-17 checkpoint with our own loss, sampler,
data and validator** (frozen weights, `checkpoints_upstream_ep17_val/` holding
a symlink, `ABL_TARGET=N+1 ABL_MAX_ITERS=60 ABL_EXTRA="++training.optimizer.lr=1e-12|
++training.ema.enabled=false|++training.stages.0.scheduler.num_warmup_epochs=0"`):

| quantity | upstream ep17 | A0 ep17 (EMA) | A0 ep17 (raw) |
|---|---|---|---|
| in-harness loss, 60 batches | 242.0 +- 1.3 | 245.6 (epoch mean, 1315 batches) | 243.9 +- 1.3 |
| ocean term | 0.024 | 0.05-0.07 | 0.05-0.07 |
| surface RMSE day 1 | 7.4 | 7.04 | 7.07 |
| day 3 | 13.0 | 18.5 | 18.0 |
| day 6 | 22.6 | 110 | 111 |
| day 10 | 58 | 129 | 125 |

So: **equal loss, 2.3x worse day-10 skill, 5x worse day-6, equal day-1.** The
EMA horizon is not the cause (raw and EMA differ by <5%).

`ERDM_LOSS_DIAG=1` (added for this) logs the weighted loss, the raw squared
error and sigma PER WINDOW SLOT. On the rolling staircase the slot index IS
the noise-level bin, so this is the per-sigma view the total hides. Both jobs
ran with the same seed, so the noise levels and draws are paired batch for
batch:

| slot | mean sigma | share of weighted loss | A0 / upstream (weighted) | (raw sq. err) |
|---|---|---|---|---|
| 1 | 0.004 | <0.1% | 1.014 | 1.006 |
| 2 | 0.015 | 1.1% | 0.979 | 0.968 |
| 3 | 0.074 | 20% | 0.986 | 0.986 |
| 4 | 0.49 | 49% | 0.991 | 0.993 |
| 5 | 5.2 | 27% | 1.047 | 1.054 |
| 6 | 118 (range 16-500) | 1.8% | **1.063** | **1.076** |

A0 is at or below upstream on the four low-noise slots and 5-8% worse on the
two highest, which together carry under 30% of the loss — and slot 6, the
sigma_max end every free-run frame is generated from, carries 1.8%. That is
how a 2.3x skill gap hides inside a 1% loss difference.

**Why it is structural, not a bug.** ERDM hands every training instance a
fixed staircase of noise levels (one global `t ~ U(0,1)`, slot w at
`tau_w = (W-w+t)/W`) and folds EDM's log-normal sampling density into the loss
as a weight instead of sampling sigma from it. The induced density of
`u = ln sigma` inside a slot is `|dtau/du| ~ sigma^(1/rho)`, so at `rho = -10`
the staircase spends `sigma^-1.1` of EDM-equivalent attention per unit `u`:
21x down at sigma 118, 104x down at 500, relative to the peak at
`sigma = e^P_mean = 7.4`. At batch 40, with a tenth of upstream's optimizer
updates per epoch, that under-attended regime is simply the last thing
learned. Nothing in the harness is wrong: the same code reproduces upstream's
skill AND its loss from upstream's weights.

## 2. The correction: a high-noise loss boost

`_utils.high_noise_boost(sigma, hn_sigma, hn_power, hn_clip)` =
`clamp((sigma/hn_sigma)^hn_power, 1, hn_clip)`, applied in
`ERDMScheduler.loss_weight` (sigma) and `RSIScheduler.loss_weight`'s
`snr_bump` branch (sigma_eff) through the same helper, so the A1 reduction
survives it (pinned with the knob on both sides). `hn_power = 0` returns
`None` and the multiply is skipped: bit-identical, which the running controls
depend on. `hn_power = 1 - 1/rho` is the exponent that would restore EDM
equivalence exactly.

| | A0-HN (`conf/loss/erdm_v2_hn.yaml`) | A2-L-HN (`conf/loss/rsi_a2l_hn.yaml`) |
|---|---|---|
| coordinate | ERDM sigma | RSI sigma_eff = gamma/(beta delta_std) |
| hn_sigma / hn_power / hn_clip | 10.0 / 2.0 / 30.0 | 3.5 / 1.0 / 3.0 |
| multiplier | 1.000 to sigma 10, 2.56 at 16, 9 at 30, 25 at 50, 30 above 55 | 1.000 to sigma_eff 3.5, 1.43 at 5, 2 at 7, 3 above 10.5 |
| top slot: share of the loss | 1.44% -> **13.9%** (x11.19) | 8.23% -> **14.2%** (x1.85) |
| top slot mean weight | x8.14 | x1.77 |
| slots 1-4 | exactly 1 | exactly 1 |
| total loss | x1.158 | x1.070 |

**Both measured on frozen weights, and the two corrections differ 6x on
purpose.** A2-L's top slot already carries **8.2% +- 0.3% of its realized
weighted loss (312 samples, job 7626157) against ERDM's 1.44% (104 samples,
job 7626106)** -- 5.7x more attention to the regime free-run generation starts
from. That is the measured form of "RSI's weighting is already nearly flat",
and it means A2-L barely has the pathology A0-HN exists to fix. Matching the
two runs on the ENDPOINT (~14% each) therefore needs very different
multipliers; matching the multiplier instead would have put 46% of A2-L's loss
on one slot and inflated its total loss x1.71. An intermediate attempt at
3.5/3.0/30 did exactly that and was discarded.

**The exponents were picked from a measurement, and the first attempt was too
weak.** The frozen-weight smoke (job 7626106, the A0 epoch-17 weights, 104
paired (batch, rank) samples, `ABL_LOSS_DIAG=1`) gives an exactly paired
control, because the boost is a deterministic function of the logged per-slot
sigma and can be divided back out of the same samples:

| slot | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| mean sigma | 0.004 | 0.016 | 0.077 | 0.518 | 5.53 | 129 |
| control share of the weighted loss | 0.02% | 1.16% | 21.5% | 49.4% | 26.5% | **1.44%** |
| ratio at hn_power 1.1, clip 10 (first try) | 1.000 | 1.000 | 1.000 | 1.000 | 1.020 | **3.79** |
| ratio at hn_power 2.0, clip 30 (shipped) | 1.000 | 1.000 | 1.000 | 1.000 | 1.042 | **11.19** |
| shipped share | 0.02% | 1.05% | 19.5% | 45.0% | 25.3% | **13.9%** |

`hn_power = 1 - 1/rho = 1.1` is the exponent that restores EDM equivalence
asymptotically, but slot 6's WEIGHT MASS sits at sigma 16-60, where a
sigma^1.1 law is worth only 1.7-6x; the samples that reach the cap carry
almost no weight. So the power law, not the cap, was the binding constraint.
Exponent 2 with cap 30 lifts the slot 11.2x for a total-loss cost of x1.158 --
inside the x1.17 budget that keeps the learning rate untouched.

**A2-L's weighting was already nearly flat** — measured per-slot mean weight
0.481 / 0.530 / 0.409 / 0.231 / 0.122 / 0.037, i.e. 13x front to back against
ERDM's ~2700x. The severity of the A0 finding is therefore mostly an artifact
of ERDM's `rho = -10` schedule, not of rolling training as such, and the A0
knob values would have been nearly inert on A2-L (its slot-6 weight mass sits
at sigma_eff 3.5-10, right at the `f` peak). The two configs are tuned to the
same ACHIEVED back-slot reweighting so `a0hn` vs `a2lhn` stays controlled.

Note the two quantities that are easy to conflate: the **mean loss weight**
ratio (x8.14 / x8.52, computable from the configs and pinned by tests) and the
**share of the realized weighted loss** (measured x11.19 for A0-HN, 1.44% ->
13.9%), which also depends on how the model's error varies with sigma inside
the slot. The A2-L share is projected from its weight profile and will be
measured by the a2lhn smoke.

## 3. The dispersion question, and why it had never been answered

The validator's `spread` is the lat-weighted RMS of the biased
(`unbiased=False`) per-pixel member std in **normalized** units; its `rmse` is
in **physical** units for surface and upper air; and the one-year evaluations
run with `truth_source=obs_climatology`, so their stored `spread_skill` block
is spread over *climatological* RMSE. Every previous "RSI is under-dispersed"
statement compared absolute spreads between models whose errors differ by
2.3x:

| | spread d10 | RMSE d10 | spread per unit error |
|---|---|---|---|
| upstream ERDM ep17 (churn on) | 0.196 | 59 | 0.0033 |
| A0 e17 | 0.308 | 125 | 0.0025 |
| A2-L e22 | 0.106 | 25 | **0.0042** |

A2-L has the *least* absolute spread and the *most* spread per unit error. The
honest answer is that nobody knew, so commit 8 adds, per group and all in
normalized units: `nrmse` (the matching denominator), `ssr` (spread/skill,
corrected by `sqrt((E+1)/(E-1))` — the biased variance and the
calibrated-ensemble mean error compose, so `sqrt((E+1)/E)` alone is wrong;
verified against synthetic calibrated ensembles at E = 8/16), fair ensemble
`crps` (sorted-member identity, pinned against the naive O(E^2) estimator) and
a two-number rank summary (`rankout` vs the calibrated `2/(E+1)`,
`rankbias`). CRPS is the proper score and therefore the tie-breaker: `ssr` can
always be driven to 1 by adding noise that makes the forecast worse.

### The SDE that could not run, and now can

RSI's only stochasticity under the default sampler is one latent per fresh
slot (`fresh_noise_scale * gamma_0 * S_c`) plus the warm-up window, whose
latents are front-loaded to nearly zero at the slots emitted first. Its
eps-family SDE (rung A5) was marked "not runnable": the drift carried
`Gamma^-1 = 1/(gamma S_c)` — order 1e3 at the real per-channel scales — while
the noise term was white in state units, so drift and diffusion disagreed by
`Gamma^-2`. `eps_mode="gamma2"` sets `eps_c(tau) = eps_scale Gamma_c(tau)^2`,
diagonal in Gamma's own basis (proposal v0.2 sec 3.5 explicitly allows this),
giving drift `-eps_scale dtau Gamma zhat` and noise
`sqrt(2 eps_scale dtau) Gamma dW`: both bounded by Gamma and mutually
consistent. It is the structural analogue of ERDM's churn, which injects
noise proportional to each slot's own sigma and is never stiff.

Oracle toy, exact-score head, horizon 1200, 32 members, S_c = (1.0, 0.63,
0.32, 0.14):

| variant | non-finite at | internal variance ratio | member spread | alpha |
|---|---|---|---|---|
| eps 0.003 scalar | -- | .500 .461 .452 .443 | .708 .680 .671 .671 | ~0 |
| eps 0.03 scalar | **step 76** | nan | nan | nan |
| eps 0.5 scalar | **step 12** | nan | nan | nan |
| eps 2.0 scalar | **step 9** | nan | nan | nan |
| eps 0.03 gamma2 | -- | .501 .460 .451 .434 | .709 .679 .671 .664 | ~0 |
| eps 0.5 gamma2 | -- | .563 .499 .472 .463 | .751 .706 .686 .683 | ~0 |
| eps 2.0 gamma2 | -- | **.707 .602 .549 .539** | **.842 .775 .738 .738** | ~0 |

Stable at 60x the largest scalar value that survives, and a genuine spread
knob: dispersion rises monotonically toward the ERDM control's ~1.0 while the
five-year shrinkage alpha stays ~0 on every channel — it moves spread, not the
mean, as proposal v0.2's M3 claims. **Useful eps in gamma2 units is O(0.5-2),
not O(0.05)**, so the sweep ladder is {0, 0.1, 0.5, 1.0, 2.0}. The residual
gap at eps 2.0 is Layer A (the fresh slot's missing posterior variance), not
the sampler.

Also in this campaign: with the real `noise_scale_path` artifact (S_c = 0.02,
0.1, 0.4) and a zero-output network over 50 rolls, `gamma2` holds every
channel at its eps=0 amplitude (2.81 / 2.36 / 2.17) across eps 0.02-0.5 while
`scalar` runs the S_c=0.02 channel from 2.81 to 257.

### First real calibration numbers (A2-L epoch 22)

The metrics ran on the real model at their first epoch end (job 7626136, the
epoch-22 weights frozen, 4 ICs x 10 members). `crps < nrmse` at every lead, as
required, and `rankbias` is ~0.004-0.015 everywhere, so the ensemble is
unbiased. The spread-skill ratio, finally dimensionless:

| lead | ssr surface | ssr upper air | ssr diagnostic | rankout surface |
|---|---|---|---|---|
| day 1 | 0.726 | 0.710 | 0.741 | 0.260 |
| day 3 | 0.726 | 0.710 | 0.759 | 0.288 |
| day 6 | 0.873 | 0.904 | 0.854 | 0.259 |
| day 10 | **1.067** | **1.074** | 0.974 | 0.212 |

(1.0 = calibrated; `rankout` reference 2/(E+1) = 0.182.)

**This reframes the whole dispersion question.** A2-L is NOT globally
under-dispersed. It is ~27% under-dispersed at days 1-3 and essentially
calibrated -- marginally OVER-dispersed -- by day 10, with a mild excess of
envelope outliers throughout consistent with the short-lead deficit. Every
earlier "RSI is under-dispersed" statement rested on comparing absolute
spreads with ERDM at 2.3x different error; in matched units the deficit is
real but confined to short lead and is not the long-lead problem it was taken
for.

Consequence for the eps sweep: a GLOBAL eps increase is the wrong instrument,
because it would push day 10 further past 1.0 while fixing days 1-3. The
knobs to reach for are the ones that act early -- `eps_tmin`/`eps_tmax`
restricted to the high-tau (early-lead) part of the sweep, or the fresh-slot
latent -- and the sweep should be scored on the short-lead `ssr` with day-10
`ssr` and `crps` as the guard rails.

## 4. A silent hyperparameter bug

`train.py::_flatten_optimizer_cfg` forwarded `muon_lr_multiplier` and `betas`
but dropped `muon_momentum`, so every `++training.optimizer.muon_momentum=...`
in every batch-40 job script was a no-op. **All batch-40 chains — prod24_b40,
prod24_bundle_b40, the Phase-5 fine-tunes, A0, A2-L — trained at the package
default 0.95, including the ones labelled `bundle`, which intended 0.85.**
Both ends of the pipe were correct and unit-tested; only the middle link was
missing, which is why it survived. `ABL_MOMENTUM` now defaults to 0.95 for
both recipes, so the fix is a no-op for the in-flight chains.

## 5. Running the new series

```bash
# one qsub per run: the link submits its own successor once it owns the lock,
# so a 12-link run occupies ONE queued slot (the small queue caps queued+held
# per user, and the controls are using those slots)
qsub -v "ABL=a0hn,ABL_RECIPE=upstream,ABL_TARGET=24,ABL_CHAIN=11" \
    polaris_ablation_train_b40.pbs
qsub -v "ABL=a2lhn,ABL_RECIPE=bundle,ABL_TARGET=24,ABL_CHAIN=11" \
    polaris_ablation_train_b40.pbs
```

Gate before launching (1 node, debug, frozen weights, `ABL_LOSS_DIAG=1`):
per-slot ratios hn/control = 1.00 x4, ~1.03, ~3-4; gnorm within 1.5x of the
1.2e5 median; `valid-cal:` present with finite `ssr` and `crps < nrmse`.

Post-hoc eps sweep on the A2-L **control** (so the sweep is not confounded by
two changes), through the eval script's cfg suffixes, which drive the
whitelisted `sampler_overrides` merge:

```bash
EVAL_JOBS="a2l_eps0:checkpoints_a2l_sstpred_b40:24:a2l+a2l_eps1:...:a2l_eps1+..."
```

Decide on `ssr_step{6,10}` closest to 1 with no `crps` degradation,
`rankout` against `2/(E+1)`, `rmse_step10` within 5% of eps 0, and the
headline climate biases unchanged.

See also: [rsi-v0.2-lag-w-implementation](rsi-v0.2-lag-w-implementation.md),
[rsi-drift-diagnosis-report](rsi-drift-diagnosis-report.md) (addendum
2026-09-15), [rsi-a0-a1-batch40-plan](rsi-a0-a1-batch40-plan.md) (the A0
recipe reference and the momentum correction note).
