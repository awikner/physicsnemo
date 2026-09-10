<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI drift vs ERDM: diagnosis report (2026-09-08)

Companion to [rsi-drift-diagnosis-brief](rsi-drift-diagnosis-brief.md), which
holds the measurements. This document holds the diagnosis: what the
investigation established, what it corrected in the brief, what it found wrong
in the code and the formulation, and which tests on the real model come next.

**How it was produced.** A multi-agent review of the proposal
(`rolling_stochastic_interpolants_proposal.md.pdf`), `rsi.py`/`erdm.py`, the
drivers, the configs, the tests and the brief, run twice independently (14
audits total), plus a literature sweep, two local numerical probes that drive
the *real* `RSIScheduler`/`ERDMScheduler` code paths
(`tools/diagnostics/rsi_drift/`), and a three-lens adversarial verification
(independent derivation, execution against the real scheduler, consistency
with the measurements) of the twelve findings that survived triage. Verifiers
were instructed to refute; several did, and the corrections are folded in
below with the per-finding status. Every number was either computed against
the shipped scheduler on this machine or is quoted from the brief. Nothing was
run on the cluster; no checkpoint was available locally.

---

## Addendum, 2026-09-09: what the Polaris runs established

Section 4's tests were run on the epoch-24 checkpoint (plan, job scripts and
full tables: [rsi-drift-polaris-test-plan](rsi-drift-polaris-test-plan.md)).
Where the results below differ from the text of sections 0-7, the results
win.

- **Test 1 (teacher-forced readout):** the RSI anchor readout (slot 6,
  t = 0.5) passes 99.6% of the truth's anomaly amplitude on-manifold (ERDM's
  back-slot denoiser 99.1%); both heads are *above* the oracle Bayes floor
  for the fast channels. There is no on-manifold contraction of the quantity
  RSI copies forward, so the mis-scaled skip (A2) is a conditioning defect
  and not a drift mechanism on the real model. New measurement: a uniform
  offset added to the window is passed to the anchor readout at 0.66-0.86 by
  RSI for the slow channels (0.77 mean) but only at 0.05-0.6 by ERDM (0.48
  mean; 0.06 for surface pressure), because ERDM must re-derive the level
  from forcing and the cleaner slots every roll while RSI's input *is* the
  previous state. That is the formulation's restoring-force asymmetry,
  quantified.
- **Test 2 (cascade with anchor trace, 4 ICs, and the 8-member 300-day
  traces):** the anchor chain holds ~0.93 of truth to roll 28, then the fast
  channels' anchors collapse (0.78 at roll 28 -> 0.38 at roll 50 for
  S_c > 0.15) while their emitted frames are still intact; the emitted frames
  follow 10-15 rolls later; the slow channels (T, z, surface pressure) follow
  by roll 70-100 through the network's cross-channel response. The
  anchor/emitted ratio stays at 0.92-0.96 throughout (no compounding readout
  shrink). ERDM stays at 1.00 at every roll and channel.
- **Test 4 (multi-roll flush):** from a true window both models dissolve a
  perturbation into weather chaos within ~24 rolls (RSI 1.5x slower); from the
  free-run window at roll 30, ERDM erases a uniform offset by roll 18 while
  RSI still carries ~1/3 of it and ~45% of a pattern shrink at roll 36. The
  off-manifold loss of restoring force is real and RSI-specific.
- **Test 6 (spread on disk):** the real RSI is under-dispersed vs ERDM by
  only 12-19% at leads 5-20 (chaotic growth compensates the fresh-slot
  deficit), 0.81x at long lead for the surface group and 1.5x *over* for the
  upper air.
- **Test 7 (alpha vs S_c):** Spearman 0.78 across 81 channels (fast channels
  collapse most); the actual per-channel scale range is 0.017-0.40, not the
  0.017-0.9 assumed in sections 0-3 and in the toys.
- **Test 3 / 5 (inference-only variants, 300 days, 8 members):** fresh-slot
  inflation 1.2 / 1.45 / 1.8 closes the day-10 spread deficit (1.45 matches
  ERDM's diagnostic spread) and leaves the drift plateau unchanged (1.8
  delays the run-away by ~5 days); `final_denoise` and `num_steps 4` both
  *accelerate* the run-away (day-30 surface RMSE 402 and 383 vs 219). Layer A
  is the dispersion defect and nothing else. **Phase 3 (five-year runs at
  1.2 / 1.45 / 1.8, plus the shipped sampler re-run and a second seed):** the
  five-year shrinkage alpha is unchanged to within 0.006 on every channel
  (surface mean 0.718 -> 0.719 / 0.717 / 0.713) and the drift plateau to
  within 1%, while the day-10 spread rises monotonically toward ERDM's
  (0.110 -> 0.130 vs 0.136). Layer A is eliminated as a cause of the
  time-mean collapse with a reproducibility check (the re-run matches the
  original file to 3 decimals; seeds agree to 0.001).

- **Phase 4 (training-side):** fine-tuning from epoch 24 with the
  pattern-shrink anchor augmentation (`loss.anchor_shrink 0.3`, everything
  else the shipped recipe) roughly halves the one-year drift and then
  saturates. One epoch: z500 bias-map RMSE 1029 vs 2059, t2m 4.98 K vs 10.5 K,
  surface RMSE-vs-climatology plateau 397 vs 611 (ERDM 63); the shrinkage
  alpha falls from 0.68 to 0.20 (2m temperature), 0.75 to 0.43 (upper-air T),
  0.76 to 0.31 (geopotential), surface pressure's collapse disappears, the
  fast v-winds improve least (0.87 to 0.6-0.7). Four epochs: the plateau is
  flat at 336 (half the excess over ERDM removed) and the global-mean bias
  keeps improving (t2m -3.65 -> -0.99 K), but the pattern alpha bottomed
  after the first epoch and creeps back (t2m 0.20 -> 0.32, z 0.31 -> 0.41),
  so the bias-map RMSE is minimal at epoch 26 (z500 1030, t2m 4.81 K). A
  wider shrink range (0.4-1.0) gains nothing (plateau 345 vs 350 at equal
  epochs) and costs 10-day skill (validation step-10 surface RMSE 139 vs
  87). The anchor traces explain the ceiling: the fine-tuned head amplifies
  its anchor (anchor/emitted 0.88 vs 0.95) and holds the slow channels at
  0.5-0.7 of their amplitude instead of 0.3-0.4, but the fast tropospheric
  winds still collapse to 0.33-0.39 (shipped 0.27) with the onset moved only
  from roll 34 to 37-46. The augmentation undoes a uniform pattern shrink;
  the fast channels leave the manifold by decorrelating, which it does not
  cover. Side effect: stratospheric specific humidity is over-amplified
  5-70x by every fine-tuned head (exclude negligible-S_c channels from the
  augmentation before production use). Cost: step-1 surface RMSE 6-11 vs
  2.3. The fresh-slot inflation on top again changes the mean by nothing.
  Best pattern-shrink checkpoint: epoch 25-26; the next lever is
  self-generated anchors (section 5), not more shrink epochs.

- **Phase 5 (new base, 2026-09-09, in progress):** the batch-40 *bundle*
  production run (linear-scaled lr with warmup and cosine, Muon momentum
  0.85) reached epoch 20 with a training loss 11% below the previous run at
  equal epochs and better 1-6 day skill, and became the base for the
  remaining fine-tunes. Its one-year baseline drifts **harder** than the old
  base: surface RMSE-vs-climatology plateau 794 vs 611, t2m level bias
  -5.5 K vs -3.65 K, onset at roll 32 vs 34, surface pressure's pattern
  amplitude down to 0.46 (old base 0.71), with the same shrinkage alpha
  (0.7-0.85 on every channel) and, once more, no change from the fresh-slot
  inflation. A better-fit head trusts its anchor at least as much: the drift
  is a property of the formulation, not of a training recipe. Running from
  that base, each to epoch 22 and scored over one year against the chain's
  own epoch 22: the pattern-shrink augmentation alone; self-generated
  anchors (`pushforward_rolls 2`, the model's own sampler rolled with the
  Layer A injection so the last k slots are anchored on its y_hat chain);
  and both together with the stratospheric-humidity channels excluded from
  the shrink. **First result (one augmented epoch, one year, 8 members):**
  self-generated anchors alone take the surface RMSE-vs-climatology plateau
  from 792 to **152** (ERDM 63), i.e. 88% of the excess is gone; t2m
  bias-map RMSE 11.5 -> 1.6 K, z500 2094 -> 542; the shrinkage alpha of
  every slow channel is zero to within 0.03 and the winds are at 0.2-0.5;
  the anchor chain holds the fast winds above 0.7 of their amplitude for
  335 rolls instead of 32; the long-lead spread equals ERDM's; and the
  day-10 skill improves (58 vs 116) at no day-1 cost. The pattern shrink is
  inferior alone (416) and harmful on top (290, with the level bias back).
  This is remedy (b) at full strength: the head learns a restoring force
  from the actual off-manifold structure of its own anchor chain, which a
  uniform shrink cannot imitate. Remaining after one epoch: a -1.2 K level, a
  +3.4 m/s u250 level, wind alphas 0.2-0.5 and a slight over-amplification
  of tropospheric T and q (1.07-1.14). The unpaired sampler gives the same
  climate (plateau 141), so the injection is a spread knob only; the finished
  plain epoch 24 drifts exactly like epoch 20 (795); the shrink with an
  added level-offset augmentation recovers its level damage (t2m -6.1 ->
  -3.8 K) but stays at 422; a second epoch of pushforward + shrink is worse
  than the first (329 vs 290). Ranking by one-year plateau: pushforward 152
  << pushforward + shrink 290-329 < shrink + level 422 < shrink 455 << plain
  795, ERDM 63. **Two pushforward epochs** remove the residual level (z500
  +13 m2/s2, t2m -0.24 K; bias-map RMSE 0.72 K vs the control's 11.5 K and
  ERDM's 0.21 K) at the same plateau (159), and the day-10 skill improves
  again (54). After two epochs the head depends on the anchor law it was
  trained on: the plain sampler creeps late in the year (353 at day 365)
  where the paired one stays flat, so the inference `fresh_noise_scale` must
  match the training rolls. **Five years, however, show the one-epoch fix
  to be a delay:** the yearly surface plateau runs 143, 211, 419, 593, 606
  (control 709-783, ERDM 65) as the anchor chain, intact through year 1,
  collapses in years 2-3 to the base model's state; the shrink variants
  stay flat (450; 320 with pushforward) but biased. The head learned its
  restoring force from anchors one or two links off truth and has none for
  the states its free run reaches after hundreds of links. Two-epoch and
  finished-model five-year runs and a long-chain (k up to 6, Euler rolls)
  variant are queued; the complete remedy is to train on the model's own
  free-run windows as anchors. Full tables in the plan's Phase 5 section.

Net verdict after the runs: Layer A as stated; Layer B is the off-manifold,
no-restoring-force branch (brief H1 sharpened), entered when the fast
channels' anchor chains go off-manifold at roll ~28, and it is a property of
what the readout is asked to trust (the previous state) rather than of any
per-roll bias; the training-side remedies in section 5 are the ones that
address it. Pattern-shrink augmentation alone removes half of the drift and
plateaus; the fast-channel collapse that remains needs the network to see its
own off-manifold anchors -- and one epoch of exactly that (self-generated
anchors with the Layer A injection, Phase 5) removes 88% of the drift excess
and closes the dispersion gap at the same time.

---

## 0. Verdict

The RSI-vs-ERDM gap has **two layers**, and the brief's hypotheses conflated
them.

**Layer A: a proven formulation defect that collapses variance, not the mean.**
The fresh back slot is seeded from a *conditional mean* (the slot-6 H1 readout
at local time tau = 1/12) plus `Gamma(0) = 1.0 * S_c` of white noise, while
the training law seeds it from a *true sample* plus the same noise
(`rsi.py:1060-1073` vs `compute_loss`, `rsi.py:738`). By the law of total
variance the copied-forward anchor is short by its own posterior variance,
which at that readout is `(0.93-1.11 S_c)^2`, i.e. about one full increment
variance. Driving the shipped sampler with *exact Bayes-optimal heads* (six
independent implementations, one of them the probe in
`tools/diagnostics/rsi_drift/oracle_toy.py`), the emitted amplitude settles at
0.64-0.70 of truth for `S_c >= 0.1`, the anchor chain at 0.33-0.63, members
lock, and lag-1 autocorrelation is inflated; ERDM through the identical
harness holds 0.91-1.00. A small *trained* nonlinear toy (real `compute_loss`,
`learned_toy.py`) reproduces the same thing: free-running variance 0.28 of
truth, ensemble spread 0.54, ERDM 1.00/1.00. **But in both probes the
time-mean pattern and the seasonal cycle are preserved** (oracle: time-mean
error <= 0.014, seasonal amplitude 1.00 +- 0.02 over 1095 forced rolls; trained
toy: shrinkage alpha 0.027). A chain of conditional means is unbiased
(DC gain 1.0000 measured). So Layer A explains the under-dispersion, the
member locking and the "spread 0.005 -> 0.10" of brief section 2.2, and it is
fixable at inference by injecting the missing variance (total ~1.4-1.55 S_c
instead of 1.0 S_c restores 0.96-1.00 in the oracle). It does **not** explain
the headline: a ~70% collapse of the 5-year *time mean* toward each channel's
global mean.

**Layer B: the time-mean collapse is a property of the trained network off its
training manifold, not of the sampler algebra or of the objective's slot
weighting.** With exact heads *neither* sampler contracts the state level
(per-roll level gain 1.00002 emitted / 0.99980 anchor for RSI; the one-roll
map's spectral radius equals the process's own lag-1 autocorrelation to 4-6
digits), so any first-moment drift must come from the network, and it must
come with two properties at once: a level bias in the anchor-producing readout
*and* no restoring force to oppose it. The scalar-gamma EDM skip
(`rsi.py:534, :569, :622`; the latent std is `gamma * S_c` but every
coefficient is evaluated at the scalar `gamma`) sets the coupling coefficient:
at the anchor readout `c_skip = 0.631` for every channel, the raw head must
regenerate 37% of the absolute state (`F_1 = 1.560 y - 0.952 a - 0.794 S z`),
and a relative shortfall `e` in a *level-copying* head contracts the
copied-forward anchor by `0.41 e` per roll. But adversarial verification
showed this cannot carry the drift on its own: measured with Bayes-optimal
heads scaled by `1 - e`, the per-roll lever is 0.40 e for `S_c = 0.017`,
0.14 e at `S_c = 0.63`, and *negative* at `S_c = 1`, so it is largest for the
channels that collapse least (surface pressure, alpha 0.49) and null for the
ones that collapse most (precipitation, cloud, v, alpha 0.93); with the
oracle's own restoring force the closed-loop shrinkage is only `alpha ~ 0.5-3 e`
and does not compound; a uniform head shortfall damages the ERDM oracle
comparably, so it is not the RSI/ERDM discriminator; and no linear per-roll
gain reproduces the 25-roll latency in the trace. The objective is also not
blind to such a bias: in the units of the raw head the anchor slot is the
*most* heavily weighted slot (8.5x slot 1) and carries 22-28% of the loss; what
the raw channel sum does hide is slow-channel error, because residuals scale
as `S_c^2` (surface pressure's whole loss is ~0.5% of the total). What remains
RSI-specific is the *state* the head is asked about: after Layer A has
developed, the window carries anchors that are smoother, less dispersed and
more persistent than any training window (`anchor_noise = 0`, so the head never
saw one), and the architecture's state pathway is amplitude-blind in the
linear limit (SourceNorm RMS-normalises the state embedding per pixel; the
output head applies a non-affine LayerNorm), so nothing trained the sign of
its response to a uniformly damped state. The observed per-channel alpha
ordering is Layer A's own ordering (anchor-chain spread 0.33 at `S_c = 1` vs
0.63 at `S_c = 0.14` in the oracle), which is what one expects if the mean
collapse is downstream of the dispersion collapse. **Layer B is therefore the
brief's H1 in a sharper form** (a learned map with no restoring force once the
window is off-manifold, with the skip's 0.4-per-roll coupling for slow channels
as an amplifier), and it is not yet demonstrated on the real model: in the
small trained toy the anchor readout sat exactly at its Bayes floor on-manifold
(slope 0.978 vs 0.975 predicted) and the climatology held. Tests 1, 2 and 4
below measure it on the epoch-24 checkpoint in minutes to an hour each.

**How the two layers plausibly chain.** The drift trace is best fit by a
~25-roll latency followed by a constant ~5%/roll contraction (delayed
geometric, rms residual 13 Pa vs 54 Pa for any fixed factor from step 0 and
17 Pa for a logistic). Layer A develops over the first 20-30 rolls (its
relaxation time is `2/S_c^2` rolls for the mid-`S_c` channels), after which
the window is off-manifold in exactly the way described above; a head that is
at its Bayes floor on-manifold can lose its restoring force there, and the
anchor chain then integrates whatever bias it has. This is the only account on
the table that produces a latency, an `S_c`-ordered collapse, and a
plateau. It is testable: the per-roll anchor gain should be ~1 for
teacher-forced windows and drop once the windows are drawn from a free run
(Test 2), and a multi-roll flush from a damped window should not restore
(Test 4).

**Bottom line for planning.** Fix Layer A now (one sampler line, no
retraining) because it is proven and cheap, but do not expect it alone to
remove the time-mean bias. Run Tests 1, 2 and 4 before any retrain; they decide
whether the head has a level bias on-manifold (then the per-channel
preconditioning retrain and re-weighting are the fix), or only off-manifold
(then training must show the network damped windows: pattern-shrink or
self-generated anchors, plus the Layer A fix so the windows stop being
damped). Do **not** run the brief's H3 knob (`anchor_noise > 0` alone): both
probes show it lowers dispersion further and leaves the mean unchanged; at the
values the ablation config intended (0.1-0.5) it would read as a null.

---

## 1. Corrections to the brief

| brief location | brief says | correct | how established |
|---|---|---|---|
| 1.3, 5 | anchor readout at tau_W = 11/72 (gamma 0.61, c_skip 0.73); F_1 ~ 1.7 y - 1.2 a - 0.9 S z | **tau_W = 1/12** (gamma 0.7647, c_skip 0.6310, c_out 0.6075, c_in 0.794); required F_1 = 1.560 y - 0.952 a - 0.794 S z. Emitted slot tau_1 = 11/12 (c_skip 0.99727, not 0.998). Three head evaluations per roll at global t = 0, 0.5, 0.5; both readouts come from the last one (`rsi.py:1002` skips the corrector on the final step, `final_denoise` False). 11/72 is 11/12 divided by W and is never evaluated. | instrumented `sample_window` (7 independent agents, `learned_toy.py` L1 block) |
| 2.2 | 10-day spread 0.005 -> 0.10 is "expected for a PF-ODE sampler whose only randomness is the interpolant latent" | Half right for the wrong reason. The *shape* to day 10 is exactly what a non-contracting sampler gives: lead-1 spread is the front-slot residual `gamma(5/6) S_c = 0.068 S_c`, and the exact-oracle lead-10/lead-1 ratio is 20.7-23.5 vs the measured 20.8, i.e. +40%/roll growth. The symptom is where the spread *saturates*: 0.63-0.71 of climatological (Layer A), not 1.0. Day-10 spread is nearly blind to a head contraction; the signature is at lead >= 120. PF-ODE sampling is not the reason: the oracle ERDM PF-ODE saturates at 0.93-1.00. | oracle probe; two verifiers |
| 3.7 | members follow a "bit-identical path" apart from the latent | Wording: `_fresh_slot` and `impose_ocean` draw a fresh per-member latent every roll, so member paths differ by construction; what is bit-identical is the ODE given the latents. | RSI-12 verifiers |
| H6 | "f peaks at sigma_eff = e^2 ~ 7.4, i.e. tau ~ 0.10" | The log-normal's mode is `exp(P_mean - P_std^2) = 1.75`, and `omega = lambda f` peaks at tau = 0.786 (sigma_eff 0.10); the anchor readout at tau = 1/12 has omega 0.036 against a peak of 0.54. See A4 for why "starved" is still the wrong conclusion. | RSI-6 verifiers |
| 3.6 item 3 | seasonal cycle "almost gone" | Retained at ~1/3 amplitude with the correct phase: refitting `RSI^2 = C^2 + lam^2 ERDM^2` on the brief's own trace gives lam = 0.33 (upper air), 0.36 (diagnostics), r = 0.74/0.86. Same damping factor as the spatial pattern, i.e. one uniform contraction. | brief-critique audit, reproduced by two verifiers |
| 3.7 | "drift is deterministic, 8 members agree to <5%" | Relative-vs-absolute error: RSI member scatter (0.078 K) is 5x ERDM's (0.0155 K). The valid locking evidence is that ERDM's ensemble-mean RMSE falls toward its 1/sqrt(8) floor while RSI's never falls. | brief-critique audit |
| 3.6, H2 | onset "sharp", "0.99^100 = 0.37 matches a ~50-step transition" | The trace is a ~25-roll latency then a constant ~5%/roll contraction; no fixed factor from step 0 fits (7-15x gap between what step 28 allows and what step 85 needs). State-dependent feedback is *not* required, latency is. | RSI-10 derivation verifier |
| 3.6 | the surface-group trace | Is 99% surface pressure: `validate_diffusion.py:1281` averages per-channel *physical-unit* RMSEs. The upper-air trace is geopotential. Read both as single-channel traces. | RSI-4 verifier |
| 5 | "ERDM emits a sample, RSI a conditional mean" | The ERDM paper also emits the denoiser output (Alg. 2 l.13); the fork's `x_bar[:,0]` at sigma_min differs from it by ~0.2% of a std. The load-bearing asymmetry is what is *copied forward*: ERDM nothing (back slot redrawn at sigma_max), RSI a conditional mean at gamma 0.765. | literature audit |
| H4 | front-slot skip contraction "compounds" | Cannot compound: `x[:,0]` and `y_hat[:,0]` are both dropped by the shift (`rsi.py:1112-1113`); only `y_hat[:,-1]` persists. H4's mechanism is real but relocated to the anchor slot, where 1 - c_skip is 0.369, not 0.002. | all audits |
| H2 intervention "anchor = x[:, -1]" | proposed as a fix | Harmful: oracle over-disperses (1.29-1.43) and, being 5/6 stale, is the *only* intervention that damps the seasonal cycle (0.94/0.77/0.44/0.33) and inflates lag-1 autocorrelation; trained toy variance 0.255 vs 0.284 shipped. Drop it. | both probes |
| H3 | `anchor_noise` in {0.5, 1} as a cheap fix | Predicted to backfire. `perturb_anchor` is train-only (`rsi.py:495-508`, called only at `:738`); the inference anchor error is a *variance deficit* (smoother than truth), not white noise, so the perturbed-joint head shrinks *more*. Oracle spread 0.72 -> 0.56 -> 0.39 at sigma_a = 0/1/2; trained toy variance 0.28 -> 0.23 -> 0.13. Only a *matched* pair (train `anchor_noise = s` AND inject `s` at `_fresh_slot`) helps, and less than the plain injection. | both probes |
| H1 Test A | one-roll teacher-forced restoring probe | Non-discriminating: in the trained toy the one-roll restoring coefficient is ~0 for *both* models (RSI -0.06, ERDM -0.0004); ERDM's restoring force acts over the W rolls it takes to flush the window. Use the multi-roll flush (Test 4). | learned probe |
| 3.5, line 63 | normalized 0 == "the channel's global-and-time mean" | The vendored normalization mean `m0_c` is an *unweighted* grid mean (verified to 0.003-0.02 sigma against the brief's own 12 zonal bands), which sits 0.40-0.43 sigma below the area-weighted mean for the thermodynamic channels. The attractor `m*_c` is near `m0_c` for z500/ULWRFsfc/DLWRFsfc but not in general: the global-mean identity has the wrong sign for SHTFLsfc and u@10hPa and is off by >2x on 8 of 19 channels. Half the attractor offset is the normalization origin; the rest needs a separate explanation. | RSI-5 verifiers (3 lenses) |
| 6, L3 (coordinator's lead) | fresh slot missing a "5-day forecast-error variance" | Wrong magnitude. Slots 1-5 observe the intermediate frames, so the anchor is effectively a *one*-step conditional mean; missing variance is ~1.0-1.6 S_c^2 on the training manifold (posterior sd 0.93-1.11 S_c). The right inflation is 1.4-1.55x in std, not 2.2x; 2x already overshoots to 1.36-1.40. | 4 independent oracle implementations |

---

## 2. Confirmed errors and deviations

| id | kind | where | what | drift relevance | status |
|---|---|---|---|---|---|
| A1 | formulation | `rsi.py:1060-1073` `_fresh_slot` | Back slot seeded from a conditional mean + `Gamma(0) = 1.0 S_c`; training law seeds from a true sample. Not marginal-preserving even with exact heads. Docstring calls slot W "the resolved slot" (`:1063`); it is the *least* resolved of the six (posterior sd/S: 0.10, 0.12, 0.23, 0.38, 0.58, 1.11 for slots 1..6). | Second-moment: under-dispersion, member locking, over-persistence. Not the time mean. | **confirmed** (3 lenses; mechanism unanimous, "source of the drift" refuted) |
| A2 | code / proposal deviation | `rsi.py:534` (`c_in`), `:569` (`z_precond`), `:622-627` (H1 skip) | EDM coefficients use scalar `gamma(tau)`; the latent std is `gamma * S_c` (`gamma_apply`, `:445-452`). Proposal section 3.4 prescribes per-band `c_in = Var[x]^-1/2`, `c_out` from `Var(v*)`, skip "near tau = 0 the anchor is"; the `:610-621` comment's "unit-variance residual" is false at the anchor slot (`Var(F_1*)` 0.30-0.37). Invisible to the A1 parity tests (`reduce_to_erdm` is exactly `S_c = 1`). Consequences: (i) H1 skip: a level-copying head's relative shortfall `e` contracts the copied-forward anchor by 0.41 e per roll, but only for small `S_c` (0.14 e at `S_c` 0.63, reversed at 1.0), and only if there is no restoring force; (ii) latent skip: an *error amplifier*, per-roll level gain `1 - 0.078 S_c e` for a latent-head error `e`, zero bias at the optimum, 6-300x weaker than (i), and unable to reach surface pressure's rate (ceiling 0.13%/roll vs 0.5-5% observed); (iii) `c_in` under-scales slow channels by up to 1.41x (train/inference-consistent). The per-channel-correct skip `1/(1 + gamma^2 S_c^2)` equals the shipped one at `S_c = 1`, so the fix is a no-op for the fastest channels; and it is **not an inference-only patch**: swapping it into a head trained against the scalar skip flips the sign and gives per-roll level *growth* (1.009-1.012 at `S_c` 0.17-0.35). | Amplifier for a slow-channel level bias, conditional on H1; not the ordering-setter, not the RSI/ERDM discriminator. | **confirmed** code fact (3 lenses); drift leverage refuted as stated; severity medium |
| A3 | code | `rsi.py:495-508` vs `:1060` | `anchor_noise` is train-only (its only caller is `compute_loss`; a value in a sampler yaml is dead) and models the wrong mismatch: the inference anchor law is `a = lam y + w` with `lam = 1 - Var(y|x) < 1` (a *shrink* plus ~1 `S_c` of independent noise), while the implemented family is `a = y + s S_c z'` with total variance `> 1`; the two families meet only at `s = 0`. Its unit is `S_c`, not `Gamma_0` (so at `rsi.yaml`'s default `gamma_0 = 0.5` a value of 1 is 2x `Gamma_0`), and it ignores `self.spectral` (white and 2.8x too large under a spectral `Gamma`). Neither of the proposal's mitigations (anchor perturbation; self-generated-anchor fine-tune) is in the shipped run or the recipe. | Dispersion: monotonically worse (oracle spread 0.645 -> 0.601/0.512/0.362/0.271 at s = 0.5/1/2/3; at the ablation's intended 0.1-0.5 only -0.3 to -7%). First moment: neutral in the oracle, marginal in the trained toy (alpha 0.027 -> 0.042). | **confirmed** (2 lenses) |
| A4 | formulation | `rsi.py:677-716` `loss_weight`, `sigma_eff` | Per-slot weight mass `E_t[omega]` = 0.481/0.530/0.409/0.231/0.122/0.037 (shares 26.6/29.3/22.6/12.8/6.8/2.06%), a bump peaked at tau = 0.786 (not monotone); `omega(1/12) = 0.036`, `omega(0) = 7.8e-8`; `lambda c_out^2` ranges 0.20-1.00 (exactly 1 at tau = 1), a 4.9x deviation from EDM's identity. With an exact stub head, *all* the compounding level-gain leverage of a relative `F_1` shortfall sits in slot 6 (0.386/roll at tau = 1/12 plus 0.021 at tau = 0; every slot with tau >= 1/6 gives a one-time offset and zero compounding). **But "starved" is the wrong reading**: in the units the network regresses (`omega c_out^2`) slot 6 is the most heavily weighted slot (8.5x slot 1), it carries 22-28% of the actual loss, and the shortfall that would give the observed 5%/roll (e = 12%) would cost 2.8% of the loss, about 4 epochs of the observed decline. ERDM at its shipped `sigma_data = 1` gives its own back slot 0.28% of the weight (7x less than RSI), so slot weighting is not the RSI/ERDM discriminator. The genuine hiding place is the raw channel sum: residuals scale as `S_c^2`, so surface pressure's entire loss is ~0.5% of the total and a level bias there is invisible to the optimizer. The one real weight hole is the fresh slot's t = 0 evaluation (`c_skip = 0.5`, `omega = 7.8e-8`). | Enabler for slow-channel bias via channel weighting, not slot weighting. | **corrected** (1 of 2 lenses refuted the "starved/orthogonal" payload; numbers confirmed) |
| A5 | formulation | `rsi.py:927-936, :1013-1020` | eps-family step adds scalar-eps isotropic noise in state units against a score `-zhat/(gamma S_c)`: formally marginal-preserving, but the explicit step is stiff by `1/(gamma S_c)^2` and diverges for slow channels at `eps_scale` 0.02-0.5 even with an exact score (both probes). Ablation A5 as written is not runnable. | none for the evaluated run (eps 0) | confirmed by both probes |
| A6 | config | `conf/loss/rsi_a1.yaml`, `rsi_a3.yaml`, `rsi_a4.yaml` | Set `parameterization: residual` while inheriting `h1_precond: edm` from `rsi.yaml`; `RSIScheduler.__init__` (`rsi.py:195-200`) raises. The ERDM-parity control and the residual ablation cannot be instantiated as shipped. | none directly; the ladder never ran | **confirmed** locally (instantiation raises) |
| A7 | code | `train_diffusion.py:862` | `load_checkpoint` is called without `metadata_dict` and nothing calls `ModelEMA.load_state_dict`, although the shadow is saved in `metadata["ema"]` (`:1196`) and the comment at `:1008-1010` says a resumed stage "will hydrate it". Each 2-epoch chained link rebuilds the EMA inside its 6-epoch warmup (effective decay 1/7, 2/7), so the evaluated "EMA weights" are ~live weights. | none for the drift (ERDM control used a genuine EMA; RSI used live-ish weights, both fine) | **confirmed** by reading |
| A8 | code | `rollout.py:163-172, :479` | `_save_state` calls `eps_prev.cpu()`; RSI's `stream_init` returns `(x, None)`, so the streaming rollout driver crashes at its first state save for RSI. | none (5-year eval used another path); blocks Campaign B | confirmed by reading |
| A9 | tests / tooling | `test/diffusion/test_rsi_scheduler.py`; `tools/data/amip/make_rsi_delta_scales.py:112`; `rsi.py:422-443` | No test sets `noise_scale_path`, so every guard runs at `S_c = 1` where A2 vanishes; no test asserts a distributional property of a multi-roll rollout (the only assertion is `|x| < 50`); `sigma_c_sstpred153.pt` cannot be produced by any shipped tool (the script hard-codes 154 channels) and `_scale` checks only the channel *count*, so the artifact's channel order is unverified. | a wrong order would mis-scale Gamma per channel | confirmed by reading |
| A10 | config | `final_denoise: false` | Both readouts are a half-step stale: anchor at tau 1/12 instead of 1/6 (`1 - c_skip` 0.369 -> 0.255, weight 0.036 -> 0.084), emitted at 11/12 instead of 1 (cosmetic: 0.01% of a std). The carried window is bit-identical either way; only the readouts move. Measured effect of the flag: oracle amplitude +5-9% (anchor chain +9-22%); the compounding per-roll level exposure of a uniformly short head falls 14-21%, not 31% (the t = 0.5 evaluation still drives the final transport step); with a 10%-short head the toy's alpha falls 13% relative. Cost 3 -> 4 head evaluations per roll (+33%). | partial mitigation of Layer A; a modest reduction of Layer B's coupling; confounds the two if used as a test | **confirmed** with refined numbers (2 lenses) |
| A11 | proposal reasoning | proposal 3.2 (rsi_proposal.txt:343-361) | "Marginal consistency for the rolling shift is guaranteed provided the anchor entering is distributed as in training" is unsatisfiable by construction: the construction supplies `E[y|window]`, whose variance is deficient by the conditional variance. "Its residual noise is absorbed into Gamma_0" is short by ~1x S_c. The drift is not conditioned on the anchor (heads receive only `c_in x`, tau, forcings; `rsi.py:582-589`): Chen et al. 2024 condition `b(x, x_0)` on the current state explicitly and sample an SDE from it; Albergo et al. 2023 prove an unconditioned velocity under a data-dependent coupling transports only the *marginals*. No published rolling-diffusion model seeds the back slot from a state estimate (ERDM Alg. 2 and Rolling Diffusion both redraw pure noise). | root of Layer A | literature + formulation audits |
| A12 | brief | see section 1 | numerical and logical errors listed above | | |

Minor: `label()` clamps tau to >= 1e-3 while `c_in` uses `gamma(0) = 1.0`
(0.3% mislabel at the tau = 0 slot; train/inference-consistent; negligible).

Things checked and found **correct**: training and eval window/forcing/ocean
alignment for RSI (init rows t..t+W vs ERDM's t+1..t+W, emitted frame k scored
at row t+k+1 for both, lag-1 `c_grid_win`, own-time `ocean_win`);
`impose_ocean` anchor-time/own-time alignment (exact to machine precision);
the coefficient integrator (sum of beta increments 1.000000 over a slot's
life, Gamma increments telescope to `gamma_1 - gamma_0`); the Heun corrector's
`(x_euler, tau_next)` pairing (exact with exact heads); `beta_floor` never
engaged; the only model-config difference between `amip_rsi_sst_pred.yaml`
and `amip_erdm_sst_pred.yaml` is `num_output_heads: 2`; the latent head's
`Gamma^2` loss weight is the correct Jacobian (its `Gamma/(1-beta)`
amplification at the front slot cancels against `d beta`).

---

## 3. What the local probes showed

Both scripts live in `tools/diagnostics/rsi_drift/` (see its README for the
harness, validation and reproduction commands). Both drive the real
schedulers; the trained toy's rollout loop is asserted bit-exact against
`sample_rollout`.

### 3.1 Oracle-head Gaussian toy (`oracle_toy.py`)

Per-channel AR(1), stationary variance 1, rho = 0.5/0.8/0.95/0.99 (S_c =
1.00/0.63/0.32/0.14), seasonal forced mean in `c_grid`, exact Bayes-optimal
heads for the scheduler's own training joint (verified to 2e-15; MC
MSE/theory 0.999), horizon 3000, 64 members, 3 ICs. Ratios vs an independent
truth control.

| run | alpha_season | internal variance | member spread | anchor-chain spread |
|---|---|---|---|---|
| ERDM oracle (shipped `erdm_v2_nochurn`) | 0.00-0.03 | .94 .99 .95 .90 | .97 1.00 .98 .95 | n/a |
| **RSI shipped** | **0.00** | **.51 .45 .42 .41** | **.72 .67 .65 .64** | **.33 .52 .61 .63** |
| `final_denoise` True | 0.00 | .55 .51 .50 .51 | .74 .71 .70 .71 | .40 .58 .67 .71 |
| `num_steps` 6 | 0.00 | .56 .50 .48 .49 | .75 .71 .69 .70 | .38 .56 .66 .69 |
| anchor = `x[:,-1]` | **.07 .30 .59 .67** | 1.66 1.78 1.81 1.67 | 1.29 1.34 1.36 1.42 | .60 1.03 1.28 1.40 |
| anchor + 1 S z' (total sqrt(2) S) | 0.00 | 1.02 .88 .87 .82 | 1.01 .94 .93 .91 | .47 .73 .87 .90 |
| anchor + 2 S z' | 0.00 | 2.5 2.2 2.2 2.1 | 1.60 1.49 1.47 1.44 | |
| train `anchor_noise` 1 (oracle re-derived) | 0.00 | .32 .28 .27 .27 | .56 .53 .52 .52 | .22 .38 .48 .51 |
| `gamma_0` 2 / 0.5 | 0.00 | .73-.65 / .24-.20 | .86-.81 / .49-.45 | |
| head scale 0.9 (uniform 10% under-supply of F_1) | .07 .15 .29 .31 | .52 .43 .29 .12 | .72 .65 .54 .34 | |
| ERDM head scale 0.9 | .17 .27 .35 .18 | .91 .86 .67 .46 | | |
| `eps_scale` 0.5 | non-finite by step ~13 on every channel with S_c < 1 | | | |

Reading: a perfect network keeps the forced response and the time mean
exactly and still collapses variance and spread; the cause is the fresh slot
(the injection knob moves the plateau monotonically, readout/ODE knobs buy
3-8 pp); a *uniform* relative head shortfall damages ERDM comparably, so the
skip leverage is RSI-specific only through *what is copied forward* and
through whatever makes the RSI head biased when ERDM's is not.

### 3.2 Trained nonlinear toy (`learned_toy.py`)

1-D ring, N = 32, one channel, seasonally modulated spatial mean maintained
by cubic restoring + Laplacian + advection, increment/state std 0.31; one
0.46 M-parameter backbone trained with the real `RSIScheduler.compute_loss`
(shipped recipe) and with `ERDMScheduler.compute_loss`, same optimizer and
seed; 5000-step rollouts, 32 members.

| variant | alpha | pattern amp | var (deseas.) | seasonal amp | spread/sigma | bias RMS |
|---|---|---|---|---|---|---|
| ERDM | -0.003 | 1.003 | 1.005 | 0.970 | 1.003 | 0.011 |
| **RSI shipped** | **0.027** | 1.059 | **0.284** | 1.033 | **0.536** | 0.328 |
| `final_denoise` True | 0.028 | 1.007 | 0.494 | 1.021 | 0.705 | 0.206 |
| anchor = `x[:,-1]` | -0.096 | 1.154 | 0.255 | 0.834 | 0.513 | 0.294 |
| anchor + 2 S z' | 0.066 | 0.938 | 0.720 | 1.005 | 0.849 | 0.088 |
| anchor + 3 S z' | 0.144 | 0.863 | 0.898 | 0.957 | 0.948 | 0.146 |
| train `anchor_noise` 1 / 2 | 0.032 / 0.042 | | 0.234 / 0.131 | | 0.488 / 0.366 | 0.342 / 0.373 |
| `h1_precond none` (retrained) | 0.035 | 1.044 | 0.289 | 1.056 | 0.540 | |
| trained with S = 1 (gamma in state units) | 0.007 | 0.994 | 0.876 | 0.993 | 0.934 | 0.034 |
| fine-tune, pattern-shrink anchors (brief H1 Test B) | -0.132 | 1.159 | 0.956 | 0.921 | 0.989 | 0.222 |
| fine-tune, self-generated anchors | 0.005 | 1.089 | 0.847 | 1.132 | 0.928 | 0.479 |
| `eps_scale` 1.0 | -0.070 | 1.188 | 152.6 | 1.62 | 12.3 | 1.19 |

Also measured: healthy short-range skill for both (RSI step-1 RMSE 0.024 vs
ERDM 0.007; RSI better at leads 1-5); slot-6 teacher-forced readout slope
0.978 vs the Bayes floor 0.975 (no super-Bayes contraction at this scale);
the one-roll restoring coefficient is ~0 for both models, while a 48-roll
flush leaves 41% of a uniform bias in RSI above its own spread floor and 0% in
ERDM; the pattern-shrink fine-tune is the only variant with a positive
one-roll restoring coefficient (0.54 at the anchor). A "static forcing"
ablation (seasonal response must be learned from the calendar scalar) changes
nothing, so the toy gives no support to H7 at this scale. The toy network is
translation-equivariant and cannot store a climatology in its weights, unlike
RollingDiT with absolute position embeddings; that is the main structural
difference from the real model and the reason the "uniform forcing" ablation
is a broken control (both models fail, ERDM worse).

---

## 4. Recommended tests on the real model (Polaris), ranked by information per GPU-hour

All on the existing `checkpoints_prod24_b40` epoch-24 weights unless stated.
Predictions are given so each test discriminates.

1. **Teacher-forced per-slot, per-channel readout slope at global t = 0.5**
   (minutes). Build true W+1 windows over a season, place them on the
   interpolant at the t = 0.5 tau staircase (not t = 0), call `heads()`, and
   report per channel and slot: `std(y_hat_w)/std(y_w)`, the regression slope
   of `y_hat_w` on `y_w`, and the slope on `x_w`. Prediction if the head is at
   its Bayes floor (toy behaviour): slot 1 ~1.000; slot 6 ~`1 - 1.2 S_c^2`
   (0.9998 for surface pressure, ~0.95 at S_c 0.3, ~0.5 at 0.9), with the
   `alpha` ordering across channels tracking `S_c`. A slot-6 slope
   *materially below* that floor for the slow channels (e.g. < 0.99 for
   surface pressure) is Layer B measured directly and makes A2's fix the
   priority; a slope at the floor means the mean collapse is an off-manifold
   effect that Test 2 must catch.
2. **Multi-roll cascade with per-roll amplitude logging** (~1 GPU-hour, 8
   members, 120 rolls, true forcings). Log per channel and roll:
   `std(y_hat[:,-1])`, `std(y_hat[:,0])`, the per-slot window std, and the
   *per-roll gain* of the anchor (`std(anchor_k)/std(anchor_{k-1})`).
   Predictions: the anchor amplitude decays from roll 1 and leads the emitted
   frame by W - 1 = 5 rolls (toy: exactly); a gain ~1 for the first ~20 rolls
   that then drops to ~0.95 is the composite chain of section 0; a gain that
   is < 1 from roll 1 for slow channels is Layer B alone. This is the test
   whose per-channel onset ordering (fast channels first) would confirm that
   Layer A triggers Layer B.
3. **Fresh-slot variance injection at inference** (one line in `_fresh_slot`,
   no retraining; ~1-3 GPU-hours per setting for 300 days, then one 5-year
   run for the winner): `anchor + Gamma(0) z + c S_c z'` with c in {1.0,
   1.2, 2.0} (totals 1.41, 1.56, 2.24 S_c), or equivalently `gamma_0`
   1.0 -> 1.45 at inference only. Predictions: member spread rises toward the
   climatological sqrt(2) sigma and the 8 members unlock; daily variance about
   the seasonal cycle rises to ~1 at c ~ 1.0-1.2 and overshoots at 2.0;
   short-range RMSE degrades by roughly one increment std per frame. If the
   5-year `alpha` is unchanged, Layer A alone is not the drift (expected). If
   `alpha` drops substantially, variance-to-mean rectification is a real
   ingredient and the fix is largely done. Caveat: the fresh slot then enters
   with more noise than any training slot ever carried (`gamma_0 = 1` is the
   trained maximum, and the label/`c_in` still say tau = 0), so the t = 0 head
   evaluation is slightly off its trained range; the trained toy tolerated
   c = 2-3 with a modest skill cost, but log short-range RMSE.
4. **Multi-roll flush restoring-force probe** (replaces the brief's Test A;
   ~1 GPU-hour). Perturb the true IC window (anomaly pattern x 0.7; uniform
   -3 K on all temperature channels), keep true forcings, run 48 rolls for
   RSI and ERDM, and report `rms(perturbed - reference)/rms(perturbation)`
   against each model's own sqrt(2) x spread decorrelation floor. Prediction:
   ERDM flushes the perturbation within ~W rolls; RSI retains a large
   fraction above its own floor. Also run it from *free-run* windows at day
   40 (off-manifold) to measure whether the retained fraction grows.
5. **Config-only knobs** (5-year each, no retraining): `final_denoise: true`
   (+33% sampling cost); `num_steps` 4. Prediction: +5-9% amplitude, a ~15%
   relative drop in `alpha` at most, little else. A large `alpha` change would
   mean the readout lag matters far more in the real model than in the toys.
   Note the flag moves both the noise budget and the readout exposure, so it
   does not separate Layer A from Layer B.
6. **Read the dispersion that is already on disk** (no compute). Both 5-year
   `eval_suite.pt` files store `spread_step{S}_{surface,upper_air,diagnostic}`
   at every logged step to 1827 (`climate_eval_suite.py:1173-1176`,
   `validate_diffusion.py:1291-1293`), so the RSI-vs-ERDM spread-saturation
   comparison the brief never made costs a `torch.load`. Compare at leads >=
   20 (ERDM's noised init staircase makes it over-dispersed vs a perfect
   ensemble before that). Prediction: RSI saturates at ~0.63-0.71 of ERDM's
   level and stays there; the day-10 numbers alone will not discriminate. Then
   add to future evals the daily variance about each model's own seasonal
   cycle and the lag-1/lag-2 autocorrelation per channel (prediction: RSI
   variance 0.2-0.5 of truth and inflated persistence).
7. **alpha_c vs S_c** (no GPU): regress the brief's per-channel `alpha`
   (section 3.5) on `S_c` from `sigma_c_sstpred153.pt`. Prediction: an
   `S_c`-ordered term (Layer A) plus a channel-independent floor (Layer B).
   While there, **verify the channel order of `sigma_c_sstpred153.pt`**
   against the wrapper's pack order (A9): a permuted artifact would mis-scale
   `Gamma` per channel and nothing in the code would notice.
8. **Do not run** `anchor_noise` alone (H3), `anchor = x[:,-1]` (H2's
   intervention), or `eps_scale > 0` (A5) on the real model; all three are
   predicted to make things worse. If H3 is tested at all, pair the training
   `anchor_noise = s` with an inference injection of `s S_c z'`.
9. **Training-side, in order of expected value per compute** (short
   fine-tunes from epoch 24, then a 1-year rollout each): (a) the
   pattern-shrink anchor augmentation `a <- <a> + s (a - <a>)`, s ~ U(0.7, 1),
   applied in `perturb_anchor` (best per compute in the toy; the only variant
   with a positive one-roll restoring force; the exact inference anchor law
   is a shrink *plus* ~1 `S_c` of independent noise, so implement both legs)
   -- run on Polaris: halves the drift within one epoch and then plateaus,
   leaving the fast-wind collapse untouched (addendum, Phase 4);
   (b) self-generated-anchor fine-tune paired with the Layer A injection
   (alone it introduced a -0.33 global-mean bias in the toy) -- the next run,
   since it is the one that shows the network the decorrelated off-manifold
   anchors the pattern-shrink augmentation cannot imitate; (c) per-channel
   preconditioning (A2 fix: evaluate `c_in`, `z_precond` and the H1 skip at
   `gamma * S_c`; reduces to the current code at `S_c = 1`, so the A1 parity
   tests still pass) - a retrain, never an inference swap; prediction:
   anchor-slot skip -> ~0.99 for slow channels so a head bias there no longer
   compounds, no change for the fastest channels; (d) re-weight the raw
   channel sum so slow channels are not invisible (residual scaling per
   channel, as ACE does), and give the fresh slot's t = 0 evaluation a
   non-zero weight; (e) `gamma` in absolute state units (`S_c = 1`) - the
   toy's best-behaved RSI (alpha 0.007, variance 0.88) at a 2x cost in step-1
   RMSE, which reframes the 2026-08-21 "gamma in increment units" fix as the
   trade it is. The bundle-recipe retrain (H8) is predicted to remain
   under-dispersed; its `alpha` may move.

---

## 5. Candidate fixes, ranked

Inference-only (this week):
- Inject the missing fresh-slot variance: total ~1.4-1.55 S_c (Test 3); or,
  better matched, add the slot-6 residual latent `Gamma(tau_W) zhat` or a
  spectrally shaped draw from the one-step increment spectrum
  (`tools/data/amip/make_rsi_spectrum.py`), since the deficit is red and the
  injection is white (which is also why 8 latent realisations agree so
  closely).
- `final_denoise: true` (partial).
- Not: `anchor = x[:,-1]`, `eps_scale > 0`.

Training-time (a short fine-tune each, in this order):
- Anchor augmentation that mimics the *actual* inference error (pattern
  shrink plus ~1 `S_c` noise, or self-generated anchors), not white noise;
  this is the only lever that addresses "no restoring force off-manifold".
- Per-channel EDM coefficients (A2) - a retrain, not a swap; removes the
  slow-channel amplifier.
- Per-channel residual weighting of the loss so slow channels carry weight
  (A4); optionally a non-zero weight at the fresh slot's t = 0 evaluation.
- Pass the anchor (or the last emitted frame) to the backbone as an explicit
  conditioning input, as Chen et al. 2024 and ATLAS (arXiv:2601.18111) do;
  this is what makes the readout a *conditional* rather than
  anchor-marginalised mean.

Formulation (needs the proposal revisited):
- Seed the back slot from a *sample*: either resolve slot W ahead with frozen
  context, or draw from the posterior at the readout (the oracle's exact fix,
  0.96-1.00).
- Lag-w anchoring on the last emitted frame (every slot anchored on a
  resolved sample; `Gamma_w` in w-step increment units). Not tested here.
- Proposal option (B) in-window anchors. Not tested here.
- If a stochastic sampler is wanted, the diffusion must be `Gamma`-shaped
  (per-channel), not isotropic in state units (A5).

---

## 6. Hypotheses rejected or reframed

- **H2 (geometric compounding from step 0)**: wrong shape (latency + constant
  rate fits); the conditional-mean chain is real but second-moment only.
- **H4 (front-slot skip)**: cannot compound; relocated to the anchor slot as
  A2/Layer B.
- **H5 (state parameterization, beta_floor, Heun pairing)**: refuted;
  integrator exact, clamp never reached, `num_steps` 2/4/6 all collapse to
  0.62-0.75 in the oracle.
- **H3 (white anchor noise)**: lowers dispersion further in both probes and
  leaves the mean alone; at the intended 0.1-0.5 it is a null.
- **H6 (loss weighting starves the anchor end)**: the slot weights are as
  computed, but in raw-head units the anchor slot is the most heavily
  weighted and carries 22-28% of the loss; the brief's `f` mode is also
  wrong (1.75, not 7.4). The real weighting issue is the `S_c^2` channel sum.
- **"The mis-scaled skip explains the collapse"** (this investigation's own
  lead L2): the code fact stands, but the lever is `S_c`-inverted relative to
  the observed alpha ordering, does not compound against a restoring force,
  hits ERDM comparably, and cannot produce the latency. Kept as an amplifier
  for slow channels, conditional on H1.
- **"The 10-day spread trace measures the contraction"**: refuted; it matches
  a non-contracting oracle to a few percent. The deficit is the saturation
  level.
- **H8 (recipe/under-training)**: a well-fit toy RSI still collapses its
  variance; cannot be the cause of Layer A. Untested for Layer B.
- **H9 (ocean handling)**: alignment exact; the toy collapse arises with no
  ocean channels. Notably the ocean block is the *only* part of the window
  restored to the exact training law each roll, which is why imposing SST
  cannot arrest the state block.
- **"The attractor is the normalization origin"**: half right (m0 is an
  unweighted grid mean 0.4 sigma from the physical mean; z500 and the
  longwave fluxes hit it), refuted as an identity (sign failures on SHTFL and
  u@10hPa; 8/19 channels off by > 2x).
- **"alpha = 1 - phi pins the forcing gain"**: a reparameterization, not a
  measurement; alpha fixes only the ratio of a per-roll level leak to a
  forcing refill.
- **H1 / H7 (no restoring force; weak forcing pathway)**: not testable by an
  oracle, and not reproduced by the small trained toy (which holds the
  seasonal cycle even when it must be learned from a calendar scalar). They
  survive, sharpened into Layer B: a level-gain deficit at the anchor
  readout, with the amplitude-blind state pathway and the 2% loss weight as
  the reasons it could be RSI-specific.

---

## 7. Open questions

- Why a ~25-roll latency? The composite chain predicts it (Layer A's
  relaxation time `2/S_c^2` for the mid-S channels), but the trace that shows
  it is 99% surface pressure, whose own relaxation time under the oracle is
  ~7000 rolls; its collapse must be inherited through the network's
  cross-channel coupling. Test 2's per-channel onsets decide.
- Why would the RSI head be biased when the ERDM head, trained on the same
  data with the same architecture, is not? Candidates: the 2% loss weight at
  the anchor slot; the amplitude-blind SourceNorm/LayerNorm pathway; the
  off-manifold anchors of a free run. Test 1 (on-manifold) vs Test 2
  (free-run windows) separates these.
- The global-mean offsets that no contraction toward a constant reproduces
  (SHTFL sign, u@10hPa outside the observed range, the water budget with
  evaporation -37% vs precipitation -8%) need a physical or architectural
  explanation of their own.
- Whether the real `S_c` (measured on non-Gaussian, spatially structured
  fields) is even the right per-channel scale for `Gamma`; the toy's
  calibration is perfect by construction, the real one is untested (A9).

---

## Appendix A. Literature (primary sources checked)

- Chen, Goldstein, Hua, Albergo, Boffi, Vanden-Eijnden, *Probabilistic
  forecasting with stochastic interpolants and Follmer processes*, ICML 2024,
  https://arxiv.org/abs/2403.13724. Drift `b_s(x, x_0)` takes the current
  state as an explicit input (Eq. 6, 14); sampler is an SDE from the point
  mass at `x_0` (Eq. 16); Thm B.4 / Eq. 42: at the base end the readout
  degenerates to `E[x_1 | x_0]`; Table 2: after 100 autoregressive steps the
  deterministic map is 54x worse than the SDE on the invariant-measure
  enstrophy while its one-step error is only 2x worse. `beta_dot(0) = 0`
  recommended for Lipschitz control at the base end.
- Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden, *Stochastic
  interpolants with data-dependent couplings*, ICML 2024,
  https://arxiv.org/abs/2310.03725. Thm 3.1 transports marginals; Cor. 3.1 /
  A.1: the conditional law requires the conditioning variable as a velocity
  input.
- Ruhling Cachay et al., *Elucidated Rolling Diffusion Models*, NeurIPS 2025,
  https://arxiv.org/abs/2506.20024 (Alg. 2: back slot redrawn from
  `N(0, sigma_max^2)` every roll; emits `y_hat`; no ablation of a
  state-estimate back slot). Ruhe et al., *Rolling Diffusion Models*, ICML
  2024, https://arxiv.org/abs/2402.09470 (same).
- Kossaifi et al., *ATLAS*, https://arxiv.org/abs/2601.18111: SI with the
  current state as an explicit drift input, SDE sampling, 60 stable
  autoregressive steps; residual parameterization accumulated error faster
  than state (caution for ablation A3).
- Srikishan et al., *TRIE*, https://arxiv.org/abs/2607.00196; Pfister,
  Holzschuh, Thuerey, *StocBench*, https://arxiv.org/html/2608.22309:
  deterministic sampling acts as a spectral filter and under-estimates
  variability; stochastic sampling preserves invariant-measure structure.
- Subich et al., https://arxiv.org/abs/2501.19374: an MSE-optimal readout
  has amplitude `rho` x truth (regression to the mean per scale); Bonavita,
  GRL 2024, doi:10.1029/2023GL107377; Lang et al., AIFS-CRPS,
  https://arxiv.org/abs/2412.15832 (MSE-trained AR models blur, CRPS-trained
  ensembles do not).
- Chattopadhyay & Hassanzadeh, https://arxiv.org/abs/2304.07029: white noise
  added to training labels did not improve stability (consistent with H3
  backfiring); Watt-Meyer et al., ACE, https://arxiv.org/abs/2310.02074: a
  deterministic one-step-MSE emulator holds a 100-year climate, so a chain
  of conditional means is not sufficient for collapse by itself.
- Cai & Lin, SOFT, https://arxiv.org/abs/2607.21080: feeding the model its
  own outputs during training beats Gaussian-noise augmentation and rollout
  tuning (ordering relevant to Test 9a/9d).
- Karras et al., EDM, https://arxiv.org/abs/2206.00364 (App. B.6): `c_skip`
  minimises `c_out` "so that the errors of F are amplified as little as
  possible", derived from the noise std actually present (basis of A2).
- *Can AI weather models predict beyond two weeks?*,
  https://arxiv.org/abs/2605.30184: removing the absolute time embedding
  yields a stable state with no seasonality (relevant to H7).

Full notes with quotes: session scratchpad `literature/notes.md`.

## Appendix B. Artifacts

- Probe code: `tools/diagnostics/rsi_drift/{oracle_toy.py, learned_toy.py,
  README.md}` (this branch). The README carries the exact commands; the
  oracle toy runs on CPU in ~15 minutes, the trained toy needs ~45 minutes on
  one 8 GB GPU. All numbers in section 3 are regenerated by those commands
  (deterministic seeds).
- The agents' audit and verification scripts lived in the session scratchpad
  under `/tmp` and did not survive the session; the report keeps their
  results, not their code. Every audit claim that mattered was reproduced by
  at least two independent scripts, and the probe scripts above are the
  durable versions.
- Workflow journals with every agent's full return value (findings, verdicts,
  probe reports):
  `~/.claude/projects/-home-awikner-repos-physicsnemo/717427ea-b5f4-4521-8a19-78d7f01d7320/subagents/workflows/wf_db30210f-e3d/journal.jsonl`
  (audits, triage, probes, first verifier votes) and
  `~/.claude/projects/-home-awikner-repos-physicsnemo-rsi-diagnosis/717427ea-b5f4-4521-8a19-78d7f01d7320/subagents/workflows/wf_97753f5c-e47/journal.jsonl`
  (follow-up verification). Result rows carry an `agentId`; the matching
  `agent-<id>.jsonl` holds the prompt that names the finding and lens.
