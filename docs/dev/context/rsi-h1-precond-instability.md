<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI A2 training instability: raw H1 readout, root cause and fix

**Status (2026-08-22): root-caused, fixed (`loss.h1_precond: edm`), confirmed
— including an ERDM fairness control.** The raw H1 state readout (missing EDM
skip/out) is the RSI-specific defect: front-slot-led, trips 1300 batches
before anything else at matched lr. A second, back-slot-led onset seen at 2x
lr turned out to be the GENERIC stability limit of the trunk at Muon 1e-3 —
ERDM itself trips at the same place (see "Second driver" below) — so with the
fix, RSI matches ERDM's stability exactly at matched lr. The first two RSI A2
production runs (Midway jobs
53918582 and 54173034, `model=amip_rsi_fancy loss=rsi`, 4xH100, Muon lr 5e-5)
both destabilized at ~11.7k optimizer steps — loss trained cleanly
1.8e5 → 1.3e3, then ran away exponentially (e-fold ~93 batches). Gradient
clipping (added after the first run NaN'd) contained the magnitude but the
model never returned to the healthy basin; the second run ground through two
more epochs in an elevated state and hit NaN at e3 b10464, where the
non-finite-loss guard aborted it.

## Root cause

The RSI proposal (sec 3.4) prescribes EDM-style preconditioning — `c_in`,
`c_out`, and a `c_skip` path — for the learned heads. The implementation
applied it to the latent head only (`z_precond`); **the H1 head was read out
raw**. Under the A2 state parameterization that makes the network's bare
output regress the clean state `y` at every tau: at the front slots this is a
near-identity copy of a 154-channel field the backbone must realize to
~gamma_1 = 0.02 precision through RMSNorm'd DiT layers, weighted by
`lambda*f(sigma_eff)`.

ERDM — the stable baseline under the identical trainer/optimizer/lr/data —
never asks its network for this: `D = c_skip x + c_out F` carries the identity
outside the network, and the EDM identity `lambda * c_out^2 == 1` caps the
trained head's output-space curvature at exactly `f(sigma) <= ~0.07`. The raw
H1 head has curvature `lambda*f` with no `c_out^2` damping:

| slot (t=0.5)          | RSI h1 raw | ERDM F  | ratio    |
|-----------------------|-----------|---------|----------|
| w=1 (front)           | 3.2e-1    | 1.7e-7  | ~1.8e6x  |
| w=3                   | 5.3e-1    | 2.3e-3  | ~230x    |
| w=6 (back)            | 7.0e-2    | 6.4e-4  | ~110x    |
| integrated over t, sum over slots | 1.94 | 0.12 | **16x** |

Mechanism of the delayed onset: the output-space stiffness is constant, but
weight-space sharpness along the state-copy direction grows as the fit
improves (progressive sharpening). When sharpness x lr crosses the stability
threshold the run enters edge-of-stability oscillation and diverges. This is
why the run trains cleanly for thousands of steps and then blows up right at
the loss floor, and why clipping cannot rescue it — the basin requires the
high-precision copy that the oscillation destroys.

## Evidence (all reproduced on Midway3, 2026-08-22)

1. **Analytic conditioning table** (above): raw H1 output-space curvature is
   16x ERDM's whole loss integrated over training; ~2e6x at the front slot.
2. **Per-term gradient decomposition, scratch init** (real config/batch, same
   weights/latents both arms; the z rows byte-match as the internal control):
   raw-H1 head gradients are 3,000–10,000x the EDM-readout version at slot 1,
   ~400x at slot 3; with `h1_precond=edm` the H1 gradients drop into the same
   band as the z-head's.
3. **Per-term decomposition at the diverged run's e1 checkpoint** (elevated
   state, quarantined in `checkpoints_diverged_54173034/` on Midway): the h1
   terms carry ~98% of the gradient power (up to 1.3e12 per term, through the
   trunk), z terms ~50x lower.
4. **Causal A/B at 2x lr (1e-4), everything else identical to production:**
   - A-arm (raw H1, job 54274459): onset at **b5795 — half of b11671** — after
     reaching the same ~1.4e3 loss floor in half the steps. Onset therefore
     tracks optimization progress, NOT batch index: the "same bad data at
     ~b11.7k" hypothesis is refuted (those batches passed cleanly earlier).
     The live per-slot loss diag (`RSI_LOSS_DIAG`) shows the h1 terms leading
     the blow-up, front slots inflating 18–30x while z terms trail 10–20x
     smaller.
   - B-arm (`h1_precond=edm`, job 54285356): same 2x lr. Starts ~12x lower at
     init (zero-init means "contract toward the state", not "predict the zero
     field") and survives 22% longer — but destabilizes too, at b7092, with a
     DIFFERENT signature: back/mid-slot-led (slots 4–6 inflate ~10x in both h1
     and z terms while slot 1 barely moves), vs A's front-slot-led blow-up
     (slot 1 inflating 30x). The front-slot pathway is gone; a second driver
     trips later.

## Second driver: the generic 2x-lr limit (RESOLVED by the ERDM control)

With the H1 fix in, the 2x-lr B-arm still left the basin at b7092 (~1600
batches past its ~1.2e3 floor), led by the back/mid window slots — initially
ambiguous between (a) 2x lr being past the trunk's stability limit for
anything (Muon hidden-weight lr is 10x base, so 2x base = Muon 1e-3) and (b)
RSI-specific residual stiffness. The control settled it: **ERDM_fancy at the
identical 2x lr destabilizes at b7145** (`mw_ctrl_erdm.sbatch`, job 54297413)
— the same place as RSI-edm within ~1%. Three-way onset at matched 2x lr:

| arm                     | onset  |
|-------------------------|--------|
| RSI raw H1 (54274459)   | b5795  |
| RSI h1_precond=edm (54285356) | b7092 |
| ERDM baseline (54297413)| b7145  |

So the back-slot-led onset is the generic sharp-at-the-floor limit shared at
2x lr by every loss; the raw-H1 excess is the only RSI-specific defect of the
HEAD, and it trips 1300 batches before the generic limit. Do not raise base
lr to 1e-4 on this trunk for ANY of these losses.

**2026-08-22 evening update — the "expected stable at 1x" extrapolation was
WRONG.** The edm production run at 5e-5 (job 54309737) reached its floor 4x
faster than raw (b~3000 vs b~11k — the readout no longer has to be learned
through the trunk) and then destabilized b3968, back-slot-led, as a SLOW
runaway (gnorm e-fold ~500 batches vs ~30-90 at 2x); no recovery over 800+
batches. Unifying invariant across all six runs: onset follows the RSI loss
floor by 300–1600 batches at every tested (lr, head) combination, while ERDM
at 5e-5 sits at its own floor indefinitely. I.e. the RSI objective's floor
region is intrinsically sharper than ERDM's — most plausibly because the
coupled interpolant makes the back/mid slots genuinely learnable and RSI's
near-flat lambda*f trains them ~100x harder than ERDM's bump.

**The lr axis is exhausted.** The 2.5e-5 arm (job 54317233) also tripped,
onset b9145, back-slot-led, same slow runaway — halving lr delays onset
(~2.3x) but does not stabilize. Consistent with the optimizers being
scale-free: Muon's orthogonalized update has CONSTANT magnitude regardless
of gradient scale (and Adam normalizes), so global-norm gradient clipping
cannot modulate step sizes here either — which is why clip=1e6 "contained
but never recovered" in run 54173034. Once the floor region's progressive
sharpening crosses the threshold for a given lr, divergence follows.

**Second objective-vs-proposal deviation, now the active fix (delta_std).**
The proposal defines the weighting SNR as beta^2 Var[Delta]/Gamma^2 with
Var[Delta] the INCREMENT variance; the shipped `delta_std: 1.0` uses state
variance, overstating the signal scale ~8-10x. Measured on
amip_dailyavg_coarse (1990, all 136 state channels, plain-zarr probe
`probes/measure_delta_std.py`): normalized 1-step increment std mean 0.117,
median 0.089, range 0.017-0.30. At the corrected `delta_std: 0.12` the
per-slot weight lambda*f becomes [0.43, 0.27, 0.16, 0.10, 0.045, 0.0034]
(front-loaded, ERDM-like emphasis on the demixing region) instead of the
shipped near-flat [0.32, 0.49, 0.53, 0.37, 0.18, 0.070] — the back slot
that leads every post-fix blow-up is de-weighted 20x and the total
network-facing curvature drops 4.3x. Result (job 54331263, lr 5e-5,
h1_precond=edm + delta_std=0.12): qualitatively better but still not
converging — the run reached a LOWER floor (6.3e2 at b4499, vs ~1.2e3
before), then entered a LIMIT CYCLE instead of a one-way runaway: excursion
to gnorm ~7.6e7 at b4859, self-recovery to gnorm ~1e4, partial descent,
second excursion at b6749, recovery again — bouncing between loss 2.4e3 and
5e3 every ~1.5-2k batches, never returning to its floor. Mid-slot-led now
(the sharpening follows the weight mass). The ocean sub-loss spikes
disproportionately (~13x) at each excursion.

**Interpretation.** RSI's objective has far more REDUCIBLE signal than
ERDM's (the coupling's whole point), so weight-space sharpness keeps growing
as the model fits; Muon's orthogonalized update has CONSTANT magnitude — no
adaptive shrinkage — so the run gets pinned at the edge-of-stability
boundary and oscillates there, destroying its best fit each cycle. The
"floors" every run reached are EoS-limited losses, not the objective's true
floor. Both remaining knobs shrink the trunk step where the blow-ups live
(probe: runaway gradients flow through the Muon-governed trunk; the AdamW
groups self-stabilize): base lr (global, slow) or `muon_lr_multiplier`
(surgical; upstream's 10x is a transplant constant, not measured). The
multiplier is now plumbed through `training.optimizer.muon_lr_multiplier`
(train.py `_flatten_optimizer_cfg` -> train_loop.py `_make_muon_optimizer`).

**Knob ladder results (all at base lr 5e-5 unless noted).** Each fix slows
the sharpening — floor residence before onset grows monotonically — but
none alone stops it:

| arm | onset | floor residence |
|---|---|---|
| raw H1 (jobs 53918582/54173034) | b11.7k | ~700 |
| + h1_precond=edm (54309737) | b3968 (floor 4x sooner) | ~1000 |
| + delta_std 0.12 (54331263) | b4627, LIMIT CYCLE (self-recovers) | ~1500 |
| edm+ds012 + muon mult 3 (54342683) | b9094 | ~2x delay |
| + noise_scale_path (incr units, 54356351) | b~6900 | ~2400 |

The channel-resolved diag (RSI_LOSS_DIAG) shows the excursions are
CHANNEL-WANDERING: healthy top-5 is stably the cloud/precip diagnostics;
at onset v@900 spikes 30x, three minutes later the spike has moved to
2m_temperature — whichever weight-space direction is currently sharpest
oscillates first. No specific term is the root; it is edge-of-stability
dynamics of a reducible-signal-rich objective under a constant-step
optimizer.

**Increment-units configuration (the plan's actual Phase-A default,
previously never run).** `noise_scale_path` was null in the shipped A2 —
every channel got 0.5-sigma WHITE corruption, 30x the natural one-step
variability of slow channels (surface_pressure 0.017) — a huge pool of
perfectly-reducible artificial-denoising signal, i.e. sharpening fuel. The
artifact is now built per-channel from the store
(`probes/build_delta_scale.py` -> `norm_stats/sigma_c_fancy154.pt` on
Midway; surface_pressure 0.017, v@900 0.29, PRATE 0.37, monthly-interp ocean
floored at 0.01). With it (gamma_0=1.0, gamma_1=0.04 in increment units,
delta_std back to 1.0): lowest healthy gnorm of any arm (4-6e3), ocean loss
30x lower, longest floor residence — but still onset at ~b6900 at
multiplier 10, and at b10747 with muon_lr_multiplier 3 stacked (job
54372888, 6k-batch floor residence).

**The schedule test settles the mechanism (job 54390053).** A new
`CosineToFloor` scheduler option in `make_scheduler` (cosine 5e-5 -> 1e-5
over 15k steps, then hold; LambdaLR, monotone, group-ratio-preserving) on
top of the full stack: the run sailed through the ENTIRE decay phase —
floor ~540-650 from b11k, clean epoch-1 checkpoint SAVED (loss 559; in
`checkpoints_fullstack/` on Midway — the first clean trained RSI
checkpoint) — and tripped exactly 1.6k steps AFTER the lr froze at its
floor (onset global step 16626 = e2 b3483), even at trunk lr 3e-5, 17x
below baseline. I.e. while lr fell the stability threshold stayed ahead;
the moment it went constant, sharpening caught up. The sharpening does not
saturate within any practical fixed-lr range: **Muon (constant-magnitude
updates) cannot take this objective to convergence at any floored lr.** The
options are perpetual ~1/t decay (training crawls) or an adaptive trunk
(AdamW).

**AdamW is not immune either (job 54423079).** Identical objective stack,
AdamW at 5e-5: floor ~505-527 (healthy gnorm 1-2.5e3, 5-10x below any Muon
arm; clean e1 checkpoint in `checkpoints_adamw/`), ~4-5k steps of floor
residence, then excursions from global step ~15k — gentler (spikes absorbed
in tens of batches, partial recoveries) but the loss still bounces 10x off
its floor. The invariant across Muon-frozen, Muon-decaying and AdamW: **~4-6k
steps of floor residence, then excursions, at any constant lr.**

**Governor + rewind (jobs 54457716 v1, 54495702 v2) — helped, then
death-spiraled.** v1 (trigger 8x, EMA merely clipped) fired 1.5k batches
late because the band chased the excursion. v2 (trigger 4x, band FROZEN
outside 2x, RewindBuffer restoring model+optimizer to a healthy snapshot)
produced clean e1/e2 checkpoints and rode to step 30k without a 1e7 runaway
— but with a new failure mode: the first rewind consumes the snapshots, a
recurring excursion then triggers with nothing to restore, the frozen band
sits far below the elevated gnorms, and the trigger re-fires every cooldown
— 8 drops ratcheted lr to 5.8e-7 with the state stranded ~10x above its
floor. A v3 would need (i) a permanent "golden" best-loss snapshot so rewind
never runs dry, and (ii) a re-arm condition (no further drops until the
state has actually returned to band). Not built — see the production recipe
below for why.

## Production recipe (current): decay-and-harvest

What is measured to WORK: the run is clean while the lr is decaying and for
a while at the floor, every excursion onset is chaotic (identical configs
tripped at b12.4k and b16.6k), and no gradient-triggered mechanism preserves
the basin once an excursion is underway. So the practical A2 recipe is:

    train with CosineToFloor (5e-5 -> 1e-5 over 15k steps), harvest the
    per-epoch checkpoints, stop at the first excursion (the non-finite
    guard + watcher make it visible); the last pre-excursion checkpoint is
    the model.

Clean checkpoints in hand on Midway (all fancy-contract, edm + increment
units): `checkpoints_fullstack/` e1 (loss 559, Muon+decay),
`checkpoints_adamw/` e1 (loss ~527, AdamW), `checkpoints_gov2/` e1
(+ e2, post-excursion — check before use). Whether e1-scale training passes
the rung gates is exactly what the climatology/bias eval decides
(`outputs/rsi_a2_eval/fullstack_e1_eval_suite.pt`, save-as-you-go).

**Open question (documented, deliberately not pursued further now):** why
the RSI objective's sharpening does not saturate — candidate: the coupled
anchor makes the back/mid-slot task near-deterministic and the model keeps
extracting fit (the design goal!), so curvature grows with fit quality
indefinitely. If the ladder later needs longer training, governor v3 or a
persistent ~1/t decay are the leads.

## The fix

`RSIScheduler(h1_precond="edm")` — H1 read out as
`y_hat = c_skip(tau) x + c_out(tau) F1` with EDM's coefficients at
`gamma(tau)`; now the default in `conf/loss/rsi.yaml`. State-parameterization
only (residual H1 has increment scale and its own story — A3). Under
`reduce_to_erdm` this is exactly ERDM's D readout, so the A1 reduction holds
for the H1 head at zero-init too. Bonus: an ERDM warm start becomes
semantically meaningful for the head (it was trained to emit F, not raw
states).

Guardrails that landed with the diagnosis: run-scoped `++checkpoint_dir` for
every launcher on Midway (the shared default `checkpoints/` silently resumed a
contaminated run once), and the `RSI_LOSS_DIAG=<N>` env knob for per-slot
h1/z loss logging.

## Related, deliberately not changed here

- `sigma_eff` uses `delta_std=1.0` while the true one-step increment std is
  ~0.1–0.3 z-scored — the loss bump sits at a different effective SNR than the
  proposal intended. A tuning knob, not a correctness bug.
- Fresh-slot anchor mismatch at `anchor_noise=0` (train anchors are true
  states; inference anchors the entering slot on a blurry conditional mean).
  The proposal's shift-consistency argument assumes anchor perturbation is on;
  revisit at A3.
- `z_precond`/`c_in` ignore the per-channel scale and spectral shape when
  `noise_scale_path`/`spectrum_path` are set — inert in A2, revisit at A4.
- `label_mode=tau` feeds a warm-started TimestepEmbedder a different domain
  than it trained on (`log_sigma_eff` matches better).
