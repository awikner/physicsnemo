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

So the back-slot-led onset is the generic Muon-1e-3 limit shared by every
loss; the raw-H1 excess is the only RSI-specific defect, and it trips 1300
batches before the generic limit. At the production lr (5e-5) ERDM is
empirically stable over its full multi-epoch production training, raw-H1 RSI
tripped at ~11.7k steps, and RSI-edm — matching ERDM's stability at matched
lr — is expected stable. Do not raise base lr to 1e-4 on this trunk for ANY
of these losses.

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
