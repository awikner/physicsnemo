<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI ablation ladder: which rungs have actually been trained (2026-09-10)

The proposal (section 5) defines the ladder as: **A0** ERDM baseline,
retrained in our harness; **A1** RSI in the uncoupled white-noise limit, a
consistency check that must match A0 within noise and "the fallback operating
point" if the coupled model drifts; **A2** + data coupling (anchor y_{w-1},
scalar white Gamma), the core hypothesis; **A3** + residual parameterization
and increment-scaled preconditioning; **A4** + spectral g(kappa) and release
profile h(tau, kappa); **A5** the eps(tau) calibration sweep on the best of
A2-A4. Secondary ablations: Gamma_0 magnitude, anchor-perturbation strength,
Gamma_1 = 0, in-window anchors, window size, steps per pass; mitigations for
anchor mismatch under rollout: the Gamma_0 / perturbation ablation and a
self-generated-anchor fine-tune.

## Mapping of every trained or evaluated run to the ladder

| rung | fancy contract (154 ch, SST anomaly channel + global-mean-SST scalar) | sst_pred contract (153 ch, ocean tail predicted, no anomaly channel) |
|---|---|---|
| **A0** ERDM | `ERDM_fancy_42_2026-08-10` -- trained UPSTREAM (amip_v2 Lightning, max_epochs 50, S_churn 1, S_noise 1.5), translated to `amip-checkpoints/translated/erdm_fancy.mdlus`; evaluated at epoch 15. In-harness ERDM training exists only as the campaign-1 stability control at 2x lr (Midway job 54297413, `mw_ctrl_erdm.sbatch`), never run to a baseline. | `erdm_sstpred_ep17..24.mdlus` -- trained UPSTREAM (Lightning, `erdm_sstpred_raw/model_epoch=N.ckpt`, batch 4), translated 2026-09-07; ep24 is the control of the drift diagnosis, ep23 the campaign-2 target. **No ERDM has been trained in this repo's harness on either contract.** |
| **A1** reduce_to_erdm | never trained | never trained |
| **A2** vanilla coupled RSI | campaign 1 (Midway 4xH100, `model=amip_rsi_fancy loss=rsi`): jobs 53918582 / 54173034 (raw H1, diverged at ~11.7k steps), 54274459 / 54285356 (2x-lr arms), 54309737 / 54317233 (edm readout, lr arms), 54331263 (measured delta_std), 54390053 -> `checkpoints_fullstack/` e1 (first clean checkpoint, loss 559), 54423079 -> `checkpoints_adamw/` e1, `checkpoints_gov2/` e1. Evaluated: `outputs/rsi_a2_eval/fullstack_e1_eval_suite.pt` (sampler `rsi_fullstack_e1.yaml`). All harvested at ~1.2 epochs (decay-and-harvest). | `rsi_sstpred_a2` retrain 2026-08-30 -> `checkpoints_sstpred/` e1 (clean harvest, evaluated; e2-e4 post-excursion); campaign-2 arms (fp32 fix, 30k-step probes, DeltaAI/Midway); Polaris `checkpoints_prod24` (1 node, batch 4, capacity queue, epoch 8 and running), `checkpoints_prod24_b40` (24 epochs, batch 40, StepLR -- "the shipped model" of the drift diagnosis), `checkpoints_prod24_bundle_b40` (24 epochs, batch 40, cosine), plus `lrtest_b40`, `bundle_b40`, `b40_probe`, `polaris_validate`, `resume_rehearsal`, `mnprobe` (recipe tests). |
| **A3** residual param. (+ anchor_noise sub-rung) | never trained | never trained. `anchor_noise` was never run (the drift report predicts it worsens dispersion). |
| **A4** spectral Gamma | never trained; no `g_l_*.pt` envelope artifact exists under `norm_stats/` (only `sigma_c_fancy154.pt`, `sigma_c_sstpred153.pt`) | never trained |
| **A5** eps sweep | `eval_rsi_amip.sbatch RSI_EVAL=calibration` exists; no record of it having been run on a real checkpoint | not run (the drift report's toy found `eps_scale 0.5` non-finite on channels with S_c < 1, and the Phase 2 sweep deliberately left eps at 0) |
| secondary / mitigations | -- | Gamma_0-magnitude at inference: `fresh_noise_scale` 1.2 / 1.45 / 1.8 (Phase 2-3). Anchor-perturbation strength: `anchor_shrink` 0.3 / 0.6 (Phase 4-5), `anchor_level_noise` (Phase 5). Self-generated-anchor fine-tune: `pushforward_rolls` 2 (Phase 5, P21/P22), K = 6 queued then killed. Steps per pass: `num_steps 4` and `final_denoise` (Phase 2). |

## Three qualifications on "A2"

1. **The trained A2 is not the proposal's plain A2.** Every clean run carries
   the campaign-1 amendments: `h1_precond: edm` (the EDM skip/out readout of
   the state head, which the proposal's section 3.4 prescribes and the
   original code omitted) and `noise_scale_path` with `gamma_0 1.0, gamma_1
   0.04` in per-channel one-day-increment units. The second item is the
   "increment-scaled preconditioning" the proposal lists under **A3**, so the
   runs sit between A2 and A3: A2's state parameterization and spatially white
   noise, with A3's per-channel noise scale and without A3's residual head.
2. **A2 was never compared with an in-harness A0.** Both ERDM controls were
   trained upstream in Lightning. Campaign 2's arm 0 proved the training-side
   forcing alignment bit-exact and the eval drive bit-identical to upstream's
   rollout, and the ERDM stability control at 2x lr was trained in-harness,
   but no in-harness ERDM has been taken to a climate. The A2-vs-A0 gap
   measured in the drift diagnosis therefore still contains any residual
   harness difference; A1 is the rung designed to remove that ambiguity.
3. **Budgets differ.** Fancy-contract RSI exists only as ~1.2-epoch harvests
   against a 15-epoch ERDM; sst_pred RSI reached 24 epochs only at global
   batch 40, which the 2026-09-04 study places nearer batch-4 at ~11 epochs,
   against a batch-4 ERDM at 24.

## A1 specifically

`conf/loss/rsi_a1.yaml` (`reduce_to_erdm: true`, residual parameterization,
`label_mode log_sigma_eff`, ERDM's sigma_min / sigma_max / rho) exists and
`train_rsi_amip.sbatch RSI_RUNG=a1` launches it, but no A1 checkpoint exists
on any cluster or contract. What does exist is the reduction pinned at the
unit level in `test/diffusion/test_rsi_scheduler.py`: whole-trajectory
sampler equality (`test_a1_reduces_to_erdm_exactly`), the denoised readout,
the loss (`test_a1_loss_reduces_to_erdm_exactly`), zero-init behaviour, `c_in`
and the forward corruption. So A1 is verified as an equality of code paths and
untested as a trained model. Given the drift result, A1 is now doubly
motivated: it is the proposal's fallback operating point, and, trained in this
harness on the sst_pred contract to the production budget, it is the missing
in-harness A0 as well (its only non-ERDM ingredient is the inert second head).

## Where the artifacts are

Polaris `$R = /eagle/lighthouse-uchicago/members/awikner/physicsnemo-rsi`:
`checkpoints_*` as listed; ERDM baselines under
`/eagle/lighthouse-uchicago/amip-checkpoints/{translated,sstpred_epochs}`.
Midway3 (not reachable non-interactively from this machine):
`checkpoints_fullstack/`, `checkpoints_adamw/`, `checkpoints_gov2/`,
`checkpoints_sstpred/`, `checkpoints_diverged_54173034/`,
`outputs/rsi_a2_eval/`. Sources: [rsi-h1-precond-instability](rsi-h1-precond-instability.md),
[rsi-stability-campaign-2](rsi-stability-campaign-2.md),
[rsi-drift-polaris-test-plan](rsi-drift-polaris-test-plan.md), the proposal PDF
(Dropbox, `rolling_stochastic_interpolants_proposal.md.pdf`), and the run
scripts under `hpc/scripts/` and `$R/*.pbs`.
