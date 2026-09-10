<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plan: A0 (ERDM) and A1 (RSI in the ERDM reduction) on the sst_pred contract, batch 40, Polaris (draft 2026-09-10)

Status: **plan and scripts ready, nothing submitted.** Open questions are at
the end; the runs start once they are answered.

## 1. What upstream trained (the sst_pred ERDM we have been comparing against)

Read from `erdm_sstpred_raw/model_epoch=24.ckpt` (`hyper_parameters`,
`optimizer_states`, `lr_schedulers`):

| item | upstream value | our harness equivalent |
|---|---|---|
| scheduler | ERDM, W 6, 2 Heun steps, sigma 0.002-500, rho -10, sigma_data 1.0, P_mean 2 / P_std 1.2, S_churn 1.0, S_tmin 0 / S_tmax 1000, S_noise 1.5, alpha 0, noise_scale_path null, ocean_loss_weight 1 | `loss/erdm_v2.yaml` (identical, field for field) |
| backbone | DiT dim 1024, 16 heads / 8 temporal, 20 blocks, scalar_dim 2, c_grid_dim 5, downsample 4, cross layers 4 / heads 8, input_embed budget (256 / 128, column encoder, d_level 16, conv2 boundary encoder, pool stats, static bias, source norm), global_cond, output_head mix / 2 experts / flat / d_level 16 | `model/amip_erdm_sst_pred.yaml` (identical; the translator mapped all 371 tensors one for one) |
| contract | 153 channels: 6 surface, 5 x 26 upper air, 15 diagnostics, 2 predicted ocean channels (SST and sea ice, monthly-interp); forcings DSWRF lead, SST, sea ice, plus 2 constants; no SST-anomaly channel, no scalar forcing | `amip_erdm_sst_pred.yaml` + the dataset overrides every RSI run used (`sst_anomaly_channel none`, `scalar_forcing none`, normalized + spatially-smoothed constants) |
| data | 1979-2014 (train_year_end 2015 exclusive), 6-hourly store, 24 h stride, window_train, 13,143 steps per epoch at batch 4 | `amip_dailyavg_coarse_train7914` + boundary store, 1,315 steps per epoch at batch 40 |
| optimizer | Muon with aux AdamW: Muon group lr 5e-4 (10 x base), AdamW group lr 5e-5, betas (0.9, 0.95), eps 1e-10, weight decay 0.01 on both, Muon momentum 0.95 | `training.optimizer` Muon, `muon_lr_multiplier 10`, `weight_decay 0.01`; momentum via `muon_momentum` (wrapper default 0.95) |
| schedule | StepLR, gamma 0.95 per epoch (lr at the epoch-24 checkpoint 1.39e-4 Muon / 1.39e-5 AdamW) | `scheduler.type StepLR sl_gamma 0.95` (what `checkpoints_prod24_b40` used) |
| precision | 32-true (fp32 storage, TF32 matmuls) | `training.amp none` + `matmul_precision high` |
| EMA | decay 0.999 (per update) | `training.ema.decay` |
| batch | 4 (4 GPUs x 1) | 40 (10 nodes x 4 x 1) |
| budget | max_epochs 50; the `ep24` checkpoint is Lightning epoch index 24 = **25 completed epochs** (global_step 328,325 = 24.98 x 13,143); checkpoints every epoch | 24 epochs = 31,560 updates at batch 40 |

Two things worth knowing about the control we have been using: it was
trained with 10 x more optimizer updates than any batch-40 run of ours can
have at equal epochs, and it is one epoch further along than its name says.

## 2. The two runs

Both use `hpc/scripts/polaris_ablation_train_b40.pbs` (`ABL=a0` / `ABL=a1`),
which is the A2 bundle script with the rung selected by one variable, so the
three rungs share every other line:

| | A0 | A1 | A2 (already trained: `checkpoints_prod24_bundle_b40`) |
|---|---|---|---|
| model | `amip_erdm_sst_pred` (1 head) | `amip_rsi_sst_pred` (2 heads; H1 carries zero loss weight under the reduction) | `amip_rsi_sst_pred` |
| loss | `erdm_v2` | `rsi_a1` (`reduce_to_erdm`, residual parameterization, `label_mode log_sigma_eff`, `h1_precond none`, `noise_scale_path null`) | `rsi` + `h1_precond edm`, `noise_scale_path` S_c, `gamma 1.0 -> 0.04` |
| checkpoint dir | `checkpoints_a0_sstpred_b40` | `checkpoints_a1_sstpred_b40` | -- |
| run name | `erdm_sstpred_a0_b40` | `rsi_sstpred_a1_b40` | `rsi_sstpred_prod24_bundle_b40` |

What A1 is: the RSI code path with beta = 1 and Gamma = ERDM's sigma
schedule, so the fresh back slot is pure noise, the anchor is read but never
used, and the z head's EDM skip/out readout is ERDM's denoiser. The unit
tests pin the sampler trajectory, the denoised readout, the loss and the
zero-init behaviour as exact equalities with `ERDMScheduler`. A1 trained to
the same budget as A0 therefore tests three things at once: that the
two-head backbone with an inert head trains like the one-head one, that the
RSI training/loss plumbing (anchor frame, W+1 stack, ocean target stack,
per-slot tau labels) introduces no artifact, and that our in-harness ERDM
reproduces the upstream model's stable climate. A0 is the in-harness A0 the
proposal asked for and the ladder never had.

Config fix that fell out of this review: `loss/rsi_a1.yaml` inherited
`h1_precond: edm` from `rsi.yaml` and set the residual parameterization, which
the scheduler refuses; the rung did not instantiate until `h1_precond: none`
was added. A matching inference sampler `sampler/rsi_a1_sstpred.yaml` and an
`a1` / `erdm` config in `polaris_rsi_drift_eval_multi_phase5.pbs` were added
so the one- and five-year evaluations run through the same cascade and
protocol as every earlier number.

## 3. Hyperparameters at batch 40 (best judgement, one recipe for both)

Upstream's recipe is a batch-4 recipe. Scaling it to batch 40 was already
worked out on this model and data for A2 (jobs 7589775/6, 7597388/9 and the
bundle test, 2026-09-04..08), and I propose to reuse that result unchanged so
that A0 / A1 / A2 differ only in the scheduler:

| knob | value | why |
|---|---|---|
| base lr | 5e-4 (AdamW group), Muon group 5e-3 via x10 | linear scaling of upstream's 5e-5 x 10; the sqrt-scaled alternative (1.58e-4, `prod24_b40`) trailed batch 4 by 12% per epoch, the linear-scaled bundle reached per-epoch parity by epoch 4 |
| warmup | 1 epoch, linear | what makes the linear peak survivable at batch 40; upstream had none because its peak was 10 x lower |
| decay | cosine to eta_min 5e-5 over the 24-epoch horizon | ends where upstream's StepLR ends at epoch 24 (0.95^24 x 5e-4 = 1.46e-4 Muon, 1.46e-5 AdamW); the cosine reaches 5e-5 / 5e-4, i.e. slightly lower, and the bundle run's late-epoch loss (196) was the best of the batch-40 runs |
| Muon momentum | 0.85 | measured gradient-noise scale B_noise ~ 6, flat over the loss range; batch 40 is ~7 x past it, so the default 0.95 (a ~20-update window) only adds lag. Upstream's 0.95 is right for batch 4 |
| weight decay | 0.01 on both groups | upstream |
| precision | fp32 + TF32 | campaign 2: bf16 is what destabilises RSI; upstream is 32-true |
| EMA decay | 0.99 | upstream's 0.999 per update, 10 updates per batch-40 step |
| activation checkpointing | first 14 of 20 levels | fits the 40 GB A100 at +7% step time (level sweep, job 7591890) |
| seed | 0 | the A2 bundle run's; the DistributedSampler then serves the same sample order to all three rungs |
| validation | 10-day rollout every 2 epochs (link ends) | 4 ICs x 10 members, ~6 min; upstream validated every epoch, the bundle every 5 |
| epochs | 24 | matches A2; see question 2 |

Expected cost: the bundle run took 73-76 min per epoch including the
checkpoint write, so 2 epochs fit one 2 h 59 link with ~25 min to spare; A0
has one output head and should be a few minutes faster. 12 links per run,
~30 h of compute each, 360 node-hours each; wall-clock is set by the queue
(links waited 0.5-6 h this week).

## 4. Execution

1. Ship `rsi.py`-independent files to `$R`: `polaris_ablation_train_b40.pbs`,
   `conf/loss/rsi_a1.yaml`, `conf/sampler/rsi_a1_sstpred.yaml`,
   `polaris_rsi_drift_eval_multi_phase5.pbs`. (No scheduler code changes are
   needed; the reduction is already in the shipped `rsi.py`.)
2. Smoke: `ABL=a1 ABL_NODES=1 ABL_CKPT=checkpoints_a1_smoke` on the debug
   queue for 30 min (and the same for a0): confirms the reduction path runs
   under DDP with the ocean target stack, gives s/step for both, and checks
   that the A1 first-batch loss is in ERDM's range (the two objectives are
   identical, so the first-batch losses should agree to the noise draw).
3. Submit two 12-link chains (`qsub -W depend=afterany` between links, every
   link resumes from the last pair and exits early at the target), plus one
   spare blacklist-aware link each. Both chains can run concurrently; the
   queue took three of ours at once this week.
4. At epochs 12 and 24: one-year evaluations (`bias1yr`, 8 members, IC
   1996-01-01, obs-climatology truth) of both, through the multi script with
   configs `erdm` and `a1`; at 24: the five-year wave (A0, A1, and the A2
   bundle epoch 24 side by side). The upstream ERDM ep24 numbers
   (`eval_bias5yr_e24`, plateau 63-67) are the reference.
5. Decision rule. A0 and A1 should agree with each other within member noise
   (the epoch-26 seeds agreed to +-0.005 in alpha and +-4 in z500 RMSE) and
   both should sit near the upstream ERDM's plateau. Three outcomes: (a)
   A0 ~ A1 ~ upstream: the harness is clean and A1 is the fallback operating
   point; the A2 drift is the coupling. (b) A0 ~ upstream but A1 drifts: the
   RSI plumbing has an artifact independent of the coupling -- the most
   consequential possibility, and exactly what the proposal reserved A1 for.
   (c) A0 drifts too: the harness or the batch-40 recipe, not RSI, is the
   problem, and every earlier comparison is confounded.

## 5. Questions before submitting

1. **Recipe for A0.** The plan gives A0 the A2 bundle recipe (linear-scaled
   lr, warmup, cosine, momentum 0.85) so the three rungs are optimizer-
   identical. The alternative is an upstream-faithful A0 (StepLR 0.95 per
   epoch, momentum 0.95, the same linear-scaled lr) whose result is
   comparable to the upstream ep24 checkpoint but not to A2. Recommendation:
   bundle recipe; the upstream checkpoint already is the upstream-recipe
   reference.
2. **Budget.** 24 epochs matches A2 and costs ~30 h each. Upstream's ep24 is
   25 epochs at 10 x the updates, and the batch-size study puts 24 epochs at
   batch 40 nearer 11 batch-4 epochs in loss. 24 keeps the A0/A1/A2
   comparison clean; matching upstream in loss would need ~50. Recommendation:
   24 now, extend both by resuming if the plateau is still moving.
3. **Concurrency.** Two 10-node chains at once, or A1 first? Concurrent
   halves the wall-clock and the queue has taken three at once.
4. **Seeds.** One seed each (seed 0, same sample order as A2), or a second A0
   seed to put an error bar on the in-harness ERDM plateau (another 360
   node-hours)? Recommendation: one each now.
5. **Evaluation cadence.** One-year at epochs 12 and 24 plus five-year at 24
   as above, or only at 24? The mid-run one-year evaluation costs one
   debug-scaling hour and would show early whether A1 tracks A0.
