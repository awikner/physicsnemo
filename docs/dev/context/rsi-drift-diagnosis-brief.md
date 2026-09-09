<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI long-rollout drift vs the ERDM control: diagnostic brief

**Written 2026-09-08 for a fresh reader (human or model) asked to reason about
WHY the RSI forecaster drifts.** Everything below is measured unless marked as a
hypothesis. Paths are on ALCF Polaris unless stated. Repo: `ai-rossby` fork of
PhysicsNeMo, branch `ai-rossby-rsi`, working tree on Polaris at
`/eagle/lighthouse-uchicago/members/awikner/physicsnemo-rsi` (referred to as
`$R` below); data root `$AI_ROSSBY_DATA=/eagle/lighthouse-uchicago/physicsnemo-zarr`.

---

## 0. One-paragraph summary

Two forecasters share one backbone (RollingDiT, 561 M params, 45x90 daily AMIP
state, 6-frame rolling window), one dataset, one normalization, one evaluation
pipeline and one downscaler. They differ only in the generative formulation
and the trained weights: **ERDM** (Elucidated Rolling Diffusion, EDM-style
denoising of a window whose back slot starts from pure noise) versus **RSI**
(Rolling Stochastic Interpolants, where each window slot is transported from
its temporal *predecessor* to its target). In 8-member, 5-year (1827-day)
free-running rollouts with prescribed SST/sea-ice, ERDM reproduces the
observed 1996-2001 climatology to within z500 bias-map RMSE 46 m2/s2 and t2m
0.22 K (better than the upstream reference 65.4 / 0.285). RSI, through the
identical pipeline, settles within ~80 days into a state whose five-year mean
is, for **every** channel and level, a **~70% collapse of the spatial pattern
toward the channel's global mean** (shrinkage coefficient alpha = 0.68-0.95,
R2 = 0.9-0.99; time-mean pattern amplitude 0.15-0.45 of observed, ERDM 1.00),
with the **seasonal cycle almost gone**. The pattern *correlation* with
observations stays high (0.89-0.96 for T, z, q), so RSI keeps the shape of the
climate and loses its amplitude. The drift is deterministic (8 members agree to
<5%), flat across training epochs 17-24 while the training loss still falls,
identical to ERDM for the first 4 steps, and independent of sampler
stochasticity (the RSI sampler runs the probability-flow ODE). Pipeline,
data, forcing normalization and evaluation are ruled out by the ERDM control.
The question for the reader is *what in the RSI formulation/recipe produces a
contracting map with no restoring force toward the forced climatology*.

---

## 1. What the two models are

### 1.1 Shared

| item | value |
|---|---|
| backbone | `RollingDiTWrapper` (`physicsnemo/experimental/models/amip_si/rolling_dit.py`, `wrappers.py`), dim 1024, 20 blocks, 16 heads (8 temporal), window_size 6, c_grid_embed 32, c_scalar_embed 16, 4 c_grid cross layers, input_embed `budget` mode (`state_encoder: column`, `boundary_encoder: conv2`, `boundary_pool_stats`, `boundary_static_bias`, `source_norm: True`), `global_cond: True`, output_head `mix` with 2 experts, `decoder: flat`, `c_grid_downsample: 4`, dropout 0 |
| params | 561.37 M |
| state | 153 channels = 6 surface + 5x26 upper-air + 15 diagnostic + **2 predicted ocean channels appended at the TAIL** (SST, sea-ice; see 1.4) |
| surface | skin_temperature, surface_pressure, 2m_temperature, 2m_specific_humidity, 10m_u, 10m_v |
| upper air (26 levels hPa) | temperature, u, v, geopotential, specific_humidity at 5,7,10,20,30,50,70,100,125,150,175,200,250,300,400,500,600,700,800,850,875,900,925,950,975,1000 |
| diagnostic | USWRFtoa, ULWRFtoa, USWRFsfc, ULWRFsfc, DSWRFsfc, DLWRFsfc, LHTFLsfc, SHTFLsfc, PRATEsfc, hcc, lcc, mcc, mn2t, mx2t, mxtpr (all 24 h means/extremes) |
| c_grid (5 ch, varying first) | DSWRFtoa_24h_lead (insolation), sea_surface_temperature_monthly_interp, sea_ice_cover_monthly_interp, geopotential_at_surface, land_sea_mask. `ocean_grid_indices = [1, 2]` |
| c_scalar (2) | calendar |
| grid / step | 45 x 90 (4 deg), 24 h per model step, `timedelta_hours: 24` |
| data | `amip_dailyavg_coarse_train7914` (ERA5 daily means 1979-2014, coarsened from 180x360 by bilinear x0.25), boundary store `amip_dailyavg_boundary_train7914`; eval reads `amip_dailyavg_coarse` (full span) |
| normalization | per-channel **scalar** mean/std (`normalize_mean_dailyavg.nc` / `normalize_std_dailyavg.nc`, dims `('level',)` for upper air) => normalized 0 == the channel's global-and-time mean. Constants: `normalize_constant_boundary: true`, `constant_boundary_stats: spatial`, `smooth_constant_lsm: true` (the corrected preparation, see 4.2) |
| sst_anomaly_channel none, scalar_forcing none | |
| downscaler (eval only) | frozen upstream `x_ddc_dit` (`amip-checkpoints/translated/x_ddc_dit.mdlus`), DataDependentInterpolant, 5 steps, spherical noise; 45x90 -> 180x360. Same weights for both models. Downscaler-only bias is tiny (z500 ~2.5, measured 2026-08) |

Channel-layout v2; `ocean_state_variables: [sst_monthly_interp, sea_ice_monthly_interp]` widens in/out channels by 2 at the tail.

### 1.2 ERDM (control) — `conf/model/amip_erdm_sst_pred.yaml`, scheduler `physicsnemo/experimental/diffusion/erdm.py`

- Window of W=6 future frames y_{1:W}; one global diffusion time t in [0,1); slot w carries sigma_bar_w(t) from an EDM rho-schedule (`sigma_min 0.002`, `sigma_max 500`, `rho -10`, `sigma_data 1.0`); shift identity sigma_bar_w(1) = sigma_bar_{w-1}(0). Front slot emitted at sigma_min, back slot **enters as pure noise at sigma_max**.
- Loss: `x_bar = y + sigma*noise`, `D = denoise(model, x_bar, sigma)` (EDM c_skip/c_out/c_in preconditioning), `loss = mean_w lambda(sigma_w) f(sigma_w) ||D - y||^2` with EDM log-normal `P_mean 2.0, P_std 1.2`. Ocean block supervised with weight 1.0. Noise `gaussian`, `alpha 0.0` (no AR(1) temporal correlation), `noise_scale_path null`.
- Sampler at eval: `erdm_v2_nochurn` — 2 Heun steps per emitted frame, `S_churn 0` (deterministic PF-ODE), `S_noise 1.0`. Emitted frame is `x_bar[:, 0]` at sigma_min (a *sample*, not the denoised mean). New back slot = fresh `sigma_max * eps`. Ocean channels re-imposed as `truth + sigma_bar_w(0) * eps` at the top of every roll.
- Weights: upstream amip_v2 `sst_pred` checkpoints, epochs 17-24, EMA-averaged state_dict (n_averaged 328,350), translated to `.mdlus` (`amip-checkpoints/sstpred_epochs/erdm_sstpred_ep{17..24}.mdlus`). Trained upstream (amip_v2) with fp32, weight decay 0.01, Muon x10 multiplier, batch 4 (per campaign-2 notes, upstream LR ~3x the fork's 5e-5 batch-4 LR; exact upstream LR/epochs not re-verified here).

### 1.3 RSI (subject) — `conf/model/amip_rsi_sst_pred.yaml` (+ `output_head.num_output_heads: 2`), scheduler `physicsnemo/experimental/diffusion/rsi.py`

Slots w = 1..W, local time tau_w(t) = (W - w + t)/W (tau=0 base end, tau=1 clean). Forward interpolant per slot with anchor a_w = y_{w-1} (true predecessor):

```
x_w(tau) = a_w + beta(tau) (y_w - a_w) + Gamma(tau) z_w,   z_w ~ N(0, I)
beta(tau) = tau                                  (linear)
Gamma(tau) = gamma(tau) * S,  gamma(tau) = gamma_0 (gamma_1/gamma_0)^tau   (geometric)
gamma_0 = 1.0, gamma_1 = 0.04
S = per-channel scale = std of the model's one-step increment in normalized units
    ($AI_ROSSBY_DATA/norm_stats/sigma_c_sstpred153.pt, from tools/data/amip/make_rsi_delta_scales.py)
```

So the back slot (w=W, tau=0) starts at **anchor + one increment-std of white noise** — never from structureless noise ("the single most consequential change relative to ERDM", per the code). The front slot (w=1) starts a roll at tau = 5/6.

Two heads from one backbone pass on `c_in(tau) * x`, `c_in = 1/sqrt(Gamma^2 + sigma_data^2)`, label = tau:

```
H1  (parameterization "state", h1_precond "edm"):
     y_hat = c_skip(tau) x + c_out(tau) F_1,   c_skip = sd^2/(g^2+sd^2), c_out = g sd/sqrt(g^2+sd^2), sd = 1
zhat (EDM skip form):  zhat = x * g/(g^2+sd^2) + F_z * sd/sqrt(g^2+sd^2)
Delta_hat = (y_hat - x + Gamma zhat) / max(1 - beta, beta_floor=1e-3)      (increment estimate)
```

Loss (`compute_loss`), y = W+1 clean frames (anchor y_0 .. y_W), t ~ U(0,1), tau = local_time(t):

```
anchors = y[:, :-1] (+ anchor_noise * S z', anchor_noise = 0.0 => NO perturbation)
x = interpolant(anchors, targets, tau, z)
err = w_1 (H1 - y)^2 + w_z (Gamma (zhat - z))^2         (both in state units; w_1 = w_z = 1)
loss = mean_b [ omega(tau_w) * sum_{C,H,W} err ]  with
omega = lambda(sig) f(sig),  sig = sigma_eff = gamma(tau)/(beta(tau) delta_std),  delta_std = 1.0
lambda = (sig^2+1)/sig^2,  f = lognormal(ln sig; P_mean 2.0, P_std 1.2)     ("snr_bump")
ocean block weight 1.0
```

Sampler (`sample_window`, 2 Heun steps, `integrator coeff`, `final_denoise False`, `eps_scale 0.0` => bit-identical PF-ODE, no stochastic forcing):

```
x <- x + (beta(tau_next)-beta(tau_cur)) Delta_hat + (Gamma(tau_next)-Gamma(tau_cur)) zhat   (exact coefficient increments)
emitted frame  = y_hat[:, 0]  (H1 readout at the LAST head evaluation, tau = 11/12 for slot 1 — a conditional MEAN)
window shift   = x[:, 1:]  ++  fresh slot = y_hat[:, -1] + Gamma(0) z        (anchor = H1 readout of slot W at tau_W = 11/72 ~ 0.15)
ocean channels = imposed at t=0 of every roll as the interpolant between anchor-time and own-time truth
```

Note the asymmetry with ERDM: in RSI the quantity copied forward across rolls (the fresh-slot anchor) is a **conditional-mean estimate of the least-informed slot** (tau ~ 0.15, gamma ~ 0.61, where c_skip = 0.73 and the network must supply ~1.7 y - 1.2 a - 0.9 S z of signal through F_1); nothing in the window is ever regenerated from noise conditioned on the forcing alone.

Weights: fork-trained `checkpoints_prod24_b40/` (24 epoch pairs; `.mdlus` = live weights, `checkpoint.0.N.pt` metadata `ema` = EMA shadow). Eval used **EMA weights** (`weights_used: ema`), epochs 17-24.

### 1.4 Ocean channels (both models)

`ocean_state_variables` = SST and sea-ice monthly-interp forcings, appended as 2 predicted state channels at the tail. Training target = truth at each frame's own time (`append_ocean_target`). At inference `impose_ocean` overwrites the 2 tail channels at the top of every roll (ERDM: truth + schedule noise; RSI: the interpolant between the two truths); the emitted ocean channels are then the model's own evolved estimate and are stripped downstream. **`skin_temperature` is a separate surface STATE channel, not one of the imposed ocean channels** — this matters for 3.6.

### 1.5 Verified train/inference consistency (RSI)

- Scheduler init line, training vs eval, identical: `W=6, num_steps=2, param=state (h1_precond=edm), beta=linear, gamma=[1.0 -> 0.04] (geometric), integrator=coeff, solver=heun, weighting=snr_bump, eps_scale=0.0, noise=gaussian, reduce_to_erdm=False`.
- Eval sampler is built from `conf/sampler/rsi_sstpred_e1.yaml` via `rollout._load_group` (OmegaConf.load + hydra instantiate, so `${oc.env:AI_ROSSBY_DATA}` resolves) and carries `noise_scale_path = .../sigma_c_sstpred153.pt`; a wrong path would have crashed at `torch.load`. (The `loss: rsi` block visible in the eval's resolved Hydra config, with gamma 0.5/0.02 and null scales, is the unused default group — the forecaster uses `model.forecaster.sampler`.)
- Dataset flags identical in train and eval: `normalize_constant_boundary True / spatial / smooth_lsm True`; runtime probe of the assembled c_grid: per-channel RMS 0.97-1.23 (`$R/probe_cgrid_scale.py`).
- Training-side window alignment verified bit-exactly 2026-09-02 (states t+1..t+W, forcings t..t+W-1, ocean targets at own times). Emit-time bug in the drivers fixed 2026-08-25 (`docs/dev/context`/memory `window-rollout-emit-time-bug`); all numbers here are post-fix.
- Mid-training validation runs on EMA weights, same scheduler instance as the loss.

---

## 2. Training recipes

### 2.1 RSI prod24_b40 (the evaluated model)

| item | value |
|---|---|
| cluster/job | Polaris, 10 nodes x 4 A100-40GB = 40 ranks, global batch 40 (batch 1/rank), 12 chained 3 h links, 2 epochs/link, 1315 steps/epoch |
| optimizer | Muon (`MuonWithAuxAdam`), lr **1.5811e-4** (= 5e-5 x sqrt(10)), `muon_lr_multiplier 10` (Muon-group lr 1.58e-3), `weight_decay 0.01`, momentum 0.95 (default) |
| schedule | StepLR gamma 0.95 per epoch; LR-compounding-on-resume bug found at epoch 14 -> chain restarted from the epoch-2 checkpoint with the fix (`train_diffusion.py` resets base lr before `make_scheduler`) |
| precision | fp32 + TF32 (`amp none`, `matmul_precision high`); bf16 destabilises RSI (campaign 2) |
| EMA | decay 0.99, warmup 6 epochs |
| grad | activation checkpointing (all 20 levels), `grad_clip_norm 1e6` (none) |
| loss overrides | `h1_precond edm`, `gamma_0 1.0`, `gamma_1 0.04`, `noise_scale_path sigma_c_sstpred153.pt`, `window_size 6` |
| seed | 0; per-epoch reshuffle via `DistributedSampler.set_epoch` |
| epochs | 24 (2026-09-04 .. 09-06) |

Per-epoch median training loss / ocean-block loss:

```
ep1 808.5/0.103  ep2 405.8/0.057  ep3 341.8/0.044  ep4 309.9/0.036  ep5 291.5/0.032
ep6 278.6/0.030  ep7 269.7/0.028  ep8 262.4/0.027  ep9 256.6/0.026  ep10 251.7/0.025
ep11 247.5  ep12 243.7  ep13 240.5  ep14 237.8  ep15 235.1  ep16 232.8  ep17 230.6
ep18 228.8  ep19 227.2  ep20 225.2  ep21 224.0  ep22 222.2  ep23 221.3  ep24 219.6 (ocean 0.021)
```

(RSI and ERDM losses are not comparable numbers.)

**LR was ~3x too low.** A 4-epoch sweep (2026-09-08) shows batch-40 per-epoch loss reaches parity with the batch-4 run (lr 5e-5) only with lr 5e-4 + Muon momentum 0.85 + 1-epoch warmup + cosine(24): epoch-4 medians b4 276.7 / b40@1.58e-4 310.0 / b40@2.81e-4 281.9 / b40 bundle 276.3. A 24-epoch batch-40 run with the bundle recipe is in progress (`checkpoints_prod24_bundle_b40/`, `prod24bundle_*.log`, chain 7599766-7599776) — its 5-year bias will show whether a better-trained RSI drifts less. Caveat from 3.3: the drift is flat across epochs 17-24 while loss falls 230 -> 219, so "more fit" has not helped so far.

### 2.2 Mid-training validation (10-day, 4 ICs x 10 replicate members, 2 sampler steps, ACC vs 1979-2014 daily climatology at 45x90)

| epoch | rmse s1 sfc | rmse s10 sfc | rmse s10 ua | ACC s10 sfc | ACC s10 ua | ACC s10 diag | spread s10 sfc |
|---|---|---|---|---|---|---|---|
| 5 | 2.970 | 94.08 | 149.5 | 0.805 | 0.827 | 0.752 | 0.109 |
| 10 | 2.637 | 92.05 | 133.8 | 0.818 | 0.848 | 0.771 | 0.103 |
| 15 | 2.453 | 86.07 | 119.1 | 0.828 | 0.862 | 0.779 | 0.102 |
| 20 | 2.275 | 80.70 | 114.4 | 0.847 | 0.876 | 0.799 | 0.104 |

RMSE in physical units averaged over the group's channels; ACC/spread in normalized units. Short-range skill is healthy and improving; **10 days is too short to see the drift** (onset ~day 28, see 3.4). Spread grows from 0.005 (step 1) to ~0.10 (step 10) — very under-dispersive relative to RMSE, expected for a PF-ODE sampler whose only randomness is the interpolant latent.

### 2.3 Earlier RSI history relevant here (from `docs/dev/context/rsi-*.md` and memory)

- Three preconditioning bugs fixed 2026-08-19 (loss Jacobian Gamma^2, `c_in`, EDM output skip on zhat); three proposal deviations fixed 2026-08-21..24 (`h1_precond edm`, per-channel S, gamma in increment units). Without the output skip a zero-init network meant "zero transport" and the lead-W frame was 180x worse than ERDM.
- RSI training sharpens its objective without saturating: every constant LR eventually hits an edge-of-stability excursion (campaign 1); **bf16 is the destabiliser** and fp32+TF32 with upstream's optimizer config passes 30k steps (campaign 2). The prod24_b40 run had no excursion.
- An epoch-1 "harvest" checkpoint (misaligned pre-08-25 protocol, raw constants) scored z500 bias-RMSE 1746 — i.e., the drift is not new with the 24-epoch model.

---

## 3. Measured results (5-year combined rollouts, 180x360 after x_DDC)

Protocol (`climate_eval_suite.py`, `eval_bias_fanout_polaris.pbs` / `eval_erdm_bias_fanout_polaris.pbs`): IC 1996-01-01 00Z (dataset row 26296), horizon 1827 daily frames, 8 members = 8 latent realizations from one checkpoint (`perturber replicate_only`, identical ICs), streaming rollout, SST/sea-ice prescribed from truth every step, truth = precomputed 1996-2001 obs daily-mean climatology at 180x360 (`norm_stats/obs_climatology_1996_2001/`, S->N, anchors verified). Bias = time-mean(pred) - obs climatology; "rmse" = lat-weighted RMS of that map. In-sample span (training 1979-2014). One member per GPU, 64 GPUs, 2:58 h.

### 3.1 Headline, cross-epoch mean +- std over epochs 17-24 (RSI has 7 epochs; ep19 lost to a killed job)

Bias-map RMSE:

| field | RSI | ERDM | upstream ref | RSI/ERDM |
|---|---|---|---|---|
| z500 (m2/s2) | 2045 +- 28 | 46.2 +- 3.6 | 65.4 | 44 |
| u250 (m/s) | 10.57 +- 0.08 | 0.684 +- 0.051 | | 15 |
| t850 (K) | 8.87 +- 0.11 | 0.196 +- 0.009 | | 45 |
| q850 (kg/kg) | 3.15e-3 +- 3e-5 | 1.30e-4 +- 1e-5 | | 24 |
| t2m (K) | 10.42 +- 0.11 | 0.219 +- 0.010 | 0.285 | 48 |
| prate (kg/m2/s) | 2.49e-5 +- 2e-7 | 2.71e-6 +- 1e-7 | | 9 |
| q2m | 4.53e-3 +- 6e-5 | 1.05e-4 +- 4e-6 | | 43 |
| u10m | 3.18 +- 0.02 | 0.178 +- 0.012 | | 18 |
| v10m | 1.68 +- 0.04 | 0.147 +- 0.006 | | 11 |

Global-mean bias:

| field | RSI | ERDM |
|---|---|---|
| z500 | -923 +- 45 | -5.0 +- 5.0 |
| t850 | -3.26 +- 0.21 K | -0.027 +- 0.015 |
| t2m | -3.47 +- 0.20 K | -0.013 +- 0.020 |
| q850 | -1.63e-3 | +5.8e-5 |
| prate | -2.1e-6 (-0.18 mm/day) | -4.7e-7 |
| u10m / v10m | +0.32 / +0.46 | -0.001 / +0.009 |
| u250 | +0.01 +- 0.52 | +0.17 +- 0.04 |

Per-epoch values (RSI e17,18,20,21,22,23,24 | ERDM e17..24) for z500 RMSE: 2101, 2047, 2022, 2022, 2031, 2035, 2059 | 46.1, 44.6, 53.5, 48.5, 42.8, 45.6, 46.1, 42.2. **Flat in epoch for both.**

### 3.2 Every channel, epoch 24 (lat-weighted global mean; obs mean for scale)

| channel | RSI bias | ERDM bias | obs mean | RSI map-RMSE | ERDM map-RMSE |
|---|---|---|---|---|---|
| skin_temperature | -3.87 K | +0.007 | 288.1 | 11.07 | 0.23 |
| surface_pressure | -311 Pa | -3.9 | 98560 | **3492 (35 hPa)** | 98 |
| 2m_temperature | -3.55 | -0.006 | 287.3 | 10.54 | 0.21 |
| 2m_specific_humidity | -2.14e-3 (-22%) | -6e-6 | 9.90e-3 | 4.6e-3 | 1.0e-4 |
| 10m_u / 10m_v | +0.23 / +0.25 | -0.005 / +0.005 | -0.36 / 0.16 | 3.18 / 1.63 | 0.17 / 0.14 |
| USWRFtoa | +10.8 W/m2 | -0.18 | 97.6 | 23.0 | 1.7 |
| ULWRFtoa | -1.7 | -0.04 | 242.3 | 24.1 | 1.3 |
| USWRFsfc | +7.0 (+29%) | -0.05 | 24.5 | 21.0 | 0.85 |
| ULWRFsfc | -28.5 | +0.05 | 396.5 | 57.6 | 1.1 |
| DSWRFsfc | -13.6 | +0.03 | 188.4 | 31.4 | 1.9 |
| DLWRFsfc | -26.0 | -0.01 | 338.6 | 53.2 | 1.2 |
| LHTFLsfc (down +) | **+31.0 (evaporation -37%)** | +0.58 | -83.9 | 49.4 | 2.7 |
| SHTFLsfc | -3.0 | -0.08 | -16.6 | 14.9 | 1.3 |
| PRATEsfc | -2.8e-6 (-8%) | -4.3e-7 | 3.36e-5 | 2.5e-5 | 2.7e-6 |
| hcc / lcc / mcc | -0.034 / +0.003 / +0.013 | ~0 | 0.34 / 0.38 / 0.25 | 0.14 / 0.18 / 0.11 | 0.01 |
| mn2t / mx2t | -4.28 / -3.58 K | ~0 | 285.4 / 289.3 | 10.8 / 10.8 | 0.25 |

Physical inconsistencies of the drifted state: evaporation down 37% but precipitation only down 8% (water budget not closed); ULWRFsfc -28 W/m2 is what sigma T^4 gives for -3.9 K (consistent), while surface pressure has a 35 hPa RMS pattern error and a -3.1 hPa global mean (mass not conserved, ~0.3%).

### 3.3 Vertical structure (epoch 24, global / tropics |lat|<20 / poles |lat|>60)

Temperature bias (K):

| level hPa | RSI global | RSI tropics | RSI poles | ERDM global |
|---|---|---|---|---|
| 10 | +0.32 | -1.2 | +4.5 | -0.02 |
| 50 | +8.7 | +12.0 | +5.8 | +0.26 |
| 100 | +6.3 | **+14.4** | -1.6 | +0.23 |
| 150 | +5.2 | +9.4 | +1.0 | +0.09 |
| 200 | -0.06 | -1.0 | +0.8 | -0.03 |
| 300 | -4.0 | -9.9 | +6.4 | -0.05 |
| 500 | -2.7 | -8.8 | +9.8 | -0.04 |
| 700 | -4.5 | -10.3 | +9.6 | -0.02 |
| 850 | -3.3 | -9.9 | +12.0 | -0.03 |
| 1000 | -3.9 | **-10.9** | **+13.4** | 0.00 |

Geopotential bias (m2/s2): tropics -4400 at 200 hPa, poles +4200; z500 tropics -2220, poles +2510. Zonal wind: stratosphere 10 hPa -11 global, poles -20 (polar night jet collapsed); tropical upper troposphere +5 to +10 m/s. Specific humidity: tropics -0.0066 kg/kg at 1000 hPa (-44% of 0.015), poles +0.0028 (+140% of 0.002).

### 3.4 Zonal bands, epoch 24 (15-degree bands S->N; obs band mean below each)

```
t2m   RSI   +31.7  +14.4   +2.9   -3.7   -8.8  -10.7  -11.1  -10.8   -5.1   +2.5  +11.1  +16.9
      ERDM  -0.14  -0.07  +0.02  +0.01  +0.00  -0.01  +0.01  -0.03  -0.04  +0.01  +0.07  +0.05
      obs    238    262    278    288    295    298    299    297    287    277    267    260
t850  RSI   +21.6  +12.3   +4.3   -2.1   -7.6   -9.9  -10.0   -9.9   -5.1   +2.5   +8.3  +12.1
z500  RSI  +4260  +3590  +1210  -1200  -2190  -2230  -2170  -2260  -1560    +20  +1150  +1630
      obs  48800  49500  52400  55700  57200  57500  57500  57300  56000  53900  52400  51500
sp    RSI  +9730  +2610   -583  -1720  -1010   -764   -761   -628   +627   -342   -167  -1060
q2m   RSI +.0039 +.0034 +.0019 -.0004 -.0035 -.0069 -.0071 -.0040 -.0009 +.0014 +.0029 +.0038
      obs  .0004  .0020  .0045  .0081  .0121  .0162  .0166  .0126  .0080  .0048  .0029  .0016
LHTFL RSI  -25.5  -11.2  +11.1  +40.6  +59.0  +52.1  +45.5  +40.5  +26.8   +5.4  -11.1  -20.6
      obs   -1.3  -17.5  -47.2  -93.8   -121   -119   -114   -102  -75.1  -43.7  -23.3   -8.2
u10hPa RSI -15.2  -31.4  -32.0  -15.8   -6.0   -1.5   -1.4   -6.4  -11.6  -13.8  -14.4   -9.6
      obs   14.0   30.7   27.3    6.8   -7.2  -13.9  -13.9   -6.9    2.2    7.4   10.8    7.0
```

The sign of the bias flips where the field crosses its global mean: warm where obs is colder than the mean, cold where warmer. Antarctica t2m +32 K (band -90..-75).

### 3.5 The decisive test: shrinkage toward the global mean

Fit `bias(lat,lon) = -alpha * (obs(lat,lon) - <obs>) + c` (lat-weighted) per channel/level. alpha = 1 is complete collapse to the global mean; R2 is the fraction of the bias map's spatial variance the fit explains.

| channel | RSI alpha ep24 | R2 | RSI alpha epoch-mean | ERDM alpha | ERDM R2 |
|---|---|---|---|---|---|
| skin_temperature | 0.681 | 0.95 | 0.684 | -0.001 | 0.00 |
| surface_pressure | 0.490 | 0.87 | 0.492 | 0.000 | 0.00 |
| 2m_temperature | 0.683 | 0.95 | 0.685 | -0.000 | 0.00 |
| 2m_specific_humidity | 0.718 | 0.97 | 0.713 | -0.002 | 0.01 |
| 10m_u / 10m_v | 0.864 / 0.871 | 0.96 / 0.92 | 0.863 / 0.872 | 0.001 / 0.003 | 0.00 |
| ULWRFtoa | 0.753 | 0.93 | 0.748 | -0.005 | 0.01 |
| ULWRFsfc / DLWRFsfc | 0.685 / 0.680 | 0.95 / 0.96 | | ~0 | 0.00 |
| LHTFLsfc / SHTFLsfc | 0.725 / 0.799 | 0.90 / 0.84 | | ~0 | 0.01 |
| PRATEsfc | 0.926 | 0.83 | 0.919 | -0.001 | 0.00 |
| hcc / lcc / mcc | 0.930 / 0.809 / 0.817 | 0.89 / 0.88 / 0.87 | | -0.02 / -0.01 / -0.02 | <0.05 |
| mn2t / mx2t | 0.658 / 0.698 | 0.92 / 0.94 | | ~0 | 0.00 |
| temperature @10/50/100/150/200 | 0.82 / 0.80 / 0.75 / 0.76 / 0.86 | 0.85-0.97 | | 0.06 / 0.01 / -0.01 / -0.01 / 0.03 | <0.35 |
| temperature @250/300/500/700/850/925/1000 | 0.71 / 0.72 / 0.69 / 0.71 / 0.74 / 0.74 / 0.73 | 0.96-0.98 | | ~-0.005 | <0.14 |
| u @10/100/250/500/850/1000 | 0.70 / 0.84 / 0.89 / 0.89 / 0.88 / 0.87 | 0.93-0.99 | | 0.07 / 0.02 / 0.005 / 0.008 / 0.007 / 0.002 | <0.38 |
| v @250/500/850 | 0.94 / 0.90 / 0.85 | 0.83 / 0.70 / 0.81 | | 0.02 | 0.01 |
| geopotential @10/100/250/500/850/1000 | 0.78 / 0.73 / 0.71 / 0.72 / 0.77 / 0.84 | 0.95-0.99 | | 0.03 / 0.004 / -0.003 / -0.002 / 0.004 / 0.008 | <0.27 |
| specific_humidity @250/500/850/1000 | 0.87 / 0.80 / 0.79 / 0.73 | 0.97 / 0.93 / 0.97 / 0.97 | | -0.02 / -0.03 / -0.01 / -0.003 | <0.17 |

Spatial-pattern amplitude of the 5-year time-mean, std(pred - <pred>) / std(obs - <obs>), and pattern correlation:

| channel | RSI amp | ERDM amp | RSI corr | ERDM corr |
|---|---|---|---|---|
| skin_temperature / 2m_temperature | 0.358 / 0.352 | 1.001 / 1.000 | 0.891 / 0.901 | 1.000 |
| surface_pressure | 0.543 | 1.000 | 0.939 | 1.000 |
| 2m_specific_humidity | 0.312 | 1.002 | 0.904 | 1.000 |
| 10m_u / 10m_v | 0.226 / 0.293 | 1.000 | 0.603 / 0.438 | 0.999 / 0.997 |
| PRATE | 0.431 | 1.007 | 0.173 | 0.994 |
| ULWRFsfc / DLWRFsfc | 0.354 / 0.348 | 1.000 | 0.891 / 0.918 | 1.000 |
| temperature @100/250/500/850/1000 | 0.283 / 0.316 / 0.328 / 0.289 / 0.305 | 1.00-1.01 | 0.89-0.96 | 1.000 |
| u @100/250/500/850 | 0.188 / 0.163 / 0.149 / 0.175 | 0.99-1.00 | 0.69-0.83 | 0.999 |
| geopotential @100/250/500/850 | 0.295 / 0.303 / 0.301 / 0.263 | 1.00 | 0.93-0.96 | 1.000 |
| specific_humidity @250/500/850/1000 | 0.201 / 0.293 / 0.250 / 0.302 | 1.00-1.03 | 0.65-0.91 | 1.000 |

Reading: RSI's climate keeps the *shape* of the observed climatology (corr 0.9+ for thermodynamic fields) at **~30% of its amplitude**; fields whose pattern can be read off the constant forcing (surface pressure over orography) collapse least (alpha 0.49); fields with the weakest persistence (v, precipitation, high cloud, u) collapse most (alpha 0.86-0.95, corr 0.2-0.6). ERDM shows no shrinkage at all (alpha ~0, amplitude 1.00).

### 3.6 Time evolution: onset, plateau, seasonal cycle

Per-step lat-weighted RMSE of the 8-member-mean prediction against the constant obs climatology (mean over epochs 17-24; "sfc" = surface group, physical units; "ua" = upper-air; "diag" = diagnostic):

```
step   RSI sfc  ERDM sfc   RSI ua  ERDM ua  RSI diag ERDM diag
   1     132.7     132.6    785.7    785.7     27.2     27.2
   4     123.3     123.0    778.0    778.0     27.0     26.9
  10     163.9     141.1    779.3    767.7     26.7     26.1
  15     169.4     102.2    752.1    739.0     25.5     24.1
  20     158.3      82.6    699.8    723.6     23.4     21.9
  25     169.2      79.0    673.1    704.7     21.8     21.0
  28     179.0      77.9    637.6    693.4     20.8     20.5
  30     206.7      78.6    613.1    683.1     20.2     20.2
  35     311.4      76.2    577.0    655.6     20.4     19.0
  40     371.2      76.9    528.7    614.6     20.2     17.8
  50     478.9      70.3    488.6    545.4     20.5     15.5
  60     535.6      63.9    489.0    476.6     20.9     13.4
  70     598.7      60.4    522.7    404.5     21.2     12.0
  85     627.0      54.1    517.1    279.8     21.4     11.3
 100     626.0      53.9    505.3    173.3     21.5     12.2
 110     622.3      50.8    493.2    146.5     21.7     13.2   <- ERDM ua minimum (mid-April)
 200     622.6      76.7    506.7    701.0     23.0     20.6   <- ERDM ua maximum (mid-July)
 300     608.5      55.0    526.6    258.4     21.2     13.0
 400     602.8      74.4    553.0    645.4     22.0     18.8
 730     605.0      82.5    561.3    654.1     22.5     23.8
1000     608.9      61.3    495.0    304.9     20.8     10.4
1461     598.2      73.3    562.0    666.5     22.3     23.4
1827     603.7      74.4    561.0    666.4     22.3     23.5
```

Four things to read off this:
1. **Identical for the first ~4 steps** (both curves are the IC's own anomaly from the climatology).
2. RSI's surface curve stops falling at step ~6 and sits at 150-180 through step 28 while ERDM's keeps falling (its ensemble mean relaxes toward the climatology as members decorrelate). **Onset of the run-away is sharp at step 28-30**, the transition takes ~50 steps (206 at step 30, 479 at 50, 599 at 70), plateau 620 from step ~85 on, then flat for 4.7 years. The drifted state is an attractor, not a random walk.
3. **ERDM's upper-air and diagnostic curves oscillate with the annual cycle** (min ~mid-April/mid-October, max ~mid-July/early-February against an annual-mean climatology); **RSI's are flat** (ua 487-562 all year). The RSI state has lost most of its seasonal cycle despite receiving the seasonally varying insolation, SST, sea-ice and calendar forcings every step.
4. No non-finite values, no RMSE jumps (stability scan clean for both).

### 3.7 Members, spread, ocean skin temperature

- 8-member per-member global skt bias (RSI ep24): -4.00, -3.89, -3.84, -3.83 ... (all within ~5%): the drift is **deterministic**, not a wandering random walk. ERDM members: -0.002, -0.016, -0.005, +0.021.
- Spread/skill ratio at step 1 ~1.2e-4 for both (identical ICs, PF-ODE samplers; only the interpolant latent differs between members).
- Skin temperature over the ocean: -0.19 K at 20 frames -> -4.6 K by frame 1827 (earlier smoke), even though SST is imposed exactly every roll in the 2 ocean channels of the same state window and is present in c_grid. Over open ocean skt should equal SST. The model's skt channel decouples from the SST it is shown — consistent with the whole-state shrinkage (alpha 0.68 for skt, same as t2m), i.e., **the forcing does not pull the state back**.
- Surface pressure: ocean -1423 Pa / land +2435 Pa (earlier smoke; consistent with the +9.7 kPa Antarctic band and negative mid-latitude bands above).

---

## 4. What is ruled out, and how

| hypothesis | status | evidence |
|---|---|---|
| evaluation pipeline / truth / alignment / downscaler | **ruled out** | ERDM through the identical pipeline: z500 46, t2m 0.22, alpha ~0. Same code path, same obs climatology, same x_DDC, same IC, same 8-member protocol. |
| constant-boundary "crush" (raw z_sfc in c_grid crushing the varying forcings through `source_norm`) — the mechanism that made upstream ERDM collapse by day 10 in Aug | **ruled out for this run** | prod24_b40 trained AND evaluated with `normalize_constant_boundary True / spatial / smooth_lsm True`; runtime c_grid RMS 0.97-1.23 per channel. (Earlier fork RSI checkpoints DID train with raw constants; this one did not.) |
| sampler stochasticity | ruled out | `eps_scale 0` => PF-ODE, bit-identical path; drift identical across 8 latent realizations. ERDM control also deterministic (`S_churn 0`). |
| under-training in epoch count | not the driver | bias flat epochs 17-24 while loss falls 230 -> 219; epoch-1 harvest (older protocol) had z500 1746. |
| numerical instability / blow-up | ruled out | fp32, no gnorm excursion in training, no non-finite/jumps in 1827-step rollouts, plateau is stable. |
| ocean imposition off / wrong window | ruled out | imposed every roll at t=0 with the correct one-step-shifted window (drivers fixed 2026-08-25; `impose_ocean` shape checks); training alignment bit-verified. |
| train/inference config mismatch (gamma, scales, W, steps) | ruled out | identical scheduler init lines; sampler yaml carries the sigma_c path; EMA weights used as in training-time validation. |
| data (stores, normalization stats) | ruled out | shared with the clean ERDM control. |

Not yet tested: everything in section 6.

---

## 5. Structural differences RSI vs ERDM that could carry the drift

| aspect | ERDM | RSI |
|---|---|---|
| what the back slot starts from | pure noise, sigma_max = 500 | anchor (model's own estimate of the previous frame) + ONE increment-std of white noise (Gamma(0) = 1.0 * S) |
| what is copied across rolls | nothing (window contents are re-noised samples; only the ODE state x_bar persists, and each slot's information about the absolute state came from the denoiser conditioned on forcing + window) | the anchor chain: y_hat[:, -1] (conditional mean of the least-informed slot) seeds the next slot; every slot's x = anchor + accumulated increments |
| emitted frame | a sample (x_bar at sigma_min) | a conditional mean (H1 readout at tau = 11/12) |
| training anchors | n/a | exact truth (`anchor_noise 0`): the network never sees an anchor off the data manifold |
| information the network MUST extract from forcing during training | large: to denoise the sigma_max slot it must generate the absolute state (season, SST pattern, land/sea contrast) from c_grid + the cleaner slots | small: the anchor already carries ~all of tomorrow's state; the increment y_w - y_{w-1} depends only weakly on absolute forcing |
| restoring force when the window is off-manifold | denoiser regenerates absolute states conditioned on forcing at every roll | none by construction unless the network learned increments that depend on the anchor's absolute level |
| readout skip at the front slot | c_skip(sigma_min) = 1 - 4e-6 | c_skip(gamma_1=0.04) = 0.998 (H1), i.e. a 0.2% contraction of x per emitted frame if F_1 does not supply +0.05 y exactly |
| readout skip at the anchor-producing slot | n/a | c_skip(tau~0.15, gamma~0.61) = 0.73; the network must output F_1 ~ 1.7 y - 1.2 a - 0.9 S z to reproduce y |
| loss weight | lambda(sigma) f(sigma), sigma = sigma_bar_w | lambda(sig_eff) f(sig_eff), sig_eff = gamma/(beta*delta_std): sig_eff = 0.04 at the emitted end, -> infinity at tau -> 0 |
| stochastic exploration per roll | none at S_churn 0, but the back slot is a fresh sample from the model's conditional | none at eps_scale 0; the only randomness is Gamma z of one increment-std |
| ensemble behaviour | members decorrelate, mean -> climatology | members stay locked (deterministic drift) |

---

## 6. Hypotheses to investigate (ranked by the author's prior), each with a discriminating test

**H1 — No restoring force: RSI's objective never teaches the forcing -> absolute-state map.** In training every slot's anchor is the true previous day, so the increment the network learns is conditioned on an on-manifold state and depends only weakly on c_grid/calendar. At inference a small systematic contraction of the copied-forward state has nothing to oppose it; the state relaxes to where the model's increments vanish, which for a network trained on increments around the climatology is a damped version of the climatological pattern (alpha ~0.7) with a damped seasonal cycle. ERDM regenerates absolute states from noise under forcing at every roll, so any deviation is corrected within a window.
*Test A (restoring-force probe, cheap, teacher-forced):* take true W+1-frame windows, add a spatially uniform perturbation to the state (e.g. -3 K to all temperature channels, or scale every channel's anomaly pattern by 0.7), build the interpolant at tau_w(0), run one roll, and measure whether the emitted frame / the fresh-slot anchor moves back toward truth (restoring), stays (neutral), or moves further (contracting). Run the same on ERDM (perturb the clean window before schedule-noising). Report the per-channel restoring coefficient. This single experiment discriminates H1 from H2/H3.
*Test B:* fine-tune (or train from scratch) with the fresh-slot anchor replaced by the model's own y_hat over a few unrolled rolls (rollout fine-tuning), or with `anchor_noise > 0` plus a *pattern-shrink* anchor augmentation (a <- <a> + s (a - <a>), s ~ U(0.7,1)) so the network sees off-manifold anchors and must use forcing to correct them.

**H2 — Compounding conditional-mean shrinkage through the anchor chain (regression to the mean).** The quantity copied forward is E[y_W | x_W] at tau ~ 0.15, an MSE-optimal estimate under substantial uncertainty; MSE estimators are shrunk toward the conditional mean of their prior, and the prior here is the training-set marginal (global mean in normalized space). A per-roll amplitude factor (1 - s) compounds: 0.99^100 = 0.37, matching a ~50-step transition to a ~0.3 amplitude plateau; the plateau exists because the forced part of the pattern is re-injected each step. ERDM emits samples and never copies a conditional mean forward.
*Test:* along a rollout, log per step the amplitude ratio std(x_k - <x_k>)/std(truth_k - <truth_k>) for a few channels, separately for (i) the emitted y_hat[:,0], (ii) the fresh-slot anchor y_hat[:,-1], (iii) the raw window states x[:, w]. A geometric decay of (ii) that leads (i) supports H2. Also teacher-forced: feed TRUE windows and measure std(y_hat_w)/std(y_w) per slot w — a ratio < 1 at w = W is the per-step shrink.
*Intervention:* replace the fresh-slot anchor by a *sample* rather than the mean, e.g. y_hat[:, -1] + Gamma(tau_W(1)) z (add back the residual latent), or by the integrated interpolant state x[:, -1] itself; or use `final_denoise True`; or `eps_scale > 0` (Euler-Maruyama forcing) so slots are samples. Any of these that removes the drift points at H2.

**H3 — Exposure bias in the plain sense (train on truth anchors, test on model anchors), `anchor_noise = 0`.** Subsumed by H1/H2 but with its own cheap knob.
*Test:* short fine-tune with `anchor_noise` in {0.5, 1.0} (white, S-scaled) and re-run the 5-year bias. If white anchor noise alone fixes it, H3; if not, the mismatch is structured (H1/H2).

**H4 — Readout skip contraction at the front slot.** With `h1_precond edm`, y_hat = 0.998 x + c_out F_1 at tau ~ 1; the network must output F_1 ~ 0.05 y - z to cancel the skip. Weight decay 0.01 + EMA shrink F_1 slightly -> a 0.1-0.2%/frame contraction. Alone this decays without a plateau, so it can only be a contributing term.
*Test:* teacher-forced front-slot readout ratio std(y_hat_1)/std(y_1) (expect 0.998-1.0); compare a rollout with `h1_precond none` weights (none trained — would need a retrain) or with the emitted frame taken as x_final - Gamma(1) zhat instead of H1.

**H5 — State-parameterization reconstruction and the beta_floor clamp.** Delta_hat = (y_hat - x + Gamma zhat)/(1 - beta) with 1 - beta as small as 1/12 at the front slot; the coefficient integrator multiplies back by Delta-beta so the algebra is exact, but the Heun corrector evaluated at tau_next pairs a lagged x with a new tau. The residual parameterization (ablation A3, never trained) removes the division.
*Test:* teacher-forced one-roll error decomposition per slot (RSI_LOSS_DIAG-style) at inference tau values; check the emitted-frame error is not dominated by the front slot's second step.

**H6 — Loss weighting starves the emitted end or the anchor end.** sigma_eff spans 0.04 (emitted) to ~infinity (tau -> 0). With P_mean 2, f peaks at sigma_eff = e^2 ~ 7.4, i.e. tau ~ 0.10 (deep back slots); lambda = (sig^2+1)/sig^2 re-inflates the small-sigma end (lambda f ~ 0.41 at sigma_eff 0.04 vs ~0.05 at 7.4). Whether the *anchor-producing* region tau ~ 1/6 .. 1/3 is under-weighted relative to its role is unclear.
*Test:* evaluate omega(tau) over the W slots at t = 0 and t = 1; compare the per-slot teacher-forced errors; try `weighting uniform` or `midrange` in a short fine-tune.

**H7 — Forcing pathway learned weak (a consequence of H1, but measurable directly).** RSI has lost the seasonal cycle.
*Test:* Jacobian/sensitivity of the emitted frame to c_grid channels (finite differences: +1 sigma insolation, +1 K SST) for RSI vs ERDM weights on the same true window; compare the norm of `input_embed` boundary-encoder weights and the c_grid cross-attention output magnitudes between the two checkpoints (same architecture, so directly comparable).

**H8 — Optimization recipe (LR 3x too low, StepLR, EMA 0.99, wd 0.01).** The bundle-recipe retrain is running (checkpoints_prod24_bundle_b40). Prediction under H1/H2: it will drift too, perhaps with a different alpha. If it does NOT drift, the drift was a fit-quality symptom and H1/H2 are downgraded.

**H9 — Ocean-block handling.** RSI imposes the interpolant a + beta(y - a) + Gamma z at t=0 and then lets the model evolve the 2 ocean channels; the emitted ocean channels are model estimates. ERDM does the analogous thing. Unlikely to drive a whole-state collapse but cheap to check: rollout with the ocean block hard-clamped to truth at every head evaluation, or with `nocean 0` weights (none trained).

Things that would *quickly* falsify the shrinkage picture: a rollout whose per-step pattern amplitude does not decay geometrically before the plateau; a teacher-forced readout ratio of 1.00 at every slot; a restoring-force probe that shows RSI pushing back as strongly as ERDM.

---

## 7. Where everything is

Polaris `$R = /eagle/lighthouse-uchicago/members/awikner/physicsnemo-rsi`:

| artifact | path |
|---|---|
| RSI checkpoints (24 epochs, live `.mdlus` + `checkpoint.0.N.pt` with EMA in metadata) | `$R/checkpoints_prod24_b40/` |
| RSI bundle-recipe retrain (in progress) | `$R/checkpoints_prod24_bundle_b40/`, logs `$R/prod24bundle_*.log`, test epochs 1-4 `$R/bundle_7597447.log`, `bundle_7597448.log` |
| RSI training logs (prod24_b40) | `$R/prod24b40_7591784.log`, `$R/prod24b40_75936*.log`; Hydra `$R/outputs/rsi_sstpred_prod24_b40/.hydra/config.yaml` |
| ERDM control checkpoints (EMA-averaged, translated) | `/eagle/lighthouse-uchicago/amip-checkpoints/sstpred_epochs/erdm_sstpred_ep{17..24}.mdlus`; raw `$R/erdm_sstpred_raw/model_epoch={17..24}.ckpt` |
| 5-year eval outputs (per epoch): `eval_suite.pt` with `climatology.{surface,upper_air,diagnostic}_{pred_mean,truth_mean,bias}` at (C[,L],180,360), `rmse_acc` per step, `headline`, `global_bias`, `member_spread`, `spread_skill`, `stability`, `config` | RSI `$R/eval_bias5yr_e{17,18,20,21,22,23,24}/`, ERDM `$R/eval_erdm_bias5yr_e{17..24}/`; per-rank logs alongside; Hydra `$R/outputs/{bias5yr,erdm_bias5yr}_e*/.hydra/config.yaml` |
| fan-out job scripts | `$R/eval_bias_fanout_polaris.pbs`, `$R/eval_erdm_bias_fanout_polaris.pbs` (16 nodes, 1 GPU per member x epoch; 2:58 of a 2:59 wall — no slack) |
| analysis scripts used for sections 3.2-3.6 | `$R/compare_bias_runs.py`, `$R/drift_structure.py`, `$R/drift_shrinkage.py` (+ `*.pbs` debug wrappers; outputs `$R/analyze_bias_runs.out`, `$R/drift_structure.out`, `$R/drift_shrinkage.out`). Reading `eval_suite.pt` needs torch => compute node (login conda torch lacks CUDA-13/cray-mpich libs). |
| obs climatology (truth) | `$AI_ROSSBY_DATA/norm_stats/obs_climatology_1996_2001/` (+ `climatology_obs_meta.json`) |
| per-channel increment scales S | `$AI_ROSSBY_DATA/norm_stats/sigma_c_sstpred153.pt` (153 values) |
| validation climatology (ACC) | `$AI_ROSSBY_DATA/norm_stats/val_clim_coarse/` |
| c_grid scale probe | `$R/probe_cgrid_scale.py` |
| batch-4 parity run (lr 5e-5, StepLR, 1 node) | `$R/checkpoints_prod24/`, `$R/prod24_7586612.log` (+ continuation 7599793) |

Repo (`ai-rossby-rsi`): `physicsnemo/experimental/diffusion/rsi.py` (RSI scheduler, 1185 lines, heavily commented), `erdm.py`, `physicsnemo/experimental/models/amip_si/{rolling_dit.py,wrappers.py}`, `examples/weather/ai_rossby/{train_diffusion.py,train_loop.py,validate_diffusion.py,climate_eval_suite.py,obs_climatology.py,rollout.py,inference.py}`, configs under `examples/weather/ai_rossby/conf/{model,loss,sampler,training,validation}/`. Context docs: `docs/dev/context/rsi-h1-precond-instability.md`, `rsi-stability-campaign-2.md`; tests `test/recipes/ai_rossby/test_rsi_scheduler.py` (incl. the A1 ERDM-reduction parity test), `test_eval_combined.py`.

Known caveats: both spans are in-sample; RSI epoch 19 is missing (job killed); the "member" axis here is latent realizations, not epochs as in amip_v2's outer axis; the ERDM control uses EMA-averaged weights loaded raw (no second EMA swap) and lands ~30% better than the upstream-quoted 65.4/0.285 (ensemble/epoch selection differences, not investigated).
