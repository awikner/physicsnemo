<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# `rsi_drift` — oracle-head Gaussian toy for the RSI long-rollout drift

`oracle_toy.py` answers one question:

> Does the RSI **formulation as implemented** (`RSIScheduler` + the shipped
> `conf/sampler/rsi_sstpred_e1.yaml` recipe) collapse / under-disperse a forced
> climate **even with a perfect network**?

ERDM (`conf/sampler/erdm_v2_nochurn.yaml`) through the identical harness is the
control. Nothing in the samplers is re-implemented: both rollouts go through the
real `sample_rollout` / `sample_window` / `_fresh_slot` / `heads` /
`precondition` code paths. Only the *network* is replaced — by the exact
Bayes-optimal denoiser of a linear-Gaussian toy climate.

## The toy climate

`C = 4` independent channels, each a Gaussian AR(1) in anomaly space with a
prescribed seasonal mean:

```
y_k[c] = mu_k[c] + a_k[c],   a_{k+1} = rho_c a_k + sqrt(1-rho_c^2) eps
mu_k[c] = A_c sin(2 pi k / P)                      Var(a) = 1 (stationary)
rho = (0.5, 0.8, 0.95, 0.99)  ->  S_c = sqrt(2 (1-rho_c)) = (1.00, .632, .316, .141)
```

`S_c` is *exactly* the one-step increment std, so the `noise_scale_path`
artifact the shipped recipe loads is self-consistent with the process by
construction — the fairest possible test of the formulation (no mis-calibrated
`S`). It is written to `sigma_c_toy.pt` with the same `(C, 1, 1)` shape as the
real `sigma_c_sstpred153.pt` (`tools/data/amip/make_rsi_delta_scales.py:122`),
which `RSIScheduler._scale` relies on for correct channel broadcasting.

Forcings mirror the real pipeline's time convention (slot `w` holds the state at
`k+w` and is conditioned on the forcing at `k+w-1`):
`c_grid[:, j] = [sin, cos]` of the phase of absolute frame `k0+j`, spatially
constant, plus a redundant `c_scalar` calendar. The oracle reads `mu` off
`c_grid[:, 0]` alone by phase rotation — exactly the information a real network
gets from insolation/calendar, and nothing more.

## The oracle heads

Under the RSI **training** joint (true anchors `a_w = y_{w-1}`, i.i.d. latents)
the window is a linear-Gaussian observation of `v = (y_0 .. y_W)`:

```
x_w = (1-beta_w) y_{w-1} + beta_w y_w + gamma_w S_c z_w
Prec     = Sigma_y^{-1} + A^T N^{-1} A,   A = [(1-beta) | beta] bidiagonal
N        = diag(nv_w),  nv_w = ((1-beta_w)^2 sigma_a^2 + gamma_w^2) S_c^2
E[v|x]   = Prec^{-1} (Sigma_y^{-1} mu + A^T N^{-1} x)
E[z_w|x] = (gamma_w S_c / nv_w) (x_w - (A E[v|x])_w)
```

`sigma_a` is the *train-time* `anchor_noise`; the perturbed-joint oracle
(`sigma_a > 0`) is what intervention E2(f) needs (`z` and `z'` enter `x` only
through their sum, so the `E[z|x]` formula stays closed-form).

The oracle module then **inverts the scheduler's preconditioning** — undoes
`c_in`, and emits the RAW heads

```
F_1 = (E[y|x] - c_skip x) / c_out ,   F_z = (E[z|x] - z_skip x) / z_out
```

so that `RSIScheduler.heads()` returns the conditional means bit-exactly. This
mirrors `test/diffusion/test_rsi_scheduler.py::_RSILinearStub`. It recovers the
global time from `label[:, 0]` (slot 1's `tau` is never clamped by `time_eps`)
and rebuilds every slot's `tau` with the scheduler's own formula, then asserts
the reconstruction reproduces the labels it was handed. ERDM's oracle is
`D = E[y_w | xbar_{1:W}]` presented through `precondition()` as
`_ERDMLinearStub` does.

## Validation (`--experiments validate`)

* **V1** `sched.heads(oracle, x, tau, ...)` vs an independently computed
  posterior mean: max abs error ~1e-15 (`h1`) / ~1e-13 (`zhat`).
* **V2** Monte-Carlo MSE of the head vs the analytic Bayes MSE
  (`diag(Prec^{-1})`), per slot x channel: ratio 1.00 +- 0.04 at n = 4096
  (MC noise is +-2.2%). A wrong `mu` bookkeeping or a wrong `S` would show up
  here as excess MSE.
* **V3** `compute_loss(oracle)` is strictly lower than `compute_loss` of five
  degraded oracles (raw head scaled by 0.5/0.95/1.05/2.0, shifted by +0.02) —
  a necessary condition for Bayes optimality, checked for RSI (anchor_noise 0
  and 1) and ERDM.
* **teacher-forced** one roll from a TRUE `W+1` window: per-slot readout
  amplitude ratio `std(yhat_w)/std(y_w)`, emitted/anchor-slot RMSE and bias,
  against the analytic Bayes std.
* **truth control** every ratio is also normalized by `metrics()` applied to
  `members` INDEPENDENT truth trajectories over the identical index range, so
  the finite-sample bias of a 3000-step variance estimate on a rho = 0.99
  series is not mistaken for a sampler defect.

## Experiments

| id | what |
|---|---|
| `E0` | ERDM oracle, shipped `erdm_v2_nochurn` — the control |
| `E1` | RSI oracle, shipped `rsi_sstpred_e1`, plus anchor-chain and per-slot instrumentation |
| `E2` | interventions: `final_denoise=True`; `num_steps=6`; fresh anchor `= x[:,-1]`; fresh anchor `+ c S z'` for c in {1,2,3} and the per-channel variance-matched c; `eps_scale` sweep; train-time `anchor_noise` in {1,2} with the **re-derived perturbed-joint oracle**; `gamma_0` in {0.5, 2}; raw-head scale error (the L2 probe) |

Metrics per run (per channel): emitted time-mean error, seasonal amplitude
ratio and `alpha_season = 1 - slope(ensemble-mean on mu)` (the toy analogue of
the brief's spatial shrinkage `alpha`), temporal variance ratio (total and
deseasonalized), member spread and its binned time curve, lag-1 autocorrelation,
plus the same suite applied to the recorded anchor chain (`y_hat[:, -1]`).

## Running

```bash
cd /home/awikner/repos/physicsnemo-rsi-diagnosis
./.venv/bin/python tools/diagnostics/rsi_drift/oracle_toy.py \
    --horizon 3000 --members 64 --ics 3 --e2-ics 2 \
    --experiments all --out <dir>
```

CPU only, float64 throughout, fully seeded. ~13 s per 3000-step / 64-member RSI
rollout (the per-`t` linear algebra is cached; only two distinct global times
are ever evaluated at `num_steps=2`). `--experiments` takes a comma list of
`validate,E0,E1,E2`. Results land in `<dir>/oracle_toy_results.json`.

## Known limitations

A linear-Gaussian oracle is *by construction* incapable of showing
off-manifold learned-map effects: no representation error, no weight decay, no
finite-capacity bias, and the conditional mean is exactly affine. So a correct
time-mean / seasonal amplitude here does **not** refute the brief's H1 (no
restoring force from the learned forcing->state map) — it only shows that H1 is
not *implied* by the formulation. The head-scale intervention is the toy's
proxy for a small learned-map error, not a substitute for it.

## Headline result (2026-09-08, horizon 3000, 64 members, 3 ICs)

Toy channels rho = 0.5/0.8/0.95/0.99, so S_c = 1.00/0.632/0.316/0.141.
All ratios normalized by the independent-truth control.

| run | `alpha_season` | var(internal)/ctrl | member spread/ctrl | anchor spread/ctrl |
|---|---|---|---|---|
| E0 ERDM oracle | 0.00-0.03 | .94 .99 .95 .90 | .97 1.00 .98 .95 | n/a |
| E1 RSI shipped | **0.00** | **.51 .45 .42 .41** | **.72 .67 .65 .64** | **.33 .52 .61 .63** |
| E2a `final_denoise=True` | 0.00 | .55 .51 .50 .51 | .74 .71 .70 .71 | .40 .58 .67 .71 |
| E2b `num_steps=6` | 0.00 | .56 .50 .48 .49 | .75 .71 .69 .70 | .38 .56 .66 .69 |
| E2c anchor = `x[:,-1]` | **.07 .30 .59 .67** | 1.66 1.78 1.81 1.67 | 1.29 1.34 1.36 1.42 | .60 1.03 1.28 1.40 |
| E2d anchor + 1·S·z' | 0.00 | 1.02 .88 .87 .82 | 1.01 .94 .93 .91 | .47 .73 .87 .90 |
| E2f train `anchor_noise=1` | 0.00 | .32 .28 .27 .27 | .56 .53 .52 .52 | .22 .38 .48 .51 |
| E2 `gamma_0=2` | 0.00 | .73 .66 .64 .65 | .86 .82 .80 .81 | .32 .58 .74 .80 |
| E2 `eps_scale=0.5` | non-finite by step ~13 on every channel with S_c < 1 | | | |

1. **A perfect network does not lose the forced response.** `alpha_season` and
   the seasonal amplitude ratio are 1.00 +- 0.02 for RSI, i.e. the toy does
   **not** reproduce the real model's `alpha = 0.68-0.95` pattern collapse.
2. **A perfect network does collapse the variance and lock the members.** The
   emitted spread saturates at 0.64-0.72 of climatological and the *anchor
   chain* — the quantity copied across rolls — at 0.33-0.63, worst for the
   largest `S_c` (least persistent channel). That per-channel ordering is the
   same ordering as the real model's `alpha`.
3. **The cause is the fresh slot's variance deficit, not the readout, not the
   ODE.** `final_denoise`/`num_steps` buy +4pp. Injecting one extra `S` of
   noise into the fresh-slot anchor recovers spread 0.91-1.01. Halving/doubling
   `gamma_0` moves the plateau monotonically (0.45-0.49 / 0.80-0.86).
4. **Train-time `anchor_noise` alone makes it worse** (spread 0.52-0.56 at
   `sigma_a = 1`): the perturbed-joint Bayes head shrinks *more*, while
   inference still injects only `Gamma_0`. It only helps if the inference-time
   injection is matched (0.72-0.87).
5. **An anchor that is temporally mis-registered reproduces the real symptom.**
   E2c (anchor = the slot-W ODE state at `tau = 1/6`, i.e. 5/6 stale) is the
   only intervention that damps the forced amplitude — to 0.94/0.77/0.44/0.33,
   the same range as the real model's 0.15-0.45 — while inflating variance and
   lag-1 autocorrelation.
6. **`eps_scale > 0` is not usable at these `S_c`.** The Langevin term adds
   isotropic noise in state units while the score is `-zhat/(gamma S_c)`;
   `eps_scale >= 0.5` diverges within ~13 rolls on every channel with
   `S_c < 1`.

---

# `learned_toy.py` — small TRAINED nonlinear toy

Companion to `oracle_toy.py`, answering the question the oracle explicitly
cannot: **what does a network that was actually TRAINED with the real
`RSIScheduler.compute_loss` do in a free rollout, and does ERDM trained on the
same data with `ERDMScheduler.compute_loss` behave differently?** All the
effects a linear-Gaussian oracle is blind to — finite capacity, weight decay,
zero-init readouts, a nonlinear/non-Gaussian target process, an imperfectly
learned forcing pathway — are in scope here.

Nothing in the schedulers is re-implemented. Training calls
`compute_loss(model, c_grid, c_scalar, y)` verbatim; rollouts go through a
loop whose only difference from `sample_rollout` is a pluggable fresh-slot
function, and the loop is asserted **bit-exact** against
`RSIScheduler.sample_rollout` / `ERDMScheduler.sample_rollout` at every run
(`results["harness"]`, `rsi_bitexact: true`). Train-time anchor interventions
use the scheduler's own `anchor_noise` or override the single
`perturb_anchor` hook.

## Toy system (`ToySystem`)

1-D periodic ring, `N = 32` sites, `C = 1`, tensors `(b, T, 1, 1, N)`:

```
m_k(x) = M p(x) (1 + 0.5 sin(2 pi k / P))                  P = 180
du/dt  = -a (u-m) - b (u-m)^3 + q (u-m)^2 + kappa Lap(u) - lam u du/dx + sig xi
```

`p` is a fixed 3-harmonic unit-std ring pattern, `xi` is low-wavenumber
(k0 = 2.5) white-in-time forcing, 2 Euler substeps of `dt = 0.125` per model
step. The state is z-scored with a **scalar** mean/std over space and time,
exactly as the real pipeline normalizes a channel, so normalized 0 IS the
channel's global-and-time mean and the brief's shrinkage fit
`bias(x) = -alpha (clim(x) - <clim>) + c` transfers verbatim.

Measured properties (seed 0, 200k steps): `S` (1-step increment std) 0.1685,
climatological pattern std 0.781, deseasonalized std 0.543, seasonal-composite
amplitude 0.300, skew -0.62, kurtosis 3.20, two-half climatology correlation
0.9993 (ergodic and stationary). True forecast-error std at leads 1..6 =
0.157, 0.211, 0.241, 0.266, 0.290, 0.307 — i.e. the fresh slot's
`Gamma(0) = 1 * S = 0.168` under-injects the lead-5 forecast variance by
**2.96x** (lead L3, measured).

`--forcing-mode` is the load-bearing knob:

* `direct` — `c_grid[j] = m_j(x)`: the forced mean pattern is handed to the
  network at every slot. *More* forcing information than the real model has.
* `uniform` — `c_grid[j] = <m_j>` (season only, no spatial pattern). The
  network is a circular-conv net, so it cannot store `m(x)` in its weights
  either: the pattern now has to be carried by the WINDOW. This is the toy
  analogue of the real model's situation, where 5 forcing channels do not
  determine a 153-channel climatology.

## Networks

One architecture for both formulations (459,794 params): circular Conv1d
stem over `[x, c_grid]`, 4 residual blocks with a learned softmax mixing
matrix across the `W` slots (the temporal-attention analogue), Fourier label
embedding of `tau` (RSI) / `log sigma` (ERDM) plus a `c_scalar` term,
**zero-init last layer** as in `RollingDiT`. RSI emits `2C` channels
(H1, F_z), ERDM `C`. AdamW, identical lr/steps/batch/seed for both.

## What it measures

Per variant, from a `horizon`-step, `members`-member free rollout off one true
IC: the shrinkage fit (`alpha`, `R2`), time-mean pattern amplitude ratio and
correlation, deseasonalized and total temporal variance ratio, seasonal
amplitude ratio, member spread (report as `spread/sigma_clim`: 1.0 is a
correctly dispersed ensemble; the JSON's `spread_saturation` divides by
`sqrt(2) sigma` so its ideal value is 0.707), the brief-3.6 trace (per-step
RMSE of the member mean against the constant climatology), plus teacher-forced
per-slot readout ratios, the H1-TestA one-roll restoring probe, and a
multi-roll **flush probe** (perturb the true IC window, follow
`rms(perturbed - reference)` for 48 rolls against the model's own
`sqrt(2) x spread` decorrelation floor).

## Running

```bash
cd /home/awikner/repos/physicsnemo-rsi-diagnosis
O=/path/to/out
./.venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py \
    --out $O/run1 --phase train --steps 9000 --ft-steps 3000 --extras
./.venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py \
    --out $O/run1 --phase roll --variants erdm,rsi_shipped --horizon 5000 \
    --members 32 --burn 500 --extras          # shard the variant list across
./.venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py \
    --out $O/run1 --phase probe --extras      # 3 concurrent procs on one GPU
./.venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py \
    --out $O/run1 --phase collect
# and the forcing ablation (rsi + erdm only):
./.venv/bin/python tools/diagnostics/rsi_drift/learned_toy.py \
    --out $O/unif --phase train --forcing-mode uniform --nets erdm --steps 6000
```

`--quick` is a 2-minute smoke of the whole pipeline. Total for the numbers in
the notes below: ~24 min training + ~20 min rollouts on one RTX 2070 SUPER.
matplotlib is not installed in this venv, so `--phase collect` writes CSVs
(`trace200.csv`, `clim_profiles.csv`, `patamp.csv`) instead of figures.

## Headline result (seed 0, 5000 steps, 32 members)

`direct` forcing, i.e. the mean pattern is given:

| variant | alpha | amp | corr | var(deseas) | var(total) | seasR | spread/sigma |
|---|---|---|---|---|---|---|---|
| erdm | -0.003 | 1.003 | 1.000 | 1.005 | 0.990 | 0.970 | 1.003 |
| rsi_shipped | 0.027 | 1.059 | 0.919 | **0.284** | 0.467 | 1.033 | **0.536** |
| rsi_final_denoise | 0.028 | 1.007 | 0.966 | 0.494 | 0.623 | 1.021 | 0.705 |
| rsi_anchor_xlast | -0.096 | 1.154 | 0.949 | 0.255 | 0.359 | 0.834 | 0.513 |
| rsi_anchor +2S z | 0.066 | 0.938 | 0.996 | 0.720 | 0.788 | 1.005 | 0.849 |
| rsi_anchor +3S z | 0.144 | 0.863 | 0.992 | 0.898 | 0.902 | 0.957 | 0.948 |
| rsi_eps_scale 1.0 | -0.070 | 1.188 | 0.901 | 152.6 | 117.4 | 1.620 | 12.3 |
| rsi train anchor_noise 1 | 0.032 | 1.061 | 0.913 | 0.234 | 0.436 | 1.046 | 0.488 |
| rsi train anchor_noise 2 | 0.042 | 1.068 | 0.896 | 0.131 | 0.359 | 1.050 | 0.366 |
| rsi ft pattern-shrink anchors | -0.132 | 1.159 | 0.976 | 0.956 | 0.931 | 0.921 | 0.989 |
| rsi ft self-generated anchors | 0.005 | 1.089 | 0.913 | 0.847 | 0.949 | 1.132 | 0.928 |
| rsi h1_precond none | 0.035 | 1.044 | 0.925 | 0.289 | 0.482 | 1.056 | 0.540 |
| rsi trained with S = 1 | 0.007 | 0.994 | 0.999 | 0.876 | 0.902 | 0.993 | 0.934 |

So: **the shipped RSI recipe reproduces the variance/spread collapse but NOT
the time-mean pattern collapse**, as long as the forcing hands the pattern
over. Its free-running deseasonalized variance is 28% of truth and its
ensemble is 54% dispersed, while ERDM is 1.00/1.00 through the same harness.
The intervention ordering is the diagnostic: everything that puts *sample*
variance back into the copied-forward slot fixes it (`+3S z` 0.90,
`S = 1` 0.88, pattern-shrink anchors 0.96, self-anchor fine-tune 0.85),
white train-time `anchor_noise` makes it monotonically **worse** (0.28 ->
0.23 -> 0.13), and `h1_precond none` changes nothing (0.289 vs 0.284), so the
EDM readout skip is not the driver.

The `uniform` forcing ablation (same system, same seed, same short-range
skill) is what turns the variance collapse into the real symptom — see the
notes in the diagnosis brief / the session report for those numbers.

### The two forcing ablations

`--forcing-mode static` (c_grid = the annual-mean pattern only; the seasonal
modulation must be learned from the `c_scalar` calendar) changes **nothing**:
RSI alpha 0.038, amp 1.052, seasonal amplitude ratio 1.011, deseasonalized
variance 0.289; ERDM -0.011 / 1.011 / 0.962 / 1.019. Short-range skill is
unchanged in both (RSI step-1 RMSE 0.0227 vs 0.0239, ERDM 0.0069 vs 0.0073),
so the toy gives **no support for H7** (weak learned forcing pathway) at this
scale: RSI learns the calendar -> seasonal-amplitude map as well as ERDM.

`--forcing-mode uniform` (no spatial pattern anywhere in the inputs) breaks
BOTH models and **inverts** the asymmetry: ERDM alpha 1.045 / amp 0.068,
RSI alpha 0.582 / amp 0.523, both with inflated variance (2.4-3.4x). The toy
network is a circular-conv net, so with no positional input nothing can pin
`m(x)` in space and the pattern rotates/decays away; RSI's anchor chain
*resists* that better than ERDM's noise-regenerated window. Do not read this
mode as an H1 confirmation — it is a broken control, and it is also the one
place where the toy is structurally unlike the real backbone, which has
absolute position embeddings on a 45x90 grid and therefore *can* store the
climatology in weights.

### Verdict transferred to the real model

* The formulation-plus-recipe, trained with the real loss, reliably produces
  a **~70% collapse of the free-running variance and a ~46% deficit of
  ensemble dispersion** with ERDM at 1.00 through the same harness. Cause:
  the fresh slot injects `Gamma(0) = 1 * S` = one 1-step increment std, while
  the quantity it replaces carries the *W-step* forecast spread
  (measured deficit 2.96x in variance at lead 5 in this toy, and
  `(S_c/sigma_lead5)^2 = 0.34` predicts the measured 0.28 variance ratio).
* It does **not** reproduce the brief's headline symptom (alpha 0.68-0.95
  collapse of the time-mean pattern). In the toy the anchor chain's
  per-roll shrinkage is at the Bayes floor -- slot-W teacher-forced
  `slope_on_truth` = 0.978 vs the 1 - Var(y|x)/Var(y) = 0.975 predicted by the
  toy's own lead-1 error -- so it shrinks only the *unpredictable* component,
  and the forced/predictable pattern passes through unshrunk. A real
  mean-pattern collapse therefore needs one of: an F_1 under-supply well
  BELOW the Bayes floor (measurable per channel on the real checkpoint), or a
  nonlinear rectification of the variance deficit into the mean that this toy
  is too weakly nonlinear to show.

### One more structural fact about the toy's RSI time-mean error

`alpha ~ 0` does not mean the toy RSI climatology is right. Its bias map has
RMS 0.328 = **42% of the climatological pattern std** (ERDM 0.0105 = 1.3%),
but regressed on `[clim, Lap(clim), d clim/dx, clim^2]` it loads on
`d clim/dx` (coefficient -0.71, i.e. the mean pattern sits ~0.7 grid points
displaced), not on `clim`. So RSI's toy climate error is a **phase/advection**
error, not the brief's shrinkage. The interventions rank identically for both
symptoms: bias RMS 0.328 (shipped) -> 0.146 (`+3S z`) -> 0.088 (`+2S z`) ->
0.034 (`S = 1` training) vs ERDM 0.0105, i.e. whatever restores the missing
latent variance also removes ~90% of the time-mean error.
