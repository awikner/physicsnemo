# RSI stability campaign 2 — surviving the edge of stability (draft 2026-08-31)

## Why a second campaign

Campaign 1 (rsi-h1-precond-instability, 2026-08) established that RSI training
sharpens its objective without saturating and hits an edge-of-stability (EoS)
excursion at any constant LR, under both Muon and AdamW. Muon/Adam are
scale-free, so gradient clipping and LR governors cannot rescue a run
mid-excursion (governor v1 chased sustained excursions; v2 death-spiraled when
rewind snapshots ran dry). The workable recipe was **decay-and-harvest**:
CosineToFloor(15k steps → 1e-5) and harvest the epoch-1 checkpoint.

The 2026-08-30 `rsi_sstpred_a2` retrain (first run with the corrected
constant-boundary preparation — see erdm-bias-gap root cause, 2026-08-29)
sharpened the picture: the excursion fired **at the floor transition itself**
(~15k steps, mid-epoch-2; loss 560 → 5–7k, gnorm 10⁴ → 10⁷) and never
recovered over 3.5 further epochs at floor 1e-5. The fancy-contract campaign
saw ~4–6k steps of floor residence first; the sst_pred contract (or the
corrected forcings) got none. Decay-and-harvest still works — e1 is clean and
under evaluation — but it caps effective training at ~1.2 epochs while the
ERDM baseline trains 23+. **Goal: stable RSI training ≥ 10 epochs**, so the
RSI-vs-ERDM comparison is run at comparable training budgets.

## Arm 0 — forcing-alignment audit of the TRAINING path (do this first)

Collaborator advice (2026-08-31, re the ERDM model, verbatim):

> One thing I can think of (that has tripped me up in the past) is to make
> sure the forcings are aligned + passed in correctly. For example, when
> denoising a noisy window at `<t+1><t+2><t+3><t+4>` to a cleaner window at
> `<t+1><t+2><t+3><t+4>`, the appropriate forcing is `<t><t+1><t+2><t+3>`.

That lag-1 convention is exactly what the W-offset investigation pinned for
the **inference** side, and our eval drive is proven bit-identical to
upstream's windowed rollout (probe_rollout_parity, 2026-08-29). **What has
never been verified at the same standard is the TRAINING side**: the window
batches `train_diffusion._train_step` hands `RSIScheduler.compute_loss`.
A misaligned training forcing (slot j conditioned on `t+j+1` instead of
`t+j`, or ocean targets read at the conditioning time instead of the frame's
own time) would not crash anything, would train to a healthy-looking loss,
and would make the learned forcing coupling systematically off-manifold — a
plausible *contributor to the sharpening itself* (the model keeps trying to
resolve an inconsistency the data cannot support). Deliverables:

1. Instrument a real training batch (the probe pattern from
   `probe_cgrid_seasonality.py` / the emit-time probe): capture
   `(y window times, c_grid window times, ocean-target times, calendar)` as
   actually passed to `compute_loss`, on the sst_pred contract with corrected
   constants, and check: states `t+1..t+W`, forcings `t..t+W-1`, ocean truth
   at each frame's own time `t+1..t+W`.
2. Same audit for the RSI-specific tensors: the anchor `a_w`, the H1 target,
   and `z_w`'s per-slot noise-scale application — each slot's τ profile must
   match the window position convention the sampler uses at inference
   (train/inference τ-alignment was the campaign-1 h1_precond lesson; recheck
   under the sst_pred contract).
3. Cross-check against upstream's training assembly (`window_train: true`
   path in amip_v2 `train_module.py`) the way the eval side was checked —
   numerically, on shared inputs, not by reading code.

Cost: half a day. If a misalignment is found, everything below is moot until
retrained.

## Candidate arms (after Arm 0), cheapest first

Probe protocol for every arm: 20k-step runs (~4.5 h on the 4×H100 node),
`RSI_LOSS_DIAG=200` on, gate = **no excursion before 30k steps** (2× the
current onset), tracked via gnorm EMA (excursion = gnorm > 20× trailing
median) and per-channel loss decomposition. Two survivors ladder to 10-epoch
runs with validation every 5 epochs; final gate = corrected-protocol combined
eval vs the ERDM sst_pred baseline.

| arm | change | hypothesis | cost |
|---|---|---|---|
| 1 | `ct_floor_lr` 1e-5 → 3e-6, `ct_decay_steps` 15k → 25k | sstpred excursion fired at the floor: the floor is simply too high for this contract | 1 probe |
| 2 | evaluate the **EMA** weights of the existing runs (decay 0.999, warmup 6 epochs — already saved in `checkpoint.0.N.pt`) | EMA may average through early excursion noise; free checkpoints already on disk | eval only |
| 3 | `weighting=snr_bump` → uniform (or clip the bump) | campaign-1's "sharpens without saturating": the bump concentrates weight on ever-harder τ as easy τ converge — a positive-feedback sharpener | 1 probe |
| 4 | raise the noise-scale floor 0.01 → 0.05 (`make_rsi_delta_scales --floor`) | floored channels (ocean tail, slow fields) get 1/σ_c² ≈ 10⁴ effective target weight; their gradients may dominate the tail of training | 1 probe + artifact |
| 5 | `muon_lr_multiplier` 3 → 2 (+ arm-1 floor) | the multiplier was tuned on the fancy contract with broken constants; corrected forcings change the loss landscape | 1 probe |
| 6 | skip-step guard: **reject** the optimizer step when gnorm > k× trailing median (no rescaling — scale-free-safe, unlike clipping/governors) | excursions start from single outlier batches; skipping them denies the spiral its first step. Different failure mode from governor v1/v2: no LR meddling, no rewind dependency | small code + 1 probe |
| 7 | `w_z` 1.0 → 0.5 (down-weight the ẑ head) | two-head interference: the ẑ regression may be the sharpening term (check the RSI_LOSS_DIAG h1/z split from campaign-1 logs first — if z dominates late, this is arm 3's cousin) | 1 probe |
| 8 | AdamW arm at 2e-5 with arm-1 schedule | AdamW held longer in campaign 1 (e1 loss 527); Muon's orthogonalized updates may interact badly with the sharpened curvature | 1 probe |

Not retrying (campaign-1 dead ends): gradient clipping alone, GnormLrGovernor
v1/v2, rewind-on-excursion, alpha/churn sampler dials (inference-side only).

## Bookkeeping

- All probes: run-scoped `++checkpoint_dir` (the shared-default resume trap),
  `wandb.mode=offline`, validation every 5 epochs on full runs.
- Baseline artifacts: `checkpoints_sstpred/` e1 (clean harvest, evaluated),
  e2–e4 (post-excursion, kept only as excursion forensics).
- The corrected-constants knobs are load-bearing for every arm:
  `++dataset.normalize_constant_boundary=true`
  `++dataset.constant_boundary_stats=spatial`
  `++dataset.smooth_constant_lsm=true` (+ `sst_anomaly_channel=none`,
  `scalar_forcing=none` on the sst_pred contract).
- Success = 10-epoch stable run whose corrected-protocol combined eval beats
  the e1 harvest and closes on the ERDM sst_pred ep-23 baseline.
