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

## Measured cost + hardware calibration (2026-09-01)

All at global batch 4 (4 GPUs x batch 1, no accumulation) — the same global
batch as upstream ERDM sst_pred and the rsi_sstpred_a2 baseline.

| config | s/step | 30k steps | peak GPU mem |
|---|---|---|---|
| Midway 4xH100, bf16 | 0.826 | 6.9 h | 50.4 GB / 95.8 GB |
| DeltaAI 4xGH200, bf16 | 0.62 | 5.2 h | 50.8 GB / 120 GB |
| DeltaAI 4xGH200, fp32 (no TF32) | 2.53 | 21.1 h | 74.3 GB / 120 GB |

Consequences: (a) fp32 costs **4.1x** bf16, not the ~1.5-2x assumed when the
arm ladder was written — fp32 arms need >=24 h walltime, and a *failing* arm
still costs only ~10 h since the excursion lands at ~15k steps; (b) fp32 peaks
at 74 GB, so it fits on BOTH H100 (95 GB) and GH200 — the earlier OOM concern
was wrong and no gradient checkpointing is needed for these arms;
(c) GH200 bf16 is ~25% faster than H100 bf16.

**TF32 is part of upstream parity, not just an optimization.** amip_v2's
`train.py` sets `torch.set_float32_matmul_precision("high")`, so upstream's
`precision: 32-true` run is fp32 *storage* with TF32 matmuls. Our
train_diffusion has the same knob (`++training.matmul_precision=high`) but
defaults it OFF, so an fp32 arm without it is both slower AND less faithful
than the run it is supposed to match. Cross-check: upstream's own checkpoint
timestamps give 5 h 40 m/epoch = **1.55 s/step**, between our bf16 (0.62) and
our TF32-less fp32 (2.53) — consistent with TF32 on. All fp32 arms therefore
carry `++training.matmul_precision=high`.

**Interactive queues.** DeltaAI's `ghx4-interactive` is the same PriorityTier
as `ghx4` but nearly empty (7 pending vs 672), at a 2 h / 4-node cap — too
short for a 30k-step arm (~8.7k steps) but ideal for de-risking: the 300-step
smokes above validated the whole aarch64 path (module load, Cray CXX/CC
override, cuDNN-SDPA workaround, 4-way DDP) and produced this table before
any 24 h job entered a 672-deep queue.

## Gradient checkpointing (measured 2026-09-02, DeltaAI interactive queue)

Implemented in `RollingDiT` (`grad_checkpoint`, wrapped over all 20 spatial +
20 temporal + 4 forcing blocks with `use_reentrant=False` — required for DDP
autograd hooks and for the non-tensor b/W/n/mask args). Switched on with
`++training.grad_checkpoint=true`, which the trainer applies to the built
backbone; deliberately NOT a model-config key, so it never lands in a
checkpoint's args.json. Skipped whenever grad is off, so eval/samplers pay
nothing. `_CONTRACT_KEYS` is unaffected, so old checkpoints still load.

fp32, 150 steps, 4x GH200, global batch 4, identical seeds:

| | peak GPU mem | s/step |
|---|---|---|
| checkpointing OFF | 74,327 MiB (72.6 GB) | 2.530 |
| checkpointing ON | **16,631 MiB (16.2 GB)** | 3.320 |
| | **4.47x less, 56.3 GB saved** | **+31%** |

Loss at batches 0/100/149 is IDENTICAL between the two runs
(7.0338e+03 / 5.6206e+03 / 4.4819e+03) — a full-scale empirical confirmation
of the unit-test equivalence (test/models/amip_si/test_grad_checkpoint.py:
bitwise-identical forward, <1e-4 relative gradient agreement).

**What this unlocks.** fp32 now fits with room to spare on every 40 GB-class
card: Polaris A100-40GB (16.2 of 40 GB), Delta gpuA100x4 (40 GB) and the
~70-node gpuA40x4 pool (48 GB). That matters because campaign 2 established
fp32 is mandatory for RSI, and until now the recipe was confined to
H100/GH200. Estimated 10-epoch cost at global batch 4 (parity preserved,
1 node x 4 GPUs): ~96 h on H100/GH200, ~287 h on A100 TF32. Polaris's
`capacity` queue (1-4 nodes, **168 h**, priority 150, allocation has 34.5k
node-hours) therefore yields ~6 epochs per submission — vs the ~1.2 epochs
the pre-fix instability capped us at — and checkpoint/resume can chain
submissions for the full 10.

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
