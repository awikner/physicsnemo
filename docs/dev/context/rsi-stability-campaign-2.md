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

## RESULTS (complete 2026-09-02) — precision was the answer

Gate: no gnorm excursion (>20x a fixed 4k-9k baseline, sustained) before
30k steps; the pre-campaign baseline broke at ~15k.

| arm | change from baseline | onset | final loss | verdict |
|---|---|---|---|---|
| (baseline) | — | ~15,000 | — | reference |
| arm0 | training-side forcing-alignment audit | n/a | n/a | ✅ PASS (26/26 slots bit-exact) |
| arm1_floor | ct_floor_lr 3e-6, decay 25k | 12,380 | — | ❌ FAIL (worse) |
| arm3_uniform | weighting=uniform | 14,337 | — | ❌ FAIL |
| arm9b_wd | weight_decay 0.01 | 14,520 | — | ❌ FAIL |
| arm9c_muon10 | muon_lr_multiplier 10 (bf16) | 9,001 | — | ❌ FAIL (earliest) |
| **arm9a_fp32** | **amp=none + matmul_precision=high** | none | 455 | ✅ **PASS** |
| **arm9_upstream_match** | **fp32+TF32 + wd 0.01 + muon x10** | none | **280.5** | ✅ **PASS** |

**Conclusion: bf16 is what destabilises RSI.** fp32 (with TF32 matmul) alone
is sufficient — arm9a. Nothing else helped: the LR floor made onset *earlier*,
flattening the snr_bump weighting did nothing, and weight decay alone bought
~0. arm9c is the sharpest negative: upstream's muon x10 LR, which is stable
for them in fp32, breaks at 9k steps in bf16 with gnorm 2140x baseline — so
**fp32 is what makes their higher LR survivable**, and LR magnitude was never
the driver. Arm0 also rules out the collaborator's lag-1 forcing-alignment
hypothesis on the training side (the eval side was already bit-proven).

**Production recipe = arm9** (i.e. upstream ERDM sst_pred's own optimizer
configuration): `++training.amp=none ++training.matmul_precision=high
++training.optimizer.weight_decay=0.01
++training.optimizer.muon_lr_multiplier=10`. It converges ~38% lower than
fp32-alone (280.5 vs 455) because the x10 LR is now usable, and gnorm still
*falls* at 30k (3.51e3 baseline -> 1.31e3) — well-conditioned, not merely
surviving. Pair with activation checkpointing (below) for 40 GB cards.

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

## Batch-size scaling: does global batch 40 hurt? (measured 2026-09-04)

Polaris' `prod` queue is the only one that reliably schedules, but its minimum
is 10 nodes, which at per-rank batch 1 forces **global batch 40** (10x the ERDM
sst_pred baseline) and cuts optimizer steps 10x for a fixed epoch budget. Jobs
`7589775/6`: 40 ranks, `steps_per_epoch=1315`, 2 epochs, lr `5e-5*sqrt(10)` =
1.5811e-4, EMA decay `0.999^10`, otherwise arm9's exact overrides — including
reproducing arm9's schedule, which was `CosineToFloor` with
`ct_floor_lr == lr`, i.e. **a perfectly constant lr** (ratio 1.0). Checked
against arm9's own `.hydra/overrides.yaml`, not the summary above; matching the
production StepLR instead would have added a schedule confound.

The comparison is valid because `_train_step` returns a **rank-local** loss (no
all_reduce) and per-rank batch is 1 in both runs — single-sample losses either
way, hence medians over a +-200-step window.

| epoch | batch 4 (arm9) | batch 40 | ratio |
|---|---|---|---|
| 0.25 | 601.9 | 1250.3 | 2.08 |
| 0.50 | 453.0 | 807.4 | 1.78 |
| 1.00 | 367.5 | 493.0 | 1.34 |
| 1.50 | 331.6 | 406.5 | 1.23 |
| 1.95 | 311.3 | 370.9 | **1.19** |

After two full epochs batch 40 has not reached what batch 4 reached in **one**.
But per UPDATE it is ~50% better throughout (at 2,600 steps: 370.3 vs 692.9).
To reach loss 370.9: batch 4 needed 12,450 steps / 0.95 epochs / 11.6 h on
1 node; batch 40 needed 2,630 steps / 2.00 epochs / **2.6 h** on 10 nodes.
**4.7x fewer updates, 2.1x more data, 4.5x faster wall-clock, 2.2x the
node-hours** — it converts node-hours (abundant: 34,480 available) into
wall-clock (scarce).

**So 24 epochs at batch 40 is NOT quality-equivalent to 24 at batch 4** — on
this evidence nearer batch-4-at-~11-epochs; matching would need ~50 epochs.
Caveat: this is epochs 0-2, where large batch is penalised hardest, and the gap
was still closing fast and decelerating (slope -0.78 -> -0.27 -> -0.11 per
epoch, trending to ~1.10-1.15). Where it lands at epoch 24 is unknown and
UNKNOWABLE from arm9, which stops at 2.28 epochs. gnorm curves nearly overlap
by epoch 2 (1466 vs 1404), so batch 40 is behind on progress, not misbehaving.

### Partial checkpointing — the level sweep (measured 2026-09-04, Polaris debug)

All-or-nothing is a bad trade on a 40 GB A100: OFF needs 74 GB (does not fit),
ON costs +31%. `RollingDiT.grad_checkpoint_levels` (surfaced as
`++training.grad_checkpoint_levels=N`) checkpoints only the first N of the 20
levels; `None` = all, so old configs are unchanged. Counted from the INPUT side
so the uncheckpointed levels are the last ones, whose activations backward
frees first rather than holding them resident through the recomputes.

Job `7591890`, 4x A100-40GB, fp32+TF32, global batch 4, 30 steps per point:

| ckpt levels | peak MiB | % of card | s/step | vs full-ckpt |
|---|---|---|---|---|
| 20 (full) | 18,183 | 44.4% | 3.345 | — |
| 16 | 31,653 | 77.3% | 3.174 | +5.1% |
| **14** | **37,013** | **90.4%** | **3.103** | **+7.2%** |
| 13 | 39,691 | 96.9% | 3.069 | +8.3% |
| 12 | — | — | — | **OOM** (died after 1 step) |
| 10 / 8 | — | — | — | **OOM** (0 steps) |

**Use 14.** 13 is the true minimum and the fastest that runs, but it sits at
96.9% of the card with 1,269 MiB spare — too thin for a multi-day run once
fragmentation drift and the validation rollouts land on top. 14 keeps 3,947 MiB
of headroom for 1.1 points of the 8.3%. Note the cliff is sharp: 13 runs, 12
dies. There is no graceful degradation to lean on.

Linear and cross-checked — per-level cost 2,678–3,368 MiB, fits
`peak = 80,245 - 3,086*levels` MiB and `s/step = 2.548 + 0.0396*levels`.
Extrapolated to zero levels that is 80,245 MiB / 2.548 s/step against the
74,327 MiB / 2.530 s/step measured independently on GH200 with checkpointing
off: step time agrees to 0.7%, memory 8% high (expected — the fit is anchored
on points sitting in the allocator's fragmentation regime).

Mathematically exact, so it is safe to change between runs:
`test/models/amip_si/test_grad_checkpoint.py` pins bitwise-identical forwards
and <1e-4 relative gradients at every level count, and the ON/OFF losses were
already identical at batches 0/100/149.

**The 2026-09-04 batch-40 prod campaign ran at 20/20** (the sweep landed after
it started, and ~2 h of saving was not worth a mid-chain recipe change against
a ±20 h queue-wait uncertainty). Apply `++training.grad_checkpoint_levels=14`
to runs submitted from here on.

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
