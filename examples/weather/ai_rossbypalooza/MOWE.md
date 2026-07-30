<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MoWE Method 0 — data loader, model, losses, and how to run it

Mixture-of-AI-Weather-Experts gate for week-2 Indian monsoon rainfall. A ViT
(DiT) gate learns per-gridpoint weights and bias corrections that blend the
daily precipitation forecasts of four **frozen** AIWP experts into one field,
scored against IMERG over the IMD gauge region. Experts are never fine-tuned —
that is Method 1.

Data provenance and the harmonised store schema live in [DATA.md](DATA.md);
measured results and the reasoning behind the loss choices are in
[`docs/dev/context/mowe-loss-formulation-findings.md`](../../../docs/dev/context/mowe-loss-formulation-findings.md).

---

## 1. Data loader

`datapipes/` — one sample is one **(initialization, lead-day τ)** pair.

**Sources.** Four harmonised expert archives at 1° on IMERG's grid
(180×360, lat 89.5→−89.5 N→S), `hindcasts_mowe/{model}/{YYYY}.zarr` with dims
`(init_time, lead_time[days, 0=IC], lat, lon)`: `pangu_s2s`, `sfno_era5`,
`graphcast` (merged e2s+wb2), `aifs_single_v2`. Truth is IMERG
`total_precipitation_24hr` in mm/day.

**Sample construction** (`index.py`). The initialization universe is the
**union** across experts, not the intersection, so an init present in only some
archives is still usable. For each (init, τ) a per-expert bit records whether
that expert has the init *and* supports the lead; pairs with fewer than
`min_experts` live experts are dropped at build time, so there is no runtime
skipping and every DDP rank derives an identical index from coordinates alone.

**Day alignment.** A sample covers `[init+(τ−1)·24h, init+τ·24h)`, and the IMERG
record is stamped `date(init)+(τ−1)`. Per-expert `precip.day_offset` corrects
models whose precip is offset — **sfno needs `day_offset: 1`** because its precip
head is forward-looking (verified by correlation against IMERG).

**Channels** (`variables.py`). Every native name from either raw schema is
normalised to a canonical `(variable, level_hPa)` key *before* any channel
lookup, so the same physical field from different models always lands in the same
slot and receives the same normalisation. **Channel 0 is always precip**;
`master_channels` in the dataset config lists the dynamical predictors
(currently z500, z850, q850, u850, v850, u250, msl, sp).

**Normalisation** (`stats.py`). Dynamical channels use ERA5 stats matched by
level *value* (raising rather than silently mismatching). Precip uses shared
IMERG statistics computed in the **log-transformed** space:
`log(1e-3 + P[m/24h])`, mean −6.379, std 0.858. The transform is read from the
stats store's own attributes, so a transform/stats mismatch is impossible.

**Missing experts** are handled at three levels: dropped at index build if the
init or lead is absent; demoted at read time if the finite fraction < 0.5 or the
precip channel is all-NaN; and in the model the mask becomes E extra input
planes plus a `-inf` softmax fill, so a masked expert gets **exactly zero**
weight. `expert_dropout` (0.30) randomly drops live experts during training, so
one checkpoint serves any expert subset. Missing ≡ 0 in z-space, which is *not*
0 mm/day once inverted (≈0.70), hence `mix()` also takes the mask as a guard.

**Sample dict:** `expert_inputs (E,1+C,180,360)`, `expert_mask (E,)`,
`target (1,180,360)` normalised, `target_mm (1,180,360)` physical, plus
`lead_days`, `init_time`, `valid_time`, `pair_idx`.

**Windows.** Training leads **7–14**, validation **8–14** (week 2). Day 15 is
unusable: sfno's `day_offset` means τ=15 needs lead 16 and the harmonised stores
stop at 15. graphcast carries `min_lead_day: 8` because its wb2-sourced inits
have no complete 24 h precip window at 168 h.

---

## 2. Model

`mowe_precip.py` — `MoWEPrecipGate(DiT)`, **3.81M parameters**
(`hidden_size 192`, `depth 4`, `num_heads 6`, `patch_size 4×4` → 4050 tokens;
`drop_path 0.15`).

Input `(B, E, 1+C, H, W)` is folded to `E·(1+C) + E` channels — every expert
block plus E constant mask planes, so the gate is *told* which experts are
absent. Output is `2E` channels split into logits and biases:

```
weights = softmax( logits.masked_fill(mask == 0, -inf) )   # over experts
P̂       = Σᵢ wᵢ · (Pᵢ + bᵢ)                                 # mix(), outside the model
```

Lead time τ enters through DiT's conditioning path. Mixing lives **outside** the
model so validation can log the weight maps. `model.mix_space` controls the space
the combination happens in and **must stay `physical`**: combining standardized
log channels makes the ensemble a weighted *geometric* mean (2 and 50 mm/day →
11.4 instead of 26.0, a 2.3× compression, and a −3.5 mm/day July bias). `log` is
retained only as an ablation.

Capacity note: the earlier 384/8 gate (24.0M params) overfit badly against only
~2,900 independent initializations — validation loss bottomed at epoch 10 then
rose 18% while training loss fell 54%.

---

## 3. The two losses that worked

Both take the error over the **training region** (monsoon box 5–35°N/60–100°E ∩
IMD gauge coverage = 378 gridpoints, cos-lat weighted) and both mix in mm/day.
`scale_mm: 9.3` divides the physical MSE so it lands near 1.0 and the tuned
learning rate and gradient clipping carry over — pure rescaling, the optimum is
unchanged.

### `loss=regional_mse_physical` — best aggregate skill

```yaml
name: regional_mse   # space: physical, scale_mm: 9.3
```

Squared error in mm/day, so it elicits the conditional **mean** and is unbiased
by construction. **Best RMSE and ACC of anything tested**, but it wins them by
hedging intensity: amplitude ratio 0.392 against ACC 0.337, i.e. essentially the
MSE-optimal shrinkage σ_p = r·σ_t. It produces **4.8% of observed 50 mm/day
events** and its CSI at 20 mm is *below equal-weight's*.

### `loss=regional_mse_physical_var` — recommended

```yaml
name: regional_mse   # space: physical, scale_mm: 9.3, var_weight: 1.0
```

Adds `var_weight · (σ_pred/σ_obs − 1)²` on the region-weighted **spatial**
standard deviation per sample, forbidding the hedging. It is the only arm that
beats equal-weight on RMSE *and* heavy-rain frequency *and* CSI simultaneously,
and it posts the best gate SEEPS (0.839, effectively tying AIFS at 0.836).
`var_weight` 3.0 overshoots — it breaks the equal-weight RMSE floor and collapses
ACC — so 1.0 is bracketed as the sweet spot.

| Loss | RMSE | bias | ACC | amp | SEEPS | exc_bias 20mm | CSI 20mm |
|---|---|---|---|---|---|---|---|
| `regional_mse_physical` | **8.96** | −0.18 | **0.337** | 0.392 | 0.853 | 0.400 | 0.164 |
| `regional_mse_physical_var` | 9.13 | −0.16 | 0.313 | 0.477 | **0.839** | 0.611 | **0.196** |
| equal_weight (bar) | 9.43 | −0.02 | 0.272 | 0.537 | 0.871 | 0.588 | 0.173 |
| aifs_single_v2 (bar) | 9.91 | 0.41 | 0.310 | 0.755 | 0.836 | 0.859 | 0.211 |

Also available: `regional_mse` (log space, `space: normalized`),
`regional_mse_bias` (log space + `bias_weight`), `regional_log_mse`. Both
log-space arms were clearly worse (RMSE 9.37–9.44, ACC ≈0.29). **MAE would not
help**: absolute error elicits the median (3.65 mm/day against an arithmetic mean
of 9.32 on IMERG July), and the median is invariant under monotone transforms.

**Loss values are not comparable across these configs** — different objectives.
Compare the metrics.

---

## 4. Train

Per-cluster wrappers, one GPU node each:

```bash
# Midway3, 4x H100 on the dedicated pedramh-gpu node
sbatch --export=ALL,RUN_NAME=mowe_prod,WANDB=true,\
EXTRA="loss=regional_mse_physical_var" tools/train_mowe_midway3_h100.sbatch

# Derecho, 4x A100
qsub -v RUN_NAME=mowe_prod,WANDB=true,EXTRA_FILE=/path/overrides.txt \
     tools/train_mowe_derecho.pbs
```

**Overrides containing commas must go through `EXTRA_FILE`.** Both Slurm's
`--export` and PBS's `-v` are comma-separated lists, so
`dataset.train.years=[2000,2024]` is silently truncated and Hydra then fails with
`no viable alternative at input`.

Defaults (`conf/training/default.yaml`): AdamW lr 3e-4 with 2-epoch warmup and
cosine decay, `max_epochs 40`, bf16, grad-clip 1.0, weight decay 0.15,
`expert_dropout 0.30`, EMA (decay 0.999, validation and best-checkpoint use EMA
weights), early stopping on the validation loss with patience 8.

Checkpoints land in `<rundir>/outputs/<run_name>/`:
* `checkpoints/` — every 5 epochs plus the final epoch, for resuming. **The final
  epoch is the most overfit; do not ship it.**
* `checkpoints_best/` — best validation loss. **Use this one.**

Resume by re-submitting with the same `RUN_NAME`.

---

## 5. Validate

Validation runs inside training every epoch and reports, for the gate, the
equal-weight ensemble, and each expert separately, all over the 378-gridpoint
training region in mm/day:

* per lead 8–14 and a mean: `rmse_lead{τ}`, `bias_lead{τ}`, `seeps_lead{τ}`
* per calendar month pooled over validation years:
  `imd_{rmse,bias,acc,seeps,amp}_{MM}` and `_mean`
* intensity-resolved at 1/5/10/20/50 mm/day: `exc_bias_{T}mm` (frequency bias
  P(pred>T)/P(obs>T)) and `csi_{T}mm`
* `loss` — the training criterion on the validation split, pairing with
  `train/loss`; plus `{source}/loss` per baseline

Two definitions to keep straight. **ACC uses a day-of-year climatology**
(`clim_mean_daily`, ±7-day smoothed) — a monthly reference leaves the monsoon
onset in both forecast and observed anomalies and inflates the correlation
(central India runs 3.06 → 8.02 mm/day within June alone). And **`amp` =
σ_pred/σ_obs** is the shrinkage diagnostic: ≈1 preserves intensity, ≈ACC means
the loss is hedging.

To re-validate a checkpoint without training, run inference (below) and score the
saved forecasts, or re-submit training with `training.max_epochs` equal to the
resumed epoch.

### Best checkpoints

Single-split runs (train 2000–2019, validate 2020–2024 Mar–Sep), on Midway3:

| Loss | Path (`/scratch/midway3/awikner/mowe_runs/outputs/…`) | Best epoch |
|---|---|---|
| `regional_mse_physical_var` (recommended) | `mowe_v6_var1/checkpoints_best/` | 29 |
| `regional_mse_physical` | `mowe_v6_physmse_ref/checkpoints_best/` | 12 |

5-fold cross-validation (20 train / 5 held-out years per fold, per-fold refitted
normalisation and climatology) on Derecho:

```
/glade/derecho/scratch/awikner/mowe_runs/outputs/mowe_cv{1..5}_phys/checkpoints_best/
/glade/derecho/scratch/awikner/mowe_runs/outputs/mowe_cv{1..5}_physvar/checkpoints_best/
```

Fold *k* holds out: 1 → 2000-04, 2 → 2005-09, 3 → 2010-14, 4 → 2015-19,
5 → 2020-24. Each contains `MoWEPrecipGate.0.<epoch>.mdlus` and
`checkpoint.0.<epoch>.pt`; `load_checkpoint` picks the latest automatically.

Both verified to exist (2026-07-30). `checkpoints_best/` keeps every
best-so-far checkpoint rather than only the final one, so the highest epoch
number is the best; `load_checkpoint` selects it automatically.

---

## 6. Infer and save gate forecasts

`tools/infer_mowe.py` loads a checkpoint, replays a split deterministically
(single process — I/O bound, no DDP), and writes a dense zarr. Run it on a GPU
node (**Midway's H100 node for inference tests**), not a login node.

```bash
python tools/infer_mowe.py \
    dataset=hindcast_derecho \
    +checkpoint=/glade/derecho/scratch/awikner/mowe_runs/outputs/mowe_cv5_physvar/checkpoints_best \
    +out=/glade/derecho/scratch/awikner/mowe_forecasts/cv5_physvar.zarr \
    +split=val +save_gate=true
```

Output `(init_time, lead_time, lat, lon)`:

* `total_precipitation_24hr` — the mixture in **mm/day**.
* with `+save_gate=true`, additionally `(init_time, lead_time, expert, lat, lon)`:
  * `gate_weights` — masked-softmax weight per expert, summing to 1 over live
    experts and exactly 0 for a masked one. **These are the fields to inspect for
    the monsoon-structure question** (active/break phases, orographic vs.
    depression rainfall).
  * `gate_biases` — the learned per-expert additive correction, in the *mixing*
    space: mm/day under the default `mix_space=physical`, standardized-log
    offsets under `log`.

`P̂ = Σᵢ wᵢ(Pᵢ + bᵢ)`, so the two gate arrays plus the harmonised expert stores
reproduce the forecast exactly. Pairs absent from the index (IMERG gaps, too few
live experts) stay **NaN** — that is not a zero forecast. The gate arrays are E×
the forecast's size, hence off by default.

> **Only the supervised region is meaningful.** The gate emits fields at all
> 64,800 gridpoints but is trained on 378, and outside them the output is
> unconstrained extrapolation — measured on a real run, the biases average
> **−15.3 mm/day (1st percentile −82)** outside versus **−0.52 mm/day** inside.
> Plotting the global bias field would be badly misleading. The store records the
> region in its `supervised_region_box` / `supervised_region_note` attributes;
> mask with `region_mask(lat, lon, box) & imd_valid_mask(...)` before analysis.

Measured on the recommended checkpoint over the 2020–2024 Mar–Sep split
(3,255 pairs, 465 inits, ~45 min on one H100 — each pair is a separate zarr
region write, so buffer in memory if that becomes a bottleneck):

| | forecast precip | gate bias |
|---|---|---|
| inside the region | mean 5.56, median 2.45, p99 32.2 mm/day | mean −0.52, p99 +8.4 mm/day |
| outside (untrained) | mean 1.05, median 0.00 | mean −15.27, p1 −82.5 |

**Learned weights inside the region** (a first look at the monsoon-structure
question): graphcast **0.475**, aifs_single_v2 0.215, sfno_era5 0.184,
pangu_s2s 0.126. The gate concentrates on graphcast and downweights pangu, which
is also the weakest expert by ACC (0.104) — so it is discriminating on skill
rather than averaging blindly.

To score saved forecasts against IMERG, reuse the streaming accumulators in
`validation.py` (`StreamingRegionalScore`, `StreamingMonthlyScores`,
`StreamingThresholdScores`) with `region_weights(lat, lon, box,
extra_mask=imd_valid_mask(...))` so the region matches training exactly.

---

## 7. Tests

```bash
pytest test/recipes/ai_rossbypalooza/ -m ""     # 166 tests; -m "" includes slow
```

Covers cross-schema channel identity, every precip unit/axis/offset combination,
conservative regridding, index union and gap handling, masked-softmax exactness,
the arithmetic-vs-geometric mixing distinction, the loss-ranking flip the
variance term is built to cause, EMA round-trips, early stopping, the day-of-year
ACC reference (a case where ACC flips sign against a monthly reference), the
intensity thresholds, and an end-to-end train→infer round trip.

## 8. Not yet built

* The plan's **primary** target is week-2 *accumulated* precip as tercile and
  exceedance probabilities; this trains and scores *daily* precip (the secondary
  target).
* The gate is deterministic (`noise_dim: null`), so BSS, RPSS, AUC, CRPS and rank
  histograms cannot be computed. A quantile/CRPS head is the natural next step.
* No FSS, no block-bootstrap significance.
* AIFS alone still beats every gate variant on July ACC (0.408 vs 0.355),
  amplitude, heavy-rain frequency, and CSI. A convex combination cannot exceed
  its inputs' amplitude, so the per-expert bias fields are the only lever with
  real headroom there.
