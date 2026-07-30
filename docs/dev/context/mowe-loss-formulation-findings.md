# MoWE Method-0: loss formulation, intensity, and Midway gotchas

Durable findings from the 2026-07-29/30 loss sweep on the `ai-rossbypalooza-mowe`
branch. Recipe: `examples/weather/ai_rossbypalooza/`. All numbers are the
IMD-coverage region (378 gridpoints = monsoon box ∩ IMD gauge mask), daily-precip
target, leads 8–14 for validation, best epoch of each run.

## The central result: RMSE and intensity pull in opposite directions

| Loss | RMSE | bias | ACC | amp | SEEPS | exc_bias 20mm | exc_bias 50mm | CSI 20mm |
|---|---|---|---|---|---|---|---|---|
| physical MSE | **8.96** | −0.18 | **0.337** | 0.392 | 0.853 | 0.400 | 0.048 | 0.164 |
| physical MSE + var 1.0 | 9.13 | −0.16 | 0.313 | 0.477 | **0.839** | 0.611 | 0.089 | **0.196** |
| physical MSE + var 3.0 | 9.48 | −0.05 | 0.271 | 0.567 | 0.863 | 0.718 | 0.161 | 0.195 |
| log-MSE + bias 0.10 | 9.37 | −0.69 | 0.294 | 0.511 | 0.912 | — | — | — |
| log-MSE + bias 0.50 | 9.44 | −0.25 | 0.291 | 0.540 | 0.892 | — | — | — |
| equal_weight (bar) | 9.43 | −0.02 | 0.272 | 0.537 | 0.871 | 0.588 | 0.135 | 0.173 |
| aifs_single_v2 (bar) | 9.91 | 0.41 | 0.310 | 0.755 | 0.836 | 0.859 | 0.385 | 0.211 |

**Plain physical MSE wins RMSE and ACC by refusing to forecast heavy rain.** It
produces 4.8% of observed 50 mm/day events and 40% of 20 mm/day events, and its
CSI at 20 mm is *below equal-weight's*. Aggregate scores alone would have
selected it; the intensity metrics are what expose it.

**Why this is structural, not a training bug.** For region-weighted anomalies,

    MSE = (Δmean)² + (σ_p − σ_t)² + 2·σ_p·σ_t·(1 − r)

Shrinking σ_p toward 0 deletes the decorrelation term, so shrinking *lowers* MSE
whenever r < 0.5, and the MSE-optimal amplitude is σ_p = r·σ_t. Measured: the
physical-MSE run converged to amp 0.392 with ACC 0.337 — within 16% of that
theoretical optimum. The intensity compression is the objective working as
specified.

**Chosen model: physical MSE + `var_weight` 1.0.** Only arm beating equal-weight
on RMSE *and* heavy-rain frequency *and* CSI simultaneously, with the best gate
SEEPS (0.839, effectively tying AIFS at 0.836). `var_weight` 3.0 brackets the
optimum from above: it breaks the equal-weight RMSE floor and collapses ACC.

**Still unbeaten by the blend:** AIFS alone leads July ACC (0.408 vs 0.355),
amplitude (0.755), heavy-rain frequency, and CSI. Note a convex combination
cannot exceed its inputs' amplitude, so reweighting alone caps the gate near
AIFS's 0.755; the per-expert bias fields are the only lever with headroom.

Expert amplitudes for reference (how much intensity each model arrives with):
aifs 0.755, sfno 0.748, graphcast 0.685, pangu 0.571 — and equal-weight lands at
0.537, *below every individual expert*, which is the variance reduction from
averaging imperfectly-correlated fields. **The blending destroys intensity that
the experts had.**

## Log space vs physical space

Mixing must happen in **mm/day**. Mixing standardized log channels makes the
ensemble a weighted *geometric* mean: for two experts at 2 and 50 mm/day that is
11.4 rather than the arithmetic 26.0, a 2.3× compression, and it produced a
−3.5 mm/day July bias. `model.mix_space` selects it; `log` is retained for
ablation only.

The loss is a separate choice from the mixing. A log-space MSE elicits the
conditional geometric mean (4.06 mm/day on IMERG July over the region, against
an arithmetic mean of 9.32 — a −56% dry bias). **MAE would not help**: absolute
error elicits the median, which is lower still at 3.65 mm/day (−61%), and the
median is invariant under monotone transforms so applying MAE in log space
changes nothing. Bias penalties (`bias_weight`) treat the symptom; moving the
error into physical space fixes the cause.

## Metric definitions that bit us

* **ACC must reference a day-of-year climatology**, not a monthly one. In central
  India the daily climatology runs 3.06 mm/day on 1 June to 8.02 on 30 June
  while the monthly value flattens it to 5.87 — that entire onset signal was
  sitting in both the forecast and observed anomalies, inflating the correlation
  ~4% overall and most in June. `clim_mean_daily` (±7-day smoothed) is the
  reference; the monthly `clim_mean` is a warned fallback.
* **Score only where the model is supervised.** The gate emits weights at all
  64,800 global gridpoints but is trained on 378, so box-wide metrics were 68%
  untrained extrapolation. All metrics now use the training region.
* **SEEPS coverage is thin in peak monsoon**: the standard p1 ∈ [0.1, 0.85]
  filter leaves 198/378 gridpoints in July and 233 in August, so monthly SEEPS
  rests on ~52% of the region there — a different, noisier sample than
  RMSE/bias/ACC.

## Overfitting (fixed)

The original 384-wide/8-deep gate (24.0M params) overfit against only ~2,900
independent initializations: validation loss bottomed at epoch 10 then rose 18%
by epoch 32 while training loss fell 54%. Fixed by 192/4 (3.81M params, 6.3×
smaller), `drop_path` 0.15, weight decay 0.15, expert dropout 0.30, EMA, and
early stopping with a separate `checkpoints_best/`. All subsequent runs
early-stop on patience around epochs 20–29 rather than degrading. Note the old
setup shipped its *last* epoch, i.e. its most overfit weights.

## Data-window constraints

Training leads are **7–14**, not 7–15. Day 15 is unusable because sfno's precip
head is forward-looking (`day_offset: 1`), so τ=15 needs lead 16 and the
harmonized stores stop at 15 — including it silently masks sfno on every such
sample. Day 7 works for all four experts, but graphcast's wb2-sourced inits have
no complete 24 h precip window at 168 h, so it carries `min_lead_day: 8`;
without that, wb2-only inits at τ=7 left samples with *zero* live experts and the
model raised mid-epoch.

## Midway operational gotchas

* **`/scratch/midway2` is not mounted on every node.** `midway3-0102` has it,
  `midway3-0025` does not, and login nodes never do. The failure is deceptive:
  tools report "no finite data found" and warn that every year is missing, which
  reads as corrupted data. Both the training sbatch and the CV stats jobs now
  precheck the mount and exit 75 with the hostname. Pin midway2-touching CPU
  jobs to a known-good node.
* **Do not pin CPU helper jobs to `pedramh-gpu`** — the training job holds all 32
  cores and 480 GB, so they never schedule (and an interactive `srun` queues
  behind them).
* **Slurm's `--export` list is comma-separated**, so any value containing a comma
  (e.g. `dataset.train.years=[2000,2024]`) is silently truncated and Hydra then
  fails with "no viable alternative at input". Pass such overrides in a file via
  `EXTRA_FILE`.
* **Hydra struct mode** refuses to override a key absent from the config, hence
  `exclude_years: []` is declared in the dataset configs rather than appended
  with `+` on the CLI.
* The `gpu` partition is 5×V100 + 5×RTX6000 + 1×A100 and usually saturated;
  `pedramh-gpu` is a dedicated 4×H100 node (`-A pi-pedramh`) and is ~7× faster
  than 3×RTX6000 at fp32 (≈3.5 min/epoch vs 26).

## Open gaps (not addressed)

* The **primary target is week-2 accumulated precip** (lead 8–14 weekly total) as
  tercile and exceedance probabilities; we train and score *daily* precip, the
  plan's secondary target.
* The gate is **deterministic** (`noise_dim: null`), so BSS, RPSS, AUC, CRPS and
  rank histograms cannot be computed.
* No FSS, no block-bootstrap significance.
* Model selection used one fixed 2000-2019/2020-2024 split; 5-fold CV
  (`tools/run_cv_midway3.sh`) was built to replace that. 2025 and the 1965–1978
  pre-satellite period remain untouched holdouts.
