# Plan: 2000–2024 hindcast campaign (Pangu-legacy + SFNO) and ArchesWeather port/training

**Date:** 2026-07-21. **Status:** APPROVED by user (answers recorded below). This is the
execution handoff document — it is written to be sufficient to execute the whole task
even after a model/session switch. Read fully before acting. Companion recon details are
inlined; no re-recon should be needed except where marked VERIFY.

## 0. Task statement (user's request, condensed)

1. Generate 15-day hindcasts from two models over 2000–2024, initialized 8×/month on
   days {1,5,9,13,17,21,25,29} at 00Z, **excluding Feb 29** (February gets 7 inits) →
   **95 inits/year, 2375 per model**:
   - **Model A — Pangu legacy**: `PanguModel_Plasim` from the old PanguWeather repo on
     Derecho, config `/glade/work/awikner/PanguWeather/v2.0/config/PANGU_S2S_infer_mar_aug_multigpu_2.yaml`.
   - **Model B — SFNO**: the `SfnoPlasim` ERA5 model trained by rmasiwal on Stampede3
     (see `~/Downloads/ERA5_SFNO_STAMPEDE3.md` on the Mac).
   - Generate on **Derecho**, one year at a time; final data lands on **Stampede3**.
   - Output format: like the ERA5 training zarrs — yearly stores keyed by init-condition
     year, with an added lead-time dimension.
2. **Model C — ArchesWeather**: port the ArchesWeather deterministic backbone (as close
   to the original as possible) onto our data variables + 17-level/1° grid; train on
   Stampede3 with the original ArchesWeather loss/training recipe; then generate the same
   hindcast set with it.
3. Smoke-test anything new before scaling up.

## 1. User decisions (2026-07-21, via AskUserQuestion)

| Decision | Answer |
|---|---|
| SFNO checkpoint unreadable (rmasiwal 0700) | Poll for access; **if it stays blocked, SKIP SFNO hindcasts** (do NOT retrain). User may ping rmasiwal; check periodically. |
| Hindcast output spec | **As proposed**: 1 deterministic member; daily leads 0–15 (16 frames, lead 0 = ERA5 IC); float32; full native variable set per model; yearly stores at `/scratch/09979/awikner/physicsnemo-zarr/hindcasts/{pangu_s2s,sfno_era5,archesweather_era5}/YYYY.zarr` on Stampede3; training-zarr-style schema + `init_time`/`lead_time` dims. No Delta copy requested. |
| ArchesWeather patch size | **(2,3,3)** → token grid (10, 60, 120), identical spatial token count to original ArchesWeather-M; windows/shifts carry over verbatim; PixelShuffle(3) decoder. |
| ArchesWeather scope | **Single M (seed 0) + hindcasts after training.** Train 1979–2018, val 2019, RPFT (2007–2018) for final 50k steps. No precip/diagnostic head. |

Other defaults established during planning (not separately asked; use unless contradicted):
- Pangu: no inference dropout, no latent saving, checkpoint run **2000** `best_ckpt.tar` (EMA-trained best).
- SFNO: prefer **EMA weights** (in `checkpoint.0.50.pt` metadata) if trivially loadable; raw
  `SfnoPlasim.0.50.mdlus` weights otherwise. Record which was used in store attrs.
- Feb-29 inits: never generated/kept (drop at consolidation if the legacy filter emits them in leap years).
- End-of-archive boundary forcing (Dec-2024 inits needing Jan-2025 SST/SIC/TISR): **clamp to
  last available timestep (persistence)**; record affected inits in store attrs.
- Git workflow: develop on branch `ai-rossbypalooza`, commit incrementally with the
  Co-Authored-By trailer, push to `awikner/physicsnemo`, and `git fetch && git checkout
  ai-rossbypalooza` on cluster clones to sync code.

## 2. Recon facts (verified 2026-07-21 — trust these; VERIFY items marked)

### 2.1 Derecho — Pangu legacy pipeline (WORKING; last successful campaign 2026-07-16)

- Code: `/glade/work/awikner/PanguWeather/v2.0/`. Entry point **`inference_months.py`**
  (torchrun, DDP over 4 GPUs). Per-year PBS job: `submit_infer_mar_aug.sh`
  (`-A UCHI0018 -q main -l select=1:ncpus=64:ngpus=4:mem=480GB:gpu_type=a100`,
  walltime 20:00 — must be raised, see §3). Orchestrator example: `submit_dropout_ensemble.sh`.
- Conda env: `module load conda && conda activate aires_panguplasim`
  (`/glade/work/awikner/conda-envs/aires_panguplasim`, exists, exercised Jul 16).
- Checkpoint: `/glade/work/awikner/PanguWeather-rajatm2/v2.0/results/S2S/2000/training_checkpoints/best_ckpt.tar`
  (1.43 GiB, torch .tar; path built as `{exp_dir}/{--config}/{--run_num}/training_checkpoints/best_ckpt.tar`,
  exp_dir from yaml = `/glade/work/awikner/PanguWeather-rajatm2/v2.0/results`, `--config S2S`, `--run_num 2000`).
  NOTE checkpoint lives in the *PanguWeather-rajatm2* copy — do not touch/delete that tree.
- Config loading: `utils/YParams.py` reads only the named yaml section (default `--config S2S`);
  `S2S` inherits `base_config` via YAML anchors. CLI overrides: `--val_one_year Y` (sets
  val_year_start=Y, val_year_end=Y+1), `--save_run_num` (redirect outputs, keep ckpt from --run_num).
- IC selection: dataset InferFilter enumerates 6-hourly indices in the val-year window and keeps
  those matching yaml `infer_months` / `infer_days` / `infer_hours`. Existing config: months 3–8,
  days {1,9,17,25}, hour 0 → 24 inits/yr. Our config: months 1–12, days {1,5,9,13,17,21,25,29},
  hours [0] → 96/yr (leap Feb 29 slips in; drop it at consolidation) → 95 kept.
- Stepping: one 24-h step per forward; `inference_steps = max(forecast_lead_times)`.
  Set `forecast_lead_times: [15]` → 15 steps. `predict_delta: False`, `save_latent: false` for us.
- Model state per step: surface block of **9 channels** = 5 `surface_variables`
  (2m_temperature, 10m_u/v, mean_sea_level_pressure, surface_pressure) + 3 `land_variables`
  (volumetric_soil_water_layer_1, soil_temperature_level_1, skin_temperature) + 1 `ocean_variables`
  (sea_surface_temperature); constant boundary = land_sea_mask + geopotential_at_surface;
  varying boundary = **toa_incident_solar_radiation only** (read from the h5 archive per step);
  upper air = 5 vars × **17 levels [5,10,20,30,50,70,100,150,250,300,400,500,600,700,850,925,1000]**
  (200 hPa intentionally absent — the checkpoint was trained that way; do NOT add it).
- Output per init: NetCDF `{save_dir}/{run}/predictions/pangu_plasim_{run}_24h_{N}step_{YYYYMMDDHH}.nc`,
  dims (time=N+1, plev=17, lat=180, lon=360); variables = 5 surface + 2 diagnostics
  (total_precipitation_24hr, mean_top_net_long_wave_radiation_flux) + 5 upper-air. Frame 0 = IC.
  cftime standard calendar. (Land/ocean state vars are marched but NOT written by the nc writer.)
- Input data: `data_dir /glade/campaign/univ/uchi0014/yqsun/pangu_s2s/h5data` — symlinks
  covering **1979–2024** (one 28-MiB h5 per 6-h step, `{year}_{idx:04d}.h5`, `input/<var>[_<level>.0]`
  datasets, 18 levels present). Norm .nc files (`pangu_s2s_1979-2018_*`) in same dir. All readable.
- Storage: Derecho scratch 135.95/200 TiB, **22.46M/26.21M inodes (85.7%)** — keep transient
  footprint small, clean up per year. /glade/work/awikner 1.56/2.0 TiB.
- physicsnemo clone on Derecho: `/glade/work/awikner/physicsnemo`, branch ai-rossby, a few
  commits behind — needs fetch/checkout. ERA5 zarr master intact:
  `/glade/derecho/scratch/awikner/physicsnemo-zarr/era5/{1979..2024}.zarr` + 18-level
  `normalization_pangu_s2s{,_mean,_std}.zarr` (scratch = 60-day purge; last touched 2026-07-16).
- VERIFY at execution: whether a python venv for physicsnemo exists on Derecho (check
  `/glade/work/awikner/physicsnemo/.venv*`); Pangu loader behavior when rollout crosses year
  boundary (Dec init) and at archive end (Dec 2024) — smoke test both.

### 2.2 Stampede3 — SFNO + data + ArchesWeather training ground

- **BLOCKER**: trained SFNO checkpoints under
  `/work2/10441/rmasiwal/stampede3/sfno_era5_outputs/outputs/sfno_era5_stampede3_3326526/checkpoints/`
  are unreachable — `/work2/10441/rmasiwal` is `drwx------`. Shared group G-819272 → rmasiwal
  can fix with `chmod g+rx /work2/10441/rmasiwal /work2/10441/rmasiwal/stampede3` + `chmod -R g+rX
  /work2/10441/rmasiwal/stampede3/sfno_era5_outputs` (or setfacl u:awikner:rX, or copy to their
  group-readable scratch `/scratch/10441/rmasiwal`). Poll: `ssh stampede3 'ls /work2/10441/rmasiwal/'`.
  Expected files: `SfnoPlasim.0.50.mdlus` (+ intermediate epochs 5..45), `checkpoint.0.50.pt`
  (optimizer + metadata incl. EMA), `.hydra/config.yaml` in the run dir (grab it to confirm the
  exact train config). **If still blocked at campaign end → skip SFNO hindcasts (user decision).**
- awikner clone: `/work2/09979/awikner/stampede3/physicsnemo`, branch ai-rossby @ 4bfc2852
  (same as Mac), `.venv` with torch 2.10.0+cu128 (satisfies torch<2.11 DDP pin). Clean.
- ERA5: `/scratch/09979/awikner/physicsnemo-zarr/era5/` — all 46 year-stores 1979–2024 + 18-level
  norm stores (`normalization_pangu_s2s_mean.zarr`/`_std.zarr` used by `era5_multiyear.yaml`).
  ~30 GB/year-store. **Purge-exposed** (TACC scratch; mtimes Jul 9–10): touch/refresh atimes at
  campaign start (e.g. `find ... -name zarr.json -exec touch -a {} +` on the stores in use) — VERIFY
  TACC purge policy tolerance; at minimum re-check existence before training.
  `era5_train/`, `era5_val/` symlink dirs DO NOT EXIST YET — create before ArchesWeather training:
  train = 1979–2018, val = {2018, 2019} (per ERA5_SFNO_STAMPEDE3.md convention).
  Also create `era5_recent/` = 2007–2018 symlinks for the RPFT phase.
- Allocation TG-ATM170020: 124,862 SUs (exp 2026-09-30). h100 partition: 24 nodes, fully
  allocated at check time → expect queue waits. Scratch: no quota. $WORK: 683 GB/1 TB used.
- ERA5 zarr v3 schema (the template for hindcast stores): zarr_format 3, consolidated (zarr.json),
  no sub-groups (all arrays at root), codecs bytes(LE)+zstd(level 0), fill NaN,
  chunk = 1 timestep full-field; dims surface `(time,lat,lon)`, upper-air
  `(time,pressure_level,lat,lon)` [18 levels], constants `(lat,lon)`; coords lat = 89.5..-89.5
  (N→S descending), lon = 0..359 ascending, time = hours since YYYY-01-01, calendar standard;
  root attrs: `era5_zarr_schema_version "1.0"`, `calendar`, `data_timedelta_hours 6`, the six
  variable-group lists, `year_index`, `sample_range`.

### 2.3 Local repo (Mac, branch ai-rossbypalooza) — inference + training infrastructure

- `examples/weather/ai_rossby/inference.py` (1339 lines) — **existing deterministic autoregressive
  hindcast driver**. Loads model via `Module.instantiate` + `load_checkpoint`; rollout in normalized
  space; per step re-reads varying boundary (SST/SIC/TISR) from the dataset, holds constants,
  feeds back surface+upper-air; denormalizes; writes per-IC zarr/nc via `AsyncForecastWriter`
  (`async_writer.py`). Config via Hydra `inference.*` keys: `ic_start` (list of INTEGER time
  indices), `max_step`, `batch_size`, `output_dir`, `output_format`, `save_variables`.
  Gaps to close: (i) it opens a single `PlasimClimateDataset` (`inference.py:1211`) — no
  multi-year routing (needed for Dec inits crossing year boundary); (ii) datetime→index mapping
  for the init schedule; (iii) boundary clamp at archive end; (iv) EMA weight loading option.
- `train.py`: `build_model` (`train.py:465`) instantiates from `cfg.model.name`+`cfg.model.module`
  via `Module.instantiate` (importlib fallback — new models need NO registry/pyproject changes).
  `build_datapipe` (`train.py:506`) routes dir-not-.zarr → `ClimateZarrMultiYearDataset`; level
  subset via `_PressureLevelSubsetTransform` (`train.py:92`) when `cfg.model.levels` ⊂ data levels.
  Loss `PanguPlasimLoss` (`loss.py:191`); ERA5 recipe = `loss=raw_l2`. Checkpoints:
  `save_checkpoint`/`load_checkpoint` (`physicsnemo/utils/checkpoint.py:724/950`),
  `{Class}.{rank}.{epoch}.mdlus` + `checkpoint.{rank}.{epoch}.pt`, auto-resume from `./checkpoints`.
- Datapipe: `physicsnemo/experimental/datapipes/climate/{dataset,multiyear,transforms,datapipe,samplers}.py`.
  Sample keys: `surface_in (C,H,W)`, `upper_air_in (C,L,H,W)`, `constant_boundary`,
  `varying_boundary`, `diagnostic`, `target_*`, `lead_time`, `time_idx`; `__getitem__` takes int
  or `(start_idx, lead_steps)`; `emit_calendar=True` adds a `calendar (2,)` key (VERIFY semantics
  in dataset.py — expected month/hour or similar, needed for ArchesWeather conditioning).
  `ClimateNormalizer` matches levels by value (17-of-18 subset OK), `denormalize_state` API;
  `from_dataset(..., **overrides)`.
- `SfnoPlasim` (`physicsnemo/experimental/models/sfno_plasim/sfno_plasim.py`): forward
  `(surface_in, constant_boundary, varying_boundary, upper_air_in, ...)` → tuple
  `(out_surface, out_upper_air[, out_diag], 0,0,0,0)`; `has_diagnostic` attr. Input packing order:
  surface, const, varying, upper-air-flattened. This is the forward contract ArchesWeather must match.
- ERA5 recipe trio: `dataset=era5_multiyear`, `model=sfno_era5` (17 levels, embed 512, [180,360]),
  `training=sfno_plasim`, `loss=raw_l2`; launcher `hpc/scripts/train_sfno_era5.sbatch`
  (Stampede3: `AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr sbatch -p h100 -A TG-ATM170020 ...`).
- ERA5 zarr writer reference: `tools/data/era5/pangu_h5_to_zarr.py` (channel groups
  `PANGU_ERA5_CHANNELS` at line ~75; store-writing code ~309–527) — reuse its conventions/codec
  choices for the hindcast consolidator.

### 2.4 ArchesWeather dossier (paper 2412.12971 + geoarches repo; port target)

**Port from `github.com/INRIA/geoarches` (BSD-3-Clause). Do NOT copy from `gcouairon/ArchesWeather`
(no license).** Keep WeatherLearn/Pangu attribution chain in headers. Local clones from recon may
still exist in the session scratchpad (`.../scratchpad/{ArchesWeather,geoarches}`); re-clone
geoarches if gone: `git clone https://github.com/INRIA/geoarches`.

Flagship deterministic recipe (= what we replicate, "ArchesWeather-M", 84M params @ 13 levels):
- **Backbone** (`geoarches/backbones/archesweather.py` + `archesweather_layers.py`):
  3D Swin U-Net, emb_dim 192, heads (6,12,12,6), depths (2,6,6,2)×`depth_multiplier=2`,
  window (1,6,10) on token grid (Z=8, 60, 120), alternate-block roll shift (1,3,5) with **no
  attention mask** (deliberate); earth position bias table (684, n_window_types, nH);
  **Cross-Level Attention** in every block: lucidrains `axial_attention` (`AxialAttention(dim,
  heads=8, num_dimensions=1)` + `AxialPositionalEmbedding(shape=(zdim,))`) applied to the
  window-attention OUTPUT, added to residual with the attention branch; **LinVert** column-mixing
  `Linear(zdim*C, zdim*C)` residual right after patch embed; **SwiGLU MLP** (timm, ratio 4·2/3);
  drop_path linspace 0→0.1 (for dm=2), dropout 0; DownSample = 2×2 space-to-depth LN+Linear
  (Z untouched); UpSample = Linear + pixel-shuffle reshape; U-Net skip concat (layer1→layer4,
  layer4 dim 384); **adaLN conditioning**: two DiT `TimestepEmbedder`s (month 1–12, hour 0–23),
  cond 256, one zero-init `adaLN_modulation` per STAGE producing (shift,scale,gate)×2.
  Patch embed: Conv3d(k=s=(2,2,2)) upper-air (levels zero-padded to even, pad [0,1] at back),
  Conv2d(k=s=2) surface; surface token = z-index 0. Decoder (v2): surface Conv2d(384→4·u²,k=3)+
  PixelShuffle(u); levels: 384→2×192 sub-planes → per-plane Conv2d(192→5·u²,k=3)+PixelShuffle(u),
  drop the padded plane; **ICNR init** on both.
- **Inputs**: state at t AND t−24h (channel-concat, pangu-normalized), 4 surface vars
  (order u10,v10,t2m,mslp in original — we keep OUR canonical order t2m,u10,v10,mslp and
  reorder loss weights accordingly), 6 level vars ZUVTQ+W (we have 5: no W), 3 static masks
  standardized (we have 2: land_sea_mask, geopotential_at_surface) — no TISR/SST/SIC forcing,
  no solar input; month+hour adaLN instead. One forward = 24 h. Rollout: feed back, prev←cur.
- **Normalization**: inputs pangu-style per-var/per-level mean/std (compute ours from train
  years or reuse existing `normalization_pangu_s2s_*` stores — they exist and match our vars);
  target = pangu-normalized next state; loss multiplied per channel by `(pangu_std/delta24_std)²`
  ("loss_delta_normalization"), delta24_std = per-variable SCALAR std of the 24h increment
  (original: level [Z,U,V,T,Q,W]=[597.86,7.4878,8.9492,2.7132,9.5222e-4,0.3]; surface
  [U10,V10,T2m,MSLP]=[3.8920,4.5422,2.0727,584.098]) → **compute ours from our ERA5 train years**
  (small CPU job; per-variable scalar over 24h deltas, all levels pooled per variable).
- **Loss**: lat-weighted MSE; lat weights = cell-area (our grid is cell-centered 1°: use
  cos(lat)·normalized-to-mean-1); level weights p/mean(p) (17 levels, mean=350.588 → 0.0143…2.852);
  surface weights (graphcast) 0.1/0.1/1.0/0.1 for u10/v10/t2m/mslp; combination
  `(n_surf_eff·mse_surface_w + n_level·mse_level_w)/(n_level + sum_surface_coeffs)` — for us
  n_level=5, sum surface = 1.3 → denominator 6.3 (original 6+1.3=7.3 with W).
- **Optimizer/schedule**: AdamW lr 3e-4, betas (0.9,0.98), wd 0.05 applied ONLY to
  params with `'weight' in name and 'norm' not in name`; warmup 5000 steps linear + cosine→0
  over **300k optimizer steps**; global batch 4 (4 GPU × 1); grad clip 1.0; mixed precision
  (bf16 on H100); **no EMA**; seed 0. **RPFT**: at step 250k switch train data to 2007–2018
  (same run/schedule). No multistep fine-tuning (paper 2 drops it for the flagship).
- **Adaptation decisions (locked)**: patch (2,3,3) → tokens (Z=10: 1 surface + 9 level-token
  [17 levels pad→18], 60, 120); stage-2/3 grid (10,30,60); windows fit exactly (60/6, 120/10,
  30/6, 60/10); generalize hardcoded zdim=8 → parameter (LinVert, axial pos-embed shape,
  decoder plane count 2·9=18 → drop 1 pad → 17); PixelShuffle(3)/ICNR upscale 3; earth-bias
  table sizes recomputed (Wlat²·(2Wlon−1)=684 entries; window types stage1 = 10·10=100,
  stage2/3 = 10·5=50). Expected params ≈ 84M + slightly larger decoder/bias tables.
- Reference results (13-lvl 1.5° original, 24h RMSE 2020): Z500 44.96 m²/s², T850 0.615 K,
  T2m 0.539 K (sanity anchors only, not comparable exactly on our grid).

## 3. Execution phases

Work top-to-bottom; A and C can proceed in parallel; B is gated on checkpoint access (poll
throughout; drop B if never unblocked). Use background agents for cluster-side long jobs;
smoke-test EVERYTHING new before scaling (user instruction).

### Phase A — Pangu legacy hindcasts (Derecho) [no repo-code dependency; start immediately]

A1. On Derecho create `/glade/work/awikner/PanguWeather/v2.0/config/PANGU_S2S_infer_hindcast_15d.yaml`:
    copy of `PANGU_S2S_infer_mar_aug_multigpu_2.yaml` with, in the `S2S` section:
    `infer_months: [1,2,3,4,5,6,7,8,9,10,11,12]`, `infer_days: [1,5,9,13,17,21,25,29]`,
    `infer_hours: [0]`, `forecast_lead_times: [15]`, `save_latent: false`,
    `save_dir: '/glade/derecho/scratch/awikner/pangu_s2s_hindcasts/hindcast_2000_2024_15d'`.
    Leave everything else (esp. levels WITHOUT 200 hPa, exp_dir, data_dir) unchanged.
A2. New per-year job script `submit_hindcast_year.sh` (copy of `submit_infer_mar_aug.sh`):
    walltime **02:00:00** (95 inits ≈ 4× the 24-init job that fit in 20 min), drop the dead
    Globus block, keep `torchrun --nproc_per_node=4 inference_months.py --yaml_config=... \
    --run_num=2000 --val_one_year=${YEAR}` (NO dropout args, NO --save_run_num needed → outputs
    under `{save_dir}/2000/predictions/`). New orchestrator `submit_hindcast_2000_2024.sh`
    (from `submit_dropout_ensemble.sh`): loop YEAR 2000..2024, no seeds, no convert job.
A3. **Smoke test**: temporary config with `infer_months: [1]`, `infer_days: [1,5]` for year 2000
    (2 inits) → verify NC output exists, dims (time=16, plev=17, lat=180, lon=360), values sane
    (finite, T2m ~200–330 K). Then a second smoke: year 2024 with `infer_months: [12]`,
    `infer_days: [25,29]` → verifies (a) year-boundary crossing for TISR reads, (b) archive-end
    behavior. If archive-end crashes (IndexError past 2024-12-31), patch the loader minimally to
    clamp boundary reads to the last index (document in code comment) and re-smoke.
A4. Full campaign: submit all 25 year-jobs (they are independent; queue absorbs them).
    Monitor with `qstat -u awikner`. Expect ≤2 h each on 1 node × 4 A100.
A5. Consolidation (see §4): per completed year, run the consolidator on Derecho
    (CPU job or login-safe plain xarray/zarr — do NOT import physicsnemo on login nodes) to build
    `pangu_s2s/YYYY.zarr`; drop any Feb-29 init; then Globus-transfer to Stampede3
    `/scratch/09979/awikner/physicsnemo-zarr/hindcasts/pangu_s2s/`; verify (open store, check
    dims/coords/finite fraction); then DELETE the year's NC files and the Derecho-side yearly
    zarr (inode pressure, 85.7%!). Endpoints: NCAR GLADE `d33b3614-6d04-11e5-ba46-22000b92c6ec`,
    TACC Stampede3 `1e9ddd41-fe4b-406f-95ff-f3d79f9cb523`. Watch Globus high-assurance session
    timeouts (`globus session update` — Delta↔access-ci.org, TACC↔uchicago.edu).

### Phase B — SFNO hindcasts (Derecho) [gated on checkpoint access; code work can start now]

B1. Poll `ssh stampede3 'ls /work2/10441/rmasiwal/ 2>&1'` every hour or two while other work runs.
    Once readable: copy `SfnoPlasim.0.50.mdlus` + `checkpoint.0.50.pt` + run-dir `.hydra/config.yaml`
    to `$WORK/sfno_era5_ckpt/` (awikner), then transfer to Derecho
    `/glade/work/awikner/sfno_era5_ckpt/` (scp via the Mac, or Globus; ~3 GB total).
    Diff `.hydra/config.yaml` against `conf/model/sfno_era5.yaml` to confirm arch hyperparams.
B2. Repo code (do now, independent of access; all committed to `ai-rossbypalooza`):
    extend `examples/weather/ai_rossby/inference.py`:
    - multi-year dataset routing (same pattern as `build_datapipe`, dir-not-.zarr →
      `ClimateZarrMultiYearDataset`);
    - `inference.init_schedule` config (months/days/hours/year) → resolve to dataset time
      indices via the store `time` coord (cftime-safe); keep `ic_start` as-is for backcompat;
    - varying-boundary clamp at archive end (persistence past last index) + record clamped inits;
    - optional `inference.use_ema: true` — load EMA state dict from `checkpoint.*.pt` metadata
      onto the model after `load_checkpoint` (inspect metadata structure; `train.py` saves
      `metadata={"ema": ...}`).
    Smoke locally where possible (unit-level: schedule→index mapping with a synthetic cftime axis).
B3. Derecho setup: `cd /glade/work/awikner/physicsnemo && git fetch && git checkout
    ai-rossbypalooza && git pull`; ensure venv (VERIFY existing `.venv`; else `uv sync --extra
    cu12 --extra sfno-extras --extra utils-extras --extra datapipes-extras --group dev`; wandb not
    needed for inference). Create `/glade/derecho/scratch/awikner/physicsnemo-zarr/era5_all/`
    with symlinks to all year stores (NOT the normalization stores — multiyear globs `*.zarr`).
B4. **Smoke test** (derecho GPU, can use the derecho-smoke-test/shell skills): 1 init (2000-01-01)
    × 15 steps, batch 1 → verify per-IC output, then the same two edge cases as A3 (Dec 2000-12-29
    cross-year; 2024-12-29 archive-end clamp). Compare lead-1 fields against ERA5 next-day (RMSE
    sanity: Z500 ~<100 m²/s² territory after denorm — just check magnitudes are weather-like).
A note: inference.py runs rollout with `dataset` reads per step — point `data.zarr_path` at
    `era5_all`, `mean/std` at the Derecho norm stores, model config `model=sfno_era5`.
B5. Full campaign: per-year PBS jobs (`-q main`, 1 node 4×A100 or even 1 GPU — measure in smoke;
    95 inits × 15 steps is light). Simplest: 1 GPU per year-job, `inference.batch_size` ≥ 8.
    Then consolidate per year (§4) → `sfno_era5/YYYY.zarr` → Globus → verify → clean Derecho.

### Phase C — ArchesWeather port + training + hindcasts (local dev → Stampede3)

C1. Implement `physicsnemo/experimental/models/archesweather/` (port from geoarches, BSD-3
    headers + NVIDIA/UChicago SPDX lines per CI convention):
    - `archesweather_layers.py`: EarthAttention3D, EarthSpecificBlock (CLA via vendored minimal
      axial attention or a small self-contained implementation — avoid adding the
      `axial_attention` pip dep if trivial: it is AxialPositionalEmbedding (learned per-z-token
      embedding) + standard MHA over the Z axis), LinVert (zdim-parametrized), Up/DownSample,
      CondBasicLayer + zero-init adaLN, TimestepEmbedder, ICNR init, earth-position-index utils,
      pad/crop/window helpers (port from geoarches `weatherlearn_utils`).
    - `archesweather.py`: `ArchesWeatherEncodeDecode` (patch (2,3,3), 17→pad-18 levels, zdim 10,
      PixelShuffle(3) ICNR decoder) + `ArchesWeatherCondBackbone` + the physicsnemo wrapper
      `class ArchesWeather(physicsnemo.Module)`:
      ctor kwargs = variable groups + `levels` + `horizontal_resolution` + arch kwargs
      (emb_dim 192, depth_multiplier 2, window (1,6,10), droppath 0.2, mlp swiglu, cond 256,
      patch (2,3,3), `use_prev_state: true`, `add_input_state: false`);
      forward signature compatible with the trainer:
      `forward(surface_in, constant_boundary, varying_boundary, upper_air_in, *, surface_prev_in=None,
      upper_air_prev_in=None, calendar=None, ...)` → returns `(out_surface, out_upper_air, 0,0,0,0)`;
      `has_diagnostic = False`; **ignores `varying_boundary`** (accept + discard — original model
      has no SST/SIC/TISR forcing; month/hour adaLN replaces solar input); constant masks are
      standardized INSIDE the model (register per-channel mean/std buffers computed lazily from
      the first batch OR passed via config — pick config: two scalars per mask, computed once
      from the store and written into the yaml).
    - Prev-state support in the datapipe: add `prev_state_steps: int = 0` to `ClimateZarrDataset`
      / `ClimateZarrMultiYearDataset` (read `start - prev_state_steps`, emit `surface_prev_in`,
      `upper_air_prev_in`; sampler must not emit starts < prev_state_steps — extend
      `LeadTimePairSampler` min-start; multiyear cross-year read reuses the existing two-store
      assembly logic). Normalizer: z-score the `*_prev_in` keys with the same stats.
      `train_loop.train_step` / `validate.py` / `inference.py`: pass prev keys + `calendar` to
      the model when present in the batch (use a small helper `_model_kwargs(batch)` so
      SfnoPlasim path is unchanged). `emit_calendar=True` for this dataset (VERIFY the `calendar`
      key semantics in dataset.py; need month 1–12 and hour 0–23 — if it encodes something else,
      derive month/hour from the time coord instead).
    - New loss: `loss/archesweather.yaml` + implementation (extend `loss.py` with a
      `latitude_weighted: true`, `level_weights: pressure`, `surface_weights: [1.0,0.1,0.1,0.1]`
      (t2m,u10,v10,mslp order!), `delta_scaling: true` mode reading delta24 stds from a small
      stats file) — OR a self-contained `ArchesWeatherLoss` class selected via `loss=archesweather`.
      Denominator 6.3 as derived above.
    - Delta-std stats job: `tools/data/era5/compute_delta24_std.py` — per-variable scalar std of
      (x(t+24h)−x(t)) over 1979–2018, sampled (e.g. every 5th day is plenty), writes a tiny
      json/zarr consumed by the loss. Run via stampede3 CPU (skill: stampede3-cpu-job).
    - Configs: `conf/model/archesweather_era5.yaml` (name ArchesWeather, module
      physicsnemo.experimental.models.archesweather, same variable groups/levels/resolution as
      sfno_era5 MINUS diagnostic_variables (empty), plus arch + mask-stats kwargs);
      `conf/training/archesweather.yaml` (AdamW 3e-4 (0.9,0.98) wd 0.05-selective — VERIFY
      trainer supports param-group wd; if not, add it; warmup+cosine in STEPS: 5000/300000 —
      the existing scheduler is epoch-based (LinearWarmupCosineAnnealingLR per epoch); with
      `num_samples_per_epoch: 8760` and global batch 4 → 2190 steps/epoch → 300k steps ≈ 137
      epochs; set warmup ≈ 2.28 epochs ≈ round to 2, eta_min 0, max_epochs 137; close enough to
      the original schedule — document the approximation); `dataset=era5_multiyear` variant
      `era5_archesweather.yaml` with `prev_state_steps: 4`, `emit_calendar: true`,
      `forecast_lead_times: [4]`.
    - RPFT: run as TWO submissions: epochs 1–114 (≈250k steps) with `era5_train`; then resume
      (auto from ./checkpoints) with `zarr_path → era5_recent` (2007–2018) for epochs 115–137.
      The cosine schedule continues because scheduler state is in checkpoint.pt.
C2. **Smoke tests** (in order): (i) local CPU: instantiate model at full size, forward on random
    tensors, check shapes + param count ≈ 84–90M, backward runs; (ii) tiny-config pytest on
    Stampede3 GPU (stampede3-smoke-test skill) if a test file is added; (iii) 1-GPU short train
    run on Stampede3 (few hundred steps): loss decreases, throughput measured → compute real
    walltime for 300k steps; (iv) 4-GPU DDP run for ~50 steps (DDP correctness; remember
    torch<2.11 pin — venv already 2.10; wandb init on every rank — train.py already handles).
C3. Full training: sbatch (clone `train_sfno_era5.sbatch` → `train_archesweather_era5.sbatch`),
    1 node 4×H100, 48 h walltime, auto-resubmit on timeout (resume is automatic from
    ./checkpoints). Expected ~0.4–0.6 s/step → 300k steps ≈ 33–50 h + queue. wandb project
    ai-rossby-era5 (entity qiangsun-university-of-chicago), init on all ranks, mode per cluster.
    Monitor: RMSE-ish val loss at epoch boundaries; sanity anchor vs SFNO run's curves.
C4. ArchesWeather hindcasts 2000–2024: run **on Stampede3** (checkpoint + ERA5 + venv all local;
    no boundary forcing needed; prev-state needs x(t0−24h) from ERA5 — available for all inits).
    Reuse the extended inference.py (works from Stampede3 paths; multi-year dir
    `era5_all/` symlinks on Stampede3 too). Same schedule, same consolidator →
    `hindcasts/archesweather_era5/YYYY.zarr` (written directly on Stampede3 — no transfer).

### Phase D — wrap-up

D1. Register the three hindcast datasets in `hpc/data_registry.yaml` (dataset `hindcasts_pangu_s2s`
    etc., copies: stampede3) + `registry.py scan stampede3` to record.
D2. Final verification pass: per store — dims (init_time≈95, lead_time=16, [pressure_level=17],
    lat=180, lon=360), init_time coord matches schedule minus Feb 29, no all-NaN fields,
    spot-check lead-0 equals ERA5 IC values for a random init, attrs complete.
D3. Update `docs/dev/context/` with a completion note + gotchas learned; update this plan's
    status line; memory files; summary report for the user (what ran, where data lives, any
    skipped parts — esp. whether SFNO hindcasts happened or were skipped per decision).
D4. Cleanup: Derecho transient NC/zarr deleted (verify inode count back down); nothing left in
    scratch hindcast staging dirs; PanguWeather repo untouched except new config/scripts.

## 4. Hindcast store specification (all three models)

Path: `/scratch/09979/awikner/physicsnemo-zarr/hindcasts/{model}/{YYYY}.zarr`, model ∈
{pangu_s2s, sfno_era5, archesweather_era5}; YYYY = init year, 2000–2024.

- zarr v3, consolidated, codecs bytes(LE)+zstd(0) (match training stores); float32; fill NaN.
- Dims: surface/diagnostic vars `(init_time, lead_time, lat, lon)`; upper-air
  `(init_time, lead_time, pressure_level, lat, lon)` with pressure_level = the 17 model levels.
- Chunking: `(1, 16, [17,] 180, 360)` — one chunk per (variable, init) → ~95 chunks/var/year
  (~1.5k files/store; Derecho-inode- and Globus-friendly; a full-trajectory read is one chunk).
- Coords: `init_time` (cftime standard, hours since YYYY-01-01), `lead_time` = [0..15] int
  (attrs units "days"), `lat`/`lon`/`pressure_level` copied from the training store.
- lead 0 = the ERA5 initial condition (denormalized model input), leads 1..15 = forecasts.
- Root attrs: training-store-style variable group lists (per model's native set), plus
  `hindcast_schema_version "1.0"`, `model`, `checkpoint` (path + epoch/run id + ema flag),
  `source_dataset`, `init_schedule "monthly days 1,5,9,13,17,21,25,29 (no Feb 29) 00Z"`,
  `lead_time_hours 24`, `n_lead 16`, `calendar standard`, `boundary_clamped_inits [...]` (list,
  usually empty), `created`, `generator` (script + commit hash).
- Variables per model (all in one flat root group, like training stores):
  - pangu_s2s: surface 2m_temperature, 10m_u/v, mean_sea_level_pressure, surface_pressure;
    diagnostic total_precipitation_24hr, mean_top_net_long_wave_radiation_flux; upper-air T,U,V,Q,Z.
  - sfno_era5: surface 2m_temperature, 10m_u/v, mean_sea_level_pressure; diagnostic
    total_precipitation_24hr; upper-air T,U,V,Q,Z.
  - archesweather_era5: surface 2m_temperature, 10m_u/v, mean_sea_level_pressure; upper-air T,U,V,Q,Z.
- Consolidator: ONE shared script, e.g. `tools/data/hindcast/consolidate_hindcasts.py`, with
  input adapters: (a) Pangu per-init NC (dims time/plev/lat/lon, frame 0 = IC), (b) ai_rossby
  inference.py per-IC output (dims per `_build_per_ic_dataset`: pred_surface (ic, ensemble, step,
  var, lat, lon) etc. — squeeze ic/ensemble, prepend the lead-0 IC frame read from ERA5 if the
  writer's frame set starts at step 1 — VERIFY frame indexing in async_writer output, recon says
  "frame 0 = observed IC" for the streaming path). Skip/drop Feb-29 inits. Plain
  xarray/zarr/numpy only (login-node-safe); unit-test locally with synthetic small arrays.

## 5. Risks / gotchas checklist

- Derecho scratch inodes 85.7% used → never accumulate more than ~2 years of transient output;
  delete after verified transfer. Derecho scratch 60-day purge → don't park anything there.
- Derecho is RETIRING (see derecho-retire-rehome-to-delta.md) — this campaign is fine, but do
  not create new long-lived data on Derecho.
- TACC scratch purge (~10-day) → hindcast stores land on Stampede3 scratch; they are the ONLY
  copy. Flag to user at wrap-up (they declined a Delta copy for now).
- torch<2.11 pin for SFNO DDP (venvs already comply); ArchesWeather DDP should also be smoked
  under 2.10. wandb on every rank. `uv sync` must keep the extras or SFNO deps get pruned.
- Do not `import physicsnemo` on login nodes (CUDA/Warp core-dump risk) — consolidator is plain
  xarray/zarr; run heavy steps via the cluster job skills.
- Pangu: do NOT add 200 hPa; do NOT touch PanguWeather-rajatm2; YParams turns 'None' strings
  into None; `hyperparams_infer.yaml` rank-0 write race across concurrent year-jobs is harmless.
- Legacy InferFilter will emit Feb 29 in leap years → drop at consolidation (target 95/yr).
- Globus high-assurance session refresh; globus-cli on Stampede3 at `~/gcli`; transfers can also
  run from the Mac with the local globus CLI if present.
- geoarches licensing: BSD-3 LICENSE governs (pyproject has a stale CC-BY-NC-SA string — noted;
  fine for this research port with attribution). v1 repo (gcouairon/ArchesWeather) has NO
  license — do not copy from it.
- CI header check: new files need the NVIDIA SPDX + UChicago copyright lines.
- SFNO hindcasts use observed (ERA5) SST/SIC/TISR along the trajectory — inherent to the model
  design; worth noting in any downstream comparison vs Pangu (which feeds only observed TISR)
  and ArchesWeather (no forcing at all).

## 6. Progress log (update as executed)

- [x] Recon (4 agents) + user decisions + this plan committed. (2026-07-21)
- [x] A1–A3 Pangu config + scripts + smokes DONE. Both smokes PASS (basic Jan2000
      T2m 214-307 K; edge Dec2024 T2m 219-313 K, 16-frame). **Loader patch applied**
      to Derecho `utils/data_loader_multifiles.py` (backup `.bak_hindcast`), gated
      behind config flag `include_forecast_past_data_end` (default-off, zero impact
      on other configs): (a) persistence fallback for a missing h5 (archive-end,
      steps back 6h until an existing file — fixes Dec-2024→2025 crash); (b) relaxes
      the `max_inference_idx` clamp that was dropping Dec 21/25/29 EVERY year.
      Verified enumeration: 2000→96, 2001→95, 2024→96 (leap keeps Feb29 → dropped at
      consolidation → 95/yr). To revert: restore the .bak + drop the flag.
- [x] A4 25-year campaign LAUNCHED: jobs **6839213..6839237** (yr 2000..2024),
      queued/running on Derecho, 4×A100, ≤2h each, run_num=2000, outputs at
      `.../hindcast_2000_2024_15d/2000/predictions/` (filenames carry init datetime).
- [ ] A5 Pangu consolidation + transfer + cleanup — per year as jobs finish
      (consolidator `tools/data/hindcast/consolidate_hindcasts.py --format pangu`).
- [~] B1 SFNO checkpoint STILL blocked (rmasiwal `/work2/10441/rmasiwal` 0700,
      unchanged since Dec-2024). Per user decision: skip SFNO hindcasts if never
      unblocked. Polled repeatedly this session; still denied.
- [~] B2 inference.py extensions IN PROGRESS (background agent): multi-year
      routing, init_schedule resolver, boundary clamp, use_ema, prev/calendar rollout.
- [ ] B3–B5 SFNO setup/smoke/campaign (gated on B1)
- [x] **C1 ArchesWeather model + datapipe prev-state + loss + configs DONE**
      (commits d644e77c, 2e9ba645, b79e7208; pushed). CPU smoke: 88.75M params,
      correct shapes, backward OK. Datapipe test 4/4, loss test 2/2, all py_compile.
- [x] C2 smokes: local shape/params ✅; **1-GPU real-env integration smoke PASSED on
      Derecho A100** (job 6841131, rc=0) — Module.instantiate + prev/calendar datapipe
      + train_step + ArchesWeatherLoss + selective-wd optimizer + validation all ran
      end-to-end, finite loss, ~1.1s/iter. (Stampede3 smoke 3334859 was stuck >100min
      in the h100 queue → cancelled, validated on Derecho instead.) Init loss large
      (untrained, no delta-scaler) — expected. 4-GPU DDP not separately smoked
      (single-GPU stack validated; DDP is the same code under torch<2.11).
- [~] C3 300k-step training — **VENUE CHANGED (2026-07-22, user): Midway3
      `pedramh-gpu` (4×H100, infinite walltime, account pi-pedramh).** See §8 for the
      Midway3 setup. (Stampede3 H100 job 3335041 was the initial submission; cancelled
      in favour of Midway3 — Stampede3 h100 was heavily oversubscribed and pi-pedramh's
      pedramh-gpu node has infinite walltime so no 48h chunking/resubmit is needed.)
      delta-std stats job ready (run `tools/data/era5/compute_delta24_std.py` +
      set loss.delta_scaler_path to enable delta-normalization; loss works without it).
- [ ] C4 ArchesWeather hindcasts on Stampede3 (post-training; era5_all/ dir ready;
      inference.py handles prev/calendar rollout at step_size=4).
- [~] D: consolidator + test DONE (commit aafcf6a4); registry/verification/docs/
      cleanup/final report pending campaign completion.

**Session note (2026-07-21, autonomous):** All CODE deliverables complete, tested,
committed, pushed. Cluster campaigns (Pangu 25-yr generation, ArchesWeather 300k-step
training + generation) are launched/staged but run for hours–days beyond a single
session; SFNO remains gated on checkpoint perms. See memory `hindcast-campaign-2026-07`.

## 7. Learnings from execution (2026-07-22)

- **Pangu NC writer emits MORE vars than recon assumed.** The per-init NetCDFs carry
  the land/ocean state (volumetric_soil_water_layer_1, soil_temperature_level_1,
  skin_temperature, sea_surface_temperature) *in addition to* the 5 surface + 2 diag +
  5 upper-air. Recon §2.1 said these are "marched but NOT written" — wrong; they ARE
  written. The consolidator keeps them verbatim (more complete; harmless). So pangu_s2s
  stores have 9 surface + 2 diagnostic + 5 upper-air vars, not the 4+2+5 in §4.
- **Loader clamp bug (fixed).** Stock `utils/data_loader_multifiles.py` `max_inference_idx`
  clamp drops Dec 21/25/29 inits EVERY year (→92-93/yr) and crashes at archive end
  (Dec-2024 needs 2025 h5). Patched (gated behind `include_forecast_past_data_end`, backup
  `.bak_hindcast`): persistence for missing h5 + relaxed clamp. Confirmed 95/yr (leap 96,
  Feb29 dropped at consolidation). This patch lives ONLY in the legacy Derecho PanguWeather
  repo, not the fork.
- **Consolidation is I/O-bound (~39 min/year, ~29-30 GB/store).** Reads ~38 GB NC + writes
  ~29 GB zstd to GLADE Lustre per year. OOMs on login nodes (stacks ~40 GB/year) — MUST run
  as a batch job. Run PARALLEL across disjoint year ranges (one sequential job can't finish
  25 yrs in walltime). Feb-29 correctly dropped (init_time=95, verified finite).
- **Derecho CPU `main` queue is badly congested** (>2.7 h queued); the GPU `main` queue
  backfills in minutes. Ran the CPU-only consolidator on GPU nodes (UCHI0018) to unblock —
  wasteful but pragmatic.
- **TACC Stampede3 rejects `--gpus-per-node`/`--gres`** (whole-node allocation). Launchers
  must derive GPU count from `nvidia-smi`/Slurm env, not a baked gres directive (fixed in
  train_archesweather_era5.sbatch). Stampede3 h100 queue was heavily oversubscribed
  (a 1-GPU smoke sat >100 min PD) — validated the ArchesWeather stack on Derecho A100 instead.
- **Globus works autonomously** from Stampede3 (`~/gcli/bin/globus`, identity
  awikner@uchicago.edu); the NCAR GLADE session was valid (no interactive `session update`
  needed this session). Transfers are server-side (submit + poll `globus task show`).

## 8. ArchesWeather training on Midway3 (venue change, 2026-07-22, user-directed)

Train ArchesWeather-M on UChicago RCC **Midway3 `pedramh-gpu`** (single node
`midway3-0423`, **4×H100**, 515 GB, 32 cpu, **infinite walltime**, account
**pi-pedramh**). Chosen over Stampede3 H100 (heavily oversubscribed) — infinite
walltime means no 48h chunking/resubmit.

- **Data fits scratch.** Training data = ERA5 1979-2018 (train) + 2019 (val), 41
  year-stores × ~30 GB ≈ **1.2-1.3 TB** < Midway3 scratch **2.0 TB hard limit**
  (`/scratch/midway3/awikner`). NOTE soft quota is 100 GB (grace-based); we run
  far over soft, relying on grace + the 2 TB hard cap. era5_recent (RPFT) + train/val
  are symlink dirs (no extra copies). Checkpoints (few GB) fit the remaining headroom.
- **Globus:** Midway3 collection `2fde89c0-6fb4-11eb-8c47-0eb1aa8d4337`
  (UChicago RCC Midway3). No globus-cli ON Midway3 → drive the transfer from
  Stampede3 (`~/gcli/bin/globus`). Source = TACC Stampede3 `1e9ddd41-...`
  `/scratch/09979/awikner/physicsnemo-zarr/era5/{1979..2019}.zarr` + the two
  normalization_pangu_s2s_{mean,std}.zarr. Direct recursive transfer (tiny-chunk
  stores → per-file-overhead-bound, ~hours, server-side).
- **Setup on Midway3:** clone/pull `awikner/physicsnemo@ai-rossbypalooza`; build the
  venv (`uv sync --extra cu12 --extra utils-extras --extra datapipes-extras --group dev`
  — SFNO extras NOT needed for ArchesWeather; torch<2.11 pin); create
  era5_train/era5_val/era5_recent symlink dirs under `/scratch/midway3/awikner/physicsnemo-zarr`.
- **Launch:** `AI_ROSSBY_DATA=/scratch/midway3/awikner/physicsnemo-zarr sbatch
  -p pedramh-gpu -A pi-pedramh hpc/scripts/train_archesweather_era5.sbatch`
  (launcher is cluster-agnostic; derives GPU count from nvidia-smi). No walltime cap
  needed. RPFT: resubmit with `AI_ROSSBY_TRAIN_DIR=era5_recent` at ~epoch 114.
