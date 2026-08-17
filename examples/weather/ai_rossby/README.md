# ai-rossby training recipe

Train and evaluate deep-learning weather/climate emulators (PanguWeather-style
transformers, SFNO, and AMIP diffusion) on the PhysicsNeMo framework. This is
the recipe a new group member uses to go from a cluster environment to a
trained model.

- **New here?** Read the top-level [`README.md`](../../../README.md) first, then
  this file.
- **Have PanguWeather models/data already?** See
  [`PANGUWEATHER_MIGRATION.md`](PANGUWEATHER_MIGRATION.md) for porting configs,
  data, and checkpoints.
- **Where does the data come from?** See [`DATA.md`](DATA.md).
- **Running the AMIP diffusion models (v1 or v2)?** Sections 1–5 below are the
  deterministic Pangu/SFNO path; the diffusion families have their own end-to-end
  guide in [§6](#6-amip-diffusion-v1-and-v2--full-run-guide).

## 1. Prerequisites

1. A PhysicsNeMo environment on your cluster — follow
   [`hpc/install.md`](../../../hpc/install.md) or the per-cluster doc
   (`hpc/delta.md`, `hpc/deltaai.md`, …).
2. Data. On Delta the converted Zarr stores already exist and the configs fall
   back to them automatically. Elsewhere, set `AI_ROSSBY_DATA` to your
   converted-Zarr root (see [`DATA.md`](DATA.md)):
   ```bash
   export AI_ROSSBY_DATA=/my/physicsnemo-zarr
   ```

## 2. How the configs are organized (Hydra)

Training is composed from five [Hydra](https://hydra.cc) config groups under
[`conf/`](conf/); each run picks one YAML per group:

| Group | Dir | Picks |
|---|---|---|
| `model` | `conf/model/` | architecture + variable groups (e.g. `sfno_e3sm`, `pangu_plasim_legacy`) |
| `dataset` | `conf/dataset/` | Zarr paths, normalization, loader knobs (e.g. `e3sm`, `plasim_sim52_year12`) |
| `training` | `conf/training/` | optimizer, EMA, multi-stage curriculum (e.g. `sfno_plasim`) |
| `loss` | `conf/loss/` | loss family (`mae`, `raw_l2`, …) |
| `validation` | `conf/validation/` | rollout validator (`off`, `rollout_5412`, …) |

Compose them on the command line and override any leaf with a dotted path:

```bash
python train.py \
    model=sfno_e3sm dataset=e3sm training=sfno_plasim loss=raw_l2 validation=off \
    training.max_epochs=100 dataset.batch_size=8 run_name=sfno_e3sm_run0
```

`conf/config.yaml` is the root (defaults + run control + wandb). Hydra changes
into `outputs/<run_name>/`, so checkpoints land in
`outputs/<run_name>/checkpoints/`.

## 3. Train

**Single GPU:**

```bash
cd examples/weather/ai_rossby
python train.py model=sfno_e3sm dataset=e3sm training=sfno_plasim run_name=sfno_e3sm_run0
```

**Multi-GPU (single node):**

```bash
torchrun --standalone --nproc-per-node=4 train.py \
    model=sfno_e3sm dataset=e3sm training=sfno_plasim run_name=sfno_e3sm_ddp
```

> **SFNO under DDP requires `torch<2.11`** and wandb initializes on every rank —
> both are handled by the pinned environment; see
> [`PANGUWEATHER_MIGRATION.md`](PANGUWEATHER_MIGRATION.md) §4.2.

Optional: stage the data to fast node-local disk first with
`dataset.stage_to_local=True` (a win only for data-bound runs; see
`PANGUWEATHER_MIGRATION.md` §4.5).

Useful overrides: `training.amp=bf16`, `training.max_epochs=N`,
`dataset.batch_size=N`, `wandb.mode=online`, `wandb.enabled=False`,
`training=sfno_plasim_curriculum` (unroll ramp). Resume by relaunching with the
same `run_name`.

## 4. Monitor

Metrics route to **[wandb](https://wandb.ai)** (default, offline — a local
`./wandb/` run, no login needed) and the console: `train/*` (loss components)
and `valid/*` (`val_loss`, and `rmse_step*`/`acc_step*` when a
`validation=rollout_*` group is composed). `wandb sync ./outputs/<run>/wandb/<run>`
uploads an offline run.

## 5. Evaluate

```bash
# Autoregressive rollout — one file per initial condition, written to a DIRECTORY
python inference.py model=sfno_e3sm dataset=e3sm \
    +inference.checkpoint_dir=/path/to/checkpoints \
    +inference.output_dir=/path/to/preds +inference.max_step=60 \
    '+inference.ic_start=[0, 60, 120]'

# Score predictions against the reference Zarr (RMSE/ACC scorecard)
python validate_cli.py dataset=e3sm \
    +validation_cli.predictions=/path/to/preds/<file> \
    +validation_cli.reference_zarr=$AI_ROSSBY_DATA/e3sm/2045.zarr \
    +validation_cli.output_md=/path/to/scores.md
```

`inference.py` requires `checkpoint_dir`, `output_dir`, `max_step`, and one of
`ic_start` / `init_schedule`. `output_dir` is a **directory** — the writer emits
one self-describing file per IC (`+inference.output_format=zarr|netcdf`, default
`zarr`).

## 6. AMIP diffusion (v1 and v2) — full run guide

Everything in this section is `physicsnemo.experimental`, derived from the
upstream `amip` codebase (see [`NOTICE`](../../../NOTICE)) and **not** part of the
supported PhysicsNeMo path. Use the deterministic Pangu/SFNO recipes above unless
you specifically need diffusion.

The port spans **two upstream generations**, both live in this tree:

| | upstream baseline | families | status |
|---|---|---|---|
| **v1** | `amip` @ `497827e` | SI, SI_X, EDM, RFM, ERDM-UNet, x_DDC-UNet | **frozen** — kept runnable so translated v1 checkpoints still load; receives no rebaseline changes, because amip_v2 deleted these families |
| **v2** | `amip_v2` @ `e0b7b60` | ERDM/RollingDiT forecaster, x_DDC/`DiTAE` downscaler | the active path; both real v2 checkpoints reproduce upstream's own forward at `max │diff│ = 0` |

Contents: [6.1 entry points](#61-entry-points) ·
[6.2 channel contracts](#62-two-channel-contracts-on-purpose) ·
[6.3 family matrix](#63-the-family-matrix) · [6.4 data](#64-data-you-need) ·
[6.5 train v1](#65-train--v1-families) · [6.6 train v2](#66-train--v2-erdm) ·
[6.7 run knobs](#67-knobs-both-generations-share) ·
[6.8 mid-training validation](#68-mid-training-validation) ·
[6.9 rollout](#69-rollout-inference) · [6.10 eval suite](#610-long-horizon-eval-suite) ·
[6.11 checkpoints](#611-bringing-in-an-upstream-checkpoint) ·
[6.12 two-stage cascade](#612-combinedmodule-forecaster--downscaler) ·
[6.13 health gates](#613-health-gates-run-these-before-trusting-a-config) ·
[6.14 cluster scripts](#614-cluster-job-scripts) ·
[6.15 gotchas](#615-known-limitations--gotchas)

### 6.1 Entry points

| Command | What it does | Applies to |
|---|---|---|
| `train_diffusion.py` | trains a forecaster — single-step **or** rolling-window | SI, SI_X, ERDM-UNet, RFM, ERDM v2 (± ocean, ± SST suite) |
| `inference.py` | autoregressive / rolling rollout → one file per IC | same, **except SI_X** (§6.15) |
| `eval_diffusion.py` | long-horizon eval suite: climatology, bias, QBO, global-mean flux, spread/skill | same as `inference.py` |
| `validate_diffusion.py` | **library only** (no CLI) — the validator behind `validation=diffusion_rollout` | — |
| `validate_cli.py` | scores an already-written prediction file against a reference Zarr | generation-agnostic |
| `tools/checkpoint_translation/amip_si.py` | upstream Lightning `.ckpt` → `.mdlus` | every family |
| `tools/checkpoint_translation/verify_v2_numerical.py` | forward-equivalence against upstream's own module | ERDM v2, x_DDC |
| `tools/data/amip/*` | build the Zarr stores + the SST artifact | — |

Two absences worth knowing up front:

- **The downscaler has no CLI.** x_DDC is trained upstream and translated in
  (§6.11); at inference it runs inside `CombinedModule`, composed in Python
  (§6.12). `train_diffusion.py` has no x_DDC branch — its scheduler's
  `compute_loss(wrapper, x_lowres, x_highres)` signature matches neither the
  single-step nor the window path.
- **`climatology_cli.py` is deterministic-only.** The diffusion equivalent is
  `eval_diffusion.py` (§6.10).

### 6.2 Two channel contracts, on purpose

The two generations differ in **channel order**, so loading a checkpoint against
the wrong contract runs happily and produces nonsense — every contract bug found
during the rebaseline was shape-preserving. Every wrapper takes
`channel_layout`, and the translator sets it from `--source-contract`.

| `channel_layout` | Meaning | Where it comes from |
|---|---|---|
| `v2` | upstream **amip_v2**: state `[surface \| diag \| upper_air]`, upper-air level-major (1000 hPa first), c_grid `[varying \| constant]` | `amip_erdm_v2`, `amip_erdm_v2_ocean`, `amip_erdm_fancy`, `amip_x_ddc_dit`; the default for `RollingDiTWrapper` / `XDDCWrapper` |
| `v1` | upstream **amip v1**: same group order, upper-air **variable-major** in config level order | `amip_x_ddc` (its `XDDCUNet` backbone is v1-only). Also what `--source-contract v1` writes into a translated artifact — so for the *other* v1 families, whose YAMLs say `fork`, add `++model.channel_layout=v1` when you run one |
| `fork` | the Phase-8 fork order (`[surface \| upper_air \| diag]`, c_grid `[constant \| varying]`) — **no upstream checkpoint matches it**, including v1's | `amip_rfm` explicitly and `amip_erdm` by wrapper default. Fine for training from scratch in this fork; wrong for loading real upstream weights |

**Why this needs saying at all.** `train_diffusion.py`, `inference.py` and
`eval_diffusion.py` all build the model from `cfg.model` and *then* load weights
into it. So the `channel_layout` that governs run-time packing is **the one in
the YAML**, not the one baked into the `.mdlus` at translation time — and the
translator sets the artifact's layout from `--source-contract` for all four
wrapper classes. `amip_x_ddc` aside, the v1-family configs say `fork`, so a
translated v1 checkpoint of those needs the layout supplied on the command line.
Use `++` — it both appends the key where the YAML omits it (`amip_erdm`) and
overrides it where the YAML sets it (`amip_rfm`), so one spelling always works:

```bash
python inference.py model=amip_erdm ++model.channel_layout=v1 …  # not fork
python inference.py model=amip_rfm  ++model.channel_layout=v1 …  # not fork
```

The SI configs need no override: `amip_si` / `amip_si_x` already say `v1`,
because they describe real v1 checkpoints rather than a from-scratch model.

**Since 2026-08-14 the drivers refuse the mismatch instead of running it.**
`Module.save` records the resolved constructor kwargs in the archive's
`args.json`, so `train_loop.assert_checkpoint_contract` compares the artifact's
`channel_layout` (and the variable lists, and `levels`) against the instantiated
`cfg.model` *before* the load, in `inference.py`, `eval_diffusion.py` and
`load_partial_weights`:

```
ValueError: channel-contract mismatch between .../AmipDiTWrapper.0.0.mdlus and cfg.model (AmipDiTWrapper):
  channel_layout:
      checkpoint: v1
      cfg.model:  fork
```

A passing run logs `channel contract verified against <file>: channel_layout='v1'`.
The guard exists because this failure class is **shape-preserving**: it permutes
the upper-air block, so `load_state_dict` is clean *and* the §6.13 shape digest is
byte-identical across a layout flip (measured — `amip_x_ddc` hashes to
`ba2d45a0801366be` either way). A warm start is the one caller that tolerates
differences, and only in the keys that are the point of it (`ocean_state_variables`
et al.) — `channel_layout` is fatal there too.

Loading through `Module.from_checkpoint` (§6.12) side-steps the question
entirely: it rebuilds the wrapper from those stored args, so it cannot disagree
with the weights.

The v1 families are **frozen**: kept working, not migrated. See
[`docs/dev/phase12_implementation_plan.md`](../../../docs/dev/phase12_implementation_plan.md)
for the seam.

### 6.3 The family matrix

Every width below is **derived** from the config's own variable lists (nothing
restates a channel count) and was read back off the instantiated wrapper:

| Config | Wrapper | Layout | Train `loss=` | Sample `sampler=` | Dataset | State grid | `in_ch` | `c_grid` | `scalar` | ocean |
|---|---|---|---|---|---|---|---|---|---|---|
| `amip_si` | `AmipDiTWrapper` | `v1`⁴ | `si` | `si` | `amip_dailyavg_coarse` | **45×90** | 151 | 5 | 2 | — |
| `amip_si_x` | `AmipDiTWrapper` | `v1`⁴ | `si_x` | `si_x` | `amip_dailyavg_coarse` | **45×90** | 151 | 5 | 3⁵ | — |
| `amip_erdm` | `ERDMWrapper` | `fork` | `erdm` | `erdm` | `amip_1981` | 180×360 | 161 | 7 | 2 | — |
| `amip_rfm` | `RollingDiTWrapper` | `fork` | `rfm` | `rfm` | `amip_1981` | 180×360 | 161 | 7 | 2 | — |
| `amip_x_ddc` | `XDDCWrapper` (`XDDCUNet`) | `v1`¹ | — (upstream)³ | —³ | `amip_dailyavg` | 180×360 | 151 | — | — | — |
| `amip_erdm_v2` | `RollingDiTWrapper` | `v2` | `erdm_v2` | `erdm`² | `amip_dailyavg_coarse` | 45×90 | 151 | 5 | 3 | — |
| `amip_erdm_v2_ocean` | `RollingDiTWrapper` | `v2` | `erdm_v2` | `erdm`² | `amip_dailyavg_coarse` | 45×90 | 153 | 5 | 3 | 2 |
| `amip_erdm_fancy` | `RollingDiTWrapper` | `v2` | `erdm_v2` | `erdm`² | `amip_dailyavg_coarse` + SST suite | 45×90 | 154 | 6 | 3 | 3 |
| `amip_x_ddc_dit` | `XDDCWrapper` (`DiTAE`) | `v2` | — (upstream)³ | —³ | `amip_dailyavg` | 180×360 | 151 | — | — | — |
| `amip_combined` | `CombinedModule` | — | — | — | — | 45×90 → 180×360 | — | — | — | — |

¹ `v1` since 2026-08-14 (it shipped as `v2`): `XDDCUNet` is the *convolutional*
denoiser, which exists only in amip v1 — v2 deleted the conv path — so every
checkpoint this config can load is a v1 artifact. The flip moves no parameter
shape, so nothing became unloadable.
² the default `sampler=from_loss` instantiates `cfg.loss` instead, which for a v2
run means `erdm_v2` — the right choice; `sampler=erdm` is the v1-statistics
variant.
³ no recipe drives the downscaler standalone: `loss/x_ddc.yaml` and
`sampler/x_ddc.yaml` exist to *build* the `DataDependentInterpolant` that
`CombinedModule` samples with (§6.12), and its arity does not fit
`train_diffusion.py` or `inference.py` (§6.15). Its listed dataset/grid is the
resolution it operates at, not a store it reads — its conditioning is always the
forecaster's own upsampled prediction.
⁴ these describe the **real v1 SI checkpoints**: 1.25B-param *coarse-state*
models on a 45×90 backbone grid with 180×360 forcings reduced by
`c_grid_downsample: 4`, the same shape as the v2 ERDM. Both verified bitwise
against upstream v1's own `DiT`, on random inputs *and* on a real batch through
our loader (`max │diff│ = 0.0000e+00`). They took over these two names on
2026-08-17; the previous `amip_si`/`amip_si_x` described a 161-channel model on a
180×360 grid that no checkpoint we hold matches, and were deleted. Pair with
`dataset=amip_dailyavg_coarse` — which is **6-hourly** ("dailyavg" names the
24-hour-accumulation variables, not the row cadence) and carries exactly this
variable set — never `amip_1981`, which serves a native-resolution state and a
different variable set.
⁵ `scalar_dim` 3 because `global_mean_co2` is **routed** out of the gridded
stream into the calendar row (`scalar_routed_boundary_variables`), keeping
`c_grid_dim` at 5 for a 4-entry varying list. The checkpoint's `c_grid_embed` is
`Conv2d(5 → 192)`, so listing CO₂ as a fourth gridded channel would not load.

`amip_combined` is documentation, not a buildable model config: `CombinedModule`
takes already-instantiated sub-modules, so it is assembled in Python (§6.12).

### 6.4 Data you need

The v1 families read the 6-hourly full-resolution `amip` store; v2 reads the
**pre-coarsened 45×90 state store paired with native-1° forcings** — upstream's
own layout (coarse state, full-res `c_grid` reduced by the model's stride-4 conv,
`c_grid_downsample: 4`).

| Store | Registry name | Used by |
|---|---|---|
| `amip/<year>.zarr` | `amip` | v1 families (`dataset=amip_1981`) |
| `amip_dailyavg/<year>.zarr` | `amip_dailyavg` | x_DDC (full-res, 180×360) |
| `amip_dailyavg_coarse/<year>.zarr` | `amip_dailyavg_coarse` | v2 forecaster state (45×90) |
| `amip_dailyavg_boundary/<year>.zarr` | `amip_dailyavg_boundary` | v2 forcings at 1°, NaN preserved |
| `norm_stats/sst_climatology.npz` | `amip_sst_climatology` | `amip_erdm_fancy` only |

Check what exists where before building anything:

```bash
python tools/data/registry.py show amip_dailyavg_coarse
python tools/data/registry.py check          # green = all declared stores present
```

Build chain (per year; the cluster wrappers in §6.14 loop over years):

```bash
# 1. HDF5 -> per-year Zarr
python tools/data/amip/amip_h5_to_zarr.py \
    --config tools/data/amip/configs/amip_dailyavg.yaml \
    --year 1981 --output $AI_ROSSBY_DATA/amip_dailyavg/1981.zarr

# 2. 45x90 coarse state store — upstream's exact blur operator
#    (F.interpolate bilinear, align_corners=False), so the forecaster trains on
#    the same manifold the downscaler undoes. --smooth-boundaries is not
#    cosmetic: a 4x4 block straddling a coastline otherwise averages ocean
#    values with the land fill.
python tools/data/amip/coarsen_zarr.py \
    --input  $AI_ROSSBY_DATA/amip_dailyavg/1981.zarr \
    --output $AI_ROSSBY_DATA/amip_dailyavg_coarse/1981.zarr \
    --factor 4 --smooth-boundaries

# 3. boundary-only store at native 1 degree — ~2.3 GB/yr instead of ~58 GB/yr
#    for a bit-identical training input. NaN is PRESERVED here on purpose: the
#    loader's own coast fade does the filling.
python tools/data/amip/extract_boundary_store.py \
    --input  $AI_ROSSBY_DATA/amip_dailyavg/1981.zarr \
    --output $AI_ROSSBY_DATA/amip_dailyavg_boundary/1981.zarr

# 4. amip_erdm_fancy ONLY — SST day-of-year climatology, fit on the TRAINING
#    years alone (--year-end is EXCLUSIVE, as upstream). Fitting the full record
#    would absorb the warming trend into the reference climatology and shrink
#    the very signal the anomaly channel exists to expose. This fork's artifact
#    is bitwise identical to upstream's committed one.
python tools/data/amip/make_sst_climatology.py \
    --zarr $AI_ROSSBY_DATA/amip_dailyavg_boundary \
    --year-start 1979 --year-end 2015 --harmonics 3 --stride 4 \
    --nan-fill 270.0 --smooth-sigma 1.5 --smooth-kernel-size 5 \
    --smooth-n-iters 10 \
    --out $AI_ROSSBY_DATA/norm_stats/sst_climatology.npz
```

### 6.5 Train — v1 families

Same Hydra composition as §2, with `train_diffusion.py` instead of `train.py`:

```bash
cd examples/weather/ai_rossby

# SI (DriftScheduler, single-step) — 45x90 coarse state, 1-degree forcings
python train_diffusion.py model=amip_si loss=si \
    dataset=amip_dailyavg_coarse training=amip_diffusion \
    validation=diffusion_rollout run_name=amip_si_run0

# SI_X (DynamicInterpolant, single-step), CO2 routed to the calendar row
python train_diffusion.py model=amip_si_x loss=si_x \
    dataset=amip_dailyavg_coarse training=amip_diffusion validation=off \
    run_name=amip_si_x_run0

# ERDM, v1 UNet backbone + v1 statistics (rolling window, W from cfg.loss)
python train_diffusion.py model=amip_erdm loss=erdm \
    dataset=amip_1981 training=amip_diffusion validation=diffusion_rollout \
    run_name=amip_erdm_run0

# RFM (rolling flow matching)
python train_diffusion.py model=amip_rfm loss=rfm \
    dataset=amip_1981 training=amip_diffusion validation=diffusion_rollout \
    run_name=amip_rfm_run0
```

**Training over more than one year.** Point `dataset.zarr_path` at the archive
*directory* rather than one `.zarr` and the loader routes to
`ClimateZarrMultiYearDataset` — `dataset=amip_dailyavg_coarse_multiyear` does
this. Until 2026-08-17 the diffusion recipe opened `ClimateZarrDataset`
unconditionally and could only ever train on a single year, while the upstream SI
and ERDM runs trained 1979–2015. Pairs and rolling windows dispatch across year
boundaries by global index, so no year is skipped at the seams:

```bash
python train_diffusion.py model=amip_si loss=si \
    dataset=amip_dailyavg_coarse_multiyear training=amip_diffusion \
    validation=diffusion_rollout run_name=amip_si_multiyear
```

**Per-channel noise scaling.** `loss/si.yaml` and `loss/si_x.yaml` ship
`noise_scale_path: null`, i.e. isotropic noise. Upstream's SI runs load a
per-channel scale (`sigma_c_lowres_26.pt`); build the equivalent from our own
store and point the config at it:

```bash
python tools/data/amip/make_noise_scales.py \
    --zarr $AI_ROSSBY_DATA/amip_dailyavg_coarse \
    --model-config examples/weather/ai_rossby/conf/model/amip_si.yaml \
    --mean $AI_ROSSBY_DATA/amip_dailyavg_coarse/normalize_mean_dailyavg.nc \
    --std  $AI_ROSSBY_DATA/amip_dailyavg_coarse/normalize_std_dailyavg.nc \
    --year-start 1979 --year-end 2015 \
    --out  $AI_ROSSBY_DATA/norm_stats/sigma_c_amip_dailyavg_coarse.pt
```

The scale is each channel's own 24-hour increment std in normalized units, so a
channel that barely moves per step stops being perturbed as hard as one that
does. **It is indexed by packed channel**, hence the required `--model-config`:
the builder derives the order from that wrapper's own `pack_state`, and an
artifact built for one `channel_layout` must not be used with another — the same
silent failure class as §6.2's contract mismatch. A sidecar `.json` records the
layout, years and channel count it was built from.

Keep `loss` and `model` paired as in §6.3: the loss config **is** the training
scheduler, and a rolling scheduler with a single-step wrapper (or the reverse) is
a shape error, not a graceful fallback. `loss.window_size` is the window length —
it must equal the model's `rolling_dit_kwargs.window_size` where the backbone has
one, because both size the same temporal tables.

`amip_erdm` and `amip_rfm` train **from scratch** on the `fork` layout, which is
self-consistent for that purpose. Starting either from real upstream v1 weights —
a warm start, an evaluation, a rollout — additionally needs
`++model.channel_layout=v1` (§6.2).

The SI pair describes the **real v1 checkpoints** (§6.3 footnote 4), so running
translated weights needs no layout override — they already say `v1`:

```bash
python inference.py model=amip_si dataset=amip_dailyavg_coarse loss=si \
    +inference.checkpoint_dir=./ckpt +inference.output_dir=./out \
    +inference.max_step=2 '+inference.ic_start=[8]'
```

**The store may serve more forcings than a model consumes.** SI-V lists 3 varying
channels where `amip_dailyavg_coarse` has 4 — upstream's run never fed
`global_mean_co2` — so the pipeline slices the stream down to the model's list
before the NaN-fill and aligns the normalizer to the same list. It logs
`varying-boundary subset active: … (indices=[1, 2, 3])`. The indices come from
name lookup, which matters here: the dropped channel is the store's *first*, so
taking the leading N would mis-assign every forcing.

### 6.6 Train — v2 ERDM

The v2 forecaster in its three shipped shapes. All three want
`training=amip_diffusion_bf16` (bf16 is measurably cheaper with no convergence
cost — `benchmarks/physicsnemo/experimental/models/amip_si/RESULTS.md`).

```bash
# (a) plain v2 — full 12e feature set, no predicted ocean
python train_diffusion.py \
    model=amip_erdm_v2 loss=erdm_v2 loss.window_size=6 \
    dataset=amip_dailyavg_coarse training=amip_diffusion_bf16 \
    validation=diffusion_rollout run_name=erdm_v2_run0

# (b) + predicted ocean channels (SST, sea ice), warm-started from (a).
#     Ocean support only ADDS parameters, so the expected report is
#     "loaded N/N keys" with ZERO skipped.
python train_diffusion.py \
    model=amip_erdm_v2_ocean loss=erdm_v2 loss.window_size=6 \
    dataset=amip_dailyavg_coarse training=amip_diffusion_bf16 \
    training.partial_checkpoint=./outputs/erdm_v2_run0/checkpoints/RollingDiTWrapper.0.50.mdlus \
    validation=off run_name=erdm_v2_ocean_run0

# (c) ERDM_fancy — the contract of the real v2 checkpoint: SST anomaly channel
#     + global_mean_sst trend scalar, nocean=3. The SST suite is a DATASET
#     switch; the model config only lists the channels it implies.
python train_diffusion.py \
    model=amip_erdm_fancy loss=erdm_v2 loss.window_size=6 \
    dataset=amip_dailyavg_coarse training=amip_diffusion_bf16 \
    dataset.sst_anomaly_channel=append \
    dataset.scalar_forcing=global_mean_sst \
    dataset.sst_climatology_path=$AI_ROSSBY_DATA/norm_stats/sst_climatology.npz \
    validation=off run_name=erdm_fancy_run0
```

What the run should print, and why each line is worth reading:

- `model step: 4 store row(s) (24 h)` — the 24-hour step reached the loader. The
  AMIP archives are 6-hourly; one row per step would advance the forcings 4×
  too fast.
- `ocean contract: nocean=2, c_grid indices=[1, 2]` — the scheduler adopted the
  predicted-ocean contract *from the model*. Without it the sampler never
  imposes the true ocean fields.
- `train/loss_ocean` alongside `train/loss` — the ocean block is ~1–2 % of a
  channel-summed loss and collapses fast, so it is logged separately; folded in,
  "learned it" and "weighted too low to matter" look identical.
- `partial checkpoint …: loaded N/M keys` — with **any** `SKIPPED` line, the two
  configs differ somewhere beyond the added ocean parameters. Stop and find out
  what before spending GPU hours.

Notes on the SST suite (Phase 12g), all enforced rather than documented-only:

- `sst_anomaly_channel: append` derives an SST-anomaly channel against the
  day-of-year climatology and divides by the ~0.6 K residual std instead of the
  ~12.3 K absolute-SST std — which is what makes the ocean warming trend legible
  (+0.40 K goes from 0.03 σ to ~0.7 σ). The channel is **derived, not stored**,
  and the model must list it in `varying_boundary_variables` in post-rescaler
  order (right after absolute SST) or `c_grid_dim` comes out one short.
- `scalar_forcing: global_mean_sst` routes the ocean-mean anomaly into the
  calendar row's third slot — **the slot CO₂ uses**. Asking for both raises
  rather than silently overwriting, so `amip_erdm_fancy` drops
  `global_mean_co2` from its varying list and leaves
  `scalar_routed_boundary_variables` empty.
- `sst_climatology_path` must point at the `.npz` from §6.4 step 4. Left `null`
  (as shipped) the whole suite is inert.

### 6.7 Knobs both generations share

- **Multi-GPU:** `torchrun --standalone --nproc-per-node=N train_diffusion.py …`.
  Requires `torch<2.11`, and wandb initializes on **every** rank — never inside
  `if rank == 0:` (that asymmetry was the root cause of the 2026-08-07 DDP-init
  NCCL hang). `wandb.allow_multigpu=false` re-arms the auto-disable guard if it
  ever resurfaces.
- **Optimizer:** Muon (`uv pip install muon`), lr 5e-5, wd 3e-6 — upstream's
  choice, in `conf/training/amip_diffusion*.yaml`.
- **Precision:** `training=amip_diffusion` is fp32; `training=amip_diffusion_bf16`
  is the same config with `amp: bf16` (no GradScaler needed — only fp16 wants one).
- **Resume** by relaunching with the same `run_name`: the run reloads its own
  `checkpoints/` (weights **and** optimizer state). `training.partial_checkpoint`
  is deliberately ignored when that resume finds something, so restarting a long
  job never rewinds to the warm-start weights.
- **EMA** is on by default (decay 0.999, 6-epoch warmup) and is persisted in the
  checkpoint metadata; `+inference.use_ema=true` swaps it in at rollout time.
- **Multi-stage curricula:** each entry in `training.stages` may carry
  `loss_overrides` (e.g. a W=3 pretrain then a W=6 finetune). The recipe rebuilds
  the DataLoader *and* the scheduler at any stage boundary where the merged
  `window_size` changes, so single-step ↔ rolling transitions are allowed.
- **The model's timestep lives in the model config** (`timedelta_hours`), not the
  dataset's, because one store feeds families with different steps. Every AMIP
  config — v1 and v2 — is 24 h over 6-hourly rows, i.e. 4 rows per step; the
  dataset's row-level `forecast_lead_times: [4]` is cross-checked against it and a
  disagreement raises. Anything that needs a *duration* rather than a stride
  (eval-suite bin widths, §6.10) derives it from the same place via
  `train_loop.model_step_hours` / `steps_per_month`, so no config restates a
  cadence it cannot verify.

### 6.8 Mid-training validation

`validation=diffusion_rollout` scores a short rollout every
`validation.every_n_epochs` (default 5 — full sampling every epoch is expensive):

```bash
python train_diffusion.py model=amip_erdm_v2 loss=erdm_v2 \
    dataset=amip_dailyavg_coarse training=amip_diffusion_bf16 \
    validation=diffusion_rollout \
    validation.rollout.log_steps='[1, 3, 6]' \
    validation.rollout.max_initial_conditions=4 \
    validation.rollout.ensemble_size=4 \
    validation.rollout.sampler.num_steps='[20, 20, 10, 10, 4, 4]' \
    run_name=erdm_v2_run0
```

- `log_steps` are lead times in **model steps**. For a rolling scheduler the
  horizon defaults to `window_size`, so every entry must fit inside it unless you
  raise `validation.rollout.horizon`.
- `sampler.num_steps` is decoupled from the training scheduler's: a single int
  applies uniformly, a list of length `horizon` gives a per-frame schedule
  (spend solver steps on the early, harder frames and taper). This is what makes
  long validation rollouts tractable.
- Spread metrics appear only when `ensemble_size > 1`. Metrics route to
  `valid/*` in wandb.

### 6.9 Rollout inference

```bash
# v2 ERDM (rolling): one sample_rollout call emits the whole horizon
python inference.py model=amip_erdm_v2 dataset=amip_dailyavg_coarse \
    sampler=from_loss loss=erdm_v2 \
    +inference.checkpoint_dir=./outputs/erdm_v2_run0/checkpoints \
    +inference.output_dir=./outputs/erdm_v2_run0/rollout \
    +inference.max_step=60 '+inference.ic_start=[0, 120, 240]' \
    +inference.sampler_num_steps=4 +inference.use_ema=true \
    +inference.output_format=zarr

# v1 single-step (SI): autoregressive, one sample() per emitted frame
python inference.py model=amip_si dataset=amip_dailyavg_coarse sampler=si loss=si \
    +inference.checkpoint_dir=./outputs/amip_si_run0/checkpoints \
    +inference.output_dir=./outputs/amip_si_run0/rollout \
    +inference.max_step=20 '+inference.ic_start=[0, 60]'
```

Required: `checkpoint_dir`, `output_dir`, `max_step`, and one of `ic_start`
(store row indices) / `init_schedule` (calendar-based). Useful extras:
`ensemble_size`, `perturber` (`deterministic` | `replicate_only` | `gaussian_ic`
— the default for `ensemble_size > 1` is `replicate_only`, since the ensemble
axis comes from the sampler's own noise), `sampler_num_steps`, `step_size`,
`ensemble_save_mode`, `writer_num_workers`.

Three things the driver does for you, all of which used to be manual:

- **Sampler choice.** `sampler=from_loss` (the default) instantiates `cfg.loss`,
  so inference uses the training scheduler. Override with `sampler=<family>` for
  a faster or different sampler — but only within a compatible arity
  (`sampler=edm` and `sampler=x_ddc` do **not** fit this driver, §6.15).
- **Ocean contract.** The scheduler adopts `nocean` / `ocean_grid_indices` from
  the loaded model, not from a config, so a checkpoint with predicted ocean
  channels is handed a correctly padded window and has the true ocean re-imposed
  each roll.
- **Step stride.** `step_size` defaults to the model's own contract (4 rows =
  24 h), so frames get the right valid time and the forcings advance at the right
  rate.

Frame 0 of each per-IC file is the IC's last window frame in physical units;
frames 1…`max_step` are the forecast. Score the result with `validate_cli.py`
(§5) — it only reads files, so it is generation-agnostic.

### 6.10 Long-horizon eval suite

`eval_diffusion.py` reuses the same rollout mechanics but scores **every**
emitted frame of a long (e.g. one-year) rollout into streaming aggregators:

```bash
python eval_diffusion.py model=amip_erdm_v2 loss=erdm_v2 \
    dataset=amip_dailyavg_coarse validation=eval_suite \
    +eval_suite.checkpoint_dir=./outputs/erdm_v2_run0/checkpoints \
    +eval_suite.output_path=./outputs/erdm_v2_run0/eval_suite.pt \
    eval_suite.horizon=365 eval_suite.sampler_num_steps=4
```

Five independently-toggled validators: `climatology` (per-variable time mean +
per-bin, e.g. monthly, climatology), `bias` (signed lat-weighted global-mean per
group), `qbo` (30°S–30°N zonal-mean U at stratospheric levels, binned, with a
zero-crossing period estimate), `global_mean` (lat-weighted global-mean flux
timeseries), and `ensemble_envelope` (spread/skill; off by default — it costs E×
a rollout). Results are a `torch.save` dict at `output_path`.

`validation=eval_suite` selects from the `validation` group but the suite is read
from `cfg.eval_suite`; the routing is a `# @package eval_suite` directive on the
first line of `conf/validation/eval_suite.yaml` (added 2026-08-14 — before that
the invocation the config itself documented could not work). Keep it on line 1:
Hydra does not look for it further down the header.

**Bin widths are derived from the model step, not configured** (2026-08-14). Each
aggregator block states `months_per_bin`, and `train_loop.steps_per_month`
converts it using `cfg.model.timedelta_hours` and the store's own cadence — 30
steps per monthly bin at the AMIP 24-hour step, 122 at a 6-hour one. Neither
number appears in the config, which is the point: it previously hard-coded
`steps_per_bin: 120` labelled "≈ 1 month", true only for a 6-hourly model, so
every AMIP "monthly" climatology was silently binned four months wide and QBO's
period estimate came out ~4× off. Setting `steps_per_bin` explicitly still wins
and logs a warning that the bins are not the stated width.

**`horizon` is in model steps and is *not* derived**, because a multi-year
archive legitimately supports a multi-year rollout. The shipped `1460` is ~1 year
at 6 h and ~4 years at the AMIP 24 h, so pass `eval_suite.horizon=365` for a
single-year store. A horizon the store cannot serve now raises, and says what
fits:

```
ValueError: no admissible initial condition: horizon=1460 x step_size=4 needs 5840
future rows (plus 20 past) but the store has 1460. Largest horizon this store
supports is 360.
```

That used to be silent: zero ICs selected, zero samples drawn, and **RMSE 0.0**
reported — a flawless-looking model that was never evaluated.

The two channel names in that config were also placeholders that matched no
model (`ua`, `DSWRFtoa`) and are now the AMIP ones: zonal wind is
`u_component_of_wind` everywhere, and the TOA fluxes are the `*toa_24h`
**diagnostics** — v1's `DSWRFtoa` is a *varying boundary*, so it was never
eligible. Override them for a non-AMIP model; both validators raise on an unknown
name.

Truth comes from the dataset's own trajectory in the same Zarr, matching
upstream's convention — there is no separately shipped oracle file.

### 6.11 Bringing in an upstream checkpoint

**Translate.** Prefer the auto-derive path (no `--model-config`): it reads the
wrapper kwargs out of the checkpoint's own `hyper_parameters` block instead of
trusting a hand-written YAML to agree with the weights.

```bash
# v2 (ERDM/RollingDiT forecaster)
python tools/checkpoint_translation/amip_si.py \
    --source .../ERDM_fancy_42_2026-08-10T13-21-13/last.ckpt \
    --source-contract v2 --target-class RollingDiTWrapper \
    --output erdm_fancy.mdlus --strict

# v2 (x_DDC / DiTAE downscaler)
python tools/checkpoint_translation/amip_si.py \
    --source .../x_DDC_42_2026-08-07T09-34-49/last.ckpt \
    --source-contract v2 --target-class XDDCWrapper \
    --output x_ddc_dit.mdlus --strict

# v1 (any of SI / SI_X / ERDM-UNet / RFM / EDM / x_DDC-UNet)
python tools/checkpoint_translation/amip_si.py \
    --source .../SI_X_AIMIP_interp_gaussian_42_2026-05-28T09-27-49/last.ckpt \
    --source-contract v1 --output si_x.mdlus --strict
```

Flags that matter: `--strict` refuses to write on any missing/unexpected key;
`--prefer-live` takes `current_model_state` (live training weights) instead of
the default EMA-averaged `state_dict`; `--target-class` forces the wrapper when
`model_name` is ambiguous (`ERDM` means v1's UNet *or* v2's RollingDiT — the
`model.backbone` field decides); `--amip-repo` / `$AI_ROSSBY_AMIP_REPO` is needed
only to unpickle the source normalizer.

**Then verify numerically.** Key parity proves nothing about channel order — a
clean `load_state_dict` and a wrong permutation look identical:

```bash
python tools/checkpoint_translation/verify_v2_numerical.py \
    --source .../last.ckpt --amip-v2-repo <amip_v2 checkout> --family erdm
```

It builds upstream's backbone and ours in one process (their `RollingDiT` /
`DiTAE` import only torch + einops, so no Lightning and no `norm_stats`), hands
both the same weights, compares the **forward**, and localises any mismatch per
channel block. `--synthetic` self-tests the harness with no checkpoint. Both real
v2 checkpoints come out at `max │diff│ = 0.0000e+00`.

`--family si` does the same for the frozen v1 single-step families against
upstream **v1**'s own `DiT` — note the repo differs, since amip_v2 deleted them
(`--amip-repo` names either; `--amip-v2-repo` still works):

```bash
python tools/checkpoint_translation/verify_v2_numerical.py \
    --source .../SI_X_AIMIP_wCO2_.../model_epoch=19.ckpt \
    --amip-repo $AI_ROSSBY_AMIP_REPO --family si --tol 0
```

**One step further — the same comparison on a real batch.** The above uses
`torch.randn`, which isolates the translation but says nothing about whether your
*config and loader* hand the model what upstream's did. That is a separate
failure mode, and for the SI checkpoints it was the live one (a 45×90 state with
180×360 forcings, which no shipped SI config described):

```bash
python tools/checkpoint_translation/verify_si_realdata.py \
    --model amip_si --dataset amip_dailyavg_coarse \
    --translated $CKPT/translated/si_v_....mdlus \
    --source $CKPT/SI_v_.../last.ckpt --amip-repo $AI_ROSSBY_AMIP_REPO
```

It builds the wrapper **from the model config** (not `Module.from_checkpoint`,
which would rebuild from the artifact's own args and prove nothing about the
config), asserts its channel contract against the artifact, loads strictly, pulls
a real sample through the recipe's own fill → normalize → route pipeline, and
checks that `c_grid` arrives at exactly `c_grid_downsample ×` the state grid.
Both SI checkpoints pass at `0.0000e+00` on real daily-average data.

**Then run it.** The translator writes a bare `Module.save()` artifact, but
`load_checkpoint` looks up `<WrapperClass>.<mp_rank>.<index>.mdlus` inside a
directory. So give it that name:

```bash
# v2: the shipped config already states channel_layout: v2
mkdir -p ckpt && cp erdm_fancy.mdlus ckpt/RollingDiTWrapper.0.0.mdlus
python inference.py model=amip_erdm_fancy dataset=amip_dailyavg_coarse \
    loss=erdm_v2 +inference.checkpoint_dir=./ckpt …

# a fork-default config (amip_erdm / amip_rfm) needs the layout stated, or the
# upper-air block is silently permuted. The SI pair already says v1.
mkdir -p ckpt_v1 && cp rfm.mdlus ckpt_v1/RollingDiTWrapper.0.0.mdlus
python inference.py model=amip_rfm ++model.channel_layout=v1 dataset=amip_1981 \
    loss=rfm +inference.checkpoint_dir=./ckpt_v1 …
```

Two log lines to read here. `channel contract verified against
RollingDiTWrapper.0.0.mdlus: channel_layout='v2'` means the artifact and
`cfg.model` agree (§6.2) — if they don't, this raises instead of loading. And
`Could not find valid checkpoint file, skipping load` is expected and harmless:
that is the *optimizer-state* file, which a translated artifact has no reason to
carry. The weights loaded if you see `Loaded model state dictionary …`.

A translated `.mdlus` is also directly usable as `training.partial_checkpoint`
(§6.6) without renaming — the same contract check applies there.

Real-checkpoint provenance, contracts and the four gotchas the live translation
exposed are in
[`docs/dev/phase8e_midway3_checkpoint_inventory.md`](../../../docs/dev/phase8e_midway3_checkpoint_inventory.md#amip_v2-checkpoints-2026-08-14).

### 6.12 `CombinedModule` (forecaster → downscaler)

The two-stage cascade — a 45×90 ERDM forecaster whose every emitted frame is
bilinear-upsampled and handed to the 180×360 x_DDC downscaler as conditioning.
There is no standalone "Combined" checkpoint upstream and none here: it is
composed from two independently-trained checkpoints, for **inference only** (it
is not trained end to end). No CLI wraps it; this is the whole driver:

```python
import sys
import hydra
from omegaconf import OmegaConf
from physicsnemo import Module
from physicsnemo.experimental.models.amip_si import CombinedModule

RECIPE = "examples/weather/ai_rossby"
sys.path.insert(0, RECIPE)                       # the recipe dir is not a package
from train_loop import adopt_ocean_contract      # noqa: E402

# from_checkpoint restores each wrapper WITH ITS STORED ARGS, so the channel
# contract comes from the artifact and cannot disagree with a YAML.
forecaster = Module.from_checkpoint("translated/erdm_fancy.mdlus").eval()
downscaler = Module.from_checkpoint("translated/x_ddc_dit.mdlus").eval()

# window_size / num_steps must be passed at CONSTRUCTION — the sigma tables are
# derived there, so assigning the attribute afterwards is silently wrong.
f_sched = hydra.utils.instantiate(
    OmegaConf.load(f"{RECIPE}/conf/loss/erdm_v2.yaml"), window_size=6, num_steps=2
)
d_sched = hydra.utils.instantiate(OmegaConf.load(f"{RECIPE}/conf/sampler/x_ddc.yaml"))
adopt_ocean_contract(f_sched, forecaster)        # nocean + c_grid indices

combined = CombinedModule(
    forecaster=forecaster, forecaster_scheduler=f_sched,
    downscaler=downscaler, downscaler_scheduler=d_sched,
).eval()

# Streaming rollout: one frame per step, checkpointable between them.
# `init` is a (b, W, C, 45, 90) oracle window — pass a BARE STATE window even
# under nocean>0; it is zero-padded here and the first roll imposes the truth.
x_bar, eps = combined.windowed_init(init)
for k in range(horizon):
    y_highres, x_bar, eps = combined.windowed_step(
        x_bar, eps,
        c_grid_win[k], c_scalar_win[k],       # (b, W, …) forcing window at step k
        ocean_win=c_grid_win[k + 1],          # forcings shifted one step forward
    )
    # y_highres is (b, C_state, 180, 360) in normalized units
```

Two properties this path is pinned to (`test/models/amip_si/test_combined_windowed.py`):

- **Streaming == batch.** `windowed_init`/`windowed_step` draw noise in the same
  order as `ERDMScheduler.sample_rollout`, so the same seed gives identical
  rolling state step for step. Otherwise streaming would be a second
  implementation free to drift.
- **The ocean tail comes off before the downscaler**, which is a pretrained
  state-width model. Because the predicted-ocean block sits at the *end* of the
  channel axis, passing it through would be a silent width error rather than an
  obvious one. A rolling state whose channel count disagrees with the forecaster
  is refused outright, since a streaming driver may resume from disk.

**`rollout.py` drives this from the command line** (2026-08-17) — the streaming
API had no driver until then:

```bash
python rollout.py model=amip_combined dataset=amip_dailyavg_coarse \
    +rollout.output_dir=./outputs/cascade \
    +rollout.ic_start=8 +rollout.horizon=120 \
    +rollout.forecaster_num_steps=2 +rollout.downscaler_num_steps=5
```

It adds six things `inference.py` structurally cannot do: two checkpoints and two
schedulers (each contract-checked before loading); output coords from the
**downscaler's** grid, read from a store rather than synthesized — `inference.py`
takes them off the driving store, which would label a 180×360 field with the
forecaster's 45 latitudes, and inventing a latitude vector would additionally risk
the row order (§6.4); one `windowed_step` per frame instead of one
`sample_rollout` over the horizon; mid-rollout resume via an atomically-written
`(x_bar, eps_prev, step)`; separate forecaster/downscaler solver budgets; and
month-buffered output instead of one file per IC.

`conf/model/amip_combined.yaml` records the v1 pairing (`amip_si_x` + `amip_x_ddc`
with their checkpoint paths) in the same shape.

### 6.13 Health gates: run these before trusting a config

```bash
# every conf/model/amip_* instantiates, forwards at its own derived widths,
# has a self-consistent channel contract, and matches its pinned shape digest
pytest test/models/amip_si/test_config_health_gates.py -q

# wrappers, schedulers, streaming rollout, translator key layouts
pytest test/models/amip_si test/diffusion test/tools/checkpoint_translation -q
```

```bash
# the checkpoint/config contract guard + the SI_X tuple unwrap
pytest test/recipes/ai_rossby/test_checkpoint_contract.py \
       test/recipes/ai_rossby/test_validate_diffusion.py -q
```

The shape-signature digest is the gate that catches this port's worst class of
bug directly: the translator once built `static_bias` at `[256, 180, 360]`
instead of `[256, 45, 90]` (data resolution vs. token grid). With
`boundary_static_bias: false` that error loads cleanly and is silently wrong —
`static_bias` is the only grid-shaped parameter in the tree. Regenerate a digest
**deliberately** when a config or backbone changes shape.

What the digest **cannot** see is a packing change: `channel_layout` moves no
parameter shape, so `amip_x_ddc` hashes to `ba2d45a0801366be` under both `v1` and
`v2`. Two things cover that gap — `test_channel_layout_is_pinned` asserts the
expected layout per config (so a flip has to be deliberate), and §6.2's run-time
guard catches an artifact that disagrees with whatever the config says.

### 6.14 Cluster job scripts

Working launchers, in the order you'd use them:

| Script | What it runs |
|---|---|
| `hpc/scripts/convert_amip_dailyavg_derecho.pbs` | H5 → per-year Zarr (source lives only on Derecho scratch) |
| `hpc/scripts/coarsen_amip_dailyavg_derecho.pbs` | the 45×90 coarse store |
| `hpc/scripts/extract_amip_boundary_stampede3.sbatch` | the 1° boundary-only stores |
| `hpc/scripts/make_sst_climatology_polaris.pbs` | the SST climatology `.npz` |
| `hpc/scripts/translate_v2_checkpoints_polaris.pbs` | translate **and** numerically verify both real v2 checkpoints |
| `hpc/scripts/verify_si_checkpoints_polaris.pbs` | same for the two real **v1 SI** checkpoints (needs the v1 checkout) |
| `hpc/scripts/run_si_coarse_configs_polaris.pbs` | the SI configs end to end: real-data A/B vs upstream + an `inference.py` rollout |
| `hpc/scripts/smoke_amip_ocean_polaris.pbs` | v2 ocean + warm start + DDP, on real data, with assertions on the log |
| `hpc/scripts/smoke_amip_v2_layout_2xA40.sbatch` | v2 channel layout under DDP |
| `hpc/scripts/smoke_amip_diffusion_2xA40.sbatch` | v1 SI wiring smoke (1 mini-epoch) |
| `hpc/scripts/bench_amip_diffusion_bf16.sbatch` | fp32-vs-bf16 throughput/memory |

### 6.15 Known limitations & gotchas

- **`sampler=edm` and `sampler=x_ddc` don't fit `inference.py`.**
  `EDMSchedulerModule.sample(initial_cond, model, …)` and
  `DataDependentInterpolant.sample(model, x_lowres, …)` have different arities
  than the `sample(model, x, c_grid, c_scalar, num_steps=…)` the driver calls.
  Both configs exist for bespoke drivers; EDM additionally has no shipped model
  config (translate to `AmipDiTWrapper` and supply your own geometry).
- **`channel_layout` is read from the YAML at run time**, not from the `.mdlus`
  — see §6.2. Mismatches are now refused rather than run, but the *correct*
  layout still has to be supplied by you.
- **SI_X's sampler returns a tuple.** `DynamicInterpolant.sample` hands back
  `(y, model_last_pred)` under its `return_model_last=True` default where the
  other single-step schedulers return a tensor. The two rollout drivers unwrap it
  as of 2026-08-14 (before that, every SI_X rollout and validation died on
  `'tuple' object has no attribute 'dim'/'narrow'`; training was unaffected). Any
  new driver calling `scheduler.sample` needs the same two-line unwrap that
  `CombinedModule.forward` has.
- **Scheduler `window_size` / `num_steps` / `l_max` / `noise` are construction
  arguments.** The sigma tables and the spherical-noise basis are built in
  `__init__`, so mutating the attribute afterwards silently mismatches. `l_max`
  in particular is tied to the grid: the shipped `sampler/x_ddc.yaml`
  (`noise: spherical`, `l_max: 180`) is sized for the real 180×360 downscaler.
- **Forcings are lagged one step under `v1`/`v2`** (slot `w` holds the state at
  step `w+1`, conditioned on the forcing at step `w`) and unlagged under `fork`.
  This is derived from `channel_layout`, not configured — but it means
  v1-checkpoint evaluations produced before 2026-08-12 were one step off.
- **Don't `import physicsnemo` on a login node** for small scripts — CUDA/Warp
  init can core-dump. Use plain xarray/numpy, or run on a compute node (the
  translation and climatology scripts do).
- **Muon needs a process group.** Single-process runs are handled, but a bare
  `python -c` experiment with the optimizer may need
  `torch.distributed.init_process_group` first.

## Files

Deterministic recipes (Pangu / SFNO / ArchesWeather):

- `train.py` / `train_loop.py` — training entrypoint + step logic. `train_loop`
  also holds the pieces every driver shares: the model step (`model_step_rows`,
  `model_step_hours`, `steps_per_month`), the predicted-ocean contract
  (`adopt_ocean_contract`), warm start (`load_partial_weights`), and the
  checkpoint/config contract guard (`assert_checkpoint_contract`, §6.2)
- `inference.py` — autoregressive rollout, one file per IC (Zarr or NetCDF);
  also the diffusion rollout driver (§6.9)
- `validate.py` / `validate_cli.py` — mid-training + after-the-fact scoring
- `climatology.py` / `climatology_cli.py` — streaming climatology aggregators
- `loss.py`, `ema.py`, `async_writer.py`, `dataset_setup.py`, `data_staging.py` —
  supporting pieces

AMIP diffusion (§6):

- `train_diffusion.py` — diffusion training entrypoint (single-step + rolling)
- `rollout.py` — two-stage cascade driver (forecaster → downscaler), streaming
- `validate_diffusion.py` — mid-training rollout validator (library, no CLI)
- `eval_diffusion.py` — long-horizon climate eval suite
- `conf/` — the Hydra config groups above (`model`, `loss`, `sampler`, `dataset`,
  `training`, `validation`)

Out-of-tree, but part of the same workflow:

- `tools/data/amip/` — store conversion, coarsening, boundary extraction, SST
  climatology
- `tools/checkpoint_translation/amip_si.py` + `verify_v2_numerical.py` —
  upstream `.ckpt` → `.mdlus`, and the forward-equivalence check
- `hpc/scripts/` — cluster launchers (§6.14)
