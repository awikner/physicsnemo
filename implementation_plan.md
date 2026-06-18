# AI-Rossby Implementation Plan — Porting PanguWeather v2.0 & amip into PhysicsNeMo

Status: **in progress (Phase 1)** · Author: Claude (analysis + port) · Updated: 2026-06-16

This plan covers the first objective in [`project_outline.md`](project_outline.md): porting all current
models, training/inference code, and mid-training & after-the-fact validation from **PanguWeather/v2.0** and
**amip** into the **PhysicsNeMo** framework. It is grounded in a deep read of all three repositories
(findings summarized in §1).

## Decisions locked in (from review)

1. **Repo strategy — Fork & in-tree.** `ai-rossby` work happens in the user's PhysicsNeMo fork
   (`awikner/physicsnemo`, local at `/work/nvme/bdiu/awikner/physicsnemo`, branch `ai-rossby`).
   Ported models live under `physicsnemo/models/...`, recipes under `examples/weather/...`, tests under
   `test/models/...`, registered via PhysicsNeMo entry-point/registry conventions. Upstream tracked for merges.
2. **Fidelity — weight-compatible, two flavors per architecture.** Every ported model ships as:
   - a **faithful** variant: an exact reproduction of the original architecture (same submodule names, same
     forward math) so existing trained checkpoints load via translation scripts; and
   - a **native** variant: the same model rebuilt on native PhysicsNeMo building blocks (`physicsnemo.nn`,
     diffusion stack, StaticCapture-friendly, AMP/cuda-graph metadata).
   Both variants implement the **same input/output contract** so a single datapipe + recipe serves both.
   (These are the "legacy/updated" flavors from review, renamed **faithful/native** to avoid colliding with
   the separate *legacy Pangu architecture* — see Naming.)
3. **Sequence — Pangu_Plasim first**, end-to-end (model → data → train → validate → translate), establishing
   the shared skeleton the diffusion port reuses. amip stochastic-interpolant model follows.
4. **Validation — on HPC.** Logic is unit-tested locally on synthetic data; real training/inference and
   numerical-fidelity checks run on the cluster (PBS: Derecho/Casper; SLURM: Midway). PhysicsNeMo's
   `DistributedManager` auto-detects SLURM and torchrun; PBS is handled via MPI/env vars.

### Naming (two orthogonal axes — read this to avoid confusion)
- **Architecture axis** (which PanguWeather network): `pangu_plasim` = the *current* `networks/pangu.py`
  model (with the training-only VAE dual-encoder + KL); `pangu_plasim_legacy` = the *predecessor*
  `networks/pangu_legacy.py` model (no VAE). "Legacy Pangu" always means this no-VAE architecture.
- **Fidelity axis** (how we port it): `faithful` (weight-compatible reproduction) vs `native` (rebuilt on
  PhysicsNeMo blocks). The faithful flavor is the default class name; native variants carry a `Native` suffix.

## 1. Source-repo findings (the facts this plan is built on)

### PanguWeather/v2.0 (deterministic, custom PyTorch)
- **Models** (`networks/`): `PanguModel_Plasim` (`pangu.py`, primary) — Pangu-style 3D Swin / Earth-Specific
  transformer with a *training-only VAE dual-encoder + KL term*, **separate surface vs. upper-air streams**,
  constant/varying boundary conditioning (TOA solar radiation special-cased into the 3D stream), optional
  `predict_delta` mode (+ `Integrator`), configurable grid (PLASIM 64×128, ERA5/S2S 180×360). Plus a
  **legacy `PanguModel_Plasim`** (`pangu_legacy.py`, no VAE; forward returns 4–5 values, selected via
  `use_legacy_model`), a vendored **SFNO** (`networks/modulus_sfno/`, Modulus/makani-derived), and `pangu_lite`.
- **Forward contract** (model-agnostic across current/legacy Pangu & SFNO):
  `forward(surface, const_boundary, varying_boundary, upper_air, train=False, ...)` →
  `(out_surface, out_upper_air[, out_diag], mu, sigma, mu2, sigma2)`.
- **Training** (`train.py`, ~4.4k lines, custom `Trainer`): DDP (`find_unused_parameters=True`), bf16 AMP
  (fp16 w/ GradScaler), **EMA**, AdamW (`fused`)/optional ZeRO-1, OneCycle/cosine/warmup schedulers,
  loss = `surface*0.25 + pl (+ diag*0.25)` or raw channel-weighted, optional VAE-KL, no grad-accum.
- **Data**: per-timestep **HDF5** `{year}_{idx:04d}.h5` (one dataset per variable), z-score stats in
  **NetCDF**; separate surface/upper-air tensors; config-driven channel lists.
- **Checkpoints**: plain dict `{iters, epoch, model_state, optimizer_state_dict, ema_state, scheduler_state_dict, ...}`;
  possible `module.` prefix; `ema_state` preferred at inference; `Integrator` std buffers reconstructed from norm stats.
- **Mid-training validation**: autoregressive rollout → lat-weighted **RMSE + ACC** (dayofyear climatology),
  long-rollout climate **bias**, ensemble forecast validation, power spectra/GIF diagnostics.
- **After-the-fact**: `inference*.py`/`long_inference.py`/`ensemble_inference.py` (shared `Stepper`), NetCDF
  output, MC-dropout + IC-perturbation ensembles, observation/event metrics.
- **Config**: YAML + argparse (`utils/YParams.py`). **Optional deps**: transformer_engine (FP8, off),
  torch_harmonics (SFNO + spectral loss), apex (optional). No flash-attn/xformers/external makani.

### amip (stochastic-interpolant latent diffusion, PyTorch Lightning)
- **Models** (`modules/`): `TrainModule` dispatching `SI`/`SI_X`/`ERDM`; primary = **`SI_X`** = **DiT** backbone +
  custom **x-prediction stochastic interpolant** (`DynamicInterpolant`, exponential integrator, ~5 steps,
  spherical-harmonic noise). Also `SI` (velocity), `ERDM` (rolling diffusion), `x_DDC` downscaler, and an
  eval-only `CombinedModule`.
- **"Latent" is a fixed bilinear ×4 downsample** (180×360→45×90), **not a learned VAE**; autoregression on the
  coarse physical grid. **Channels = 151** = surface(6)+diagnostic(15)+multilevel(5×26).
- **Conditioning splits three ways**: `cond` (current state, channel-concat), `c_grid` (5 spatial forcings:
  SST, sea-ice, TOA insolation + orography, LSM; stride-4 conv-embedded), `c_scalar` (calendar `[sod, doy]`
  cyclic + `co2` acyclic).
- **Training** (Lightning): **Muon** optimizer (2 param groups, ≥2D weights at 10×lr) + EMA
  (`EMAWeightAveraging`), `precision=32-true`, `StepLR`. **All diffusion is hand-written** (no `diffusers`).
- **Diffusion math** (`modules/diffusion/`): interpolant `X_t=(1-t)x+t·y+(1-t)σ√t·noise`; x-prediction loss;
  exponential/log-uniform sampler `x_next = r·x_t + (1-r)·x1_pred + σ(1-t)·dW`. Spherical-harmonic noise
  via `torch_harmonics`. 2D RoPE on physical lat/lon; `SphereConv2d`; pole-aware `sphere_pad`.
- **Data**: per-timestep **HDF5** + per-var/per-level z-score stats in **NetCDF**; conditioning forcings;
  `cftime` calendars; global-scalar CO₂ appended to calendar.
- **Checkpoints**: Lightning `.ckpt`; backbone under `model.*`, per-channel noise buffer under
  `scheduler.noise_scales`; full config under `hyper_parameters` (+ sibling `config.yml`).
- **Validation**: lat-weighted RMSE at {1,3,5,10}-day leads + power spectra (mid-training); climatology/bias,
  **QBO time-height**, global-mean t2m timeseries, ensemble envelopes (`evals/`, `bias.py`). No ACC.
- **Live coupling to `old/`**: `modulate_fused`, `MLP` (`old/fa_basics.py`), `contractions.*` — must travel
  with the port.

### PhysicsNeMo (target framework, v2.2.0a0)
- **Pangu already ships** (`physicsnemo/models/pangu/pangu.py`, `Pangu(Module)`) with a full recipe
  (`examples/weather/pangu_weather/`) — **but** it's the standard 721×1440 ERA5 single-tensor `forward(x)`
  variant; it does **not** match `PanguModel_Plasim` (dual-stream/boundary/VAE/delta). It's a **template**, not
  a drop-in. Reusable building blocks: Earth-Specific attention, patch embed/recovery.
- **`physicsnemo.Module`** (`physicsnemo/core/module.py`): base class with **JSON-serializable `__init__`
  args** (auto-captured), `.mdlus` checkpoints (ZIP of `model.pt`+`args.json`+`metadata.json`),
  `.save()`/`.load()`/`.from_checkpoint()`, `ModelMetaData` capability flags, entry-point + runtime registry.
- **No Trainer abstraction** — example-based loops over building blocks: `DistributedManager`
  (`physicsnemo/distributed`), `save_checkpoint`/`load_checkpoint` (`physicsnemo/utils/checkpoint.py`, FSDP/DTensor-aware),
  `LaunchLogger` (`physicsnemo/utils/logging`), `StaticCaptureTraining` (`physicsnemo/utils/capture.py`,
  CUDA-graphs+AMP+GradScaler).
- **Diffusion** (`physicsnemo/diffusion`): prediction-agnostic, protocol-based (x0/score/epsilon — **no
  v/velocity loss, no interpolants, no flow-matching, no latent-diffusion/VAE example**). EDM preconditioners,
  noise schedulers (`LinearGaussianNoiseScheduler` base), samplers/solvers (`sample()`, Euler/Heun/stochastic),
  losses (`MSEDSMLoss` w/ `*_to_x0_fn` callbacks), guidance, multi-diffusion. Backbones: `SongUNet*`, `DhariwalUNet`,
  **`DiT`**. **TopoDiff** example is the canonical "custom scheduler/solver in user code" pattern.
- **Datapipes** (`physicsnemo/datapipes`): `ERA5HDF5Datapipe` expects **per-year** HDF5 with a single
  `fields(T,C,H,W)` array + `.npy` stats — **mismatch** with both source repos (per-timestep, per-variable,
  NetCDF stats). Legacy `Datapipe` base + new `Reader/Transform/Dataset/DataLoader` arch both available.
- **Metrics** (`physicsnemo/metrics`): `acc` (lat-weighted), lat-weighting reductions, `mse`/`rmse`
  (unweighted), `crps`/`kcrps`, `power_spectrum`, ensemble metrics. (RMSE lat-weighting = compose mse + reductions.)
- **Config**: Hydra + OmegaConf throughout examples (flat vs. composed `conf/base/*` groups).
- **Distributed/HPC**: `DistributedManager` auto-detects ENV(torchrun)/SLURM/OpenMPI; no native PBS (use MPI/env).
- **Tests**: `test/common` validators — `validate_forward_accuracy`, `validate_checkpoint`,
  jit/cuda-graph/amp/onnx validators gated by `ModelMetaData` flags. Per-model `test/models/<name>/test_<name>.py`.

## 2. Target architecture in the fork

```
ai-rossby (= awikner/physicsnemo, branch ai-rossby)
├── physicsnemo/
│   ├── models/
│   │   ├── pangu_plasim/               # NEW — phases 1 & 6
│   │   │   ├── __init__.py
│   │   │   ├── pangu_plasim.py         # PanguPlasim — faithful port of pangu.py (VAE)         [P1]
│   │   │   ├── pangu_plasim_legacy.py  # PanguPlasimLegacy — faithful port of pangu_legacy.py  [P1]
│   │   │   ├── pangu_plasim_native.py  # PanguPlasimNative + PanguPlasimLegacyNative           [P6]
│   │   │   ├── layers.py               # EarthSpecific{Layer,Block,Attention3D}, patch embed/recovery,
│   │   │   │                           #   earth_position_index, up/down sample, mask, Integrator
│   │   │   └── vae.py                  # training-only dual-encoder + KL (used by PanguPlasim)
│   │   ├── sfno_plasim/                # NEW — phase 7 (reuse physicsnemo SFNO/makani where possible)
│   │   └── stochastic_interpolant/     # NEW — phase 8+ (DiT faithful + native)
│   ├── diffusion/
│   │   └── noise_schedulers/
│   │       └── interpolant.py          # NEW — DynamicInterpolant/Drift/DataDependent as NoiseScheduler+Solver
│   ├── datapipes/climate/
│   │   └── plasim_hdf5.py              # NEW — reads native per-timestep HDF5 + NetCDF stats, channel routing
│   └── metrics/climate/
│       └── ai_rossby.py                # NEW — anything missing (dayofyear-clim ACC aggregator, bias, QBO helpers)
├── examples/weather/ai_rossby/         # NEW — recipes (Hydra)
│   ├── conf/                           # model/data/training/validation config groups
│   ├── train.py                        # shared training loop (deterministic + diffusion modes)
│   ├── inference.py                    # autoregressive rollout (+ ensemble)
│   ├── validate.py                     # after-the-fact metrics + plots
│   └── README.md
├── tools/checkpoint_translation/       # NEW
│   ├── pangu_plasim.py                 # old .tar dict -> .mdlus (faithful variants)
│   └── amip_si.py                      # Lightning .ckpt -> .mdlus
├── test/models/pangu_plasim/           # NEW — forward/constructor/checkpoint/optim validators + synthetic data
├── skills/                             # NEW — Claude skills for dev/test/optimization (per outline)
└── hpc/                                # NEW — PBS (Derecho/Casper) + SLURM (Midway) job templates
```

## 3. Phased delivery

### Phase 1 — Pangu_Plasim faithful ports (BOTH architectures, weight-compatible) ← *substantively complete*
Port both PanguWeather Pangu networks into `physicsnemo/experimental/models/pangu_plasim/` as
`physicsnemo.Module`s, faithful flavor:
- **`PanguPlasim`** — faithful port of the current `pangu.py` model (with the training-only VAE dual-encoder + KL).
- **`PanguPlasimLegacy`** — faithful port of the predecessor `pangu_legacy.py` model (no VAE; forward returns
  the same six- (or seven- with diagnostics) tuple shape as `PanguPlasim` eval mode, with **all four** latent
  slots zero-tensor placeholders — matches the original source so downstream code targets one return shape).
  Ported *together* with `PanguPlasim` so they share the `layers.py` building blocks.

Per MOD-002a, both models live in `physicsnemo/experimental/models/` while iteration is ongoing. They are
re-exported via entry-points in `pyproject.toml` so `Module.from_checkpoint` and the smoke-test workflow
work as for production models. **Promotion** to `physicsnemo/models/pangu_plasim/` happens once (a) the
MOD-008b non-regression fixtures stabilize across ≥1 fork release cycle, and (b) the Phase-5 fidelity gate
validates the faithful flavor against a real PanguWeather checkpoint.

Shared Earth-Specific blocks / patch embed-recovery / up-down sample / mask / Integrator go in `layers.py`.
Refactor both constructors from the `params`/YParams blob to explicit **JSON-serializable kwargs** while keeping
internal math and **submodule names bit-identical** (so checkpoints map cleanly). Preserve the
`(surface, const_boundary, varying_boundary, upper_air, ...)` forward contract for both.

**Coding-standards compliance** (`CODING_STANDARDS/MODELS_IMPLEMENTATION.md`):
- **MOD-003** docstrings: `r"""` prefix, NumPy-style sections (`Parameters` / `Forward` / `Outputs` /
  `Notes` / `Examples`), tensor shapes in `:math:` LaTeX, double-backtick inline code,
  `name : type, optional, default=value` single-line param format.
- **MOD-005** shape validation at the top of `forward`, guarded by
  `if not torch.compiler.is_compiling():` and using the standardized
  ``"Expected ... got shape {actual_shape}"`` format.
- **MOD-006** `jaxtyping.Float[torch.Tensor, "..."]` annotations on `__init__` and public-method
  tensor arguments.
- **MOD-007** both models declare `__model_checkpoint_version__ = "1.0"`.
- **MOD-011** the original source's broken `USE_TE` opt-in (referenced `te.Linear` etc. without a guarded
  `transformer_engine` import) is **removed** from `layers.py`. Reintroduce cleanly behind
  `check_version_spec("transformer_engine", ...)` if/when FP8 is actually wanted.

**Tests** (per model):
- *Unit* (CPU, login-node-runnable):
  - **MOD-008a** `test_pangu_plasim_constructor` — 3-variant sweep (baseline / `upper_air_boundary` /
    `diagnostic_variables`).
  - **MOD-008b** `test_pangu_plasim_non_regression` — load committed reference
    `test/models/pangu_plasim/data/<ClassName>_v1.0.pth` (seeded by `init_seed=0`, `input_seed=42`,
    `forward_seed=123`) and compare forward output. Note: MOD-008b's example template overrides params
    with raw `randn` — that saturates this transformer to `NaN`, so we seed `torch.manual_seed` *before*
    the constructor (letting the default trunc-normal / Kaiming initializers run deterministically) and
    document the deviation.
  - **MOD-008c** `test_pangu_plasim_checkpoint` — `.mdlus` roundtrip via `Module.from_checkpoint`.
- *Smoke* (Delta `gpuA40x4-interactive`): per the smoke-test contract in `hpc/delta.md` — instantiate on
  CUDA, forward + backward + AdamW step on synthetic tiny tensors, `save_checkpoint`/`from_checkpoint`
  roundtrip.

### Phase 2 — `PlasimClimateDatapipe` ← *substantively complete*
At `physicsnemo/experimental/datapipes/plasim/` per MOD-002a. After comparative review (see
`pangu_plasim_reuse_plan.md` Phase 2 discussion) the underlying store format is **Zarr v3 via xarray**,
not the per-year HDF5 the original ERA5 pattern uses — Zarr supports the dual sigma + pressure level
systems PanguWeather configs need (`use_sigma_levels=True` w/ `Z` and `Z_2` coord systems coexisting),
irregular time axes for sparse `train_data_sets.json` date ranges without padding, and async chunk-granular
sharding for DDP. xarray + zarr + netCDF4 promoted to core deps in `pyproject.toml`.

Components:

- **Converter** (`tools/data/plasim/pangu_h5_to_zarr.py`): CLI reading the same PanguWeather v2.0 YAML
  config (e.g. `SFNO_PLASIM_H5_DERECHO_5412.yaml`) the user already has, walks the per-timestep
  `{year}_{idx:04d}.h5` archive, parses sigma / pressure level keys by numeric matching, writes a Zarr v3
  store. Channel-group bookkeeping, calendar, and timedelta go into the store `.attrs`.
- **Dataset** (`dataset.py`): `PlasimClimateDataset` — `torch.utils.data.Dataset` opening the Zarr lazily,
  concatenating sigma upper-air vars (first) and pressure upper-air vars (second) along the variable axis
  so `PanguPlasim.forward` consumes a single `(n_upper, n_levels, H, W)` tensor.
- **Sampler** (`samplers.py`): `LeadTimePairSampler` — multi-lead-time `(start, lead)` pair generator,
  shuffle-determined-by-`(seed, epoch)`, DDP-aware via rank/world_size positional slicing (matches
  `DistributedSampler`'s contract).
- **Transform** (`transforms.py`): `PlasimNormalizer` — loads PanguWeather's NetCDF per-variable mean/std,
  subsets pressure stats to the model's pressure levels, applies broadcast-shaped z-score to surface /
  upper-air / varying-boundary / target tensors. Constant boundaries and diagnostics pass through by
  default (toggleable). `.to(device)` moves buffers; composes as the dataset's `transform=` arg.

**Tests** (`test/datapipes/plasim/`):
- *Unit* (CPU, login-node-runnable): 13 cases covering layout-from-attrs, sample shapes / paired-target
  semantics / int-index shorthand / out-of-range, sampler determinism / DDP positional partitioning /
  validation, normalizer alignment / near-zero-mean+unit-std output / transform composition. All use the
  real PLASIM Zarr fixture at `$AI_ROSSBY_TEST_DATA/plasim/smoke_month.zarr` (30 days, PanguWeather sim52
  year 100, generated by the converter). Skip cleanly when the fixture is missing.
- *Smoke* (Delta `gpuA40x4-interactive`): 2 tests — real Zarr → iterate N batches on cuda:0 (shape /
  dtype / device / channel-routing contract per `hpc/delta.md`), and real Zarr → normalizer → PanguPlasim
  forward + backward + AdamW step on cuda:0 (the end-to-end pipeline gate). The model-forward smoke
  applies `torch.nan_to_num(0)` to constant_boundary + varying_boundary before the model — PLASIM's `lsm`
  carries NaN at the poles by convention; a proper `NanFillTransform` is a follow-up.

**Completed Phase 2 follow-ups** (commits `f0a7d412`, `578c3630`, etc.):
- ✅ `PlasimClimateDatapipe(Datapipe)` wrapper.
- ✅ Batched-async Zarr reads → 3× faster, now beats PanguH5 (commit `578c3630`).
- ✅ `predict_delta` mode in `PlasimNormalizer` (constructor `predict_delta=True` +
  `delta_std_path=...`). Tendency computation is `target = (raw_target − raw_state) / delta_std`
  (no mean subtraction per PanguWeather convention). Delta-std NetCDF generated via
  `tools/data/plasim/compute_delta_stats.py` (a small CLI walking the Zarr to compute per-variable
  per-level tendency std).
- ✅ `NanFillTransform` — composable CPU-side transform with per-variable fill dict
  (`{"sst": 273.15, ...}`) + `default=0.0` + `strict=False`. Default scope is
  `constant_boundary + varying_boundary`. Strict mode raises if any NaN survives the fill
  (sentinel for stats changes). `ComposeTransform` chains nan-fill → normalizer.
- ✅ Yearly-repeating boundary substitution: `boundary_zarr_path` (single-year), or
  `yearly_repeating_boundary=True` + `leap_boundary_zarr_path` + `non_leap_boundary_zarr_path`
  (PanguWeather convention; cycles via `cftime.is_leap_year(prog_year)` and day-of-year mapping).
  When all three boundary kwargs are unset, varying boundaries come from the prognostic Zarr at the
  same time index (the pre-Phase-2-follow-up behavior).

**Deferred (not yet implemented)**:
- Bias-correction `.npy` loader for the `bias_data_dir` files (separate per-variable / per-level
  annual + diurnal-cycle 2D fields). This is a distinct concept from boundary substitution —
  applied at training-time / inference-time to model outputs or to inputs as a residual correction.
  Lives more naturally in the training recipe (Phase 3) once the recipe defines exactly when
  bias correction runs (pre-loss vs post-output).

### Phase 2 follow-up: shared `ClimateZarrDataset` + unified data format ← *complete*

After the initial PLASIM datapipe shipped (`PlasimClimateDataset` / `PlasimClimateDatapipe`),
we generalized the loader and built per-dataset converters for ERA5 and E3SM under one schema.
The loader is metadata-driven (channel groups + level coords + calendar from store ``attrs``),
so a single class handles all three datasets.

- **Rename + alias** (`3ed6c25c`): `PlasimClimateDataset` → `ClimateZarrDataset`,
  `PlasimMultiYearDataset` → `ClimateZarrMultiYearDataset`,
  `PlasimStoreLayout` → `ClimateZarrStoreLayout`, in a new sub-package
  [`physicsnemo.experimental.datapipes.climate`](physicsnemo/experimental/datapipes/climate/).
  PLASIM-flavored names retained as backward-compat aliases.
- **Climatology schema v1.1** (`faf6eb1b`): every `{var}` climatology array carries a leading
  `stat` axis (`mean`, `std`). PLASIM sources have no separate std → NaN-filled.
- **Yearly-repeating boundary tests** (`bd285dc7`): cftime + numpy.datetime64 robustness for the
  three boundary modes (inline / single-year / yearly-repeating leap+non-leap).
- **cftime everywhere** (`39eb9dd2`, `d95eca63`): all xarray opens force
  `decode_times=CFDatetimeCoder(use_cftime=True)` so the time-coord semantics are uniform
  across PLASIM (pre-1582 year 1) and ERA5/E3SM (post-1582 dates). Loader bench shows < 1%
  impact on the hot path; see
  [`benchmarks/.../RESULTS.md`](benchmarks/physicsnemo/experimental/datapipes/plasim/RESULTS.md)
  cftime parity check.
- **ERA5 converters** (`ed5ae9d7`): per-year H5→Zarr, 5 normalization variants (pangu_s2s ±
  withnino / log_precip), climatology+std Zarr.
- **E3SM converters** (`041dbfec`, `481f79d2`, `4573e437`): per-year H5→Zarr (uppercase var
  names, hybrid pressure levels in hPa, noleap calendar), normalization, climatology+bias with
  soil-level (`levgrnd`) decomposition into per-depth flat 2D channels.
- **Full-archive SLURM scripts** (`71048006`): three sbatch jobs at
  [`hpc/scripts/convert_{plasim,era5,e3sm}_full_archive.sbatch`](hpc/scripts/) covering PLASIM
  (12–132), ERA5 (1979–2018), E3SM (2015–2049). All target
  `/work/hdd/bdiu/awikner/physicsnemo-zarr/{dataset}/`.

### Phase 3 — Training recipe (shared, deterministic mode) ← *v1 in progress*
Hydra config groups translated from the YParams YAML schema. Custom loop on `DistributedManager` +
`save/load_checkpoint` + `LaunchLogger` + `StaticCaptureTraining`, reproducing: AdamW(`fused`)/ZeRO-1,
OneCycle/cosine/warmup, bf16 AMP, EMA, loss combination + VAE-KL + `predict_delta`/`Integrator`. Modular
(pluggable model/loss/optimizer/scheduler) so the diffusion port reuses the loop.

**v1 (PanguPlasimLegacy, deterministic, no VAE-KL)** at
[`examples/weather/ai_rossby/`](examples/weather/ai_rossby/):

- [`loss.py`](examples/weather/ai_rossby/loss.py): `PanguPlasimLoss` — per-variable + cos(lat) weighted
  L1 / L2 residual on surface + upper-air + (optional) diagnostic; diagnostic head off for `LEGACY` config.
  Both MSE and MAE supported via `loss_type`.
- [`ema.py`](examples/weather/ai_rossby/ema.py): `ModelEMA` with PanguWeather decay=0.999, warmup
  ramp `(1+epoch)/(warmup_epochs+1)`.
- [`train_loop.py`](examples/weather/ai_rossby/train_loop.py): `make_optimizer`/`make_scheduler`/`train_step`
  factories. OneCycleLR (`oc_pct_start=0.1`, `oc_div_factor=1e5`, `oc_final_div_factor=0.00025` per
  `PANGU_PLASIM_H5_DERECHO_0514.yaml`) for `PanguPlasimLegacy`; LinearWarmup + CosineAnnealing reserved for
  the VAE variant.
- [`train.py`](examples/weather/ai_rossby/train.py): Hydra entrypoint composing
  `model` / `scheduler` / `loss` groups, wiring `DistributedManager` + DDP + `LaunchLogger` + `ModelEMA` +
  `save/load_checkpoint`. Drives a `PlasimClimateDatapipe` with `PlasimNormalizer` + `NanFillTransform`
  attached as the dataset's CPU-side transform.
- [`conf/`](examples/weather/ai_rossby/conf/): top-level `config.yaml` + `model/pangu_plasim_legacy.yaml`
  + `scheduler/{onecycle,cosine_warmup}.yaml` + `loss/{mae,mse}.yaml`.
- [`hpc/scripts/pangu_plasim_legacy_shake_out.sbatch`](hpc/scripts/pangu_plasim_legacy_shake_out.sbatch):
  SLURM script for the longer real-data shake-out on Delta `gpuA40x4` (non-interactive, 4× A40,
  `torchrun --standalone --nproc-per-node=4`).

**Tests** at [`test/recipes/ai_rossby/`](test/recipes/ai_rossby/):
- *Unit* (21 cases, CPU): `test_loss.py` (cos-lat weights, identity, per-var amplification, gradient flow,
  unknown-type rejection), `test_ema.py` (warmup clamp, post-warmup decay, apply/restore round-trip,
  apply-twice raises, state-dict round-trip), `test_train_loop.py` (AdamW factory, OneCycleLR + cosine
  composition + unknown rejections, end-to-end loss reduction on a toy model).
- *Smoke* (Delta `gpuA40x4-interactive`): `test_smoke_single_gpu.py` — real Zarr → datapipe → 2 train steps
  on a tiny PanguPlasimLegacy + checkpoint roundtrip on cuda:0. `test_smoke_ddp.py` —
  `torchrun --standalone --nproc-per-node=2` 2-GPU DDP variant that all-gathers params after one step to
  assert byte-identical sync.
- Longer shake-out: SLURM script above; not a smoke test.

**v2 — PanguPlasim (VAE) wired** (commits leading up to *Phase 3 v2*):

- ✅ `vae_kl_loss(mu_q, logvar_q, mu_p, logvar_p)` in
  [`examples/weather/ai_rossby/loss.py`](examples/weather/ai_rossby/loss.py)
  — faithful port of PanguWeather v2.0 `utils/losses.Kl_divergence_gaussians`.
- ✅ [`train_loop.train_step`](examples/weather/ai_rossby/train_loop.py) gains
  `vae_kl_weight` kwarg. Branches on whether the model returned real tensor latents
  (`torch.Tensor`) vs the int `0` placeholders the legacy model emits.
- ✅ [`train.py`](examples/weather/ai_rossby/train.py) `build_model` selects
  PanguPlasim vs PanguPlasimLegacy via `cfg.model.model_type`; the VAE-KL weight
  comes from `cfg.loss.vae_kl_weight`.
- ✅ Hydra: [`conf/model/pangu_plasim.yaml`](examples/weather/ai_rossby/conf/model/pangu_plasim.yaml)
  + [`conf/loss/{mae,mse}_with_kl.yaml`](examples/weather/ai_rossby/conf/loss/);
  `conf/scheduler/cosine_warmup.yaml` already in place from v1.
- ✅ Tests: 9 new unit cases (KL analytic properties, train_step VAE-on / VAE-off
  branches, sanity loss-reduction) + 1 new Delta GPU smoke
  ([`test_smoke_vae_single_gpu.py`](test/recipes/ai_rossby/test_smoke_vae_single_gpu.py))
  exercising the full VAE training path with `vae_kls_nonzero > 0` guard. All
  30 non-smoke recipe tests stay green.

**Deferred (Phase 3 v3)**:
- bf16 AMP via `StaticCaptureTraining` (currently disabled by default; `cfg.amp=True` no-op until wired).
- Fused AdamW / ZeRO-1 / gradient clip enable path (config keys present, factories pending).
- Long-validation rollout + bias correction (Phase 4 territory; rolls into the recipe via the validation
  hooks already stubbed in `train.py`).

### Phase 4 — Validation (mid-training + after-the-fact)
Mid-training: autoregressive rollout → lat-weighted RMSE (mse+reductions) + ACC (`physicsnemo.metrics.climate.acc`
with dayofyear climatology), long-rollout bias, ensemble validation, power spectra. After-the-fact:
`inference.py` (shared stepper, IC-perturbation + MC-dropout ensembles) → NetCDF; `validate.py` →
RMSE/ACC/spectra/bias/CRPS + plots. Port the DDP all-reduce `MetricsAggregator` behavior.
**Tests**:
- *Unit*: each new metric (dayofyear-clim ACC aggregator, bias, CRPS, QBO) against analytic/reference values
  on synthetic tensors.
- *Smoke* (Delta): same metrics run on real CUDA tensors; for the aggregator, a 2-GPU DDP smoke verifies the
  all-reduce produces the single-GPU value.

### Phase 5 — Checkpoint translation + numerical fidelity
`tools/checkpoint_translation/pangu_plasim.py`: normalize `module.` prefix, prefer `ema_state`, remap keys →
faithful module state_dict (handles both `PanguPlasim` and `PanguPlasimLegacy`), reconstruct `Integrator`
buffers from norm `.nc`, emit `.mdlus`.
**Tests**:
- *Unit*: small synthetic source-format ckpt → translator → load into faithful model → forward shape OK.
- *Smoke* (Delta interactive): same as unit but on CUDA; tolerance only on shape/dtype.
- **Fidelity gate** (Delta non-interactive `gpuA40x4`): run the *original* PanguWeather inference to capture
  reference outputs, load the translated checkpoint into the faithful model, assert rollout outputs match
  within tolerance. This is the one Phase 5 task that doesn't fit the interactive queue's 1-hour cap.

### Phase 6 — Pangu_Plasim native variants
Rebuild both architectures on native PhysicsNeMo blocks (`PanguPlasimNative`, `PanguPlasimLegacyNative`),
reusing the shipped Pangu's Earth-Specific attention/patch ops where compatible; StaticCapture-friendly, richer
`ModelMetaData`. Same I/O contract, own configs; trained via the Phase-3 recipe. (New training runs; not
checkpoint-compatible with the faithful variants — that's the faithful variants' role.)
**Tests**: same unit + smoke contract as Phase 1, both variants.

### Phase 7 — SFNO (rest of v2.0) — *in scope*
Map the vendored SFNO to PhysicsNeMo's SFNO (makani plugin) or port the vendored copy as `sfno_plasim`
(faithful + native). (Legacy Pangu moved into Phase 1; `pangu_lite` deferred unless needed.)
**Tests**: same unit + smoke contract as Phase 1, applied to the SFNO variants.

### Phase 8+ — amip stochastic-interpolant model (second major effort)
- `interpolant.py` NoiseScheduler + Solver reproducing `DynamicInterpolant`/`DriftScheduler`/`DataDependentInterpolant`
  (+ `ERDMScheduler` later); spherical-harmonic noise; per-channel `noise_scales` buffer.
- DiT model (faithful w/ `old/` deps carried over + native on `physicsnemo.models.dit`); fixed bilinear-×4
  latent as deterministic pre/post-processing; three-way conditioning (state/`c_grid`/`c_scalar` w/ calendar+CO₂).
- Muon optimizer + EMA into the shared recipe; rollout/ensemble inference (RNG-checkpointed resume; S2S hindcasts);
  evals (climatology/bias, QBO, global-mean timeseries, ensemble envelopes); Lightning-`.ckpt` translation script.
**Tests**:
- *Unit*: solver step + 5-step rollout on synthetic state, finite output, shapes preserved.
- *Smoke* (Delta): same on CUDA; one DiT forward + backward + AdamW step; checkpoint roundtrip.
  Datapipe smoke for amip's reader (real fixture, per `hpc/delta.md`). Lightning-ckpt translator smoke.

### Cross-cutting (throughout)
- **Claude skills** (`skills/`) for model dev, test scaffolding, and optimization (per outline objective).
  Two skills are wired up for the smoke-test workflow: `delta-smoke-test` (submit a pytest target to
  `gpuA40x4-interactive`) and `delta-shell` (interactive A40 srun).
- **HPC docs and job templates** (`hpc/`): `install.md` (portable uv + system-PyTorch recipe), `delta.md`
  (NCSA Delta specifics — partition, account, env, smoke-test patterns); PBS templates for Derecho/Casper and
  SLURM templates for Midway/Delta non-interactive added as those clusters come online.
- **Robust unit + smoke tests** as the base-class contract for new models/datapipes/metrics (outline objective).
  Smoke-test contract is normative — see `hpc/delta.md`.
- **Config system**: Hydra groups mirroring the model/data/training/validation separation the outline asks for.

### Smoke tests on Delta interactive queue (normative)

Every newly added feature — model, datapipe, metric, training-recipe component, checkpoint translator,
interpolant solver — ships **both** a CPU-runnable unit test **and** a GPU smoke test on Delta's
`gpuA40x4-interactive` partition. Full contract: `hpc/delta.md`. Highlights:

- Smoke tests are marked `@pytest.mark.smoke` and `@pytest.mark.cuda`, live alongside the unit tests in
  `test/`. `pytest -m "smoke and cuda"` selects them on a GPU node.
- Run on **1 node, ≤ 4 A40 GPUs, ≤ 5 min wall** (interactive queue cap is 1 hr).
- Synthetic tiny tensors **except** for data-loading code, which must read a real fixture from
  `$AI_ROSSBY_TEST_DATA` (gitignored scratch path, symlinked at `test/_data` for IDE convenience).
- DDP smoke tests use exactly 2 GPUs.
- Anything that can't fit (Phase 5 fidelity gate, Phase 3 real-data shake-out) goes to `gpuA40x4`
  non-interactive with its own job script under `hpc/scripts/`.

## 4. Reuse map (build off what exists)

| Need | Reuse from PhysicsNeMo | New work |
|---|---|---|
| Model base/checkpoint/registry | `physicsnemo.Module`, `.mdlus`, entry points | refactor constructors to JSON-serializable args |
| Earth-Specific attention/patch ops | shipped `Pangu` building blocks | dual-stream + boundary + VAE + delta wrapping |
| Training infra | `DistributedManager`, `save/load_checkpoint`, `LaunchLogger`, `StaticCaptureTraining` | the loop, EMA, loss combo, VAE-KL, Muon |
| Diffusion core | protocol API, `sample()`, samplers, `MSEDSMLoss`, DiT/SongUNet | interpolant scheduler + solver (net-new) |
| Data | `Datapipe` base / Reader-Transform arch | native HDF5+NetCDF reader, channel routing |
| Metrics | `acc`, lat-weight reductions, `crps`, `power_spectrum` | dayofyear-clim ACC aggregator, bias, QBO |
| Tests | `test/common` validators | per-model/datapipe test files + fixtures |
| HPC | `DistributedManager` SLURM/MPI detection | PBS job templates |

## 5. Key risks & mitigations
- **Constructor refactor breaks weight mapping.** Keep submodule names/order identical in the faithful variants;
  gate translation behind the Phase-5 fidelity test against original outputs.
- **Interpolant has no framework analog.** Follow the TopoDiff "custom scheduler/solver in user code" pattern;
  validate the sampler against amip outputs before wiring into training.
- **Data-format mismatch.** Custom datapipe reading native files (no mass migration); validate against the
  original loader's tensors on identical files.
- **PBS clusters.** No native detection — provide tested PBS templates that set torchrun/MPI env vars.
- **VAE/KL & Muon fidelity.** Port both faithfully in faithful/training paths; make them optional/pluggable in
  the native variants.

## 6. Resolved & remaining items

Resolved (was §6 in earlier drafts):
1. **HPC specifics** — Delta first. Smoke-test partition `gpuA40x4-interactive`, account `bdiu-delta-gpu`,
   env via `pytorch-conda/2.8` + uv `--system-site-packages`. Repo at `/work/nvme/bdiu/awikner/physicsnemo`,
   test data under `$AI_ROSSBY_TEST_DATA` = `/work/nvme/bdiu/awikner/physicsnemo_test_data`
   (symlinked at `test/_data`, gitignored). Full recipe: `hpc/delta.md`,
   portable template: `hpc/install.md`.
2. **SFNO scope** — *in scope* (Phase 7), faithful + native variants, same unit + smoke contract as Pangu_Plasim.

Remaining:
- **Phase 5 fidelity-gate job script** — non-interactive `gpuA40x4` submission script + tolerance choices.
  Drafted when the translator lands.
- **Per-cluster docs as we extend beyond Delta** — Derecho/Casper (PBS), Midway (SLURM) need their own
  `hpc/<cluster>.md` mirroring `hpc/delta.md`.
