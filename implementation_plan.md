# AI-Rossby Implementation Plan — Porting PanguWeather v2.0 & amip into PhysicsNeMo

Status: **in progress (Phase 1)** · Author: Claude (analysis + port) · Updated: 2026-06-16

This plan covers the first objective in [`project_outline.md`](project_outline.md): porting all current
models, training/inference code, and mid-training & after-the-fact validation from **PanguWeather/v2.0** and
**amip** into the **PhysicsNeMo** framework. It is grounded in a deep read of all three repositories
(findings summarized in §1).

## Decisions locked in (from review)

1. **Repo strategy — Fork & in-tree.** `ai-rossby` work happens in the user's PhysicsNeMo fork
   (`awikner/physicsnemo`, local at `/Users/Alexander/Documents/UChicago/physicsnemo`, branch `ai-rossby`).
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

### Phase 1 — Pangu_Plasim faithful ports (BOTH architectures, weight-compatible) ← in progress
Port both PanguWeather Pangu networks into `physicsnemo/models/pangu_plasim/` as `physicsnemo.Module`s,
faithful flavor:
- **`PanguPlasim`** — faithful port of the current `pangu.py` model (with the training-only VAE dual-encoder + KL).
- **`PanguPlasimLegacy`** — faithful port of the predecessor `pangu_legacy.py` model (no VAE; forward returns
  4–5 values, `mu2`/`sigma2` = None; VAE loss skipped). Ported *together* with `PanguPlasim` so they share
  the `layers.py` building blocks.

Shared Earth-Specific blocks / patch embed-recovery / up-down sample / mask / Integrator go in `layers.py`.
Refactor both constructors from the `params`/YParams blob to explicit **JSON-serializable kwargs** while keeping
internal math and **submodule names bit-identical** (so checkpoints map cleanly). Preserve the
`(surface, const_boundary, varying_boundary, upper_air, ...)` forward contract for both.
**Tests** (per model): `validate_forward_accuracy` (commit reference), `_constructor`, `validate_checkpoint`
roundtrip, `_optims`; conservative `ModelMetaData` flags (amp/bf16), expanded as validated.

### Phase 2 — `PlasimClimateDatapipe`
Custom datapipe reading the native per-timestep HDF5 + NetCDF z-score stats, reproducing channel routing
(surface/upper-air/boundary/diagnostic + TOA-solar special case), `predict_delta`, and lead-time/multi-year
sampling with `DistributedSampler`. Normalization as composable transforms reading the existing `.nc` stats.
Tests: synthetic small HDF5 + stats fixtures; shape/device/channel/grid checks via `test/datapipes/common`.

### Phase 3 — Training recipe (shared, deterministic mode)
Hydra config groups translated from the YParams YAML schema. Custom loop on `DistributedManager` +
`save/load_checkpoint` + `LaunchLogger` + `StaticCaptureTraining`, reproducing: AdamW(`fused`)/ZeRO-1,
OneCycle/cosine/warmup, bf16 AMP, EMA, loss combination + VAE-KL + `predict_delta`/`Integrator`. Modular
(pluggable model/loss/optimizer/scheduler) so the diffusion port reuses the loop. Validate: single-step smoke
on synthetic locally; short real run on HPC.

### Phase 4 — Validation (mid-training + after-the-fact)
Mid-training: autoregressive rollout → lat-weighted RMSE (mse+reductions) + ACC (`physicsnemo.metrics.climate.acc`
with dayofyear climatology), long-rollout bias, ensemble validation, power spectra. After-the-fact:
`inference.py` (shared stepper, IC-perturbation + MC-dropout ensembles) → NetCDF; `validate.py` →
RMSE/ACC/spectra/bias/CRPS + plots. Port the DDP all-reduce `MetricsAggregator` behavior.

### Phase 5 — Checkpoint translation + numerical fidelity
`tools/checkpoint_translation/pangu_plasim.py`: normalize `module.` prefix, prefer `ema_state`, remap keys →
faithful module state_dict (handles both `PanguPlasim` and `PanguPlasimLegacy`), reconstruct `Integrator`
buffers from norm `.nc`, emit `.mdlus`. **Fidelity gate (HPC)**: run the *original* PanguWeather inference to
capture reference outputs, load the translated checkpoint into the faithful model, assert rollout outputs match
within tolerance.

### Phase 6 — Pangu_Plasim native variants
Rebuild both architectures on native PhysicsNeMo blocks (`PanguPlasimNative`, `PanguPlasimLegacyNative`),
reusing the shipped Pangu's Earth-Specific attention/patch ops where compatible; StaticCapture-friendly, richer
`ModelMetaData`. Same I/O contract, own configs; trained via the Phase-3 recipe. (New training runs; not
checkpoint-compatible with the faithful variants — that's the faithful variants' role.)

### Phase 7 — SFNO (rest of v2.0)
Map the vendored SFNO to PhysicsNeMo's SFNO (makani plugin) or port the vendored copy as `sfno_plasim`
(faithful + native). (Legacy Pangu moved into Phase 1; `pangu_lite` deferred unless needed.)

### Phase 8+ — amip stochastic-interpolant model (second major effort)
- `interpolant.py` NoiseScheduler + Solver reproducing `DynamicInterpolant`/`DriftScheduler`/`DataDependentInterpolant`
  (+ `ERDMScheduler` later); spherical-harmonic noise; per-channel `noise_scales` buffer.
- DiT model (faithful w/ `old/` deps carried over + native on `physicsnemo.models.dit`); fixed bilinear-×4
  latent as deterministic pre/post-processing; three-way conditioning (state/`c_grid`/`c_scalar` w/ calendar+CO₂).
- Muon optimizer + EMA into the shared recipe; rollout/ensemble inference (RNG-checkpointed resume; S2S hindcasts);
  evals (climatology/bias, QBO, global-mean timeseries, ensemble envelopes); Lightning-`.ckpt` translation script.

### Cross-cutting (throughout)
- **Claude skills** (`skills/`) for model dev, test scaffolding, and optimization (per outline objective).
- **HPC job templates** (`hpc/`) for PBS (Derecho/Casper) + SLURM (Midway); document `DistributedManager` env.
- **Robust unit + smoke tests** as the base-class contract for new models/datapipes/metrics (outline objective).
- **Config system**: Hydra groups mirroring the model/data/training/validation separation the outline asks for.

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

## 6. Open items to confirm
1. **HPC specifics** (gates Phases 3 & 5): which cluster first, data/checkpoint paths there, module/conda env,
   account/queue for the smoke run and the fidelity check.
2. **Scope** (gates Phase 7): is SFNO in scope now, or deferred until Pangu_Plasim + the stochastic-interpolant
   model both land? (Legacy Pangu already folded into Phase 1.)
