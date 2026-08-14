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
# Autoregressive rollout to a file
python inference.py model=sfno_e3sm dataset=e3sm \
    +inference.checkpoint_dir=/path/to/checkpoints \
    +inference.output_path=/path/to/preds.nc +inference.max_step=60

# Score predictions against the reference Zarr (RMSE/ACC scorecard)
python validate_cli.py dataset=e3sm \
    +validation_cli.predictions=/path/to/preds.nc \
    +validation_cli.reference_zarr=$AI_ROSSBY_DATA/e3sm/2045.zarr \
    +validation_cli.output_md=/path/to/scores.md
```

## 6. AMIP diffusion (experimental)

The diffusion recipes (`train_diffusion.py`, `validate_diffusion.py`,
`eval_diffusion.py`, `model=amip_*`) are **experimental** — derived from the
`amip` codebase (see [`NOTICE`](../../../NOTICE)) and not part of the supported
path. Use the deterministic Pangu/SFNO recipes above unless you specifically
need diffusion.

### Two channel contracts, on purpose

The AMIP port spans two upstream generations, and **which one a config belongs to
is not cosmetic** — the contracts differ in channel order, so loading a
checkpoint against the wrong one runs happily and produces nonsense. Every
wrapper takes `channel_layout`, and the translator sets it from
`--source-contract`.

| `channel_layout` | Meaning | Configs |
|---|---|---|
| `v2` | upstream **amip_v2**: state `[surface \| diag \| upper_air]`, upper-air level-major (1000 hPa first), c_grid `[varying \| constant]` | `amip_erdm_v2`, `amip_erdm_v2_ocean`, `amip_erdm_fancy`, `amip_x_ddc_dit` |
| `v1` | upstream **amip v1**: same group order, upper-air variable-major in config level order | translated v1 checkpoints (`--source-contract v1`) |
| `fork` | the Phase-8 fork order (`[surface \| upper_air \| diag]`, c_grid `[constant \| varying]`) — **no upstream checkpoint matches it** | pinned by the frozen `amip_rfm` / `amip_erdm` |

The v1 families (SI, SI_X, EDM, RFM, ERDM-UNet) are **frozen**: kept working, not
migrated, because amip_v2 deleted them. See
[`docs/dev/phase12_implementation_plan.md`](../../../docs/dev/phase12_implementation_plan.md)
for the seam.

### Supported config set

| Config | What it is |
|---|---|
| `amip_erdm_v2` | the v2 ERDM forecaster (full feature set, no predicted ocean) |
| `amip_erdm_v2_ocean` | + 12f predicted ocean channels (SST, sea ice) |
| `amip_erdm_fancy` | + 12g SST anomaly channel and `global_mean_sst` scalar; matches the real `ERDM_fancy` checkpoint |
| `amip_x_ddc_dit` | the v2 downscaler (`DiTAE`) — **the only denoiser amip_v2 kept** |
| `amip_x_ddc` | the v1 convolutional downscaler (`XDDCUNet`), frozen |
| `amip_si`, `amip_si_x`, `amip_erdm`, `amip_rfm` | frozen v1 families |
| `amip_combined` | forecaster + downscaler composition from two checkpoints |

Two things worth knowing before running any of them:

- **The model's timestep lives in the model config** (`timedelta_hours`), not the
  dataset's, because the same store feeds families with different steps. All AMIP
  configs are 24 h over 6-hourly rows, i.e. 4 store rows per step.
- **Translating a v2 checkpoint**: use the auto-derive path (no `--model-config`)
  and then verify numerically with
  `tools/checkpoint_translation/verify_v2_numerical.py` — key parity cannot catch
  a channel permutation. See
  [`docs/dev/phase8e_midway3_checkpoint_inventory.md`](../../../docs/dev/phase8e_midway3_checkpoint_inventory.md#amip_v2-checkpoints-2026-08-14).

## Files

- `train.py` / `train_loop.py` — training entrypoint + step logic
- `inference.py` — autoregressive rollout to NetCDF
- `validate.py` / `validate_cli.py` — mid-training + after-the-fact scoring
- `loss.py`, `ema.py`, `climatology.py`, `data_staging.py` — supporting pieces
- `conf/` — the Hydra config groups above
