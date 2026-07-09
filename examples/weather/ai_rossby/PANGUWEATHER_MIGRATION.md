# Migrating from PanguWeather to the ai-rossby recipe

This guide explains how to bring a model, its configuration, its data, and
its trained weights over from a **PanguWeather-style training repo** (the
`PanguWeather-e3sm` / `PanguWeather-plasim` v2.0 forks, which are themselves
derived from NVIDIA's `makani` / FourCastNet training stack) into the
**ai-rossby** recipe in this repository.

It is written primarily around the **SFNO-E3SM** model, but the moving parts
(config translation, data conversion, checkpoint conversion, launch) are the
same for every model family the recipe supports (`PanguPlasim`,
`PanguPlasimLegacy`, `SfnoPlasim`, the AMIP diffusion models, …), so the
generic path is called out at each step.

---

## 0. Mental model: what changed and what didn't

| Concern | PanguWeather | ai-rossby (this repo) |
|---|---|---|
| **Config format** | One flat YAML with named blocks; a block is selected at launch with `--config=SFNO` | [Hydra](https://hydra.cc) *compositional groups* — `model=`, `dataset=`, `training=`, `loss=`, `validation=` each pick one YAML from `conf/<group>/` |
| **Launch** | `python train.py --yaml_config=<file> --config=SFNO --epochs=N …` | `python train.py model=sfno_e3sm dataset=e3sm training=sfno_plasim …` (or `torchrun` for multi-GPU) |
| **Data on disk** | Per-timestep / per-year **HDF5** files | A single **Zarr** store per split (converted once with the tools in `tools/data/`) |
| **Checkpoint** | `torch.save(dict, "ckpt.tar")` blob with `model_state` / `ema_state` | PhysicsNeMo `.mdlus` file (`Module.save`) — architecture + weights in one file |
| **Model code** | Vendored SFNO / Pangu blocks | `physicsnemo` library components (see `pangu_plasim_reuse_plan.md`) |
| **Logging sink** | MLflow | **wandb** (default, `offline` mode) + console |

The *numerical contract* (variable groups, level count, normalization,
architecture hyperparameters, loss) is preserved. Migration is mostly a
matter of re-expressing the same numbers in the new locations.

---

## 1. Translating configuration

### 1.1 Where each PanguWeather config block lands

A PanguWeather flat YAML (e.g. `E3SM_SFNO_H5_STAMPEDE_jsw_256.yaml`, config
block `SFNO`) is split across five Hydra groups:

```
PanguWeather  E3SM_SFNO_H5_STAMPEDE_jsw_256.yaml  (--config=SFNO)
│
├── SFNO architecture block ........... → conf/model/sfno_e3sm.yaml
├── channel / variable / level lists ... → conf/model/sfno_e3sm.yaml (variable groups)
├── data paths + loader knobs .......... → conf/dataset/e3sm.yaml
├── optimizer / scheduler / epochs / EMA → conf/training/sfno_plasim.yaml
├── loss (l1 / l2) ..................... → conf/loss/mae.yaml (l1) or raw_l2.yaml (l2)
└── validation forecast_lead_times ..... → conf/validation/rollout_5412.yaml
```

At launch you compose them:

```bash
python train.py \
    model=sfno_e3sm \
    dataset=e3sm \
    training=sfno_plasim \
    loss=mae \
    validation=rollout_5412
```

Anything you want to tweak without editing a file, override on the CLI with a
dotted path, e.g. `training.max_epochs=100 dataset.batch_size=8`.

### 1.2 SFNO architecture keys (1:1 mapping)

The SFNO block maps key-for-key into `conf/model/sfno_e3sm.yaml`. The names
are intentionally identical to the makani/PanguWeather SFNO block:

| PanguWeather SFNO key | ai-rossby `conf/model/*.yaml` key | Notes |
|---|---|---|
| `nettype: SFNO` | `name: SfnoPlasim`, `module: physicsnemo.experimental.models.sfno_plasim` | class + import path for `Module.instantiate` |
| `embed_dim` | `embed_dim` | `256` for the `jsw_256` variant |
| `num_layers` | `num_layers` | |
| `num_blocks` | `num_blocks` | |
| `scale_factor` | `scale_factor` | |
| `filter_type` | `filter_type` | `linear` |
| `operator_type` | `operator_type` | `dhconv` |
| `spectral_transform` | `spectral_transform` | `sht` |
| `hard_thresholding_fraction` | `hard_thresholding_fraction` | |
| `normalization_layer` | `normalization_layer` | `instance_norm` |
| `activation_function` | `activation_function` | `gelu` |
| `mlp_ratio`, `use_mlp` | `mlp_ratio`, `use_mlp` | |
| `encoder_layers` | `encoder_layers` | |
| `pos_embed` | `pos_embed` | |
| `big_skip` | `big_skip` | |
| `spectral_layers` | `spectral_layers` | |
| `complex_network`, `complex_activation`, `use_complex_kernels` | same | |
| `factorization`, `rank`, `separable` | same | |
| `sparsity_threshold`, `drop_rate`, `drop_path_rate` | same | |
| `checkpointing` | `checkpointing` | activation checkpointing depth |
| `img_shape` / `(nlat, nlon)` | `horizontal_resolution: [lat, lon]` | E3SM: `[180, 360]` |
| `data_grid` | `data_grid` | `equiangular` |

> **General rule:** the model YAML *is* the model constructor's kwargs. Every
> key except the four identity keys (`name`, `module`, `target`,
> `model_type`) is forwarded verbatim to the model class `__init__`
> (`build_model()` in `train.py`). To see the full accepted set for a family,
> read the class docstring — e.g. `SfnoPlasim` in
> `physicsnemo/experimental/models/sfno_plasim/sfno_plasim.py`.

### 1.3 Variable groups and levels

PanguWeather encodes channels as flat index ranges; ai-rossby names them
explicitly and routes them into the model's four input groups + one output
group. For SFNO-E3SM (`conf/model/sfno_e3sm.yaml`):

```yaml
surface_variables:          [TREFHT, U10, PSL]            # prognostic 2D
upper_air_variables:        [T, U, V, RELHUM, Z3]         # prognostic 3D (× levels)
constant_boundary_variables:[TOPO, PCT_GLACIER, PCT_NATVEG, PFTDATA_MASK]  # static
varying_boundary_variables: [SST, ICE, sol_in]            # time-varying forcing
diagnostic_variables:       [PRECT]                        # output-only diagnostic
levels: [ ... 18 hybrid-pressure levels in hPa ... ]
horizontal_resolution: [180, 360]
```

Channel-count bookkeeping is derived automatically:
`in_chans = n_surface + n_const + n_varying + n_upper × n_levels`, and
`out_chans = n_surface + n_diag + n_upper × n_levels`.

**Ordering matters and must match the Zarr store.** The variable-group lists
must be in the same order the data converter wrote them (see
`tools/data/e3sm/pangu_h5_to_zarr.py`, `PANGU_E3SM_CHANNELS`) and the same
order the loss expects. Upper-air variables are assumed to be in
sigma-then-pressure order to match `PlasimClimateDataset`.

> **E3SM caveat already baked in:** the reference config's land variables
> (`SOILWATER_10CM`, `TSOI_10CM`) are *not* written by the per-year converter
> (they only exist in the climatology store), so they are intentionally
> absent from `sfno_e3sm.yaml`. If you regenerate the Zarr with those
> channels, add them here in the same position.

### 1.4 Data, optimizer, loss, validation blocks

| PanguWeather key | ai-rossby location | Group file (SFNO-E3SM) |
|---|---|---|
| `train_data_path` | `dataset.zarr_path` | `conf/dataset/e3sm.yaml` |
| `valid_data_path` | `dataset.val_zarr_path` | |
| `global_means_path` / `global_stds_path` | `dataset.mean_path` / `dataset.std_path` | (Zarr with mean+std) |
| `batch_size` | `dataset.batch_size` | |
| `num_data_workers` | `dataset.num_workers` | |
| `dt` / `forecast_lead_times` (train) | `dataset.forecast_lead_times` | |
| mask fill value | `dataset.nan_fill_values` / `nan_fill_default` | e.g. `{SST: 270.0}` |
| normalize boundary/diagnostic | `dataset.normalize_constant_boundary`, `dataset.normalize_diagnostic` | `True` for E3SM |
| `optimizer_type` | `training.optimizer.type` | `conf/training/sfno_plasim.yaml` |
| `lr`, `weight_decay` | `training.optimizer.lr`, `.weight_decay` | |
| `scheduler` + warmup | `training.stages[*].scheduler` | `LinearWarmupCosineAnnealingLR`, `num_warmup_epochs`, `eta_min`, … |
| `max_epochs` | `training.max_epochs` | |
| `ema`, `ema_decay` | `training.ema.enabled`, `.decay`, `.warmup_epochs` | |
| `enable_amp` / `--no_amp` | `training.amp` (`none` / `bf16` / `fp16`) | |
| `loss: l1` / `l2` | `loss=mae` / `loss=raw_l2` | `conf/loss/*.yaml` |
| per-variable loss weights | `loss.surface_var_weights`, `loss.upper_air_var_weights`, … | |
| `valid rollout forecast_lead_times` | `validation.rollout.log_steps` | `conf/validation/rollout_5412.yaml` |

> **Multi-stage curriculum.** PanguWeather typically fine-tunes autoregressive
> rollout in a second run. ai-rossby expresses this as a list of `stages` in
> one training config (see `conf/training/sfno_plasim_curriculum.yaml`): each
> stage carries its own `num_epochs`, `unroll_steps`, optional `batch_size`,
> and a fresh scheduler. The global epoch counter runs continuously across
> stages so checkpoint resume and EMA warmup keep working.

---

## 2. Converting the data (HDF5 → Zarr)

ai-rossby reads a Zarr store, not PanguWeather's per-timestep HDF5. Convert
once per split with the matching tool under `tools/data/`:

```bash
# One year of E3SM (1460 6-hourly steps) → one Zarr store
python tools/data/e3sm/pangu_h5_to_zarr.py \
    --year 2041 --sample-range 0 1460 \
    --output /path/to/zarr/e3sm/2041.zarr

# repeat for the validation year, e.g. 2045
```

For other sources use the sibling converters: `tools/data/era5/pangu_h5_to_zarr.py`,
`tools/data/plasim/pangu_h5_to_zarr.py`. The channel groups written by the
converter (`PANGU_E3SM_CHANNELS` etc.) **define** the variable ordering your
model config must match — treat the converter as the source of truth.

Normalization statistics (`mean_path` / `std_path`) point at a small Zarr of
per-channel mean/std. If you already have PanguWeather's `global_means.npy` /
`global_stds.npy`, either regenerate the stats store or point the config at a
converted equivalent; the `PlasimNormalizer.from_dataset(...)` call in
`build_datapipe()` is what consumes them.

For large conversions on an HPC cluster, the repo ships CPU-job skills
(`delta-cpu-job`, `derecho-cpu-job`, `midway3-cpu-job`, `stampede3-cpu-job`).

> **Where the E3SM Zarr already lives on Delta.** The converted stores are
> checked into `conf/dataset/e3sm.yaml`, so you don't need to re-convert on
> Delta — just point the run at them (they are the default paths):
>
> | Purpose | Path on Delta |
> |---|---|
> | Training split (year 2041) | `/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/2041.zarr` |
> | Validation split (year 2045) | `/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/2045.zarr` |
> | Normalization stats (mean+std, 2015–2050) | `/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/normalization_2015-2050.zarr` |
>
> These sit on Delta's `/work/hdd` (the `bdiu` project's hardened-disk
> allocation). On other clusters, convert your own copy and override the four
> `dataset.*_path` keys as in §4.1.

---

## 3. Converting a trained checkpoint (`.tar`/`.pt` → `.mdlus`)

PanguWeather saves a `torch.save(dict, "ckpt.tar")` blob (the `.tar`
extension is a convention — it is *not* a tarball) containing `model_state`,
`ema_state`, and optimizer/scheduler state. The translators in
`tools/checkpoint_translation/` load that blob, remap the state-dict keys to
the ai-rossby wrapper layout, instantiate a fresh model from your model YAML,
load the weights, verify there are no missing/unexpected keys, and write a
PhysicsNeMo `.mdlus`.

**SFNO:**

```bash
python tools/checkpoint_translation/sfno_plasim.py \
    --source   /path/to/panguweather_sfno_ckpt.tar \
    --model-config examples/weather/ai_rossby/conf/model/sfno_e3sm.yaml \
    --output   /path/to/sfno_e3sm.mdlus \
    --strict
```

**Pangu family** (analogous script, no `sfno.` re-prefix):

```bash
python tools/checkpoint_translation/pangu_plasim.py \
    --source   /path/to/panguweather_pangu_ckpt.tar \
    --model-config examples/weather/ai_rossby/conf/model/pangu_plasim_legacy.yaml \
    --output   /path/to/pangu_plasim.mdlus \
    --strict
```

Key points:

- **EMA is preferred by default.** When `ema_state` is present it is used
  (PanguWeather's documented inference-time preference). Pass
  `--prefer-model-state` to take raw `model_state` instead.
- **Wrapper prefix.** `SfnoPlasim` holds the base SFNO under `self.sfno`, so
  every source key is re-prefixed with `sfno.`. The Pangu wrappers keep the
  base keys, so no re-prefix is applied. DDP (`module.`) and
  `torch.compile` (`_orig_mod.`) prefixes are stripped iteratively either way.
- **`--strict` is your friend.** It refuses to write the `.mdlus` if any key
  is missing or unexpected — the fastest way to catch a config that doesn't
  match the checkpoint (wrong `embed_dim`, wrong channel counts, etc.). Run
  with `--verbose` to see the first few mismatched keys.
- **The `--model-config` must describe the *same architecture* the checkpoint
  was trained with.** This is the single most common failure: a key mismatch
  almost always means the model YAML disagrees with the source run's SFNO
  block. Re-check §1.2 / §1.3.

To convert a checkpoint for a **new model family**, follow the same pattern:
copy one of these scripts and adjust `translate_state_dict()` for that
wrapper's submodule prefix.

### Resuming training vs. converting for inference

The translators produce a weights-only `.mdlus` for inference or as a warm
start. ai-rossby's own training checkpoints (written by `save_checkpoint` to
`./checkpoints`) additionally carry optimizer state and EMA; `train.py`
auto-resumes from `./checkpoints` on restart (`load_checkpoint`). A translated
`.mdlus` is not an optimizer-state resume — it seeds the weights only.

---

## 4. Running the training script

The entrypoint is `examples/weather/ai_rossby/train.py` (Hydra app,
`conf/config.yaml` is the root). Hydra changes into `outputs/<run_name>/` at
launch, so checkpoints land in `outputs/<run_name>/checkpoints/`.

### 4.1 Single GPU

```bash
cd examples/weather/ai_rossby
python train.py \
    model=sfno_e3sm \
    dataset=e3sm \
    training=sfno_plasim \
    loss=mae \
    validation=rollout_5412 \
    run_name=sfno_e3sm_run0
```

Point the data at your own paths if they differ from `conf/dataset/e3sm.yaml`:

```bash
python train.py model=sfno_e3sm dataset=e3sm training=sfno_plasim \
    dataset.zarr_path=/my/e3sm/2041.zarr \
    dataset.val_zarr_path=/my/e3sm/2045.zarr \
    dataset.mean_path=/my/e3sm/norm.zarr \
    dataset.std_path=/my/e3sm/norm.zarr
```

### 4.2 Multi-GPU (single node)

```bash
cd examples/weather/ai_rossby
torchrun --standalone --nproc-per-node=4 train.py \
    model=sfno_e3sm dataset=e3sm training=sfno_plasim \
    validation=rollout_5412 run_name=sfno_e3sm_ddp
```

DDP is set up automatically when `world_size > 1`. Optional throughput knobs
(off by default, safe to leave off) live in the training config:
`training.ddp_bucket_cap_mb`, `training.ddp_static_graph`,
`training.jit_compile`, `training.use_static_capture`, `training.amp`.

> **SFNO under DDP requires `torch<2.11`.** torch 2.11 regressed DDP for the
> SFNO models: `DistributedDataParallel.__init__` raises an int-overflow in
> `_verify_params_across_processes` on the SFNO parameter set. The recipe
> pins `torch>=2.10.0,<2.11.0` (and matching torchvision) in `pyproject.toml`
> for exactly this reason — the checked-in `uv.lock` resolves to torch 2.10.0
> / NCCL 2.27.5, which matches the validated 4×A100 SFNO benchmark. When
> re-syncing an environment, keep the pin; a floating `torch>=2.10` will pull
> 2.11 and break multi-GPU SFNO. The Pangu family is unaffected by the 2.11
> bug.

> **wandb is initialized on *every* rank under DDP** (only rank 0 logs
> metrics). This is deliberate: wandb's background threads (service IPC /
> console capture / GPU monitor) grab the GIL inside CUDA calls, and running
> wandb on rank 0 alone desyncs the ranks and deadlocks the gradient
> all-reduce mid-epoch. Initializing wandb on all ranks makes that jitter
> symmetric, keeping NCCL collectives in lockstep — mirroring the
> PanguWeather reference trainer. This is handled automatically in
> `_maybe_init_wandb`; no flag needed. See §5.4.

> **When re-syncing with `uv sync`, include the recipe's optional extras**
> (`sfno-extras`, `utils-extras`, `datapipes-extras`). Without them `uv sync`
> prunes torch-harmonics/tensorly (SFNO), wandb/mlflow, and
> zarr/xarray/netCDF4/dask, which silently breaks SFNO training. The
> `hpc/scripts/sync-all-clusters.sh` helper already passes these on every
> cluster.

### 4.3 Multi-node

Launch `torchrun` per node with a shared rendezvous (`--nnodes`,
`--node-rank`, `--master-addr`, `--master-port`) under your scheduler. The
repo ships SLURM/PBS launch skills per cluster
(`delta-smoke-test`, `derecho-smoke-test`, `midway3-smoke-test`,
`stampede3-smoke-test`, `deltaai-smoke-test`, `dsi-smoke-test`) and example
sbatch files in `hpc/scripts/`. Cluster-specific setup is documented in
`hpc/<cluster>.md`.

### 4.4 Useful overrides

| Goal | Override |
|---|---|
| Longer run | `training.max_epochs=100` |
| Bigger batch | `dataset.batch_size=8` |
| Enable bf16 | `training.amp=bf16` |
| wandb live streaming (default is offline) | `wandb.mode=online` (see §5.4) |
| Disable wandb | `wandb.enabled=False` |
| Curriculum (unroll ramp) | `training=sfno_plasim_curriculum` |
| Change checkpoint cadence | `checkpoint_save_interval=5` |
| Resume | just relaunch with the same `run_name` — it reloads `./checkpoints` |
| Per-batch loss TSV (bench) | `bench.per_batch_tsv=/path/loss.tsv` |
| Stage data to node-local disk | `dataset.stage_to_local=True` (see §4.5) |

### 4.5 Staging data to node-local disk (fast)

On clusters where the Zarr stores live on a shared parallel filesystem (Delta
keeps the E3SM stores on Lustre `/work/hdd`), a data-bound run can pay an I/O
tax reading chunks over the network while every rank and every concurrent job
contends on the same filesystem. `dataset.stage_to_local=True` copies the
stores to fast **node-local disk** (`/tmp`) once at startup, then rewrites the
dataset paths to the local copies:

```bash
torchrun --standalone --nproc-per-node=4 train.py \
    model=sfno_e3sm dataset=e3sm training=sfno_plasim \
    dataset.stage_to_local=True
```

How it works (`data_staging.py`):

- **Once per node.** `local_rank 0` on each node does the copy; a global
  barrier holds the other ranks until every node's copy is complete, so no
  rank reads a store mid-copy. Each node stages to its own `/tmp`.
- **Fast.** A Zarr store is thousands of independent chunk files (one E3SM
  year ≈ 17.5k files / 30 GB). A serial `cp -r` pays one Lustre round-trip
  per file, serialized — that is what overran an earlier 25-min budget.
  Staging instead issues **many concurrent copies** (default 256) via a
  thread pool over kernel `sendfile` (zero-copy). No extra package is needed —
  stdlib `shutil.copyfile` + threads is optimal here, confirmed by
  measurement (below); `mpifileutils`/`dcp` isn't even installed on Delta.
- **Idempotent.** A completed copy drops a `<store>.stage_complete` marker;
  a requeued job landing on the same node skips re-copying. A killed job's
  partial copy (no marker) is removed and re-staged, never read as valid.
- **De-duplicated.** `mean_path` and `std_path` usually point at the same
  normalization store — it is copied once.

**Where the time goes (measured on Delta, one E3SM year → node-local `/tmp`):**
the copy against a *warm* cache is **write-bound at ~2.2 GB/s** to `/tmp` —
identical for `shutil.copyfile` vs a raw `sendfile` loop, and flat from 64 to
512 workers, so there is nothing to gain from a fancier copy primitive. The
Lustre **read** side scales hard with concurrency (~1.6 GB/s at 128 readers →
~12 GB/s at 512), so the worker count's real job is to hide *cold*-read
latency. That is decisive: a full **cold** year (30 GB / 17.5k files, source
untouched) stages in **~47 s at 256 workers** (0.64 GB/s) versus ~2 min at 64
workers and ~7.7 min extrapolated for a serial `cp` — a ~10× speedup that is
why the default is 256, not a token handful. Bump to 512 for even more
cold-read concurrency if your source filesystem is under heavy contention.

Knobs (all in `conf/dataset/e3sm.yaml`, overridable on the CLI):

| Goal | Override |
|---|---|
| Enable staging | `dataset.stage_to_local=True` |
| Custom local dir (default `$SLURM_TMPDIR`/`$TMPDIR`/`/tmp`) | `dataset.stage_dir=/local/nvme` |
| Copy concurrency (default 256) | `dataset.stage_num_workers=512` |

> **When it helps.** Staging is a win only when the run is **I/O-bound** and
> node-local disk has room (≈60 GB for E3SM train+val+norm; Delta `/tmp` is
> ~1.3 TB). For the compute-bound SFNO-E3SM speed benchmark it buys nothing,
> so it is **off by default**. Enable it for data-bound configs or when many
> jobs hammer the same shared store.

---

## 5. Validation & inference tools

### 5.1 Mid-training validation (built into `train.py`)

When `dataset.val_zarr_path` is set, every epoch computes a **single-step
validation loss** (`val_loss`, the lat-weighted L1/L2 from `PanguPlasimLoss`)
— this is the direct analogue of PanguWeather's `validation_step` MSE.

When `validation=rollout_*` is composed and `validation.rollout.enabled=True`,
the `RolloutValidator` (`validate.py`) additionally runs **multi-step
autoregressive rollouts** on the held-out year and logs, per lead time and
per channel group (`surface` / `upper_air` / `diagnostic`):

- `rmse_step{n}_{group}` — latitude-weighted RMSE, in **physical units**
- `acc_step{n}_{group}` — anomaly correlation (only when a climatology is
  provided)

It is DDP-safe (streaming per-rank sums, single all-reduce in `finalize()`)
and **ensemble-aware** via the `Perturber` API
(`validation.rollout.perturber` ∈ `deterministic` / `replicate` /
`gaussian_ic`, with `ensemble_size` and `perturber_scales`).

Lead times come straight from the PanguWeather validation block:
`validation.rollout.log_steps: [1, 12, 20, 40, 60]` (60 steps × 6 h = 15-day
rollout in `rollout_5412.yaml`). `validation.every_n_epochs` controls cadence.

All of these route to the console **and to wandb** (the default logging
backend) — training metrics under the `train/` namespace and validation
metrics (`val_loss`, `rmse_step*`, `acc_step*`) under `valid/`. See §5.4.

### 5.2 Inference (autoregressive rollout to a file)

```bash
python inference.py \
    model=sfno_e3sm dataset=e3sm \
    +inference.checkpoint_dir=/path/to/checkpoints \
    +inference.output_path=/path/to/preds.nc \
    +inference.max_step=60 \
    +inference.ic_start=[0, 60, 120, 180]
```

### 5.3 After-the-fact scoring (`validate_cli.py`)

Score a saved prediction file against the reference Zarr, without loading the
model — reuses the same streaming RMSE/ACC aggregators:

```bash
python validate_cli.py \
    dataset=e3sm \
    +validation_cli.predictions=/path/to/preds.nc \
    +validation_cli.reference_zarr=/path/to/e3sm/2045.zarr \
    +validation_cli.output_json=/path/to/scores.json \
    +validation_cli.output_md=/path/to/scores.md \
    +validation_cli.climatology_zarr=/path/to/climatology.zarr   # optional, enables ACC
```

### 5.4 Logging with wandb (default backend)

ai-rossby uses **[wandb](https://wandb.ai)** as its logging backend, replacing
PanguWeather's MLflow. It is **on by default** (`conf/config.yaml`) in
`offline` mode, so it works with no login and no network — every run writes a
local `./wandb/<run>/` directory under the Hydra run dir. `LaunchLogger`
routes all training and validation metrics to it automatically:

- `train/…` — total loss + per-group components (surface / upper_air /
  diagnostic / vae_kl) per minibatch and per epoch
- `valid/…` — `val_loss`, and (when a `validation=rollout_*` group is
  composed) `rmse_step{n}_{group}` and `acc_step{n}_{group}`

Install it (bundled in the recipe's `requirements.txt`):

```bash
pip install -r examples/weather/ai_rossby/requirements.txt   # brings in wandb
```

Common controls (all overridable on the CLI):

| Goal | Override |
|---|---|
| Stream to the wandb cloud live | `wandb.mode=online` (run `wandb login` once first) |
| Set project / team | `wandb.project=my-proj wandb.entity=my-team` |
| Name the run | `wandb.name=sfno_e3sm_run0` (defaults to `run_name`) |
| Disable wandb entirely | `wandb.enabled=False` |
| Upload an offline run afterwards | `wandb sync ./outputs/<run_name>/wandb/<run>` |

**Multi-GPU:** wandb is created on *every* rank, but only rank 0 routes
`LaunchLogger` metrics to it — the other ranks' runs stay empty. Running
wandb on all ranks (rather than rank 0 alone) is required for DDP: its
background threads grab the GIL inside CUDA calls, and doing so on one rank
only desyncs the ranks and deadlocks the gradient all-reduce. Initializing on
all ranks makes the jitter symmetric so NCCL stays in lockstep. `physicsnemo`'s
`initialize_wandb` names the per-rank runs (`..._Process_<rank>_...`) and gives
them a shared DDP group tag, so they group cleanly in the wandb UI. No flag is
needed — this is automatic (`_maybe_init_wandb` in `train.py`).

If the `wandb` package isn't installed, the trainer prints one warning and
falls back to console logging — no crash.

> **Migrating dashboards from MLflow.** The metric *names* differ from
> PanguWeather (`val_loss` / `rmse_step*` vs. MLflow's `Validation error`),
> but the quantities are the same or richer. If you specifically need MLflow
> instead, `LaunchLogger` still supports it: create the run with
> `initialize_mlflow(...)` and pass `use_mlflow=True` to
> `LaunchLogger.initialize(...)` in `train.py`, mirroring
> `examples/weather/pangu_weather/train_pangu_era5.py`.

---

## 6. End-to-end SFNO-E3SM checklist

1. **Convert data** → `python tools/data/e3sm/pangu_h5_to_zarr.py --year … --output …` for each split + a normalization store.
2. **Point the config** at your stores in `conf/dataset/e3sm.yaml` (or override on CLI).
3. **Verify the model YAML** (`conf/model/sfno_e3sm.yaml`) matches the source SFNO block and the converter's channel order (§1.2 / §1.3).
4. *(Optional)* **Convert a warm-start checkpoint** → `tools/checkpoint_translation/sfno_plasim.py … --strict`.
5. **Launch**: `torchrun --standalone --nproc-per-node=<N> train.py model=sfno_e3sm dataset=e3sm training=sfno_plasim validation=rollout_5412 run_name=…`.
6. **Watch** `val_loss` + `rmse_step*` in wandb (default, offline — see §5.4) or the console.
7. **Evaluate** with `inference.py` → `validate_cli.py` for a final scorecard.

---

## References in this repo

- Recipe code: `examples/weather/ai_rossby/{train.py, train_loop.py, loss.py, validate.py, validate_cli.py, inference.py}`
- Extra deps (wandb): `examples/weather/ai_rossby/requirements.txt`
- Configs: `examples/weather/ai_rossby/conf/`
- Original PanguWeather reference trainer: `examples/weather/pangu_weather/`
- Checkpoint translators: `tools/checkpoint_translation/`
- Data converters: `tools/data/{e3sm,era5,plasim}/pangu_h5_to_zarr.py`
- Component-reuse rationale: `pangu_plasim_reuse_plan.md`
- Per-cluster HPC setup: `hpc/<cluster>.md`
