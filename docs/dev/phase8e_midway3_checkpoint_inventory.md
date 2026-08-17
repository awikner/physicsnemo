# Phase 8e — Midway3 Checkpoint Inventory (predicted)

> **Superseded in part (2026-08-14, Phase 12h).** This doc predicts where
> upstream **amip v1** Lightning ckpts live. Real **amip_v2** checkpoints now
> exist and are validated — see
> [the v2 section below](#amip_v2-checkpoints-2026-08-14) — so for v2 work start
> there. The v1 punch list stays valid for the frozen families.

Reference doc for the Lightning `.ckpt` translator work. Derived from the
configs vendored from upstream `amip` (commit `497827e`,
`/work/nvme/bdiu/awikner/amip/configs/`), specifically by scraping every
`checkpoint:`, `partial_checkpoint:`, `forecaster_checkpoint:`, and
`downscaler_checkpoint:` reference. Use this as a punch list when
hunting for trained `.ckpt` blobs on Midway3 before wiring them through
the Phase 8e translator.

## Path convention

Upstream amip Lightning runs land under
`{log_dir}/{run_name}/{ckpt_name}.ckpt`, where:

- `log_dir` on Midway3 is **`/project/pedramh/ayz/AMIP_logs/`**.
- `run_name = {ModelName}_{Variant}_{Seed}_{ISO-timestamp}` — for
  example `SI_X_AIMIP_interp_gaussian_42_2026-05-27T17-37-14`. Seed is
  almost always `42`.
- `ckpt_name` is one of `last`, `model_epoch=NN`,
  `model_epoch=NN_step=NNNNN_best`.

Other clusters used by the same upstream training campaign for
cross-reference (in case ckpts were rsynced):

- `/glade/derecho/scratch/ayz/AMIP_logs/` (NCAR Derecho)
- `/mnt/home/azhou/ceph/data/logs/` (Flatiron CCQ)

## Likely checkpoint ↔ config pairs

| Checkpoint family | Likely Midway3 run subdir(s) | Upstream config | Local equivalent |
|---|---|---|---|
| **SI_X** (DynamicInterpolant) | `SI_X_AIMIP_interp_gaussian_42_2026-05-27T17-37-14/last.ckpt`<br>`SI_X_AIMIP_spec_42_2026-05-26T21-15-41/last.ckpt` | `configs/SI_midway_AIMIP.yaml`<br>(filename is `SI_midway`, but `model_name: SI_X` inside) | `model=amip_si_x` + `loss=si_x` |
| **SI** (DriftScheduler) | `SI_AIMIP_interp_gaussian_v_42_2026-06-08T09-01-42/last.ckpt` | `configs/SI_midway_AIMIP_V.yaml`<br>(`model_name: SI`) | `model=amip_si` + `loss=si` |
| **x_DDC** (super-res cascade) | `x_DDC_x_DDC_42_2026-05-20T16-21-23/last.ckpt`<br>`x_DDC_x_DDC_42_2026-04-16T17-08-57/model_epoch=24_step=72700_best.ckpt` | `configs/DDC_midway_AIMIP.yaml`<br>(`model_name: x_DDC`) | **Ported (Phase 8f, F6)** — `model=amip_x_ddc` + `loss=x_ddc` / `sampler=x_ddc`; see below |
| **Combined** (forecaster + downscaler) | Typically no standalone ckpt — `combined_midway.yaml` references both an SI_X forecaster and an x_DDC downscaler ckpt | `configs/combined_midway.yaml` | **Ported (Phase 8f, F6)** — `CombinedModule` composes two independently-translated checkpoints at runtime; no standalone ckpt to translate (see `conf/model/amip_combined.yaml`) |

### ERDM / RFM notes

I don't see any Midway-targeted configs for ERDM or RFM. Every
ERDM run referenced in the vendored configs sits under
`/glade/derecho/scratch/ayz/AMIP_logs/` or
`/mnt/home/azhou/ceph/data/logs/`. RFM has no checkpoint references
in any vendored config. **Translator live-validation of ERDM / RFM
will most likely need to pull a ckpt from Derecho or CCQ rather than
Midway3** — verify before scoping that part of Phase 8e.

## Naming-convention quirks that matter for the translator

- **Filename ≠ class.** The yaml filename and the model class don't
  have to match: `SI_midway_AIMIP.yaml` actually trains SI_X. The
  translator must key on `model.model_name` *inside* the yaml, never
  on the filename.
- **`_V` variant.** Suffixes like `SI_AIMIP_..._v_...` /
  `SI_midway_AIMIP_V.yaml` mark the V variant (velocity-prediction /
  drift target). It changes scheduler hyperparameters; the translator
  needs to read these from the saved Lightning hparams blob, not
  guess from the run name.
- **Other variant suffixes.** `interp_gaussian`, `spec`, `wCO2`,
  `forcings_smooth` are noise / forcing variants. They change
  scheduler args (e.g. `noise: gaussian` vs `noise: spectral`) and
  add varying-boundary channels (e.g. `wCO2` adds the global-mean
  CO2 channel). All survive the translator as scheduler-config
  tweaks — no architectural changes.
- **Seed in the run name.** Seed is always `42` in the configs I've
  scraped. If you find a run dir whose seed isn't `42`, it's a
  hand-run experiment outside the standard training scripts.

## Translator status (post-transfer)

The Phase 8e translator
[`tools/checkpoint_translation/amip_si.py`](tools/checkpoint_translation/amip_si.py)
has been live-validated against the
[`/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/`](/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/)
tree on Delta NVMe. Six `last.ckpt` runs in scope (excluding the
not-ported `x_DDC` family); results:

| Run subdir | Result | Notes |
|---|---|---|
| `SI_AIMIP_interp_gaussian_v_42_2026-06-02T20-10-55` | OK | Auto-derive + load + forward, all clean. |
| `SI_X_AIMIP_interp_gaussian_42_2026-05-28T09-27-49` | OK | 1.2B params, n_averaged=41625 (EMA-mature). |
| `SI_X_AIMIP_spec_42_2026-05-25T19-52-10` | OK | Spectral noise variant — same backbone, different scheduler config. |
| `SI_X_AIMIP_wCO2_42_2026-05-24T19-28-58` | OK | Translator auto-trims `varying_boundary_variables` to match `c_grid_dim=5` (drops the trailing entry; upstream routes that one through the c_scalar path). |
| `SI_X_AIMIP_wCO2_interp_gaussian_42_2026-05-30T08-32-59` | OK | Same wCO2 auto-trim path. |
| `SI_V_new_42_2026-05-20T20-47-08` | **xfail** | Predates the vendored amip commit `497827e`. Uses an older `ScalarEmbedder` (plain `Linear(scalar_dim → c_scalar_embed_dim)`); the vendored `CalendarEmbedding` wraps the input in sinusoidal embeddings, so `scalar_embedder.out_proj` shape is incompatible. Would need either re-vendoring the older backbone or a hand-written shim — out of scope for Phase 8e MVP. |

The unit + live test stack lives at
[`test/tools/checkpoint_translation/test_amip_si.py`](test/tools/checkpoint_translation/test_amip_si.py)
(26 unit + 6 live, of which 1 xfail and 5 pass).

### x_DDC translator status (Phase 8f, F6)

`XDDCWrapper` + the `x_DDC` → `XDDCWrapper` translation path landed
this phase (`test_live_translation_round_trips_xddc`, parametrized
over the same
[`/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/`](/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/)
tree, filtered to `x_DDC*` run dirs). Only the `decoder_type: unet`
denoiser is vendored (`XDDCUNet`) — both real Midway3 x_DDC checkpoints
use it (their configs carry only a `decoder:` UNet block, no `dit:`
block, and `decoder_type` defaults to `"unet"` upstream when absent).
`decoder_type: dit` (the DiTAE autoencoder denoiser,
`modules/models/DiTAE.py`) is **not vendored** — the translator raises
`NotImplementedError` if it's ever encountered; nothing in scope needs
it today.

**Live-validated** (CPU, this dev node — `/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/` is mounted directly here even without a GPU; `torch.load` + a CPU forward pass is enough to validate the translator, just slower than on a GPU). `_collect_midway_xddc_ckpts()` actually found **five** `x_DDC*` run dirs on Delta (more than the two originally known about at Phase 8e time) — all five translate → load → forward cleanly:

| Run subdir | Result | Notes |
|---|---|---|
| `x_DDC_x_DDC_42_2026-05-20T16-21-23` | OK | |
| `x_DDC_x_DDC_42_2026-04-16T17-08-57` | OK | Only has `model_epoch=24_step=72700_best.ckpt`, no `last.ckpt` — `_collect_midway_xddc_ckpts()` falls back to the best-epoch file. |
| `x_DDC_x_DDC_42_2026-06-01T09-19-35` | OK | Not in the original Phase 8e inventory scrape — discovered by the collector. |
| `x_DDC_x_DDC_AIMIP_train_noise_42_2026-05-22T16-07-43` | OK | `train_noise` variant — same UNet decoder, different training-noise scheduler hyperparameters (doesn't affect the wrapper/backbone shape). |
| `x_DDC_x_DDC_AIMIP_train_noise_42_2026-05-23T18-41-12` | OK | Same as above. |

`Combined` still has no standalone checkpoint to translate — see
`conf/model/amip_combined.yaml` for how to compose
`CombinedModule` from the two checkpoints above at runtime instead.

## Translator wiring (when a ckpt is in hand)

Default invocation (auto-derive wrapper kwargs from the ckpt's
`hyper_parameters` block — recommended for upstream ckpts whose
channel layout doesn't match the in-repo wrapper defaults):

```bash
python tools/checkpoint_translation/amip_si.py \
    --source /work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/SI_X_AIMIP_interp_gaussian_42_2026-05-28T09-27-49/last.ckpt \
    --output /work/nvme/bdiu/awikner/translated-mdlus/si_x_aimip_interp_gaussian.mdlus
```

Override the wrapper config with an explicit YAML (use this when the
ckpt's hparams block is stale or when you want to swap channel groups
post-hoc):

```bash
python tools/checkpoint_translation/amip_si.py \
    --source /path/to/last.ckpt \
    --model-config examples/weather/ai_rossby/conf/model/amip_si_x.yaml \
    --output /path/to/output.mdlus
```

Other flags worth knowing:

- `--prefer-live` swaps the source from `state_dict` (EMA-averaged,
  default) to `current_model_state` (live training weights).
- `--strict` refuses to write the output when there are missing or
  unexpected keys.
- `--target-class` forces a specific wrapper class
  (`AmipDiTWrapper` / `RollingDiTWrapper` / `ERDMWrapper`); useful
  when the ckpt's `model_name` hint is wrong.
- `--model-name` overrides the source `model_name` detection
  (e.g. for ckpts with a missing hparams block).

When you find an actual `.ckpt`, dump the Lightning hparams blob from
inside the ckpt to sanity-check the auto-derive output:

```python
import torch, sys
sys.path.insert(0, "/work/nvme/bdiu/awikner/amip")  # required to unpickle the data normalizer
ckpt = torch.load("/path/to/last.ckpt", map_location="cpu", weights_only=False)
print(ckpt["hyper_parameters"]["config"]["model"])
```

## Cross-reference

- `physicsnemo/experimental/diffusion/__init__.py` — vendored
  scheduler families (SI, SI_X, ERDM, RFM, EDM).
- `physicsnemo/experimental/models/amip_si/wrappers.py` — wrappers
  the translator's output needs to hydrate.
- `implementation_plan.md`, Phase 8e — translator design notes,
  including the prefix-stripping helper shared with `pangu_plasim.py`.

## amip_v2 checkpoints (2026-08-14)

The first real v2-trained checkpoints, both `project: amip_v2`. Fetched off
retiring Derecho scratch by Globus (task `56188e3e`, 9.6 GB) to
**`/eagle/lighthouse-uchicago/amip-checkpoints/`** on Polaris; their original
Derecho paths were `ayz/AMIP_logs/` and `katyr/ai-models/CMU_model/`, and both
were trained at TACC (`/scratch/10512/azhou4`, reading
`/scratch/09979/awikner/ERA5/h5_dailyavg` for the downscaler).

| | forecaster | downscaler |
|---|---|---|
| run | `ERDM_fancy_42_2026-08-10T13-21-13` | `x_DDC_42_2026-08-07T09-34-49` |
| size / params | 5.1 GB / 561.61M | 4.5 GB / 389.62M |
| source `model_name` | `ERDM` + `backbone: DiT` → **`RollingDiTWrapper`** | `x_DDC` + `decoder_type: dit` → **`XDDCWrapper(decoder_type="dit")`** |
| our config | `conf/model/amip_erdm_fancy.yaml` | `conf/model/amip_x_ddc_dit.yaml` |
| contract | 154 ch (151 state + **3 ocean**: SST, anomaly, ice), `c_grid_dim` 6, `scalar_dim` 3, 45×90 state / 180×360 forcings | 302 → 151 @ 180×360 |
| features exercised | 12b packing, full 12e set, 12f ocean tail, 12g `sst_anomaly_channel: append` + `scalar_forcing: global_mean_sst` | 12h `DiTAE` |
| translated | `--strict`: 373 kept, 1 scheduler dropped, **0 unknown / 0 missing** | 192 kept, **0 dropped / 0 missing** |
| **numerical vs upstream** | **max │diff│ 0.0000e+00** | **max │diff│ 0.0000e+00** |

Translated artifacts: `amip-checkpoints/translated/{erdm_fancy,x_ddc_dit}.mdlus`.
Reproduce end to end with `hpc/scripts/translate_v2_checkpoints_polaris.pbs`
(translate both, then compare against upstream's own forward).

### Translating a v2 ckpt

```bash
python tools/checkpoint_translation/amip_si.py \
    --source .../last.ckpt --source-contract v2 \
    --target-class RollingDiTWrapper \
    --output erdm_fancy.mdlus --strict
```

No `--model-config`: the auto-derive path reads the ckpt's own hparams. Prefer it
for v2 — it is now the exercised path, and it derives the channel contract rather
than trusting a hand-written YAML to agree with the weights.

**Verify numerically, not just by key parity.** Every contract bug in this
rebaseline was shape-preserving, so a clean `load_state_dict` says nothing about
channel order:

```bash
python tools/checkpoint_translation/verify_v2_numerical.py \
    --source .../last.ckpt --amip-v2-repo <amip_v2 checkout> --family erdm
```

It builds upstream's backbone and ours in one process (their `RollingDiT` /
`DiTAE` import only torch + einops, so no Lightning or `norm_stats` needed), hands
both the same weights, and compares the forward, localising any mismatch per
channel block. `--synthetic` self-tests the harness without a checkpoint.

### Gotchas the real v2 ckpts exposed

- **Backbone kwargs live under the backbone-named key** — `model.ERDM.DiT`, not
  `model.ERDM.model` (v1's `ERDM_Unet.yaml` carries `DiT` *and* `UNet`). Reading
  the wrong key silently yields `{}`: class-default geometry, `scalar_dim` → 2.
- **`nlat`/`nlon` are absent from every upstream ERDM config** — `RollingDiT`
  defaults them to 45×90, the *coarse state* grid, while the data resolution is
  180×360. The wrapper's `horizontal_resolution` must be the state grid
  (`data.horizontal_resolution / downsample_factor`).
- **`model_name: ERDM` is ambiguous** across the rebaseline; `model.backbone`
  decides between `ERDMWrapper` (v1 UNet) and `RollingDiTWrapper` (v2).
- **The SST anomaly is a derived channel** — not in the store's
  `varying_boundary_variables`, but counted in the source's `c_grid_dim`. The
  translator splices it in via `climate.grid_forcing_names`; a model config must
  list it too (post-rescaler order) or `c_grid_dim` comes out one short.

## Real v1 SI checkpoints (2026-08-14)

The first v1 SI-family checkpoints verified against upstream, off **Derecho
scratch** (`/glade/derecho/scratch/ayz/AMIP_logs/`, retiring) and now on Polaris
Eagle at `/eagle/lighthouse-uchicago/amip-checkpoints/`.

| | SI-V | SI-X (wCO2) |
|---|---|---|
| run | `SI_AIMIP_interp_gaussian_v_42_2026-06-02T20-10-55` | `SI_X_AIMIP_wCO2_interp_gaussian_42_2026-05-30T08-32-59` |
| file used | `last.ckpt` (11.3 GB) — **stale, see below** | `model_epoch=19.ckpt` (11.3 GB) — **no `last.ckpt` in that run** |
| source `model_name` / block | `SI` / `model.SI.model` | `SI_X` / `model.SI_X.model` |
| our config | `conf/model/amip_si.yaml` | `conf/model/amip_si_x.yaml` |
| contract | 151 ch state, `c_grid_dim` 5, `scalar_dim` 2 | same, `scalar_dim` 3 (CO2 routed) |
| backbone | dim 1536, 24 blocks, 8 ca-blocks, patch 1, **45×90** | same, wider embeds (192/192) |
| params | 1247.77M | 1248.85M |
| keys (`--strict`) | 308 kept, 1 scheduler dropped, 0 unknown / 0 missing | 312 / 1 / 0 / 0 |
| **numerical, random inputs** | **max │diff│ 0.0000e+00** | **max │diff│ 0.0000e+00** |
| **numerical, REAL daily-avg batch** | **max │diff│ 0.0000e+00** | **max │diff│ 0.0000e+00** |
| `inference.py` rollout | writes a forecast file | writes a forecast file |

**SI-V's `last.ckpt` is not the trained weights.** It reports epoch 4 /
global_step 7700 and its mtime (Jun 3) predates `model_epoch=20…24.ckpt`
(Jun 5–7) in the same directory, so it looks like a leftover from an earlier
segment. Translation fidelity is checkpoint-independent, so the verification
below stands as a statement about the converter — but anyone wanting SI-V's
actual weights should take `model_epoch=24.ckpt`.

Reproduce with `hpc/scripts/verify_si_checkpoints_polaris.pbs` (translate +
verify on random inputs) then `hpc/scripts/run_si_coarse_configs_polaris.pbs`
(re-translate SI-X, real-data A/B, rollout). Both need upstream **v1** at the
vendored commit `497827e`; the Polaris copy came from a `git archive` of that
commit shipped over Globus, because ALCF's proxy blocks `git clone`.

### These are COARSE-STATE models, and they now own the SI config names

Their own `config.yml` states `nlat/nlon: 45×90` with `in_channels: 302`
(= 151 × 2, `[x_noised, cond]` concatenated) and `c_grid_downsample: 4` — a
45×90 state with 180×360 forcings reduced by the backbone's strided conv, the
same shape as the v2 ERDM.

The `amip_si.yaml` / `amip_si_x.yaml` that shipped before 2026-08-17 described a
*different* model — 16 surface variables (161 channels) on a 180×360 grid — that
no checkpoint we hold matches. Those were **deleted** and the names reassigned to
these two contracts, so `model=amip_si` / `model=amip_si_x` now mean the real
weights. Consequences:

- **translate with auto-derive**; the config now agrees with the checkpoints, but
  the auto-derive path is still the one under test and needs no YAML;
- **pair with `dataset=amip_dailyavg_coarse`** (coarse state + 1° boundary
  store), not `amip_1981`, which serves a native-resolution state and the older
  variable set.

On the data: `amip_dailyavg` is **6-hourly** (`data_timedelta_hours: 6` —
"dailyavg" names the 24-hour-accumulation *variables*, not the row cadence) and
carries exactly this variable set, so it is the right store to train this model
type on. What differs from the upstream runs is the raw archive and the
statistics: they used `/project/pedramh/AMIP/h5` with `normalize_mean_interp.nc`,
we use `normalize_*_dailyavg.nc`. So *reproducing their checkpoints* is a
different exercise from *training a model of the same type*, and the bitwise
verification below is a statement about the converter, not about skill.

### What running them exposed

Three defects, none visible from key parity or from the random-input check:

1. **No varying-boundary subset in the shared pipeline.** SI-V lists 3 varying
   channels where the store serves 4 (upstream's run never fed
   `global_mean_co2`), and `NanFillTransform` — sized from the *model's* list —
   indexed a 4-channel tensor and raised `IndexError`. `inference.py` had handled
   this since the Pangu-S2S port; `train_diffusion._build_dataset` had not. Now
   `dataset_setup.resolve_varying_subset` + `VaryingBoundarySubset`, with the
   normalizer aligned to the model's list. The dropped channel is the store's
   **first**, so the indices must come from name lookup — "take the leading N"
   would mis-assign every forcing.
2. **The translator deleted scalar-routed channels instead of routing them.**
   Its trim heuristic predated the wrappers having
   `scalar_routed_boundary_variables`, so `global_mean_co2` was dropped from
   `varying_boundary_variables` outright. Same geometry, but not runnable: with
   nothing routed, `resolve_scalar_forcing` yields a 2-wide calendar row against
   a model wanting 3. It now routes when the source's `scalar_dim == 2 + n` *and*
   the channels are name-matched — the second condition matters, because the
   `fancy` configs' `scalar_dim: 3` is the `global_mean_sst` trend scalar and
   would otherwise have "routed" sea ice into the calendar row.
3. **`ForcingAssembler._scalar_of` collapsed the batch axis** (`reshape(-1)`),
   so batched callers hit `cat` of `(B, 2)` and `(1,)`. The rank error was the
   lucky outcome: with shapes aligned, every sample would have received one CO2
   value averaged across the batch. Reduces over spatial axes only now.

`AmipDiTWrapper` also gained `scalar_routed_boundary_variables` (additive; an
empty list reproduces the old arithmetic), because SI-X cannot otherwise be
expressed: its 4-entry varying list against a `Conv2d(5 -> 192)` `c_grid_embed`
only reconciles if CO2 rides the calendar row.
