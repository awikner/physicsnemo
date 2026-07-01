# Phase 8e — Midway3 Checkpoint Inventory (predicted)

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

**Pending real-checkpoint validation**: the live test above is written
and passes its unit-level dry runs (synthetic hparams), but hasn't yet
been run against the two real x_DDC `.ckpt` files on Delta (that
requires a GPU session with the checkpoint tree mounted — tracked as
part of the Phase 8f HPC validation pass alongside the F1/F3/F7 Delta
runs). Update this row with OK/xfail + notes once that run completes:

| Run subdir | Result | Notes |
|---|---|---|
| `x_DDC_x_DDC_42_2026-05-20T16-21-23` | _pending Delta run_ | |
| `x_DDC_x_DDC_42_2026-04-16T17-08-57` | _pending Delta run_ | Only has `model_epoch=24_step=72700_best.ckpt`, no `last.ckpt`. |

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
