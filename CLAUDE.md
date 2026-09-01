# CLAUDE.md — ai-rossby fork project brief
Orientation for working in this repo. This is the **ai-rossby** fork of NVIDIA
PhysicsNeMo, maintained by the UChicago group for climate emulation (PLASIM,
ERA5, E3SM, AMIP) with SFNO / Pangu-Weather models.

- **Active branches:**
  - `ai-rossby` — the core fork; PRs usually target this, not upstream `main`.
  - `ai-rossbypalooza` — **new, currently being set up.** Two-week hackathon
    branch for the mixture-of-AI-weather-experts monsoon rainfall project (see
    below). Everyone is just starting on this branch — expect fast-moving,
    partially-scaffolded code here.
- **Recipe:** `examples/weather/ai_rossbypalooza/` (`train.py`, `conf/`, `DATA.md`,
  `data_staging.py`).
- **Durable engineering context:** [`docs/dev/context/`](docs/dev/context/) —
  read these before touching training envs, the data pipeline, or cluster storage.

## Data pipeline (one loader, config-selected)
- **Loader:** a single shared class `ClimateZarrDataset` (alias
  `PlasimClimateDataset`) — `physicsnemo/experimental/datapipes/climate/dataset.py`.
  It reads each Zarr store's variable groups + level coords from the store's own
  `attrs`, so the same class serves ERA5 / E3SM / PLASIM / AMIP. Dataset selection
  is by config (`cfg.dataset.zarr_path`), not by different classes.
- **Normalizer:** `ClimateNormalizer` (= `PlasimNormalizer`),
  `physicsnemo/experimental/datapipes/climate/transforms.py`. Matches pressure
  levels **by value** (raises on a missing level, never silently misaligns).
- **Data catalog:** `hpc/data_registry.yaml` + `tools/data/registry.py`
  (show/check/scan) + `tools/data/sync_dataset.py` (Globus sync, `--stage-raw`,
  `--rehydrate`). Per-cluster data root via the `AI_ROSSBY_DATA` env var.

## Clusters & storage topology
| Cluster | Role | Notes |
|---|---|---|
| **Delta** (NCSA) | **intended persistent master** (`/work/hdd`) | not purged; shared bdiu group quota. GPU: A40 + A100 partitions. |
| **Stampede3** (TACC) | conversion + working copy (`$SCRATCH`) | no inode limit; H100. globus-cli at `~/gcli`. |
| **Derecho** (NCAR) | master **RETIRING** (`/glade/derecho/scratch`) | inode-limited (~26.2M-file cap); being decommissioned → Delta. |
| DeltaAI (NCSA) | GH200/aarch64 training | shares Delta `/work`; env caveats in context notes. |
| Midway3, DSI | (UChicago) | not yet holding converted data. |

Cross-cluster zarr replication uses `hpc/scripts/replicate_tar.sh` (tar-bundle →
Globus → untar; ~5× faster than per-file Globus for these tiny-chunk stores).

## Current state (2026-07-21)
- **Phase 11 complete** — all datasets converted + consolidated; `registry.py
  check` green. See [phase11-data-consolidation](docs/dev/context/phase11-data-consolidation.md).
- **ERA5 normalization fixed** to 18 levels (200 hPa was missing) — both combined
  and separate norm stores, all clusters.
- **DEFERRED (do not start without the user):** retire Derecho scratch, re-home
  `e3sm`/`plasim_plev`/`amip` gap-ranges to Delta persistent storage. See
  [derecho-retire-rehome-to-delta](docs/dev/context/derecho-retire-rehome-to-delta.md).
- **`ai-rossbypalooza` just kicked off** — see the hackathon section below for
  scope, methods, and who owns what.

## Gotchas that will bite you (details in `docs/dev/context/`)
- **Jobs run in a container, not a venv.** One multi-arch image on NGC PyTorch
  serves Delta, DeltaAI, Stampede3, Midway3, Polaris and Derecho; launch through
  `hpc/scripts/container_run.sh`. Contract: [`hpc/containers.md`](hpc/containers.md);
  measured facts and traps:
  [container-migration](docs/dev/context/container-migration.md). **DSI is the only
  venv cluster** (no container runtime, no module system).
  Image conversion happens on **DeltaAI**, not Delta — Delta's apptainer 1.5.1
  bundles a mksquashfs that segfaults on this image.
- **Multi-GPU (DDP):** `torch < 2.11` (2.11/2.12 break DDP) — now enforced by the
  image's base tag rather than by `uv sync` luck; init wandb on *every* rank —
  **never inside `if rank == 0:`** (a rank-0-only init hangs DDP's first NCCL
  collective; root-caused + fixed 2026-08-07, see
  `docs/dev/wandb_ddp_hang_fix_plan.md`; `wandb.allow_multigpu=false` re-arms
  an auto-disable guard if it resurfaces). *Venv path only:* `uv sync` must include
  `--extra sfno-extras --extra utils-extras --extra datapipes-extras` or it
  silently prunes SFNO/zarr deps.
- **DeltaAI (GH200):** the container retires three documented workarounds here —
  the Cray-PE `CXX=CC` that broke TorchInductor, the import-broken conda `wandb`,
  and `AI_ROSSBY_NO_CUDNN_SDPA` (bf16 SDPA at the v2 geometry measured working on
  GH200 in-container). *Venv path only:* install wandb into `.venv-deltaai`;
  `torchrun` isn't on the venv PATH.
- **Globus high-assurance sessions time out** — refresh with `globus session
  update <domain>` (Delta ↔ `access-ci.org`, TACC ↔ `uchicago.edu`).
- **Don't `import physicsnemo` on a login node** for small scripts — CUDA/Warp
  init can core-dump; use plain xarray/numpy.
- **Latitude orientation is NOT uniform across datasets.** `era5`/`plasim`/`imerg`
  are N→S; **`amip*` and `e3sm` are S→N** (both correct — the raw archives are S→N
  and the data is stored verbatim). Never assume a row order positionally: read the
  `lat` coord or the `lat_row_order` attr. The raw H5 archives carry *no* lat coord,
  so converters must verify against physical anchors — `tools/data/check_lat_orientation.py`
  does this, and the AMIP converter now asserts it at ingest. Full audit +
  outstanding repairs: [lat-orientation-audit](docs/dev/context/lat-orientation-audit.md).

## Conventions
- Commit messages end with the `Co-Authored-By` trailer; branch before committing
  on `main`; commit/push only when asked.
- CI header check requires the NVIDIA SPDX copyright line; add the UChicago line
  alongside it (see existing files).

---

# Hackathon project (`ai-rossbypalooza`): Mixtures of AI Weather Experts for Week-2 Monsoon Rainfall

Two-week hackathon, just starting. Goal: test whether learned, state-dependent
combination of AI weather prediction (AIWP) models (Methods 0/1/2) beats the
EMOS baseline for week-2 (lead days 8–14) Indian summer monsoon rainfall, and
whether it fixes the intensity-blurring problem that plain averaging doesn't
(AIWP models smooth/compress rainfall intensity with lead time, worst in
exactly this 8–14 day range).

**Success criterion:** Methods 0/1/2 outperform EMOS on the metrics below, at
moderate-to-heavy rain intensities specifically (not just aggregate RMSE), and
ideally recover known monsoon structure (active/break phases, orographic vs.
depression rainfall) in the learned gate weights.

### Classical baseline
- **EMOS** — the sole classical blending method being run, and the bar the
  learned methods (0/1/2 below) must clear. Other classical options (BMA,
  quantile mapping, GBT baselines, ex-post-optimal fixed weights) were
  discarded.

### Methods (mixture of AI weather experts)
- **Method 0 — Vanilla MoWE.** Append a ViT gate on top of the expert rollouts
  (per Chakraborty et al. 2025); gate ingests expert forecasts + dynamical
  fields + lead time, emits per-gridpoint weights + a bias term. Underlying
  expert models are frozen — only the gate is trained. Extending the published
  (deterministic, temperature/wind, ≤48h) version to precipitation and leads
  out to 14+ days, with quantile/CRPS output heads for calibration.
- **Method 1 — MoWE + fine-tuning.** Same architecture as Method 0, but the
  underlying expert models are fine-tuned (not frozen) jointly with the gate.
- **Method 2 — Optimal perturbation.** Model weights remain unchanged; instead
  learn a state-dependent optimal initial-condition perturbation. Based on
  Optimal perturbation (Vonich & Hakim, 2026) —
  [https://journals.ametsoc.org/view/journals/aies/5/3/AIES-D-26-0009.1.xml](https://journals.ametsoc.org/view/journals/aies/5/3/AIES-D-26-0009.1.xml),
  which found that even just a learned mean perturbation gives a substantial
  accuracy gain. Can use NCEP/DA-style ICs from background covariance, or a
  learned state-dependent perturbation generator. Challenge: the perturbation
  is only known post-hoc at training time, so inference needs a generative
  perturbation model.

Mixing (Methods 0/1) can happen either post-hoc (conditioned on lead time,
mixing just the regional target) or during autoregressive rollout (requires
mixing the full state, not just the regional target).

*Note: PBC (probabilistic bias correction, Guan et al. 2026) and the
latent-space multi-model data-assimilation idea (cf. MM-EnKF) are **not** being
pursued for this project.*

### Data
Hindcasts are being generated from the following models only (GenCast,
NeuralGCM, GraphCast, etc. are **not** in scope):
- **Pangu-S2S** (in-house, twice-weekly to weekly May–Jul init, 45–50-day rollouts)
- **SFNO-S2S** (in-house, same init cadence/rollout as above)
- **Arches-S2S**
- **AIFS** — maybe, not confirmed

0.25° resolution, native time resolution unless noted otherwise.

Verification: IMERG (satellite, land+ocean) + IMD 0.25° gauge analysis (land),
ERA5 precip as a sensitivity check, IFS-ENS as external benchmark. Domain
5–35°N, 60–100°E, JJAS. Gate/correction predictors: expert precip + dynamical
fields, lead time, monsoon-phase indices, climatological quantiles, lagged obs.

*Data catalog for this project (available/needed) — TODO, not filled in yet.*

### Targets & metrics
Primary target: week-2 (lead 8–14d) accumulated precip as tercile and
exceedance probabilities at fixed (50/100/150/200 mm/wk) and climatological
(75/90/95th pct) thresholds, 1° resolution. Secondary: daily precip at leads
8–14. Extension: weekly totals at weeks 3–6.

Metrics: RMSE + Bias first, then BSS/RPSS/AUC vs. static climatology (matching
Aitken et al. 2026 protocol for direct comparability), plus CRPS/CRPSS, SEEPS,
FSS (spatial structure), rank histograms + spread-skill (calibration).
Significance via block bootstrap.

Evaluation protocol: leave-one-year-out CV over 2000–2024 for all model
selection, 1965–1978 pre-satellite hold-out, 2025 as operational test.
Held-out periods evaluated exactly once, on day 11.

### Schedule
- **Week 1:** read papers/brainstorm; RMSE+Bias eval of individual models;
  generate long hindcasts for in-house models (Pangu-S2S, SFNO-S2S); stand up
  ensembling infra + initial Method 0 blending; (stretch) start MoE setup.
- **Week 2:** finish + train MoE model; finalize analysis framework, compare
  methods; (stretch) extend to probabilistic forecasts.

### Code layout / infra reuse
- **Dataset/normalizer/dataloader:** reuse the existing `ClimateZarrDataset` /
  `ClimateNormalizer` pipeline as-is — not a hackathon-specific rewrite.
- **Training code:** will differ per method (Method 0 trains only the gate,
  Method 1 also fine-tunes the expert models, Method 2 trains a perturbation
  generator) — not a shared `train.py`.
- **Storage/data root:** config-driven via Hydra, not a separate env var —
  cluster/dataset paths for the hindcast archives are specified in the Hydra
  config (`conf/`), same pattern as `cfg.dataset.zarr_path` in the core fork.
- Where the Methods 0/1/2 code itself (gate, fine-tuning loop, perturbation
  generator) lives under `examples/weather/ai_rossbypalooza/` — still TODO.

### Roles
- **Alex** (`awikner`) — expert manager + hindcast generator
- **Zilu** (`ZiluM`) — Method 2
- **Daniel** (`danielboscu`) — Method 1
- **Seung-yoon** (`syback`) — analysis + NWP model data acquisition
- **Linqiang** (`helq1116`) — base ensembling models + analysis
- **Vadim** (`VadimLimousin`) — Methods 0/1/2 as needed

### Related reading
- MoWE (Chakraborty et al. 2025) — base architecture for Methods 0/1.
- Physical realism of blended models — PhysMericsWeather (Kasteleyn et al. 2026): arXiv:2606.10642
- Method 2 base paper — Optimal perturbation (Vonich & Hakim, 2026): https://journals.ametsoc.org/view/journals/aies/5/3/AIES-D-26-0009.1.xml
