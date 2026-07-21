# CLAUDE.md — ai-rossby fork project brief

Orientation for working in this repo. This is the **ai-rossby** fork of NVIDIA
PhysicsNeMo, maintained by the UChicago group for climate emulation (PLASIM,
ERA5, E3SM, AMIP) with SFNO / Pangu-Weather models.

- **Active branch:** `ai-rossby` (PRs usually target the fork, not upstream `main`).
- **Recipe:** `examples/weather/ai_rossby/` (`train.py`, `conf/`, `DATA.md`,
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

## Gotchas that will bite you (details in `docs/dev/context/`)

- **Multi-GPU SFNO:** `torch < 2.11` (2.11/2.12 break DDP); init wandb on *every*
  rank; `uv sync` must include `--extra sfno-extras --extra utils-extras --extra
  datapipes-extras` or it silently prunes SFNO/zarr deps.
- **DeltaAI (GH200):** the inherited conda `wandb` is broken — install wandb into
  `.venv-deltaai`; `torchrun` isn't on the venv PATH.
- **Globus high-assurance sessions time out** — refresh with `globus session
  update <domain>` (Delta ↔ `access-ci.org`, TACC ↔ `uchicago.edu`).
- **Don't `import physicsnemo` on a login node** for small scripts — CUDA/Warp
  init can core-dump; use plain xarray/numpy.

## Conventions

- Commit messages end with the `Co-Authored-By` trailer; branch before committing
  on `main`; commit/push only when asked.
- CI header check requires the NVIDIA SPDX copyright line; add the UChicago line
  alongside it (see existing files).
