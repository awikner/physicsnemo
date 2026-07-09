---
marp: true
title: ai-rossby — onboarding
paginate: true
theme: default
---

<!--
Onboarding deck for new group members. Sparse, presenter-driven (~18 slides).
Build to PDF:  make -C docs/onboarding onboarding.pdf
(needs marp-cli:  npx @marp-team/marp-cli   or   npm i -g @marp-team/marp-cli)
Speaker detail lives in the HTML-comment presenter notes on each slide.
-->

# ai-rossby

### Deep-learning weather & climate emulators on PhysicsNeMo

Group onboarding

<!-- notes: One-liner — we train ML weather/climate emulators. This deck: what it is, then clone → train → evaluate. Everything here links to the repo docs; slides are the map, docs are the detail. -->

---

## Why

- Train **fast surrogate models** for weather & climate (PLASIM, ERA5, E3SM, AMIP)
- One codebase, many model families — shared data pipeline, training loop, evaluation
- Built on **NVIDIA PhysicsNeMo** (we maintain a fork)

<!-- notes: The goal is a unified recipe: swap a config, get a different model/dataset, same infra. Emulators run orders of magnitude faster than the physical simulators they learn from. -->

---

## Lineage

- **Fork of NVIDIA PhysicsNeMo** (Apache-2.0) — the framework
- Deterministic models derived from **PanguWeather v2.0** (→ makani / FourCastNet → Pangu-Weather)
- Diffusion models derived from the **amip** codebase *(experimental)*
- See [`NOTICE`](../../NOTICE) for provenance & licensing

<!-- notes: We didn't build from scratch — we ported. That's why some code is "vendored" and why the NOTICE matters. Diffusion is a collaborator's work, still experimental. -->

---

## Supported models

| Family | What | Status |
|---|---|---|
| PanguPlasim / Legacy | Pangu-style transformer | ✅ |
| SfnoPlasim / SFNO-E3SM | Spherical FNO | ✅ |
| AMIP diffusion (SI/EDM/…) | latent diffusion | ⚠️ experimental |

**Start with the deterministic Pangu/SFNO recipes.**

<!-- notes: For onboarding we focus on the supported path. Diffusion works but isn't the blessed path yet. -->

---

## The data: Zarr stores

- Recipe reads **Zarr** — one store per split + small normalization/climatology stores
- Sources: **E3SM, PLASIM, ERA5, AMIP** (raw per-timestep HDF5 → Zarr via `tools/data/`)
- On **Delta** the converted stores already exist (group-readable)

<!-- notes: You rarely convert data yourself on Delta — it's already there. Elsewhere you convert once with the tools/. The converter defines the channel ordering the model expects. -->

---

## Data: pointing at it

```bash
# On Delta: works with no setup (configs fall back to the shared location)
# Elsewhere: set one env var
export AI_ROSSBY_DATA=/my/physicsnemo-zarr
```

Details + the converted-store table: [`examples/weather/ai_rossby/DATA.md`](../../examples/weather/ai_rossby/DATA.md)

<!-- notes: The env-var-with-fallback scheme is the one thing to remember. Everything under AI_ROSSBY_DATA/ has e3sm/, plasim/, etc. -->

---

## Repo tour

- `examples/weather/ai_rossby/` — the recipe (**train / inference / validate**)
- `examples/weather/ai_rossby/conf/` — Hydra configs (model / dataset / training / loss / validation)
- `tools/` — data conversion + checkpoint translation
- `hpc/` — per-cluster install & run docs
- `docs/dev/` — internal design/plan history

<!-- notes: Two dirs matter day-to-day: the recipe and its conf/. The rest is support. -->

---

## Configs are composable (Hydra)

Pick one YAML per group, override any leaf on the CLI:

```bash
python train.py \
  model=sfno_e3sm dataset=e3sm training=sfno_plasim \
  loss=raw_l2 validation=off \
  training.max_epochs=100 dataset.batch_size=8
```

<!-- notes: This is the whole UX. No editing files to change a run — compose groups + dotted overrides. -->

---

## Set up the environment

- Portable recipe: [`hpc/install.md`](../../hpc/install.md)
- Per-cluster: `hpc/delta.md`, `hpc/deltaai.md`, `hpc/derecho.md`, …
- `uv` + the system PyTorch module; a per-cluster venv

<!-- notes: Follow the per-cluster doc for your machine. The env is uv-managed. On DeltaAI the venv is aarch64 and separate. -->

---

## Train — single GPU

```bash
cd examples/weather/ai_rossby
python train.py \
  model=sfno_e3sm dataset=e3sm training=sfno_plasim \
  run_name=my_first_run
```

Checkpoints → `outputs/my_first_run/checkpoints/`

<!-- notes: This is the "does it work" command. Point them here on day one. -->

---

## Train — multi-GPU

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  model=sfno_e3sm dataset=e3sm training=sfno_plasim \
  run_name=my_first_ddp
```

⚠️ SFNO+DDP needs **torch<2.11** (handled by the pinned env)

<!-- notes: DDP is automatic when world_size>1. The torch pin + all-rank wandb are handled for you; just know they exist. -->

---

## Monitor

- **wandb** (default, offline — local `./wandb/`, no login)
- `train/*` loss components · `valid/*` val_loss / RMSE / ACC
- `wandb sync ...` to upload; `wandb.mode=online` to stream

<!-- notes: Offline by default so it works with no network. Sync later if you want the dashboard. -->

---

## Evaluate

```bash
python inference.py model=sfno_e3sm dataset=e3sm \
  +inference.checkpoint_dir=... +inference.output_path=preds.nc

python validate_cli.py dataset=e3sm \
  +validation_cli.predictions=preds.nc \
  +validation_cli.reference_zarr=$AI_ROSSBY_DATA/e3sm/2045.zarr
```

<!-- notes: Roll out to a file, then score it. Produces an RMSE/ACC scorecard. -->

---

## Clusters & scheduling

- **Delta / DeltaAI** (NCSA), **Derecho** (NCAR), **Stampede3** (TACC), **Midway3** & **DSI** (UChicago)
- SLURM/PBS smoke-test skills per cluster + example sbatch in `hpc/scripts/`
- Use **your** account/allocation (edit the sbatch headers)

<!-- notes: Six clusters are wired up. The sbatch scripts are templates — swap in your username/allocation. -->

---

## Where to get help

- **Start:** top-level [`README.md`](../../README.md) → recipe [`README.md`](../../examples/weather/ai_rossby/README.md)
- Data: [`DATA.md`](../../examples/weather/ai_rossby/DATA.md) · Porting: `PANGUWEATHER_MIGRATION.md`
- Tools: [`tools/README.md`](../../tools/README.md) · History: `docs/dev/`
- **Ask:** _<add group contact / channel here>_

<!-- notes: The docs cover the mechanics; ping the group channel for the rest. Fill in the contact before presenting. -->

---

## First task

1. Set up your cluster env (`hpc/<cluster>.md`)
2. Run the single-GPU SFNO-E3SM train command
3. Watch `train/loss` fall
4. Roll out + score with `inference.py` → `validate_cli.py`

**Welcome aboard.**

<!-- notes: Concrete first-day goal: one training run + one evaluation. If that works, the environment and data are good. -->
