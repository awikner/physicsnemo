# Data acquisition & conversion

The recipe reads **Zarr** stores (one per split) plus small normalization /
climatology stores. This guide covers (1) how the configs find data, (2) the
already-converted stores on Delta, and (3) converting raw source files yourself.

## 1. How the configs find data (`AI_ROSSBY_DATA`)

Dataset configs (`conf/dataset/*.yaml`) resolve their paths under environment
variables with a **shared-Delta fallback**, so:

- On **Delta** (as a `delta_bdiu` member) they work with no setup — the paths
  fall back to the group's shared location.
- **Elsewhere**, set the env var(s) once:

```bash
export AI_ROSSBY_DATA=/my/physicsnemo-zarr          # converted Zarr stores (main)
# advanced / only for specific configs:
export AI_ROSSBY_PLASIM_STATS=/my/plasim/sigma_data # PLASIM .nc mean/std
export AI_ROSSBY_AMIP_STATS=/my/amip/h5             # AMIP .nc mean/std
export AI_ROSSBY_AMIP_CKPT=/my/amip-checkpoints     # amip diffusion checkpoints
```

`${AI_ROSSBY_DATA}` is expected to contain per-source subdirectories:
`e3sm/`, `plasim/`, `amip/`, `era5_sfno_s2s/`, `era5/`. Override any single
path on the CLI, e.g. `dataset.zarr_path=/my/store.zarr`.

## Getting the data onto a cluster (registry + Globus)

The converted Zarr lives on several clusters; the master copy is on Derecho
scratch, a second copy on Stampede3 (see `docs/dev/phase11_implementation_plan.md`).
`hpc/data_registry.yaml` records **which datasets exist and on which clusters**,
and two tools drive it:

```bash
# What exists and where (+ gaps / at-risk single copies):
python tools/data/registry.py show
python tools/data/registry.py check

# Before training on a cluster, pull a dataset there if it isn't already:
python tools/data/sync_dataset.py e3sm --to stampede3 --dry-run   # preview the Globus plan
python tools/data/sync_dataset.py e3sm --to stampede3             # transfer (needs `globus` CLI + auth)

# Ship a dataset's *raw* h5 to a cluster so it can be converted there:
python tools/data/sync_dataset.py era5 --to stampede3 --stage-raw

# After a scratch purge, restore what should be on a cluster from a peer:
python tools/data/sync_dataset.py --rehydrate derecho
```

`sync_dataset.py` transfers into that cluster's `AI_ROSSBY_DATA` root, so a
subsequent run finds the data with no path edits. After a transfer, record it
with `registry.py scan <cluster> --write`. (One-time: fill the
`globus_collection` UUIDs in `hpc/data_registry.yaml`.)

To bring up the datasets that still need converting (ERA5, AMIP, PLASIM-plev,
E3SM — all staged to Stampede3 and converted on the `spr` queue, then replicated
to the Derecho master), follow the step-by-step
[`hpc/phase11_globus_runbook.md`](../../../hpc/phase11_globus_runbook.md).

## 2. Already-converted stores on Delta

Under `/work/hdd/bdiu/awikner/physicsnemo-zarr/` (group-readable), so you can
point a run at them without re-converting:

| Source | Store(s) | Notes |
|---|---|---|
| E3SM | `e3sm/2041.zarr` (train), `e3sm/2045.zarr` (val), `e3sm/normalization_2015-2050.zarr`; years `2041`–`2049` + `climatology_bias.zarr` also present | SSP245 AMIP run, 18 hybrid-pressure levels, 180×360 |
| PLASIM | `plasim/12.zarr`, `plasim/13.zarr`; sigma stats under `/work/nvme/bdiu/awikner/PLASIM/.../sigma_data/` | sim52, 64×128 |
| ERA5 (S2S) | `era5_sfno_s2s/1981.zarr`, `era5/normalization_pangu_s2s_*.zarr` | PanguWeather S2S feature set |
| AMIP | `amip/1981.zarr`, stats under `/work/hdd/bdiu/awikner/AMIP/h5/` | diffusion (experimental) |

> **Where does the *raw* source data come from?** The raw per-timestep HDF5
> archives (E3SM simulation output, PLASIM 2100-year runs, ERA5 PanguWeather
> archives, AMIP) are produced/curated outside this repo. If you need the raw
> inputs (not the converted Zarr), ask the data owner — **TODO: name the
> contact / archive location for each source** (E3SM run, PLASIM sim52, ERA5,
> AMIP). The converters below assume you already have the source HDF5.

## 3. Converting raw source → Zarr

Each source has a converter under `tools/data/<source>/`. See
[`tools/README.md`](../../../tools/README.md) for the full index. Typical flow
(E3SM shown; ERA5/PLASIM/AMIP are analogous):

```bash
# 1. One year of per-timestep HDF5 → one Zarr store
python tools/data/e3sm/pangu_h5_to_zarr.py \
    --input-dir /path/to/E3SM/.../h5/sigma_data \
    --year 2041 --sample-range 0 1460 \
    --output $AI_ROSSBY_DATA/e3sm/2041.zarr

# 2. Normalization statistics (mean/std) → a small Zarr
python tools/data/e3sm/build_normalization_zarr.py \
    --source-dir /path/to/E3SM/.../h5/sigma_data \
    --output $AI_ROSSBY_DATA/e3sm/normalization_2015-2050.zarr

# 3. (optional) Climatology + bias store, for ACC in validation
python tools/data/e3sm/build_climatology_zarr.py \
    --source-dir /path/to/E3SM/.../h5/sigma_data \
    --output $AI_ROSSBY_DATA/e3sm/climatology_bias.zarr
```

The channel groups the converter writes (e.g. `PANGU_E3SM_CHANNELS` in
`tools/data/e3sm/pangu_h5_to_zarr.py`) **define** the variable ordering your
model config must match — treat the converter as the source of truth.

For large conversions on a cluster, the repo ships CPU-job skills
(`delta-cpu-job`, `derecho-cpu-job`, …) and example sbatch scripts in
`hpc/scripts/convert_*.sbatch` (edit the paths/account for your site).
