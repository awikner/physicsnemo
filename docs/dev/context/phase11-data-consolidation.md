<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 11 — data conversion + multi-cluster consolidation

> Status as of 2026-07-20. Full plan: [`../phase11_implementation_plan.md`](../phase11_implementation_plan.md);
> runbook: [`../../../hpc/phase11_globus_runbook.md`](../../../hpc/phase11_globus_runbook.md);
> registry: [`../../../hpc/data_registry.yaml`](../../../hpc/data_registry.yaml).
> See also [derecho-retire-rehome-to-delta](derecho-retire-rehome-to-delta.md)
> for the topology change that supersedes "master = Derecho".

**Phase 11 complete (2026-07-16).** All four remaining datasets converted on
Stampede3 `spr`, consolidated on the Derecho master + Stampede3 second copy,
registry verified: `registry.py check` → "all datasets: complete and durably
stored." Key commit `d92b17a0` on `ai-rossby`.

**Final coverage** (converted on Stampede3): ERA5 1979-2024 (46 yrs + norm),
AMIP **1978-2022** (45 yrs — 2023-24 raw lack `global_mean_co2`, dropped per
user), PLASIM-plev 12-104 (93), E3SM 2015-2049 (35 + norm + climatology_bias).

## Key gotchas learned

- **Globus identity mapping.** The TACC "Stampede3 GCS v5.4 Filesystems"
  collection `1e9ddd41…` maps the **uchicago.edu** identity to local user
  `awikner`; Delta needs **access-ci.org**. Fix a timed-out high-assurance
  session with `~/gcli/bin/globus session update <domain>`. globus-cli is
  installed at `~/gcli` on Stampede3 (drives all cluster-to-cluster transfers).
- **E3SM full archive location.** The complete 35-yr E3SM archive is on
  **Stampede3** at `/scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101`
  (collaborator jwan4, group-readable). The Derecho/Delta `.../h5/sigma_data`
  dirs hold only a 2-year subset (2015, 2045) — do NOT use them.
- **Tar-bundle for cross-cluster zarr replication** (`hpc/scripts/replicate_tar.sh`).
  Converted stores are 16k–61k tiny per-timestep chunk files each; per-file Globus
  crawls (~70 MB/s, stalls near the end on data-channel faults). pack → send →
  unpack is ~5× faster. Run pack/unpack on compute nodes (Derecho `develop` queue
  resolves to `cpudev`). **Derecho scratch is inode-limited** (see below), so
  large untars there can hit "Disk quota exceeded."
- **E3SM conversion verified bit-exact vs source h5** (worst |diff|=0 across all
  vars/levels, incl. a mid-series timestep). noleap calendar → 1460 steps/yr. The
  4 static land fields (TOPO, PCT_*, PFTDATA_MASK) are ~62.6% NaN (ocean) by
  design; RELHUM>100 is source-origin, not a conversion artifact. ERA5/AMIP use a
  REAL calendar (leap yrs = 1464 steps).

## Derecho scratch is at its file-count (inode) quota — key constraint

`gladequota` shows `/glade/derecho/scratch/awikner` at a **26,214,400-file hard
limit** (space only ~63% of 200 TiB — inodes are the binding constraint, since
zarr stores are millions of tiny chunk files). Adding PLASIM-sigma (~1.5M files)
overran it (untar failed "Disk quota exceeded", left partial stores). So the
Derecho master cannot grow without freeing inodes / a quota bump. **Stampede3
scratch has NO file-count limit** (`lfs quota` shows limit 0) → the place for
inode-heavy zarr. This is why PLASIM-sigma's second copy went to Stampede3, and
ultimately why Derecho is being retired as master.

## Post-Phase-11 follow-ups (all resolved)

- **ERA5 persistent copy on Delta** `/work/hdd/.../physicsnemo-zarr/era5` (46 yrs
  + norm; file counts verified). Commit `4373c47f`.
- **PLASIM-sigma → Stampede3** (Derecho inode-full): Delta (persistent) +
  Stampede3 copies, verified vs Delta source. Commit `0cd69f63`.
- **Tar + raw staging cleaned up**: `zarr-tars/` removed on all clusters (~7 TB);
  Stampede3 `raw/` staging deleted (5.7 TB) — durable raw sources intact (Delta
  era5/amip + Z200 stats, Derecho plev, jwan4 e3sm).

## ERA5 normalization fix — 200 hPa (2026-07-17 / 07-20)

The full-archive `era5` Zarr is 18 pressure levels (incl 200 hPa) but
`normalization_pangu_s2s.zarr` was built from the 17-level
`pangu_s2s_1979-2018_mean.nc` (the SFNO_S2S_0003 ablation drops 200). Fixed by
rebuilding from the `pangu_s2s_Z200` variant (the **complete** 18-level stats —
`_Z200.nc`, all 5 upper-air vars × 18 levels, 17 shared levels bit-identical to
the 17-lvl file). Both the **combined** `normalization_pangu_s2s.zarr` and the
**separate** `_mean.zarr`/`_std.zarr` rebuilt to 18-lvl and deployed to all 3
clusters (old kept as `.17lvl_bak`). Commits `a1629106` + follow-up.

- The normalizer `ClimateNormalizer` (= `PlasimNormalizer`,
  `physicsnemo/experimental/datapipes/climate/transforms.py`) matches levels **by
  value** → an 18-vs-17 mismatch RAISES loudly (`"No near match for level 200"`),
  never silently misaligns. So no model trained mis-normalized.
- E3SM/AMIP/PLASIM norms were already clean. The **benchmark** path
  (`era5_sfno_s2s` 17-lvl dataset + 17-lvl model configs) is self-consistent.
  Model configs stay 17-lvl per user — to *train on* 200 hPa with the full
  archive, `cfg.model.levels` must also go to 18.
- Build-script caveat: use plain xarray — do NOT `import physicsnemo` on a login
  node (CUDA/Warp init core-dumps). The `_nearest_indices` value-match
  (`tol = max(1e-3, 1e-3·|level|)`) is trivial to inline.

## Node-local staging (not MPI)

`examples/weather/ai_rossby/data_staging.py` is a thread-pool `sendfile` copier,
**not** MPI (MPI/`dcp` deliberately rejected — the copy is write-bound ~2.2 GB/s,
so MPI adds no throughput). Done + measured on **Delta only**; the code is generic
(handles SLURM + PBS job IDs) but unvalidated on DeltaAI/Derecho/Stampede3/
Midway3/DSI, and has no tests. Config flags in `conf/dataset/e3sm.yaml`:
`stage_to_local` (default `False`), `stage_dir`, `stage_num_workers` (256).

## Tooling map

- `tools/data/registry.py` (show/check/scan), `tools/data/sync_dataset.py`
  (`--stage-raw`, `--rehydrate`), `hpc/data_registry.yaml`.
- `hpc/scripts/`: `convert_*_stampede3.sbatch`, `convert_*_derecho.pbs`,
  `replicate_tar.sh`, `run_phase11_stampede3.sh`.
- `hpc/phase11_globus_runbook.md`, `docs/dev/phase11_implementation_plan.md`.
