<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 11 — Stampede3 conversion runbook

**All remaining conversion runs on Stampede3 (`spr`).** Derecho stays the master
(final Zarr store) but no longer converts. So every raw archive is staged to
Stampede3 via Globus, converted on the `spr` queue, then the finished Zarr is
replicated back to the Derecho master. Stampede3 keeps its copy as the second
replica.

| Dataset | Raw source → Stampede3 | Raw size | Target years |
|---|---|---|---|
| ERA5  | Delta `bgong1/data/h5data` | (full ERA5 archive) | 1979-2024 |
| AMIP  | Delta `awikner/AMIP/h5` | (full AMIP archive) | 1978-2024 |
| PLASIM-plev | Derecho `…/sim52/h5/plev_data` | **565 G / 184k files** ⚠ | 12-104 |
| E3SM  | Derecho archive root | 130 G + 4.3 G boundary_data | 2015-2049 |

Notes before you start:
- **plev is 184k tiny `.h5` files (565 G).** That per-file overhead makes it the
  slowest transfer by far. The converter only builds years 12-104 by default
  (`YEARS_LO/YEARS_HI` in the sbatch), but the recursive stage still copies all
  of 7-132 — fine on Stampede3's ~100 TB, just budget the time.
- **E3SM** stages the whole archive root; `boundary_data` (4.3 G, unused) tags
  along. `climatology.nc` + the normalization `data_*_mean.nc` live *inside*
  `h5/sigma_data/`, so they come with the main transfer.
- Stampede3's own `ERA5/h5` is a *different* (non-Pangu) archive missing
  `mean_sea_level_pressure` — do **not** convert it; ERA5 must come from Delta.

The collection UUIDs and paths below are already in
[`hpc/data_registry.yaml`](data_registry.yaml); `sync_dataset.py` emits every
command. Re-run any with `--dry-run` to preview.

---

## 0. Prereq: `globus-cli` authenticated

The UUIDs are filled in (Delta `7e936164…`, Derecho/GLADE `d33b3614…`,
Stampede3/TACC `1e9ddd41…`). Authenticate once from wherever you run the CLI
(your laptop: `pipx install globus-cli && globus login`), or drive the same
transfers from the [Globus web app](https://app.globus.org/file-manager).
Transfers run server-side — no `ssh`/`kinit` needed.

## 1. Stage all four raw archives → Stampede3

```bash
cd tools/data
python sync_dataset.py era5        --to stampede3 --stage-raw   # Delta   → $SCRATCH/raw/era5/h5data
python sync_dataset.py amip        --to stampede3 --stage-raw   # Delta   → $SCRATCH/raw/amip
python sync_dataset.py plasim_plev --to stampede3 --stage-raw   # Derecho → $SCRATCH/raw/plasim_plev/plev_data
python sync_dataset.py e3sm        --to stampede3 --stage-raw   # Derecho → $SCRATCH/raw/e3sm
```

Each prints/submits one recursive `globus transfer` (checksum sync, so re-runs
are idempotent). Destinations are exactly what the converters read (`RAW=`/
`H5_DIR=` in the sbatch scripts). Start plev first — it's the long pole. Wait for
**SUCCEEDED** (`globus task list`) per dataset before converting it.

## 2. Convert on Stampede3 (`spr`)

```bash
# on a Stampede3 login node, repo at $WORK/physicsnemo (ai-rossby)
sbatch hpc/scripts/convert_era5_stampede3.sbatch          # 1979-2024 + normalization
sbatch hpc/scripts/convert_amip_stampede3.sbatch          # 1978-2024
sbatch hpc/scripts/convert_plasim_plev_stampede3.sbatch   # 12-104 (YEARS_LO/HI to widen)
sbatch hpc/scripts/convert_e3sm_stampede3.sbatch          # 2015-2049 + norm + clim/bias
```

Each takes a whole `spr` node (`-N1 -n1 -c112`) and sizes the H5 reader pool to
all 112 cores. `skip-if-exists`, so a re-submit resumes. Output →
`$SCRATCH/physicsnemo-zarr/<dataset>/<year>.zarr`.

Sanity-check the first year before trusting a full run (ERA5 especially — this
is the missing-variable check):

```bash
python - <<'PY'
import xarray as xr
ds = xr.open_zarr("$SCRATCH/physicsnemo-zarr/era5/1979.zarr")
print(ds.dims); print(list(ds.data_vars))       # expect mean_sea_level_pressure present
PY
```

## 3. Record the copies, then replicate to the Derecho master

```bash
cd tools/data
python registry.py scan stampede3 --write   # records every converted year now on Stampede3

# Replicate all four Stampede3 → Derecho master:
python sync_dataset.py era5        --to derecho
python sync_dataset.py amip        --to derecho
python sync_dataset.py plasim_plev --to derecho
python sync_dataset.py e3sm        --to derecho

python registry.py scan derecho --write      # records the master copies
```

Now each year-store lives on **both** Stampede3 (second copy) and Derecho
(master).

### Stats stores (normalization / climatology-bias)

`sync_dataset.py` moves per-year stores; the small named stat stores travel in a
one-off `globus transfer <stampede3>:<root> <derecho>:<root> --batch -` (same
roots as Step 3), feeding:

```
era5/normalization_pangu_s2s.zarr        era5/normalization_pangu_s2s.zarr        --recursive
e3sm/normalization_2015-2050.zarr        e3sm/normalization_2015-2050.zarr        --recursive
e3sm/climatology_bias.zarr               e3sm/climatology_bias.zarr               --recursive
```

(AMIP normalization stays `.nc`; the recipe reads it directly.)

## 4. Verify

```bash
python registry.py check      # every dataset "ok"; no MISSING, no single-copy AT-RISK
```

Green `check` = all six datasets on the master **and** a second copy, and any
purge is recoverable with `sync_dataset.py --rehydrate <cluster>`.

---

## Rehydrate after a purge (ongoing)

```bash
python tools/data/sync_dataset.py --rehydrate derecho     # or: stampede3
```

Reads the registry, scans what's actually present, re-pulls only the missing
year-stores from a peer that still has them.
