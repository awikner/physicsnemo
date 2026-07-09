# Phase 11 — Complete the data conversion + multi-cluster data catalog

Status: **planned** · Author: Claude (data inventory + plan) · Created: 2026-07-09

## Goal

Three deliverables:

1. **Convert all remaining raw data** to the unified ai-rossby Zarr format, so
   every supported model can train on its full multi-year archive (not just the
   single-year benchmark stores). **Conversion runs on Stampede3** (`spr` CPU
   queue), not Delta — Delta can't hold the full converted set.
2. **Consolidate the complete dataset on Derecho scratch** as the master copy
   (`/glade/derecho/scratch/awikner/physicsnemo-zarr/`), populated by copying the
   converted Zarr from Stampede3 (which keeps its own copy — the second replica).
3. **Stand up a multi-location data registry + Globus tooling** that records
   **every** location of each dataset (peer clusters *and* the raw source), so
   that if a copy is lost (e.g. a scratch purge) it can be re-transferred — and
   so any cluster can pull the data it needs before training.

---

## 0. Constraints & decisions

| # | Item | Resolution |
|---|---|---|
| **Delta storage** | Delta `/work/hdd` (bdiu quota) **cannot hold the full converted archive** on top of the raw H5. | ⇒ Convert on **Stampede3** (`spr` queue, `TG-ATM170020`): ship raw there, convert, copy the Zarr to Derecho (master). |
| **Stampede3 scratch** | `$SCRATCH` ≈ **100 TB** — comfortably fits the full raw + converted archive. | Delete each year's raw after its Zarr is written as hygiene, but no tight source-by-source sequencing is needed. 90-day purge (see D4). |
| **D1 (resolved)** | **Master location.** | **Derecho scratch.** `/glade/campaign` lacks the quota for this volume, so campaign is out. Accept the **60-day purge**; mitigate with the multi-location registry (11c) + re-hydration (11d) — never rely on a single copy. |
| **D2 (resolved)** | E3SM upper bound. | **2015–2049** (2050 excluded). |
| **D3 (resolved)** | PLASIM-plev raw source. | **Derecho** `/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/plev_data` — per-year **`.h5`**, complete (same format as the already-converted 7–78). The existing converter + `configs/sim52_plev_full.yaml` handles it directly; **no enrichment / no sigma merge.** Convert in-place on Derecho or ship to Stampede3 (see 11a routing note). |
| **D4 (resolved)** | Conversion + second copy. | **Stampede3** is both the **conversion cluster** (`spr`) and the **second copy** — converted Zarr stays on its `$SCRATCH` and is copied to Derecho (master). Both scratches are volatile but on independent clocks (**Derecho 60-day, Stampede3 90-day**); Delta `/work/hdd` raw is the durable fallback. |

---

## Data gap (target vs. converted)

Authoritative targets (A. Wikner, 2026-07-09):

| Source | Target years | Converted today | **To convert** | Raw location |
|---|---|---|---|---|
| **E3SM** | 2015–2049 (35) | 2041–2049 (9) | **2015–2040** (26) | Delta `…/E3SM/…/h5/sigma_data` |
| **ERA5** | 1979–2024 (46) | 1981 (1) | **45 yrs** | Delta `bgong1/data/h5data` |
| **AMIP** | 1978–2024 (47) | 1981 (1) | **46 yrs** | Delta `AMIP/h5` (`{year}_{idx}.h5`) |
| **PLASIM sigma** | *(reference)* 12–104 (93) | 12–104 (93) | — | Delta/Stampede3 `…/sim52/h5/sigma_data` |
| **PLASIM plev** | = sigma → 12–104 | 7–78 (72) | **79–104** (26) | Derecho `…/sim52/h5/plev_data` (`.h5`, complete) |

Plus **PLASIM & AMIP normalization stats `.nc → .zarr`** (E3SM & ERA5 are already
Zarr). Rough scale: a few TB of new Zarr; the raw H5 to move is comparable or
larger — plan multi-hour Globus transfers.

---

## Sub-phase 11a — Convert on Stampede3 (`spr`), copy to Derecho

> **Execution status.** Two of the four remaining sources turned out to have
> their raw already on **Derecho**, so they convert there (128-core PBS,
> validated) straight into the master — no Globus: **PLASIM-plev** `12-104`
> (job `6681544`) and **E3SM** `2015-2049` (job `6681752`). Only **ERA5** and
> **AMIP** genuinely need Globus (correct raw only on Delta). The paste-and-go
> checklist for those two is **[`hpc/phase11_globus_runbook.md`](../../hpc/phase11_globus_runbook.md)**.

Conversion runs on **Stampede3** (`-p spr -A TG-ATM170020`; repo/venv on `$WORK`
per `hpc/stampede3.md`), writing Zarr to `$SCRATCH`; finished stores are copied
to Derecho (11b/11d) and kept on Stampede3 as the second replica.

1. **Ship raw H5 → Stampede3 `$SCRATCH/raw/<source>/`** via Globus (11d):
   - **E3SM / ERA5 / AMIP** raw from Delta `/work/hdd` (gap table).
   - **PLASIM-plev** raw `.h5` is on **Derecho** already (years 7–132), so plev
     converts **in-place on Derecho** — no Globus round-trip. Script:
     `hpc/scripts/convert_plasim_plev_derecho.pbs` (PBS, `-A UCHI0014`, `main`
     queue, 128 cores; `YEARS_LO..YEARS_HI` default 12–104 to match sigma).
     Output lands directly on the Derecho master; copy to Stampede3 for the
     second copy. **[in progress — job running on Derecho]**
2. **Port the converters to Stampede3 SLURM.** The existing
   `hpc/scripts/convert_*_full_archive.sbatch` (per-year loop + skip-if-exists)
   become `convert_*_stampede3.sbatch` (`-p spr -A TG-ATM170020`, `$SCRATCH`
   paths). Target ranges:
   - **E3SM** `seq 2015 2049` (→ 2015–2040), **ERA5** `seq 1979 2024`,
     **AMIP** `seq 1978 2024`.
   - **PLASIM plev** — existing converter + `configs/sim52_plev_full.yaml`;
     dynamic year-discovery + skip-if-exists picks up 79–104. Complete `.h5`,
     **no sigma merge / no precip computation**. (Decide plev's extra 7–11 vs.
     backfilling sigma to 7.)
   - **Stats → Zarr** via `build_normalization_zarr.py` for PLASIM & AMIP; repoint
     `conf/dataset/{plasim_*,amip_1981}` `mean_path`/`std_path` to `.zarr` (keep
     the `$AI_ROSSBY_*` env indirection from Phase 10).
3. **Scratch hygiene:** delete each year's raw after its Zarr is written and copy
   finished sources to Derecho as they complete. With ~100 TB scratch (§0) there's
   ample headroom — strict source-by-source sequencing isn't required.
4. Parallelize with SLURM job arrays (one year per task); each converter sizes its
   pool from `SLURM_CPUS_PER_TASK`.

## Sub-phase 11b — Master on Derecho scratch (via copy from Stampede3)

- **Copy each converted source Stampede3 → Derecho** via Globus (11d) as 11a
  finishes it, into
  `/glade/derecho/scratch/awikner/physicsnemo-zarr/{e3sm,era5,amip,plasim,plasim_plev}/`.
- **Migrate the existing converted stores** (E3SM 2041–2049, PLASIM 12–104,
  plev 7–78, ERA5-S2S 1981, AMIP 1981, all norm/clim stores) from Delta → Derecho
  (and → Stampede3 for the second copy). Result: the **complete archive on
  Derecho scratch = master**; the Stampede3 copy is the second replica.
- Set `AI_ROSSBY_DATA` to the `physicsnemo-zarr` root on each cluster
  (`/glade/derecho/scratch/...` on Derecho, `$SCRATCH/physicsnemo-zarr` on Stampede3).
- **Resilience (D4):** master on Derecho scratch + second copy on Stampede3
  `$SCRATCH`. Both are *volatile* (60-day purge) but on **independent clocks**,
  so a purge on one is re-pulled from the other via `--rehydrate` (11d). A light
  `touch`/read cron (or the training jobs) keeps active data warm. The durable
  last resort is the **raw H5 on Delta `/work/hdd`** (persistent) for E3SM /
  ERA5 / AMIP / PLASIM-sigma. The **plev raw `.h5` is on Derecho scratch
  (volatile)** but self-contained; if plev durability matters, copy the plev
  `.h5` (or the Zarr) to a persistent FS (Delta `/work/hdd` or `/glade/work`).

## Sub-phase 11c — Multi-location data registry (the "ongoing list")

`hpc/data_registry.yaml`, committed to the repo — the source of truth for *what
exists* and *every place it lives*, so a lost copy is always re-transferable.

```yaml
clusters:
  delta:    { data_root: /work/hdd/bdiu/awikner/physicsnemo-zarr, globus_collection: <UUID> }
  deltaai:  { alias_of: delta }        # shares Delta /work — physically the same data
  derecho:  { data_root: /glade/derecho/scratch/awikner/physicsnemo-zarr, globus_collection: <NCAR-UUID>, volatile: true }  # master; scratch 60-day purge
  stampede3: { data_root: /scratch/09979/awikner/physicsnemo-zarr, globus_collection: <TACC-UUID>, volatile: true }        # second copy (D4); scratch purge
  # midway3, dsi …

datasets:
  e3sm:
    subdir: e3sm
    years: "2015-2049"
    stats: [normalization_2015-2050.zarr, climatology_bias.zarr]
    raw_source: { cluster: delta, path: /work/hdd/bdiu/awikner/E3SM/.../h5/sigma_data }  # ultimate fallback: re-convert
    copies:                      # EVERY current Zarr location (for re-transfer if one is lost)
      - { cluster: derecho,   years: "2015-2049" }   # master
      - { cluster: stampede3, years: "2015-2049" }   # second copy
      - { cluster: delta,     years: "2041-2049" }   # partial (whatever fits on Delta)
  era5:   { subdir: era5,  years: "1979-2024", raw_source: {...}, copies: [ {cluster: derecho, years: "1979-2024"}, {cluster: stampede3, years: "1979-2024"} ] }
  amip:   { subdir: amip,  years: "1978-2024", ... }
  plasim: { subdir: plasim, years: "12-104", ... }
  plasim_plev: { subdir: plasim_plev, years: "12-104",
                 raw_source: { cluster: derecho, path: /glade/derecho/scratch/awikner/PLASIM/.../h5/plev_data },
                 copies: [ {cluster: derecho, years: "12-104"}, {cluster: stampede3, years: "12-104"} ] }
```

- **`copies:` is a list** — the multi-location requirement. A dataset with a
  single non-`raw_source` copy on a `volatile` cluster is flagged by the checker
  as **at-risk** (one purge from loss).
- **`raw_source`** is the durable last resort: if all Zarr copies are lost, the
  data is re-converted from raw (which persists on Delta).
- **Reconciler:** `tools/data/registry.py` — `scan <cluster>` walks a cluster's
  `data_root` and updates that cluster's entries in `copies:`; `check` diffs
  manifest vs. disk (catches purges) and lists at-risk (single-copy) datasets.

## Sub-phase 11d — Globus transfer + re-hydration tooling

Thin CLI over the **Globus CLI** (`globus transfer`), driven by the registry:

- **`tools/data/sync_dataset.py <dataset> --to <cluster> [--years A-B]`** — if the
  target already has it (per registry), no-op; else pick a source from `copies:`
  that has the range and `globus transfer` into the target's `data_root`, then
  add the new location to `copies:`. `deltaai` resolves to `delta` (shared FS).
- **`--rehydrate <cluster>`** — restore everything the registry *says* should be
  on a cluster but `check` found missing (post-purge recovery): pull each missing
  dataset from a peer that still has it, or, if none, re-run the 11a conversion
  from `raw_source`.
- `--dry-run` prints the plan; `--sync-level checksum` for integrity. Document
  per-cluster Globus endpoint activation; capture the collection UUIDs into the
  registry (look up once on globus.org: NCSA Delta, NCAR GLADE, TACC, UChicago).

## Sub-phase 11e — Integration & verification

- Update `examples/weather/ai_rossby/DATA.md` (Phase 10): the canonical "where is
  the data / get it here" answer becomes the registry + `sync_dataset.py`
  (replacing the "on Delta the stores exist" note). First step on a new cluster:
  `python tools/data/sync_dataset.py <dataset> --to <cluster>`.
- **Round-trip + resilience test:** convert one gap year on Stampede3 → copy to
  Derecho → `registry.py scan derecho` shows it → `sync_dataset.py e3sm --to <cluster>` trains there with
  no path edits → simulate a purge (drop the Derecho copy from `copies:` + disk)
  → `sync_dataset.py --rehydrate derecho` restores it from a peer.
- Commit the registry + tooling; keep `copies:` updated on every convert/transfer
  (that *is* the ongoing list).

---

## Execution order
1. **11c registry + 11d tooling** first — needed to move raw *to* Stampede3 for
   11a; independent of the conversions themselves.
2. **Stampede3 env** ready (repo/venv on `$WORK` per `hpc/stampede3.md`).
3. **11a conversions** on Stampede3 `spr` (long SLURM jobs, source-by-source);
   copy each finished source Stampede3 → Derecho (11b).
4. **11b** also migrates the existing Delta stores → Derecho (+ Stampede3);
   assemble the master.
5. **11e** integration + the resilience round-trip.

## Acceptance criteria
- [ ] All targets converted (E3SM 2015–2049, ERA5 1979–2024, AMIP 1978–2024,
      PLASIM sigma 12–104, plev matching sigma; PLASIM & AMIP stats in `.zarr`).
- [ ] Complete archive on Derecho scratch; `registry.py check derecho` clean.
- [ ] `hpc/data_registry.yaml` records **≥2 locations (or 1 copy + `raw_source`)**
      for every dataset; the checker flags any single-copy-on-volatile dataset.
- [ ] `sync_dataset.py` moves a dataset to a fresh cluster (train with no path
      edits) **and** `--rehydrate` restores a simulated-purged copy from a peer.
