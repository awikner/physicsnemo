<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Latitude-orientation audit of every Zarr store in the registry (2026-08-14)

Full audit of whether each converted store's **labelled** latitude matches the data
actually stored under it. ~1,440 store-copies checked across Delta, Stampede3, Derecho
and Polaris. Four distinct defects were found; the AMIP family was wrong everywhere.

**346 stores repaired**: 322 AMIP-family relabelled (Stampede3 180, Polaris 90,
Derecho 47, Delta 5) and 24 ERA5 stores data-flipped (Derecho 21, Stampede3 3).
Outstanding: Delta's four ERA5 stores, blocked by a full filesystem — see §3.

The one-line reason this went unnoticed for so long: the raw per-timestep H5 archives
**carry no latitude coordinate**, so each converter's lat array was an assumption, and
nothing in the pipeline ever contradicted it. `registry.py check` counts stores; it
cannot see an upside-down one. `ClimateZarrDataset` reads array order verbatim and
never consults the lat *values*, so training runs happily on a mislabelled store.

## Ground truth: what row order the raw archives actually have

Established from the data, never from the look of the numbers — Antarctic land
fraction and ice-sheet geopotential, TOA insolation at both solstices (polar
day/night is pure astronomy), and Jan-vs-Jul polar temperature. 5–6 independent
tests per archive, unanimous, and for three archives independently confirmed by a
`climatology.nc` sitting beside the H5 that *does* carry a lat coordinate.

| raw archive | true row order | corroboration |
|---|---|---|
| ERA5 `h5data` (Delta) | **S→N** (row 0 = South Pole) | `1979-2018_mean_climatology.nc` lat −89.5…89.5 |
| AMIP `h5` (Delta) | **S→N** | 6/6 physical tests |
| ERA5 `h5_dailyavg` (amip_dailyavg source) | **S→N** | 6/6 physical tests |
| E3SM `sigma_data` (Stampede3) | **S→N** | `climatology.nc` lat −89.5…89.5 |
| PLASIM sigma (Delta) | **N→S** | orography + insolation + `tas` |
| PLASIM plev (Derecho) | **N→S** | `climatology.nc` lat 87.86…−87.86, matching the YAML Gaussian list |

Landmarks worth remembering: the polar-band contrast is ~0.87 vs ~0.18 land fraction
and ~18,700 vs ~1,450 m²/s² surface geopotential, and June-solstice insolation is
~1.8e6 vs ~0 J/m². These are not subtle.

## What was wrong

### 1. The whole AMIP family, every store, every cluster — FIXED

`tools/data/amip/amip_h5_to_zarr.py` read `grp[v][...]` verbatim while its configs
declared `lat: [89.5 … -89.5]`, and both source archives are S→N. Every field in
every store it ever wrote was upside-down relative to its label. Confirmed bit-exact
against raw *and* independently by the `land_sea_mask` / orography / insolation
anchors, which cannot be dismissed as a threshold artefact.

Affected and repaired: `amip` (45 Stampede3, 45 Derecho, 5 Delta incl. the four
1981 quarter subsets), `amip_dailyavg` (45 Stampede3, 1 Derecho),
`amip_dailyavg_coarse` (45 Stampede3, 1 Derecho), `amip_dailyavg_boundary`
(45 Stampede3) — **232 stores**.

**Fix chosen: relabel the coordinate to ascending S→N, data bytes untouched**
(`tools/data/amip/relabel_lat_ascending.py`). This was selected as the option
optimised for data loading:

* `ClimateZarrDataset` reads array order verbatim and never uses the lat values, so
  relabelling leaves the read path at **zero per-sample transformation** — the
  optimum. Reversing the data instead buys no loading speed at all.
* It avoids rewriting ~2.7 TB, so no re-replication across three clusters.
* It preserves bit-exact agreement with upstream amip_v2 and with **every existing
  checkpoint** — all of them were trained on exactly these S→N bytes.
* `norm_stats/sst_climatology.npz` stays valid (it was fitted on this grid).
* E3SM already ships correct S→N labels, so this is not a new convention here.

The converter now **verifies** the declared coordinate against the row order the data
actually has, at ingest, and refuses to write a mismatched store
(`--allow-lat-mismatch` downgrades it to a warning). The shipped configs' lat arrays
were changed to ascending, and stores get a `lat_row_order` attr.

### 2. Derecho's ERA5 1979–1999 (21 stores) — FIXED

Those stores never received the 2026-07-24 fix: no `lat_flipped_to_NtoS` attr at all,
and they failed every absolute anchor. 2000–2024 on Derecho were already correct.

Repaired 2026-08-14 with `flip_lat_zarr.py` (PBS job 7117204 on the `develop` queue,
stores 6-wide): 15 arrays reversed in each of the 21 stores, no errors, and a
follow-up anchor check over **all 46 stores reports zero FLIPPED**.

**Separate gap found in the same 21 stores, NOT fixed:** they were never augmented
either — 19 arrays where 2000–2024 have 24, missing `skin_temperature`,
`surface_pressure`, `soil_temperature_level_1`, `volumetric_soil_water_layer_1` and
the derived OLR, and no `augmented_full_surface` attr. The July-24 commit did the
augmentation *and* the flip together, and only 2000–2024 received either on Derecho.
So Derecho is orientation-correct but still channel-incomplete for 1979–1999; treat
Stampede3 (or Delta, once repaired) as the era5 reference, not Derecho.
Fixing it needs `augment_era5_full_surface.py` against the raw archive, which lives
on Delta.

### 3. Three half-repaired ERA5 arrays — FIXED on Stampede3, BLOCKED on Delta

An interrupted `flip_lat_zarr.py` run leaves an array flipped over a *prefix* of
60-timestep blocks and upside-down over the rest, while the store still advertises
`lat_flipped_to_NtoS=True`. Ranges below were mapped bit-exactly against raw, and the
Delta and Stampede3 copies were confirmed byte-identical by checksum first.

| store | array | upside-down range | seam |
|---|---|---|---|
| era5/1987 | `sea_surface_temperature` | t=0…1459 (all) | never flipped at all |
| era5/1988 | `temperature` | t=1260…1463 (204 steps, all 18 levels) | 1260 = 60×21 |
| era5/1989 | `temperature` | t=420…1459 (1040 steps, all 18 levels) | 420 = 60×7 |

1987's SST was the one surface field *not* NaN-fill-patched, which is why it fell out
of the repair's `--vars` list. Repaired on Stampede3 with
`tools/data/era5/repair_lat_ranges.py` and re-verified.

**Delta is blocked by storage, not by tooling.** The `bdiu` project on `/work` is out
of usable space (project-view `df` shows 29 TB of 30 TB used; quota 28.3 TB against a
29.8 TB soft limit). Small writes land — a 12 MB `dd` and a store-attr update both
succeeded — but every bulk write fails:

* in-place chunk rewrites fail with `EDQUOT` (a repair of 1987 got 59 timesteps in,
  and the *revert* failed too),
* and a Globus transfer of replacement stores stalled with `nice_status =
  QUOTA_EXCEEDED` after 4 MB.

So **Delta's era5/1987 `sea_surface_temperature` is currently mixed**: t=0…58 correct,
t=59…1459 upside-down. That is recorded in the store's own `lat_repair_interrupted`
attr. 1988, 1989 and `era5_sfno_s2s/1981` are untouched and uniformly as originally
converted (`era5_sfno_s2s` reports 0 arrays flipped and no guard, so nothing is
half-done there).

The fix chosen is **replacement rather than in-place repair**, since untarring writes
new files instead of overwriting quota-blocked chunks. The repaired Stampede3 stores
are already packed and waiting — 3 tars, 92 GB, at
`stampede3:/scratch/09979/awikner/zarr-tars/era5-latfix/{1987,1988,1989}.zarr.tar`.
Once there is headroom on Delta, from Stampede3:

```bash
G=~/gcli/bin/globus
S=/scratch/09979/awikner/zarr-tars/era5-latfix
D=/work/hdd/bdiu/awikner/zarr-tars/era5-latfix
cd $S && for t in *.tar; do echo "$t $t"; done | \
  $G transfer 1e9ddd41-fe4b-406f-95ff-f3d79f9cb523:$S \
              7e936164-de58-4e3d-85da-21aa23c07169:$D \
    --batch - --sync-level mtime --label "ai-rossby era5 lat-fix 1987-1989"
```

then on Delta `bash /work/hdd/bdiu/awikner/lat_audit/delta_swap_era5.sh`, which
verifies each tar, swaps one store at a time, re-checks the new copy's anchors and
repair manifest, and restores the original on any failure.

**Space budget:** each store is ~31 GB, so the swap needs ~62 GB of transient headroom
per year (tar + new store) if the old copy is kept until verified. If Delta stays
tight, delete each broken store *before* transferring its tar — net usage is then flat,
and correct verified copies exist on both Stampede3 and Derecho meanwhile.

`era5_sfno_s2s/1981` has **no Stampede3 copy**, so it cannot be replaced this way. It
needs either free space plus `flip_lat_zarr.py --store $Z/era5_sfno_s2s/1981.zarr`
(resumable, currently EDQUOT), or a re-convert from the Delta raw archive.

### 4. `era5_sfno_s2s/1981` (Delta only) — BLOCKED on the same quota

All 19 arrays upside-down: converted before the 2026-07-24 fix and never flipped.
The label is right and the rows are not, so the DATA is what needs reversing. See the
note at the end of §3 — there is no Stampede3 copy to replace it with.

## What was already correct

* `plasim` 93/93 and `plasim_plev` 93/93 — bit-exact vs an N→S raw archive under N→S labels.
* `e3sm` 35/35 — bit-exact vs an S→N raw archive under **S→N labels**, i.e. correct.
  Delta's 9-year copy is checksum-identical to Stampede3's.
* `era5` 43/46 on Delta and Stampede3; Derecho 2000–2024.
* `imerg` 26/26 and `imd` 125/125 on all three clusters.
* `e3sm/climatology_bias.zarr` (32 decisive fields, 0 flipped), the IMERG
  quantile/SEEPS stores, and `hindcasts/pangu_s2s` 25/25.
* Every `normalization_*.zarr` has **no lat dimension** — per-channel scalars, no exposure.

## Two convention inconsistencies (each internally consistent)

* **`imd` disagrees across clusters.** Derecho's copy is S→N (native, never passed
  through `reorient_imd_lat.py`); Delta's and Stampede3's are N→S. Both are correctly
  labelled, but code that assumes the project N→S convention reads Derecho's upside-down.
* **`e3sm` is S→N everywhere** while ERA5/PLASIM/IMERG are N→S. Correct, but do not
  "fix" the labels without also reversing the data. The AMIP family now joins E3SM on
  S→N, so *two* conventions exist by design; rely on the `lat` coordinate (or the
  `lat_row_order` attr), never on a positional assumption.

## Polaris — RELABELLED

`amip_dailyavg_coarse` (45) and `amip_dailyavg_boundary` (45) were upside-down, as
expected (they were tar-shipped from Stampede3's then-flipped stores). Both sets were
relabelled 2026-08-14: **45 + 45 relabelled, 0 skipped, 0 failed**, anchors unanimous
S→N. The independent post-check did not run — the Polaris session dropped back to MFA
immediately afterwards — so re-confirm when convenient:

```bash
python tools/data/check_lat_orientation.py --anchors-only --band 60 \
    --stores "/eagle/lighthouse-uchicago/physicsnemo-zarr/amip_dailyavg_coarse/[0-9]*.zarr"
python tools/data/check_lat_orientation.py --anchors-only \
    --stores "/eagle/lighthouse-uchicago/physicsnemo-zarr/amip_dailyavg_boundary/[0-9]*.zarr"
```

`norm_stats/sst_climatology.npz` needs **no change**: it was fitted on the boundary
store's grid and is consumed index-aligned with it, and the relabel did not move any
data, so it stays valid.

## Tooling (new)

| Tool | Purpose |
|---|---|
| [`tools/data/check_lat_orientation.py`](../../../tools/data/check_lat_orientation.py) | The auditor. Physical anchors → cross-array correlation → temporal seam detection → exact-vs-raw. Exit 1 on any defect, so it gates. `--anchors-only` is a 1–2 read/store screen. |
| [`tools/data/_common/lat_orientation.py`](../../../tools/data/_common/lat_orientation.py) | Shared anchors, used by both the auditor and the converter's ingest assertion. |
| [`tools/data/amip/relabel_lat_ascending.py`](../../../tools/data/amip/relabel_lat_ascending.py) | Metadata-only AMIP repair. Re-verifies from the data before touching anything; idempotent. |
| [`tools/data/era5/repair_lat_ranges.py`](../../../tools/data/era5/repair_lat_ranges.py) | Reverses one array over a timestep range, for half-flipped arrays. Fails closed on an unverified range; block-resumable. |
| [`tools/data/test_check_lat_orientation.py`](../../../tools/data/test_check_lat_orientation.py) | 7 tests on synthetic stores, ~6 s, no cluster data — plants each defect class and asserts it is caught. |

Three things the tooling learned the hard way, all now encoded:

* **Compare correlation magnitudes, not signed values.** Surface pressure vs
  orography is legitimately r = −0.99; judging on sign calls it reversed.
* **Geostrophy is a *relative* test.** A whole-store flip reverses both `dΦ/dy` and
  the sign of `f`, and the two cancel, so it pins a wind to its geopotential — not to
  the globe.
* **A single day is not a season.** The Jan-1 polar temperature contrast can be ~4 K
  and occasionally the wrong sign; it must not outvote the time-invariant anchors.
  Likewise a subset store may contain no solstice, so a date match needs a tolerance.

## Re-running the audit

```bash
# fast screen, any dataset/cluster
python tools/data/check_lat_orientation.py --anchors-only \
    --stores "$AI_ROSSBY_DATA/amip_dailyavg/*.zarr" --workers 12

# gold standard, where the raw archive is co-located (get --raw-order from the table above)
python tools/data/check_lat_orientation.py --stores "$AI_ROSSBY_DATA/era5/[0-9]*.zarr" \
    --h5-dir /work/hdd/bdiu/bgong1/data/h5data --raw-order "S->N" --workers 64
```

Caveat when interpreting a raw comparison on Delta: `/work/hdd/bdiu/bgong1/data/h5data`
is **not** bit-identical to the (deleted) Stampede3 staging the era5 stores were built
from — `skin_temperature`, the soil fields and OLR differ in value there (NaN fill and
a differing derived-OLR source), so they match only by correlation. That is why
correlation-only evidence never on its own reports a seam.
