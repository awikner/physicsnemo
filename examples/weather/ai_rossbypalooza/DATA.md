# ai-rossbypalooza data catalog (MoWE week-2 monsoon rainfall)

All paths are the **Derecho copies** — nothing in this recipe reads from the
DSI cluster. Base: `/glade/derecho/scratch/awikner/`.

## The unified harmonized stores (what training reads)

`tools/harmonize_hindcasts.py` converts every expert archive into ONE schema
under **`hindcasts_mowe/{model}/{YYYY}.zarr`**:

- dims `(init_time, lead_time, lat, lon)` — `lead_time` in whole **days as
  values** (0 = the IC where present), 1° IMERG/ERA5 grid (180×360, lat N→S);
- every variable a flat 2-D field named by the **ERA5 long name** with an
  integer `_{level}` suffix for pressure levels (`geopotential_500`), with
  per-variable `units` attrs;
- accumulated / time-mean variables refer to the **preceding 24 h** in ERA5
  units: `total_precipitation_24hr` in **m**, `mean_top_net_long_wave_
  radiation_flux` in W m⁻²; leads without a full 24 h window (lead 0; day 7
  for wb2-sourced graphcast precip) are NaN.

| expert | built from | inits | notes |
|---|---|---|---|
| `pangu_s2s` | `physicsnemo-zarr/hindcasts/pangu_s2s/` (subset + 3-D flattened) | 95/yr (days 1,5,9,…,29 @00Z), 2000–2024 | 21-variable subset (below) |
| `sfno_era5` | `physicsnemo-zarr/hindcasts/sfno_era5/` (subset + flattened) | same | **v4** checkpoint (`SfnoPlasim.0.30.mdlus_ema_v4parity`, per the source store's own attrs); same subset |
| `graphcast` | `hindcasts_dsi/zarr/graphcast_e2s` **merged with** `graphcast_wb2` (e2s wins per (init, variable); `init_source` coord records provenance) | union, 2000–2024 | regridded 0.25°→1° (1-D conservative); `u_component_of_wind_250` comes only from wb2 (NaN on e2s-only inits); wb2-sourced precip is NaN at lead day 7 |
| `aifs_single_v2` | `hindcasts_dsi/zarr/aifs_single_v2` | 91/yr, 2000–2024 | regridded 0.25°→1° |

Not converted (still in the raw archives only): aifs_single_v1, aifs_single_v1p1,
aurora_e2s — descoped 2026-07-28.

### The master variable list (`tools/mowe_subset_variables.txt`)

**Every harmonized store is filtered to this 21-variable list** — variables
outside it are not converted at all:
`mean_sea_level_pressure`, `sea_surface_temperature`,
`soil_temperature_level_1`, `surface_pressure`,
`volumetric_soil_water_layer_1`, `total_precipitation_24hr`,
`mean_top_net_long_wave_radiation_flux`, `specific_humidity_{1000,925,850,700,600}`,
`u/v_component_of_wind_{850,500,250}`, `geopotential_{850,500,250}`.

### Variables per harmonized expert (intersection with the list)

- `aifs_single_v2` (11): `surface_pressure`, `soil_temperature_level_1`,
  `volumetric_soil_water_layer_1`, `specific_humidity_{1000,925,850}`,
  `u/v_component_of_wind_850`, `geopotential_{500,850}`,
  `total_precipitation_24hr`.
- `graphcast` (10): `mean_sea_level_pressure`,
  `specific_humidity_{1000,925,850}`, `u/v_component_of_wind_850`,
  `geopotential_{500,850}`, `u_component_of_wind_250` (wb2-only; NaN on
  e2s-only inits), `total_precipitation_24hr`.
- `pangu_s2s` / `sfno_era5` (21): the full list.

Conversion provenance: the only value-changing transforms were
`graphcast_wb2 total_precipitation_6hr` → trailing-24h sum, the 0.25°→1°
conservative regrid of the two DSI models, and NaN-ing lead-0 diagnostics;
everything else was rename/reshape/drop (all archives already carried tp as
daily accumulation in m — verified empirically 2026-07-28, see the plan).

## Truth and derived stores

| store | path (Derecho) | producer |
|---|---|---|
| IMERG daily precip (mm/day, 1°, 2000-06→2025-04) | `physicsnemo-zarr/imerg/{YYYY}.zarr` | `tools/data/precip/h5_to_zarr.py` |
| IMD gauge analysis (land, native 33×35 grid; not used by the loader yet) | `physicsnemo-zarr/imd/{YYYY}.zarr` | same |
| ERA5 normalization (mean/std, 18 plev) | `physicsnemo-zarr/era5/normalization_pangu_s2s_{mean,std}.zarr` | `tools/data/era5/build_normalization_zarr.py` |
| **Model v1** IMERG precip norm stats in log space — mean −6.379, std 0.858 in `log(1e-3 + P[m/24h])` (2000–2019) | `physicsnemo-zarr/normalization/imerg_precip_stats_log.zarr` | `tools/compute_precip_norm.py --log-epsilon 1e-3 --log-units m` (done 2026-07-28) |
| Linear precip stats — mean 2.154, std 6.958 mm/day (2001–2018; superseded by the log store for v1) | `physicsnemo-zarr/normalization/imerg_precip_stats.zarr` | `tools/compute_precip_norm.py` |
| SEEPS climatology (p1, t2 per month × gridpoint, **2000–2019**) | `physicsnemo-zarr/normalization/imerg_seeps_climatology.zarr` | `tools/compute_seeps_climatology.py` (recomputed 2026-07-28) |

Units note: harmonized expert precip is in **m per 24 h** (ERA5 units); the
IMERG truth is **mm/day**. The loader's `PrecipSpec(units="m")` bridges them
(everything is mm/day inside the model/metrics).

## Day-alignment convention — VERIFIED 2026-07-28

A sample at (init, τ) pairs each expert's daily precip for
`[init+(τ−1)·24h, init+τ·24h)` with the IMERG record stamped
`date(init) + (τ−1)` days (records stamped 00Z on day X cover `[X, X+1)`;
inits are 00Z). `tools/verify_precip_alignment.py` results:

| expert | day_offset | evidence |
|---|---|---|
| pangu_s2s | **0 (confirmed)** | decisive at short leads: corr 0.61/0.59/0.48 at τ=2/3/4 for offset 0 vs ≤0.37 for ±1 |
| sfno_era5 | 0 (undeterminable — see flag below) | corr ≈ 0 at ALL leads/offsets |
| graphcast | 0 (by construction) | week-2 signal too weak to discriminate (Δcorr ≤ 0.01); shares the converter convention pangu verified |
| aifs_single_v2 | 0 (by construction) | same (stores only have leads ≥ 7, where skill ≈ 0) |

### Latitude-orientation incident — FOUND AND REPAIRED 2026-07-28

A physical-orientation audit (July NH-vs-SH z500 asymmetry + ITCZ
seasonality) found the **IMERG stores and all 25 pangu_s2s stores
(source + harmonized) data-flipped in latitude** relative to their
(correct) N→S coordinate. Root cause: ascending source latitudes swapped
for a ref-store N→S label without flipping the data
(`consolidate_hindcasts.py` pangu path — patched; IMERG had the analogous
converter issue). Repaired **in place** with `flip_lat_zarr.py` on all
clusters; norm stats (flip-invariant, unchanged) and the SEEPS climatology
(+`clim_mean`) regenerated; audit now passes for every store
(`~/mowe_tools/check_lat_orientation.py` on Derecho).

**Retractions**: the earlier "sfno precip is broken" and "graphcast/aifs/
sfno 4× wet bias" flags were artifacts of scoring correctly-oriented
models against flipped truth (and pangu only *looked* aligned because it
was flipped consistently with IMERG). Post-fix alignment numbers are in
`~/mowe_tools/verify_postflip.log`.

### Remaining data-quality notes

1. pangu's `mean_top_net_long_wave_radiation_flux` clamps at exactly 0 at
   its upper tail (sfno's does not).

## Setup order (one-time, on Derecho)

1. `qsub tools/harmonize_derecho.pbs` — builds all 100 harmonized stores
   (graphcast merge + aifs_v2 regrid + pangu/sfno subsets; self-resubmits;
   sentinels under `hindcasts_mowe/.harmonize_done/`).
2. `python tools/compute_precip_norm.py …` — **done 2026-07-28**.
3. `python tools/compute_seeps_climatology.py …` — **done 2026-07-28**.
4. `python tools/verify_precip_alignment.py --dataset-yaml
   conf/dataset/hindcast_derecho.yaml` → pin `day_offset` per expert.
5. Train: `python train.py` / `torchrun --standalone --nproc-per-node=4
   train.py`; variants: `training=all_experts`, `loss=regional_log_mse`.
