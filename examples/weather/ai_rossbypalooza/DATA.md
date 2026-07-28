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
| `sfno_era5` | `physicsnemo-zarr/hindcasts/sfno_era5/` (subset + flattened) | same | v3 checkpoint; same subset |
| `graphcast` | `hindcasts_dsi/zarr/graphcast_e2s` **merged with** `graphcast_wb2` (e2s wins per (init, variable); `init_source` coord records provenance) | union, 2000–2024 | regridded 0.25°→1° (1-D conservative); `u_component_of_wind_250` comes only from wb2 (NaN on e2s-only inits); wb2-sourced precip is NaN at lead day 7 |
| `aifs_single_v2` | `hindcasts_dsi/zarr/aifs_single_v2` | 91/yr, 2000–2024 | regridded 0.25°→1° |

Not converted (still in the raw archives only): aifs_single_v1, aifs_single_v1p1,
aurora_e2s — descoped 2026-07-28.

### Pangu/SFNO subset (`tools/mowe_subset_variables.txt`)

`mean_sea_level_pressure`, `sea_surface_temperature`,
`soil_temperature_level_1`, `surface_pressure`,
`volumetric_soil_water_layer_1`, `total_precipitation_24hr`,
`mean_top_net_long_wave_radiation_flux`, `specific_humidity_{1000,925,850,700,600}`,
`u/v_component_of_wind_{850,500,250}`, `geopotential_{850,500,250}`.

### Variables per harmonized expert

- `aifs_single_v2` (24): `2m_temperature`, `2m_dewpoint_temperature`,
  `surface_pressure`, `soil_temperature_level_{1,2}`,
  `volumetric_soil_water_layer_{1,2}`, `total_column_water`,
  `specific_humidity_{1000,925,850}`, `temperature_{1000,925,850}`,
  `u/v_component_of_wind_{50,200,850}`, `geopotential_{200,500,850}`,
  `total_precipitation_24hr`.
- `graphcast` (20): `2m_temperature`, `mean_sea_level_pressure`,
  `specific_humidity_{1000,925,850}`, `temperature_{1000,925,850}`,
  `u/v_component_of_wind_{50,200,850}`, `geopotential_{200,500,850}`,
  `u_component_of_wind_250` + `geopotential_50` (wb2-only),
  `total_precipitation_24hr`.
- `pangu_s2s` / `sfno_era5` (21): the subset above. (No 2 m temperature or
  upper-air temperature — excluded by the subset spec.)

Conversion provenance: the only value-changing transforms were
`graphcast_wb2 total_precipitation_6hr` → trailing-24h sum, the 0.25°→1°
conservative regrid of the two DSI models, and NaN-ing lead-0 diagnostics;
everything else was rename/reshape (all archives already carried tp as daily
accumulation in m — verified empirically 2026-07-28, see the plan).

## Truth and derived stores

| store | path (Derecho) | producer |
|---|---|---|
| IMERG daily precip (mm/day, 1°, 2000-06→2025-04) | `physicsnemo-zarr/imerg/{YYYY}.zarr` | `tools/data/precip/h5_to_zarr.py` |
| IMD gauge analysis (land, native 33×35 grid; not used by the loader yet) | `physicsnemo-zarr/imd/{YYYY}.zarr` | same |
| ERA5 normalization (mean/std, 18 plev) | `physicsnemo-zarr/era5/normalization_pangu_s2s_{mean,std}.zarr` | `tools/data/era5/build_normalization_zarr.py` |
| IMERG precip norm stats — mean 2.154, std 6.958 mm/day (2001–2018) | `physicsnemo-zarr/normalization/imerg_precip_stats.zarr` | `tools/compute_precip_norm.py` (done 2026-07-28) |
| SEEPS climatology (p1, t2 per month × gridpoint, 2001–2018) | `physicsnemo-zarr/normalization/imerg_seeps_climatology.zarr` | `tools/compute_seeps_climatology.py` (done 2026-07-28) |

Units note: harmonized expert precip is in **m per 24 h** (ERA5 units); the
IMERG truth is **mm/day**. The loader's `PrecipSpec(units="m")` bridges them
(everything is mm/day inside the model/metrics).

## Day-alignment convention

A sample at (init, τ) pairs each expert's daily precip for
`[init+(τ−1)·24h, init+τ·24h)` with the IMERG record stamped
`date(init) + (τ−1)` days (records stamped 00Z on day X cover `[X, X+1)`;
inits are 00Z). The per-expert `day_offset` knob in
`conf/dataset/hindcast_derecho.yaml` is pinned by
`tools/verify_precip_alignment.py` (run against the harmonized stores).

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
