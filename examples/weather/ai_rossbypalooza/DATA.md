# ai-rossbypalooza data catalog (MoWE week-2 monsoon rainfall)

All paths are the **Derecho copies** — nothing in this recipe reads from the
DSI cluster. Base: `/glade/derecho/scratch/awikner/`.

## Expert hindcast archives

| expert | schema | root (Derecho) | native res / cadence | precip variable |
|---|---|---|---|---|
| pangu_s2s | `consolidated` | `physicsnemo-zarr/hindcasts/pangu_s2s/` | 1°, 24 h (lead 0 = IC, 95 inits/yr: days 1,5,9,…,29 @00Z) | `total_precipitation_24hr` |
| sfno_era5 | `consolidated` | `physicsnemo-zarr/hindcasts/sfno_era5/` | 1°, 24 h (same schedule; v3 checkpoint, rebuilt Jul 25) | `total_precipitation_24hr` |
| graphcast_e2s | `dsi` | `hindcasts_dsi_1deg/zarr/graphcast_e2s/` | 0.25°→**1° (regridded)**, 6 h + daily lead axes, lead days 7–21 | `tp` |
| aurora_e2s | `dsi` | `hindcasts_dsi_1deg/zarr/aurora_e2s/` | same | `tp` |
| aifs_single_v1 | `dsi` | `hindcasts_dsi_1deg/zarr/aifs_single_v1/` | same (twice-weekly + daily merged) | `tp` |
| aifs_single_v1p1 | `dsi` | `hindcasts_dsi_1deg/zarr/aifs_single_v1p1/` | same, **2019–2024 only** | `tp` |
| aifs_single_v2 | `dsi` | `hindcasts_dsi_1deg/zarr/aifs_single_v2/` | same | `tp` |
| graphcast_wb2 | `dsi` | `hindcasts_dsi_1deg/zarr/graphcast_wb2/` | same (twice-weekly, from the WB2 NetCDF archive) | `tp` |

- **Schema `consolidated`** (`tools/data/hindcast/consolidate_hindcasts.py`):
  `(init_time, lead_time[day index], [pressure_level,] lat, lon)`, canonical
  ERA5 names, ClimateZarr group-list attrs.
- **Schema `dsi`** (`tools/data/hindcast/dsi_hindcast_to_formats.py` Format 2):
  `(init_time, prediction_timedelta[hours] | prediction_timedelta_daily[days],
  lat, lon)`, flat level-baked names (`z_500`, `2t`, `tp`), attrs
  `channel_variables_6h/_daily`. **The loader only reads the 1° copies**
  under `hindcasts_dsi_1deg/` produced by `tools/regrid_dsi_to_1deg.py`
  (1-D conservative pooling onto the IMERG grid; the native 0.25° originals
  stay at `hindcasts_dsi/zarr/`). Submit `tools/regrid_dsi_derecho.pbs` to
  (re)build them.

## Truth and derived stores

| store | path (Derecho) | producer |
|---|---|---|
| IMERG daily precip (mm/day, 1°, 2000-06→2025-04) | `physicsnemo-zarr/imerg/{YYYY}.zarr` | `tools/data/precip/h5_to_zarr.py` |
| IMD gauge analysis (land, native 33×35 grid; not used by the loader yet) | `physicsnemo-zarr/imd/{YYYY}.zarr` | same |
| ERA5 normalization (mean/std, 18 plev) | `physicsnemo-zarr/era5/normalization_pangu_s2s_{mean,std}.zarr` | `tools/data/era5/build_normalization_zarr.py` |
| IMERG precip norm stats (shared mean/std for all precip channels + target) | `physicsnemo-zarr/normalization/imerg_precip_stats.zarr` | `tools/compute_precip_norm.py` (this recipe) |
| SEEPS climatology (p1, t2 per month × gridpoint) | `physicsnemo-zarr/normalization/imerg_seeps_climatology.zarr` | `tools/compute_seeps_climatology.py` (this recipe) |

The common grid everywhere is IMERG's exact 1° ERA5 grid: 180×360, lat N→S
89.5..−89.5, lon 0..359.

## Precip conventions (per expert) — VERIFICATION REQUIRED

The stores carry **no units/interval attrs**. The `precip:` block of each
expert in `conf/dataset/hindcast_derecho.yaml`
(`{var, axis: 6h|daily, kind: accum|rate|cumulative, units: m|mm|kg m-2 s-1,
day_offset}`) is a placeholder until pinned by

```
python tools/verify_precip_alignment.py \
    --dataset-yaml conf/dataset/hindcast_derecho.yaml
```

(run on Derecho after the 1° regrid finishes). It reports, per expert and
`day_offset ∈ {−1, 0, +1}`, the monsoon-box pattern correlation and the
magnitude ratio vs IMERG — ratio ≈ 1000 ⇒ units are m; ratio growing with
lead ⇒ cumulative-since-init. **Record the verified values in the yaml and
in the table below.**

| expert | axis | kind | units | day_offset | verified on |
|---|---|---|---|---|---|
| (fill in after verify_precip_alignment.py) | | | | | |

## Day-alignment convention

A sample at (init, τ) pairs each expert's daily precip for
`[init+(τ−1)·24h, init+τ·24h)` with the IMERG record stamped
`date(init) + (τ−1)` days (records stamped 00Z on day X cover `[X, X+1)`;
inits are 00Z).

## Setup order (one-time, on Derecho)

1. `qsub tools/regrid_dsi_derecho.pbs` — 0.25°→1° regrid of all 131 DSI
   stores (~450 GB; self-resubmits until done; sentinels under
   `hindcasts_dsi_1deg/zarr/.regrid_done/`).
2. `python tools/compute_precip_norm.py --imerg-root .../imerg
   --years 2001-2018 --out .../normalization/imerg_precip_stats.zarr`
3. `python tools/compute_seeps_climatology.py --imerg-root .../imerg
   --years 2001-2018 --out .../normalization/imerg_seeps_climatology.zarr`
4. `python tools/verify_precip_alignment.py --dataset-yaml
   conf/dataset/hindcast_derecho.yaml` → fix the yaml + table above.
5. Train: `python train.py` (single GPU) or
   `torchrun --standalone --nproc-per-node=4 train.py`;
   all-experts-per-sample variant: `python train.py training=all_experts`;
   log-loss variant: `python train.py loss=regional_log_mse`.
