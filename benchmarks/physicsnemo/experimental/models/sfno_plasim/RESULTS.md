# SFNO_PLASIM_S2S benchmark — ai-rossby vs. PanguWeather v2.0

This document is the decision artifact for the SFNO_S2S 1981 training
comparison between the new ai-rossby `SfnoPlasim` (vendored Modulus SFNO +
PLASIM routing wrapper at
[`physicsnemo/experimental/models/sfno_plasim/`](../../../../../../physicsnemo/experimental/models/sfno_plasim/))
and the original PanguWeather v2.0 SFNO_v2 at
`/work/nvme/bdiu/awikner/PanguWeather/v2.0/networks/modulus_sfno/`.

## Setup

| Item | Value |
|---|---|
| Hardware | 4× NVIDIA A100, Delta `gpuA100x4-interactive` |
| Account | `bdiu-delta-gpu` |
| Precision | fp32 (both repos) |
| Data | ERA5 1981 (1460 timesteps); converted to
[`/work/hdd/bdiu/awikner/physicsnemo-zarr/era5_sfno_s2s/1981.zarr`](/work/hdd/bdiu/awikner/physicsnemo-zarr/era5_sfno_s2s/) for ai-rossby |
| Reference config | [`config/SFNO_S2S_0003_test.yaml`](/work/nvme/bdiu/awikner/PanguWeather/v2.0/config/SFNO_S2S_0003_test.yaml) (PanguWeather v2.0) |
| Per-rank batch size | 4 |
| Global batch size | 16 |
| DataLoader workers | 4 |
| Random seed | 0 (`global_seed 0` for PanguWeather; `seed: 0` for ai-rossby) |
| EMA | OFF (warmup is 6 epochs anyway → no behavioral difference at 1 epoch) |
| Gradient clipping | OFF |
| ZeRO-1 | OFF (both repos) |
| Profiling | OFF (nsys disabled on PanguWeather side) |
| Epochs | 1 |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01 |
| Scheduler | LinearWarmupCosineAnnealingLR, num_warmup_epochs=5, eta_min=1e-8 |
| Loss | `raw_l2` (unweighted MSE; per-var weights still apply via `surface_weight`/`upper_air_weight`) |

## Channel groups & levels (both repos)

| Group | Vars |
|---|---|
| Surface (incl. land + ocean) | `2m_temperature`, `10m_u/v_component_of_wind`, `mean_sea_level_pressure`, `surface_pressure`, `volumetric_soil_water_layer_1`, `soil_temperature_level_1`, `skin_temperature`, `sea_surface_temperature` (9 channels) |
| Upper-air | `temperature`, `u_component_of_wind`, `v_component_of_wind`, `specific_humidity`, `geopotential` × 17 pressure levels (85 channels) |
| Constant boundary | `land_sea_mask`, `geopotential_at_surface` (2) |
| Varying boundary | `toa_incident_solar_radiation` (1) |
| Diagnostic | `total_precipitation_24hr`, `mean_top_net_long_wave_radiation_flux` (2) |
| Pressure levels | 5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa |

## How to reproduce

1. **One-time ERA5 1981 conversion with the SFNO_S2S channel set** (sub-minute on `cpu-interactive`):

   ```bash
   srun --partition=cpu-interactive --account=bdiu-delta-cpu \
        --time=00:15:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=64g \
        --job-name=era5-1981-sfno_s2s \
       bash -lc 'cd /work/nvme/bdiu/awikner/physicsnemo && source .venv/bin/activate && \
                python tools/data/era5/pangu_h5_to_zarr.py \
                    --year 1981 \
                    --channel-config benchmarks/physicsnemo/experimental/models/sfno_plasim/era5_sfno_s2s_channels.json \
                    --output /work/hdd/bdiu/awikner/physicsnemo-zarr/era5_sfno_s2s/1981.zarr'
   ```

2. **Submit both benchmark runs** (they're independent, run in parallel):

   ```bash
   sbatch hpc/scripts/bench_sfno_ai_rossby.sbatch
   sbatch hpc/scripts/bench_sfno_panguweather.sbatch
   ```

3. **Parse + summarize**:

   ```bash
   python benchmarks/physicsnemo/experimental/models/sfno_plasim/compare.py \
       --ai-rossby-tsv     /work/hdd/bdiu/awikner/sfno_bench/ai_rossby_<jobid>.tsv \
       --panguweather-log hpc/scripts/logs/bench-sfno-panguweather-<jobid>.out
   ```

## Headline numbers

*Run pending — fill in after benchmark jobs complete.*

Expected table format (compare.py output gets pasted here):

```
| Metric | ai-rossby (`SfnoPlasim`) | PanguWeather v2.0 (`SFNO_v2`) |
|---|---|---|
| batches/epoch | … | … |
| final-batch loss | … | … |
| median-batch loss | … | … |
| wall (epoch, s) | … | … |
| samples/s | … | … |
```

## Per-batch loss curve

*Run pending. Plot from `ai_rossby_<jobid>.tsv` and `bench-sfno-panguweather-<jobid>.out` once the
runs complete.*

## Conclusion

*To be filled in based on results.*

Decision rule (per the plan): if max relative |Δ loss| < 5% at any batch and
the wall-clock numbers are within ±10%, the ai-rossby vendor + wrapper is
functionally equivalent to PanguWeather's reference; we keep proceeding with
ai-rossby. Larger divergence → investigate (likely candidates: pos_embed
init RNG ordering, `module.` DDP prefix on save/load, instance_norm running
stats).
