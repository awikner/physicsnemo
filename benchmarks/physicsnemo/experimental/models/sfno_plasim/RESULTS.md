# SFNO_PLASIM_5412 benchmark — ai-rossby vs. PanguWeather v2.0

This document is the decision artifact for the SFNO_PLASIM_5412 training
comparison between the new ai-rossby `SfnoPlasim` (vendored Modulus SFNO +
PLASIM routing wrapper at
[`physicsnemo/experimental/models/sfno_plasim/`](../../../../../../physicsnemo/experimental/models/sfno_plasim/))
and the original PanguWeather v2.0 SFNO at
`/work/nvme/bdiu/awikner/PanguWeather/v2.0/networks/modulus_sfno/`.

Earlier rounds targeted SFNO_S2S_0003_test (ERA5 1981, embed_dim=512,
180×360) but that model does not fit on a 40 GB A100 in fp32 even at
batch_size=2/rank. We switched to PLASIM SFNO_5412 (sim52, embed_dim=256,
64×128, legendre-gauss) — the smaller config originally designed for the
PLASIM data.

## Setup

| Item | Value |
|---|---|
| Hardware | 4× NVIDIA A100, Delta `gpuA100x4-interactive` |
| Account | `bdiu-delta-gpu` |
| Precision | fp32 (both repos) |
| Data | PLASIM sim52 year 12 (1464 6-hourly timesteps at 64×128, legendre-gauss); ai-rossby reads [`/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim/12.zarr`](/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim/), PanguWeather reads the source H5 at `/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data/` |
| Reference config | [`config/SFNO_PLASIM_H5_DERECHO_5412_test.yaml`](/work/nvme/bdiu/awikner/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_5412_test.yaml) (PanguWeather v2.0) |
| Per-rank batch size | 8 |
| Global batch size | 32 |
| DataLoader workers | 8 |
| Random seed | 0 (`global_seed 0` for PanguWeather; `seed: 0` for ai-rossby) |
| EMA | OFF for ai-rossby; PanguWeather config has `use_ema: True` + `ema_warmup_epochs: 6` (does not activate at 1 epoch) |
| Gradient clipping | OFF |
| ZeRO-1 | OFF (both repos) |
| Profiling | OFF (nsys disabled on PanguWeather side) |
| Epochs | 1 |
| forecast_lead_times | `[1]` (PanguWeather's `[1, 12, 20, 40, 60]` overridden to `[1]` via sed for parity with ai-rossby's single-step training) |
| Train years | year 12 only (`train_year_end: 14` overridden to `13` via sed) |
| Optimizer | AdamW, lr=1e-4, weight_decay=3e-6 |
| Scheduler | LinearWarmupCosineAnnealingLR, num_warmup_epochs=5, eta_min=1e-8 |
| Loss | `raw_l2` (unweighted MSE; per-var weights still apply via `surface_weight`/`upper_air_weight`) |

## Channel groups & levels (both repos)

| Group | Vars |
|---|---|
| Surface | `pl`, `tas` (2 channels) |
| Upper-air (sigma) | `ta`, `ua`, `va`, `hus` × 10 sigma levels (40 channels) |
| Upper-air (pressure) | `zg` × 10 pressure levels (10 channels) |
| Constant boundary | `lsm`, `sg`, `z0` (3) |
| Varying boundary | `sst`, `rsdt`, `sic` (3) |
| Diagnostic | `pr_6h` (1) |
| Sigma levels | 0.0383, 0.1191, 0.21085, 0.31685, 0.4368, 0.5668, 0.69935, 0.82335, 0.9241, 0.9833 |
| Pressure levels | 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000 Pa |

## How to reproduce

1. **One-time PLASIM year 12 conversion** (already done; see
   [`hpc/scripts/convert_plasim_full_archive.sbatch`](../../../../../../hpc/scripts/convert_plasim_full_archive.sbatch)
   and `tools/data/plasim/configs/sim52_full_coverage.yaml`).

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
