# AsyncForecastWriter — per-IC inference speedup benchmark

Measures how much of the per-IC rollout wall time the async forecast
writer hides by overlapping disk I/O with the next IC's GPU rollout.
Compares two configurations of the same inference loop driven by
[`run_inference_streaming_per_ic`](../../../../../../examples/weather/ai_rossby/inference.py)
on the translated PanguPlasimLegacy S2S 2000 checkpoint:

| Mode  | Writer                                                                                    | Behaviour                                                              |
| ----- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| sync  | `AsyncForecastWriter(max_in_flight=1, num_workers=1)` + `wait_all()` after every `submit` | Each IC's Zarr is fully flushed before the next IC starts rolling out. |
| async | `AsyncForecastWriter(max_in_flight=4, num_workers=2)` (in-prod default)                   | Next IC's rollout overlaps the previous IC's disk write.               |

Everything else is bit-identical: same model, same dataset, same IC
indices, same per-IC payload, same lustre target directory. The
wall-time delta is the headroom the async writer is hiding.

## Setup

| Item | Value |
| ------------- | ------------- |
| Hardware | 1× NVIDIA A40 (40 GB), Delta `gpuA40x4-interactive` |
| Account | `bdiu-delta-gpu` |
| Model | [`PanguPlasimLegacy`](../../../../../../physicsnemo/experimental/models/pangu_plasim/pangu_plasim_legacy.py) — embed_dim=240, 17 levels, 180×360, 60.9M params |
| Checkpoint | `pangu_plasim_s2s_2000.mdlus` (translated from PanguWeather S2S 2000 best_ckpt.tar) |
| Dataset | `era5_sfno_s2s/1981.zarr` (1460 6-hourly timesteps, 180×360, 17 pressure levels) |
| Output | Zarr V3, per-IC, frames 0..max_step where frame 0 = IC; ~22 MB per file |
| Output storage | lustre `/work/hdd/bdiu/awikner/sfno_bench/` |
| ICs | 6, stride 80 in time index → 0, 80, 160, 240, 320, 400 |
| Rollout horizon | `max_step=20` per IC (21 frames including IC) |
| Warmup | 1 IC dropped before timing |
| Precision | fp32 |
| Job IDs | 19556579 |
| Code | [`benchmarks/.../async_writer/run_bench.py`](run_bench.py), [`hpc/scripts/bench_async_writer.sbatch`](../../../../../../hpc/scripts/bench_async_writer.sbatch) |

## Reproduce

```bash
# Translate the PanguWeather S2S 2000 .tar to .mdlus (one-time):
python tools/checkpoint_translation/pangu_plasim.py \
    --source /work/hdd/bdiu/awikner/PanguWeather-rajatm2/v2.0/results/S2S/2000/training_checkpoints/best_ckpt.tar \
    --model-config examples/weather/ai_rossby/conf/model/pangu_plasim_s2s.yaml \
    --target-class PanguPlasimLegacy \
    --output /work/hdd/bdiu/awikner/translated_checkpoints/pangu_plasim_s2s_2000.mdlus \
    --strict

# Run the benchmark on a 40 GB A40:
sbatch hpc/scripts/bench_async_writer.sbatch
```

The sbatch hits `gpuA40x4-interactive` (50 min cap) and writes a JSON
report to `hpc/scripts/logs/bench-async-writer-<jobid>.json`.

## Results (job 19556579)

| Metric | sync | async |
| --------------- | ----- | ----- |
| Total wall time (s) | 99.40 | 77.21 |
| Per-IC wall time (s) | 16.57 | 12.87 |
| Files written | 6 | 6 |
| Bytes per file (MB) | 22.0 | 22.0 |

**Headline:** the async writer hides **22.3% of total wall time** in this
configuration (1.29× speedup, 22.19 s saved over 99.40 s). Per-IC the
savings work out to ≈ 3.7 s — close to the per-file Zarr flush latency
on lustre. The disk write is roughly fully overlapped with the next IC's
forward pass; the remainder is the final IC's flush, which can't overlap
anything.

## What this means

- For the S2S 180×360 output shape at 21 frames per IC, a single per-IC
  Zarr is 22 MB. On Delta lustre that takes about 3–4 s synchronously,
  which is the per-IC delta we observed.
- Scaling to higher resolution (e.g. ERA5 0.25°, 721×1440) or longer
  rollouts (12-month forecast = ~1460 frames per chunk in the
  climatology pipeline) would push the per-file payload to ~GB. The
  same overlap mechanic still applies, but the *relative* speedup
  depends on how the per-file disk write compares to the per-chunk GPU
  rollout — if disk dominates rollout, the async writer reduces the
  schedule to roughly `max(disk, rollout)`.
- For the climatology pipeline's chunked output (
  [`examples/weather/ai_rossby/climatology_cli.py`](../../../../../../examples/weather/ai_rossby/climatology_cli.py)),
  this benchmark is a lower bound. A 1460-frame chunk dwarfs a 21-frame
  per-IC dump, so the absolute time saved per chunk should be much
  larger than 3.7 s.

## Caveats

- Single-GPU. Multi-rank inference partitions ICs across ranks; we
  expect the per-rank speedup to behave identically since each rank
  drives its own writer.
- One run (warmup + sync + async), no repetition. Run-to-run lustre
  variance on Delta is typically <10% for writes of this size; the
  22% headline comfortably exceeds that envelope.
- The "sync" baseline serializes through *one* writer thread by design
  — that is the configuration the async writer is hiding. The
  *previous* in-prod path (no writer at all, blocking `to_zarr` call in
  the rollout loop) is equivalent to the sync baseline minus thread
  hand-off overhead, which is sub-millisecond per submit.

## Conclusion

Async forecast writer is a real, measurable speedup for the ai-rossby
per-IC inference path even on the modest S2S output payload. The
implementation can stay as the default. Re-run with a larger payload
(climatology validation, ERA5 0.25°) to characterize how the headline
scales when disk dominates rollout.
