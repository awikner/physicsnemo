# AMIP SI diffusion — fp32 vs. bf16-native benchmark

Phase 8f (F3) decision artifact: does training the AMIP SI diffusion
recipe (`AmipDiTWrapper` around `AmipDiT`) under bf16 autocast tank
convergence relative to fp32, and what does it buy in throughput /
peak memory? This motivates the wall-time benchmark Phase 8f exists
to unblock (per `phase8f_completion_plan.md`).

## Setup

| Item | Value |
|---|---|
| Hardware | 2× NVIDIA A40, Delta `gpuA40x4-interactive` |
| Account | `bdiu-delta-gpu` |
| Data | AMIP 1981 Zarr (`/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr`) |
| Model | `AmipDiTWrapper` / `AmipDiT`, `dim=128`, `num_blocks=4`, `num_heads=4`, `patch_size=4`, `c_grid_downsample=1` (shrunk to fit 2× A40, matches `smoke_amip_diffusion_2xA40.sbatch`) |
| Loss | `si` (`DriftScheduler`) |
| Optimizer | Muon (`MuonWithAuxAdam`), lr=5e-5, weight_decay=3e-6 (Phase 8f F1) |
| EMA | enabled, decay=0.999, warmup_epochs=0 (bench run is too short to warm up) |
| Epochs | 3 |
| Per-rank batch size | 2 |
| Precision (arm A) | fp32 (`training=amip_diffusion`, `amp: none`) |
| Precision (arm B) | bf16-native (`training=amip_diffusion_bf16`, `amp: bf16`) |
| Validation | off (throughput/memory/loss only) |

## How to reproduce

```bash
sbatch hpc/scripts/bench_amip_diffusion_bf16.sbatch
```

Runs both arms back-to-back on the same allocation. Reads its per-batch
TSVs from `/work/hdd/bdiu/awikner/amip_si_bench/{fp32,bf16}_<jobid>.tsv`
and its peak-GPU-memory + final-loss lines from the sbatch log at
`hpc/scripts/logs/bench-amip-diffusion-bf16-<jobid>.out`.

## Headline numbers

Job [19852637](../../../../../../hpc/scripts/logs/bench-amip-diffusion-bf16-19852637.out),
2× A40, 3 epochs (2190 batches/arm).

| Metric | fp32 | bf16 |
|---|---|---|
| first-10-batch mean loss | 6.504e6 | 6.504e6 |
| last-10-batch mean loss | 5.329e6 | 5.333e6 |
| steady-state samples/s (batches 50→end) | 19.87 | 19.93 |
| peak GPU memory (GB) | 1.77 | 1.57 |

## Conclusion

**bf16 does not tank convergence** — the two loss trajectories track
each other almost exactly throughout the run (both start ~6.50e6,
both end ~5.33e6; the largest first-vs-last-10-batch gap between arms
is <0.1%). **bf16 gives a measurable peak-memory win** (1.57 GB vs.
1.77 GB, ~11% lower) but **no measurable throughput win** at this
smoke-test model size (`dim=128`, 4 blocks — steady-state samples/s
is within noise of fp32, 19.93 vs. 19.87). That's expected at this
scale: bf16's throughput advantage comes from tensor-core utilization
on large matmuls, and a `dim=128` toy backbone is far too small to be
compute-bound on an A40 — most of the wall-time here is data loading /
kernel-launch overhead, not GEMM time. The memory win alone (smaller
activations) is consistent with the Phase 8f decision rule.

**Decision**: adopt bf16 as the wrapper's declared default
(`MetaData.bf16 = True` / `MetaData.amp = True`, landed in this same
Phase 8f commit) per the "no convergence regression + measurable win"
rule — the memory win is real and convergence is unaffected. The
throughput win is expected to materialize at production model size
(`dim=384`, 6 blocks, per `conf/model/amip_si.yaml`) where the
backbone is actually compute-bound; re-running this benchmark at
production size is a natural follow-up but out of scope for the F3
smoke-scale decision artifact.
