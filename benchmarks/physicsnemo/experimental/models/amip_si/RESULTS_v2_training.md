<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# v2 per-batch training step — our port vs upstream amip_v2

How fast is one training step of the two v2 families in this fork, against
upstream `amip_v2` @ `e0b7b60` running the same geometry on the same GPU, and what
can be done to make ours faster?

**Answer up front, in three parts.**

1. **The port is at parity.** Every pair measured under identical settings is within
   ±0.5% and reports byte-identical peak memory, on A100, H200 and GH200.
2. **One parity gap was ours to close.** Upstream's `train.py` calls
   `torch.set_float32_matmul_precision("high")` and our `train_diffusion.py` set
   nothing, so identical models ran at different matmul precision — worth
   **1.5–1.8×** and now available as `++training.matmul_precision=high`.
3. **Then the profile bought more than the settings did.** Nsight Systems showed
   that after TF32, ~70% of a step is fp32 attention on sm80 CUTLASS kernels that
   Hopper cannot accelerate and the TF32 flag cannot reach. Running *only*
   attention in bf16 (`++training.attention_dtype=bf16`) takes the full stack to
   **4.31× (x_DDC)** and **3.84× (ERDM)** over shipped fp32 — and, since upstream
   has no equivalent knob, **2.0–2.3× faster than upstream** at matched settings.

Every knob is opt-in; no shipped default moved.

## Method

`benchmarks/physicsnemo/experimental/models/amip_si/bench_v2_training_step.py`
times one full optimizer step — forward, loss, backward, `optimizer.step()` — on
identical synthetic tensors.

| Item | Value |
|---|---|
| Ours | `conf/model/amip_erdm_v2.yaml`, `conf/model/amip_x_ddc_dit.yaml` |
| Upstream | `anthonyzhou-1/amip_v2` @ `e0b7b60`, `configs/ERDM_co2.yaml`, `configs/DDC.yaml` |
| Batch | 1 (what upstream's configs train at) |
| Precision baseline | fp32 — upstream's configs say `precision: 32-true` |
| Optimizer | AdamW or Muon, the **same** on both sides |
| Iterations | 20, after 5 discarded warmup |

Choices that decide whether the numbers mean anything:

* **Each side is built from its own config**, and the harness **hard-errors if the
  parameter counts disagree** — different geometry makes a timing comparison
  meaningless rather than merely noisy.
* **`torch.cuda.synchronize()` brackets every step.** Without it CUDA's async
  launch queue makes a step look nearly free.
* **The median, not the mean**, over 20 iterations; one allocator flush skews a
  mean.
* **One model alive at a time.** The first version of this harness timed upstream
  while our model, gradients and optimizer state were still resident, and so
  reported upstream as 2.9 GB hungrier than ours. It was not; that number was my
  harness. Each side is now freed (`del`, `gc`, `empty_cache`,
  `reset_peak_memory_stats`) before the next is built, and the sides then report
  *identical* peaks — which is the evidence the isolation works.
* **Synthetic tensors, deliberately.** The subject is model-and-scheduler speed;
  a real loader adds Lustre variance that swamps it. End-to-end iteration time is
  a separate measurement off `cfg.bench.per_batch_tsv`.
* **The same scheduler kwargs on both sides** (`--scheduler upstream`). Ours ship
  `noise: spherical` for x_DDC, which upstream's interpolant refuses outright, so
  leaving it on would time a fork-only feature and read it as a port regression.

### Reproduce

```bash
# Delta (x86) or DeltaAI (aarch64) — the script picks the venv from uname -m
python benchmarks/physicsnemo/experimental/models/amip_si/bench_v2_training_step.py \
    --family erdm --side both --amip-repo /work/nvme/bdiu/awikner/amip_v2 \
    --optimizer muon --iters 20

# the 1.80x knob
python .../bench_v2_training_step.py --family x_ddc --side both \
    --amip-repo <repo> --tf32
```

Upstream is a plain clone; it needs no Polaris (`git clone
git@github.com:anthonyzhou-1/amip_v2.git && git checkout e0b7b60`).

## Results

All GH200 120GB unless noted, batch 1, 20 timed iterations after 5 warmup, median.
ERDM uses Muon (upstream's configured optimizer), x_DDC AdamW.

### Parity: same settings, both sides

| config | ours | upstream | ratio | peak mem |
|---|---|---|---|---|
| x_DDC fp32 (A100-40GB) | 656.4 ms | 656.7 ms | 1.000x | 11.67 G both |
| x_DDC fp32 +TF32 (A100) | 364.7 ms | 365.3 ms | 0.998x | 11.67 G both |
| x_DDC bf16 (A100) | 143.9 ms | 143.9 ms | 1.000x | 9.70 G both |
| x_DDC fp32, Muon (A100) | 738.0 ms | 736.1 ms | 1.003x | 10.26 G both |
| x_DDC fp32 | 314.1 ms | 314.2 ms | 1.000x | 11.71 G both |
| x_DDC +TF32 | 208.5 ms | 208.6 ms | 1.000x | 11.71 G both |
| ERDM fp32 | 2519.5 ms | 2525.5 ms | 0.998x | 66.58 G both |
| ERDM +TF32 | 1632.9 ms | 1627.1 ms | 1.004x | 66.58 G both |
| ERDM fp32 (H200-141GB) | 2316.0 ms | 2315.3 ms | 1.000x | 66.58 G both |
| ERDM bf16+compile | 515.4 ms | 517.5 ms | 0.996x | 38.74 G both |

Byte-identical peak memory on every pair, which is the strongest single statement
about the port: same shapes, same allocations, same order.

### The optimization ladder for our side

fp32 weights everywhere except where noted; each row adds to the one above.

| step | x_DDC | vs fp32 | ERDM | vs fp32 |
|---|---|---|---|---|
| fp32 (as shipped) | 314.1 ms / 11.71 G | 1.00x | 2519.5 ms / 66.58 G | 1.00x |
| `matmul_precision=high` | 208.5 ms / 11.71 G | 1.51x | 1632.9 ms / 66.58 G | 1.54x |
| `+ attention_dtype=bf16` | **91.8 ms / 10.77 G** | **3.42x** | **793.0 ms / 58.49 G** | **3.18x** |
| `+ torch.compile` | **72.9 ms / 10.15 G** | **4.31x** | **655.8 ms / 52.65 G** | **3.84x** |
| full bf16 autocast + compile | 51.3 ms / 8.76 G | 6.12x | 515.4 ms / 38.74 G | 4.89x |

Against **upstream at the same TF32 settings**, the bf16-attention rows are
**2.28x** (x_DDC) and **2.03x** (ERDM) faster, because that knob exists only on our
side: it works by routing our call sites through ``amip_si._attention.sdpa``, and
upstream calls ``F.scaled_dot_product_attention`` directly.

### Where the time went, before and after

Nsight Systems, GH200, `--nvtx`:

| kernel group | fp32 | TF32 |
|---|---|---|
| `fmha_cutlass*_f32_aligned_*_sm80` (attention) | 43.6% (ERDM) / 46.2% (x_DDC) | **69.7% / 70.9%** |
| cuBLAS GEMMs | ~50% | ~10% |

ERDM's four largest GEMMs: **6416 ms -> 705 ms** across the traced steps, the top
one moving from `sm80_xmma_gemm_f32f32_f32f32` to
`sm90_xmma_gemm_f32f32_tf32f32`. Its attention kernels over the same change:
**3488.8 -> 3510.7 ms** backward and **1520.8 -> 1531.2 ms** forward, i.e.
unchanged. They are hand-written CUTLASS rather than cuBLAS, so the TF32 flag does
not reach them, and they are **sm80** builds: fp32 has no flash implementation, so
PyTorch selects the mem-efficient backend whose fp32 path targets Ampere and runs
as-is on Hopper.

That is the whole argument for `attention_dtype=bf16` — it is the one part of the
step that Hopper could not accelerate, and it was 70% of it.

### Kernel counters (Nsight Compute)

`ncu --nvtx --nvtx-include "timed_step/" --kernel-name "regex:fmha|flash|xmma_gemm|sgemm"`,
per-launch on GH200. Durations are longer than the nsys timeline because counter
collection replays each kernel; the ratios are the point.

| kernel | dur | compute% | dram% | occ% | tensor% |
|---|---|---|---|---|---|
| x_DDC fp32 GEMM `sm80_xmma_..._f32f32_tilesize128x128x8` | 717 µs | 85.3 | 2.9 | 12.5 | **0.0** |
| x_DDC TF32 GEMM `sm90_xmma_..._tf32f32_tilesize128x128x32` | **92 µs** | 83.3 | 17.1 | 17.8 | **88.7** |
| x_DDC attention `fmha_cutlassF_f32_aligned_64x64_rf_sm80`, fp32 | 2799 µs | 51.1 | 0.9 | 17.0 | 32.3 |
| the same kernel with TF32 enabled | **2799 µs** | 51.1 | 0.9 | 17.0 | 32.3 |
| x_DDC attention `pytorch_flash::flash_fwd_kernel`, bf16 | **325 µs** | 51.3 | 2.1 | 12.3 | 48.4 |
| ERDM attention `fmha_cutlassF_..._sm80`, TF32 | 15298 µs | 54.4 | 1.1 | 18.5 | 33.5 |
| ERDM TF32 GEMM `sm90_xmma_..._tilesize128x128x32` | 579–739 µs | 85.8–87.8 | 22–26 | 18.6 | **92.8–98.4** |

What the counters settle:

* **fp32 GEMMs use no tensor cores at all** (0.0%), running at 85% of *fp32* peak.
  TF32 lifts the same GEMMs to ~90% tensor utilization and **7.8x** faster
  (717 -> 92 µs). That is the TF32 win, measured rather than inferred.
* **The attention kernel is unchanged by TF32** — identical duration, identical
  32.3% tensor, identical 0.9% DRAM — confirming at counter level what the timeline
  implied. bf16 replaces it with flash at **325 µs, 8.6x faster**, which is what
  produces the 2.27x step-level speedup.
* **Nothing is bandwidth-bound.** DRAM throughput is 0.9–2.1% for attention and
  17–26% for the TF32 GEMMs. That rules out a whole class of optimizations —
  `channels_last`, layout changes, fusing memory-bound elementwise work — none of
  which had anything to reclaim. A useful negative result.
* **What remains is compute-saturated or parallelism-starved, not fixable in the
  model.** The TF32 GEMMs sit at 84–88% compute and 90%+ tensor. Flash attention is
  at 51% compute and 12.3% occupancy, i.e. latency-bound on available parallelism,
  where the lever is batch size (measured: 59.3 ms/sample at batch 2 versus 80.8 at
  batch 1) rather than any restructuring.
* One loose end, recorded and not chased: ERDM's TF32 run still contains three small
  `sm80_xmma_gemm_f32f32_f32f32` kernels at 0.1–0.3% tensor (7.3, 8.5, 40.5 µs) —
  cuBLAS picking SIMT kernels for shapes too small to profit from TF32. Together
  well under 1% of the step.

## Practical notes

* **ERDM needs ~67 GB/GPU in fp32.** A 40 GB A100 cannot train the shipped geometry
  at batch 1 and neither can upstream's code, which OOMs at the same point
  (39.44 GiB ours, 39.48 GiB theirs). bf16 autocast + compile brings it to
  **38.74 GB**, which does fit; TF32 + bf16 attention + compile reaches 52.65 GB,
  which does not.
* **bf16 needs `disable_cudnn_sdpa` on GH200.** With DeltaAI's inherited torch 2.10,
  bf16 attention raises `cuDNN Frontend error: No valid execution plans built`
  rather than falling back. A100 is unaffected.
* **Muon costs ~12% throughput and saves 1.4 GB** versus AdamW (738.0 vs 656.4 ms on
  A100). Upstream's configs specify it, so compare Muon-to-Muon.
* **x_DDC at batch 2** is 59.3 ms/sample versus 80.8 at batch 1 (bf16) — 1.36x better
  per-sample throughput for 14.35 GB.
* **Shrunken geometry misleads about fixed costs.** At 1/8 backbone width the
  spherical-noise option looked like a 1.6x tax; at production geometry it is
  654.2 ms against 656.4, i.e. nothing.

## What this says about optimizing our side

1. **TF32 is a parity fix, not a bonus.** Upstream's `train.py` calls
   `torch.set_float32_matmul_precision("high")`; our `train_diffusion.py` set
   nothing, so identical models ran at different matmul precision. `train.py` in
   this repo has done the same for Pangu/SFNO since its benchmarks (~15% there).
   Now available as `++training.matmul_precision=high` (plus `allow_tf32`,
   `cudnn_benchmark`), **off by default** because it changes a trained model's
   numerics.
2. **bf16 is the largest single lever** and is already wired
   (`training=amip_diffusion_bf16`). `CONVERGENCE.md` is the evidence that it does
   not tank convergence for the v1 SI family; the equivalent check for v2 has not
   been run.
3. **Muon costs throughput and buys memory.** Upstream's configs specify it, so
   quote Muon numbers when comparing against upstream runs and AdamW numbers only
   against each other.
4. **Shrunken geometry misleads about fixed costs.** At 1/8 backbone width the
   spherical-noise option looked like a 1.6× tax; at production geometry it costs
   nothing, because the fixed transform is dwarfed by a 20-block dim-1024
   backbone. Any figure here at `--shrink > 1` is a harness self-test, not a
   result.

## Found while benchmarking, unrelated to speed

* **`noise: spherical` crashed on torch_harmonics ≥ 0.9.** The generator built
  coefficients of width `l_max + 1`; 0.8.0 reports `mmax = lmax + 1` and 0.9.1
  reports `mmax = lmax`, and `sht.py` asserts on the mismatch. Since
  `conf/sampler/x_ddc.yaml` ships that option, every x_DDC rollout on a newer
  install failed with a bare `AssertionError`. Fixed to read `isht.mmax`.
* **Our x_DDC sampler defaults diverge from upstream's** (`spherical`/
  `exponential` vs `gaussian`/`uniform`). Harmless for speed; it is an
  inference-parity question, since upstream's x_DDC cannot run the spherical
  option at all.
