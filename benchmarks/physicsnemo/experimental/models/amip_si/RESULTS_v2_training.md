<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# v2 per-batch training step — our port vs upstream amip_v2

How fast is one training step of the two v2 families in this fork, against
upstream `amip_v2` @ `e0b7b60` running the same geometry on the same GPU, and what
can be done to make ours faster?

**Answer up front.** The port is at **parity** — every measured pair is within
±0.5% and reports byte-identical peak memory. The speed that was available came
from settings, not from the port: enabling TF32 matmuls is worth **1.80×** on
x_DDC, and upstream already trains that way while our diffusion recipe did not.

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
<!-- filled from the JSON records under bench_v2_results/ -->

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
