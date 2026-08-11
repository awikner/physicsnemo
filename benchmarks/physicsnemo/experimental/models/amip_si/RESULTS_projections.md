<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# RollingDiT input/output projections (Phase 12e)

Ports of upstream amip_v2's `tools/bench_input_embed.py` /
`bench_output_head.py`. Both are CPU-only and take seconds; they exist to
answer "is this feature worth the parameters" *before* committing a run.

## Output head — does the mixture actually buy anything?

`bench_output_head.py` fits each head variant directly on the real EDM target
`F_target = a(σ)·y + b(σ)·ε` under the real `f(σ)` loss weighting, from an
idealised trunk (a frozen random embedding of `y` and `ε`), so the number
reflects **head expressiveness alone**.

`--dim 128 --steps 150`, window σ = {0.007, 0.030, 0.162, 1.241, 15.99, 500}
(the reference ERDM schedule's span at global `t=0`):

| head | params | final weighted MSE |
|---|---|---|
| legacy (fixed `Linear`) | 69,015 | 787.4 |
| mix `K=1` | 71,982 | 455.3 |
| mix `K=2` | 110,940 | 450.8 |
| mix `K=2` + `decoder: column` | 185,730 | **133.9** |

This is the quantitative form of upstream's argument: the legacy head applies
the *same* output matrix at every σ, while the target is a σ- and
channel-dependent blend of a signal readout and a noise readout. A
σ-conditioned per-channel gain (`K=1`) already cuts the achievable loss ~42%
for +3 k parameters; the column decoder — which shares one readout across the
26 pressure levels instead of 130 independent columns — is where the large win
is (−83% vs legacy).

Caveat: this is a *fitting-capacity* measurement on synthetic targets, not a
forecast-skill result. It says the legacy head is representationally
bottlenecked, not how much that costs in RMSE.

## Input projection — do the forcings have a voice?

`bench_input_embed.py` reports each variant's parameter count, the realised
channel budget, and how far the token representation moves under a +1σ shift
of the boundary forcings / SST alone / the trend scalar alone, **relative to a
+1σ shift of the whole state**.

`--dim 256 --nlat 8 --nlon 16` (ratios; higher = that source has more say):

| variant | params | d_state | d_bnd | d_cal | bnd/state | sst/state | trend/state |
|---|---|---|---|---|---|---|---|
| budget-flat | 81,712 | 160 | 64 | 32 | 0.624 | 0.375 | 0.316 |
| budget-column | 114,928 | 160 | 64 | 32 | 0.609 | 0.324 | 0.441 |
| budget-column-conv1 | 72,240 | 160 | 64 | 32 | 0.665 | 0.326 | 0.385 |
| budget-column-nostats | 109,168 | 160 | 64 | 32 | 0.643 | 0.378 | 0.349 |
| budget-column-nobias | 106,736 | 160 | 64 | 32 | 0.559 | 0.309 | 0.330 |
| **budget-column, d_boundary = 3/8·dim** | 157,264 | 128 | 96 | 32 | **1.066** | 0.525 | 0.530 |
| budget-column-nosourcenorm | 114,672 | 160 | 64 | 32 | 0.594 | 0.190 | 0.729 |

The load-bearing row is the second-to-last: raising the boundary budget from
1/4 to 3/8 of `dim` lifts boundary influence from 0.61 to 1.07 relative to the
state — i.e. the budget knob does what it claims, which is the whole point
versus the legacy concat (where CO₂ reaches the model through roughly two
effective channels of 1024). Note also the last row: dropping `source_norm`
makes the *relative* loudness of the sources swing (SST down to 0.19, trend up
to 0.73), which is exactly the accident-of-channel-count effect the RMS
normalisation exists to remove.

Reproduce:

```bash
python benchmarks/physicsnemo/experimental/models/amip_si/bench_output_head.py --dim 128 --steps 150
python benchmarks/physicsnemo/experimental/models/amip_si/bench_input_embed.py --dim 256 --nlat 8 --nlon 16
```
