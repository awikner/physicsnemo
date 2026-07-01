# Phase 8f Completion Plan

Post-smoke, post-commit state (2026-06-26): Phase 8-pre-1 through 8e are
committed and live-validated on Delta. This document lays out the plan
for everything remaining under Phase 8's umbrella.

## Scope

Everything remaining in Phase 8 apart from the SI_V_new translator
xfail (see [phase8e_midway3_checkpoint_inventory.md](phase8e_midway3_checkpoint_inventory.md)),
which stays as a known documented limitation.

## Delivery order (dependency-first)

### F1 — Muon optimizer + `MuonWithAuxAdam` param-group plumbing

**Why first**: the smoke works around it with AdamW; wiring it in
unblocks apples-to-apples convergence checks against upstream amip
runs. Small change, high reuse.

**Files**:

- `examples/weather/ai_rossby/train_loop.py:make_optimizer` — add a
  ``Muon`` branch. Requires `pip install git+https://github.com/KellerJordan/Muon`
  in the venv + pyproject.
- New helper `physicsnemo.experimental.models.amip_si.wrappers` —
  each wrapper class gets a `muon_param_groups()` method mirroring
  upstream `get_dit_muon_param_groups()` /
  `get_rolling_dit_muon_param_groups()` /
  `get_erdm_muon_param_groups()`. Splits ≥2D matmul weights (Muon
  group) from biases + norms + embeddings (aux Adam group).
- `train_diffusion.py:main` — when the optimizer branch is Muon,
  call `wrapper.muon_param_groups()` and hand to `MuonWithAuxAdam`.
- Test: `test/recipes/ai_rossby/test_muon_param_groups.py` — for each
  wrapper, verify param counts sum to `sum(p.numel() for p in
  model.parameters())` and no param is in both groups.

**Effort**: ~2h code + 30 min Delta A40 smoke.

### F2 — `wCO2` translator name-aware trim heuristic

**Why here**: cheap, closes a known correctness footgun before F3+ starts
producing more translations. Keeps the log-message truthful about which
channel got dropped.

**Files**:

- `tools/checkpoint_translation/amip_si.py:wrapper_kwargs_from_hparams` —
  when trimming `varying_boundary_variables` to match `c_grid_dim`,
  prefer to drop entries whose name matches known-scalar-routed
  channels (`global_mean_co2`, `co2*`) rather than the trailing entry.
  Fall back to trailing when no name match.
- `test/tools/checkpoint_translation/test_amip_si.py` — new unit test
  covering the CO2-first trim heuristic.

**Effort**: ~30 min.

### F3 — bf16-native diffusion variants

**Why now**: unblocks the fp32 → bf16 wall-time benchmark that motivates
Phase 8f. Fast Delta run confirms bf16 doesn't tank loss vs fp32.

**Files**:

- `physicsnemo/experimental/models/amip_si/wrappers.py` — each wrapper's
  `MetaData` gains `bf16: True`, `amp: True`, `cuda_graphs: False`
  (the diffusion loop's iterative sample() is not cuda-graph friendly).
- `conf/training/amip_diffusion.yaml` — `amp: bf16` alternative stanza.
- `hpc/scripts/bench_amip_diffusion_bf16.sbatch` — 2× A40 fp32 vs bf16
  benchmark against 1981 Zarr. Emit throughput + peak GPU memory + final
  loss.
- `benchmarks/physicsnemo/experimental/models/amip_si/RESULTS.md` —
  fp32 vs bf16 comparison table + reproduction command.

**Effort**: ~1h code + ~30 min Delta A40 run.

### F4 — DiffusionRolloutValidator per-step sample-limit knob

**Why here**: needed by F5's long-horizon eval suite. Currently the
validator's inference sampler `num_steps` overrides only work through
the training-yaml default; extending to a per-emit-step schedule caps
sampling cost at long horizons.

**Files**:

- `examples/weather/ai_rossby/validate_diffusion.py` — extend
  `sampler_num_steps` to accept a schedule (list of ints, one per
  emitted frame, or a single int applied uniformly).
- `conf/validation/diffusion_rollout.yaml` — document the schedule
  option.
- `test/recipes/ai_rossby/test_validate_diffusion.py` — new unit test
  covering the per-frame override.

**Effort**: ~1h.

### F5 — Eval suite: climatology / bias / QBO / global-mean / ensemble envelopes

**Why last (of the "core" 8f items)**: largest piece; consumes bf16 +
per-step-sample-limit from F3 + F4 for practical runtimes on long
rollouts.

**Files**:

- New module `physicsnemo.experimental.metrics.climate.aggregators` (or
  reuse Phase 4c's `StreamingTimeMean` / `StreamingBinned*` if they
  already exist) — the aggregators run on-GPU in physical units,
  DDP-safe with all-reduce in finalize().
- New validator class `examples/weather/ai_rossby/eval_diffusion.py`:
  * `ClimatologyValidator` — per-variable time-mean over a validation
    year, vs. an oracle climatology on disk.
  * `BiasValidator` — signed lat-weighted mean difference vs.
    climatology.
  * `QBOValidator` — 30° S–30° N zonal-mean U-component at 10, 30, 50
    hPa vs. observed QBO period.
  * `GlobalMeanTimeseriesValidator` — surface energy budget +
    top-of-atmosphere fluxes.
  * `EnsembleEnvelopeValidator` — spread/skill ratios across an
    ensemble.
- Hydra group `conf/validation/eval_suite.yaml` selecting which
  aggregators to run.
- Per-validator unit tests + one live Delta run against a trained
  diffusion ckpt.

**Effort**: ~2 days (largest item).

### F6 — `x_DDC` super-resolution cascade + `CombinedModule`

**Why last**: currently blocks translating 8 of the 13 Midway3
checkpoints (both x_DDC families + Combined). But: full-recipe port,
not a small delta. Needs its own vendoring pass, wrapper, config, and
translator branch.

**Files**:

- Vendor `x_DDC` backbone into
  `physicsnemo/experimental/models/amip_si/x_ddc.py` (mirrors what
  Phase 8a did for the diffusion backbones).
- Vendor scheduler if x_DDC uses a distinct one (upstream `configs/DDC_*.yaml`
  reuses SI/SI_X schedulers, so this is likely a no-op).
- New `XDDCWrapper` in `wrappers.py` — mirrors `AmipDiTWrapper` but
  the backbone consumes a low-res forecast + emits a hi-res residual.
- New `CombinedModule` — thin composition wrapping a forecaster
  (`AmipDiTWrapper` / `RollingDiTWrapper`) + a downscaler
  (`XDDCWrapper`).
- Translator: extend
  `tools/checkpoint_translation/amip_si.py:_MODEL_NAME_TO_WRAPPER` to
  include `x_DDC` → `XDDCWrapper` and `Combined` →
  `CombinedModuleWrapper`. Update the "not supported" error to a hard
  translation path.
- Recipe configs: `conf/model/{amip_x_ddc,amip_combined}.yaml`,
  `conf/loss/{x_ddc}.yaml` if needed.
- Live-validate on the 4 unsupported Midway3 ckpts (2 x_DDC + 2
  Combined pairs) via the translator's live-test parametrization —
  will bump the 5/6 pass count to 12/13.
- Update `phase8e_midway3_checkpoint_inventory.md` with the new
  supported set.

**Effort**: ~1.5 days.

### F7 — Convergence smoke (multi-epoch verification)

**Why last (bookkeeping)**: the 1-epoch smoke verified wiring; a 10-epoch
run on 1981 verifies that loss actually decreases, per the "smoke passed
but loss ~5e6" observation logged in the wiring smoke session.

**Files**:

- `hpc/scripts/smoke_amip_diffusion_convergence_2xA40.sbatch` — 10
  epochs of amip_si on 1981 Zarr, 2× A40 interactive (1h wall). Track
  per-batch loss to a TSV for a matplotlib convergence plot.
- `benchmarks/physicsnemo/experimental/models/amip_si/CONVERGENCE.md` —
  short writeup: expected loss shape, observed shape, sanity check.

**Effort**: ~30 min sbatch draft + 1h Delta run.

## Dependency graph

```
F1 (Muon) ─┬─→ F5 (eval suite)
           │
F2 (wCO2) ─┴───────────────┐
                            │
F3 (bf16) ──→ F4 (sampler ─┼─→ F7 (convergence smoke)
              schedule)     │
                            │
F6 (x_DDC + Combined) ──────┘
```

F1, F2, F6 are independent of each other. F3 and F4 chain into F5 and
F7. Practically: F2 in parallel with F1, then F3, F4 in parallel with F6,
then F5, then F7.

## Estimated total effort

| Item | Effort |
|---|---|
| F1 — Muon | 2 h + 30 min |
| F2 — wCO2 trim | 30 min |
| F3 — bf16 | 1 h + 30 min |
| F4 — sampler schedule | 1 h |
| F5 — eval suite | 2 d |
| F6 — x_DDC + Combined | 1.5 d |
| F7 — convergence smoke | 30 min + 1 h |
| **Total** | **~4 developer days** |

## Out of scope (permanently)

- SI_V_new translator xfail — older ScalarEmbedder variant, documented
  in `phase8e_midway3_checkpoint_inventory.md`.
