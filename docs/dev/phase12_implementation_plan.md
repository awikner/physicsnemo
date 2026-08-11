# Phase 12 — amip_v2 rebaseline (ERDM / x_DDC / Combined parity)

Status: **12a-12e complete (2026-08-11); 12f+ planned** · Author: Claude (analysis + plan) · Created: 2026-08-07

Phase 8 ported the amip repo as of its last public commit
(`497827e` "BIG changes", 2026-06-17) in full — all five diffusion
families, backbones, wrappers, translator, Muon, bf16, eval suite
(see `implementation_plan.md` §Phase 8 and `phase8f_completion_plan.md`).
Since then the collaborator has replaced amip with
**[anthonyzhou-1/amip_v2](https://github.com/anthonyzhou-1/amip_v2)** — a
staged cleanup plus ~7 weeks of new work. Phase 12 brings the fork's AMIP
port up to parity with amip_v2.

## Baseline pin

- **amip_v2 @ `e0b7b60`** (2026-08-06) — **confirmed as the baseline by
  the user (2026-08-07).** The repo is still moving (the README lags the
  config dir), so upstream work past `e0b7b60` is out of scope for
  Phase 12; fold it into a future rebaseline pass rather than chasing
  HEAD mid-phase.
- amip_v2's history starts from a *private* snapshot already ahead of
  v1's public HEAD — v2 is the only record of the intermediate work.
  There is no usable v1..v2 commit trail; parity is established by
  file-level comparison (done for this plan) and by the cross-check
  tests specified below.
- Currently supported upstream configs: `ERDM_co2` (base),
  `ERDM_fancy` (all features), `ERDM_ocean`, `DDC`, `DDC_tiny`,
  `combined`, `combined_co2`. (The README's `ERDM.yaml` /
  `ERDM_sst_anom.yaml` / `ERDM_gmsst.yaml` / `ERDM_ocean_anom.yaml`
  were moved to `configs/old/` in the "SST changes" commit.)

## Decisions locked in (user review, 2026-08-07)

| Topic | Decision |
|---|---|
| **Authoritative repo** | amip_v2 for everything it still contains (ERDM, x_DDC, Combined, data pipeline, configs). |
| **v1-only families** (SI, SI_X, EDM, RFM; single-step DiT, ERDMUnet) | **Keep, frozen on the v1 contract.** Not deleted, not migrated. They stay loadable for the translated Midway3 checkpoints; they are excluded from every Phase 12 contract change. |
| **Channel layout** | **Adopt v2's order** (upper-air level-major, levels flipped so 1000 hPa leads) for the ERDM / x_DDC / Combined wrappers. Translator handles v1-trained checkpoints via a permutation shim (or re-translation). |
| **Data store** | **Pre-coarsened 45×90 Zarr** built by a new converter — same size/speed win as v2's memmap fast store, keeps the Phase-8 Zarr-only decision. No memmap port. |
| **Scope** | **Full parity with v2**: contract migration, bug parity, loader semantics, input/output projections, cross-attention, global conditioning, SST forcing suite, ocean prediction, Combined/rollout rewrite, translator update, health gates. |

### The dual-contract seam (read this before touching wrappers)

Keeping the v1 families frozen while migrating ERDM/x_DDC to v2's channel
order means **two packing contracts coexist permanently**:

- `AmipDiTWrapper` (SI/SI_X/EDM) and the RFM/ERDMUnet paths keep the v1
  variable-major `_flatten_upper_air` and all current defaults. Their
  tests, configs, and translator branches are untouched. Mark each with a
  `.. note:: frozen on the amip-v1 contract (Phase 12)` docstring and a
  comment in their YAMLs.
- `RollingDiTWrapper`, `XDDCWrapper`, `CombinedModule` (and anything new
  in Phase 12) move to the v2 layout. New shared pack/unpack helpers are
  **separate functions** (`_flatten_upper_air_v2` / `_unflatten_upper_air_v2`
  or a `layout=` arg defaulting per class) so the frozen path cannot drift.
- The datapipe/normalizer are layout-agnostic (they emit `(V, L, H, W)`
  blocks; packing happens in the wrappers), so no dataset change is
  needed for the seam itself.

## v1 → v2 delta (facts the plan is built on)

**Deleted upstream:** SI/SI_X/EDM/RFM schedulers + configs; DiT
(single-step), ERDMUnet, AE, Decoder, Unet backbones; non-windowed
training branches everywhere (`window_train` is the only mode); the two
data loaders `amip_new.py` (1723 LOC) + `amip_fast.py` (920 LOC) merged
into `data/amip.py` (1220 LOC) with `MemmapStore` / `H5Store` backends
behind a 3-method interface (`key_for`, `read_frames`, `check_range`);
dead layers (`cross_attention.py`, `conv.py`, `basics.py`,
`distributions.py`, `spherical_harmonics.py`); dead config keys.

**Contract / correctness changes:**
- `assemble_input` / `disassemble_input` (`common/utils.py`): upper-air
  block is now `rearrange(multilevel.flip(2), "b c l h w -> b (l c) h w")`
  — level-major, 1000 hPa first (v1: variable-major, config order).
  `state_layout(dataconfig)` derives block sizes from the data config so
  layout and config can't drift.
- Fixed an **off-by-one in level indexing** for val-loss logging and
  plotting (z500 was actually z600).
- Fixed a **double-downsample** in the fast-store path (state was
  already coarse, then coarsened again).
- Ocean block (when enabled) appends **after** upper air:
  `[ surface | diagnostic | upper air | ocean ]` — channels 0..150 keep
  their meaning; append/strip/impose live in `ERDMScheduler`, not in
  `assemble_input`.
- Recommended scheduler stats: `sigma_data: 1.0` (unit-variance
  normalization; v1 + our port default 0.5), `rho: -10`, `P_mean: 2.0`,
  `P_std: 1.2`, `S_churn: 1.0`, `S_noise: 1.5`.
- Upstream keeps old checkpoints loadable: the `legacy` input/output
  projection modes are **bit-identical** to v1's; only new-mode runs
  retrain the projections.

**New upstream features:**
- **Input projection** (`modules/layers/input_embed.py`,
  `RollingDiTInputEmbed`): `mode: legacy | budget`. Budget mode gives
  each source (state / boundary / calendar) an explicitly sized,
  RMS-normalised slice of `dim` (`d_boundary`, `d_calendar`, `d_co2`,
  `co2_linear`, `state_encoder: column` with shared per-level `Linear` +
  level embedding, `boundary_encoder: conv2` with spherical padding,
  `boundary_pool_stats`, `boundary_static_bias`, `source_norm`).
- **Output head** (`modules/layers/output_head.py`,
  `RollingDiTOutputHead`): `mode: legacy | mix` — σ-conditioned
  per-output-channel gated mixture over `num_experts` linear readouts
  (`out_c = Σ_k (1/K + gate_{k,c}(c_noise)) · (W_k h)_c`);
  `decoder: column` mirrors the column encoder.
- **Forcing cross-attention**: `CausalForcingCrossAttentionBlock`
  (`c_grid_cross_layers`, `c_grid_cross_heads`) — temporal causal
  cross-attention from hidden state to the forcing stream on the last N
  blocks; AdaLN-Zero gated.
- **Global conditioning**: `global_cond: true` routes day-of-year (+ CO₂
  / trend scalar) into the AdaLN conditioning vector next to flow-time
  (`TimestepEmbedder(num_conds=...)`).
- **SST forcing suite** (`data/sst_forcing.py`,
  `scripts/make_sst_climatology.py`, artifact
  `norm_stats/sst_climatology.npz` — per-gridpoint harmonic fit of the
  day-of-year cycle over training years only): config keys
  `sst_anomaly_channel: none|append|replace`,
  `scalar_forcing: auto|none|co2|global_mean_sst`,
  `sst_anomaly_scale` / `sst_scalar_scale` (fitted stats by default;
  physical units for +ΔT experiments).
- **Ocean-state prediction** (`data.ocean_state_variables`): SST /
  sea-ice (+ optionally the SST anomaly) as *predicted* tail channels,
  supervised against each frame's own-time boundary truth (`W+1`
  boundary frames served, `[:W]` forcing / `[1:]` truth), **imposed**
  (`truth + sigma_bar(0)·noise`) into the window at inference so
  predictions never feed back; `train/loss_ocean` logged separately;
  warm start from an `ERDM_co2` checkpoint with zero skipped keys via
  `training.partial_checkpoint` / `load_partial_weights`.
- **`forcing_from_raw`** (`data/amip.py`): the single choke point where
  stored boundary channels become model inputs (z-score, SST rescaling,
  CO₂ pop, calendar row) — shared by train / val / rollout / bias /
  debug entrypoints so the channel contract can't fork.
- **Pre-downsampled memmap store**: state pre-coarsened to 45×90,
  2 TB → 160 GB, one array read per frame (our answer: pre-coarsened
  Zarr, see 12c).
- **Combined/rollout rewrite**: `windowed_init` (oracle init: noise up
  the first W real frames) / `windowed_step` (denoise front frame →
  downscale → shift → append noise); `rollout.py` streams monthly
  NetCDF via a background thread pool.
- **Repo health gates** (`tools/check_repo.py`): compile / imports /
  instantiate-signature-hash / synthetic-smoke / deadscan, with
  monotonicity baselines. Bench tools `bench_input_embed.py`,
  `bench_output_head.py`, `bench_store_io.py`.
- `common/inference.py` (single build-model + load-checkpoint helper),
  `scripts/make_constant_boundary.py` (constant-boundary cache npz).
- Boundary NaN handling: `smooth_masked_boundary` (iterated masked
  Gaussian smoothing near coasts) behind `smooth_nan_boundaries` +
  `mask_fill` — our datapipe currently has `NanFillTransform` only.

## Sub-phases (dependency order)

### 12a — Bug-parity audit + config rebaseline *(no new features)*

1. **Off-by-one audit**: check `validate_diffusion.py`,
   `eval_diffusion.py`, and any plotting/level-label code in the fork
   for v1's level-index bug (z500 labeled/logged as z600). Fix + unit
   test pinning a known level to its channel index.
2. **Double-downsample audit**: trace the recipe-side pre-downsample in
   `train_diffusion.py` + wrappers (`c_grid_downsample` vs the
   dataset-side resolution) and prove by test that the state is
   coarsened exactly once on every path (train / val / inference /
   climatology). We never ported the fast store, so we *likely* don't
   have the bug — the test is the deliverable.
3. **Scheduler config rebaseline**: new `conf/loss/erdm_v2.yaml` +
   `conf/model/amip_erdm_v2.yaml` mirroring upstream `ERDM_co2.yaml`
   (`sigma_data 1.0`, `rho -10`, `P_mean 2.0`, `P_std 1.2`,
   `S_churn 1.0`, `S_tmin 0 / S_tmax 1000`, `S_noise 1.5`,
   `num_steps 2`, `num_blocks 20`, `dim 1024`). Our `ERDMScheduler`
   already has the churn knobs — this is config, not code.
4. **Freeze markers**: docstring + YAML comments on the v1-only
   families (see the dual-contract seam above). No functional change;
   full existing test suite stays green.

**Effort**: ~1 day.

**12a delivered** *(2026-08-07)*:

- **Off-by-one audit — bug NOT present.** The fork never ported v1's
  headline-level plotting (the buggy `l_plot = -10/-13/-6` hardcoded
  positions in v1 `train_module.py`, off by one slot each in the
  26-level list). Every level selection in the fork is by value:
  `eval_diffusion.py` resolves `levels.index(lvl)` and raises on a
  missing level; `validate_diffusion.py` reduces over the level axis
  without labeling; `inference.py` writes the level coordinate from the
  same config list that orders the axis. Regression pin:
  `test_qbo_validator_resolves_level_values_to_channel_indices`
  (out-of-storage-order request + per-slot sentinel values through the
  real `_band_mean` selection).
- **Double-downsample audit — structurally impossible today.** Zero
  resample calls exist in `train_diffusion.py` / `validate_diffusion.py`
  / `eval_diffusion.py` / all schedulers; wrappers default
  `c_grid_downsample=1` so both streams share the dataset grid. The only
  resample anywhere is x_DDC's *intentional* corruption operator
  (already pinned by its roundtrip test). New invariant test:
  `test_rolling_dit_wrapper_preserves_input_resolution` (forward output
  spatial dims == input; packed c_grid on the state grid) — this is the
  guard that stays meaningful when 12c makes the store the single place
  resolution changes.
- **Config rebaseline**: [`conf/loss/erdm_v2.yaml`](../../examples/weather/ai_rossby/conf/loss/erdm_v2.yaml)
  (sigma_data 1.0, S_churn 1.0, S_noise 1.5, S_tmax 1000, alpha 0) +
  [`conf/model/amip_erdm_v2.yaml`](../../examples/weather/ai_rossby/conf/model/amip_erdm_v2.yaml)
  (`RollingDiTWrapper`, dim 1024 / 20 blocks / 16+8 heads, upstream
  daily-avg 151-channel variable set, 45×90). Instantiation covered by
  `test_hydra_diffusion_configs_instantiate` (151 in_channels verified).
  Header documents what needs code, not config: scalar-CO₂ routing
  (12d), input_embed/output_head/global_cond/cross-attn (12e), c_grid
  resolution decision (12c). Runnable only once 12c's dataset exists.
- **Freeze markers** on the six v1-only modules
  (`dynamic_interpolant.py`, `x_interpolant.py`, `edm.py`, `rfm.py`,
  `dit.py`, `erdm_unet.py`), the two frozen wrapper classes
  (`AmipDiTWrapper`, `ERDMWrapper` docstring notes), seven frozen YAMLs
  (`model/amip_{si,si_x,rfm,erdm}.yaml`, `loss/{si,si_x,rfm}.yaml`) and
  a SUPERSEDED note on `loss/erdm.yaml` (v1-era stats).
- **Suites green**: 201 passed (recipes + amip_si) + 59 passed
  checkpoint-translation, CPU (`-m "not smoke and not cuda"`). No GPU
  smoke needed — 12a changed no GPU code paths (configs, docstrings,
  tests only); the two new tests are CPU unit tests.

### 12b — Channel-contract migration (ERDM / x_DDC / Combined)

5. Add v2 pack/unpack helpers (level-major + flip) and a
   `state_layout()`-equivalent derived from the dataset config (mirrors
   upstream `common/utils.py:state_layout` so the layout is stated once).
6. Migrate `RollingDiTWrapper`, `XDDCWrapper`, `CombinedModule`
   pack/unpack to the v2 layout. Frozen families untouched.
7. **Translator**: `tools/checkpoint_translation/amip_si.py` grows a
   `--source-contract {v1,v2}` flag. v2 checkpoints translate directly;
   v1 ERDM/x_DDC checkpoints get a channel permutation applied to every
   channel-indexed tensor (input projection rows, output head columns,
   `noise_scales` buffer). Previously-translated `.mdlus` files are
   **re-translated from the original Lightning ckpts** (cheap, and
   avoids a `.mdlus`-to-`.mdlus` migration tool).
8. **Cross-check tests** (CPU): pack/unpack round-trip; bit-parity of
   our v2 pack against upstream `assemble_input`/`disassemble_input`
   on synthetic `(b, c, l, h, w)` tensors (vendored reference copy in
   the test, not an amip_v2 import); permutation-shim correctness
   (translate a synthetic v1 ckpt, assert forward equivalence against
   the v1-layout model fed permuted inputs).

**Effort**: ~1.5 days.

**12b delivered** *(2026-08-07; code complete, cluster steps pending)*:

- **Design deviation — layout kwarg instead of weight permutation.** The
  migrated wrappers gained a ``channel_layout`` constructor kwarg that
  travels with the ``.mdlus`` args: ``"v2"`` (default; upstream amip_v2:
  ``[surface | diag | upper_air]`` with the upper-air block level-major,
  1000 hPa first; c_grid ``[varying | constant]``), ``"v1"`` (upstream
  v1: same group order, upper-air variable-major), and ``"fork"``
  (RollingDiTWrapper only — the historical Phase-8 order). The
  translator's ``--source-contract {v1,v2}`` flag simply sets the kwarg,
  so runtime packing always matches the trained channel-indexed weights —
  no weight surgery, and checkpoint provenance is self-describing.
- **Discovered latent Phase-8e bug — RESOLVED (user decision
  2026-08-07: fix within the v1 contract).** The fork's Phase-8 wrappers
  pack ``[surface | upper_air | diag]`` with c_grid
  ``[constant | varying]`` — but real upstream-v1 checkpoints were
  trained on ``[surface | diag | upper_air]`` / ``[varying | constant]``
  (v1 ``assemble_input`` / ``assemble_forcing``), and the translator
  never permuted. Translated v1 checkpoints for the SI/SI_X/EDM (AmipDiT)
  and ERDM-UNet families therefore ran with permuted channels at the
  input/output projections. **Fix:** the frozen ``AmipDiTWrapper`` /
  ``ERDMWrapper`` also gained the ``channel_layout`` kwarg, restricted
  to the two v1-era contracts ``{"fork", "v1"}`` (never ``"v2"`` — their
  families were deleted upstream). Default stays ``"fork"``
  (bit-identical pre-12b behavior for existing configs/tests); the
  translator now sets ``channel_layout="v1"`` for every auto-derived
  target and **raises** if ``--source-contract v2`` is combined with a
  frozen-family target. All previously-translated ``.mdlus`` files for
  these families must be re-translated (cluster follow-up below).
- Mixin (`_RollingPackUnpackMixin`) is layout-aware with class default
  ``"fork"`` → the frozen ``ERDMWrapper`` is bit-identical to pre-12b
  (pinned by test); ``RollingDiTWrapper`` defaults ``"v2"`` with
  ``conf/model/amip_rfm.yaml`` pinning ``channel_layout: fork`` (frozen
  behavior preserved) and ``amip_erdm_v2.yaml`` / ``amip_x_ddc.yaml``
  declaring ``v2`` explicitly. ``XDDCWrapper`` accepts ``{"v1","v2"}``
  (its group order always matched upstream). ``CombinedModule`` needed
  no changes — it converts via ``unpack → dict → pack``, so it inherits
  layout correctness from its sub-wrappers.
- ``state_layout()`` on the rolling mixin + ``XDDCWrapper`` (upstream
  ``common/utils.py:state_layout`` mirror; ``nocean=0`` until 12f).
- **Tests** (18 new; 278 total green):
  ``test/models/amip_si/test_channel_layouts.py`` — bit-parity of v1/v2
  packs against vendored upstream ``assemble_input`` references,
  fork-layout bit-identity, c_grid ``[varying | constant]`` parity,
  pack/unpack round-trips for every layout, ``.mdlus`` roundtrip
  preserving ``channel_layout``, invalid-layout rejection, and the
  fixed v1↔v2 channel permutation property the translator relies on;
  plus 3 translator tests (default v1, explicit v2, frozen-target
  warning + no kwarg).
- **Cluster follow-ups — DELIVERED** *(2026-08-07, Delta jobs
  20916803 → 20920825, worktree
  ``/work/nvme/bdiu/awikner/physicsnemo-amip-v2``)*:
  * **Live re-translation sweep (job 20916803): 10 PASSED + 1
    documented XFAIL** (``SI_V_new``, pre-497827e ScalarEmbedder) over
    every real v1 Lightning ckpt at
    ``/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/`` —
    translate → ``.mdlus`` → reload → finite forward, with
    ``channel_layout="v1"`` provenance asserted through the roundtrip.
    (No stale persisted ``.mdlus`` artifacts existed to regenerate; the
    sweep is the re-translation validation.)
  * **v2-layout GPU smoke (job 20920825): PASSED** —
    ``RollingDiTWrapper(channel_layout=v2)`` + ``loss=erdm_v2``
    (sigma_data 1.0, churn on) rolling W=3 stage on the real 1981 AMIP
    Zarr, 2×A40 DDP, 20 capped iterations, finite losses, checkpoint
    saved, 8.1 GB peak. An uncapped probe (job 20919698) ran 200+
    batches with finite fluctuating losses before being cancelled.
  * **Four latent bugs found & fixed by the smoke campaign** (the
    rolling-window branch of ``train_diffusion.py`` had *never*
    executed): (1) ``SequenceDataset`` constructed with kwargs it
    doesn't accept; (2) ``_pack_window`` read bare keys where the
    dataset emits ``{key}_seq`` stacks, and ``calendar`` was missing
    from ``_SEQ_KEYS`` entirely; (3) no DDP sampler in window mode
    (every rank saw identical batches) — now a ``DistributedSampler``;
    (4) ``training.stages[*].max_iterations`` silently ignored by the
    diffusion trainer. Plus an infra recurrence: **wandb under DDP
    still hangs the NCCL watchdog** on the diffusion path (the
    regression-2 signature from commit ``1a1b843b``; the later
    every-rank strategy from ``fad7c0bb`` did not prevent it — rank 0
    SIGABRT after 480 s, peer rank reports ``value cannot be converted
    to type int without overflow``). The smoke runs
    ``wandb.enabled=false``; **flag for the ai-rossbypalooza team**:
    any multi-GPU diffusion run should disable wandb or revisit the
    auto-disable guard.
  * Delta venv note: ``muon`` had been pruned from the shared venv by a
    ``uv sync`` (the documented gotcha) — reinstalled.

### 12c — Daily-avg conversion + pre-coarsened 45×90 Zarr store

> **The daily-avg dataset has not been converted yet** (confirmed
> 2026-08-07). The fork's existing AMIP Zarr (Phase 11, 1978–2022) is
> the v1-era variable set; v2 trains on the **daily-averaged** archive
> (`*_24h` diagnostics, `*_monthly_interp` SST/ice, `global_mean_co2`,
> 24 h model steps over a 6 h-spaced store). 12c is therefore a
> two-step data delivery, not just a coarsening pass.

9. **Daily-avg H5 → full-res Zarr conversion**: extend
   `tools/data/amip/amip_h5_to_zarr.py` (or a sibling channel-config
   JSON) for the daily-avg variable set from upstream `ERDM_co2.yaml`;
   convert the archive to a new `amip_dailyavg` Zarr. Convert the
   norm stats (`normalize_mean_dailyavg.nc` / `normalize_std_dailyavg.nc`)
   into our stats layout. Note the source archive currently lives on
   Derecho scratch (see parked open item 2) — run the conversion while
   the source is reachable.
10. New tool `tools/data/amip/coarsen_zarr.py`: full-res daily-avg Zarr →
    45×90 Zarr (area/bilinear coarsening matched to upstream's
    `downsample_factor: 4` pipeline — must match v2's blur semantics
    because the downscaler's `x0` is defined by it). Boundary channels
    kept at full res (the model downsamples `c_grid` internally).
11. Registry entries in `hpc/data_registry.yaml` (`amip_dailyavg`,
    `amip_dailyavg_coarse`), sbatch conversion scripts,
    `conf/dataset/{amip_dailyavg,amip_dailyavg_coarse}.yaml`.
12. Loader benchmark coarse vs full-res Zarr (reuse the datapipe bench
    harness) recorded in `benchmarks/`.

**Effort**: ~2 days + conversion jobs.

**12c.9–11 delivered** *(2026-08-07; conversion runs in progress)*:

- **Channel config** [`tools/data/amip/configs/amip_dailyavg.yaml`](../../tools/data/amip/configs/amip_dailyavg.yaml)
  — role lists = the upstream `ERDM_co2` 151-channel contract
  (`specific_humidity`, 6 surface vars, `*_monthly_interp` SST/ice,
  CO₂+`DSWRFtoa_24h_lead` varying); the archive superset (soil, snow,
  dewpoint, SST/ice climatology + plain variants, `specific_total_water`,
  `vertical_velocity`) preserved via `extra_variables`. Verified against
  the live archive (68,676 files on Derecho, 225 keys/file) and the
  vendored stats. **No converter code change needed** — same
  `{year}_{idx:04d}.h5` + `/input` convention.
- **Norm stats**: `normalize_{mean,std}_dailyavg.nc` vendored verbatim
  from amip_v2@`e0b7b60` into `tools/data/amip/norm_stats/` (27 KB each,
  already in `ClimateNormalizer` layout — the plan's "convert the stats"
  step turned out to be a no-op).
- **Coarsen tool** [`tools/data/amip/coarsen_zarr.py`](../../tools/data/amip/coarsen_zarr.py)
  — the exact amip_v2 blur operator (`F.interpolate` bilinear,
  `align_corners=False`); state NaN-free (hard fail), boundaries
  NaN-filled (SST 270 K / ice 0) then coarsened (store self-contained
  for `c_grid_downsample=1`; `boundary_zarr_path` → full-res store for
  upstream-parity runs); extras skipped by default; coarse lat/lon =
  block means; provenance attrs; time rides the batch axis so output is
  independent of `--time-block`. 7 CPU tests incl. bit-parity vs a
  literal transcription of upstream `bilinear.py`.
- Registry entries (`amip_dailyavg`, `amip_dailyavg_coarse`), dataset
  configs (`conf/dataset/amip_dailyavg{,_coarse}.yaml` — coarse pairs
  with `model=amip_erdm_v2` + `loss=erdm_v2`,
  `normalize_diagnostic: True` per upstream), Derecho PBS scripts
  (per-year skip-if-exists loops).
- **12c.12 in progress**: 1981 test conversion + coarsening submitted on
  Derecho (jobs 7045083/7045084, dependency-chained; worktree
  `/glade/work/awikner/physicsnemo-amip-v2`). **Full-run blocker
  discovered**: Derecho scratch sits at 25.58 M of the ~26.2 M inode
  cap — the full-res 47-year Zarr (~3.4 M chunk files at time-chunk 1)
  cannot live there. Coarse store (~200 k files, time-block-64 chunks)
  fits anywhere. Routing options (interacts with the parked
  derecho-retire item): (a) per-year convert→ship→delete rotation via
  `replicate_tar.sh`, (b) Globus the ~3.9 TB raw archive to Stampede3
  ($SCRATCH has no inode cap — the Phase 11 conversion pattern) and
  convert there, (c) larger time-chunks (degrades random-access reads).
  Decision pending with the user.

### 12d — Loader-semantics parity (`forcing_from_raw` equivalent)

13. Centralize boundary→input assembly in one place on our side (extend
    `ClimateNormalizer` / a dedicated transform): z-score + SST
    rescaling hook + CO₂-pop into `c_scalar` + calendar row. All five
    consumers (`train_diffusion`, `validate_diffusion`,
    `eval_diffusion`, `inference`, `climatology_cli`) route through it —
    the "channel contract cannot fork" property upstream built.
14. Port `smooth_masked_boundary` (iterated masked Gaussian smoothing,
    `mask_fill` dict) as a composable transform beside
    `NanFillTransform`; config keys `smooth_nan_boundaries`,
    `smooth_sigma`, `smooth_kernel_size`, `smooth_n_iters`.
15. Constant-boundary cache: evaluate — Zarr already gives us cheap
    constant reads; port only if profiling says otherwise (document the
    call either way).

**Effort**: ~1.5 days.

**12d delivered** *(2026-08-10)*:

- **The choke point** —
  [`ForcingAssembler`](../../physicsnemo/experimental/datapipes/climate/forcing.py)
  performs upstream `forcing_from_raw` steps 3–4: pops spatially-uniform
  boundary channels (CO₂) out of the gridded stream and appends them to the
  calendar row, with a uniformity check, a `reduce={mean,first}` choice
  (`first` = upstream's literal `boundary[0,0,0]` read), and upstream's
  "no calendar ⇒ no scalar, CO₂ stays in the grid" degradation. Steps 1–2
  stay where the fork already had them (`ClimateNormalizer`; a
  `sst_rescaler` hook reserved for 12g, invoked *before* the pop so SST is
  still gridded, matching upstream's ordering).
- **One construction site** —
  [`dataset_setup.py`](../../examples/weather/ai_rossby/dataset_setup.py)
  (`build_forcing_pipeline`) now defines the fill → normalize → route chain
  once; all five consumers route through it (`train_diffusion`,
  `eval_diffusion` (inherits `_build_dataset`), `inference`,
  `climatology_cli`, `train` for the NaN-fill). Both normalization
  placements are first-class: `normalize_in_dataset=True` (chain includes
  the normalizer) and `False` (recipe normalizes per batch at use — the
  `as_normalizer()` proxy keeps one object for both directions so no
  rollout-helper signature changed). `assert_matches(wrapper)` is the
  anti-fork guard, called in all four model-owning recipes.
- **Two real divergences found and fixed by centralizing** (both pinned by
  regression tests):
  1. **Order** — `train_diffusion` composed `nan_fill(normalizer(sample))`,
     substituting *physical-unit* fills (SST 270 K) into z-scored space
     (≈ +20σ over every masked gridpoint). The other three recipes fill
     first. This would have hit the 12c AMIP configs, which are the first
     to set non-zero boundary fills (`sea_surface_temperature_monthly_interp:
     270.0`). Test `test_inverted_order_would_leave_the_raw_fill_value`
     documents the old behavior explicitly.
  2. **Scope** — `train_diffusion` / `climatology_cli` filled boundary NaN
     only; `train` / `inference` also fill prognostic-surface + diagnostic
     NaN (PLASIM/ERA5 SST land-NaN otherwise reaches the loss). The shared
     builder uses the broad scope everywhere (a no-op for the NaN-free AMIP
     daily-avg state).
- **Boundary smoothing (12d.14)** — `smooth_masked_boundary` +
  `_smooth_fill_channel` ported faithfully (iterated Dirichlet diffusion,
  circular longitude) and wired into `NanFillTransform` behind
  `smooth_nan_boundaries` / `smooth_sigma` / `smooth_kernel_size` /
  `smooth_n_iters`. Deviation from the plan's wording: it lives *inside*
  `NanFillTransform` rather than beside it, mirroring upstream where
  `_fill_mask` chooses hard-fill vs smooth-fill per variable from the same
  `mask_fill` dict — one place for the fill values instead of two.
  Boundaries only, matching upstream. `amip_dailyavg*` configs enable it
  (upstream defaults).
- **Constant-boundary cache (12d.15) — NOT needed, structurally.**
  Upstream's `.npz` exists because its per-timestep HDF5 layout would
  re-read time-invariant maps from every file. The fork's Zarr stores each
  constant once (no time axis) and
  `ClimateZarrDataset._eager_load_constants` reads them once at init,
  returning the *same tensor object* to every sample —
  `test_constant_boundary_cache.py` pins zero per-sample reads (identity
  check) plus the real hazard of a shared cached tensor: that the 12d
  chain stays copy-on-write and never mutates it.
- **Config**: `conf/model/amip_erdm_v2.yaml` now carries
  `scalar_routed_boundary_variables: [global_mean_co2]` + `scalar_dim: 3`,
  giving exactly upstream `ERDM_co2.yaml`'s contract (`c_grid_dim` 5 /
  `scalar_dim` 3, verified end-to-end against the wrapper).
- **Tests**: 32 new CPU cases (16 assembler/smoothing, 13 pipeline
  ordering + guard, 3 constant-boundary cache); **1089 → whole affected
  tree green**, including a fix for two pre-existing converter-test stubs
  that lacked the `write_batch` arg.

> **12c/12d seam — CLOSED via option (b)** *(user decision 2026-08-11)*.
> Upstream keeps forcings at **native 1° resolution** and reduces them inside
> the model with a stride-4 conv (`c_grid_downsample: 4`); only the *state* is
> coarsened. The fork now matches that:
>
> * `ClimateZarrDataset` sources **constant** boundaries from the boundary
>   store when one is configured, so the whole `c_grid` sits on one grid.
>   Previously constants came from the prognostic store and varying boundaries
>   from the boundary store, and the wrapper's `pack_c_grid` concat raised on
>   the 45×90 vs 180×360 mismatch — i.e. the mixed-resolution pairing was
>   documented in 12c but did not actually work. Now tested end-to-end
>   (dataset → pack → forward at `c_grid_downsample=4`).
> * [`extract_boundary_store.py`](../../tools/data/amip/extract_boundary_store.py)
>   copies just the 6 boundary variables out of the full-res archive
>   (~2.3 GB/yr vs ~58, bit-identical, **NaN preserved** so the runtime
>   masked-Gaussian fade runs exactly where upstream's does). That is what
>   makes 1° forcings affordable to move: ~105 GB for 1978–2022 instead of
>   the ~2.6 TB full archive.
> * `conf/dataset/amip_dailyavg_coarse.yaml` now points `boundary_zarr_path`
>   at that store and `conf/model/amip_erdm_v2.yaml` sets
>   `c_grid_downsample: 4` — so the **default** pairing is upstream parity.
>   The coarse store's own coarsened boundary arrays go unused (kept so it
>   remains usable standalone).
>
> Consequences: the runtime `smooth_nan_boundaries` knobs are now **live**
> (the boundary store carries NaN), and **12e's `boundary_pool_stats` is
> unblocked** — within-cell boundary variance still exists at 1°, which is
> exactly what that feature reads. The earlier recommendation (a)
> (`coarsen_zarr.py --smooth-boundaries`) stays available and tested for
> anyone using the standalone coarse store, but no re-coarsening was needed:
> the state channels are bit-identical either way, so the shipped coarse
> store required no rebuild.

### 12e — Backbone feature parity (RollingDiT)

16. Vendor `input_embed.py` (`RollingDiTInputEmbed`, budget mode + all
    sub-switches) and `output_head.py` (`RollingDiTOutputHead`, mix
    mode) into `physicsnemo/experimental/models/amip_si/layers/`.
17. Extend our `RollingDiT` with `input_embed=` / `output_head=` /
    `global_cond=` / `c_grid_cross_layers=` / `c_grid_cross_heads=`
    kwargs + `CausalForcingCrossAttentionBlock`; adopt upstream's
    `dit_block.py` factoring where it simplifies the vendored code.
    **`legacy` modes must remain bit-identical** to the current
    forward — pinned by a non-regression test against the existing
    committed reference outputs.
18. Port `tools/bench_input_embed.py` / `bench_output_head.py` as
    optional benchmarks under `benchmarks/`.
19. Tests: constructor sweep over `{legacy, budget} × {legacy, mix} ×
    {global_cond} × {cross_layers 0/2}`; forward-shape + finite; budget
    slice-width bookkeeping (`d_state = dim − d_boundary − d_calendar`);
    Muon param-group split still covers every new parameter exactly once.

**Effort**: ~2 days.

**12e delivered** *(2026-08-11)*:

- **Vendored layers** (`layers/input_embed.py`, `layers/output_head.py`),
  module/parameter names preserved so upstream-trained checkpoints translate
  1:1: `RollingDiTInputEmbed` (budget mode + `SourceNorm`,
  `BoundaryEncoder` conv1/conv2 with spherical padding + exact pooled
  mean/std, `ScalarForcingEmbedder` with the reserved `d_co2` affine head,
  `ColumnStateEncoder`) and `RollingDiTOutputHead` (σ-gated mixture over
  `num_experts` readouts + `ColumnDecoder`), both carrying the `nocean`
  plumbing 12f needs (inert at `nocean=0`).
- **`RollingDiT` extended** with `input_embed=` / `output_head=` /
  `global_cond=` / `c_grid_cross_layers=` / `c_grid_cross_heads=` /
  `window_size=` / `state_layout=`, plus
  `CausalForcingCrossAttentionBlock` (temporal causal cross-attention from
  the hidden state to the in-window forcing stream, zero-init gate).
  `state_layout` is **derived by the wrapper** from its variable lists, never
  restated in a config.
- **Legacy is bit-identical, proven not assumed.** A reference was generated
  from the pre-12e commit in a temp worktree and committed as
  `test/models/amip_si/data/rolling_dit_legacy_v1.pt`: default kwargs
  reproduce the same 48 state-dict keys, the same 244,260 parameters, and a
  **bit-for-bit equal** forward. Budget/mix modes leave the replaced
  submodules unbuilt (no dead weights in the state dict).
- **Muon grouping fixed** — a latent bug this feature would have hit: the
  wrapper's `muon_param_groups` listed only the legacy modules, so under
  12e the new `input_embed` / `output_head` / `forcing_blocks` parameters
  would have been **silently dropped from the optimizer**. Now mirrors
  upstream: cross-attention Linears → Muon, but its 2-D `temporal_pos` /
  `query_pos` position tables → AdamW. Tested: every backbone parameter in
  exactly one group, for all five feature combinations.
- **Benchmarks ported** with measured results in
  `benchmarks/.../amip_si/RESULTS_projections.md`. They confirm upstream's
  motivation quantitatively: the legacy fixed-`Linear` head is
  representationally bottlenecked on the σ-dependent EDM target (weighted
  MSE 787 → 455 for a σ-conditioned gain, → 134 with the column decoder),
  and raising `d_boundary` from ¼ to ⅜ of `dim` lifts boundary influence
  from 0.61 to 1.07 relative to the state.
- **Config**: `conf/model/amip_erdm_v2.yaml` now mirrors upstream
  `ERDM_co2.yaml`'s model block exactly (budget 640+256+128, `d_co2` 48,
  column state encoder, conv2 boundary encoder + pooled stats + static bias,
  `global_cond`, 4 cross-attention layers, mix head K=2) — **561.6M
  parameters**, matching upstream's ~500M forecaster target.
- **Tests**: 63 new CPU cases (`test_rolling_dit_features.py`) — legacy
  bit-identity, a 36-way `{embed} × {head} × {global_cond} × {cross}`
  forward sweep, budget slice bookkeeping, cross-attention identity-at-init
  **and causality over the window** (perturbing a later frame's forcing
  leaves earlier outputs bit-unchanged while the last frame does respond),
  global-cond column semantics, head zero-at-init + gating, column layout
  validation, and the Muon coverage checks. **1224 tests green.**

> **Note on small widths** (real, not a test artifact):
> `state_encoder='column'` needs `d_state >= 24` and a `scalar_dim >= 3`
> budget needs `d_calendar > 8`, because each sub-block has an 8-channel
> floor. Upstream's `dim=1024` clears both by far; tiny unit-test models must
> pass explicit budgets. Kept upstream's rounding math rather than
> special-casing, since changing it would change parameter shapes and break
> checkpoint compatibility.

### 12f — ERDM scheduler parity: ocean channels + warm start

20. Port the "Ocean channels" section of upstream `erdm.py`
    (`ocean_truth`, `append_ocean_target`, `pad_state`, `strip_ocean`,
    `impose_ocean`) into our `ERDMScheduler`; `compute_loss` gains
    `return_parts` and the recipe logs `train/loss_ocean` separately.
21. Dataset: serve `W+1` boundary frames for rolling-window samples
    (`[:W]` forcing, `[1:]` ocean truth) — extension to
    `SequenceDataset` boundary emission.
22. **Partial-checkpoint warm start**: `training.partial_checkpoint` in
    `train_diffusion.py` + a `load_partial_weights` helper with the
    upstream guarantee surfaced (report skipped keys loudly; zero
    skipped keys expected for ocean-variant warm starts).
23. Tests: append/strip/impose round-trip; imposition-is-total (predicted
    ocean never survives a roll); warm-start zero-skipped-keys on a
    synthetic ERDM→ERDM-ocean pair; Delta A40 smoke of one ocean-variant
    mini-epoch.

**Effort**: ~1.5 days.

### 12g — SST forcing suite

24. Vendor `data/sst_forcing.py` logic as a transform/normalizer hook
    (slots into 12d's choke point); port
    `scripts/make_sst_climatology.py` →
    `tools/data/amip/make_sst_climatology.py` (reads our Zarr; emits the
    harmonic-fit climatology artifact — keep upstream's `.npz` format so
    artifacts are interchangeable).
25. Dataset config keys: `sst_anomaly_channel`, `scalar_forcing`,
    `sst_anomaly_scale`, `sst_scalar_scale`, `sst_climatology_path`;
    register the artifact in the data registry.
26. Tests: anomaly ≈ 0 over land / continuous across coasts on a
    synthetic coastline; append vs replace channel counts; scalar
    mutual-exclusion (`co2` vs `global_mean_sst`); `auto` mode matches
    historical behavior; scale-override arithmetic.

**Effort**: ~1 day.

### 12h — Combined/rollout rewrite, translator, gates

27. Rewrite `CombinedModule` on upstream's `windowed_init` /
    `windowed_step` semantics; `inference.py`/rollout path drives it via
    our existing `AsyncForecastWriter` with month-buffered output
    (upstream's thread-pool NetCDF writer maps 1:1 onto what we already
    have).
28. **Translator for v2 checkpoints**: new key layouts (input_embed /
    output_head / cross-attn / ocean modules), `ERDM_fancy` and
    `ERDM_ocean` variants, `combined` two-ckpt loading. **No v2-trained
    checkpoints exist yet** (confirmed 2026-08-07) — unit-test the v2
    path on synthetic Lightning-style ckpts now; live validation is a
    standing follow-up that runs when the first real v2 training lands.
    In the meantime the **12b permutation shim is the live-validated
    path**: exercise it against the existing real v1 ERDM / x_DDC /
    Combined checkpoints (Midway3 inventory), asserting forward
    equivalence with the frozen-contract loading of the same weights.
29. **Health gates**: adopt the useful subset of `check_repo.py` as
    tests — an instantiate-all-AMIP-configs test with a state-dict
    shape-signature hash (regression gate against silent widening, the
    exact failure mode upstream's ocean warm-start doc warns about), and
    a synthetic-tensor smoke over every supported config. Deadscan
    omitted (pytest + CI already cover imports/compile).
30. Update `phase8e_midway3_checkpoint_inventory.md` +
    `PANGUWEATHER_MIGRATION.md`/README pointers to describe the dual
    contract and the supported config set.

**Effort**: ~2 days.

**Total**: ~12–13 developer days + conversion/validation jobs.

## Dependency graph

```
12a (audit/configs) ──→ 12b (contract) ──→ 12e (backbone) ──→ 12f (ocean) ──→ 12h (combined/translator/gates)
                          │                                      ▲
12c (coarse Zarr) ────────┤                                      │
                          └──→ 12d (loader semantics) ──→ 12g (SST) ──┘
```

12a and 12c are independent starters; 12b gates everything
contract-shaped; 12d gates the SST suite; 12h consumes all of it.

## Tests contract (normative, per `hpc/delta.md`)

Every sub-phase ships CPU unit tests + a Delta `gpuA40x4-interactive`
smoke. Parity with upstream is pinned by vendored-reference cross-checks
(synthetic tensors, no amip_v2 import at test time) for: pack/unpack
layout, budget input-embed forward, mix output-head forward, ocean
append/strip/impose, SST anomaly transform. The frozen v1 families'
existing suites must stay green after every sub-phase.

## Open items / questions for the user

1. ~~**v2 checkpoint inventory**~~ — **resolved 2026-08-07: none exist
   yet; only v1 checkpoints.** 12h.27 adjusted: synthetic-ckpt unit
   tests for the v2 path now, live validation deferred to the first
   real v2 training run; the 12b permutation shim is live-validated
   against the existing v1 checkpoints instead.
2. **Derecho-resident artifacts** — **parked per user (2026-08-07),
   ignore for now.** For the record: amip_v2's configs point at
   `/glade/derecho/scratch/{awikner,ayz}/ERA5/...` (daily-avg H5 archive
   + fast stores + logs), and Derecho scratch is retiring
   (`derecho-retire-rehome-to-delta.md`, DEFERRED). Revisit alongside
   that item; until then 12c plans against the fork's own converted
   AMIP Zarr as input (subject to item 4).
3. ~~**Rebaseline cadence**~~ — **resolved 2026-08-07: `e0b7b60` is the
   baseline.** Upstream work past it is out of Phase 12 scope (see
   Baseline pin).
4. ~~**Daily-avg vs 6-hourly**~~ — **resolved 2026-08-07: the daily-avg
   dataset has not been converted yet.** 12c expanded into a two-step
   data delivery: daily-avg H5 → full-res `amip_dailyavg` Zarr (+
   dailyavg norm stats), then coarsen to 45×90.
