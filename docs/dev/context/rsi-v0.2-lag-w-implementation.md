<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RSI proposal v0.2 implemented: the lag-W sample anchor, and the A0 / A2-L baselines (2026-09-10)

Branch `worktree-rsi-ablation-review` (on top of `rsi-drift-diagnosis`),
five commits: scheduler core, recipe + validator, artifact tool + configs +
job scripts, evaluation drivers, probes + docs. Everything below is shipped
to Polaris `$R` (`backups_20260910_v02/` holds the replaced copies).

## 1. What changed and why

The v0.1 rolling interpolant anchored slot `w` on frame `w-1`. At inference
the fresh back slot's anchor is therefore the state head's readout of slot
`W` at local time `1/(2W)`: a **conditional mean**. Iterating it contracts
the free-run climate (one-year surface plateau 611-795 vs ERDM's 63; no
perturbation-based mitigation repaired it; see
[rsi-anchor-fix-report-for-proposal-revision](rsi-anchor-fix-report-for-proposal-revision.md)).
Proposal v0.2 makes the coupling a **lag-W sample anchor**: slot `w`
interpolates from `y_{w-W}` to `y_w`, so the fresh slot is anchored on the
frame **emitted** at the same roll, a completed sample, and every anchor is
a sample by construction.

### `RSIScheduler(anchor_lag=L)` (`physicsnemo/experimental/diffusion/rsi.py`)

| item | v0.1 (L = 1, unchanged default) | v0.2 (L = W) |
|---|---|---|
| training sample | `W+1` frames | `W+L` frames (`2W`); `anchor_frames = L`, `history_frames = (L-1) + pushforward_rolls` |
| anchors / targets | `y[:, :W]` / `y[:, 1:]` | `y[:, :W]` / `y[:, L:]` |
| fresh slot anchor | readout slot `W` (index `W-1`) | readout slot `W-L+1` (index `W-L` = 0 at `L = W`: the emitted frame) |
| `Gamma(0)` | 1-step increment scale | L-step increment scale (`noise_scale_path` built with `--lag L`) |
| ocean imposition at `t = 0` | anchor-time truth = the conditioning window | anchor-time truth = the boundary `L` steps behind the own-time window (`anchor_bnd_win`; required at `L > 1`) |
| init | `init_frames = W+1` (`y_0 .. y_W`) | `init_frames = W+L` (`y_{1-L} .. y_W`) |
| drivers | forcing trajectory from the IC | `L-1` pre-IC boundary frames prepended (`traj_lead`); IC lower bound `(L-1)` steps |
| `reduce_to_erdm` (A1) | lag-independent | lag-independent (pinned by test) |

`_gather_window` now clamps negative starts to frame 0 (they wrapped to the
end of the trajectory before). Pushforward rolls (`pushforward_rolls = K`)
seed from `W+L` truth frames and slice the forcing / own-time / anchor-time
windows shifted by `L-1`. The `RSI_TRACE_PATH` trace records the fresh-slot
anchor from slot `W-L`, so at `L = W` `anchor_std / emit_std` is 1 by
construction (`polaris_results.py trace` notes this).

Drivers: `validate_diffusion.py` (`first_past = (L-1) * step`, trajectory
prepended, `traj_lead` passed only when non-zero), `inference.py`,
`rollout.py` (`_traj_windows` returns the lead; `run_rollout(traj_lead=)`
gathers `k+lead`, `k+1+lead` and the anchor-time window; the RSI resume crash
on `eps_prev = None` is fixed), `CombinedModule.windowed_step(...,
anchor_bnd_win=)`, `rsi_drift_probes.py` (lag-aware init / forcing rows /
readout split). `train_diffusion.py` needed no plumbing change (the
`history_frames` route already delivers `W+L+K` frames and the ocean stack is
the own-time boundary of every frame); it records `rsi_anchor_lag` in the
checkpoint metadata and refuses a resume at a different lag.

Configs: `loss/rsi_a2l.yaml` (`anchor_lag 6`, state head + EDM readout,
`gamma 1.0 -> 0.04` in W-step increment units, `sigma_c_sstpred153_lag6.pt`,
`fresh_noise_scale 1.0`, no anchor perturbation, no pushforward),
`sampler/rsi_a2l_sstpred.yaml` (key-for-key mirror, pinned by
`test_rsi_ladder_configs.py`), `loss/rsi.yaml` now carries an explicit
`anchor_lag: 1` (rung A2, diagnostic). Job scripts:
`polaris_ablation_train_b40.pbs ABL=a2l` (+ `ABL_MAX_ITERS` smoke knob),
`polaris_rsi_drift_eval_multi_phase5.pbs` cfg `a2l`.

Tests: `test/diffusion/test_rsi_scheduler.py` + `test_rsi_ocean.py`
(parametrized over `L in (1, 2, W)`), `test/recipes/ai_rossby/
test_train_diffusion_rsi.py`, `test_validate_diffusion.py`,
`test_rsi_ladder_configs.py`, `test_rollout_cascade.py`,
`test/models/amip_si/test_combined_windowed.py`,
`test/tools/data/test_make_rsi_delta_scales.py`. The `L = 1`
parametrizations are bit-identical to the pre-change behaviour.

## 2. Finding: the shipped noise-scale artifacts were 6-hour, not one-step, increments

`tools/data/amip/make_rsi_delta_scales.py` (which produced
`sigma_c_fancy154.pt` / `sigma_c_sstpred153.pt` on 2026-09-02) computed
`np.diff` over store ROWS. The AMIP daily-average stores are 6-hourly
(`data_timedelta_hours: 6`, 1460 rows a year) under a 24-hour model step, so
those artifacts are the std of the **6-hour** increment of 24-hour running
means. Reproduced exactly on Polaris on 2026-09-10 (`--lag 1 --step-rows 1
--compare`: ratio 1.000 on all 153 channels, which also verifies the shipped
file's channel order). The tool now counts the lag in model steps
(`--step-rows` derived from the store cadence) and writes a `.json` sidecar.

| artifact (`$AI_ROSSBY_DATA/norm_stats/`) | increment | mean over 153 ch | surface_pressure (ch 1) | v@900 (ch 43) | ratio to shipped |
|---|---|---|---|---|---|
| `sigma_c_sstpred153.pt` (shipped, used by every A2 run) | 1 row = 6 h | 0.121 | 0.017 | 0.29 | 1 |
| `sigma_c_sstpred153_lag1.pt` (new, for the record) | 1 step = 24 h | 0.382 | 0.058 | 0.90 | ~3.2x |
| `sigma_c_sstpred153_lag6.pt` (new, **A2-L**) | 6 steps = 6 days | 0.649 | 0.111 | 1.28 | 1.7x (fast) to 12x (upper-level geopotential) |

Consequence for the record: every lag-1 A2 run (`gamma_0 = 1.0` "in
increment units") injected about a third of the one-day increment std into
the fresh slot, and the drift campaign's `fresh_noise_scale 1.45` sweep was
partly compensating for this. A2-L uses the correctly scaled W-step artifact.

## 3. The runs (Polaris, 10 nodes x 4 ranks, global batch 40, 24 epochs, seed 0)

**A0** -- `ABL=a0 ABL_RECIPE=upstream`, `model=amip_erdm_sst_pred loss=erdm_v2`,
`checkpoints_a0_sstpred_b40`: Muon x10, base lr 5e-4 (upstream's 5e-5 scaled
linearly), momentum 0.95, weight decay 0.01, StepLR 0.95/epoch after a
1-epoch warmup, fp32 + TF32, EMA 0.99, GC 14 levels, validation every epoch.
Chain of 13 links submitted 2026-09-10 22:0x UTC (7603431-7603443). **Links
7603431-7603436 all died within a minute**: rank 18 on `x3208c0s37b1n0`
raised `CUDA-capable device(s) is/are busy or unavailable` inside
`init_process_group` although the matmul preflight passed on all 40 ranks.
The host was added to `$R/bad_nodes.txt` (the script's blacklist check now
resubmits away from it) and six replacement links were appended
(7603477-7603482, `afterany` on 7603443). Link 7603437 started 22:35 UTC on a
clean allocation: `world_size=40`, `steps_per_epoch=1315`, ~3.5 s/step
(1315 steps ~ 77 min, so a 3-hour link holds ~2 epochs plus validation).

**A2-L** -- `ABL=a2l ABL_RECIPE=bundle`, `model=amip_rsi_sst_pred
loss=rsi_a2l`, `checkpoints_a2l_sstpred_b40`: the recipe the lag-1 A2 bundle
run reached batch-4 loss parity with (lr 5e-4 / Muon 5e-3, 1-epoch warmup,
cosine to 5e-5 over 24 epochs, momentum 0.85, weight decay 0.01, fp32 +
TF32, EMA 0.99, GC 14), validation every epoch. Smoke first (debug queue, 1
node, `checkpoints_a2l_smoke`): job 7603511 with `ABL_TARGET=1` failed in
`make_scheduler` because a 1-epoch cosine equals the 1-epoch warmup (a
smoke-config artifact; use `ABL_TARGET=24 ABL_MAX_ITERS=60`); job 7603531
passed: stage line `anchor_lag=6, anchor_frames=6, history_frames=5 (loader
window 12 state frames)`, `steps_per_epoch=13138` at 4 ranks (13143 at
lag 1), finite losses (2.3e4 -> 2.0e4 over the first batches, ~3 s/step on
one node), epoch-end checkpoints `RollingDiTWrapper.0.{1,2}.mdlus` written,
and the in-training rollout validation ran through the lag-6 path (12 init
frames, 5 pre-IC boundary frames, horizon 10, 4 ICs x 10 members). Chain of
13 links submitted 23:1x UTC: 7603546-7603558 (`afterany`, prod -> small).

Not touched: `rsi-sstpred-prod24` (7586612, 1 node, capacity queue, running
since 2026-09-06, with 7599793 held behind it) -- the batch-4 lag-1 A2
production training. It was not part of the September test runs and was left
running.

## 4. Evaluation protocol (v0.2 section 5)

Through `polaris_rsi_drift_eval_multi_phase5.pbs` (`EVAL_JOBS=name:ckpt:epoch:cfg`,
cfg `erdm` for A0, `a2l` for A2-L, `RSI_TRACE_PATH` on): one-year at epochs
12 and 24, five-year at 24; plateau, shrinkage alpha per channel class, level
bias, spread, ten-day validation. A2-L predictions: anchor/emitted amplitude
ratio 1.0 from roll 1 (by construction), no fast-wind onset, day-1 skill near
ERDM with the slow channels ahead. A0 is checked against the upstream
ep17/ep23/ep24 evaluations on disk (`$R/eval_bias5yr_e17..e24`) at epochs 12
and 24 as in [rsi-a0-a1-batch40-plan](rsi-a0-a1-batch40-plan.md) section 6.

## 5. Gotchas met

- `test_train_diffusion_rsi_smoke.py` runs the recipe in a subprocess, which
  imports the venv's *installed* (editable, main-checkout) `physicsnemo`;
  from a worktree run it with `PYTHONPATH=<worktree>` or the new
  `anchor_lag` key is "unexpected".
- The per-level lazy zarr slice in the old artifact tool re-read every
  level-spanning chunk 26 times; each upper-air variable is now read once per
  year (the three artifacts took ~8 minutes on a loaded login node).
- `qsub` from a chain link's own resubmission must carry every new knob in
  its `-v` list (`ABL_MAX_ITERS` added).
