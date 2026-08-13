<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-License-Identifier: Apache-2.0
-->

# wandb × DDP NCCL-watchdog hang — fix plan

Status: **COMPLETE (W1–W4, 2026-08-07) — root cause fixed, multi-GPU wandb re-enabled** · Created:
2026-08-07 (during Phase 12b cluster validation) · Owner: TBD

## Symptom

Multi-GPU (DDP/NCCL) launches of `examples/weather/ai_rossby/`
recipes with `wandb.enabled=true` hang in **`DistributedDataParallel.__init__`**
— the *first* NCCL collective (`_verify_param_shape_across_processes`)
— until the NCCL watchdog kills the job:

- Rank A: `ProcessGroupNCCL's watchdog got stuck for 480 seconds
  without making progress ... could be triggered by another thread
  holding the GIL inside a CUDA api` → **SIGABRT** (after a debug-dump
  attempt that itself takes ~8 min).
- Rank B (peer fallout): `RuntimeError: value cannot be converted to
  type int without overflow` from the same verify call — **garbage
  read from the dying peer, NOT the torch-2.11 overflow regression**
  (venv confirmed at torch 2.10.0+cu128). The two regressions share an
  error string; check torch's version before assuming the pin drifted.

## Evidence (Delta, 2×A40 `gpuA40x4-interactive`, worktree
`/work/nvme/bdiu/awikner/physicsnemo-amip-v2`, torch 2.10.0, wandb 0.27.0)

| Job | Config | wandb | Outcome |
|---|---|---|---|
| 20916803 | `train_diffusion.py`, RollingDiT v2-layout smoke | on (offline) | hang signature on one rank (partially masked by a missing-`muon` crash on the peer) |
| 20918380 | same, muon restored | on (offline) | **clean repro**: rank 0 watchdog-stuck 480 s → SIGABRT; rank 1 overflow |
| 20919698 | same + `wandb.enabled=false` | **off** | DDP init instant; trained 200+ finite batches until walltime-scancel |
| 20920825 | same, off + `max_iterations` fix | **off** | **PASSED** end-to-end |

Toggling exactly one variable (wandb) flips the outcome — root cause
is wandb's presence during/before the first NCCL collective.

## History (why this is a *recurrence*, not a new bug)

1. **`1a1b843b`** (2026-07-07) fixed two regressions: pinned
   `torch<2.11`, and **auto-disabled wandb when `world_size>1`**
   (escape hatch `wandb.allow_multigpu=true`) after diagnosing wandb's
   background threads (service IPC / console capture / GPU-stats
   monitor) grabbing the GIL inside CUDA calls and stalling NCCL
   progress. Verified on Delta `gpuA100x4`.
2. **`fad7c0bb`** ("Wire ai-rossby train/val logging to wandb as the
   default backend") **removed the auto-disable** and adopted the
   *init-on-every-rank* strategy (`train.py:_maybe_init_wandb`; rank 0
   drives `LaunchLogger`, other ranks open throwaway offline runs for
   thread-jitter symmetry, mirroring the PanguWeather reference
   trainer). `docs/dev/context/sfno-ddp-requirements.md` §2 records
   "wandb works multi-GPU now".
3. **Phase 12b** (2026-08-07): the hang is back on the diffusion path
   despite the every-rank strategy — and at **DDP init**, not the
   mid-epoch all-reduce desync the strategy was designed against. The
   every-rank symmetry argument cannot help at init: the collective
   blocks on whichever rank's wandb threads stall it, symmetric or not.

## What differs from the validated (July, A100, `train.py`/SFNO) setup

Testable deltas, any of which may be the trigger:

- **Recipe path**: `train_diffusion.py` (hang) vs `train.py`
  (validated). Same `_maybe_init_wandb` helper, different surrounding
  code and model classes.
- **Partition/GPU**: A40 (`gpub` nodes) vs A100.
- **wandb version drift**: venv is a shared, evolving env
  (ai-rossbypalooza work); wandb is at 0.27.0 today — the July
  validation ran whatever was installed then. wandb's service/IPC
  internals change across minor versions.
- **`initialize_wandb`** (`physicsnemo/utils/logging/wandb.py`) sets
  only `wandb.Settings(init_timeout=...)` — the GPU-stats monitor and
  console capture (the two thread families named in the original
  diagnosis) are both left ON.

## Plan

### W1 — Immediate mitigation (restore the proven guard) — ~2 h

Restore `1a1b843b`'s behavior inside `_maybe_init_wandb` (shared by
`train.py` and `train_diffusion.py`, so one change covers both):

1. When `dist.world_size > 1` and NOT `cfg.wandb.allow_multigpu`
   (new key, default `false`): log one loud line
   (`"wandb auto-disabled under DDP (world_size=N); set
   wandb.allow_multigpu=true to override — see
   docs/dev/wandb_ddp_hang_fix_plan.md"`) and return `False` on every
   rank. Single-GPU behavior unchanged (full wandb).
2. Multi-GPU runs keep console logging + the bench TSV. Post-hoc
   `wandb sync` of offline dirs is NOT a workaround (offline mode still
   spawns the threads — that's what jobs 20916803/20918380 ran).
3. `conf/config.yaml`: add `wandb.allow_multigpu: false` with a comment
   pointing here.
4. Unit tests (CPU): guard returns False on every rank when
   `world_size>1` + flag unset; returns the normal per-rank result when
   flag set or `world_size==1`. (Mock `dist`; no real wandb needed.)
5. Remove the now-redundant `wandb.enabled=false` override from
   `hpc/scripts/smoke_amip_v2_layout_2xA40.sbatch` (the guard covers
   it), keeping a comment breadcrumb.
6. Doc sync: update `docs/dev/context/sfno-ddp-requirements.md` §2 and
   the CLAUDE.md gotcha line — "init wandb on every rank" is *not*
   sufficient; the guard is authoritative until W3 lands. Cross-link
   this plan.
7. **Coordinate with the ai-rossbypalooza team** (their multi-GPU runs
   share these recipes/venv): announce the guard + escape hatch before
   landing, since anyone relying on multi-GPU wandb dashboards loses
   them until W3.

**W1 delivered** *(2026-08-07, branch ``ai-rossby-amip-v2``)*: guard in
``train.py:_maybe_init_wandb`` (rank-0 warning, silent on other ranks;
missing ``allow_multigpu`` key defaults to guarded), ``wandb.allow_multigpu:
False`` in ``conf/config.yaml`` with rewritten multi-GPU comment, 6 CPU
unit tests (``test/recipes/ai_rossby/test_wandb_ddp_guard.py`` — guard
on/off per world_size, escape hatch, rank silence, enabled=False
short-circuit, missing-key default), smoke sbatch's ``wandb.enabled=false``
override removed (guard covers it), ``sfno-ddp-requirements.md`` §2 +
CLAUDE.md gotcha updated. 232 recipe/amip tests green.
**Live-validated on the exact hang-repro config** (Delta job 20921623,
2×A40, no manual wandb override): guard warning logged, DDP init
cleared in ~14 s, smoke passed end-to-end.

**Item 7 (palooza heads-up) — CLOSED 2026-08-13, no action needed.** It existed
because W1's guard *disabled* multi-GPU wandb by default, which would have taken
the palooza team's dashboards away silently. W2 found the real root cause
(rank-0-only ``_maybe_init_wandb``), W3/W4 fixed it, and the default went back to
``allow_multigpu: True`` after the 93-minute validation — so nothing is taken
away. Checked at close: ``examples/weather/ai_rossbypalooza/train.py`` already
initializes wandb on **every** rank (``mode="disabled"`` on non-zero ranks rather
than skipping the init), i.e. the team independently has the correct pattern. The
only thing worth mentioning at merge is that ``wandb.allow_multigpu=false``
exists as an escape hatch if a DDP-init hang ever reappears.

### W2 — Root-cause isolation matrix (Delta, one sbatch, ~1 GPU-hour)

> **W2 discovery (2026-08-07, before running the matrix):**
> `train_diffusion.py` called `_maybe_init_wandb` **only on rank 0**
> (`if dist.rank == 0:` around the init) — precisely the asymmetric
> configuration the every-rank strategy exists to prevent, and the
> configuration the July diagnosis identified as the deadlock trigger.
> `train.py` calls it unconditionally on every rank. This is now the
> **primary hypothesis**: it explains why the July validation
> (`train.py`, every-rank, A100) passed while the diffusion path hangs,
> at the SAME wandb version. **The version-drift cell is dropped**:
> wandb was already 0.27.0 in `1a1b843b`'s lockfile — no drift
> occurred. The matrix below is revised accordingly: every cell runs
> the every-rank call-site fix; the pre-fix rank-0-only baseline is not
> re-run (documented failing twice, jobs 20916803/20918380).
>
> Revised cells (`hpc/scripts/isolate_wandb_ddp_hang_2xA40.sbatch`):
> M1a/b/c = every-rank fix + wandb on, three repeats (primary);
> M2 = + stats monitor off; M3 = + console off;
> M4 = `wandb.init_after_ddp=true`; M5 = `wandb.nonzero_rank_mode=disabled`;
> M6 = `train.py` control on A40.

One `hpc/scripts/isolate_wandb_ddp_hang_2xA40.sbatch` running a
sequence of ≤3-min 2×A40 micro-trainings (tiny RollingDiT, 5 capped
iterations each, `wandb.allow_multigpu=true` so the guard doesn't mask
the experiment), with the NCCL flight recorder armed
(`TORCH_NCCL_DUMP_ON_TIMEOUT=1`,
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120`, per-rank dump dir) and a short
`timeout` wrapper so a hung cell records FAIL and moves on:

| Cell | Variable under test |
|---|---|
| 0 | baseline, wandb on — repro control (expect FAIL) |
| 1 | GPU-stats monitor off (`wandb.Settings(x_disable_stats=True)` / env `WANDB_DISABLE_GPU=true` per wandb 0.27 naming — check both) |
| 2 | console capture off (`settings.console="off"`) |
| 3 | **init wandb AFTER the DDP wrap** (move the `_maybe_init_wandb` call; the first collective completes before any wandb thread exists) |
| 4 | rank>0 `mode="disabled"` instead of `"offline"` (no threads on non-zero ranks; deliberately re-tests the asymmetry claim) |
| 5 | wandb service/start-method variants (`WANDB_START_METHOD`, `x_disable_service` if still supported in 0.27) |
| 6 | cells 0+3 repeated on `train.py` (tiny SFNO) — is the diffusion path special, or did the July validation simply predate the trigger? |
| 7 | wandb pinned to the July-era version (recover from `uv.lock` history at `1a1b843b`/`fad7c0bb`) — version-drift check |

Deliverable: findings table appended to this doc + flight-recorder
dumps for every FAIL cell (each dump names the stuck rank's last
collective + stack — see the diagnosis tip in
`sfno-ddp-requirements.md`).

Expectation to confirm or kill: cells 1–3 target the GIL-in-CUDA-API
mechanism directly; cell 3 is the strongest candidate for a durable
fix because it removes the init-time race categorically regardless of
which thread family is guilty.

**W2 findings** *(2026-08-07, Delta job 20921753, 2×A40, wandb 0.27.0,
torch 2.10.0)*:

| Cell | Configuration | Result |
|---|---|---|
| (pre-fix baseline) | rank-0-only init + wandb on | **FAIL ×2** (jobs 20916803/20918380 — watchdog hang at DDP init; not re-run) |
| M1a / M1b / M1c | **every-rank call-site fix** + wandb on (offline) | **PASS ×3** (200 s / 65 s / 64 s; M1a includes first-run warmup) |
| M2 | every-rank + GPU/system-stats monitor off | PASS (64 s) |
| M3 | every-rank + console capture off | PASS (65 s) |
| M4 | `wandb.init_after_ddp=true` | PASS (63 s) |
| M5 | `wandb.nonzero_rank_mode=disabled` | PASS (66 s) |
| M6 | `train.py` control + wandb on | machinery PASS — wandb init + DDP wrap + first `train_step` all reached; the cell then errored on an unrelated model/dataset level-count mismatch (13 vs 10) in the cell's config pairing, 43 s, no hang |

No NCCL flight-recorder dumps were produced (no hangs anywhere).

**Verdict: root cause CONFIRMED = the rank-0-only `_maybe_init_wandb`
call in `train_diffusion.py`.** With the every-rank call-site fix
(landed in commit `19f36128`), wandb-on DDP passes 3/3 on the exact
hang-repro config, and every sub-variant (stats off / console off /
init-after-DDP / thread-free non-zero ranks) also passes — none of
them is needed for the fix; the knobs remain available as belt-and-
suspenders options. The July validation-vs-12b-hang discrepancy is
fully explained: `train.py` always had the every-rank call,
`train_diffusion.py` never did.

### W3 — Durable fix + re-enable — ~half day + validation runs

Based on the W2 table (decision rule: prefer ordering fixes over
settings fixes over version pins):

1. Land the winning combination inside `_maybe_init_wandb` /
   `initialize_wandb` (likely: init-after-DDP + stats monitor off +
   console off; keep every-rank vs rank0-only per cell 4's verdict).
2. Validation gate to flip `wandb.allow_multigpu` default → `true`
   (or drop the guard):
   - the 12b v2-layout smoke with wandb ON passes 3/3 consecutive
     submissions (init-time hang is intermittent-shaped: demand
     repeats);
   - one ≥1 h 2-GPU run (real config, e.g. `sfno_plasim` or the ERDM
     smoke uncapped) survives with wandb ON — this re-tests the
     *mid-epoch* desync that motivated the every-rank strategy, which
     W2's short cells cannot see.
3. If no combination passes: guard stays permanently, multi-GPU logging
   remains console+TSV, and we file the flight-recorder evidence
   upstream (wandb + pytorch issue trackers) — the dumps from W2 are
   exactly what both projects ask for.

**W3 delivered** *(2026-08-07)*: the durable fix is the every-rank
call-site in `train_diffusion.py` (landed with W2, commit `19f36128`)
— no settings changes needed. Re-enable gate met in full:

- 3/3 wandb-on passes on the hang-repro config (W2 cells M1a/b/c);
- **93-minute full-epoch 2×A40 run with wandb ON: PASSED** (Delta job
  20921857, non-interactive `gpuA40x4`, 729 batches, both ranks'
  offline runs created, checkpoint saved, flight-recorder dir empty —
  zero hangs). This covers the mid-epoch desync regime the short cells
  cannot see.

`wandb.allow_multigpu` default flipped to `true` in `conf/config.yaml`
— multi-GPU wandb dashboards are restored. The guard code remains as
an opt-out (`allow_multigpu: false` re-arms it); the code-level
fallback for configs *missing* the key stays conservative (guarded).
Validation sbatch kept at
`hpc/scripts/validate_wandb_multigpu_2xA40.sbatch` for reruns.

### W4 — Regression guards + docs — ~1 h

1. The wandb-ON smoke variant from W3.2 becomes a keepable sbatch knob
   (`WANDB_ON=1`), documented in `hpc/delta.md`.
2. Final state of `sfno-ddp-requirements.md` §2 + the CLAUDE.md gotcha
   rewritten to whatever W3 concluded (single source of truth: this
   doc's findings table).
3. Project memory updated (the "init wandb on every rank" guidance is
   superseded or re-scoped).

## Risks / notes

- **Shared venv**: wandb version experiments (W2 cell 7) mutate the
  env the palooza team trains in — run them in a throwaway venv or
  restore immediately; never leave the venv on an experimental pin.
- **`uv sync` pruning**: any env work here must use the full extras
  incantation or it silently strips sfno/utils/datapipes extras (see
  `sfno-ddp-requirements.md` §3 — this already bit 12b via `muon`).
- **Intermittency**: job 20916803's ranks got *past* DDP init once
  (one rank died at the muon import *after* the wrap), so the hang is
  timing-dependent. Single PASSes prove little — hence the 3/3 rule in
  W3.2.
- The `int overflow` error string is shared by two unrelated causes
  (torch-2.11 regression vs dying-peer fallout). Always check
  `torch.__version__` first; don't re-pin on sight.

## Effort

| Item | Estimate |
|---|---|
| W1 mitigation + tests + docs | ~2 h |
| W2 isolation matrix | ~2 h scripting + ~1 GPU-hour |
| W3 durable fix + validation | ~0.5 day + 2 short GPU runs |
| W4 guards + docs | ~1 h |
| **Total** | **~1.5 developer days** |
