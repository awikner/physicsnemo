# Phase 10 — Release Preparation (hand-off to the research group)

Status: **planned (decisions locked 2026-07-09)** · Author: Claude (release-readiness review) · Created: 2026-07-09

## Goal

Get the ai-rossby fork ready to hand to other members of the group. A new
member should be able to: **clone → set up an environment on a cluster → train
one of the supported models → evaluate it**, guided by discoverable docs, with
a checkout that runs without editing hardcoded personal paths, and with correct
licensing/attribution. This phase is documentation + de-personalization +
licensing + a focused code cleanup + repo hygiene — *not* new modeling work.

The findings below come from a four-track review of the researcher-authored
surface (recipe, `tools/`, `physicsnemo/experimental/*`, `hpc/`, root docs).
Every actionable item carries `file:line` so it can be executed directly.

---

## 0. Decisions (locked 2026-07-09)

| # | Decision | **Locked answer** | Impact on this plan |
|---|---|---|---|
| **D1** | Audience | **Internal group (private repo)** | Shared cluster paths/accounts stay (group infra); collaborator usernames are low-risk (note-only, not a scrub blocker). Config/argparse **path defaults still get fixed** — that's usability, not privacy. No public-grade security review; NVIDIA CI/CONTRIBUTING get a banner, not a rewrite (10e). |
| **D2** | Provenance/license of `amip` & `PanguWeather v2.0` | **Mixed / collaborator-owned** | Not a full legal audit, but a real checkpoint: get the amip and PanguWeather contributors' **OK to share within the group**, and add a `NOTICE` documenting lineage + each part's owner. Since diffusion is "experimental" (D3), the amip-derived code is lower-stakes; the PanguWeather-derived Pangu/SFNO is the supported path and needs its contributors' sign-off. See 10c. |
| **D3** | Recipe scope | **Pangu/SFNO supported; diffusion experimental; healda is upstream (n/a)** | Deterministic Pangu/SFNO is the documented path. AMIP diffusion ships **marked experimental** — no need to close the `phase8f` punch list or fix its `NotImplementedError` branches; just don't let it break imports and label it clearly. **`healda` is NOT the researcher's code** — it's upstream NVIDIA (`physicsnemo/experimental/{datapipes,models}/healda`, authored by C. Adams & P. Harrington, PhysicsNeMo PRs #1555/#1612, present on `main`). It rides along with the fork like all upstream, nothing in the recipe imports it, and its env-var reads are NVIDIA's DA-recipe design — **no release action beyond simply not documenting it as part of ai-rossby.** |
| **D4** | Branch/repo | **Ship `ai-rossby` branch as-is** | No merge to `main`; members clone `ai-rossby`. Stays a physicsnemo fork. NVIDIA CI stays (documented as upstream/non-running); no wholesale CI replacement. 10f branch work drops to "tag + document the clone target." |

**Net effect on priority:** with licensing downgraded from a public "P0 veto" to
an internal "collaborator OK + NOTICE," the critical path is **10b
(de-personalize so a fresh checkout runs) → 10a (onboarding docs) → 10d
(correctness fixes)**, with 10c/10e/10f as lighter follow-ups.

---

## Sub-phase 10a — Documentation for onboarding

**Problem:** the top-level `README.md:1` is still the verbatim upstream
"# NVIDIA PhysicsNeMo" framework README — a new member lands on generic
framework copy with zero signal this fork exists or where to start. The one
genuinely excellent end-to-end doc (`examples/weather/ai_rossby/PANGUWEATHER_MIGRATION.md`)
is buried five directories deep and framed for someone *migrating from*
PanguWeather (assumes they already own PanguWeather HDF5 + checkpoints), not a
fresh member.

### Docs to CREATE (priority order)
1. **Top-level `README.md`** — replace/prepend a "what is this fork" section:
   purpose (ai-rossby weather/climate emulators built on PhysicsNeMo),
   relationship to upstream NVIDIA, supported recipes, and a prominent pointer
   to the recipe README + `hpc/install.md`. *Highest-leverage single change.*
   Keep the upstream README content below a divider or move it to `docs/`.
2. **`examples/weather/ai_rossby/README.md`** (does not exist) — the missing
   recipe entry point: what it trains (SFNO-E3SM, PanguPlasim/Legacy,
   SfnoPlasim, and — per D3 — AMIP diffusion), the Hydra `conf/` group layout,
   and the exact clone→env→train→evaluate path. Distill from
   `PANGUWEATHER_MIGRATION.md`, reframed for a fresh user (not a migrator).
3. **Data-acquisition guide** (biggest content hole) — where E3SM / PLASIM /
   ERA5 / AMIP raw data comes from and how it feeds
   `tools/data/*/*_h5_to_zarr.py`, `build_normalization_zarr.py`,
   `build_climatology_zarr.py`. Today every doc starts from "you already have
   the HDF5." Include the on-cluster locations of the already-converted Zarr
   stores (e.g. Delta E3SM at `/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/`,
   per `PANGUWEATHER_MIGRATION.md §2`).
4. **`tools/README.md`** (or `tools/data/README.md`) — index the data-conversion
   and checkpoint-translation scripts, only documented inline today.
5. **Planning-docs index / map** — one short doc explaining what the root
   `phase*`/`*_plan.md` files are (dev history + status) so a new member isn't
   misled by stale "in progress" plans.

### Docs to REORGANIZE
- **Relocate internal dev logs out of root** into `docs/dev/` (or `planning/`) —
  keep for history, don't let them compete with onboarding docs:
  `implementation_plan.md`, `pangu_plasim_reuse_plan.md`,
  `phase9_implementation_plan.md`, `phase8f_completion_plan.md`,
  `phase8e_midway3_checkpoint_inventory.md`, `project_outline.md`, and this
  file. **`phase8e_midway3_checkpoint_inventory.md` exposes collaborator
  directory layouts** — review before any release beyond the immediate group.
- **Reconcile the upstream NVIDIA meta-docs** that will mislead a group member:
  `FAQ.md`, `CONTRIBUTING.md` (describes an NVIDIA CLA process),
  `SECURITY.md`, `CHANGELOG.md` (55 KB NVIDIA framework changelog),
  `v2.0-MIGRATION-GUIDE.md` (NVIDIA v1→v2, not Pangu→ai-rossby). Either add a
  one-line "this is upstream NVIDIA — see <fork doc> for the group workflow"
  banner to each, or replace with fork-specific versions (CONTRIBUTING is the
  most urgent given group members *will* contribute).
- **Surface `PANGUWEATHER_MIGRATION.md`** from the new README(s) — keep the
  file, but link its clone→train→evaluate substance rather than leaving it
  discoverable only by its migration-specific title.

---

## Sub-phase 10b — De-personalize so a fresh checkout runs

**Problem (HIGH):** committed config **defaults** and argparse **defaults**
point at one user's cluster paths, so a fresh checkout fails on the first run.
No secrets were found (clean), so this is portability, not a leak — but it is
the single biggest usability blocker.

### Config default paths → placeholders (loaded on every run)
Replace the hardcoded `zarr_path`/`val_zarr_path`/`mean_path`/`std_path`
defaults with Hydra required-overrides (`???`) or env interpolation
(`${oc.env:AI_ROSSBY_DATA}/...`), and document the override in 10a:
- `conf/dataset/plasim_sim52_train_val.yaml:9-12`
- `conf/dataset/plasim_sim52_year12.yaml:9,11,12`
- `conf/dataset/amip_1981.yaml:11,17,18`
- `conf/dataset/era5_sfno_s2s_1981.yaml:9,11,12`
- `conf/dataset/e3sm.yaml:10-13`
- `conf/model/amip_combined.yaml:35,40` (hardcoded `.ckpt` checkpoint paths)

### argparse `default=` and literal paths in tools
- `tools/data/era5/pangu_h5_to_zarr.py:136`, `build_normalization_zarr.py:101`,
  `build_climatology_zarr.py:57` — `default=Path("/work/hdd/bdiu/bgong1/...")`
  (points at a *different* user's dir). Make the input dir a required arg.
- `tools/data/e3sm/pangu_h5_to_zarr.py:123`, `build_normalization_zarr.py:56` —
  hardcoded E3SM source-dir fallback → required arg.
- `tools/checkpoint_translation/amip_si.py:181` — `else Path("/work/nvme/bdiu/awikner/amip")`
  default repo → required/`--amip-repo` arg.
- `tools/checkpoint_translation/_validate_translations.py:16,25-42` — a personal
  validation harness: `sys.path.insert(0, "/work/.../awikner/...")` + a `CASES`
  table of absolute author paths. **Either delete from the release or move to
  `docs/dev/` and gate behind an env var** (it only runs on the author's
  account and is dead weight otherwise).

### `healda` datapipe — UPSTREAM NVIDIA, not a release concern (D3)
Correction to the initial review: `physicsnemo/experimental/{datapipes,models}/healda/`
is **upstream NVIDIA code** (authored by C. Adams & P. Harrington, PhysicsNeMo
PRs #1555/#1612, on `main`), not researcher-added. Its hard `os.environ[...]`
`KeyError` reads (`configs/static_data.py:49,62`, `loaders/era5.py:217`) are
NVIDIA's own DA-recipe design and are **out of scope** — no fix, no
`.env.example`, no active exclusion. It rides along with the fork like the rest
of upstream physicsnemo; nothing in the ai-rossby recipe imports it. Only
action: don't reference it from ai-rossby onboarding docs.

### Committed run artifacts + collaborator identifiers
- **Untrack `hpc/scripts/logs/*.out`** (15 committed job logs full of
  `/work/.../awikner/...`) and add `hpc/scripts/logs/` to `.gitignore`.
- Scrub collaborator usernames from committed code/configs (docs are note-only):
  `bgong1` (`tools/data/era5/*`), `rajatm2`
  (`conf/model/pangu_plasim_s2s.yaml:12`, `_validate_translations.py:33`),
  `ayz` (docs).
- Strip the `# Vendored from /work/nvme/bdiu/awikner/amip @ 497827e` provenance
  paths in ~23 files under `physicsnemo/experimental/models/amip_si/**` and
  `physicsnemo/experimental/diffusion/**` (keep the commit hash, drop the
  personal path — fold the real citation into the NOTICE from 10c).
- Reference-comment paths in configs: `conf/model/sfno_e3sm.yaml:9`,
  `conf/model/pangu_plasim_s2s.yaml:12` — genericize or move to NOTICE.

### Site-specific scripts (LOW — expected, but document)
`hpc/scripts/*.sbatch|*.pbs` and `sync-all-clusters.sh` legitimately carry
`--account=bdiu-*`/`UCHI0018` and `/work/.../awikner/...` roots. These are
*meant* to be edited per user — add a single "replace `awikner`/`bdiu`/allocation
with yours" callout in `hpc/install.md` rather than templating every script.

---

## Sub-phase 10c — Licensing, attribution, provenance

**State:** `LICENSE.txt` = Apache-2.0 (NVIDIA upstream) ✅. **No `NOTICE`
file.** 48/51 sampled new `.py` files carry an Apache SPDX header, but **all
of them claim *only* NVIDIA copyright** although `git log --diff-filter=A`
shows they were authored by the researcher (UChicago). The CI hook
`test/ci_tests/header_check.py:33` requires a `Copyright.*NVIDIA.*` line to
exist — so attribution can be fixed by **adding** a second copyright line, not
replacing (non-breaking to CI).

**Scope for an internal, mixed-ownership release (D1+D2):** not a full external
legal audit — a collaborator-OK checkpoint plus a `NOTICE` that records lineage.

### P0 — collaborator sign-off + lineage record (tie to D2)
1. **Get the `PanguWeather v2.0` contributors' OK** to share the derived
   `pangu_plasim`/`sfno_plasim` models + `loss.py` with the group (this is the
   *supported* path). Record the source + owner in `NOTICE`.
2. **Get the `amip` author's OK** to ship the derived diffusion recipe
   (`examples/weather/ai_rossby/*_diffusion.py`, `physicsnemo/experimental/diffusion/`,
   `tools/checkpoint_translation/amip_si.py`) as an **experimental** component.
   Lower-stakes given D3, but still record it in `NOTICE`.
3. **Confirm the EDM diffusion core** (`physicsnemo/experimental/diffusion/__init__.py`,
   `EDMSchedulerModule`) tracks the **Apache-2.0** physicsnemo EDM, not the
   non-commercial (CC-BY-NC-SA) NVIDIA EDM reference implementation.

> If any contributor declines, the affected code is pulled from the release
> surface (diffusion is already "experimental," so it can be dropped cheaply).
> A public release later would re-escalate 1–2 to a formal license audit.

### P1 — attribution correctness
4. **Add a `NOTICE`** at repo root acknowledging: the `pangu_plasim`/`sfno_plasim`
   + `loss.py` derivation from PanguWeather v2.0 (and its makani/FourCastNet
   lineage); the diffusion recipe/models derivation from `amip`; each with its
   license + copyright. Fold the ~23 "Vendored from" citations here.
5. **Fix copyright headers** on researcher-authored files — add
   `SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago` (keep
   the NVIDIA line so CI passes; for wholly-original files make UChicago the
   primary holder). ~48 files.

### P2 — hygiene
6. **Add SPDX headers to the 3 headerless files** (they fail the repo's own
   policy): `tools/checkpoint_translation/_validate_translations.py`,
   `tools/data/e3sm/__init__.py`, `tools/data/era5/__init__.py`.
7. **README fork/provenance note** (pairs with 10a #1).

---

## Sub-phase 10d — Code cleanup (release blockers)

Good news: **no `TODO`/`FIXME`/`breakpoint()`/`pdb` anywhere in scope, and no
stray `print()` in the recipe** (it uses logging). Real issues cluster in the
vendored Pangu utils and a few user-facing edges.

### Correctness / robustness
- `physicsnemo/experimental/models/pangu_plasim/_pangu_utils.py:856,860,892,896`
  — bare `except:` wrapping `assert` and re-labelling **any** exception (incl.
  `KeyboardInterrupt`) as a lat/lon-count error. Narrow to `except AssertionError`.
- `_pangu_utils.py:897` — copy-paste bug: `PolarPad3d.forward` raises a message
  naming **`PolarPadding2D`**. Fix the class name.
- `_pangu_utils.py:759,787,795` — stray `nn.Conv2d(...)` expression statements
  constructed and discarded (no assignment) inside a "true upsampling with 3D
  conv" class → the class looks half-finished. Verify it's unused; remove or
  complete.
- `examples/weather/ai_rossby/validate_diffusion.py:111` — `assert
  pred_ensemble.shape[0] % ensemble_size == 0` validates **user input**
  (stripped under `python -O`) → raise `ValueError` (as sibling lines 249/252/285
  already do).
- `tools/checkpoint_translation/amip_si.py:280,359` — `NotImplementedError` on
  user-facing paths (unsupported `model_name`; `decoder_type != "unet"`).
  Document as known limitations in `tools/README.md` (10a #4).

### Internal references leaking to users
- `tools/checkpoint_translation/amip_si.py:282,364` — user-facing exception
  messages tell the reader to "see `phase8e_midway3_checkpoint_inventory.md`"
  (an internal doc that won't ship). Reword.

### Cleanup (low risk)
- `_pangu_utils.py:369,372,516,519,576,691` — commented-out `#print(...)` debug
  lines; `:735-746` — a commented-out alternate `__init__` block. Remove.

### Structural — `sys.path.insert` sibling-import pattern (decide once)
Recipe entrypoints (`inference.py:65`, `eval_diffusion.py:52`,
`validate_cli.py:54`, `train_diffusion.py:60`, `climatology_cli.py:70`) and 7
`tools/data/**` scripts insert their own dir onto `sys.path` to import siblings.
Fragile (breaks under `-m`/install/symlink) and inconsistent. For release,
either (a) document "run from the recipe dir" explicitly, or (b) make the
recipe + tools proper packages. Lower priority than 10a/10b but affects the
"does a new user's first command work" test in 10f.

> Note: `pangu_plasim.py` (native) and `pangu_plasim_legacy.py` both ship — a
> deliberate compatibility path, not stale debris, but confirm both are
> "supported" under D3 and say so in the recipe README.

---

## Sub-phase 10e — Repo & CI hygiene

- **CI (D4 = ship the fork as-is):** `.github/workflows/` are all upstream
  NVIDIA (`blossom-ci.yml`, `github-nightly-*.yml`, `github-pr.yml`,
  `merge-queue-blossom-passthrough.yml`) and target NVIDIA infra. Don't rewrite
  them — add a one-line note (README/recipe README) that CI is upstream NVIDIA's
  and doesn't run for this fork. *Optional, low-effort:* a single lint + recipe
  smoke-test workflow so group PRs get basic signal.
- **`.gitignore`:** add `hpc/scripts/logs/` (see 10b). Confirm `checkpoints/`,
  `.venv*`, `test/_data`, `.env` already ignored ✅.
- **`healda` (upstream NVIDIA):** not researcher code — rides along with the
  fork; nothing in the recipe imports it. No action beyond keeping it out of
  ai-rossby docs.
- **`CONTRIBUTING.md` (D1 = internal):** it's the NVIDIA CLA process, which will
  confuse group contributors. Add a short "internal contribution notes for the
  group" section at the top (branch/PR conventions, who reviews) rather than a
  full rewrite.
- **Pre-commit:** `.pre-commit-config.yaml` runs the header check + others;
  ensure it passes after 10c header edits.

---

## Sub-phase 10f — Release logistics & verification

- **Branch (D4 = ship `ai-rossby` as-is):** no merge to `main`. Tag a release
  point on `ai-rossby` (e.g. `v0.1-group`) and document `git clone -b ai-rossby`
  as the clone command in the README so members land on the right branch.
- **Fresh-clone smoke:** the acceptance test for this phase — on a clean
  `ai-rossby` checkout with only documented env vars set, a new user runs the
  documented first command (env setup → one `train.py` invocation on a
  supported Pangu/SFNO recipe) and it starts training. This is what 10a+10b are
  in service of; run it on one cluster (Delta or DeltaAI) end-to-end before
  declaring release-ready.
- **Announce surface:** short "supported recipes (Pangu/SFNO; diffusion =
  experimental) + where to start + who to ask" note for the group (can live
  atop the recipe README).

---

## Sub-phase 10g — Onboarding presentation (PDF deck)

A **sparse, presenter-driven kickoff deck (~15–25 slides)** that is the first
thing a new group member sees — it gives the conceptual overview *and* walks
through the hands-on quickstart, then hands off to the markdown docs (10a) for
detail. It does **not** duplicate the docs: slides carry the arc + the exact
commands; the speaker carries the detail; links point into the 10a docs as the
single source of truth.

### Toolchain (locked: text source in repo → PDF)
- **Source:** Markdown slides under `docs/onboarding/` built with **Marp**
  (`marp-cli`) — Markdown-native, matches the sparse style, diffs cleanly, and
  stays in sync with the other markdown docs. (**Quarto** is the alternative if
  we later want slides with figures generated from data — e.g. embedding a
  validation scorecard or a training curve; Marp is the lighter default.)
- **Build:** a documented one-liner (`marp docs/onboarding/onboarding.md --pdf
  --allow-local-files`) plus a `make onboarding-pdf` / small script so it's
  reproducible; optionally a CI job that rebuilds the PDF on change.
- **Commit the built PDF** (`docs/onboarding/ai-rossby-onboarding.pdf`) so
  members can read it without installing the toolchain; keep the `.md` source
  next to it. Use Marp presenter notes (`<!-- notes -->`) for the speaker.

### Slide arc (~15–25 slides, sparse)
1. Title — what "ai-rossby" is, in one line (1)
2. Why: the goal — unified weather/climate emulators on PhysicsNeMo (1)
3. Lineage: a fork of NVIDIA PhysicsNeMo; models derived from PanguWeather v2.0
   (Pangu/SFNO) + amip (diffusion, experimental) (1)
4. Supported model families: PanguPlasim/Legacy, SfnoPlasim, SFNO-E3SM
   (+ diffusion marked experimental) — one sentence each (2–3)
5. Data: sources (E3SM / PLASIM / ERA5 / AMIP), the Zarr format, where the
   converted stores already live on the clusters, normalization/climatology (2)
6. Repo tour: `examples/weather/ai_rossby/` (train/inference/validate),
   `conf/` Hydra groups (model/dataset/training/loss/validation), `tools/`,
   `hpc/` (1–2)
7. Environment setup on a cluster (uv + system torch; pointer to
   `hpc/install.md` and the per-cluster docs) (1–2)
8. **Hands-on — train a supported recipe:** the exact single-GPU and
   `torchrun` multi-GPU commands for SFNO-E3SM / PanguPlasim (2)
9. Monitoring: wandb (offline default), checkpoints, `val_loss` / `rmse_step*` (1)
10. **Hands-on — evaluate:** `inference.py` → `validate_cli.py` scorecard (1–2)
11. Clusters + accounts + the SLURM/PBS skills available (1)
12. Where to get help: the docs map, who to ask, contributing conventions (1)

### Dependencies & scope
- **Depends on 10a** (README + recipe README + data-acquisition guide) — the
  deck distills them, so build it *after* the docs stabilize to avoid drift.
- Reflects the locked scope: Pangu/SFNO as the supported path, diffusion shown
  as experimental, no mention of upstream-only components (e.g. HealDA).
- Internal audience (D1): fine to include cluster names/accounts and the
  "who to ask" slide; no public-scrub needed.

---

## Suggested execution order (per the locked decisions)

1. **10c P0 (collaborator OK)** — kick off the amip/PanguWeather sign-offs
   *first* since they're async (waiting on people), even though internal makes
   them a checkpoint, not a veto. If a contributor declines, drop that piece
   (diffusion is already droppable) before investing in its docs.
2. **10b** — de-personalize config/argparse defaults (the real blocker: unblocks
   a new user running anything); untrack logs; genericize provenance comments.
3. **10a** — onboarding docs (README + recipe README + data-acquisition guide),
   which the runnable checkout from 10b now supports.
4. **10d** — code cleanup (correctness fixes + the `_pangu_utils.py` cluster).
5. **10c P1/P2** — `NOTICE` + copyright-header attribution + 3 missing headers.
6. **10e / 10f** — gitignore/CONTRIBUTING banner, tag `ai-rossby`, and the
   fresh-clone smoke gate on a cluster.
7. **10g** — build the onboarding PDF deck *last*, once 10a docs are stable
   (it distills them, so building earlier risks drift).

## Acceptance criteria
- [ ] A fresh `ai-rossby` clone + documented env setup + one documented command
      trains a supported Pangu/SFNO recipe on a cluster (10f smoke), with **no**
      edits to hardcoded personal paths.
- [ ] `README.md` identifies the fork and routes a new member to a working
      quickstart; recipe README + data-acquisition guide exist.
- [ ] `NOTICE` present; amip + PanguWeather licenses documented and compatible;
      researcher-authored files carry correct copyright; 0 headerless files.
- [ ] No committed run logs, collaborator usernames scrubbed from code/configs,
      no hard `KeyError` env reads on a documented setup.
- [ ] Correctness fixes in 10d landed; CI (fork-appropriate) is green.
- [ ] Onboarding deck (`docs/onboarding/`) builds to PDF reproducibly, covers
      the overview + clone→train→evaluate arc in ~15–25 sparse slides, and the
      built PDF is committed.
