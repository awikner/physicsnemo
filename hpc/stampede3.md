# TACC Stampede3 — install & smoke-test recipe

The realization of `hpc/install.md` for **TACC Stampede3** (SLURM, Lmod on RHEL). Sister document
to `hpc/delta.md`; follows the same ai-rossby smoke-test workflow.

> **✅ Verified 2026-07-02** — Option B cu128 install (**torch 2.11.0+cu128**, exact match to the
> 12.8 Nsight); **smoke passed on the `h100` partition** (see Smoke-test results). Project code is
> **`TG-ATM170020`** (uppercase — TACC rejects the lowercase form as "Unknown project"). A few
> items remain `TBD` (checklist at bottom).

---

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | SLURM (Lmod modules) |
| GPU hardware | NVIDIA H100 SXM, 80 GB |
| GPU account / allocation | `TG-ATM170020` |
| CPU account / allocation | `TG-ATM170020` |
| Default smoke-test partition (GPU) | **`h100`** (24 nodes, H100 — verified); submit via `sbatch` / `idev` |
| Default data-conversion partition (CPU) | **`spr`** (616 nodes, 112-core Sapphire Rapids) — verified 2026-08-19; `skx`/`icx` also open, `skx-dev` for ≤ 2 h tests |
| Interactive allocator | `idev` (TACC's node-grab; preferred over bare `srun --pty`) |
| Walltime caps | **2 days** on every production partition — enforced by the partition's **QoS**, not the partition (`sinfo` reports `infinite`); `skx-dev` caps at 2 h. Table below |
| Single-node constraint | ✅ all smoke tests + data-conversion jobs run on 1 node |
| Repo path | `$WORK/physicsnemo` |
| Test-data path | `$WORK/physicsnemo_test_data` (symlinked at `test/_data`) |
| Heterogeneous jobs (`hetjob`) | ✅ **supported** across partitions — verified 2026-08-19 (`h100` + `spr`); see below |
| GPU GRES | ❌ **not tracked** (`Gres=(null)`, `SelectType=select/linear`) — nodes allocate whole; `--gres=gpu:N` / `--gpus-per-node` are unusable |

### Partition & QoS limits (verified 2026-08-19)

`sinfo` reports `TIMELIMIT=infinite` for every production partition — the real caps live in each
partition's **QoS**, so read `sacctmgr`, not `sinfo`:

```bash
scontrol show partition h100 | grep -o 'QoS=[^ ]*'
sacctmgr show qos where name=qh100,qspr format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitJobsPU
```

| Partition | Nodes | Per node | QoS | MaxWall | Nodes/user | Running/user | Submitted/user |
|---|---|---|---|---|---|---|---|
| `h100` | 24 | 96 cores, 1006 GB, 4× H100 SXM 80 GB | `qh100` | 2 days | 4 | **2** | 4 |
| `spr` | 616 | 112 cores, 124 GB | `qspr` | 2 days | 96 | 24 | 36 |
| `skx` | 1160 | 48 cores, 186 GB | `qskx` | 2 days | 256 | 40 | 60 |
| `icx` | 224 | 80 cores, 250 GB | `qicx` | 2 days | 48 | 12 | 20 |
| `skx-dev` (default) | 72 | 48 cores, 186 GB | `qdevelopment` | **2 h** | 16 | 2 | 4 |
| `pvc` | 18 | 96 cores, 1006 GB, Intel PVC | `qpvc` | 2 days | 4 | 2 | 4 |
| `nvdimm` | 3 | 80 cores, 3.8 TB | `qnvdimm` | 2 days | 1 | 2 | 4 |

**`qh100` allows only 2 running jobs / 4 nodes per user** — the tightest budget on the machine.
Plan multi-GPU experiments around that, and remember a hetjob spends one running-job slot in
*each* component's QoS.

## Authentication

TACC uses 2FA at a single password prompt: type your **password immediately followed by a
comma and your TOTP code**, e.g. `mypassword,123456`. The `stampede3` SSH alias carries
`ControlMaster`/`ControlPersist 8h`, so this happens once per day (see `hpc/mac-setup.md`).

## Filesystem conventions (TACC)

| Filesystem | Size / policy | Use for |
|---|---|---|
| `$HOME` | ~25 GB, backed up | dot-files only — never venvs or data |
| `$WORK` / `$STOCKYARD` | ~1 TB, persistent, no purge | repo clone, venv, test fixtures |
| `$SCRATCH` | ~100 TB, **purged after 90 days without access** | training data, converted Zarr, job logs |

**Rule (all clusters):** repo clone + `.venv` + `$AI_ROSSBY_TEST_DATA` live on the persistent
filesystem (`$WORK`); large Zarr archives and run outputs live on `$SCRATCH`.

## System stack — install strategy

TACC historically ships current PyTorch modules. **Step 0: check what's available** before
choosing Option A vs B (see `hpc/install.md`):

```bash
module avail cuda                 # note the newest cuda/12.x
module avail python3              # note python3/3.12.x
module load python3/3.12          # or the exact version found
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"   # may error if no torch module
```

**Verified:** ai-rossby uses **Option B with `--extra cu12`** (CUDA 12.8) — `uv sync` pulled
**torch 2.11.0+cu128**, an exact match to the system Nsight (12.8). This is what
`sync-all-clusters.sh` uses for `[stampede3]`. (uv picked a `$WORK/miniconda3` Python 3.12 as the
base interpreter; that's fine.)

## One-time setup

```bash
# 1. uv (one-time, per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone into $WORK (persistent), NOT $HOME or $SCRATCH.
cd $WORK
git clone git@github.com:awikner/physicsnemo.git   # ForwardAgent serves the Mac's GitHub key
cd physicsnemo && git checkout ai-rossby

# 3. Install (Option B cu128). TWO Stampede3 login-node gotchas — both required:
unset VIRTUAL_ENV
export UV_CACHE_DIR=$SCRATCH/.uv-cache      # $HOME is tiny; cache the multi-GB tree on $SCRATCH
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1  # see vmem note ↓
uv sync --extra cu12 --group dev --python 3.12

# 4. Test-data area on $WORK
mkdir -p $WORK/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=$WORK/physicsnemo_test_data   # add to ~/.bashrc
```

> **⚠️ Login-node vmem cap.** The Stampede3 login node limits each process to **8 GB** virtual
> memory (`ulimit -v 8388608`). A default `uv sync` of the cu128 tree exceeds it and dies with
> *"memory allocation of N bytes failed"*. The `UV_CONCURRENT_*=1` exports above throttle uv so it
> stays under the cap. Compute nodes have no such limit but also lack outbound internet, so the
> download must run on the login node — hence the throttle. `UV_CACHE_DIR` on `$SCRATCH` likewise
> avoids the tiny `$HOME` quota.

Verify (login node, CPU-only):

```bash
python -c "import torch, physicsnemo; print('torch', torch.__version__, 'cuda', torch.version.cuda, '/ physicsnemo', physicsnemo.__version__)"
```

### ⚠️ Pre-flight for MULTI-GPU training: the venv above is torch 2.11

The 2026-07-02 verification above installed **torch 2.11.0+cu128**, and that is
fine for the single-GPU smoke it was verified with. It is **not** fine for
multi-GPU training: torch 2.11/2.12 regressed DDP for these models
(`docs/dev/context/sfno-ddp-requirements.md`), which is why `pyproject.toml`
pins `torch>=2.10.0,<2.11.0`. The `uv sync --extra cu12` line above predates
that pin being enforced here, so a fresh sync today resolves *below* 2.11
anyway — the risk is an **existing** venv built before it.

Check before submitting any >1-GPU job:

```bash
cd $WORK/physicsnemo && source .venv/bin/activate
python -c "import torch; print(torch.__version__)"      # want 2.10.x, NOT 2.11+
```

If it reports 2.11 or newer, rebuild:

```bash
cd $WORK/physicsnemo
unset VIRTUAL_ENV
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
rm -rf .venv
# The extras are NOT optional: without them uv silently prunes the SFNO/zarr
# deps and the recipe fails at import time, not at resolve time.
uv sync --extra cu12 --group dev --python 3.12     --extra sfno-extras --extra utils-extras --extra datapipes-extras
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

You do not have to remember this: `hpc/scripts/train_rsi_amip.sbatch` refuses to
start a >1-GPU run on torch >= 2.11 rather than let a degraded DDP job burn 48
hours of allocation. Single-GPU runs are unaffected and are allowed through.

## Smoke-test contract

Identical to `hpc/delta.md` — a smoke test is `@pytest.mark.smoke` **and** `@pytest.mark.cuda`,
single-node, ≤ 5 min wall, tiny synthetic tensors (datapipes read one real fixture from
`$AI_ROSSBY_TEST_DATA`). Target:

```bash
pytest -m "smoke and cuda" -x -q test/
```

## Job-script templates

### Interactive node via `idev` (TACC-preferred)

`idev` grabs a compute node and drops you into a shell on it:

```bash
idev -p <GPU_PARTITION> -N 1 -n 1 -t 01:00:00 -A TG-ATM170020    # partition TBD
# once on the node:
cd $WORK/physicsnemo && source .venv/bin/activate
pytest -m "smoke and cuda" -x -q test/models/<feature>/
```

The `stampede3-shell` skill wraps this.

### `sbatch --wait` (non-interactive smoke, blocks until done)

TACC discourages bare `srun` from login nodes; the portable non-interactive pattern is
`sbatch --wait` (blocks, writes to a file, then you tail it). Save as
`hpc/scripts/smoke_stampede3.sbatch`:

```bash
#!/bin/bash
#SBATCH -p <GPU_PARTITION>          # TBD
#SBATCH -A TG-ATM170020
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J pn-smoke
#SBATCH -o hpc/scripts/logs/smoke-%j.out

set -euo pipefail
module load <cuda-module>           # TBD; add python3 module if Option A
cd $WORK/physicsnemo
source .venv/bin/activate
pytest -m "smoke and cuda" -x -q "${TARGET:-test/}"
```

Submit and block: `sbatch --wait hpc/scripts/smoke_stampede3.sbatch` then tail the log.
The `stampede3-smoke-test` skill wraps this.

**TBD:** confirm whether direct `srun` works from the Stampede3 login node — if so, the
Delta-style streaming `srun ... bash -lc '...'` one-liner can replace `sbatch --wait`.

### Heterogeneous jobs — one allocation spanning `h100` + `spr`

**Supported and verified 2026-08-19.** Slurm is `23.11.11` with `SchedulerType=sched/backfill`
(required — hetjobs are *only* scheduled by the backfill loop) and `JobSubmitPlugins=(null)`, so
nothing site-local rejects them. Useful when a GPU rollout needs a fat CPU-side co-process
(e.g. Zarr writers / dataloader feeders on `spr` while the model trains on `h100`).

Both the in-script `hetjob` separator and the `:` CLI form are accepted:

```bash
#!/bin/bash
#SBATCH -J het-rollout
#SBATCH -o hpc/scripts/logs/het-%j.out
#SBATCH -p h100
#SBATCH -N 1 -n 1 -t 02:00:00 -A TG-ATM170020
#SBATCH hetjob
#SBATCH -p spr
#SBATCH -N 1 -n 1 -t 02:00:00 -A TG-ATM170020

srun --het-group=0 <gpu-side command>     # lands on h100
srun --het-group=1 <cpu-side command>     # lands on spr
```

```bash
# equivalent CLI form
sbatch -A TG-ATM170020 -p h100 -N 1 -n 1 -t 02:00:00 \
     : -p spr  -N 1 -n 1 -t 02:00:00  job.sh
```

Verify acceptance without burning SUs — `sbatch --test-only` runs TACC's submit filter once per
component and prints a concrete cross-partition placement:

```
$ sbatch --test-only hetjob_test.sbatch
--> Verifying access to desired queue (h100)...OK      # filter runs per component
--> Verifying access to desired queue (spr)...OK
sbatch: Job 3420806 to start at 2026-08-20T01:08:34 using 208 processors on c521-041,c562-008
```

A real submission registers as two job records under one leader, and `scancel <leader>` kills
the whole set:

```
JobId=3420812 HetJobId=3420812 HetJobOffset=0  Partition=h100  JobState=PENDING
JobId=3420813 HetJobId=3420812 HetJobOffset=1  Partition=spr   JobState=PENDING
   HetJobIdSet=3420812-3420813
```

**Caveats:**

- **`ibrun` is not hetjob-aware** (no `het` handling in the wrapper). Launch with
  `srun --het-group=N`. A *single* MPI communicator spanning both components
  (`srun --het-group=0,1`) is **untested** here — with `SwitchType=(null)` and
  `TopologyPlugin=topology/default`, don't assume it works across the SPR and H100 fabrics.
- **All components must start simultaneously**, so queue wait is gated by the scarcer side.
  `h100` runs near-fully-allocated; a 1-node/1-min test estimated a start ~9 h out.
- **Each component is a separate job record under its own QoS** — a hetjob spends from both
  budgets, and `qh100` only permits 2 running jobs per user (see the QoS table above).
- **`idev` cannot do this** — it's TACC's own allocator, not a hetjob front end. Use `sbatch`,
  or `salloc` with the `:` syntax for interactive.

## Profiling with Nsight

`nsys`/`ncu` ship with the CUDA / NVHPC modules. Load `cuda/12.8` (or `nvidia/25.3`, the NVHPC
SDK) to put them on PATH — this matches ai-rossby's **cu128 (12.8)** torch exactly, so `ncu`
kernel profiling attaches cleanly. Stampede3 also offers `cuda/12.4` and `cuda/13.1`; **12.8 is
the right pick** (13.1 would exceed the cu128 wheels and `cuda/12.4`'s Nsight is older).

```bash
module load cuda/12.8            # or nvidia/25.3
nsys --version ; ncu --version   # ⚠️ record exact versions on first login
```

**TBD:** exact `nsys`/`ncu` versions, and confirm `cuda/12.8` matches the H100 nodes' driver.

## Data-conversion CPU jobs

CPU-only preprocessing (HDF5→Zarr, climatology/bias, normalization stats) runs on **`spr`**
(112 cores/node, 2-day cap) under `TG-ATM170020`, via `idev` (interactive) or
`sbatch`. Scripts read `SLURM_CPUS_PER_TASK` to size their `multiprocessing.Pool`. See the
`stampede3-cpu-job` skill.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| 2026-07-02 | 2.11.0+cu128 | H100 SXM (`h100`) | **PASS** | `pangu_plasim`: 2 passed, 34 deselected, 99 s |

---

## First-login verification checklist (clears the TBDs)

- [ ] GPU partition name — `sinfo -s | grep -i h100`
- [x] CPU partition name — **`spr`** (also `skx`, `icx`, `skx-dev`)
- [x] Per-partition walltime caps — **2 days via QoS** (`sinfo` says `infinite`); `skx-dev` 2 h
- [ ] CUDA module version — `module avail cuda`
- [ ] Does TACC ship torch ≥ 2.10? → Option A vs B decision (record here)
- [ ] Does bare `srun` work from the login node, or is `idev`/`sbatch` required?
- [ ] `$WORK` and `$SCRATCH` resolved paths — `echo $WORK $SCRATCH`
