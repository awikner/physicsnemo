# NCAR Derecho — install & smoke-test recipe

The realization of `hpc/install.md` for **NCAR Derecho** (**PBS** scheduler, Lmod, NCAR
environment). Sister document to `hpc/delta.md`. Derecho is the one cluster in scope that uses
**PBS Pro**, not SLURM — job control and directives differ throughout (see the callout below).

> **⚠️ SKELETON — authored from the Phase 9 plan, not yet verified on the cluster.**
> Every value tagged **`TBD`** must be confirmed on first login (checklist at the bottom) and
> this banner removed once smoke tests pass. Trust `hpc/delta.md` for anything left open.

---

## PBS vs. SLURM — the key difference

| Action | SLURM (Delta etc.) | **PBS (Derecho)** |
|---|---|---|
| Submit batch | `sbatch job.sh` | `qsub job.pbs` |
| Queue status | `squeue` | `qstat` |
| Cancel | `scancel <id>` | `qdel <id>` |
| Interactive | `srun --pty bash` | `qsub -I ...` |
| Directive prefix | `#SBATCH` | `#PBS` |
| Queues | `sinfo` | `qstat -Q` |
| Job id var | `$SLURM_JOB_ID` | `$PBS_JOBID` |
| CPU count var | `$SLURM_CPUS_PER_TASK` | `$NCPUS` |

Directives do **not** expand shell variables (e.g. `#PBS -o` cannot use `$PBS_JOBID`).

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** (Lmod modules, NCAR environment) |
| GPU hardware | 4× NVIDIA A100 (40 GB) per GPU node |
| GPU account / project | `UCHI0018` |
| CPU account / project | `UCHI0014` |
| Default smoke-test queue (GPU) | **TBD** — plan guess `develop` (`qstat -Q`; confirm it's GPU-capable) |
| Default data-conversion queue (CPU) | **TBD** — `main` or `cpu` |
| Walltime cap (interactive/dev queue) | **TBD** — plan guess ~1 h |
| Single-node constraint | ✅ all smoke tests + data-conversion jobs run on 1 node |
| Repo path | `/glade/work/awikner/physicsnemo` |
| Test-data path | `/glade/work/awikner/physicsnemo_test_data` (symlinked at `test/_data`) |

## Filesystem conventions (NCAR / GLADE)

| Filesystem | Size / policy | Use for |
|---|---|---|
| `/glade/home/awikner/` | ~50 GB, backed up | dot-files only |
| `/glade/work/awikner/` | ~2 TB, persistent | repo clone, venv, test fixtures |
| `/glade/derecho/scratch/awikner/` | ~30 TB, **purged after 60 days without access** | training data, Zarr, job outputs |
| `/glade/campaign/` | long-term project storage (separate allocation) | finalized multi-year Zarr archives |

**Existing data:** collaborator amip checkpoints at `/glade/derecho/scratch/ayz/AMIP_logs/`
(referenced in `phase8e_midway3_checkpoint_inventory.md`).

## System stack — install strategy

NCAR provides PyTorch via a module (`py-torch`) or conda on some environments. **Step 0:**

```bash
module load ncarenv/24.12          # or latest — base NCAR environment (exact ver TBD)
module avail cuda                  # note newest cuda/12.x
module avail python                # note python/3.12.x
module avail torch py-torch        # is a torch module offered, and at what version?
```

Derecho's system CUDA/Nsight is **12.9** (`cuda/12.9.0`; nsys 2025.1.3 / ncu 2025.2.0 — verified),
and its conda torch is 2.8 (< 2.10), so ai-rossby uses **Option B with `--extra cu129`** (new in
`pyproject.toml`) to build torch 2.10+cu129 — an *exact* match to the system Nsight (see the
Phase 9 plan § 9f and `hpc/install.md` § Step 7). Do **not** use `cu12`/cu128 here; cu129 matches
Derecho's profiler exactly.

**TBD:** `--extra cu129` has not been resolved by uv anywhere yet — **Derecho is its first use**,
so `uv sync --extra cu129` here also validates the lock. Commit the resulting `uv.lock` update.

## One-time setup

```bash
# 1. uv (one-time, per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # add to ~/.bashrc

# 2. Clone into /glade/work (persistent), NOT /glade/home or scratch.
cd /glade/work/awikner
git clone git@github.com:awikner/physicsnemo.git    # ForwardAgent serves the Mac's GitHub key
cd physicsnemo && git checkout ai-rossby

# 3. Load CUDA 12.9 (matches the system Nsight), then Option B with cu129:
module load cuda/12.9.0
unset VIRTUAL_ENV
uv sync --extra cu129 --group dev --python 3.12       # cu129 = exact Nsight-12.9 match

# 4. Test-data area on /glade/work
mkdir -p /glade/work/awikner/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=/glade/work/awikner/physicsnemo_test_data   # add to ~/.bashrc
```

Verify (login node, CPU-only):

```bash
python -c "import torch, physicsnemo; print('torch', torch.__version__, 'cuda', torch.version.cuda, '/ physicsnemo', physicsnemo.__version__)"
```

## Smoke-test contract

Identical to `hpc/delta.md` — `@pytest.mark.smoke` **and** `@pytest.mark.cuda`, single-node,
≤ 5 min wall, synthetic tiny tensors (datapipes read one real fixture from
`$AI_ROSSBY_TEST_DATA`). Target: `pytest -m "smoke and cuda" -x -q test/`.

## Job-script templates

### PBS smoke script (`hpc/scripts/smoke_derecho.pbs`)

```bash
#!/bin/bash
#PBS -A UCHI0018
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N pn-smoke
#PBS -j oe
#PBS -o hpc/scripts/logs/

set -euo pipefail
module load ncarenv/24.12 cuda/12.9.0 python/3.12.5   # exact versions TBD
cd /glade/work/awikner/physicsnemo
source .venv/bin/activate
pytest -m "smoke and cuda" -x -q "${TARGET:-test/}"
```

Submit from the repo root (so the relative `-o` path resolves): `qsub hpc/scripts/smoke_derecho.pbs`.
PBS writes `pn-smoke.o<jobid>` into `hpc/scripts/logs/`. The `derecho-smoke-test` skill wraps
this and blocks on completion.

### Interactive GPU shell (`qsub -I`)

```bash
qsub -I -A UCHI0018 -q develop \
  -l walltime=01:00:00 -l select=1:ncpus=8:ngpus=1:mem=64gb
# once on the node:
module load ncarenv/24.12 cuda/12.9.0 python/3.12.5
cd /glade/work/awikner/physicsnemo && source .venv/bin/activate
pytest -m "smoke and cuda" -x test/models/<feature>/
```

The `derecho-shell` skill wraps this.

## Profiling with Nsight

`nsys`/`ncu` load with `cuda/12.9.0`. Verified versions:

| Tool | Version | CUDA |
|---|---|---|
| Nsight Systems (`nsys`) | 2025.1.3 | 12.9 |
| Nsight Compute (`ncu`) | 2025.2.0 | 12.9 |
| `nvcc` | 12.9 | — |

ai-rossby's torch is **cu129 (12.9)** — an *exact* match, so `ncu` attaches cleanly (see
`hpc/install.md` § Step 7). Profile from inside a PBS GPU job (via `derecho-shell` /
`derecho-smoke-test`), writing the large output to scratch:

```bash
module load cuda/12.9.0
cd /glade/work/awikner/physicsnemo && source .venv/bin/activate
nsys profile   -o /glade/derecho/scratch/awikner/nsys_%p python -m <target> ...
ncu   --set full -o /glade/derecho/scratch/awikner/ncu_%p python -m <target> ...
```

## Data-conversion CPU jobs

CPU-only preprocessing runs under project `UCHI0014` on the CPU queue (`main`/`cpu`, **TBD**).
Scripts read `$NCPUS` (PBS's per-job CPU count; fall back to `os.cpu_count()`) to size their
`multiprocessing.Pool`. See the `derecho-cpu-job` skill.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| _pending first run_ | | A100 40 GB | | |

---

## First-login verification checklist (clears the TBDs)

- [ ] Queue structure — `qstat -Q` (confirm GPU dev queue name + that it has GPUs)
- [ ] Walltime cap on the interactive/dev queue
- [ ] CPU queue name for `UCHI0014`
- [ ] `ncarenv` / `python` exact module versions — `module avail` (`cuda/12.9.0` confirmed)
- [ ] Validate `uv sync --extra cu129` resolves (Derecho = cu129's first use); commit `uv.lock`
- [ ] Confirm `$NCPUS` is the right CPU-count var for conversion pool sizing
