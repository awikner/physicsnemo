# TACC Stampede3 — install & smoke-test recipe

The realization of `hpc/install.md` for **TACC Stampede3** (SLURM, Lmod on RHEL). Sister document
to `hpc/delta.md`; follows the same ai-rossby smoke-test workflow.

> **⚠️ SKELETON — authored from the Phase 9 plan, not yet verified on the cluster.**
> Every value tagged **`TBD`** must be confirmed on first login (see the checklist at the
> bottom) and this banner removed once smoke tests pass. Until then, trust `hpc/delta.md`
> for anything this doc leaves open.

---

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | SLURM (Lmod modules) |
| GPU hardware | NVIDIA H100 SXM, 80 GB |
| GPU account / allocation | `tg-atm170020` |
| CPU account / allocation | `tg-atm170020` |
| Default smoke-test partition (GPU) | **TBD** — likely `gpu-h100` or `h100` (`sinfo -s \| grep -i h100`) |
| Default data-conversion partition (CPU) | **TBD** — likely `skx` or `normal` |
| Interactive allocator | `idev` (TACC's node-grab; preferred over bare `srun --pty`) |
| Walltime caps | **TBD** (`sinfo -o "%P %l"` for per-partition limits) |
| Single-node constraint | ✅ all smoke tests + data-conversion jobs run on 1 node |
| Repo path | `$WORK/physicsnemo` |
| Test-data path | `$WORK/physicsnemo_test_data` (symlinked at `test/_data`) |

## Authentication

TACC uses 2FA at a single password prompt: type your **password immediately followed by a
comma and your TOTP code**, e.g. `mypassword,123456`. The `stampede3` SSH alias carries
`ControlMaster`/`ControlPersist 8h`, so this happens once per day (see `hpc/mac-setup.md`).

## Filesystem conventions (TACC)

| Filesystem | Size / policy | Use for |
|---|---|---|
| `$HOME` | ~25 GB, backed up | dot-files only — never venvs or data |
| `$WORK` / `$STOCKYARD` | ~1 TB, persistent, no purge | repo clone, venv, test fixtures |
| `$SCRATCH` | ~10 TB, **purged after 90 days without access** | training data, converted Zarr, job logs |

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

- **If TACC provides torch ≥ 2.10** → **Option A**: `uv venv --system-site-packages`,
  reuse the system torch (tight NCCL/MPI integration, no multi-GB download). This is the
  preferred path on Stampede3 if available.
- **If torch < 2.10 or absent** → **Option B**: load only the `cuda/12.x` module, let uv pull
  torch from the `pytorch-cu128` index: `uv sync --extra cu12 --group dev --python 3.12`.

**TBD:** record which option was used and the exact module versions once verified.
`hpc/scripts/sync-all-clusters.sh` currently assumes Option B for Stampede3 — if Option A wins,
override the `[stampede3]` entry in that script's `SYNC_CMD` map (e.g. `uv sync --group dev`).

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

# 3. Load the system stack (exact modules TBD — see strategy above), then:
unset VIRTUAL_ENV
uv sync --extra cu12 --group dev --python 3.12      # Option B; adjust for Option A

# 4. Test-data area on $WORK
mkdir -p $WORK/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=$WORK/physicsnemo_test_data   # add to ~/.bashrc
```

Verify (login node, CPU-only):

```bash
python -c "import torch, physicsnemo; print('torch', torch.__version__, 'cuda', torch.version.cuda, '/ physicsnemo', physicsnemo.__version__)"
```

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
idev -p <GPU_PARTITION> -N 1 -n 1 -t 01:00:00 -A tg-atm170020    # partition TBD
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
#SBATCH -A tg-atm170020
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

CPU-only preprocessing (HDF5→Zarr, climatology/bias, normalization stats) runs on the CPU
partition (`skx`/`normal`, **TBD**) under `tg-atm170020`, via `idev` (interactive) or
`sbatch`. Scripts read `SLURM_CPUS_PER_TASK` to size their `multiprocessing.Pool`. See the
`stampede3-cpu-job` skill.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| _pending first run_ | | H100 SXM | | |

---

## First-login verification checklist (clears the TBDs)

- [ ] GPU partition name — `sinfo -s | grep -i h100`
- [ ] CPU partition name — `sinfo -s`
- [ ] Per-partition walltime caps — `sinfo -o "%P %l"`
- [ ] CUDA module version — `module avail cuda`
- [ ] Does TACC ship torch ≥ 2.10? → Option A vs B decision (record here)
- [ ] Does bare `srun` work from the login node, or is `idev`/`sbatch` required?
- [ ] `$WORK` and `$SCRATCH` resolved paths — `echo $WORK $SCRATCH`
