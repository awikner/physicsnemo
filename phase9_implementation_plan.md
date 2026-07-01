# Phase 9 — Multi-Cluster Development Setup

## Goal

Establish a frictionless development workflow where code is written on a
personal Mac and executed on any of the six active HPC clusters with a single
command per day for authentication. Covers: SSH config, environment bootstrap,
environment propagation when packages change, cluster-specific documentation,
Claude skills, and a minimal smoke-test validation on each cluster.

**Clusters in scope (Polaris deferred):**

| Cluster | Site | Scheduler | GPU hardware | GPU account | CPU account |
|---|---|---|---|---|---|
| Delta | NCSA | SLURM | 4× NVIDIA A40 48 GB | `bdiu-delta-gpu` | `bdiu-delta-cpu` |
| DeltaAI | NCSA | SLURM | 4× NVIDIA H100 NVL 80 GB | `bdiu-dtai-gh` | — |
| Stampede3 | TACC | SLURM | NVIDIA H100 SXM 80 GB | `tg-atm170020` | `tg-atm170020` |
| Derecho | NCAR | PBS | 4× NVIDIA A100 40 GB | `UCHI0018` | `UCHI0014` |
| Midway3 | UChicago RCC | SLURM | NVIDIA V100/A100 | `pi-pedramh` | `pi-pedramh` |
| DSI | UChicago DSI | SLURM | A40/A100/L40S/H100/H200 | `general_group` | `general_group` |

**Already done:** Delta is fully set up (`hpc/delta.md`, three skills,
verified smoke tests). Phase 9 extends that baseline to the other five clusters.

---

## Sub-phase 9a — Mac SSH config + `morning-login`

**Done on the personal Mac.** No files committed to the repo (SSH config is
user-local), but `hpc/mac-setup.md` is committed as the canonical reference.

### Files

- `hpc/mac-setup.md` ← **committed to repo**, documents the SSH config and
  `morning-login` script so the setup is reproducible.
- `~/.ssh/config` ← **not in repo**, created on Mac from the template in
  `hpc/mac-setup.md`.
- `~/.ssh/controlmasters/` ← socket directory (one file per active connection).
- `~/bin/morning-login` ← **not in repo**, executable script.

### SSH config template (goes into `~/.ssh/config`)

```sshconfig
# ── Shared defaults ─────────────────────────────────────────────────────────
# ForwardAgent lets the Mac's ssh-agent serve the GitHub key on remote hosts,
# so `git pull / git clone` works without a per-cluster GitHub key.
# ControlMaster reuses a single connection for 8 h; MFA prompts fire once.
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3

# ── NCSA Delta ──────────────────────────────────────────────────────────────
Host delta
    HostName login.delta.ncsa.illinois.edu
    User awikner
    ForwardAgent yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 8h

# ── NCSA DeltaAI ────────────────────────────────────────────────────────────
Host deltaai
    HostName login.deltaai.ncsa.illinois.edu
    User awikner
    ForwardAgent yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 8h

# ── TACC Stampede3 ──────────────────────────────────────────────────────────
# TACC 2FA: enter password,TOTP-code at the single password prompt
# (e.g., "mypassword,123456"). ControlMaster avoids repeating this.
Host stampede3
    HostName stampede3.tacc.utexas.edu
    User awikner
    ForwardAgent yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 8h

# ── NCAR Derecho ────────────────────────────────────────────────────────────
Host derecho
    HostName derecho.hpc.ucar.edu
    User awikner
    ForwardAgent yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 8h

# ── UChicago Midway3 ────────────────────────────────────────────────────────
Host midway3
    HostName midway3.rcc.uchicago.edu
    User awikner
    ForwardAgent yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 8h

# ── UChicago DSI ────────────────────────────────────────────────────────────
# SSH key-based auth only (no DUO). ControlMaster still helps with latency.
Host dsi
    HostName login.ds.uchicago.edu
    User awikner
    ForwardAgent yes
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 8h
```

### `morning-login` script

```bash
#!/usr/bin/env bash
# morning-login — establish ControlMaster connections to all HPC clusters.
# Run once per day. You will be prompted for MFA/DUO per cluster that requires
# it; after that all SSH/SCP/rsync commands skip re-authentication for 8 h.
#
# Usage:  morning-login [cluster …]
#   No args → connects to all clusters.
#   With args → connects only to named clusters.

set -euo pipefail
CLUSTERS=(delta deltaai stampede3 derecho midway3 dsi)
targets=("${@:-${CLUSTERS[@]}}")

mkdir -p ~/.ssh/controlmasters

for c in "${targets[@]}"; do
    if ssh -O check "$c" &>/dev/null; then
        echo "[✓] $c — already connected"
    else
        printf "[…] %s — connecting" "$c"
        # -fNM: background after auth (-f), no remote command (-N), master (-M).
        # For DUO clusters the DUO prompt fires before -f backgrounds the process.
        if ssh -fNM "$c" 2>/dev/null; then
            echo " → OK"
        else
            echo " → FAILED (check hostname / credentials)"
        fi
    fi
done
echo "Done."
```

Install:

```bash
mkdir -p ~/bin
cp morning-login ~/bin/morning-login
chmod +x ~/bin/morning-login
# Make sure ~/bin is on PATH — add to ~/.zshrc if not:
# export PATH="$HOME/bin:$PATH"
```

### GitHub key forwarding

`ForwardAgent yes` in the SSH config makes the Mac's `ssh-agent` available on
each cluster. Confirm the key is loaded:

```bash
ssh-add -l                        # should show your key
ssh-add ~/.ssh/id_ed25519         # add if missing (or id_rsa / id_ecdsa)
```

On macOS the agent is persistent across reboots. No per-cluster GitHub key
needed. Test on a cluster:

```bash
ssh -T git@github.com             # run this on the remote host; should say "Hi awikner!"
```

---

## Sub-phase 9b — Per-cluster install

For each cluster below, the steps are:

1. `ssh <cluster>` (ControlMaster handles auth)
2. Clone repo (uses ForwardAgent for GitHub key)
3. Identify system stack; pick Option A or B from `hpc/install.md`
4. `uv sync`
5. Create test-data area
6. Write `hpc/<cluster>.md`
7. Write Claude skills

All five are independent and can be done in any order, but **DeltaAI first**
(most like Delta, lowest risk), **Midway3 second** (amip checkpoints already
there → unlocks translator live tests on a second cluster immediately).

### Path conventions (code vs. scratch)

Every cluster distinguishes between a **persistent work/project filesystem**
(for code, venv, and test fixtures — survives indefinitely) and a **scratch
filesystem** (for training data, converted Zarr archives, and job outputs —
large but subject to purge policies). These must not be confused.

| Cluster | Code + venv path | Test-data path | Training data / scratch path | Purge policy |
|---|---|---|---|---|
| Delta | `/work/nvme/bdiu/awikner/physicsnemo` | `/work/nvme/bdiu/awikner/physicsnemo_test_data` | `/work/hdd/bdiu/awikner/physicsnemo-zarr/` | No purge (quota-limited) |
| DeltaAI | `/work/nvme/bdiu/awikner/physicsnemo` ² | `/work/nvme/bdiu/awikner/physicsnemo_test_data` | `/scratch/bdiu/awikner/physicsnemo-zarr/` ² | TBD |
| Stampede3 | `$WORK/physicsnemo` | `$WORK/physicsnemo_test_data` | `$SCRATCH/physicsnemo-zarr/` | `$SCRATCH` purged after 90 days no access |
| Derecho | `/glade/work/awikner/physicsnemo` | `/glade/work/awikner/physicsnemo_test_data` | `/glade/derecho/scratch/awikner/physicsnemo-zarr/` | Scratch purged after 60 days no access |
| Midway3 | `/project/pedramh/awikner/physicsnemo` ³ | `/project/pedramh/awikner/physicsnemo_test_data` | `/scratch/midway3/awikner/physicsnemo-zarr/` | Scratch purge policy TBD |
| DSI | `/net/projects/general_group/awikner/physicsnemo` ⁴ | same root + `_test_data` | `/net/scratch/awikner/physicsnemo-zarr/` ⁴ | TBD |

**Rule:** the repo clone, `.venv`, and any test fixtures referenced by
`$AI_ROSSBY_TEST_DATA` always live on the persistent filesystem. Large Zarr
archives and training-run outputs live on scratch. The `sync-all-clusters.sh`
script only touches the code path.

² Delta and DeltaAI may share a `/work/nvme` filesystem — verify on first
login with `df -h /work/nvme` on each cluster. If shared, a single clone
serves both clusters but needs **separate venvs** (different GPU hardware /
CUDA builds). Name the DeltaAI venv `.venv-deltaai`. The DeltaAI scratch
path is also TBD until first login.

³ Verify exact path for `pi-pedramh` project storage on Midway3
(`ls /project/pedramh/`).

⁴ Verify correct project and scratch paths on DSI on first login
(`df -h ~` and check the cluster docs at https://cluster-policy.ds.uchicago.edu/).

### 9b-1 — NCSA DeltaAI

**Hardware:** H100 NVL GPU nodes (80 GB, NVLink-connected pairs).

**Expected system stack:** identical NCSA module environment to Delta. Verify:

```bash
module list | grep -i cuda       # expect cudatoolkit/25.x_12.x
module list | grep -i nccl
```

**Install:** same Option B as Delta (uv + `cu12` extra, Python 3.12). If the
Delta and DeltaAI `/work` filesystems are **separate**, clone fresh and run
`uv sync`. If they are **shared**, skip the clone step and just create
`.venv-deltaai`:

```bash
cd /work/nvme/bdiu/awikner/physicsnemo
unset VIRTUAL_ENV
uv venv --name .venv-deltaai      # or: uv sync with VIRTUAL_ENV unset
uv sync --extra cu12 --group dev --python 3.12
source .venv-deltaai/bin/activate
```

**TBD on first login:**
- Exact interactive GPU partition name (likely `gpuH100x4-interactive` but verify
  with `sinfo | grep -i h100`)
- Whether non-interactive partition is `gpuH100x4` (2-day walltime)
- Whether `aws-ofi-nccl` module exists for multi-GPU NCCL optimization

**Files:**
- `hpc/deltaai.md` — cluster facts, install notes, partition names, templates
- `.claude/skills/deltaai-smoke-test/SKILL.md`
- `.claude/skills/deltaai-shell/SKILL.md`

### 9b-2 — TACC Stampede3

**Hardware:** NVIDIA H100 SXM 80 GB, SLURM scheduler.

**Auth:** TACC uses 2FA. At the SSH password prompt, enter `password,TOTP`
(e.g., `mypassword,123456`). ControlMaster handles the rest of the day.

**Module system:** Lmod (`module` command). Typical TACC pattern:

```bash
module load cuda/12.6   # or latest available
module load python3/3.12
```

Verify:
```bash
python3 -c "import sys; print(sys.version)"
nvcc --version
```

**Install strategy:** Option A preferred if TACC's Python module has
PyTorch ≥ 2.10; otherwise Option B (uv manages Python). TACC historically
ships current PyTorch modules. Confirm:

```bash
python3 -c "import torch; print(torch.__version__)"
```

If torch ≥ 2.10: use `--system-site-packages` Option A. Otherwise Option B
with `--extra cu12`.

**Queue names (TBD — verify with `sinfo`):**
- H100 GPU batch: likely `gpu-h100` or `h100`
- H100 GPU interactive: TACC provides `idev` command (allocates a node
  interactively); equivalent to `srun --pty` elsewhere
- CPU: `skx` or `normal`

**TACC filesystem conventions:**
- `$HOME` (~25 GB, backed up) — dot-files only, never venvs or data
- `$WORK` / `$STOCKYARD` (~1 TB, persistent, no purge) — repo clone + venv
  + test fixtures live here
- `$SCRATCH` (~10 TB, purged after 90 days without access) — training data,
  converted Zarr archives, job output logs live here
- TACC's interactive launcher: `idev -p <partition> -N 1 -n 1 --time 01:00:00`
  (allocates a node and drops into a shell; preferred over `srun --pty` on TACC)

**Files:**
- `hpc/stampede3.md`
- `.claude/skills/stampede3-smoke-test/SKILL.md`
- `.claude/skills/stampede3-shell/SKILL.md`
- `.claude/skills/stampede3-cpu-job/SKILL.md`

### 9b-3 — NCAR Derecho

**Hardware:** 4× NVIDIA A100 40 GB per GPU node, PBS scheduler.

**Scheduler difference:** PBS uses `qsub`/`qstat`/`qdel` instead of SLURM's
`sbatch`/`squeue`/`scancel`. Job directives use `#PBS` instead of `#SBATCH`.
Interactive sessions use `qsub -I` instead of `srun --pty`. This is the most
significant change from the other clusters.

**Module system:** Lmod + NCAR environment. Typical:

```bash
module load ncarenv/24.12   # or latest — loads base NCAR environment
module load cuda/12.5       # or latest 12.x
module load python/3.12.5   # or latest
```

Verify `torch` availability and version per `hpc/install.md` step 0. NCAR
often provides PyTorch via a `py-torch` module or a conda install. If not
available as a module, use Option B.

**Queue structure (TBD — verify with `qstat -Q`):**
- GPU batch: `gpu` (walltime cap TBD, typically 12–24 h; project charge `UCHI0018`)
- GPU interactive/development: `develop` (1 h walltime, suitable for smoke tests)
- CPU batch: `main` or `cpu` (project `UCHI0014`)

**PBS job script template** (smoke test):

```bash
#!/bin/bash
#PBS -A UCHI0018
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -N pn-smoke
#PBS -j oe
#PBS -o <repo>/hpc/scripts/logs/smoke-${PBS_JOBID}.out

set -euo pipefail
module load ncarenv/24.12 cuda/12.5 python/3.12.5
cd /glade/work/awikner/physicsnemo
source .venv/bin/activate
pytest -m "smoke and cuda" -x -q test/
```

Submit: `qsub hpc/scripts/smoke.pbs`

Interactive shell on Derecho:

```bash
qsub -I -A UCHI0018 -q develop \
  -l walltime=01:00:00 -l select=1:ncpus=8:ngpus=1:mem=64gb
```

**NCAR filesystem conventions:**
- `/glade/home/awikner/` (~50 GB, backed up) — dot-files only
- `/glade/work/awikner/` (~2 TB, persistent) — repo clone, venv, test fixtures
- `/glade/derecho/scratch/awikner/` (~30 TB, purged after 60 days no access) —
  training data, Zarr archives, job outputs
- `/glade/campaign/` — long-term project storage (separate allocation; suitable
  for finalized multi-year Zarr archives once they are no longer being written)

**Existing data:** collaborator amip checkpoints are at
`/glade/derecho/scratch/ayz/AMIP_logs/` (referenced in
`phase8e_midway3_checkpoint_inventory.md`).

**Files:**
- `hpc/derecho.md`
- `hpc/scripts/smoke_derecho.pbs` (PBS job script for smoke tests)
- `.claude/skills/derecho-smoke-test/SKILL.md` (PBS-aware)
- `.claude/skills/derecho-shell/SKILL.md`
- `.claude/skills/derecho-cpu-job/SKILL.md`

### 9b-4 — UChicago Midway3

**Hardware:** NVIDIA V100 or A100 GPUs, SLURM scheduler.

**Auth:** DUO. ControlMaster handles the day.

**Module system:** Lmod. Typical:

```bash
module load cuda/12.4     # or latest available
module load python/3.12   # or Anaconda
```

**Queue structure (TBD — verify with `sinfo`):**
- GPU partition for `pi-pedramh` account: likely `beagle3` or `gpu`
- CPU partition: likely `caslake` or `bigmem`
- Walltime limits: typically 48 h non-interactive, shorter for interactive

**UChicago RCC filesystem conventions:**
- `/home/awikner/` (~30 GB, backed up) — dot-files only
- `/project/pedramh/awikner/` (project quota, persistent) — repo clone, venv,
  test fixtures; shared with the pedramh PI group
- `/scratch/midway3/awikner/` (large, purge policy TBD) — training data, Zarr
  archives, job outputs

**Existing data:**
- Collaborator amip checkpoints: `/project/pedramh/ayz/AMIP_logs/` — already
  used in Phase 8e live tests; unlocks x_DDC translator validation here
- Delta mirror of same checkpoints: `/work/nvme/bdiu/awikner/amip-checkpoints/`

**Files:**
- `hpc/midway3.md`
- `.claude/skills/midway3-smoke-test/SKILL.md`
- `.claude/skills/midway3-shell/SKILL.md`
- `.claude/skills/midway3-cpu-job/SKILL.md`

### 9b-5 — UChicago DSI

**Hardware:** A40 / A100 / L40S / H100 / H200 GPUs, SLURM scheduler.

**Auth:** SSH key only (no DUO); first login uses CNet password, subsequent
logins use SSH key. ControlMaster still reduces connection overhead.

**DSI filesystem conventions:**
- `/home/awikner/` — login node accessible; small quota
- `/net/projects/general_group/awikner/` (TBD exact path — verify) — project
  storage, persistent; repo clone, venv, test fixtures live here
- `/net/scratch/awikner/` (TBD — verify) — large scratch; training data, Zarr
  archives, job outputs
- Internal storage network runs at 100 Gbps between nodes but is not
  internet-routable; data transfers in/out require Globus (Phase 10)

**No module system.** Users manage software via Conda/MicroMamba. Since we use
`uv`, we go directly to Option B (uv manages Python + wheels). We do need the
system CUDA driver to be visible on compute nodes. Verify:

```bash
nvidia-smi   # confirms driver version → determines max CUDA version
ls /usr/local/cuda* 2>/dev/null || echo "No /usr/local/cuda"
```

CUDA 12.x drivers should be available on the H100/A100 nodes; use `--extra cu12`.

**QoS tiers:**
- `general`: preemptable, 12 h wall, max 24 concurrent jobs
- `protected`: non-preemptable, 2 h wall, max 1 concurrent job
- `interactive`: preempts general, 4 h wall, max 1 concurrent session

For smoke tests, use `--qos=interactive` (fast, non-preemptable during the 4-hour
window). For longer jobs, `--qos=protected` (2 h non-preemptable) or `--qos=general`.

**Requesting GPUs on DSI (SLURM resource syntax):**

```bash
#SBATCH --account=general_group
#SBATCH --qos=interactive
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1          # 1 GPU; omit for CPU jobs
```

To request a specific GPU type: `--gres=gpu:H100:1` or `--gres=gpu:A100:1`.

**Interactive shell** (equivalent of `delta-shell`):

```bash
srun --account=general_group --qos=interactive --time=01:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 --pty bash
```

**Files:**
- `hpc/dsi.md`
- `.claude/skills/dsi-smoke-test/SKILL.md`
- `.claude/skills/dsi-shell/SKILL.md`

---

## Sub-phase 9c — Environment propagation script

After a `git push` from the Mac, run `hpc/scripts/sync-all-clusters.sh` to
pull and re-sync every cluster. Requires ControlMaster connections to be active
(run `morning-login` first).

```bash
#!/usr/bin/env bash
# hpc/scripts/sync-all-clusters.sh
# Pull the ai-rossby branch and re-run `uv sync` on all configured clusters.
# Requires active ControlMaster connections (run `morning-login` first).
#
# Usage:  sync-all-clusters.sh [branch]   default: ai-rossby

set -euo pipefail
BRANCH="${1:-ai-rossby}"

declare -A REPO_DIRS=(
    # Persistent work/project filesystems only — never scratch.
    # Training data (Zarr archives) are on each cluster's scratch separately
    # and are not managed by this script.
    [delta]="/work/nvme/bdiu/awikner/physicsnemo"
    [deltaai]="/work/nvme/bdiu/awikner/physicsnemo"   # same /work NFS as Delta; verify
    [stampede3]="\$WORK/physicsnemo"                  # $WORK is persistent; $SCRATCH is not
    [derecho]="/glade/work/awikner/physicsnemo"       # /glade/work is persistent; scratch is not
    [midway3]="/project/pedramh/awikner/physicsnemo"
    [dsi]="/net/projects/general_group/awikner/physicsnemo"   # verify path on first login
)
declare -A VENV_NAMES=(
    [delta]=".venv"
    [deltaai]=".venv-deltaai"   # separate venv even if shared /work
    [stampede3]=".venv"
    [derecho]=".venv"
    [midway3]=".venv"
    [dsi]=".venv"
)

for cluster in "${!REPO_DIRS[@]}"; do
    repo="${REPO_DIRS[$cluster]}"
    venv="${VENV_NAMES[$cluster]}"
    echo "── $cluster ─────────────────"
    ssh "$cluster" bash -lc "
        set -euo pipefail
        cd $repo
        git fetch origin
        git checkout $BRANCH
        git pull --ff-only origin $BRANCH
        unset VIRTUAL_ENV
        uv sync --extra cu12 --group dev --python 3.12
        echo 'uv sync: OK'
    " && echo "  → OK" || echo "  → FAILED (see above)"
done
echo "Done."
```

**Notes:**
- `--ff-only` prevents accidental merges; if the pull fails, the user needs to
  SSH in manually and resolve.
- Stampede3 uses `\$WORK` (evaluated remotely, not on Mac).
- If Delta and DeltaAI share `/work`, the `git pull` runs twice on the same
  files but `uv sync` runs against different venvs — both are still correct.

---

## Sub-phase 9d — Cross-cluster smoke tests

After each cluster is set up in 9b, run the existing smoke suite to confirm
the install works end-to-end. The test target is the same as on Delta:

```bash
pytest -m "smoke and cuda" -x -q test/
```

Executed via the cluster's skill (once written) or manually. Document results
in the cluster's `hpc/<cluster>.md` under a **Smoke-test results** section:

```
| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| 2026-07-xx | 2.10.x+cu128 | A40 | PASS | — |
```

No new test code is needed — Phase 9 validates infrastructure, not new features.

---

## Sub-phase 9e — `hpc/mac-setup.md`

Committed to the repo so future sessions (or collaborators) can reproduce
the Mac-side setup. Covers:

- `mkdir ~/.ssh/controlmasters && chmod 700 ~/.ssh/controlmasters`
- The full `~/.ssh/config` template (from 9a above)
- `morning-login` script text + install instructions
- How to add the GitHub key to `ssh-agent` on macOS (persistent across reboots
  via Keychain: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`)
- How to check ControlMaster status (`ssh -O check <host>`) and close a
  socket early (`ssh -O exit <host>`)
- Security note on `ForwardAgent yes`: safe for trusted HPC systems; never
  enable on untrusted jump hosts

---

## File manifest

### New files committed to repo

| Path | Description |
|---|---|
| `phase9_implementation_plan.md` | This document |
| `hpc/mac-setup.md` | SSH config template + morning-login instructions |
| `hpc/deltaai.md` | DeltaAI cluster doc (written during 9b-1) |
| `hpc/stampede3.md` | Stampede3 cluster doc (9b-2) |
| `hpc/derecho.md` | Derecho cluster doc (9b-3) |
| `hpc/midway3.md` | Midway3 cluster doc (9b-4) |
| `hpc/dsi.md` | DSI cluster doc (9b-5) |
| `hpc/scripts/sync-all-clusters.sh` | Multi-cluster pull+sync script |
| `.claude/skills/deltaai-smoke-test/SKILL.md` | DeltaAI GPU smoke skill |
| `.claude/skills/deltaai-shell/SKILL.md` | DeltaAI interactive shell skill |
| `.claude/skills/stampede3-smoke-test/SKILL.md` | Stampede3 GPU smoke skill |
| `.claude/skills/stampede3-shell/SKILL.md` | Stampede3 interactive shell skill |
| `.claude/skills/stampede3-cpu-job/SKILL.md` | Stampede3 CPU job skill |
| `.claude/skills/derecho-smoke-test/SKILL.md` | Derecho PBS smoke skill |
| `.claude/skills/derecho-shell/SKILL.md` | Derecho PBS interactive shell skill |
| `.claude/skills/derecho-cpu-job/SKILL.md` | Derecho PBS CPU job skill |
| `.claude/skills/midway3-smoke-test/SKILL.md` | Midway3 GPU smoke skill |
| `.claude/skills/midway3-shell/SKILL.md` | Midway3 interactive shell skill |
| `.claude/skills/midway3-cpu-job/SKILL.md` | Midway3 CPU job skill |
| `.claude/skills/dsi-smoke-test/SKILL.md` | DSI GPU smoke skill |
| `.claude/skills/dsi-shell/SKILL.md` | DSI interactive shell skill |

### Not in repo (Mac-local)

| Path | Description |
|---|---|
| `~/.ssh/config` | SSH client config (cluster aliases, ControlMaster) |
| `~/.ssh/controlmasters/` | ControlMaster socket files (ephemeral) |
| `~/bin/morning-login` | Daily auth script |

---

## Items deferred to Phase 10

- **Globus data transfer** — endpoints already exist on all clusters;
  Phase 10 covers setting up a Globus Personal endpoint on the Mac, defining
  transfer paths between clusters, and adding a `globus-transfer` Claude skill.
- **ALCF Polaris** — allocation/setup deferred.
- **GitHub Actions CI** — automating `git pull + uv sync` cluster-side via
  a webhook/Actions workflow is a Phase 10 option once the manual sync script
  is proven.

---

## TBD items (verify on first login per cluster)

These need live inspection and are marked "TBD" in the cluster docs until done:

| Cluster | Question |
|---|---|
| DeltaAI | Exact interactive partition name (guess: `gpuH100x4-interactive`) |
| DeltaAI | Is `/work/nvme` the same NFS mount as Delta? (determines one-repo vs. two) |
| DeltaAI | Does `aws-ofi-nccl` module exist for multi-GPU runs? |
| Stampede3 | Exact H100 partition name (`sinfo | grep -i h100`) |
| Stampede3 | CUDA module version (`module avail cuda`) |
| Stampede3 | `idev` vs `srun --pty` preference for interactive work |
| Derecho | PBS queue structure (`qstat -Q`) |
| Derecho | CUDA module version, PyTorch module availability |
| Derecho | Walltime limits on `develop` (interactive) queue |
| Midway3 | GPU partition name for `pi-pedramh` account (`sinfo`) |
| Midway3 | Confirm correct project scratch path for clone |
| DSI | System CUDA driver version on GPU nodes (`nvidia-smi` on a compute node) |
| DSI | Confirm correct project storage path for clone |

---

## Execution order

Phase 9 picks up after the repo is cloned on the personal Mac:

```
git clone git@github.com:awikner/physicsnemo.git   # or pull the ai-rossby branch
cd physicsnemo
git checkout ai-rossby
```

Then:

1. **9e (mac-setup.md)** — write the doc (requires knowing the SSH config
   template; doesn't require any cluster access). Write first so it's committed
   before the Mac clone.
2. **9a** — apply SSH config + install `morning-login` on Mac, verify each
   cluster is reachable with `ssh <alias> hostname`.
3. **9b-1 DeltaAI** — first cluster; lowest risk; confirms the workflow.
4. **9b-4 Midway3** — second; amip checkpoints already present.
5. **9b-2 Stampede3**, **9b-3 Derecho**, **9b-5 DSI** — remaining three;
   parallelize if ControlMaster connections allow.
6. **9c** — write and test `sync-all-clusters.sh` once ≥ 2 clusters are up.
7. **9d** — smoke tests as each cluster comes online; update results table in
   each `hpc/<cluster>.md`.

Estimated total effort: ~4–6 h active work (mostly waiting for installs and
queueing for smoke tests), spread across one or two sessions.
