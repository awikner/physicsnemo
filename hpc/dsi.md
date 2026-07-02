# UChicago DSI — install & smoke-test recipe

The realization of `hpc/install.md` for the **UChicago DSI cluster** (SLURM, **no module
system**). Sister document to `hpc/delta.md`.

> **⚠️ SKELETON — authored from the Phase 9 plan, not yet verified on the cluster.**
> Values tagged **`TBD`** must be confirmed on first login (checklist at the bottom) and this
> banner removed once smoke tests pass. The QoS tiers and GPU request syntax below are from the
> DSI cluster policy and are expected to be correct; the storage paths and driver version are
> the main unknowns.

---

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | SLURM (**no Lmod** — software via Conda/MicroMamba; we use uv) |
| GPU hardware | mixed: A40 / A100 / L40S / H100 / H200 |
| GPU + CPU account | `general_group` |
| Smoke-test QoS | `interactive` (see QoS table) |
| Single-node constraint | ✅ all smoke tests + data-conversion jobs run on 1 node |
| Repo path | `/net/projects/general_group/awikner/physicsnemo` (**TBD** — verify) |
| Test-data path | `/net/projects/general_group/awikner/physicsnemo_test_data` (symlinked at `test/_data`) |

## Authentication

SSH **key-based** (no Duo). The very first login uses your CNet password to register the key;
after that it's key-only. The `dsi` SSH alias still carries `ControlMaster` to cut connection
latency (see `hpc/mac-setup.md`).

## Filesystem conventions (DSI)

| Filesystem | Policy | Use for |
|---|---|---|
| `/home/awikner/` | small quota, login-node | dot-files only |
| `/net/projects/general_group/awikner/` | project storage, persistent (**TBD** exact path) | repo clone, venv, test fixtures |
| `/net/scratch/awikner/` | large scratch (**TBD** policy) | training data, Zarr archives, job outputs |

The internal storage network runs at 100 Gbps between nodes but is **not internet-routable** —
data transfers in/out require **Globus** (deferred to Phase 10). Verify paths on first login
with `df -h ~` and the cluster docs at https://cluster-policy.ds.uchicago.edu/.

## QoS tiers

| QoS | Preemption | Wall cap | Concurrency | Use for |
|---|---|---|---|---|
| `interactive` | preempts `general` | 4 h | 1 session | **smoke tests**, interactive debugging |
| `protected` | non-preemptable | 2 h | 1 job | short non-preemptable runs |
| `general` | preemptable | 12 h | 24 jobs | longer / many-job batches |

Smoke tests use `--qos=interactive` (fast, non-preemptable during its 4 h window).

## System stack — install strategy

**No module system**, so go straight to **Option B** (uv manages Python + wheels; see
`hpc/install.md`). We only need the **system CUDA driver** visible on GPU compute nodes. Verify
on a compute node (not the login node):

```bash
nvidia-smi                                  # driver version → max CUDA version
ls /usr/local/cuda* 2>/dev/null || echo "No /usr/local/cuda"
```

CUDA 12.x drivers are expected on the H100/A100 nodes → `uv sync --extra cu12 --group dev
--python 3.12`. **TBD:** confirm the driver version supports cu128 wheels.

## One-time setup

```bash
# 1. uv (one-time, per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # add to ~/.bashrc

# 2. Clone into project storage (persistent), NOT /home or /net/scratch.
cd /net/projects/general_group/awikner       # verify this path first
git clone git@github.com:awikner/physicsnemo.git    # ForwardAgent serves the Mac's GitHub key
cd physicsnemo && git checkout ai-rossby

# 3. Option B install (no modules to load):
unset VIRTUAL_ENV
uv sync --extra cu12 --group dev --python 3.12

# 4. Test-data area on project storage
mkdir -p /net/projects/general_group/awikner/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=/net/projects/general_group/awikner/physicsnemo_test_data   # add to ~/.bashrc
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

### Streaming smoke via `srun` (blocks until pytest exits)

```bash
srun --account=general_group --qos=interactive --time=00:30:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 \
  --job-name=pn-smoke \
  bash -lc 'cd /net/projects/general_group/awikner/physicsnemo && \
            source .venv/bin/activate && \
            pytest -m "smoke and cuda" -x -q <TARGET>'
```

To pin a GPU type: `--gres=gpu:H100:1` or `--gres=gpu:A100:1`. Bump `--gres=gpu:2` for DDP.
The `dsi-smoke-test` skill wraps this.

### Interactive GPU shell

```bash
srun --account=general_group --qos=interactive --time=01:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 --pty bash
# once on the node:
cd /net/projects/general_group/awikner/physicsnemo && source .venv/bin/activate
```

The `dsi-shell` skill wraps this.

## Profiling with Nsight

DSI has no module system, so `nsys`/`ncu` come from the system CUDA install (or a Conda /
MicroMamba env). **TBD on first login** — check on a GPU node:

```bash
which nsys ncu || ls /usr/local/cuda*/bin/{nsys,ncu} 2>/dev/null
nsys --version ; ncu --version
```

Then pick the ai-rossby CUDA extra (`cu12`=12.8 or `cu129`=12.9) to be **≤** the system Nsight's
CUDA — exact match preferred (see `hpc/install.md` § Step 7). Record the versions + chosen extra
here once verified.

## Data conversion

DSI has no dedicated CPU-job skill (per the Phase 9 manifest). CPU-only preprocessing can run
under `--qos=general` (12 h) without `--gres`, using the same `srun ... bash -lc '...'` pattern;
scripts read `$SLURM_CPUS_PER_TASK` to size their pool. Add a `dsi-cpu-job` skill later if this
becomes routine.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| _pending first run_ | | H100/A100 | | |

---

## First-login verification checklist (clears the TBDs)

- [ ] Project storage path — `df -h ~`, confirm `/net/projects/general_group/awikner/`
- [ ] Scratch path + purge policy — confirm `/net/scratch/awikner/`
- [ ] CUDA driver version on a GPU compute node — `nvidia-smi` (supports cu128 wheels?)
- [ ] Confirm `--gres=gpu:<type>:N` type labels available — `sinfo -o "%P %G"`
- [ ] Confirm QoS tiers / caps match this doc — `sacctmgr show qos`
