# UChicago DSI — install & smoke-test recipe

The realization of `hpc/install.md` for the **UChicago DSI cluster** (SLURM, **no module
system**). Sister document to `hpc/delta.md`.

> **✅ Verified 2026-07-02** — installed (Option B cu129, **torch 2.12.1+cu129**) at
> `/net/projects2/laude/awikner/physicsnemo`; the `general`-partition smoke **passed on an H100**
> (see Smoke-test results). ai-rossby uses **cu129 on the `general` partition** (runs + profiles
> natively).
>
> **⚠️ Two different stacks — use `general`, not `dev`:** the `general` partition
> (H100/A100/H200/L40S) is modern: driver **595.71.05** (max CUDA **13.2**), toolkits to 13.1,
> **nsys 2026.1.3** (the newest in the fleet). The old `dev` A40 nodes match the public docs
> (driver 550, max CUDA 12.4, nsys 2023.1.2) and are frequently drained.

---

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | SLURM (**no Lmod** — software via Conda/MicroMamba; we use uv) |
| GPU hardware | `general`: A100 / L40S / H100 / H200 (driver 595.71.05, max CUDA 13.2) · `dev`: A40 (old driver 550, often drained) |
| Smoke-test partition | **`general`** (modern stack) — **not** `dev` (old A40s) |
| ai-rossby CUDA | **cu129** (Option B; native on the 595 driver, profiled by nsys 2026.1.3) |
| GPU + CPU account (SLURM) | `general_group` (`--account=general_group`) — the *job* account, **not** a storage group |
| Storage group | **`laude`** — of my Unix groups (`ai-science`, `monsoon`, `laude`), the one with writable project storage |
| Smoke-test QoS | `interactive` (see QoS table) |
| Single-node constraint | ✅ all smoke tests + data-conversion jobs run on 1 node |
| Repo path | `/net/projects2/laude/awikner/physicsnemo` (verified writable, 275 TB) |
| Test-data path | `/net/projects2/laude/awikner/physicsnemo_test_data` |

## Authentication

SSH **key-based** (no Duo). The very first login uses your CNet password to register the key;
after that it's key-only. The `dsi` SSH alias still carries `ControlMaster` to cut connection
latency (see `hpc/mac-setup.md`).

## Filesystem conventions (DSI)

| Filesystem | Policy | Use for |
|---|---|---|
| `/home/awikner/` | small quota, login-node | dot-files only |
| `/net/projects2/laude/awikner/` | project storage (`laude` group), persistent, 275 TB | repo clone, venv, test fixtures |
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
`hpc/install.md`). The `general` partition's driver **595.71.05 supports up to CUDA 13.2**
(verified on an H100 node), so ai-rossby uses **`--extra cu129`** (CUDA 12.9) — native on this
driver and fleet-consistent with Derecho / Midway3 / DeltaAI. (cu130 / CUDA 13 would also work
given the 13.1 toolkit, but cu129 keeps DSI aligned.) No forward-compat is needed on `general`;
**avoid the old `dev` A40 nodes** (driver 550, max CUDA 12.4).

## One-time setup

```bash
# 1. uv (one-time, per-user)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # add to ~/.bashrc

# 2. Clone into project storage (persistent), NOT /home or /net/scratch.
cd /net/projects2/laude/awikner       # verify this path first
git clone git@github.com:awikner/physicsnemo.git    # ForwardAgent serves the Mac's GitHub key
cd physicsnemo && git checkout ai-rossby

# 3. Option B install (no modules to load). Put uv's cache on scratch — /home is small:
unset VIRTUAL_ENV
export UV_CACHE_DIR=/net/scratch/awikner/.uv-cache    # avoid small-/home quota (bit Midway3/Stampede3)
uv sync --extra cu129 --group dev --python 3.12

# 4. Test-data area on project storage
mkdir -p /net/projects2/laude/awikner/physicsnemo_test_data
export AI_ROSSBY_TEST_DATA=/net/projects2/laude/awikner/physicsnemo_test_data   # add to ~/.bashrc
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
  bash -lc 'cd /net/projects2/laude/awikner/physicsnemo && \
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
cd /net/projects2/laude/awikner/physicsnemo && source .venv/bin/activate
```

The `dsi-shell` skill wraps this.

## Profiling with Nsight

`nsys` is in `/usr/local/bin` (no module system). Verified on a `general`-partition H100 node:

| Tool | Version | Notes |
|---|---|---|
| Nsight Systems (`nsys`) | **2026.1.3** (`general` nodes) | newest in the fleet; profiles CUDA 12.x/13.x natively |
| Nsight Compute (`ncu`) | under `/usr/local/cuda-13.1/bin` (verify) | not on the default login PATH |

ai-rossby's torch is **cu129 (12.9)** and the `general` driver is 595.71.05 (max 13.2), so `nsys`
2026.1.3 profiles it **natively — no forward-compat needed** (unlike the old `dev` A40 nodes, whose
`nsys` 2023.1.2 is too old). Profile from inside a `general` GPU job, output to scratch:

```bash
cd /net/projects2/laude/awikner/physicsnemo && source .venv/bin/activate
nsys profile -o /net/scratch/awikner/nsys_%p python -m <target> ...
```

## Data conversion

DSI has no dedicated CPU-job skill (per the Phase 9 manifest). CPU-only preprocessing can run
under `--qos=general` (12 h) without `--gres`, using the same `srun ... bash -lc '...'` pattern;
scripts read `$SLURM_CPUS_PER_TASK` to size their pool. Add a `dsi-cpu-job` skill later if this
becomes routine.

## Smoke-test results

| Date | torch version | GPU type | Result | Notes |
|---|---|---|---|---|
| 2026-07-02 | 2.12.1+cu129 | H100 (`general`) | **PASS** | `pangu_plasim`: 2 passed, 34 deselected, 207 s |

---

## First-login verification checklist (clears the TBDs)

- [ ] Project storage path — `df -h ~`, confirm `/net/projects2/laude/awikner/`
- [ ] Scratch path + purge policy — confirm `/net/scratch/awikner/`
- [ ] CUDA driver version on a GPU compute node — `nvidia-smi` (supports cu128 wheels?)
- [ ] Confirm `--gres=gpu:<type>:N` type labels available — `sinfo -o "%P %G"`
- [ ] Confirm QoS tiers / caps match this doc — `sacctmgr show qos`
