# ALCF Polaris — cluster facts & multi-node recipe

The realization of `hpc/install.md` for **ALCF Polaris** (PBS Pro, HPE Cray EX with
Slingshot/CXI). Polaris is the group's **largest A100 resource** and, once the
`--cpu-bind` trap below is avoided, the **best multi-node machine we have measured** for
SFNO-E3SM — 2.1× Delta's inter-node all-reduce bandwidth.

> ⚠️ **Read the "The `--cpu-bind` trap" section before running anything multi-node.**
> Omitting one `mpiexec` flag costs **9.1×** on inter-node bandwidth, silently.

---

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** (`qsub`/`qstat`), launcher is **PALS `mpiexec`** |
| Login host | `polaris.alcf.anl.gov` (ssh alias `polaris`, ControlMaster persistent) |
| Allocation | `lighthouse-uchicago` (16,874 node-hours as of 2026-08-07) |
| Node count | **560** compute nodes |
| GPU node geometry | 4× NVIDIA **A100-SXM4-40GB**, NVLink **NV4 all-pairs**, driver 570.124.06 |
| CPU | 1× AMD **EPYC 7543P** (32 cores / 64 threads), **4 NUMA domains**, 503 GB RAM |
| Fabric | **2× HPE Slingshot CXI NICs per node** (`cxi0`/`cxi1`, `hsn0`/`hsn1`) |
| Fabric topology | Dragonfly: **10 groups (`g0`–`g9`) × 56 nodes** (`tier1`/`tier0` in `pbsnodes`) |
| Mixed fabric flag | `ss11=True` on **336** nodes, `False` on **224** |
| Rank env | PALS exports `PMI_RANK` / `PMI_SIZE` / `PMI_LOCAL_RANK` (**not** `SLURM_*`) |

GPU/NUMA affinity is inverted — GPU *i* sits on NUMA node *3−i*:

| GPU | CPU affinity | NUMA |
|---|---|---|
| GPU0 | 24-31,56-63 | 3 |
| GPU1 | 16-23,48-55 | 2 |
| GPU2 | 8-15,40-47 | 1 |
| GPU3 | 0-7,32-39 | 0 |

## Queues

| Queue | Nodes | Walltime |
|---|---|---|
| `debug` | 1–2 | 5 min – **1 h** |
| `debug-scaling` | 1–10 | 5 min – **1 h** |
| `prod` | **10**–496 | 5 min – 24 h |
| `preemptable` | 1–10 | up to **72 h** |
| `demand` | 1–56 | 1 h |

`debug` takes **up to 2 nodes**, not 1 — it is the fastest turnaround for the 1-vs-2-node
comparisons we care about. `prod` has a **10-node minimum**, so anything between 3 and 9
nodes must go to `debug-scaling` (≤1 h) or `preemptable`.

## Storage

| Path | Notes |
|---|---|
| `$HOME` | small; fine for scripts and job logs |
| `/eagle/MDClimSim` | project space — ⚠️ **100% full** (42 TB free of 7.9 PB, 2026-08-07) |
| `/grand/...` | other project spaces exist; none allocated to us yet |

**The `/eagle` capacity is a live blocker** for staging E3SM/ERA5 Zarr archives *under
`/eagle/MDClimSim`*. Check `df -h` before planning any data move. The group's own
allocation `/eagle/lighthouse-uchicago` is where the checkout and the AMIP Zarr live:

| Path | Contents |
|---|---|
| `/eagle/lighthouse-uchicago/members/awikner/physicsnemo` | the fork checkout + `.venv` (torch 2.10.0+cu128) |
| `/eagle/lighthouse-uchicago/physicsnemo-zarr/amip_dailyavg_coarse` | 45×90 forecaster store, 1978–2022 (134 GB) |
| `/eagle/lighthouse-uchicago/physicsnemo-zarr/amip_dailyavg_boundary` | 1° forcings, 1978–2022 (20 GB) |
| `/eagle/lighthouse-uchicago/physicsnemo-zarr/norm_stats` | daily-avg mean/std NetCDFs |

**Pushing code here:** GitHub SSH auth from the Polaris login node fails
(`Permission denied (publickey)`). Ship commits with a git bundle instead:

```bash
# local
git bundle create /tmp/amipv2.bundle <last-polaris-sha>..<branch>
scp /tmp/amipv2.bundle polaris:/eagle/lighthouse-uchicago/members/awikner/
# polaris
cd /eagle/lighthouse-uchicago/members/awikner/physicsnemo
git fetch ../amipv2.bundle '<branch>:refs/remotes/bundle/<branch>' && git merge --ff-only FETCH_HEAD
```

## Software stack

```bash
module use /soft/modulefiles
module load conda/2025-09-25     # torch 2.8.0, NCCL 2.28.3
conda activate base
```

`conda/2025-09-25` (the plain variant) exports **no** `NCCL_*` or `FI_*` variables, which
makes it a clean base to layer the documented NCCL env onto. Its torch is **2.8.0** —
below physicsnemo's `torch>=2.10.0` pin but *within* the `< 2.11` DDP constraint from
[sfno-ddp-requirements](../docs/dev/context/sfno-ddp-requirements.md). A uv venv per
`hpc/install.md` Option B is the likely path; not yet built here.

Only **libfabric 2.2.0rc1** (`libfabric.so.1.28.0`) is installed. Every
`/soft/libraries/aws-ofi-nccl/*` plugin was compiled against libfabric **1.22.0**, which is
gone. In practice this does **not** matter (measured below), but it explains the confusing
module state.

---

## The `--cpu-bind` trap

**Always pass `--cpu-bind depth -d <N>` to `mpiexec`.** Without it, inter-node NCCL
bandwidth collapses by **9.1×**, with no error, warning, or log line.

Measured — 2 nodes, 8 ranks, all-reduce of a 4.404 GiB fp32 buffer (the real SFNO-E3SM
`embed_dim=512` gradient), job `7368993`:

| Case | Change from baseline | busbw |
|---|---|---|
| A | `--cpu-bind depth -d 8` (baseline) | **36.93 GB/s** |
| **B** | **no `--cpu-bind`** | **4.08 GB/s** ⬅ |
| C | `--cpu-bind depth -d 16` | 36.96 GB/s |
| D | `+ NCCL_SOCKET_IFNAME=hsn` | 36.90 GB/s |
| E | conda `-aws-nccl-1.6.0` module, no `--cpu-bind` | 3.97 GB/s |

Every case varies exactly one thing from A. The signature of the unbound case is a
bandwidth curve that **falls** with message size (12.2 → 4.6 GB/s) instead of rising to a
plateau — if you see that, you are unbound.

Why: aws-ofi-nccl drives its libfabric/CXI progress engine on the CPU. Under PALS' default
placement the four ranks are not spread across the four NUMA domains, so the progress
threads starve. ALCF's PyTorch page does mention `--cpu-bind depth -d 16`, but frames it as
a *dataloader* concern — it understates this badly; for large-message NCCL it is a 9×
effect.

Delta never showed this because its jobs run under `srun` with `--cpus-per-task`, which
applies CPU binding by default. *(Inference from the Delta configuration, not separately
measured there.)*

### Ruled out by measurement — don't re-litigate

Three plausible-sounding causes were each tested and are **not** responsible:

| Hypothesis | Verdict |
|---|---|
| aws-ofi-nccl plugin build (`v1.9.1-aws` vs `v1.9.1-aws-libfabric-1.22.0`) | ❌ both 36.9–37.0 GB/s (job `7368980`) |
| The 7 extra `FI_CXI_*` knobs (`RX_MATCH_MODE=software`, `RDZV_THRESHOLD=2000`, …) | ❌ 37.07 vs 38.79 GB/s, i.e. noise (job `7368976`) |
| `NCCL_SOCKET_IFNAME=hsn` | ❌ case D above, within noise |

## NCCL environment

From [ALCF's NCCL guide](https://docs.alcf.anl.gov/polaris/applications-and-libraries/libraries/nccl/)
(source: [argonne-lcf/user-guides](https://github.com/argonne-lcf/user-guides)); verified working:

```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/2.2.0rc1/lib64:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
```

Confirm it took — NCCL should log:

```
NET/OFI Selected Provider is cxi (found 2 nics)
Using network AWS Libfabric
```

`found 2 nics` is the Polaris signature (Delta reports `found 1 nics`).

ALCF documents that `NCCL_COLLNET_ENABLE` and friends cause **hangs/timeouts** with
Megatron-DeepSpeed; if a job hangs at init, try
`unset NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET`.

### Broken conda module variants

| Module | Plugin dir it sets | On disk? |
|---|---|---|
| `conda/2025-09-25-aws-nccl-1.6.0` | `v1.6.0-libfabric-1.22.0` | ✅ (but see below) |
| `conda/2025-09-25-aws-nccl-1.9.1` | `v1.9.1-libfabric-1.22.0` | ❌ **missing** |
| `conda/2025-09-26-aws-nccl-1.9.1` | `v1.9.1-libfabric-1.22.0` | ❌ **missing** |

The `-1.9.1` modules point at a directory that does not exist (the real one is
`v1.9.1-**aws**-libfabric-1.22.0`). Because they also set `NCCL_NET="AWS Libfabric"`, NCCL
**refuses to fall back** and dies with `Failed to initialize any NET plugin` →
`ncclInvalidUsage`. The `-1.6.0` module's plugin also failed to run under the documented
env in job `7368980`.

**Recommendation: don't use the `-aws-nccl-*` modules.** Load plain `conda/2025-09-25` and
export the block above. Worth reporting the broken modulefiles to ALCF.

---

## Job template — multi-node PyTorch

```bash
#!/bin/bash
#PBS -A lighthouse-uchicago
#PBS -q debug                       # ≤2 nodes; debug-scaling for ≤10
#PBS -l select=2:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle      # REQUIRED on Polaris
#PBS -l place=scatter
#PBS -j oe

cd "${PBS_O_WORKDIR}"
NODEFILE=${PBS_NODEFILE}
NNODES=$(wc -l < "${NODEFILE}")
PPN=4

module use /soft/modulefiles
module load conda/2025-09-25
conda activate base
# ... NCCL env block from above ...

export MASTER_ADDR=$(head -n1 "${NODEFILE}")
export MASTER_PORT=29500

mpiexec -n $(( NNODES * PPN )) --ppn ${PPN} --hostfile "${NODEFILE}" \
        --cpu-bind depth -d 8 \
        python your_script.py
```

`-l filesystems=` is mandatory or the job is rejected. `place=scatter` gives one node per
chunk. In Python, derive rank from `PMI_RANK`/`PMI_SIZE`/`PMI_LOCAL_RANK` — Polaris has no
`SLURM_*`. `hpc/scripts/allreduce_probe.py` reads both and is a working reference.

For dataloader-heavy jobs use `-d 16` and `num_workers` 4–8 (ALCF caps useful workers at
16). Set `CUDA_VISIBLE_DEVICES` *before* importing `mpi4py`/`horovod` if you use them.

## Measured performance

All-reduce of the 4.404 GiB SFNO-E3SM `embed_dim=512` gradient, correctly bound:

| | Delta (1 NIC) | Polaris (2 NICs) |
|---|---|---|
| 1 node (NVLink) | 31.2 ms / 227.21 GB/s | 34.4 ms / 206.16 GB/s |
| 2 nodes | 478.8 ms / 17.28 GB/s | **223.9 ms / 36.96 GB/s** |

**Polaris is 2.14× Delta inter-node** — consistent with 2 NICs vs 1.

Projected SFNO-E3SM (`embed_dim=512`, fp32, 1 sample/GPU, `checkpointing=2`). Delta rows
are measured; Polaris rows are projected from identical A100 compute plus the measured
all-reduce:

| | 1 node | 2 nodes | Efficiency |
|---|---|---|---|
| Delta (measured) | 7.62 samples/s | 10.26 samples/s | 67% |
| Polaris (projected) | ~7.6 samples/s | **~14–15 samples/s** | ~95% |

Delta hides ~224 ms of its 479 ms all-reduce behind backward; Polaris' entire 224 ms should
hide, leaving the step compute-bound. **Not yet validated by running the actual model on
Polaris.**

## Running SFNO-E3SM (4 nodes × 4 GPUs = 16 GPUs, global batch 16)

`dataset.batch_size` is **per rank**, so `=1` across 16 ranks gives a global batch of 16.

```bash
# test: 60 iterations, no validation/wandb, prints s/batch  (debug-scaling, 30 min)
AI_ROSSBY_DATA=<zarr-root> qsub -v MODE=test,AI_ROSSBY_DATA \
    hpc/scripts/polaris_train_sfno_e3sm.pbs

# full: preemptable is the ONLY queue that takes 4 nodes for >1 h
AI_ROSSBY_DATA=<zarr-root> qsub -q preemptable -l walltime=48:00:00 \
    -v MODE=full,AI_ROSSBY_DATA hpc/scripts/polaris_train_sfno_e3sm.pbs
```

Defaults to `model=sfno_e3sm_512` (`embed_dim=512`, `checkpointing=2`,
1,182,099,456 params); pass `-v MODEL=sfno_e3sm` for the 256 variant.

**`prod` cannot run this job** — it has a 10-node minimum. `preemptable` jobs can be
killed at any time, so `full` mode sets `checkpoint_save_interval=1`.

`polaris_rank_env.sh` maps PALS' `PMI_*` onto the `RANK`/`WORLD_SIZE`/`LOCAL_RANK` that
physicsnemo's `DistributedManager` expects — it recognises generic-env, SLURM and OpenMPI,
but **not** PALS, and without the shim `initialize_env()` raises on `int(None)`.

## Reproduction scripts

| Script | Purpose |
|---|---|
| `hpc/scripts/polaris_train_sfno_e3sm.pbs` | SFNO-E3SM training, test + full modes |
| `hpc/scripts/polaris_rank_env.sh` | PALS `PMI_*` → torchrun-style rank env shim |
| `hpc/scripts/allreduce_probe.py` | Launcher-agnostic (PMI + SLURM) all-reduce benchmark |
| `hpc/scripts/polaris_topo_probe.pbs` | Node/fabric inventory + 1/2/4-node bandwidth curve |
| `hpc/scripts/polaris_nccl_cause_ab.pbs` | The `--cpu-bind` isolation (job `7368993`) |
| `hpc/scripts/polaris_nccl_plugin_ab.pbs` | Plugin-build isolation (job `7368980`) |
| `hpc/scripts/polaris_nccl_env_ab.pbs` | Env A/B (job `7368976`) — ⚠️ control was invalid |
| `hpc/scripts/delta_allreduce_probe.sbatch` | Delta side of the head-to-head |

## Not yet done

- **No E3SM/ERA5 Zarr staged here** — blocked on `/eagle/MDClimSim` capacity. (The AMIP
  daily-avg coarse + boundary stores *are* staged, under `/eagle/lighthouse-uchicago`.)
- **SFNO-E3SM has never actually been trained on Polaris.** `polaris_train_sfno_e3sm.pbs`
  mirrors the validated Delta invocation and its NCCL env / `--cpu-bind` are measured, but
  the script itself has not been executed here — expect to shake out venv and data paths
  on the first `MODE=test` run. The throughput table above is projection.
- Whether `ss11=False` nodes (224 of 560) differ in bandwidth is **untested** — all
  measurements landed in group `g1`/`x3204`.

## Gotchas

- **Missing `--cpu-bind` ⇒ 9.1× slower inter-node.** Silent. See above.
- **`-l filesystems=` is mandatory**; jobs without it are rejected.
- **`prod` has a 10-node minimum** — 3–9-node runs need `debug-scaling` or `preemptable`.
- **Login nodes have no usable GPU** (`nvidia-smi` → `Failed to initialize NVML`).
- **`/soft/pbs/all_reduce_perf`** is a 2023 build needing CUDA 11 — unusable with current
  modules. Use `hpc/scripts/allreduce_probe.py`.
- **ALCF MFA is a single-use token** — the ssh ControlMaster (`ControlPersist yes`) is worth
  more here than on the Duo clusters. `morning-login polaris`.
- **Compute nodes have no direct internet**; the conda module sets
  `http_proxy=http://proxy.alcf.anl.gov:3128`.
