<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Container migration: measured facts and traps (2026-09-01)

Replacing the per-cluster `uv` environments with one multi-arch container built
on NVIDIA's NGC PyTorch image. Contract and usage live in
[`hpc/containers.md`](../../../hpc/containers.md); this note records what was
*measured*, so the next person does not re-derive it.

## Why

`hpc/scripts/sync-all-clusters.sh` maintained six divergent `uv sync` lines
across three CUDA extras, and they kept drifting off the `torch>=2.10,<2.11` pin
that `pyproject.toml` sets because 2.11 deadlocks SFNO DDP: the same
`--extra cu129` line resolved **2.12.1** on Derecho/Midway3/DSI and **2.11.0**
then **2.10.0** on Stampede3. Most per-cluster landmines in this directory are
environment artifacts, not hardware facts.

## The load-bearing design decision, and its evidence

**Keep NGC's torch; do not let the pyproject pin reinstall it.**

`nvcr.io/nvidia/pytorch:26.01-py3` ships `torch 2.10.0a0+a36e1d39eb`. Under PEP
440 **`2.10.0a0 < 2.10.0`**, so `torch>=2.10.0` does *not* accept it and a plain
`uv pip install .` silently replaces NVIDIA's build with the PyPI wheel — the
same shadowing failure `hpc/deltaai.md` documents for the venv.
`hpc/containers/make_requirements.sh` therefore exports the closure from
`uv.lock` and strips torch, torchvision, triton and torch's private `nvidia-*`
CUDA wheels, with a build-time guard that fails if the lock ever introduces an
`nvidia-*` wheel that is not torch-owned.

Verified inside the image on both arches:

```
torch        2.10.0a0+a36e1d39eb.nv26.01.42222806    <- NVIDIA build intact
torch.cuda   13.1        nccl (2,29,2)      cudnn 91701
apex         ok (FusedLayerNorm)                     <- the payoff
```

`apex` importing is the direct proof: its C extensions are compiled against
NGC's exact torch and would break had torch been swapped.
`physicsnemo/experimental/models/modulus_sfno/sfnonet.py` imports
`apex.normalization.FusedLayerNorm` optionally, so this is real, not incidental.

**Dropping the `cu12`/`cu129`/`cu13` extras entirely** is what removes the
per-cluster CUDA split rather than hiding it. Beyond torch they add only cuml,
pylibraft, cupy and DALI; cuml/pylibraft/cupy have **zero** imports in
`physicsnemo/` or `test/`, and every DALI import is `OptionalImport`-guarded.
Closure: 267 packages with `cu13` → **211** without.

## Measured results

| Host | Arch / GPU | Result |
|---|---|---|
| Delta login | x86_64 | full import set OK |
| Delta `gpuA40x4-interactive` | A40, sm_86, driver 595.71.05 | **GPU check passed**: bf16 matmul, NCCL 2.29.2, bf16 SDPA at the v2 geometry |
| DeltaAI login | aarch64 | full import set OK |
| DeltaAI `ghx4-interactive` | GH200 120GB, sm_90 | **GPU check passed**, incl. bf16 SDPA at the v2 geometry |
| Delta `gpuA40x4-interactive` | 1x A40 | `pytest -m "smoke and cuda"`: **21 passed, 20 skipped, 1 pre-existing failure** |
| Delta `gpuA40x4-interactive` | 2x A40 | **2-GPU DDP smoke passed** (`test_pangu_plasim_legacy_ddp_smoke`) — the gate for the torch<2.11 pin together with wandb-on-every-rank |

A CUDA 13.1 container works under plain `apptainer exec --nv` on driver 595 with
**no** NGC-entrypoint sourcing and no `LD_LIBRARY_PATH` fixup — worth knowing,
because `apptainer exec` bypasses `/opt/nvidia/nvidia_entrypoint.sh`, which is
where NGC normally decides whether to prepend `/usr/local/cuda/compat`.

Image sizes: **8.8 GB** x86_64, **9.1 GB** aarch64.

## DeltaAI workarounds the container eliminates

Three of the four documented in `hpc/deltaai.md` are gone:

1. **`CXX=CC` from the Cray PE** breaking TorchInductor (40 test failures → 0).
   `container_run.sh` unsets `CC`/`CXX`; verified by injecting `CXX=CC` on the
   host and observing it unset inside.
2. **Import-broken conda `wandb`.** The image's `wandb 0.27.0` imports cleanly.
3. **`AI_ROSSBY_NO_CUDNN_SDPA=1`.** The failure is a property of the host stack's
   cuDNN 9.20; the image ships 9.17.1, and bf16 SDPA at exactly
   `(B,H,S,D)=(6,16,4050,64)` was measured working on a real GH200 in-container.
   The forced export is removed from `train_rsi_amip.sbatch`; set it by hand if
   it ever regresses.
4. **Torch shadowing** (`uv pip uninstall torch torchvision triton`) is
   structurally gone — there is no venv to shadow.

## Traps discovered (each cost a cycle)

- **Delta cannot build images — and this breaks NCSA's *documented* method.**
  The Delta container guide says "Docker images can be converted to Apptainer sif
  format via the `apptainer pull` command", and `/sw/external/NGC/README` gives it
  verbatim. apptainer 1.5.1 bundles **mksquashfs 4.7.5 (2026/03/01)**, which fails
  with `FATAL ERROR: Bug in orderer` at the default 128 procs and SIGSEGV (exit
  139) at `-processors 8`. Download and rootfs extraction succeed; only squashing
  fails. It is **not** the `/tmp`-exhaustion failure the guide documents
  (`No space left on device`, cured by `rm -rf /tmp/build-temp*`): reproduced with
  no litter present, 1099 GB free, fresh cache/tmpdir. Control:
  `docker://alpine:3.20` converts fine on the same host, so it is size-dependent.
  Delta ships a working system mksquashfs 4.4 in `/usr/sbin`, but **apptainer
  ignores `$PATH` for its bundled helpers** (tested), so it cannot be redirected
  without root. **Worth an NCSA ticket**, since it breaks their own procedure.
  → DeltaAI (apptainer 1.4.2) converts fine, shares `/work/nvme` with Delta, and
  `apptainer pull --arch` never executes layers, so it builds **both** arches.
- **`apptainer pull` needs ~3x the final image on disk** and running it under
  `/work` fails with `EDQUOT`. The `bdiu` project quota is shared across
  `/work/nvme` and `/work/hdd`, and `/work/hdd` is over its own soft limit
  (20.79T of 19.53T) — see [lat-orientation-audit](lat-orientation-audit.md).
  `/work/nvme/bdiu` still takes a 2 GB write, so it is the cumulative unpack, not
  a hard block. Stage on node-local `/tmp` (~1.1 TB Delta, ~1.6 TB DeltaAI).
- **`/tmp` is node-local.** A script staged on a login node does not exist on the
  compute node. Bit both the driver probes and the first GPU check; put job
  payloads on `/work`.
- **`srun` from a DeltaAI login node fails** with `Error configuring
  interconnect`. Use `sbatch` (`--wait` for synchronous use).
- **torch-harmonics 0.9.x has no aarch64 distribution at all** — x86_64-only
  wheels, *no sdist* (0.8.1 is the last with one, below the `>=0.9.0` pin). So
  the `--no-binary` recipe formerly in
  [sfno-ddp-requirements](sfno-ddp-requirements.md) cannot work. Install from the
  `NVIDIA/torch-harmonics` git tag on arm64.
- **`cdo` has no arm64 apt candidate** in the NGC arm64 image. Made amd64-only;
  the PLASIM postprocessor chain is x86_64-only anyway.
- **PhysMetrics.Weather upstream now ships `requires-python`**, so
  [PhysMetrics.md](PhysMetrics.md)'s unconditional insertion produces a duplicate
  TOML key and the install fails. Also, installing it `--no-deps` (to protect
  torch) means its own deps must be named — `pyshtools`, `seaborn`, `gcsfs`.
- **nvfuser leaks a top-level `tools` package** into site-packages (just
  `gen_nvfuser_version.py` + `memory.py`), and because it is a *regular* package
  it shadows this repo's PEP 420 namespace `tools/` directories. PEP 420 gives a
  regular package precedence over namespace portions found anywhere on
  `sys.path`, **regardless of order**, so `PYTHONPATH` cannot fix it. This breaks
  `from tools.data...` / `from tools.harmonize_hindcasts ...` in 12 repo modules
  — production code included, not just tests. nvfuser never imports it, so
  `hpc/containers/strip_leaked_tools.py` deletes it at build time and asserts
  both that the shadow is gone and that nvfuser still imports. Expect this class
  of collision from any base image: `tools` is a very common top-level name.
- **cartopy publishes no linux-aarch64 wheel**; it builds from sdist against the
  libgeos/libproj headers.
- Of the 210 lock-pinned packages, an audit found **exactly one** with no
  aarch64-installable distribution (torch-harmonics). No other arm64 gaps.

## Pre-existing bugs this work surfaced

None were caused by the migration; they were found by actually running things.

1. **Comments inside backslash-continuations** in
   `smoke_amip_diffusion_2xA40.sbatch` and
   `smoke_amip_diffusion_convergence_2xA40.sbatch`. Bash ends the command at the
   comment, so `torchrun` received 15 of its 24 args and the remaining args ran as
   a bogus command. Since the dropped `num_ca_blocks=2` left the config's 8 >
   `num_blocks=4`, which `AmipDiT` asserts on, the 2xA40 smoke could not have
   passed. Introduced 2026-08-17 in `84725058`. **Fixed.** A sweep of all 60+ job
   scripts found only these two.
2. **The DDP smoke test was unselectable.** `test_smoke_ddp.py` carried
   `@pytest.mark.multigpu`, which `conftest.py` does not register (it registers
   `multigpu_dynamic` / `multigpu_static`), so conftest skipped it under *either*
   flag with "Unknown pytest.mark.multigpu". Present since `88e9ebad`
   (2026-06-18). Its docstring names the intended invocation, so the fix was
   unambiguous. **Fixed** — and the test passes.
3. **Two context notes were factually wrong** — see the traps above:
   `sfno-ddp-requirements.md`'s torch-harmonics `--no-binary` recipe and
   `PhysMetrics.md`'s unconditional `requires-python` insertion. **Both
   corrected.**
4. **`test_plasim_datapipe_with_workers_drives_pangu_plasim` fails at HEAD** —
   *left alone deliberately.* The test passes `forecast_lead_times=[1, 4]` and
   `physicsnemo/experimental/datapipes/climate/dataset.py:155` rejects multiple
   distinct leads ("Multi-lead training is single-step only"). That guard arrived
   2026-08-13 in `a505b4c9` without updating the test, which was last touched
   2026-06-23. Whether the guard is too strict or the test should move to a single
   lead is a datapipe-semantics decision, not a container one.

## Open

- **Driver check on Stampede3 `h100` and Midway3 `pedramh-gpu`** — CUDA 13.1
  needs r580+. Delta/DeltaAI are 595.71.05 and Polaris 580.65.06; these two are
  unmeasured (probe jobs queued a long time behind other work). If either is
  older, bind the host `cuda-compat` or drop to the NGC 25.06 (CUDA 12.9) line.
- **Polaris multi-node NCCL over CXI from inside a container** — the hardest
  remaining item. Gate on `hpc/scripts/allreduce_probe.py` reproducing the
  ~36.93 GB/s busbw baseline in `hpc/polaris.md`; 4.08 GB/s means CPU binding or
  the OFI plugin bind is wrong.
- Remaining job scripts, `hpc/install.md`, `CLAUDE.md` and the per-cluster docs
  are not yet converted — deliberately, until the pilots are proven on hardware.
- DSI stays on `uv sync`: no container runtime, no module system.
