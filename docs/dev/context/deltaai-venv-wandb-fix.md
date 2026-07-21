<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeltaAI (GH200/aarch64) venv wandb fix + launch caveats

> Point-in-time note (2026-07-09). Verify against the current venv before relying on it.

On NCSA DeltaAI (aarch64, GH200), the Option-A venv `.venv-deltaai` inherits the
conda module (`python/miniforge3_pytorch/2.10.0`) via `--system-site-packages`,
and that conda env's **wandb is broken**: `ImportError: cannot import name
'Imports' from wandb.proto.wandb_telemetry_pb2` (protobuf-generated-code
mismatch). Because `physicsnemo/utils/logging/wandb.py` imports wandb eagerly at
module load (guarded only by `check_version_spec("wandb")`, which passes since
wandb *is* installed), this crashes ALL physicsnemo logging imports — so
`train.py` dies before `wandb.enabled=False` can take effect.

**Fix (2026-07-09):** `uv pip install -U wandb` inside `.venv-deltaai` (installed
wandb 0.28.0). The venv's own site-packages precede the conda site-packages on
`sys.path`, so the working wandb shadows the broken conda copy. Verified
`from physicsnemo.utils.logging import LaunchLogger` imports cleanly afterward.

Also note: on DeltaAI `torchrun` is NOT on the venv PATH (torch is inherited from
the conda module, whose bin isn't added by venv activation). For single-GPU runs
use plain `python` + set `RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
MASTER_PORT=<port>` so `DistributedManager` takes the env path (else it hits the
SLURM path and fails to resolve `MASTER_ADDR` without `srun`).

See also: [sfno-e3sm-compute-bound-gh200](sfno-e3sm-compute-bound-gh200.md),
[sfno-ddp-requirements](sfno-ddp-requirements.md).
