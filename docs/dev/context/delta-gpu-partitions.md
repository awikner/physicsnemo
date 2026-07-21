<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Delta GPU partitions (A40 vs A100)

> Point-in-time reference (2026-07-07). Verify partitions/paths before relying on them.

NCSA Delta (account `bdiu-delta-gpu`) has **two** GPU partitions with different
hardware, and it matters for multi-GPU NCCL behavior:

- `gpuA40x4-interactive` — 4× **A40** (PCIe, no NVLink)
- `gpuA100x4-interactive` — 4× **A100** (NVLink)

The ai-rossby SFNO-PlaSim single-node speed benchmark
(`hpc/scripts/bench_sfno_ai_rossby.sbatch`) runs on **A100** (`gpuA100x4`, 4
GPUs). When reproducing / smoke-testing that benchmark, use the **A100**
partition — don't default to A40, or you introduce an interconnect variable.

PLASIM year-12 Zarr (train) + year-13 (val) live at
`/work/hdd/bdiu/awikner/physicsnemo-zarr/plasim/{12,13}.zarr`; sigma stats at
`/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data/data_12-132_{mean,std}_sigma.nc`.

Delta's ControlMaster SSH sockets drop frequently (MFA-gated Kerberos+Duo);
`ssh -O check delta` to test, and re-establish the session (interactive
Kerberos+Duo login) when it's down.

See also: [sfno-ddp-requirements](sfno-ddp-requirements.md).
