<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DEFERRED: retire Derecho scratch, re-home data to Delta

> **Decided 2026-07-21. Deferred — do NOT start the transfer without the user.**

Retire **Derecho scratch** as the data master. Its file-count (inode) quota
(~26.2M-file hard limit) can't hold the full dataset family (millions of tiny
zarr chunk files each — see
[phase11-data-consolidation](phase11-data-consolidation.md)). New intended
persistent home = **Delta `/work/hdd`**.

## Durability gap this creates

Dropping Derecho leaves 3 datasets with their full range only on **volatile
Stampede3** (Delta holds only partials):

| Dataset | On Delta (persistent) | Missing-persistent range (only on volatile Stampede3) |
|---|---|---|
| **e3sm** | 2041-2049 | **2015-2040** |
| **plasim_plev** | 7-78 | **79-104** |
| **amip** | 1981 | **1978-1980, 1982-2022** |

`era5`, `plasim`, `era5_sfno_s2s` already have a full persistent copy on Delta.

## Pending tasks (deferred)

1. **Transfer the missing ranges Stampede3 → Delta** (~3.3 TB). Delta `/work/hdd`
   had ~8.3 TB free (30 T, 73% used — SHARED bdiu group; +3.3 TB → ~84%). Use the
   proven tar path (`hpc/scripts/replicate_tar.sh`): pack on Stampede3 (`spr`) →
   Globus send → unpack on Delta (`cpu`, account `bdiu-delta-cpu`).
2. **Flip topology in docs** — `hpc/phase11_globus_runbook.md`,
   `docs/dev/phase11_implementation_plan.md`,
   `examples/weather/ai_rossby/DATA.md` still say "master = Derecho scratch."
3. **Decommission the Derecho scratch data** (~5.3 TB: e3sm/era5/plasim_plev/amip)
   — delete to free space + inodes rather than wait for the 60-day purge.
4. **Re-run `registry.py check`** to confirm durability post-migration.

## Already done (2026-07-21)

`hpc/data_registry.yaml` cluster roles updated (commit `22bf02b3`): delta →
`intended persistent master`, derecho → `master (RETIRING…)`, stampede3 →
`conversion + working copy`. The `copies` still reflect the current physical
layout (Derecho data is still present until the migration + decommission).
