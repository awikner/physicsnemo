#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PALS -> torchrun-style rank env shim for ALCF Polaris.
#
# physicsnemo's DistributedManager.initialize() recognises exactly three
# launchers (physicsnemo/distributed/manager.py):
#   * generic env  — RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
#   * SLURM        — SLURM_PROCID / SLURM_NPROCS / SLURM_LOCALID
#   * OpenMPI      — OMPI_COMM_WORLD_{RANK,SIZE,LOCAL_RANK}
#
# Polaris uses PALS mpiexec, which exports PMI_RANK / PMI_SIZE /
# PMI_LOCAL_RANK — none of which match. Without this shim
# `initialize_env()` raises on `int(os.environ.get("RANK"))` (None).
#
# Wrap the python command with this script so each rank translates its own
# PMI_* into the generic names before the interpreter starts:
#
#   mpiexec -n 16 --ppn 4 --hostfile "$PBS_NODEFILE" --cpu-bind depth -d 8 \
#       bash hpc/scripts/polaris_rank_env.sh python train.py ...
#
# MASTER_ADDR / MASTER_PORT must already be exported by the job script (they
# are the same for every rank, so they don't belong here).

set -uo pipefail

: "${PMI_RANK:?not running under PALS mpiexec — PMI_RANK unset}"
: "${PMI_SIZE:?not running under PALS mpiexec — PMI_SIZE unset}"
: "${MASTER_ADDR:?export MASTER_ADDR in the job script before mpiexec}"
: "${MASTER_PORT:?export MASTER_PORT in the job script before mpiexec}"

export RANK="${PMI_RANK}"
export WORLD_SIZE="${PMI_SIZE}"
# PALS sets PMI_LOCAL_RANK; fall back to rank % gpus-per-node if it's absent.
export LOCAL_RANK="${PMI_LOCAL_RANK:-$(( PMI_RANK % ${GPUS_PER_NODE:-4} ))}"

exec "$@"
