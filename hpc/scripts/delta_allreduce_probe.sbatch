#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Delta half of the Delta-vs-Polaris fabric head-to-head. Runs the IDENTICAL
# benchmark (hpc/scripts/allreduce_probe.py) that job 7368955 ran on Polaris,
# so the two all-reduce bandwidth curves are directly comparable.
#
# Why this exists: the first Delta-vs-Polaris comparison was unfair. Polaris
# had a measured synthetic all-reduce, while Delta's was INFERRED from training
# step times (0.7795 s/batch at 2 nodes minus 0.5246 s at 1 node). Inferring
# through DDP's comm/compute overlap is not the same measurement.
#
# RESULT (job 20915536), 4.404 GiB = the SFNO-E3SM embed_dim=512 gradient:
#   1 node  (NVLink)        31.2 ms -> 227.21 GB/s busbw
#   2 nodes (1 Slingshot)  478.8 ms ->  17.28 GB/s busbw
# The curve rises and plateaus (9.87 -> 16.45 -> 17.07 -> 17.23 -> 17.28 GB/s),
# i.e. cleanly bandwidth-bound — the healthy shape.
#
# This also reconciles with training: DDP hid ~224 ms of the 479 ms all-reduce
# behind the backward pass, leaving the +255 ms/step actually observed. The
# earlier ~14 GB/s inference was in the right range but conservative.
#
# Submit:
#   sbatch hpc/scripts/delta_allreduce_probe.sbatch
#
# Env knobs:
#   USE_OFI  1 (default) load aws-ofi-nccl so NCCL uses libfabric/cxi;
#            0 leave stock NCCL (TCP-socket fallback) for an A/B.

#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200g
#SBATCH --job-name=delta-allreduce-probe
#SBATCH --output=hpc/scripts/logs/delta-allreduce-probe-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs
# shellcheck disable=SC1091
source .venv/bin/activate

# Same transport setup as the winning 2-node training run (job 20915328).
if [ "${USE_OFI:-1}" = "1" ]; then
  module load cudatoolkit/25.3_12.8 libfabric/1.22.0 aws-ofi-nccl/1.14.2 \
    && echo "NCCL transport: aws-ofi-nccl (NCCL_NET=${NCCL_NET:-unset}, FI_PROVIDER=${FI_PROVIDER:-unset})" \
    || echo "WARNING: aws-ofi-nccl load FAILED"
else
  echo "NCCL transport: stock NCCL (USE_OFI=0)"
fi

export PPN=4
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"

echo "=================================================================="
echo " Delta all-reduce probe (mirror of Polaris job 7368955)"
echo "   job    : ${SLURM_JOB_ID}"
echo "   nodes  : ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "--- NVLink topology ---"
nvidia-smi topo -m 2>/dev/null | head -8
echo "--- fabric NICs ---"
ls /sys/class/cxi 2>/dev/null || echo "(no /sys/class/cxi)"
ls /sys/class/infiniband 2>/dev/null || echo "(no infiniband)"

for N in 1 2; do
  [ "${N}" -gt "${SLURM_NNODES}" ] && continue
  export MASTER_PORT=$(( 29600 + N ))
  echo
  echo "---------- ${N} node(s), $(( N * PPN )) ranks ----------"
  # --gpu-bind=none so every task sees all 4 GPUs and can pick by SLURM_LOCALID.
  srun --nodes="${N}" --ntasks="$(( N * PPN ))" --ntasks-per-node="${PPN}" \
       --gpu-bind=none --overlap \
       python hpc/scripts/allreduce_probe.py
done

echo
echo "PROBE_DONE nodes=${SLURM_NNODES}"
