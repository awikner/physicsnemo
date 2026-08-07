#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DDP scaling benchmark for SFNO-E3SM at embed_dim=512, full fp32, 1 sample/GPU.
#
# Purpose: confirm the embed_dim=512 variant *runs* under DDP on Delta A100
# nodes, and measure per-batch training speed at 1 node (4 GPUs) and 2 nodes
# (8 GPUs).
#
# Unlike hpc/scripts/train_sfno_era5.sbatch (which hard-errors on NNODES>1),
# this launcher is multi-node capable: torchrun c10d rendezvous under `srun`,
# one agent per node. The SAME launch path is used for 1 and 2 nodes so the two
# numbers are apples-to-apples (no --standalone special case for 1 node).
#
# Submit:
#   sbatch --nodes=1 hpc/scripts/bench_sfno_e3sm_512_ddp.sbatch
#   sbatch --nodes=2 hpc/scripts/bench_sfno_e3sm_512_ddp.sbatch
#
# Env knobs:
#   EMBED_DIM  model embed dim               (default 512)
#   CKPT_LIST  activation-checkpointing settings to try, in order (default "2 3")
#              MEASURED (job 20914814): checkpointing=0 OOMs at embed_dim=512
#              fp32 even at 1 sample/GPU — 38.16 GiB allocated on a 39.49 GiB
#              A100. So 0 is NOT in the default list; starting there just burns
#              ~10 min of a 1-hour interactive allocation. The script falls
#              through to the next value on failure rather than aborting.
#   ITERS      training iterations per run   (default 60)

#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --job-name=bench-sfno-e3sm-512
#SBATCH --output=hpc/scripts/logs/bench-sfno-e3sm-512-%j.out

# NOTE: deliberately no `-e` — an OOM in one CKPT variant must not kill the job.
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs
# shellcheck disable=SC1091
source .venv/bin/activate

export EMBED_DIM="${EMBED_DIM:-512}"
export ITERS="${ITERS:-60}"
CKPT_LIST="${CKPT_LIST:-2 3}"

export NNODES="${SLURM_NNODES:-1}"
export NPROC=4                             # A100x4 node
WORLD=$(( NNODES * NPROC ))

# --- NCCL inter-node transport -----------------------------------------------
# Delta compute nodes have NO InfiniBand. The fast fabric is HPE Slingshot
# (`hsn0`, Cassini/CXI) — the mlx5_* devices visible on the LOGIN node are not
# present on gpua*. Stock NCCL has no CXI transport, so it silently falls back
# to TCP sockets over hsn0.
#
# Final measured numbers for this exact model (embed_dim=512, fp32, 1 sample/
# GPU, checkpointing=2), 60 steps, median of steady state:
#   1 node  (job 20914814)            0.5246 s/batch    7.62 samples/s
#   2 nodes stock NCCL (job 20914912) 3.5051 s/batch    2.28 samples/s   <- 6.7x WORSE
#   2 nodes + aws-ofi   (job 20915328) 0.7795 s/batch  10.26 samples/s   <- 67% scaling
# So the fallback made 2 nodes far slower than 1. Loading the plugin is a 4.5x
# swing and the difference between negative and useful scaling.
#
# Corroborated by a direct all-reduce probe of the same 4.404 GiB gradient
# (hpc/scripts/delta_allreduce_probe.sbatch, job 20915536): 478.8 ms ->
# 17.28 GB/s busbw at 2 nodes, vs 227.21 GB/s intra-node over NVLink.
#
# aws-ofi-nccl routes NCCL over libfabric's cxi provider instead. The module
# sets NCCL_NET="AWS Libfabric", FI_PROVIDER=cxi and LD_PRELOADs libfabric.
# 1.14.2 is the RH9.4 build (1.19.2 requires the RH9.6 image); it is built
# against CUDA 12.8, matching the venv's torch 2.10.0+cu128.
if [ "${USE_OFI:-1}" = "1" ]; then
  module load cudatoolkit/25.3_12.8 libfabric/1.22.0 aws-ofi-nccl/1.14.2 \
    && echo "NCCL transport: aws-ofi-nccl (NCCL_NET=${NCCL_NET:-unset}, FI_PROVIDER=${FI_PROVIDER:-unset})" \
    || echo "WARNING: aws-ofi-nccl load FAILED — NCCL will fall back to TCP sockets"
else
  echo "NCCL transport: stock NCCL (USE_OFI=0) — expect TCP-socket fallback over hsn0"
fi

# makani allocator fix — reclaims reserved-but-unallocated fragmentation.
# Must be exported before CUDA initializes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Surface NCCL transport selection (IB vs sockets) on the multi-node run.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OUT_DIR=/work/hdd/bdiu/awikner/sfno_bench
mkdir -p "${OUT_DIR}"

export ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/2041.zarr
export VAL_ZARR_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/2045.zarr
export NORM_ZARR=/work/hdd/bdiu/awikner/physicsnemo-zarr/e3sm/normalization_2015-2050.zarr
for d in "${ZARR_PATH}" "${NORM_ZARR}"; do
  [ -d "${d}" ] || { echo "ERROR: ${d} not present." >&2; exit 1; }
done

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export RDZV_ID="${SLURM_JOB_ID}"

echo "=================================================================="
echo " SFNO-E3SM DDP benchmark"
echo "   job            : ${SLURM_JOB_ID}"
echo "   nodes          : ${NNODES}  (${SLURM_JOB_NODELIST})"
echo "   gpus/node      : ${NPROC}   world_size=${WORLD}"
echo "   embed_dim      : ${EMBED_DIM}"
echo "   precision      : fp32 (training.amp=none)"
echo "   batch/GPU      : 1          global batch=${WORLD}"
echo "   iters per run  : ${ITERS}"
echo "   rendezvous     : c10d @ ${MASTER_ADDR}:${MASTER_PORT}"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for CKPT in ${CKPT_LIST}; do
  export CKPT
  export TAG="e3sm512_n${NNODES}_ckpt${CKPT}_${SLURM_JOB_ID}"
  export TSV="${OUT_DIR}/${TAG}.tsv"
  echo
  echo "########## RUN ${TAG} : checkpointing=${CKPT} ##########"

  srun --ntasks="${NNODES}" --ntasks-per-node=1 --gpu-bind=none \
       bash hpc/scripts/_bench_sfno_e3sm_node.sh
  RC=$?

  echo "---- ${TAG} exit=${RC}"
  if [ -f "${TSV}" ]; then
    echo "---- TSV: ${TSV}  ($(wc -l < "${TSV}") lines)"
  fi

  if [ "${RC}" -eq 0 ]; then
    echo "---- SUCCESS at checkpointing=${CKPT}; not trying further variants."
    break
  fi
  echo "---- FAILED at checkpointing=${CKPT}; trying next variant if any."
done

echo
echo "BENCH_DONE job=${SLURM_JOB_ID} nodes=${NNODES}"
