#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-node task body for hpc/scripts/bench_sfno_e3sm_512_ddp.sbatch.
#
# Runs as ONE task per node under srun. Starts a background nvidia-smi memory
# sampler, launches this node's torchrun agent, then reports the node's peak
# GPU memory. Kept as a separate file (rather than `srun bash -c "..."`) so the
# long torchrun arg list doesn't need a second level of shell quoting, and so
# the sampler shares the training step's resources instead of needing a
# separate, `--overlap`-requiring srun step.
#
# All inputs arrive via exported env from the sbatch script.

set -uo pipefail

NODEID="${SLURM_NODEID:-0}"
MEMLOG="/tmp/${TAG}_mem_${NODEID}.log"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 \
    > "${MEMLOG}" 2>/dev/null &
SAMPLER=$!

python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC}" \
    --rdzv-backend=c10d \
    --rdzv-id="${RDZV_ID}" \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    examples/weather/ai_rossby/train.py \
      --config-dir=examples/weather/ai_rossby/conf \
      --config-name=config \
      run_name="${TAG}" \
      model=sfno_e3sm \
      loss=raw_l2 \
      training=sfno_plasim \
      validation=off \
      dataset=e3sm \
      seed=0 \
      model.embed_dim="${EMBED_DIM}" \
      model.checkpointing="${CKPT}" \
      training.amp=none \
      training.max_epochs=1 \
      training.stages.0.num_epochs=1 \
      training.stages.0.max_iterations="${ITERS}" \
      training.grad_clip_norm=0.0 \
      training.ema.enabled=False \
      training.optimizer.fused=True \
      dataset.zarr_path="${ZARR_PATH}" \
      dataset.val_zarr_path="${VAL_ZARR_PATH}" \
      dataset.mean_path="${NORM_ZARR}" \
      dataset.std_path="${NORM_ZARR}" \
      dataset.batch_size=1 \
      dataset.num_workers=8 \
      wandb.enabled=False \
      wandb.mode=disabled \
      bench.per_batch_tsv="${TSV}"
RC=$?

kill "${SAMPLER}" 2>/dev/null
wait "${SAMPLER}" 2>/dev/null

PEAK=$(sort -n "${MEMLOG}" 2>/dev/null | tail -1)
echo "NODE_PEAK_MIB node=${NODEID} host=$(hostname) peak=${PEAK:-unknown} rc=${RC}"

exit "${RC}"
