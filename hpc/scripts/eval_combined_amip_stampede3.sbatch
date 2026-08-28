#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Upstream-parity bias evaluation of the amip_v2 COMBINED cascade on
# Stampede3 — the protocol behind the reference headline table (upstream
# bias.py): forecaster rolls at 45x90, every emitted frame is downscaled by
# x_DDC-DiT to 180x360 and scored against the FULL-RES store; IC 1996-01-01,
# 1,827 daily frames (1996-01-02..2001-01-01), forecaster sampler
# num_steps=2 (checkpoint native), headline = lat-weighted mean + RMSE of
# the time-mean bias for z500/u250/t850/q850/t2m/prate/q2m/u10m/v10m.
#
# Runs on Stampede3 because the full-res amip_dailyavg store (~58 GB/yr)
# lives only there. Self-contained single job (TACC MFA: no job chains).
#
# Optional env:
#   EVAL_MODEL     amip_combined_fancy (default; reproduces the upstream
#                  ERDM table) | amip_combined_rsi_fancy (the parity RSI arm)
#   EVAL_HORIZON   emitted frames (default 1827; set 5 for a smoke test)
#   EVAL_IC_DATE   default 1996-01-01 (resolved to a store row at runtime,
#                  logged loudly — the archive starts 1978, never hand-index)
#   AI_ROSSBY_DATA        default $SCRATCH/physicsnemo-zarr
#   AI_ROSSBY_AMIP_CKPT   default /work2/09979/awikner/stampede3/amip-checkpoints
#
# Submit:
#   sbatch hpc/scripts/eval_combined_amip_stampede3.sbatch
#   EVAL_MODEL=amip_combined_rsi_fancy sbatch hpc/scripts/eval_combined_amip_stampede3.sbatch
#   EVAL_HORIZON=5 sbatch -t 0:30:00 hpc/scripts/eval_combined_amip_stampede3.sbatch   # smoke

# Stampede3's slurm rejects --gpus-per-node ("Invalid generic resource"):
# h100 nodes are allocated whole via -N/-n, no gres request.
#SBATCH -p h100
#SBATCH -A TG-ATM170020
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --job-name=eval-combined-amip
#SBATCH --output=hpc/scripts/logs/eval-combined-amip-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs

# The deployed trees on Stampede3 carry no venv of their own: source the main
# checkout's venv, then pin PYTHONPATH to THIS tree so `physicsnemo` resolves
# here and not from the venv's editable install of the other checkout (that
# silent mis-resolution has bitten before — GroupNotFoundError on configs the
# other tree predates).
# shellcheck disable=SC1091
source "${AI_ROSSBY_VENV:-/work2/09979/awikner/stampede3/physicsnemo/.venv}/bin/activate"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

export AI_ROSSBY_DATA="${AI_ROSSBY_DATA:-${SCRATCH}/physicsnemo-zarr}"
export AI_ROSSBY_AMIP_CKPT="${AI_ROSSBY_AMIP_CKPT:-/work2/09979/awikner/stampede3/amip-checkpoints}"

MODEL="${EVAL_MODEL:-amip_combined_fancy}"
HORIZON="${EVAL_HORIZON:-1827}"
IC_DATE="${EVAL_IC_DATE:-1996-01-01}"
# Optional forecaster-checkpoint override (e.g. a newer epoch of the same
# architecture); EVAL_TAG names the output dir so arms don't overwrite.
FC_CKPT="${EVAL_FORECASTER_CKPT:-}"
TAG="${EVAL_TAG:-${MODEL#amip_combined_}}"
OUTDIR="./outputs/eval_combined_${TAG}"
mkdir -p "${OUTDIR}"
EXTRA=()
if [ -n "${FC_CKPT}" ]; then
  EXTRA+=("++model.forecaster.checkpoint=${FC_CKPT}")
  echo "forecaster checkpoint override: ${FC_CKPT}"
fi
# Free-form extra hydra overrides (word-split), e.g. dataset knobs a model
# variant needs: EVAL_EXTRA_ARGS="dataset.sst_anomaly_channel=none ..."
if [ -n "${EVAL_EXTRA_ARGS:-}" ]; then
  read -r -a _extra_args <<< "${EVAL_EXTRA_ARGS}"
  EXTRA+=("${_extra_args[@]}")
  echo "extra overrides: ${EVAL_EXTRA_ARGS}"
fi

nvidia-smi || true
echo "== ${MODEL}: IC ${IC_DATE}, horizon ${HORIZON}, forecaster num_steps=2 =="

# loss=erdm_v2 satisfies Hydra composition only — the combined path builds
# both stages from the model config's own sampler blocks and never
# instantiates cfg.loss. sampler_num_steps default is 4: the =2 override is
# REQUIRED for parity (the v2 checkpoints are trained/validated at 2).
# torch.distributed.run --standalone avoids the bare-python MASTER_ADDR trap.
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    examples/weather/ai_rossby/eval_diffusion.py \
    --config-dir=examples/weather/ai_rossby/conf \
    --config-name=config \
    "model=${MODEL}" \
    loss=erdm_v2 \
    dataset=amip_dailyavg_multiyear \
    validation=eval_suite \
    "++eval_suite.horizon=${HORIZON}" \
    "++eval_suite.ic_date=${IC_DATE}" \
    ++eval_suite.sampler_num_steps=2 \
    ++eval_suite.headline.enabled=True \
    ++eval_suite.partial_save_every_frames=365 \
    "+eval_suite.output_path=${OUTDIR}/eval_suite.pt" \
    "run_name=eval_combined_${TAG}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "done -> ${OUTDIR}/eval_suite.pt"
