#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rolling Stochastic Interpolants (RSI) training on the AMIP daily-avg archive.
# Cluster-agnostic Slurm launcher, same shape as train_sfno_era5.sbatch.
#
# The ablation ladder is a `RSI_RUNG` env var, and the ORDER MATTERS: the plain
# rolling interpolant (a2) is trained and evaluated before the residual
# parameterization (a3) or spectral noise (a4) are attempted, so that each
# result is attributable to one change. See examples/weather/ai_rossby/README.md
# section 6.6b.
#
#   RSI_RUNG=a1   loss=rsi_a1  ERDM-equivalence sanity — should track a
#                              loss=erdm_v2 run within noise. Run this FIRST;
#                              a gap is an implementation artifact, not a result.
#   RSI_RUNG=a2   loss=rsi     THE VANILLA MODEL (default). Coupled rolling SI,
#                              state parameterization, scalar white Gamma.
#   RSI_RUNG=a3   loss=rsi_a3  + residual parameterization.
#   RSI_RUNG=a4   loss=rsi_a4  + spectral Gamma. Needs RSI_SPECTRUM (fit first
#                              with hpc/scripts/fit_rsi_spectrum.sbatch).
#
# a5 is NOT here: eps(tau) is a post-hoc sampler dial, so that sweep runs
# against a trained checkpoint via validate_diffusion.py, with no retraining.
#
# Required env:
#   AI_ROSSBY_DATA   physicsnemo-zarr root (holds amip_dailyavg_coarse/ and
#                    amip_dailyavg_boundary/)
# Optional env:
#   RSI_RUNG           a1 | a2 | a3 | a4                (default: a2)
#   RSI_SPECTRUM       g(l) artifact, required for a4
#   RSI_WARMSTART      an ERDM v2 .mdlus to warm-start head 1 from. Expect ONLY
#                      `output_head_aux.*` in the skipped list — anything else
#                      means the contracts disagree, so read that report.
#   RSI_MODEL          model config (default: amip_rsi_v2). Use
#                      `amip_rsi_fancy` to train on the ERDM_fancy contract
#                      (154 ch, nocean 3, SST-anomaly channel + global_mean_sst
#                      trend scalar) — the only contract that is comparable to
#                      the real trained `erdm_fancy` checkpoint. It additionally
#                      needs the SST suite on the dataset side, which
#                      RSI_SST=1 turns on.
#   RSI_SST            1 => append the SST anomaly channel and route the
#                      global_mean_sst trend scalar. Needs RSI_SST_CLIM.
#   RSI_SST_CLIM       path to sst_climatology.npz (see
#                      tools/data/amip/make_sst_climatology.py).
#   RSI_DATASET        dataset config (default: amip_dailyavg_coarse). Use
#                      `amip_dailyavg_coarse_multiyear` when AI_ROSSBY_DATA holds
#                      a DIRECTORY of per-year stores rather than one year — the
#                      loader routes on the path not ending in `.zarr`.
#   RSI_EXTRA          extra Hydra overrides, space-separated. This is how a
#                      SHORT sanity run is expressed, e.g.
#                        RSI_EXTRA="++training.max_epochs=1
#                                   ++training.stages.0.num_epochs=1
#                                   ++training.stages.0.max_iterations=200"
#   AI_ROSSBY_CLUSTER  deltaai | delta | stampede3 | midway3 | dsi
#   AI_ROSSBY_VENV     venv dir (default: .venv, or .venv-deltaai on deltaai)
#   WANDB_PROJECT / WANDB_ENTITY / WANDB_MODE
#
# Submit:
#   Stampede3 :  AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr RSI_RUNG=a2 \
#                sbatch -p h100 -A TG-ATM170020 -t 48:00:00 \
#                hpc/scripts/train_rsi_amip.sbatch
#   DeltaAI   :  AI_ROSSBY_CLUSTER=deltaai AI_ROSSBY_DATA=... RSI_RUNG=a2 \
#                sbatch -p ghx4 -A bdiu-dtai-gh -t 48:00:00 \
#                hpc/scripts/train_rsi_amip.sbatch
#
# !! Stampede3 pre-flight: DDP is broken on torch >= 2.11 and pyproject pins
#    torch<2.11, but hpc/stampede3.md's verified venv is 2.11.0+cu128. Rebuild
#    it at 2.10.x/cu128 before any MULTI-GPU run — the guard below refuses
#    rather than letting a silently-degraded DDP run burn an allocation.

# NOTE: no --gpus-per-node / --mem directive here on purpose. TACC Stampede3's
# `h100` partition configures no gres (GPUs come with the whole node), so a
# --gpus-per-node request is rejected outright with "Invalid generic resource
# (gres) specification"; Delta/DeltaAI need one. Pass it at submit on the
# clusters that want it, and the GPU count below is discovered either way.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train-rsi-amip
#SBATCH --output=hpc/scripts/logs/train-rsi-amip-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs

# Runs in the ai-rossby container (hpc/containers/). container_run.sh detects the
# cluster, resolves the .sif and bind mounts, and scrubs the host toolchain env.
# The two DeltaAI workarounds that used to live here are handled differently now:
#  * CXX=CC / CC=cc from the Cray PE: container_run.sh unsets CC/CXX outright, so
#    TorchInductor uses the image's own toolchain on every cluster.
#  * cuDNN bf16 attention: no longer forced off. The documented failure
#    (hpc/deltaai.md: "no valid execution plans built" for bf16 SDPA at the v2
#    geometry) is a cuDNN-version property of the host stack's 9.20. The image
#    ships 9.17.1, and bf16 SDPA at exactly (B,H,S,D)=(6,16,4050,64) was measured
#    working on a real GH200 inside this container (2026-09-01). Setting
#    AI_ROSSBY_NO_CUDNN_SDPA=1 by hand still works if it ever regresses; carrying
#    it unconditionally would disable the cuDNN attention backend for nothing.
CONTAINER_RUN="$(pwd)/hpc/scripts/container_run.sh"

: "${AI_ROSSBY_DATA:?set AI_ROSSBY_DATA to the physicsnemo-zarr root}"
export AI_ROSSBY_DATA

for d in amip_dailyavg_coarse amip_dailyavg_boundary; do
  if [ ! -d "${AI_ROSSBY_DATA}/${d}" ]; then
    echo "ERROR: ${AI_ROSSBY_DATA}/${d} not present." >&2
    echo "       Check hpc/data_registry.yaml for which clusters hold it" >&2
    echo "       (note: DeltaAI holds a 1979-1981 subset under" >&2
    echo "        /work/nvme/bdiu/awikner/physicsnemo-zarr that the registry" >&2
    echo "        does not list — enough for a sanity run, not for production)." >&2
    exit 1
  fi
done

# Prefer what SLURM granted; fall back to what the node actually exposes, since
# a gres-less partition reports neither.
NPROC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [ -z "${NPROC}" ]; then
  NPROC="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
fi
NPROC="${NPROC:-1}"
[ "${NPROC}" -ge 1 ] 2>/dev/null || NPROC=1
NNODES="${SLURM_NNODES:-1}"
if [ "${NNODES}" -gt 1 ]; then
  echo "ERROR: multi-node (${NNODES}) not wired here — add rdzv args for >1 node." >&2
  exit 1
fi

# --- torch<2.11 DDP guard (the Stampede3 pre-flight blocker) -----------------
# docs/dev/context/sfno-ddp-requirements.md: torch 2.11/2.12 regressed DDP for
# these models, and pyproject pins torch>=2.10,<2.11. Refuse a multi-GPU run on
# a non-compliant build rather than produce a quietly-wrong 48-hour result.
TORCH_VER="$("$CONTAINER_RUN" python -c 'import torch; print(torch.__version__.split("+")[0])')"
if [ "${NPROC}" -gt 1 ]; then
  "$CONTAINER_RUN" python - "$TORCH_VER" <<'PY' || exit 1
import sys
major, minor = (int(p) for p in sys.argv[1].split(".")[:2])
if (major, minor) >= (2, 11):
    sys.stderr.write(
        f"ERROR: torch {sys.argv[1]} with >1 GPU. DDP is broken on 2.11/2.12 for\n"
        f"       these models (docs/dev/context/sfno-ddp-requirements.md) and\n"
        f"       pyproject pins torch>=2.10,<2.11. Rebuild/repull the container\n"
        f"       image at a torch 2.10.x NGC base, or\n"
        f"       submit with --gpus-per-node=1 to run single-GPU.\n"
    )
    raise SystemExit(1)
PY
fi
echo "torch ${TORCH_VER} on ${NPROC} GPU(s)"

# --- Rung -> config ----------------------------------------------------------
RUNG="${RSI_RUNG:-a2}"
EXTRA=()
case "${RUNG}" in
  a1) LOSS=rsi_a1 ;;
  a2) LOSS=rsi ;;
  a3) LOSS=rsi_a3 ;;
  a4) LOSS=rsi_a4
      : "${RSI_SPECTRUM:?rung a4 needs RSI_SPECTRUM (run hpc/scripts/fit_rsi_spectrum.sbatch first)}"
      EXTRA+=("loss.spectrum_path=${RSI_SPECTRUM}") ;;
  *)  echo "ERROR: RSI_RUNG must be a1|a2|a3|a4, got '${RUNG}'." >&2
      echo "       a5 is a SAMPLER sweep — see conf/validation/rsi_calibration.yaml." >&2
      exit 1 ;;
esac
if [ -n "${RSI_WARMSTART:-}" ]; then
  EXTRA+=("training.partial_checkpoint=${RSI_WARMSTART}")
fi
DATASET="${RSI_DATASET:-amip_dailyavg_coarse}"
MODEL="${RSI_MODEL:-amip_rsi_v2}"
if [ "${RSI_SST:-0}" = "1" ]; then
  : "${RSI_SST_CLIM:?RSI_SST=1 needs RSI_SST_CLIM (the sst_climatology.npz path)}"
  EXTRA+=("dataset.sst_anomaly_channel=append"
          "dataset.scalar_forcing=global_mean_sst"
          "dataset.sst_climatology_path=${RSI_SST_CLIM}")
fi
# Word-split on purpose: RSI_EXTRA is a list of Hydra overrides.
if [ -n "${RSI_EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA+=(${RSI_EXTRA})
fi

WANDB_PROJECT="${WANDB_PROJECT:-ai-rossby-rsi}"
WANDB_MODE="${WANDB_MODE:-online}"
if [ -n "${WANDB_ENTITY:-}" ]; then WANDB_ENTITY_ARG="wandb.entity=${WANDB_ENTITY}"; else WANDB_ENTITY_ARG="wandb.entity=null"; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi || true

# `python -m torch.distributed.run`, not the `torchrun` console script — kept for
# consistency with the rest of hpc/scripts, where it is required because DeltaAI's
# module-provided torch does not put torchrun on PATH.
"$CONTAINER_RUN" python -m torch.distributed.run \
    --nnodes=1 \
    --nproc-per-node="${NPROC}" \
    --standalone \
    examples/weather/ai_rossby/train_diffusion.py \
        --config-dir=examples/weather/ai_rossby/conf \
        --config-name=config \
        run_name="${RSI_RUN_NAME:-rsi_${RUNG}_run0}" \
        "model=${MODEL}" \
        loss="${LOSS}" \
        loss.window_size=6 \
        "dataset=${DATASET}" \
        training=amip_diffusion_bf16 \
        validation=diffusion_rollout \
        seed=0 \
        wandb.enabled=True \
        wandb.mode="${WANDB_MODE}" \
        wandb.project="${WANDB_PROJECT}" \
        "${WANDB_ENTITY_ARG}" \
        "${EXTRA[@]}"

echo
echo "RSI rung=${RUNG} (loss=${LOSS}) launched on ${NPROC} GPU(s); wandb project=${WANDB_PROJECT}"
