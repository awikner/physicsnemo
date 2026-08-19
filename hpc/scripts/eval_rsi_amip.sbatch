#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The two gates every RSI ablation rung must clear before the next one is
# attempted (examples/weather/ai_rossby/README.md section 6.6b):
#
#   RSI_EVAL=climatology  the long-horizon climate suite — ClimatologyValidator
#                         + BiasValidator + GlobalMeanTimeseries +
#                         EnsembleEnvelope (spread/skill). Gate: biases at
#                         parity with, or better than, the ERDM v2 baseline.
#   RSI_EVAL=rollout40    the 40-year AMIP-period rollout. Same eval_diffusion
#                         driver, just a much longer eval_suite.horizon plus
#                         enough monthly bins to actually cover it. Gate:
#                         stable (no drift/blow-up), global means conserved,
#                         QBO and seasonal cycle recognisable against the
#                         ERDM v2 40-year run.
#   RSI_EVAL=calibration  the eps(tau) spread sweep (rung a5). NOT a retrain:
#                         eps is a post-hoc sampler dial, so this runs against
#                         an already-trained checkpoint. Sweeps RSI_EPS_LIST.
#
# Required env:
#   AI_ROSSBY_DATA   physicsnemo-zarr root
#   RSI_CKPT         checkpoint dir of the trained rung
# Optional env:
#   RSI_EVAL         climatology | rollout40 | calibration   (default: climatology)
#   RSI_RUNG         label for run_name / output paths       (default: a2)
#   RSI_LOSS         the loss config the checkpoint was TRAINED with
#                    (default: rsi). This must match, or the interpolant
#                    profiles differ from the forward process the heads were
#                    regressed against — which is silently wrong, not an error.
#   RSI_EPS_LIST     calibration sweep values (default: "0.0 0.01 0.05 0.1 0.2")
#   RSI_ROLLOUT_YEARS  rollout40 horizon in years            (default: 40)
#
# Submit (Stampede3):
#   AI_ROSSBY_DATA=$SCRATCH/physicsnemo-zarr RSI_CKPT=./outputs/rsi_a2_run0/checkpoints \
#   RSI_EVAL=climatology sbatch -p h100 -A TG-ATM170020 -t 12:00:00 \
#       hpc/scripts/eval_rsi_amip.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=eval-rsi-amip
#SBATCH --output=hpc/scripts/logs/eval-rsi-amip-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p hpc/scripts/logs

CLUSTER="${AI_ROSSBY_CLUSTER:-}"
case "${CLUSTER}" in
  deltaai)
    module load python/miniforge3_pytorch/2.10.0
    VENV="${AI_ROSSBY_VENV:-.venv-deltaai}" ;;
  *)
    VENV="${AI_ROSSBY_VENV:-.venv}" ;;
esac
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

: "${AI_ROSSBY_DATA:?set AI_ROSSBY_DATA to the physicsnemo-zarr root}"
: "${RSI_CKPT:?set RSI_CKPT to the trained checkpoint directory}"
export AI_ROSSBY_DATA

MODE="${RSI_EVAL:-climatology}"
RUNG="${RSI_RUNG:-a2}"
LOSS="${RSI_LOSS:-rsi}"
OUTDIR="./outputs/rsi_${RUNG}_eval"
mkdir -p "${OUTDIR}"

COMMON=(
  --config-dir=examples/weather/ai_rossby/conf
  --config-name=config
  model=amip_rsi_v2
  "loss=${LOSS}"
  loss.window_size=6
  dataset=amip_dailyavg_coarse
)

nvidia-smi || true

case "${MODE}" in
  climatology)
    # Gate 1. `sampler=from_loss` (the default) is deliberate: it reuses the
    # TRAINING scheduler, so the interpolant profiles cannot drift from the
    # ones the heads were regressed against.
    python examples/weather/ai_rossby/eval_diffusion.py \
        "${COMMON[@]}" \
        validation=eval_suite \
        "+eval_suite.checkpoint_dir=${RSI_CKPT}" \
        "+eval_suite.output_path=${OUTDIR}/eval_suite.pt" \
        "run_name=rsi_${RUNG}_climatology"
    echo "climatology/bias suite -> ${OUTDIR}/eval_suite.pt"
    ;;

  rollout40)
    # Gate 2. Same driver as gate 1 — eval_diffusion IS the long-horizon
    # rollout; 40 years is a horizon setting, not a different program.
    # (rollout.py is the two-stage CASCADE driver and wants model=amip_combined,
    # so it is the wrong tool for a forecaster-only stability check.)
    #
    # eval_suite.horizon is in MODEL STEPS, and the AMIP model steps 24 h.
    # n_bins x months_per_bin is what the binned aggregators actually cover, so
    # the bin count is raised to match — a longer horizon otherwise keeps
    # rolling but stops accumulating past the twelfth bin, which would quietly
    # reduce a 40-year run to a 1-year climatology.
    YEARS="${RSI_ROLLOUT_YEARS:-40}"
    HORIZON=$(( YEARS * 365 ))
    NBINS=$(( YEARS * 12 ))
    python examples/weather/ai_rossby/eval_diffusion.py \
        "${COMMON[@]}" \
        validation=eval_suite \
        "++eval_suite.horizon=${HORIZON}" \
        "++eval_suite.climatology.n_bins=${NBINS}" \
        "++eval_suite.bias.n_bins=${NBINS}" \
        "+eval_suite.checkpoint_dir=${RSI_CKPT}" \
        "+eval_suite.output_path=${OUTDIR}/rollout40.pt" \
        "run_name=rsi_${RUNG}_rollout40"
    echo "${YEARS}-year rollout (${HORIZON} steps, ${NBINS} monthly bins) -> ${OUTDIR}/rollout40.pt"
    ;;

  calibration)
    # Rung a5. eps(tau) preserves the marginals in the exact-score limit, so
    # the whole sweep runs against ONE trained checkpoint with no retraining.
    for e in ${RSI_EPS_LIST:-0.0 0.01 0.05 0.1 0.2}; do
      echo "=== eps_scale=${e} ==="
      python examples/weather/ai_rossby/validate_diffusion.py \
          "${COMMON[@]}" \
          validation=rsi_calibration \
          sampler=rsi \
          "++sampler.eps_scale=${e}" \
          "++checkpoint_dir=${RSI_CKPT}" \
          "run_name=rsi_${RUNG}_eps${e}"
    done
    echo "eps sweep done — read spread-skill ratio + rank histograms per lead band."
    ;;

  *)
    echo "ERROR: RSI_EVAL must be climatology|rollout40|calibration, got '${MODE}'." >&2
    exit 1 ;;
esac
