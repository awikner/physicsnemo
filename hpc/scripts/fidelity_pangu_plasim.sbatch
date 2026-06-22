#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 5 fidelity gate — non-interactive (queues longer than the
# `gpuA40x4-interactive` 1-hour cap can accommodate).
#
# This script does three things, in order:
#   1. Translate a PanguWeather PANGU_PLASIM checkpoint (.pt) → ai-rossby
#      .mdlus via tools/checkpoint_translation/pangu_plasim.py.
#   2. Run PanguWeather's own inference on the source .pt to capture
#      reference predictions at a known IC. (Requires PanguWeather's
#      conda env + access to its source tree.)
#   3. Run tools/checkpoint_translation/fidelity_compare.py to assert
#      the translated model's predictions match the reference within
#      tolerance.
#
# All four PANGU_PLASIM / SFNO_PLASIM combinations share this harness
# — change the env-var inputs at the top to point at the right .pt /
# YAML / target-class.
#
# REQUIRED INPUTS (override via --export or edit below):
#   PW_CHECKPOINT    PanguWeather .pt checkpoint
#   PW_CONFIG_NAME   PanguWeather config section (e.g. PANGU, SFNO)
#   PW_YAML          PanguWeather YAML config (e.g. config/PANGU_PLASIM_*.yaml)
#   TARGET_YAML      ai-rossby YAML (conf/model/{pangu_plasim_legacy,pangu_plasim,sfno_plasim}.yaml)
#   TARGET_CLASS     PanguPlasimLegacy | PanguPlasim | SfnoPlasim
#   IC_INDEX         Reference IC index (integer; default 0)
#   TOLERANCE        Max-abs-diff tolerance (default 1e-4)
#
# Submit with:
#   sbatch --export=ALL,PW_CHECKPOINT=/path,PW_YAML=/path,TARGET_YAML=/path,TARGET_CLASS=PanguPlasimLegacy \
#          hpc/scripts/fidelity_pangu_plasim.sbatch

#SBATCH --partition=gpuA40x4
#SBATCH --account=bdiu-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --job-name=pn-fidelity
#SBATCH --output=hpc/scripts/logs/fidelity-pangu-%j.out

set -euo pipefail

# Defaults — required inputs that must be exported, or fail loudly.
: "${PW_CHECKPOINT:?need PW_CHECKPOINT (PanguWeather .pt checkpoint path)}"
: "${PW_YAML:?need PW_YAML (PanguWeather YAML config path)}"
: "${PW_CONFIG_NAME:=SFNO}"
: "${TARGET_YAML:?need TARGET_YAML (ai-rossby model YAML)}"
: "${TARGET_CLASS:?need TARGET_CLASS (PanguPlasim | PanguPlasimLegacy | SfnoPlasim)}"
: "${IC_INDEX:=0}"
: "${TOLERANCE:=1e-4}"

cd /work/nvme/bdiu/awikner/physicsnemo

OUT_DIR=hpc/scripts/logs/fidelity-${SLURM_JOB_ID}
mkdir -p "${OUT_DIR}"

MDLUS_PATH=${OUT_DIR}/translated.mdlus
REF_PATH=${OUT_DIR}/reference_predictions.nc

echo "=== inputs ==="
echo "PW_CHECKPOINT = ${PW_CHECKPOINT}"
echo "PW_YAML       = ${PW_YAML}"
echo "PW_CONFIG_NAME = ${PW_CONFIG_NAME}"
echo "TARGET_YAML   = ${TARGET_YAML}"
echo "TARGET_CLASS  = ${TARGET_CLASS}"
echo "IC_INDEX      = ${IC_INDEX}"
echo "TOLERANCE     = ${TOLERANCE}"
echo "OUT_DIR       = ${OUT_DIR}"

# ---------------------------------------------------------------------- #
# Step 1: translate                                                       #
# ---------------------------------------------------------------------- #
source .venv/bin/activate
echo
echo "=== step 1/3: translate ${PW_CHECKPOINT} → ${MDLUS_PATH} ==="
TRANSLATOR=tools/checkpoint_translation/pangu_plasim.py
if [ "${TARGET_CLASS}" = "SfnoPlasim" ]; then
    TRANSLATOR=tools/checkpoint_translation/sfno_plasim.py
fi
python "${TRANSLATOR}" \
    --source "${PW_CHECKPOINT}" \
    --model-config "${TARGET_YAML}" \
    --output "${MDLUS_PATH}" \
    --strict

# ---------------------------------------------------------------------- #
# Step 2: run PanguWeather's own inference for the reference             #
# ---------------------------------------------------------------------- #
# Activated the PanguWeather conda env (mirrors bench_sfno_panguweather.sbatch).
set +u
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh
conda activate /projects/bdiu/awikner/.conda/envs/aires_panguplasim_cu126
set -u

PW_DIR=/work/nvme/bdiu/awikner/PanguWeather/v2.0
echo
echo "=== step 2/3: PanguWeather inference → ${REF_PATH} ==="
# PanguWeather's inference utility is run_inference.py; the exact flag
# names vary by checkpoint. We launch the documented form below, and
# the user is expected to tailor the args list once for their actual
# checkpoint structure. The output should land at ${REF_PATH} as a
# NetCDF with per-channel-group pred_* + input arrays (one frame per
# rollout step).
# NOTE: this stage requires real configuration that depends on your
# checkpoint. Edit the line below to match your model / data setup.
#
# python "${PW_DIR}/run_inference.py" \
#     --yaml_config="${PW_YAML}" \
#     --config="${PW_CONFIG_NAME}" \
#     --checkpoint="${PW_CHECKPOINT}" \
#     --ic_index=${IC_INDEX} \
#     --output="${REF_PATH}"

# Until the inference invocation is wired, fail loudly so the operator
# sees this step is the one that needs adapting per-checkpoint.
echo "ERROR: step 2/3 is intentionally stubbed — edit fidelity_pangu_plasim.sbatch"
echo "       to call PanguWeather's actual inference utility against ${PW_CHECKPOINT}"
echo "       and write per-frame predictions to ${REF_PATH}."
exit 2

# ---------------------------------------------------------------------- #
# Step 3: compare                                                         #
# ---------------------------------------------------------------------- #
# Switch back to the ai-rossby venv for the comparison.
conda deactivate || true
set +u
source .venv/bin/activate
set -u
echo
echo "=== step 3/3: compare ${MDLUS_PATH} vs ${REF_PATH} ==="
python tools/checkpoint_translation/fidelity_compare.py \
    --mdlus "${MDLUS_PATH}" \
    --model-config "${TARGET_YAML}" \
    --target-class "${TARGET_CLASS}" \
    --reference "${REF_PATH}" \
    --frame ${IC_INDEX} \
    --tolerance ${TOLERANCE} \
    --device cuda:0
RC=$?

echo
echo "=== exit code: ${RC} ==="
exit ${RC}
