#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Convert one DSI AIWP hindcast model into per-sample .h5 + per-year zarr.
# Submit: sbatch --job-name=dsi_<model> convert_model_dsi.sbatch <model>
#SBATCH --account=general_group
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/awikner/hindcasts_dsi/logs/%x_%j.out
#SBATCH --error=/net/scratch/awikner/hindcasts_dsi/logs/%x_%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch convert_model_dsi.sbatch <model>}"
PY="$HOME/miniconda3/envs/hindcast_conv/bin/python"
CONV="$HOME/hindcast_tools/dsi_hindcast_to_formats.py"
OUT="/net/scratch/awikner/hindcasts_dsi"
W=32
MON="/net/monsoon/marchakitus"

declare -a SD
case "$MODEL" in
  graphcast_e2s)    KIND=zarr; DS=reforecast;      SD=("$MON/reforecast/forecasts_graphcast_e2s");;
  aurora_e2s)       KIND=zarr; DS=reforecast;      SD=("$MON/reforecast/forecasts_aurora_e2s");;
  aifs_single_v2)   KIND=zarr; DS=reforecast;      SD=("$MON/reforecast/forecasts_AIFS_v2");;
  aifs_single_v1)   KIND=zarr; DS=monsoon_paper;   SD=("$MON/monsoon_paper_archived/AIFS/output_twice_weekly_paper_0z" "$MON/monsoon_paper_archived/AIFS/output_daily_paper_0z");;
  aifs_single_v1p1) KIND=zarr; DS=monsoon_paper;   SD=("$MON/monsoon_paper_archived/AIFS_1.1/output_daily");;
  graphcast_wb2)    KIND=nc;   DS=monsoon_paper;   SD=("$MON/monsoon_paper/graphcast/inference_gcast_twice_weekly_0z");;
  *) echo "unknown model: $MODEL" >&2; exit 1;;
esac

SRCARGS=()
for d in "${SD[@]}"; do SRCARGS+=(--source-dir "$d"); done

echo "[$(date)] host=$(hostname) model=$MODEL kind=$KIND workers=$W"
echo "sources: ${SD[*]}"
srun "$PY" "$CONV" --model "$MODEL" --source-kind "$KIND" "${SRCARGS[@]}" \
    --out-root "$OUT" --years 2000-2024 --n-workers "$W" \
    --source-dataset "$DS" --overwrite
echo "[$(date)] DONE $MODEL"
