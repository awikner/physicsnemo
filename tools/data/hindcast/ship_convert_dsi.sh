#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Incremental convert-on-DSI + Globus-ship to Derecho, staying under a hard
# monsoon staging cap. For each (model, year) chunk: convert -> Globus-ship the
# chunk's h5 + zarr from /net/monsoon -> Derecho -> delete the staged chunk.
# Sequential, so at most ONE chunk (~<=0.5 TB) is ever staged: well under 1 TB.
#
# Resumable: a marker in $STAGE/done/<model>_<year>.done is written after a
# chunk transfers + verifies; reruns skip completed chunks.
#
# Env overrides: W (workers, default 12), MODELS, STAGE.
# Run:  nohup bash ship_convert_dsi.sh >/dev/null 2>&1 &   (log at $STAGE/driver.log)

set -uo pipefail
PY="$HOME/miniconda3/envs/hindcast_conv/bin/python"
CONV="$HOME/hindcast_tools/dsi_hindcast_to_formats.py"
GL="$HOME/venvs/globus/bin/globus"
STAGE="${STAGE:-/net/monsoon/awikner/hindcast_stage}"
DONE="$STAGE/done"
DSI_COLL=d6f6509d-3ff2-4c79-b953-d404f06d243d      # CeTD Group Monsoon Storage (root /net/monsoon)
DST_COLL=d33b3614-6d04-11e5-ba46-22000b92c6ec      # NCAR GLADE
DST=/glade/derecho/scratch/awikner/hindcasts_dsi
COLLROOT=/awikner/hindcast_stage                   # $STAGE as seen by DSI_COLL (root=/net/monsoon)
MON=/net/monsoon/marchakitus
W="${W:-12}"
CAP_GB="${CAP_GB:-950}"                             # hard safety cap on monsoon staging
LOG="$STAGE/driver.log"
mkdir -p "$STAGE" "$DONE"
exec >>"$LOG" 2>&1

echo "======== [$(date)] driver start (host=$(hostname) W=$W cap=${CAP_GB}GB) ========"

cfg_for() {  # -> "kind dataset srcdir [srcdir...]"
  case "$1" in
    graphcast_e2s)    echo "zarr reforecast    $MON/reforecast/forecasts_graphcast_e2s";;
    aurora_e2s)       echo "zarr reforecast    $MON/reforecast/forecasts_aurora_e2s";;
    aifs_single_v2)   echo "zarr reforecast    $MON/reforecast/forecasts_AIFS_v2";;
    aifs_single_v1)   echo "zarr monsoon_paper $MON/monsoon_paper_archived/AIFS/output_twice_weekly_paper_0z $MON/monsoon_paper_archived/AIFS/output_daily_paper_0z";;
    aifs_single_v1p1) echo "zarr monsoon_paper $MON/monsoon_paper_archived/AIFS_1.1/output_daily";;
    graphcast_wb2)    echo "nc   monsoon_paper $MON/monsoon_paper/graphcast/inference_gcast_twice_weekly_0z";;
    *) return 1;;
  esac
}

stage_gb() { du -sb "$STAGE"/h5 "$STAGE"/zarr 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s/1e9}'; }

do_chunk() {
  local model="$1" year="$2"
  local marker="$DONE/${model}_${year}.done"
  [ -f "$marker" ] && { echo "[skip] $model $year (already done)"; return 0; }
  read -r kind ds rest <<<"$(cfg_for "$model")" || { echo "[ERR] unknown model $model"; return 1; }
  local sd=($rest) srcargs=() d
  for d in "${sd[@]}"; do srcargs+=(--source-dir "$d"); done

  echo "[$(date)] CONVERT $model $year (W=$W)"
  if ! nice -n 10 "$PY" "$CONV" --model "$model" --source-kind "$kind" "${srcargs[@]}" \
        --out-root "$STAGE" --years "$year-$year" --n-workers "$W" \
        --source-dataset "$ds" --overwrite; then
    echo "[ERR] convert failed $model $year"; return 1
  fi

  local used; used=$(stage_gb)
  echo "[$(date)] staged now: ${used} GB"
  if [ -n "$used" ] && [ "$used" -gt "$CAP_GB" ]; then
    echo "[ABORT] staged ${used} GB exceeds cap ${CAP_GB} GB — stopping to protect the 1 TB limit"; return 1
  fi

  shopt -s nullglob
  local idirs=("$STAGE"/h5/"$model"/init_"${year}"*)
  local zstore="$STAGE/zarr/$model/$year.zarr"
  if [ ${#idirs[@]} -eq 0 ] && [ ! -d "$zstore" ]; then
    echo "[skip] $model $year produced no inits"; touch "$marker"; return 0
  fi

  local batch; batch=$(mktemp)
  local idir nm
  for idir in "${idirs[@]}"; do
    nm=$(basename "$idir")
    echo "--recursive $COLLROOT/h5/$model/$nm $DST/h5/$model/$nm" >>"$batch"
  done
  [ -d "$zstore" ] && echo "--recursive $COLLROOT/zarr/$model/$year.zarr $DST/zarr/$model/$year.zarr" >>"$batch"
  echo "[$(date)] SHIP $model $year ($(wc -l <"$batch") items)"

  local tid; tid=$("$GL" transfer "$DSI_COLL" "$DST_COLL" --batch "$batch" \
      --label "dsi_conv ${model} ${year}" --sync-level mtime --preserve-mtime \
      --notify off --format unix --jmespath task_id 2>>"$LOG")
  rm -f "$batch"
  [ -z "$tid" ] && { echo "[ERR] transfer submit failed $model $year"; return 1; }
  echo "[$(date)] task=$tid ; waiting..."
  if ! "$GL" task wait "$tid" --timeout 21600 >>"$LOG" 2>&1; then
    echo "[ERR] ship wait failed/timeout $model $year task=$tid"; return 1
  fi
  local st; st=$("$GL" task show "$tid" --format unix --jmespath status 2>>"$LOG")
  [ "$st" = SUCCEEDED ] || { echo "[ERR] ship status=$st $model $year task=$tid"; return 1; }

  rm -rf "${idirs[@]}" "$zstore"
  touch "$marker"
  echo "[$(date)] DONE $model $year (shipped + cleaned)"
}

MODELS="${MODELS:-graphcast_e2s aurora_e2s aifs_single_v2 aifs_single_v1 aifs_single_v1p1 graphcast_wb2}"
for model in $MODELS; do
  if [ -n "${YEARS:-}" ]; then
    years="$YEARS"
  else
    case "$model" in
      aifs_single_v1p1) years=$(seq 2019 2024);;
      *)                years=$(seq 2000 2024);;
    esac
  fi
  for y in $years; do
    if ! do_chunk "$model" "$y"; then
      echo "[STOP] failure at $model $y — exiting (rerun to resume)"; exit 1
    fi
  done
done
echo "======== [$(date)] ALL CHUNKS COMPLETE ========"
