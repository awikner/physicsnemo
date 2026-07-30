#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# DSI-side incremental convert + ship, engineered to NOT saturate /net/monsoon:
#   * runs on a COMPUTE node (not the login node)
#   * conversion output goes to fast /net/scratch2 (monsoon does READS ONLY while converting)
#   * each chunk is tarred (zstd for the uncompressed h5) onto /net/monsoon, then
#     Globus-shipped to Derecho and the monsoon tar deleted
#   * ONE ship overlaps the next convert (endpoint-driven Globus vs compute), so
#     monsoon never sees a heavy simultaneous read+write from us
#   * staging footprints stay tiny: scratch2 ~<=1 chunk, monsoon ~<=2 chunk-tars
# Resumable via markers in $SCR/done.
#
#SBATCH --job-name=dsi_convship
#SBATCH --account=general_group
#SBATCH --partition=Monsoon
#SBATCH --qos=monsoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=/net/scratch2/awikner/hindcast_work/convship_%j.log

set -uo pipefail
PY="$HOME/miniconda3/envs/hindcast_conv/bin/python"
CONV="$HOME/hindcast_tools/dsi_hindcast_to_formats.py"
GL="$HOME/venvs/globus/bin/globus"
SCR=/net/scratch2/awikner/hindcast_work           # fast convert output (monsoon reads only)
MTAR=/net/monsoon/awikner/hindcast_tars           # monsoon tar staging (kept tiny)
MTAR_COLL=/awikner/hindcast_tars                  # $MTAR as seen by DSI_COLL (root=/net/monsoon)
DONE="$SCR/done"
DSI_COLL=d6f6509d-3ff2-4c79-b953-d404f06d243d
DST_COLL=d33b3614-6d04-11e5-ba46-22000b92c6ec
DTAR=/glade/derecho/scratch/awikner/hindcast_tars # Derecho tar landing (untar loop consumes)
MON=/net/monsoon/marchakitus
W="${W:-24}"                                      # modest: don't saturate monsoon reads
mkdir -p "$SCR/h5" "$SCR/zarr" "$MTAR" "$DONE"
echo "======== [$(date)] convship start host=$(hostname) W=$W ========"

cfg_for() {
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

ship_pid=""
ship_chunk() {   # background: globus the chunk's tars -> Derecho, then delete monsoon tars + mark done
  local model="$1" year="$2" batch tid st
  batch=$(mktemp)
  [ -f "$MTAR/${model}_${year}_h5.tar.zst" ] && echo "$MTAR_COLL/${model}_${year}_h5.tar.zst $DTAR/${model}_${year}_h5.tar.zst" >>"$batch"
  [ -f "$MTAR/${model}_${year}_zarr.tar" ]    && echo "$MTAR_COLL/${model}_${year}_zarr.tar $DTAR/${model}_${year}_zarr.tar" >>"$batch"
  if [ ! -s "$batch" ]; then rm -f "$batch"; touch "$DONE/${model}_${year}.done"; return 0; fi
  tid=$("$GL" transfer "$DSI_COLL" "$DST_COLL" --batch "$batch" --sync-level mtime --preserve-mtime \
        --notify off --label "cs ${model} ${year}" --format unix --jmespath task_id 2>>"$SCR/ship.log")
  rm -f "$batch"
  if [ -z "$tid" ]; then echo "[ERR] ship submit $model $year" | tee -a "$SCR/ship.log"; return 1; fi
  if "$GL" task wait "$tid" --timeout 21600 >>"$SCR/ship.log" 2>&1; then
    st=$("$GL" task show "$tid" --format unix --jmespath status 2>>"$SCR/ship.log")
    if [ "$st" = SUCCEEDED ]; then
      rm -f "$MTAR/${model}_${year}_h5.tar.zst" "$MTAR/${model}_${year}_zarr.tar"
      touch "$DONE/${model}_${year}.done"
      echo "[$(date)] SHIPPED $model $year (tid=$tid)"
      return 0
    fi
  fi
  echo "[ERR] ship $model $year tid=$tid" | tee -a "$SCR/ship.log"; return 1
}

do_chunk() {
  local model="$1" year="$2"
  [ -f "$DONE/${model}_${year}.done" ] && { echo "[skip] $model $year"; return 0; }
  local kind ds rest; read -r kind ds rest <<<"$(cfg_for "$model")" || { echo "[ERR] model $model"; return 1; }
  local sd=($rest) srcargs=() d; for d in "${sd[@]}"; do srcargs+=(--source-dir "$d"); done

  echo "[$(date)] CONVERT $model $year -> scratch2 (W=$W)"
  if ! "$PY" "$CONV" --model "$model" --source-kind "$kind" "${srcargs[@]}" \
        --out-root "$SCR" --years "$year-$year" --n-workers "$W" --source-dataset "$ds" --overwrite; then
    echo "[ERR] convert $model $year"; return 1
  fi

  shopt -s nullglob
  local idirs=("$SCR"/h5/"$model"/init_"${year}"*)
  local zstore="$SCR/zarr/$model/$year.zarr"
  if [ ${#idirs[@]} -eq 0 ] && [ ! -d "$zstore" ]; then echo "[skip] $model $year no inits"; touch "$DONE/${model}_${year}.done"; return 0; fi

  echo "[$(date)] TAR $model $year -> monsoon"
  if [ ${#idirs[@]} -gt 0 ]; then
    ( cd "$SCR/h5/$model" && tar -I 'zstd -1 -T0' -cf "$MTAR/${model}_${year}_h5.tar.zst" $(basename -a "${idirs[@]}") )
  fi
  [ -d "$zstore" ] && ( cd "$SCR/zarr/$model" && tar -cf "$MTAR/${model}_${year}_zarr.tar" "$year.zarr" )
  rm -rf "${idirs[@]}" "$zstore"          # free scratch2 immediately

  [ -n "$ship_pid" ] && wait "$ship_pid"  # keep at most ONE ship in flight (bounds monsoon tars)
  ship_chunk "$model" "$year" &
  ship_pid=$!
}

MODELS="${MODELS:-graphcast_e2s aurora_e2s aifs_single_v2 aifs_single_v1 aifs_single_v1p1 graphcast_wb2}"
for model in $MODELS; do
  if [ -n "${YEARS:-}" ]; then years="$YEARS"
  elif [ "$model" = aifs_single_v1p1 ]; then years=$(seq 2019 2024)
  else years=$(seq 2000 2024); fi
  for y in $years; do do_chunk "$model" "$y" || { echo "[STOP] $model $y"; [ -n "$ship_pid" ] && wait "$ship_pid"; exit 1; }; done
done
[ -n "$ship_pid" ] && wait "$ship_pid"
echo "======== [$(date)] ALL CHUNKS DONE ========"
