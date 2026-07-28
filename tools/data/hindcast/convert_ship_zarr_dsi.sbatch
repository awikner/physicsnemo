#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# DSI side, DIRECT-to-monsoon + Globus-recursive (no scratch2, no tar):
#   * convert source -> Format-2 zarr written straight onto /net/monsoon
#     (monsoon writes are fast ~369 MB/s; scratch2 small-file READ was the 7 MB/s
#     killer, so we avoid scratch2 entirely)
#   * Globus-transfer the store RECURSIVELY monsoon -> Derecho final location
#     (few big ~43 MB chunk files; no tar/untar), then a tiny .ready sentinel
#   * delete the monsoon store; sequential (full monsoon bw for each phase),
#     retries for the flaky NFS, self-chaining across 12 h boundaries
# The per-sample h5 is regenerated on Derecho from these zarr stores (develop q).
#
#SBATCH --job-name=dsi_zarrship
#SBATCH --account=general_group
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/net/monsoon/awikner/hindcast_stage/zarrship_%j.log

set -uo pipefail
PY="$HOME/miniconda3/envs/hindcast_conv/bin/python"
CONV="$HOME/hindcast_tools/dsi_hindcast_to_formats.py"
GL="$HOME/venvs/globus/bin/globus"
STAGE=/net/monsoon/awikner/hindcast_stage        # convert output ON monsoon (Globus-readable)
STAGE_COLL=/awikner/hindcast_stage               # STAGE as seen by DSI_COLL (root=/net/monsoon)
DONE="$STAGE/done"
DSI_COLL=d6f6509d-3ff2-4c79-b953-d404f06d243d
DST_COLL=d33b3614-6d04-11e5-ba46-22000b92c6ec
DST=/glade/derecho/scratch/awikner/hindcasts_dsi
MON=/net/monsoon/marchakitus
W="${W:-12}"                                      # moderate: limit monsoon read+write load
mkdir -p "$STAGE/zarr" "$DONE" "$STAGE/.rdy"
echo "======== [$(date)] zarrship(direct) start host=$(hostname) W=$W ========"

# On SLURM walltime, SIGTERM arrives ~30s before SIGKILL. Cancel any in-flight
# ship so it does not orphan (a server-side Globus transfer survives scancel and
# would then race the successor's --overwrite re-convert -> FILE_NOT_FOUND faults).
CUR_TID=""
_on_term() {
  [ -n "$CUR_TID" ] && "$GL" task cancel "$CUR_TID" >/dev/null 2>&1
  echo "[$(date)] SIGTERM: cancelled in-flight ship ${CUR_TID:-none}"; exit 143
}
trap _on_term TERM

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

TOTAL_CHUNKS=131
submitted_succ=0
maybe_submit_successor() {
  [ "$submitted_succ" = 1 ] && return
  local nd; nd=$(ls "$DONE"/*.done 2>/dev/null | wc -l)
  [ "$nd" -ge "$TOTAL_CHUNKS" ] && return
  sbatch --parsable --dependency="afterany:${SLURM_JOB_ID:-0}" \
      --export="ALL,MODELS=${MODELS:-},YEARS=${YEARS:-}" \
      "$HOME/hindcast_tools/convert_ship_zarr_dsi.sbatch" >/dev/null 2>&1 && submitted_succ=1 \
      && echo "[$(date)] chained successor (afterany ${SLURM_JOB_ID:-0}) MODELS='${MODELS:-all}'; progress ${nd}/${TOTAL_CHUNKS}"
}

ship_chunk() {   # globus the store recursively -> Derecho, drop a .ready sentinel, free monsoon
  local model="$1" year="$2" tid st
  local store="$STAGE/zarr/$model/$year.zarr"
  [ -d "$store" ] || { touch "$DONE/${model}_${year}.done"; return 0; }
  tid=$("$GL" transfer --recursive "$DSI_COLL:$STAGE_COLL/zarr/$model/$year.zarr" \
        "$DST_COLL:$DST/zarr/$model/$year.zarr" --sync-level mtime --preserve-mtime \
        --notify off --label "zs ${model} ${year}" --format unix --jmespath task_id 2>>"$STAGE/ship.log")
  [ -z "$tid" ] && { echo "[ERR] ship submit $model $year" | tee -a "$STAGE/ship.log"; return 1; }
  CUR_TID="$tid"   # track for the SIGTERM trap
  if "$GL" task wait "$tid" --timeout 14400 >>"$STAGE/ship.log" 2>&1 && \
     [ "$("$GL" task show "$tid" --format unix --jmespath status 2>>"$STAGE/ship.log")" = SUCCEEDED ]; then
    # completion sentinel (transferred AFTER the store -> guarantees store is complete)
    touch "$STAGE/.rdy/${model}_${year}.ready"
    "$GL" transfer "$DSI_COLL:$STAGE_COLL/.rdy/${model}_${year}.ready" \
        "$DST_COLL:$DST/zarr/.ready/${model}_${year}.ready" --notify off \
        --format unix --jmespath task_id >/dev/null 2>>"$STAGE/ship.log" || true
    rm -rf "$store"; rm -f "$STAGE/.rdy/${model}_${year}.ready"
    touch "$DONE/${model}_${year}.done"; CUR_TID=""
    echo "[$(date)] SHIPPED $model $year (tid=$tid)"; return 0
  fi
  CUR_TID=""
  echo "[ERR] ship $model $year tid=$tid" | tee -a "$STAGE/ship.log"; return 1
}

do_chunk() {
  local model="$1" year="$2"
  [ -f "$DONE/${model}_${year}.done" ] && { echo "[skip] $model $year"; return 2; }
  local kind ds rest; read -r kind ds rest <<<"$(cfg_for "$model")" || { echo "[ERR] model $model"; return 1; }
  local sd=($rest) srcargs=() d; for d in "${sd[@]}"; do srcargs+=(--source-dir "$d"); done

  echo "[$(date)] CONVERT(zarr->monsoon) $model $year (W=$W)"
  if ! "$PY" "$CONV" --model "$model" --source-kind "$kind" "${srcargs[@]}" --formats zarr \
        --out-root "$STAGE" --years "$year-$year" --n-workers "$W" --source-dataset "$ds" --overwrite; then
    echo "[ERR] convert $model $year"; return 1
  fi
  [ -d "$STAGE/zarr/$model/$year.zarr" ] || { echo "[skip] $model $year no store"; touch "$DONE/${model}_${year}.done"; return 0; }

  local a=0
  until ship_chunk "$model" "$year"; do
    a=$((a+1)); [ "$a" -ge 6 ] && { echo "[ERR] ship $model $year after 6 attempts"; return 1; }
    echo "[warn] ship attempt $a failed; retry in 180s"; sleep 180   # ride out transient Globus 409s / congestion
  done
}

MODELS="${MODELS:-graphcast_e2s aurora_e2s aifs_single_v2 aifs_single_v1 aifs_single_v1p1 graphcast_wb2}"
for model in $MODELS; do
  if [ -n "${YEARS:-}" ]; then years="$YEARS"
  elif [ "$model" = aifs_single_v1p1 ]; then years=$(seq 2019 2024)
  else years=$(seq 2000 2024); fi
  for y in $years; do
    do_chunk "$model" "$y"; rc=$?
    case "$rc" in
      0) maybe_submit_successor ;;
      2) : ;;
      *) echo "[STOP] $model $y (rc=$rc) — not chaining (needs attention)"; exit 1 ;;
    esac
  done
done
echo "======== [$(date)] ALL ZARR CHUNKS DONE (this job) ========"
