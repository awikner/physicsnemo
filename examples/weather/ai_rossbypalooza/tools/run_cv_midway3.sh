#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# 5-fold cross-validation over 2000-2024: each fold trains on 20 years and
# validates on the held-out 5.
#
#   fold 1  val 2000-2004   fold 4  val 2015-2019
#   fold 2  val 2005-2009   fold 5  val 2020-2024  (the original split)
#   fold 3  val 2010-2014
#
# Per fold, the precip normalisation stats AND the SEEPS/ACC climatology are
# refitted on that fold's TRAINING years only. Without this the 20-year
# climatology would contain the held-out years, leaking them into both the ACC
# anomaly reference and the SEEPS categories -- mild, but a reviewer would
# rightly call it. (The ERA5 dynamical stats are left as-is: they come from
# ERA5 rather than the target, and only condition the inputs.)
#
# Each fold is a CPU stats job, then a GPU training job depending on it. Folds
# chain one after another so a single node runs them serially.
#
#   bash run_cv_midway3.sh [LOSS_CONFIG] [AFTER_JOBID] [MODE]
#
# MODE is all (default), stats, or train. The per-fold stats are
# loss-INDEPENDENT, so `stats` can run on CPU while a GPU sweep is still
# deciding which loss to cross-validate; `train` then submits only the GPU
# folds, chained serially, assuming the stats stores already exist.
#
# AFTER_JOBID optionally delays the first submitted job until an existing job
# finishes, so an in-flight sweep is not disturbed.
set -uo pipefail

LOSS="${1:-regional_mse_physical}"
AFTER="${2:-}"
MODE="${3:-all}"
case "$MODE" in all|stats|train) ;; *) echo "bad MODE '$MODE'"; exit 2;; esac
REPO=/scratch/midway3/awikner/physicsnemo
RECIPE="$REPO/examples/weather/ai_rossbypalooza"
NORM=/scratch/midway2/awikner/physicsnemo-zarr/normalization
IMERG=/scratch/midway2/awikner/physicsnemo-zarr/imerg
RUNDIR=/scratch/midway3/awikner/mowe_runs
mkdir -p "$RUNDIR"

fold_val_lo=(2000 2005 2010 2015 2020)
fold_val_hi=(2004 2009 2014 2019 2024)

prev="$AFTER"
for k in 0 1 2 3 4; do
    f=$((k + 1))
    vlo=${fold_val_lo[$k]}
    vhi=${fold_val_hi[$k]}
    excl=$(seq -s, "$vlo" "$vhi")
    # Training years = 2000-2024 minus the fold, as comma-separated ranges.
    if [ "$vlo" -eq 2000 ]; then
        tspec="$((vhi + 1))-2024"
    elif [ "$vhi" -eq 2024 ]; then
        tspec="2000-$((vlo - 1))"
    else
        tspec="2000-$((vlo - 1)),$((vhi + 1))-2024"
    fi

    stats_id=""
    if [ "$MODE" != "train" ]; then
    dep_stats=""
    [ -n "$prev" ] && dep_stats="--dependency=afterany:$prev"
    # pedramh-gpu (no GPU requested), NOT caslake: /scratch/midway2 is not
    # mounted on every caslake node -- midway3-0025 silently has no mount and
    # the tools then report "no finite data" rather than a missing filesystem.
    # Our dedicated node is verified to mount it. The precheck makes any future
    # recurrence obvious instead of looking like a data problem.
    stats_id=$(sbatch $dep_stats -A pi-pedramh -p pedramh-gpu -N 1 -c 4 --mem=32G \
        -t 01:00:00 -J "cv${f}_stats" -o "$RUNDIR/cv${f}_stats.log" --wrap="\
set -e
if [ ! -d $IMERG/2015.zarr ]; then
  echo \"FATAL: /scratch/midway2 not mounted on \$(hostname) -- resubmit elsewhere\"
  exit 75
fi
source $REPO/.venv-mowe/bin/activate
python $RECIPE/tools/compute_precip_norm.py --imerg-root $IMERG \
  --years $tspec --log-epsilon 1e-3 --log-units m \
  --out $NORM/imerg_precip_stats_log_cv${f}.zarr --commit cv${f}
python $RECIPE/tools/compute_seeps_climatology.py --imerg-root $IMERG \
  --years $tspec --out $NORM/imerg_seeps_climatology_cv${f}.zarr --commit cv${f}" \
        2>&1 | grep -oP 'Submitted batch job \K\d+')
    echo "fold $f: stats job $stats_id (train years $tspec)"
    prev="$stats_id"
    fi

    if [ "$MODE" = "stats" ]; then
        continue
    fi

    dep_train=""
    if [ -n "$stats_id" ]; then
        dep_train="--dependency=afterok:$stats_id"
    elif [ -n "$prev" ]; then
        dep_train="--dependency=afterany:$prev"
    fi
    # The overrides contain commas, which Slurm's --export list would treat as
    # separators and truncate, so hand them over in a file instead.
    ov="$RUNDIR/cv${f}_overrides.txt"
    cat > "$ov" <<EOV
loss=$LOSS dataset.train.years=[2000,2024] dataset.train.exclude_years=[$excl] dataset.val.years=[$vlo,$vhi] dataset.normalization.precip_stats=$NORM/imerg_precip_stats_log_cv${f}.zarr dataset.normalization.seeps_climatology=$NORM/imerg_seeps_climatology_cv${f}.zarr
EOV
    train_id=$(sbatch $dep_train \
        --export=ALL,RUN_NAME="mowe_cv${f}",WANDB=true,EXTRA_FILE="$ov" \
        "$RECIPE/tools/train_mowe_midway3_h100.sbatch" \
        2>&1 | grep -oP 'Submitted batch job \K\d+')
    echo "fold $f: train job $train_id (val $vlo-$vhi, loss $LOSS)"
    prev="$train_id"
done

echo
echo "Submitted mode=$MODE for 5 folds. Collect with:"
echo "  grep -H 'training complete' $RUNDIR/train_h100_*.log"
