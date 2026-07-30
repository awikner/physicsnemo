#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# 5-fold cross-validation on Derecho for BOTH candidate losses, one 4xA100 node
# per run. 5 CPU stats jobs + 10 GPU training jobs (5 folds x 2 losses).
#
#   fold 1  val 2000-2004   fold 4  val 2015-2019
#   fold 2  val 2005-2009   fold 5  val 2020-2024
#   fold 3  val 2010-2014
#
# Unlike the Midway driver, the folds are NOT chained: Derecho has many GPU
# nodes, so all ten run concurrently and each training job depends only on its
# own fold's stats job.
#
# Per fold the precip normalisation stats AND the SEEPS/ACC climatology are
# refitted on that fold's TRAINING years only -- otherwise the 20-year
# climatology contains the held-out years and leaks them into the ACC anomaly
# reference and the SEEPS categories. The two losses SHARE a fold's stats (they
# depend on the split, not the objective). The ERA5 dynamical stats are left
# alone: they come from ERA5 rather than the target and only condition inputs.
#
#   bash run_cv_derecho.sh [--stats-only|--train-only]
set -uo pipefail

MODE="${1:-all}"
REPO=/glade/work/awikner/physicsnemo
RECIPE="$REPO/examples/weather/ai_rossbypalooza"
NORM=/glade/derecho/scratch/awikner/physicsnemo-zarr/normalization
IMERG=/glade/derecho/scratch/awikner/physicsnemo-zarr/imerg
RUNDIR=/glade/derecho/scratch/awikner/mowe_runs
ACCT=UCHI0018
mkdir -p "$RUNDIR"

LOSSES="regional_mse_physical regional_mse_physical_var"
fold_val_lo=(2000 2005 2010 2015 2020)
fold_val_hi=(2004 2009 2014 2019 2024)

for k in 0 1 2 3 4; do
    f=$((k + 1))
    vlo=${fold_val_lo[$k]}
    vhi=${fold_val_hi[$k]}
    excl=$(seq -s, "$vlo" "$vhi")
    if [ "$vlo" -eq 2000 ]; then
        tspec="$((vhi + 1))-2024"
    elif [ "$vhi" -eq 2024 ]; then
        tspec="2000-$((vlo - 1))"
    else
        tspec="2000-$((vlo - 1)),$((vhi + 1))-2024"
    fi

    stats_id=""
    if [ "$MODE" != "--train-only" ]; then
        stats_id=$(qsub -A "$ACCT" -q develop -N "cv${f}_stats" \
            -l select=1:ncpus=4:mem=40GB -l walltime=01:00:00 -j oe \
            -o "$RUNDIR/cv${f}_stats.log" - <<EOS
module load ncarenv 2>/dev/null || true
cd $REPO && source .venv/bin/activate
python $RECIPE/tools/compute_precip_norm.py --imerg-root $IMERG \
  --years $tspec --log-epsilon 1e-3 --log-units m \
  --out $NORM/imerg_precip_stats_log_cv${f}.zarr --commit cv${f}
python $RECIPE/tools/compute_seeps_climatology.py --imerg-root $IMERG \
  --years $tspec --out $NORM/imerg_seeps_climatology_cv${f}.zarr --commit cv${f}
EOS
)
        echo "fold $f: stats $stats_id (train years $tspec)"
    fi
    [ "$MODE" = "--stats-only" ] && continue

    for loss in $LOSSES; do
        tag=$([ "$loss" = "regional_mse_physical" ] && echo phys || echo physvar)
        ov="$RUNDIR/cv${f}_${tag}_overrides.txt"
        cat > "$ov" <<EOV
loss=$loss dataset.train.years=[2000,2024] dataset.train.exclude_years=[$excl] dataset.val.years=[$vlo,$vhi] dataset.normalization.precip_stats=$NORM/imerg_precip_stats_log_cv${f}.zarr dataset.normalization.seeps_climatology=$NORM/imerg_seeps_climatology_cv${f}.zarr
EOV
        dep=""
        [ -n "$stats_id" ] && dep="-W depend=afterok:$stats_id"
        # -v is comma-separated, so the overrides go in via EXTRA_FILE.
        jid=$(qsub $dep -A "$ACCT" -N "cv${f}_$tag" \
            -v RUN_NAME="mowe_cv${f}_$tag",WANDB=true,EXTRA_FILE="$ov" \
            -o "$RUNDIR/train_cv${f}_${tag}.log" \
            "$RECIPE/tools/train_mowe_derecho.pbs")
        echo "  fold $f $tag -> $jid (val $vlo-$vhi)"
    done
done

echo
echo "Collect with:  grep -H 'training complete' $RUNDIR/train_cv*.log"
