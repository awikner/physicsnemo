#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Week-2 per-gridpoint skill panels (six sources + gate deltas) on Derecho.
# CPU only, develop queue: no checkpoint, no GPU, one I/O-bound pass over the
# validation split. See README.md in this folder.
#
#   mkdir -p /glade/derecho/scratch/$USER/mowe_runs   # PBS -o needs it to exist
#   qsub tools/acc_panels/run_week2_acc_panels_derecho.sh
#   qsub -v metric=rmse,matched=--no-matched ...      # any cfg key; no commas
#
#PBS -A URIC0009
#PBS -q develop
#PBS -l select=1:ncpus=16:mem=120GB
#PBS -l walltime=01:00:00
#PBS -N week2_acc_panels
#PBS -j oe
#PBS -o /glade/derecho/scratch/dboscu/mowe_runs/week2_acc_panels.log

set -uo pipefail
module load ncarenv 2>/dev/null || true

repo=${repo:-/glade/u/home/dboscu/ai-rossbypalooza-mowe/physicsnemo}
scratch=${scratch:-/glade/derecho/scratch/dboscu}
recipe=$repo/examples/weather/ai_rossbypalooza

declare -A cfg=(
    [rundir]=$scratch/mowe_runs
    [forecast]=/glade/derecho/scratch/syback/mowe_forecasts/cv5_physvar.zarr
    [dataset_config]=$recipe/conf/dataset/hindcast_derecho.yaml
    [climatology]=$scratch/physicsnemo-zarr/normalization/imerg_seeps_climatology_daily.zarr
    [cartopy_data]=$scratch/cartopy_data
    [metric]=acc
    [matched]=--matched # the expert sources do not fully overlap with each other
    [batch_size]=8
    [num_workers]=8
    [out]=$recipe/tools/acc_panels/plots/week2_${cfg[metric]}_panels.png
)
for key in "${!cfg[@]}"; do
    [[ -v $key ]] && cfg[$key]=${!key}
done
: "${cfg[out]:=${cfg[rundir]}/week2_${cfg[metric]}_panels.png}"

# Absolute, so activation no longer depends on the working directory.
venv=$repo/.venv/bin/activate
[[ -f $venv ]] || { echo "FATAL: no venv at $venv"; exit 2; }
source "$venv"

for key in forecast dataset_config climatology; do
    [[ -e ${cfg[$key]} ]] || { echo "FATAL: $key missing: ${cfg[$key]}"; exit 2; }
done
mkdir -p "${cfg[rundir]}"

echo "[$(date)] metric=${cfg[metric]} ${cfg[matched]} -> ${cfg[out]}"
python "$recipe/tools/acc_panels/plot_week2_acc_panels.py" \
    --forecast       "${cfg[forecast]}" \
    --dataset-config "${cfg[dataset_config]}" \
    --climatology    "${cfg[climatology]}" \
    --cartopy-data   "${cfg[cartopy_data]}" \
    --metric         "${cfg[metric]}" \
    --batch-size     "${cfg[batch_size]}" \
    --num-workers    "${cfg[num_workers]}" \
    --out            "${cfg[out]}" \
    "${cfg[matched]}"

rc=$?
echo "[$(date)] week2 panels exit $rc"
exit $rc
