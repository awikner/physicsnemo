#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AMIP 1981 H5 → Zarr conversion (Phase 8b test). After three buffered-mode
# failures (NODE_FAIL → TIMEOUT → OOM → TIMEOUT — see logs from jobs
# 19574177 / 19574650 / 19629473 / 19640283), the converter was refactored
# to write the Zarr in a streaming fashion (tools/data/amip/amip_h5_to_zarr.py).
# The streaming write keeps peak memory bounded by --write-batch instead of
# the full year, which removes the GC slowdown that was killing throughput
# past file ~750.
#
# Submission strategy: split the year into two halves so each half fits the
# cpu-interactive partition's 1h wall cap. Both jobs run in parallel; the
# resulting sibling Zarrs are concatenated by a follow-up xarray step.
#
# Submit both halves with:
#   sbatch --export=ALL,SAMPLE_LO=0,SAMPLE_HI=730,OUT_SUFFIX=_a \
#          hpc/scripts/convert_amip_1981_test.sbatch
#   sbatch --export=ALL,SAMPLE_LO=730,SAMPLE_HI=1460,OUT_SUFFIX=_b \
#          hpc/scripts/convert_amip_1981_test.sbatch

#SBATCH --partition=cpu-interactive
#SBATCH --account=bdiu-delta-cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --job-name=pn-conv-amip-1981
#SBATCH --output=hpc/scripts/logs/convert-amip-1981-%j.out

set -euo pipefail
cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

# Defaults (full year) when invoked without env-var overrides.
SAMPLE_LO=${SAMPLE_LO:-0}
SAMPLE_HI=${SAMPLE_HI:-1460}
OUT_SUFFIX=${OUT_SUFFIX:-}

OUT_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981${OUT_SUFFIX}.zarr
mkdir -p "$(dirname "$OUT_PATH")"

echo "[$(date)] converting AMIP year 1981 files [${SAMPLE_LO}, ${SAMPLE_HI}) → $OUT_PATH"
python tools/data/amip/amip_h5_to_zarr.py \
    --config tools/data/amip/configs/amip_default.yaml \
    --year 1981 \
    --sample-range "$SAMPLE_LO" "$SAMPLE_HI" \
    --output "$OUT_PATH" \
    --time-chunk 1 \
    --write-batch 50

echo
echo "[$(date)] done. Output:"
ls -la "$OUT_PATH" | head -5
echo
echo "[$(date)] Zarr schema sanity:"
python -c "
import xarray as xr
ds = xr.open_zarr('$OUT_PATH')
print('  dims:', dict(ds.sizes))
print('  pressure_level:', ds.pressure_level.values.tolist())
print('  time[0]:', ds.time.values[0])
print('  time[-1]:', ds.time.values[-1])
print('  total data_vars:', len(ds.data_vars))
print('  attrs.surface_variables count:', len(ds.attrs.get('surface_variables', [])))
print('  attrs.pressure_upper_air_variables count:', len(ds.attrs.get('pressure_upper_air_variables', [])))
print('  vertical_velocity in data_vars:', 'vertical_velocity' in ds.data_vars)
print('  vertical_velocity in role list:', 'vertical_velocity' in ds.attrs.get('pressure_upper_air_variables', []))
"
