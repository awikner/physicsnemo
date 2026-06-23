#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 8b test conversion — full year 1981 AMIP H5 → Zarr on the cpu
# partition. Validates the new converter on a full-year payload before
# the multi-year archive job is staged.
#
# Output: /work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
#
# Submit with:
#   sbatch hpc/scripts/convert_amip_1981_test.sbatch

#SBATCH --partition=cpu
#SBATCH --account=bdiu-delta-cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --job-name=pn-conv-amip-1981
#SBATCH --output=hpc/scripts/logs/convert-amip-1981-%j.out

set -euo pipefail
cd /work/nvme/bdiu/awikner/physicsnemo
source .venv/bin/activate

OUT_PATH=/work/hdd/bdiu/awikner/physicsnemo-zarr/amip/1981.zarr
mkdir -p "$(dirname "$OUT_PATH")"

echo "[$(date)] converting AMIP year 1981 ..."
python tools/data/amip/amip_h5_to_zarr.py \
    --config tools/data/amip/configs/amip_default.yaml \
    --year 1981 \
    --output "$OUT_PATH" \
    --time-chunk 1

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
