#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Stage the AMIP daily-average stores Stampede3 -> Delta (2026-08-18).
#
# Written when Polaris went down mid-verification and Delta turned out to hold
# NONE of the daily-average AMIP data: its `amip` store is 1981-only, full-res,
# and predates the contract (`sea_surface_temperature` rather than
# `..._monthly_interp`, `specific_total_water` rather than `specific_humidity`).
# Only Stampede3 and Polaris have amip_dailyavg_coarse + amip_dailyavg_boundary.
#
# Ships the STATE and BOUNDARY stores together, always for the same year set. The
# loader pairs them by file name and raises on a state year with no boundary year,
# because boundary reads are indexed by day-of-year — any store answers a read, so
# a missing year would silently be served another year's SST and ice.
#
# Sizes, measured through Globus rather than taken on faith (1979 coarse store):
# one 2-D field-year is 183 chunks + zarr.json = 18.36 MiB stored (23.63 MiB raw,
# so ~1.29x blosc), and the store holds 154 2-D-equivalents per year (24 2-D
# fields + 5 upper-air vars x 26 levels) => 2.76 GiB = 2.97 GB/yr, against the
# registry's recorded 3.0 GB/yr. ~28k files/yr. The boundary store adds roughly
# 0.44 GB/yr. So: ~11 GB for a 3-year smoke subset, ~136 GB for 1979-2015.
#
# Per-file Globus is the slow path for these tiny-chunk stores (tar-bundling is
# ~5x faster — hpc/scripts/replicate_tar.sh) but tar needs a shell on the SOURCE,
# and Stampede3's is behind TACC MFA. At ~28k files/yr per-file is minutes per
# year, which is fine unattended; prefer replicate_tar.sh for the full archive.
#
# PREREQUISITE (interactive, once): Delta's collection needs a consent grant, or
# every call dies with "requires you to grant consent":
#
#   globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/7e936164-de58-4e3d-85da-21aa23c07169/data_access]'
#
# Delta is high-assurance, so also refresh with `globus session update access-ci.org`
# if a call reports the session too old; Stampede3 pairs with `uchicago.edu`.
#
# Usage:
#   hpc/scripts/stage_amip_dailyavg_to_delta.sh 1979 1981          # print the batch
#   hpc/scripts/stage_amip_dailyavg_to_delta.sh 1979 1981 --go     # submit it
#   hpc/scripts/stage_amip_dailyavg_to_delta.sh 1979 2015 --go     # the full range

set -euo pipefail

Y0=${1:?first year (inclusive)}
Y1=${2:?last year (inclusive)}
GO=${3:-}

# Endpoint IDs and roots come from hpc/data_registry.yaml's `clusters:` block.
SRC_EP=1e9ddd41-fe4b-406f-95ff-f3d79f9cb523          # TACC Stampede3
DST_EP=7e936164-de58-4e3d-85da-21aa23c07169          # NCSA Delta
SRC_ROOT=/scratch/09979/awikner/physicsnemo-zarr
# NOT /work/hdd: that allocation is over its soft quota (20.79T of 19.53T, 21.48T
# hard). /work/nvme/bdiu had 2.6T of headroom. Override with DST_ROOT= if the
# hdd allocation has been trimmed and you want the registry's canonical root.
DST_ROOT=${DST_ROOT:-/work/nvme/bdiu/awikner/physicsnemo-zarr}

batch() {
    for store in amip_dailyavg_coarse amip_dailyavg_boundary; do
        for ((y = Y0; y <= Y1; y++)); do
            echo "--recursive $store/$y.zarr $store/$y.zarr"
        done
        # Per-channel stats live at each store root and are resolution-independent.
        for f in normalize_mean_dailyavg.nc normalize_std_dailyavg.nc; do
            echo "$store/$f $store/$f"
        done
    done
}

NYEARS=$((Y1 - Y0 + 1))
echo "# $NYEARS year(s) $Y0-$Y1, both stores"
echo "# ~$(python3 -c "print(f'{$NYEARS * 3.41:.0f}')") GB, ~$(python3 -c "print(f'{$NYEARS * 28300:,d}')") files (coarse ~2.97 GB/yr measured + boundary ~0.44 GB/yr)"
echo "# $SRC_EP:$SRC_ROOT  ->  $DST_EP:$DST_ROOT"
batch

if [[ "$GO" != "--go" ]]; then
    echo "# (dry run — re-run with --go as the third argument to submit)" >&2
    exit 0
fi

# --sync-level checksum so a re-run after an interrupted transfer resumes rather
# than re-sending, and so a partially written year is repaired rather than kept.
batch | globus transfer "$SRC_EP:$SRC_ROOT" "$DST_EP:$DST_ROOT" \
    --batch - \
    --sync-level checksum \
    --preserve-mtime \
    --label "amip dailyavg $Y0-$Y1 S3->Delta"
echo
echo "Track with:  globus task list  /  globus task wait <task-id>"
echo "Then verify: AI_ROSSBY_DATA=$DST_ROOT python tools/data/registry.py check"
