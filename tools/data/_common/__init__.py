# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by the per-dataset conversion CLIs under tools/data/.

Most ai-rossby data-conversion scripts (PLASIM, ERA5, E3SM) share the same
shape of work: read a PanguWeather-style mean/std NetCDF (Z = pressure, Z_2 =
sigma), read a CDO climatology NetCDF (plev / lev), walk a directory of
per-channel bias .npy files with level-encoded filenames, and emit Zarr stores
with the unified ai-rossby schema (`sigma_level`, `pressure_level`, `stat`,
`dayofyear`, `hour_of_day`). This module factors out the bits that don't change
per dataset.
"""
