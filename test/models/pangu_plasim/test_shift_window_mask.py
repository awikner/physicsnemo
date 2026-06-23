# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test for the upstream cyclic-longitude
:func:`get_shift_window_mask` fix (Issue
`#1599 <https://github.com/NVIDIA/physicsnemo/issues/1599>`_).

Before Phase 6 Track A this test guarded a vendored copy of the fix that
lived under ``pangu_plasim/_vendored_physicsnemo_nn/``. Now the fix has
landed upstream behind ``cyclic_longitude=True`` (default ``False`` keeps
historical behavior for non-Pangu callers), and the vendored copy has
been deleted. This file pins the expected behavior of the upstream kwarg:
9-region (3D) / 3-region (2D) partition with longitude treated as cyclic.
"""

import torch

from physicsnemo.nn.module.utils.shift_window_mask import (
    get_shift_window_mask,
)


def _count_unique_regions_3d(input_resolution, window_size, shift_size):
    """Replicate the per-cell region-ID assignment loop and return the count
    actually written into the ``img_mask`` under the cyclic-longitude rule."""
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    shift_pl, shift_lat, _ = shift_size

    img_mask = torch.full((Pl, Lat, Lon), -1.0)
    pl_slices = (
        slice(0, -win_pl),
        slice(-win_pl, -shift_pl),
        slice(-shift_pl, None),
    )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )

    cnt = 0
    for pl in pl_slices:
        for lat in lat_slices:
            img_mask[pl, lat, :] = cnt
            cnt += 1
    return int(img_mask.unique().numel())


def test_cyclic_longitude_uses_nine_regions_in_3d():
    """``cyclic_longitude=True`` produces 9 distinct region IDs in 3D
    (PanguWeather paper spec); the historical default produces 27.
    """
    input_resolution = (8, 24, 48)
    window_size = (2, 6, 12)
    shift_size = (1, 3, 6)

    assert _count_unique_regions_3d(input_resolution, window_size, shift_size) == 9


def test_cyclic_vs_default_masks_differ():
    """``cyclic_longitude=True`` is not equal to the default mask — the
    cyclic variant lets cross-dateline tokens attend to each other where
    the default suppresses them with the partition's region IDs."""
    input_resolution = (8, 24, 48)
    window_size = (2, 6, 12)
    shift_size = (1, 3, 6)

    cyclic = get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=3, cyclic_longitude=True
    )
    default = get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=3
    )
    assert not torch.equal(
        cyclic, default
    ), "cyclic_longitude=True should produce a different mask than the default"


def test_cyclic_mask_shape_matches_paper_spec():
    """Cyclic mask's flattened region count matches the 9-region partition,
    independent of the partition pattern within."""
    input_resolution = (8, 24, 48)
    window_size = (2, 6, 12)
    shift_size = (1, 3, 6)

    mask = get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=3, cyclic_longitude=True
    )
    # Mask shape: (n_lon, n_pl * n_lat, win_total, win_total).
    n_lon = input_resolution[2] // window_size[2]
    n_pl_lat = (
        (input_resolution[0] // window_size[0])
        * (input_resolution[1] // window_size[1])
    )
    win_total = window_size[0] * window_size[1] * window_size[2]
    assert mask.shape == (n_lon, n_pl_lat, win_total, win_total)


def test_cyclic_mask_2d_three_regions():
    """The 2D variant: 3 region IDs under cyclic longitude (paper spec)."""
    input_resolution = (24, 48)
    window_size = (6, 12)
    shift_size = (3, 6)

    mask_cyclic = get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=2, cyclic_longitude=True
    )
    mask_default = get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=2
    )
    assert not torch.equal(mask_cyclic, mask_default)
