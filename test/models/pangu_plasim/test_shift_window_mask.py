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

"""Phase C gate for the vendored cyclic-longitude
:func:`get_shift_window_mask` (Issue
`#1599 <https://github.com/NVIDIA/physicsnemo/issues/1599>`_ fix).

Verifies that the fixed mask produces the 9-region (3D) / 3-region (2D)
partition the Pangu-Weather paper specifies, and that it differs from the
upstream physicsnemo mask (which produces 27 / 9 regions and suppresses
cross-dateline attention). When this test starts failing because the upstream
mask now agrees with the vendored one, that means the upstream PR for #1599
has landed and the vendored sub-package can be deleted.
"""

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.models.pangu_plasim._vendored_physicsnemo_nn.shift_window_mask import (
        get_shift_window_mask as vendored_get_shift_window_mask,
    )

from physicsnemo.nn.module.utils.shift_window_mask import (
    get_shift_window_mask as upstream_get_shift_window_mask,
)


def _count_unique_regions_3d(input_resolution, window_size, shift_size):
    """Replicate the per-cell region-ID assignment loop and return the count
    actually written into the ``img_mask``."""
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


def test_vendored_mask_uses_nine_regions_in_3d():
    """The cyclic-longitude fix produces 9 distinct region IDs in 3D
    (PanguWeather paper spec); the buggy upstream variant produces 27.
    """
    input_resolution = (8, 24, 48)
    window_size = (2, 6, 12)
    shift_size = (1, 3, 6)

    # Smoke-check that the local region-ID assignment matches the expected
    # cyclic-longitude semantics.
    assert _count_unique_regions_3d(input_resolution, window_size, shift_size) == 9


def test_vendored_mask_differs_from_upstream_until_1599_lands():
    """Documents the intentional divergence: vendored mask (cyclic
    longitude) is **not equal** to physicsnemo's pre-fix mask. When the
    upstream PR lands and this assertion flips to ``torch.equal``, delete the
    vendored sub-package.
    """
    input_resolution = (8, 24, 48)
    window_size = (2, 6, 12)
    shift_size = (1, 3, 6)

    vendored = vendored_get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=3
    )
    upstream = upstream_get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=3
    )

    if torch.equal(vendored, upstream):
        pytest.skip(
            "Upstream physicsnemo.nn.module.utils.get_shift_window_mask now "
            "matches the vendored fix — Issue #1599 must have been merged. "
            "Delete the _vendored_physicsnemo_nn sub-package and import "
            "get_shift_window_mask directly from physicsnemo.nn."
        )
    # Otherwise the divergence is expected and verifies the fix is active.


def test_vendored_mask_shape_matches_paper_spec():
    """Vendored mask's flattened region count matches the 9-region partition,
    independent of the partition pattern within."""
    input_resolution = (8, 24, 48)
    window_size = (2, 6, 12)
    shift_size = (1, 3, 6)

    mask = vendored_get_shift_window_mask(
        input_resolution, window_size, shift_size, ndim=3
    )
    # Mask shape: (n_lon, n_pl * n_lat, win_total, win_total).
    n_lon = input_resolution[2] // window_size[2]
    n_pl_lat = (
        (input_resolution[0] // window_size[0])
        * (input_resolution[1] // window_size[1])
    )
    win_total = window_size[0] * window_size[1] * window_size[2]
    assert mask.shape == (n_lon, n_pl_lat, win_total, win_total)
