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

r"""Cyclic-longitude shift-window attention mask.

This is a **patched** copy of
:func:`physicsnemo.nn.module.utils.shift_window_mask.get_shift_window_mask`
fixing `physicsnemo#1599
<https://github.com/NVIDIA/physicsnemo/issues/1599>`_. The bug: upstream
partitions the longitude axis with ``lon_slices``, producing 27 region IDs
(3D) / 9 (2D) and suppressing cross-dateline attention. The Pangu-Weather
paper's text is explicit:

    "Along the longitude dimension, the leftmost and rightmost indices are
    actually close to each other. If half windows appear at both leftmost and
    rightmost positions, they are directly merged into one window."

The fix is to iterate only over ``pl_slices × lat_slices`` (3D) or
``lat_slices`` (2D), leaving the entire longitude axis with a single region
ID so the dateline tokens attend across the discontinuity. The result is the
9-region (3D) / 3-region (2D) mask the original Pangu-Weather pseudocode
produces.

When the upstream PR for #1599 lands, replace imports of this module with
``physicsnemo.nn.module.utils.shift_window_mask`` and delete this file.
"""

import torch

from physicsnemo.nn.module.utils import window_partition


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    r"""Shift-window attention mask with the cyclic-longitude fix from #1599.

    Parameters
    ----------
    input_resolution : tuple of int
        :math:`(Pl, Lat, Lon)` (3D) or :math:`(Lat, Lon)` (2D).
    window_size : tuple of int
        :math:`(W_{pl}, W_{lat}, W_{lon})` (3D) or :math:`(W_{lat}, W_{lon})` (2D).
    shift_size : tuple of int
        :math:`(S_{pl}, S_{lat}, S_{lon})` (3D) or :math:`(S_{lat}, S_{lon})` (2D).
    ndim : int, optional, default=3
        Window dimensionality (3 or 2).

    Returns
    -------
    torch.Tensor
        Attention mask of shape
        :math:`(n_{lon}, n_{pl} \cdot n_{lat}, W_{pl} W_{lat} W_{lon}, W_{pl} W_{lat} W_{lon})`
        (3D) or :math:`(n_{lon}, n_{lat}, W_{lat} W_{lon}, W_{lat} W_{lon})` (2D).
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, _ = shift_size
        img_mask = torch.zeros((1, Pl, Lat, Lon, 1))

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

        # Longitude is cyclic — do not partition it. (#1599 fix.)
        cnt = 0
        for pl in pl_slices:
            for lat in lat_slices:
                img_mask[:, pl, lat, :, :] = cnt
                cnt += 1

        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, _ = shift_size
        img_mask = torch.zeros((1, Lat, Lon, 1))

        lat_slices = (
            slice(0, -win_lat),
            slice(-win_lat, -shift_lat),
            slice(-shift_lat, None),
        )

        # Longitude is cyclic — do not partition it. (#1599 fix.)
        cnt = 0
        for lat in lat_slices:
            img_mask[:, lat, :, :] = cnt
            cnt += 1

        win_total = win_lat * win_lon
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    mask_windows = window_partition(img_mask, window_size, ndim=ndim)
    mask_windows = mask_windows.view(
        mask_windows.shape[0], mask_windows.shape[1], win_total
    )
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask
