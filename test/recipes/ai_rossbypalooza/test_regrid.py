# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for separable 1-D conservative regridding (datapipes/regrid.py)."""

from __future__ import annotations

import numpy as np
import pytest

from datapipes.regrid import Regridder, grids_equal


def _era5_quarter_deg():
    """Pole-inclusive 0.25-degree grid, lat N->S (the DSI store layout)."""
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(0.0, 360.0, 0.25)
    return lat, lon


def _era5_one_deg():
    """IMERG/ERA5 1-degree grid: centers 89.5..-89.5 N->S, lon 0..359."""
    lat = np.linspace(89.5, -89.5, 180)
    lon = np.arange(0.0, 360.0, 1.0)
    return lat, lon


@pytest.fixture(scope="module")
def regridder():
    src_lat, src_lon = _era5_quarter_deg()
    dst_lat, dst_lon = _era5_one_deg()
    return Regridder(src_lat, src_lon, dst_lat, dst_lon)


def test_weight_normalization(regridder):
    np.testing.assert_allclose(regridder.a_lat.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(regridder.a_lon.sum(axis=0), 1.0, atol=1e-12)


def test_constant_field_preserved(regridder):
    x = np.full((721, 1440), 3.25, dtype=np.float32)
    out = regridder(x)
    assert out.shape == (180, 360)
    np.testing.assert_allclose(out, 3.25, rtol=1e-6)


def test_pole_rows_zero_weight(regridder):
    # cos(+/-90) = 0: the pole-inclusive source rows contribute nothing.
    assert regridder.a_lat[0, 0] == 0.0
    assert regridder.a_lat[-1, -1] == 0.0


def test_lat_only_field_matches_brute_force(regridder):
    """cos-weighted band average of a lat-only field, checked by quadrature."""
    src_lat, _ = _era5_quarter_deg()
    field = np.cos(np.deg2rad(src_lat)) ** 2  # smooth, lat-only
    x = np.broadcast_to(field[:, None], (721, 1440)).copy()
    out = regridder(x)
    # Brute force: for target row i (cell [89-i, 90-i]), integrate
    # field*cos over contributing source bands.
    for i in (0, 45, 90, 179):
        hi, lo = 90.0 - i, 89.0 - i
        lats = src_lat
        band_hi = np.minimum(np.clip(lats + 0.125, -90, 90), hi)
        band_lo = np.maximum(np.clip(lats - 0.125, -90, 90), lo)
        overlap = np.clip(band_hi - band_lo, 0.0, None)
        w = overlap * np.cos(np.deg2rad(lats))
        expected = (w * field).sum() / w.sum()
        np.testing.assert_allclose(out[i, 0], expected, rtol=1e-10)


def test_lon_wraparound(regridder):
    """Target cell centered at lon 0 pulls source columns from both ends.

    Target cell [-0.5, 0.5); source cells are 0.25 deg wide with edges at
    center +/- 0.125, so the analytic overlap weights are: col 0 -> 0.25,
    col 1 (0.25 deg) -> 0.25, col 2 (0.5 deg) -> 0.125, col 1439
    (359.75 == -0.25) -> 0.25, col 1438 (359.5 == -0.5) -> 0.125.
    """
    col0 = regridder.a_lon[:, 0]
    expected = np.zeros(1440)
    expected[[0, 1, 2, 1438, 1439]] = [0.25, 0.25, 0.125, 0.125, 0.25]
    np.testing.assert_allclose(col0, expected, atol=1e-12)


def test_area_mean_conserved(regridder):
    """Global cos-lat mean is preserved for an arbitrary smooth field."""
    src_lat, src_lon = _era5_quarter_deg()
    dst_lat, dst_lon = _era5_one_deg()
    rng = np.random.default_rng(0)
    # Smooth random field: low-order spherical-harmonic-ish combination.
    lat2d = np.deg2rad(src_lat)[:, None]
    lon2d = np.deg2rad(src_lon)[None, :]
    x = (
        1.0
        + 0.5 * np.sin(lat2d) * np.cos(2 * lon2d)
        + 0.25 * np.cos(3 * lat2d) * np.sin(lon2d)
        + 0.05 * rng.standard_normal((721, 1440))
    )
    out = regridder(x)

    def area_mean(field, lat):
        w = np.cos(np.deg2rad(lat))[:, None]
        return (field * w).sum() / (w * np.ones_like(field)).sum()

    src_mean = area_mean(x, src_lat)
    dst_mean = area_mean(out, dst_lat)
    # 1-D conservative pooling: means agree closely (not bit-exact because
    # target rows re-normalize by their own coverage).
    np.testing.assert_allclose(dst_mean, src_mean, rtol=1e-4)


def test_tiny_grid_and_shape_validation():
    src_lat = np.linspace(3.0, -3.0, 9)  # 0.75-deg-ish tiny grid
    src_lon = np.arange(0.0, 360.0, 45.0)
    dst_lat = np.array([1.5, -1.5])
    dst_lon = np.arange(0.0, 360.0, 90.0)
    r = Regridder(src_lat, src_lon, dst_lat, dst_lon)
    x = np.ones((4, 9, 8), dtype=np.float32)  # leading (channel) dim broadcasts
    out = r(x)
    assert out.shape == (4, 2, 4)
    np.testing.assert_allclose(out, 1.0, rtol=1e-6)
    with pytest.raises(ValueError, match="does not match source grid"):
        r(np.ones((5, 8)))


def test_nan_propagates(regridder):
    x = np.ones((721, 1440), dtype=np.float32)
    x[300, 700] = np.nan
    out = regridder(x)
    assert np.isnan(out).any()
    assert np.isnan(out).sum() < 10  # localized, not global


def test_grids_equal():
    a = np.linspace(89.5, -89.5, 180)
    assert grids_equal(a, a.copy())
    assert not grids_equal(a, a + 0.01)
    assert not grids_equal(a, np.linspace(90.0, -90.0, 721))
