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

"""Separable 1-D conservative lat/lon regridding (0.25 deg -> 1 deg).

Used *offline* by ``tools/regrid_dsi_to_1deg.py`` to coarsen the DSI
hindcast stores onto the 1 deg IMERG/ERA5 grid before training; the
training datapipe only ever reads pre-regridded stores.

The scheme is 1-D conservative applied per axis: each source point is
treated as a cell (edges at midpoints between neighbours, clipped to
+/-90 in latitude, periodic in longitude) and each target cell averages
the overlapping source cells weighted by overlap length — times
``cos(lat)`` on the latitude axis so the pooling is area-weighted.
Weight rows are normalized to sum to 1, so constant fields are preserved
exactly. numpy-only on purpose: the offline tool must stay login-node
safe (no physicsnemo/torch import).
"""

from __future__ import annotations

import numpy as np


def _cell_edges(centers: np.ndarray, *, clip: tuple[float, float] | None = None,
                periodic_span: float | None = None) -> np.ndarray:
    """Cell edges from 1-D centers (monotonic, either direction).

    Interior edges are midpoints; outer edges extrapolate half the adjacent
    spacing, then are clipped (latitude) or wrapped by the periodic gap
    (longitude).
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 1 or c.size < 2:
        raise ValueError("need at least 2 grid centers")
    d = np.diff(c)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError("grid centers must be strictly monotonic")
    mid = (c[:-1] + c[1:]) / 2.0
    if periodic_span is not None:
        gap = periodic_span - abs(c[-1] - c[0])
        if gap <= 0:
            raise ValueError("longitude centers span more than the period")
        sign = 1.0 if d[0] > 0 else -1.0
        first = c[0] - sign * gap / 2.0
        last = c[-1] + sign * gap / 2.0
    else:
        first = c[0] - d[0] / 2.0
        last = c[-1] + d[-1] / 2.0
    edges = np.concatenate([[first], mid, [last]])
    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])
    return edges


def _overlap_matrix(src_edges: np.ndarray, dst_edges: np.ndarray,
                    *, periodic_span: float | None = None) -> np.ndarray:
    """(n_dst, n_src) interval-overlap lengths between two edge sets."""
    src_lo = np.minimum(src_edges[:-1], src_edges[1:])
    src_hi = np.maximum(src_edges[:-1], src_edges[1:])
    dst_lo = np.minimum(dst_edges[:-1], dst_edges[1:])
    dst_hi = np.maximum(dst_edges[:-1], dst_edges[1:])
    shifts = (0.0,) if periodic_span is None else (0.0, periodic_span, -periodic_span)
    out = np.zeros((dst_lo.size, src_lo.size))
    for s in shifts:
        lo = np.maximum(dst_lo[:, None], src_lo[None, :] + s)
        hi = np.minimum(dst_hi[:, None], src_hi[None, :] + s)
        out += np.clip(hi - lo, 0.0, None)
    return out


class Regridder:
    """Separable conservative regridder between two regular lat/lon grids.

    Latitude rows of ``a_lat (n_dst_lat, n_src_lat)`` carry
    overlap x cos(src_lat) weights normalized to sum 1 (pole rows of a
    pole-inclusive source get zero weight naturally); longitude columns of
    ``a_lon (n_src_lon, n_dst_lon)`` carry periodic overlap weights
    normalized to sum 1. Application is two matmuls:
    ``out = a_lat @ x @ a_lon`` for ``x (..., H_src, W_src)``.
    """

    def __init__(self, src_lat: np.ndarray, src_lon: np.ndarray,
                 dst_lat: np.ndarray, dst_lon: np.ndarray) -> None:
        self.src_lat = np.asarray(src_lat, dtype=np.float64)
        self.src_lon = np.asarray(src_lon, dtype=np.float64)
        self.dst_lat = np.asarray(dst_lat, dtype=np.float64)
        self.dst_lon = np.asarray(dst_lon, dtype=np.float64)

        a_lat = _overlap_matrix(
            _cell_edges(self.src_lat, clip=(-90.0, 90.0)),
            _cell_edges(self.dst_lat, clip=(-90.0, 90.0)),
        )
        cos_lat = np.cos(np.deg2rad(self.src_lat))
        cos_lat[np.isclose(np.abs(self.src_lat), 90.0)] = 0.0  # exact 0 at poles
        a_lat *= cos_lat[None, :]
        row_sums = a_lat.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("target latitude cell has no source coverage")
        self.a_lat = a_lat / row_sums[:, None]

        a_lon = _overlap_matrix(
            _cell_edges(self.src_lon, periodic_span=360.0),
            _cell_edges(self.dst_lon, periodic_span=360.0),
            periodic_span=360.0,
        )
        col_sums = a_lon.sum(axis=1)
        if np.any(col_sums <= 0):
            raise ValueError("target longitude cell has no source coverage")
        self.a_lon = (a_lon / col_sums[:, None]).T  # (n_src_lon, n_dst_lon)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Regrid ``x (..., H_src, W_src)`` -> ``(..., H_dst, W_dst)``.

        NaNs propagate *locally* by design: a target cell is NaN iff a source
        cell with nonzero weight is NaN. (A plain dense matmul would poison
        the whole output through its zero weights, since ``0 * NaN = NaN``.)
        Regridded DSI stores should not contain NaNs; the offline tool checks
        afterwards.
        """
        x = np.asarray(x)
        if x.shape[-2] != self.src_lat.size or x.shape[-1] != self.src_lon.size:
            raise ValueError(
                f"field shape {x.shape[-2:]} does not match source grid "
                f"({self.src_lat.size}, {self.src_lon.size})"
            )
        x64 = x.astype(np.float64)
        nan_mask = np.isnan(x64)
        if nan_mask.any():
            out = self.a_lat @ np.where(nan_mask, 0.0, x64) @ self.a_lon
            touched = self.a_lat @ nan_mask.astype(np.float64) @ self.a_lon
            out[touched > 0.0] = np.nan
        else:
            out = self.a_lat @ x64 @ self.a_lon
        return out.astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)


def grids_equal(a: np.ndarray, b: np.ndarray, *, tol: float = 1e-4) -> bool:
    """True when two 1-D coordinate arrays describe the same grid."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol, rtol=0.0))
