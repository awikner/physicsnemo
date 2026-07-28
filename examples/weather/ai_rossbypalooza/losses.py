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

"""Regional precipitation losses for the MoWE gate.

The mixture is computed globally; the loss is evaluated only inside a
lat/lon box (the monsoon domain, cf. ``physmetrics/plot_tcwv_bias.py``'s
``DEFAULT_REGION = (5, 35, 60, 100)``), cos-lat weighted, with per-sample
NaN masking on the target (IMERG can have missing cells).

Both losses take ``(pred_norm, target_norm, target_mm)``:

* :class:`RegionalPrecipMSE` — weighted MSE in normalized space by default;
  ``space="physical"`` denormalizes with the shared IMERG stats first
  (linear, differentiable — the two differ only by a constant ``std^2``).
* :class:`RegionalPrecipLogMSE` — always physical:
  ``MSE(log1p(clamp(pred_mm, 0) / eps), log1p(target_mm / eps))``, which
  compresses the heavy tail so moderate/heavy intensity errors are not
  drowned out by the largest events.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


def region_mask(
    lat: np.ndarray, lon: np.ndarray, box: Sequence[float]
) -> torch.Tensor:
    """(H, W) bool mask of the box ``(lat_min, lat_max, lon_min, lon_max)``.

    Longitude supports wraparound boxes (``lon_min > lon_max``); the monsoon
    box does not wrap but tests do.
    """
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in box)
    la = torch.as_tensor(np.asarray(lat, dtype=np.float64))
    lo = torch.as_tensor(np.asarray(lon, dtype=np.float64))
    lat_ok = (la >= lat_min) & (la <= lat_max)
    if lon_min <= lon_max:
        lon_ok = (lo >= lon_min) & (lo <= lon_max)
    else:
        lon_ok = (lo >= lon_min) | (lo <= lon_max)
    mask = lat_ok[:, None] & lon_ok[None, :]
    if not mask.any():
        raise ValueError(f"region box {tuple(box)} selects no gridpoints")
    return mask


def region_weights(
    lat: np.ndarray,
    lon: np.ndarray,
    box: Sequence[float],
    *,
    lat_weighted: bool = True,
    extra_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """(H, W) float32 weights: box mask [x extra_mask] x cos(lat).

    ``extra_mask`` (bool, (H, W)) intersects the box — e.g. the IMD
    data-availability mask so training/metrics only see gridpoints with
    IMD gauge coverage.
    """
    mask = region_mask(lat, lon, box)
    if extra_mask is not None:
        mask = mask & extra_mask.to(torch.bool)
    w = mask.to(torch.float64)
    if lat_weighted:
        cos = torch.cos(
            torch.deg2rad(torch.as_tensor(np.asarray(lat, dtype=np.float64)))
        ).clamp(min=0.0)
        w = w * cos[:, None]
    if float(w.sum()) <= 0:
        raise ValueError("region weights sum to zero")
    return w.to(torch.float32)


def imd_valid_mask(
    imd_store: str,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    var: str = "total_precipitation_24hr",
    min_finite_frac: float = 0.99,
    coord_tol: float = 1e-3,
) -> torch.Tensor:
    """(H, W) bool mask of gridpoints with IMD gauge coverage.

    The IMD analysis lives on a native 1-degree India grid (lat 6.5..38.5,
    lon 66.5..100.5 — cell centers identical to the global IMERG/ERA5 1-deg
    grid) with ~69% NaN over ocean / station-free cells. A gridpoint is
    "valid" when its finite fraction over the store's records is at least
    ``min_finite_frac`` (the NaN pattern is a static coverage mask).
    Coordinates are matched by value, so grid orientation is irrelevant.
    """
    import xarray as xr

    with xr.open_zarr(imd_store, consolidated=True) as ds:
        vals = ds[var].values  # (T, h, w) native India grid
        imd_lat = ds["lat"].values.astype("float64")
        imd_lon = ds["lon"].values.astype("float64")
    frac = np.isfinite(vals).mean(axis=0)
    valid_native = frac >= float(min_finite_frac)

    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    mask = np.zeros((lat.size, lon.size), dtype=bool)
    lat_idx = np.full(imd_lat.size, -1)
    lon_idx = np.full(imd_lon.size, -1)
    for i, v in enumerate(imd_lat):
        j = np.argmin(np.abs(lat - v))
        if abs(lat[j] - v) <= coord_tol:
            lat_idx[i] = j
    for i, v in enumerate(imd_lon):
        j = np.argmin(np.abs(lon - v))
        if abs(lon[j] - v) <= coord_tol:
            lon_idx[i] = j
    if (lat_idx < 0).all() or (lon_idx < 0).all():
        raise ValueError(
            f"IMD grid ({imd_lat[0]}..{imd_lat[-1]}) does not align with the "
            f"target grid — expected shared 1-degree cell centers"
        )
    for i in range(imd_lat.size):
        if lat_idx[i] < 0:
            continue
        for k in range(imd_lon.size):
            if lon_idx[k] >= 0 and valid_native[i, k]:
                mask[lat_idx[i], lon_idx[k]] = True
    if not mask.any():
        raise ValueError("IMD validity mask is empty")
    return torch.from_numpy(mask)


def _squeeze_channel(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1) if x.ndim == 4 and x.shape[1] == 1 else x


def denormalize_precip(
    x_norm: torch.Tensor,
    *,
    mean: float,
    std: float,
    transform=None,
) -> torch.Tensor:
    """Normalized precip -> physical mm/day.

    ``transform`` is the dataset's optional ``LogPrecipTransform`` (model
    v1): stats then live in log space and the inverse maps back to mm/day
    (clamped at 0). ``None`` = plain linear stats in mm/day.
    """
    x = x_norm * std + mean
    if transform is None:
        return x
    return transform.inverse(x)


def _weighted_regional_mean(
    err: torch.Tensor, weights: torch.Tensor, finite: torch.Tensor
) -> torch.Tensor:
    """Per-sample weighted mean over the region, NaN cells excluded."""
    w = weights.unsqueeze(0) * finite.to(err.dtype)  # (B, H, W)
    denom = w.sum(dim=(-2, -1)).clamp(min=1e-12)
    return ((err * w).sum(dim=(-2, -1)) / denom).mean()


class RegionalPrecipMSE(nn.Module):
    """Cos-lat-weighted MSE over the region box. See the module docstring."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        box: Sequence[float],
        *,
        space: str = "normalized",
        precip_mean: float = 0.0,
        precip_std: float = 1.0,
        precip_transform=None,
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        if space not in ("normalized", "physical"):
            raise ValueError(f"space must be normalized|physical, got {space!r}")
        self.space = space
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.precip_transform = precip_transform
        self.register_buffer(
            "weights",
            region_weights(
                lat, lon, box, lat_weighted=lat_weighted, extra_mask=extra_mask
            ),
        )

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_mm: torch.Tensor,
    ) -> torch.Tensor:
        pred = _squeeze_channel(pred_norm)
        t_norm = _squeeze_channel(target_norm)
        t_mm = _squeeze_channel(target_mm)
        if self.space == "physical":
            pred = denormalize_precip(
                pred,
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
            target = t_mm
        else:
            # With the model-v1 log transform, "normalized" space is the
            # standardized log(eps + P) space.
            target = t_norm
        finite = torch.isfinite(target)
        err = (pred - torch.nan_to_num(target)) ** 2
        return _weighted_regional_mean(err, self.weights, finite)


class RegionalPrecipLogMSE(nn.Module):
    """Log-transformed regional MSE (always in physical mm/day units)."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        box: Sequence[float],
        *,
        precip_mean: float,
        precip_std: float,
        precip_transform=None,
        epsilon_mm: float = 0.1,
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        if epsilon_mm <= 0:
            raise ValueError(f"epsilon_mm must be positive, got {epsilon_mm}")
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.precip_transform = precip_transform
        self.epsilon_mm = float(epsilon_mm)
        self.register_buffer(
            "weights",
            region_weights(
                lat, lon, box, lat_weighted=lat_weighted, extra_mask=extra_mask
            ),
        )

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_mm: torch.Tensor,
    ) -> torch.Tensor:
        pred_mm = denormalize_precip(
            _squeeze_channel(pred_norm),
            mean=self.precip_mean,
            std=self.precip_std,
            transform=self.precip_transform,
        ).clamp(min=0.0)
        t_mm = _squeeze_channel(target_mm)
        finite = torch.isfinite(t_mm)
        lp = torch.log1p(pred_mm / self.epsilon_mm)
        lt = torch.log1p(torch.nan_to_num(t_mm).clamp(min=0.0) / self.epsilon_mm)
        err = (lp - lt) ** 2
        return _weighted_regional_mean(err, self.weights, finite)


def build_loss(
    cfg_loss,
    *,
    lat,
    lon,
    box,
    precip_mean,
    precip_std,
    precip_transform=None,
    extra_mask=None,
) -> nn.Module:
    """Dispatcher on ``cfg.loss.name`` (ai_rossby ``build_loss`` convention)."""
    name = str(cfg_loss.get("name", "regional_mse"))
    lat_weighted = bool(cfg_loss.get("lat_weighted", True))
    if name == "regional_mse":
        return RegionalPrecipMSE(
            lat,
            lon,
            box,
            space=str(cfg_loss.get("space", "normalized")),
            precip_mean=precip_mean,
            precip_std=precip_std,
            precip_transform=precip_transform,
            lat_weighted=lat_weighted,
            extra_mask=extra_mask,
        )
    if name == "regional_log_mse":
        return RegionalPrecipLogMSE(
            lat,
            lon,
            box,
            precip_mean=precip_mean,
            precip_std=precip_std,
            precip_transform=precip_transform,
            epsilon_mm=float(cfg_loss.get("epsilon_mm", 0.1)),
            lat_weighted=lat_weighted,
            extra_mask=extra_mask,
        )
    raise ValueError(f"unknown loss name '{name}'")
