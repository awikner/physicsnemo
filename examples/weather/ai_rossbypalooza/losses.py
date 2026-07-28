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

from typing import Sequence

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
) -> torch.Tensor:
    """(H, W) float32 weights: box mask x cos(lat), zero outside."""
    mask = region_mask(lat, lon, box)
    w = mask.to(torch.float64)
    if lat_weighted:
        cos = torch.cos(
            torch.deg2rad(torch.as_tensor(np.asarray(lat, dtype=np.float64)))
        ).clamp(min=0.0)
        w = w * cos[:, None]
    if float(w.sum()) <= 0:
        raise ValueError("region weights sum to zero")
    return w.to(torch.float32)


def _squeeze_channel(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1) if x.ndim == 4 and x.shape[1] == 1 else x


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
        lat_weighted: bool = True,
    ) -> None:
        super().__init__()
        if space not in ("normalized", "physical"):
            raise ValueError(f"space must be normalized|physical, got {space!r}")
        self.space = space
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.register_buffer(
            "weights", region_weights(lat, lon, box, lat_weighted=lat_weighted)
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
            pred = pred * self.precip_std + self.precip_mean
            target = t_mm
        else:
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
        epsilon_mm: float = 0.1,
        lat_weighted: bool = True,
    ) -> None:
        super().__init__()
        if epsilon_mm <= 0:
            raise ValueError(f"epsilon_mm must be positive, got {epsilon_mm}")
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.epsilon_mm = float(epsilon_mm)
        self.register_buffer(
            "weights", region_weights(lat, lon, box, lat_weighted=lat_weighted)
        )

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_mm: torch.Tensor,
    ) -> torch.Tensor:
        pred_mm = (
            _squeeze_channel(pred_norm) * self.precip_std + self.precip_mean
        ).clamp(min=0.0)
        t_mm = _squeeze_channel(target_mm)
        finite = torch.isfinite(t_mm)
        lp = torch.log1p(pred_mm / self.epsilon_mm)
        lt = torch.log1p(torch.nan_to_num(t_mm).clamp(min=0.0) / self.epsilon_mm)
        err = (lp - lt) ** 2
        return _weighted_regional_mean(err, self.weights, finite)


def build_loss(cfg_loss, *, lat, lon, box, precip_mean, precip_std) -> nn.Module:
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
            lat_weighted=lat_weighted,
        )
    if name == "regional_log_mse":
        return RegionalPrecipLogMSE(
            lat,
            lon,
            box,
            precip_mean=precip_mean,
            precip_std=precip_std,
            epsilon_mm=float(cfg_loss.get("epsilon_mm", 0.1)),
            lat_weighted=lat_weighted,
        )
    raise ValueError(f"unknown loss name '{name}'")
