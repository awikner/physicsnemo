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

    The IMD analysis lives on a native 1-degree India grid with ~69% NaN
    over ocean / station-free cells. A gridpoint is "valid" when its finite
    fraction over the store's records is at least ``min_finite_frac`` (the
    NaN pattern is a static coverage mask). Latitudes share the global
    half-degree cell centers; IMD LONGITUDES are offset by half a cell
    (66.5, 67.5, ... vs the global integer centers), so each valid IMD
    cell marks every overlapping global column (those within half a cell
    width) — the mask dilates by at most one column at region edges.
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
    half = 0.5 + coord_tol  # half a 1-degree cell

    def overlapping(coords: np.ndarray, v: float) -> np.ndarray:
        d = np.abs(coords - v)
        exact = np.nonzero(d <= coord_tol)[0]
        return exact if exact.size else np.nonzero(d <= half)[0]

    mask = np.zeros((lat.size, lon.size), dtype=bool)
    lat_rows = [overlapping(lat, v) for v in imd_lat]
    lon_cols = [overlapping(lon, v) for v in imd_lon]
    if not any(r.size for r in lat_rows) or not any(c.size for c in lon_cols):
        raise ValueError(
            f"IMD grid (lat {imd_lat[0]}..{imd_lat[-1]}, "
            f"lon {imd_lon[0]}..{imd_lon[-1]}) does not overlap the target "
            f"1-degree grid"
        )
    for i, rows in enumerate(lat_rows):
        if not rows.size:
            continue
        for k, cols in enumerate(lon_cols):
            if cols.size and valid_native[i, k]:
                mask[np.ix_(rows, cols)] = True
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


def normalize_precip(
    x_mm: torch.Tensor,
    *,
    mean: float,
    std: float,
    transform=None,
) -> torch.Tensor:
    """Physical mm/day -> normalized precip (inverse of :func:`denormalize_precip`).

    Used to score a baseline that is *defined* in physical space (e.g. the
    equal-weight arithmetic mean) with a loss that operates in normalized
    space, so its loss and its RMSE describe the same forecast.
    """
    x = x_mm if transform is None else transform.forward(x_mm)
    return (x - mean) / std


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
        pred_space: str = "normalized",
        bias_weight: float = 0.0,
        var_weight: float = 0.0,
        scale_mm: float | None = None,
        precip_mean: float = 0.0,
        precip_std: float = 1.0,
        precip_transform=None,
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        if space not in ("normalized", "physical"):
            raise ValueError(f"space must be normalized|physical, got {space!r}")
        if pred_space not in ("normalized", "physical"):
            raise ValueError(
                f"pred_space must be normalized|physical, got {pred_space!r}"
            )
        self.space = space
        # Space the incoming prediction lives in = the space the mixture was
        # formed in (cfg.model.mix_space). With mix_space=physical the
        # mixture is an arithmetic mean in mm/day and this loss transforms it
        # into log space before taking the squared error.
        self.pred_space = pred_space
        # Composite loss: add ``bias_weight * (regional mean error in mm/day)^2``
        # to penalise systematic wet/dry drift directly. A log-space MSE alone
        # elicits the conditional GEOMETRIC mean, which for monsoon rainfall
        # sits ~56% below the arithmetic mean (measured on IMERG July over the
        # IMD region), so the log term has no incentive to be unbiased in
        # mm/day. Units of bias_weight are (mm/day)^-2: at 0.02 a -3.6 mm/day
        # bias costs ~0.26, roughly a quarter of a typical log-MSE value,
        # while a -0.5 mm/day bias costs a negligible 0.005.
        # Note the penalty uses the per-batch regional mean error, so it also
        # lightly penalises error variance (E[m^2] = bias^2 + var/n); with a
        # few thousand weighted gridpoints per batch that term is small.
        if bias_weight < 0:
            raise ValueError(f"bias_weight must be >= 0, got {bias_weight}")
        self.bias_weight = float(bias_weight)
        # Amplitude matching: add var_weight * (sigma_pred/sigma_obs - 1)^2,
        # where sigma is the region-weighted SPATIAL standard deviation in
        # mm/day of each sample. MSE decomposes as
        # bias^2 + (sp - st)^2 + 2*sp*st*(1 - r), so shrinking the forecast
        # toward its own mean removes the decorrelation term: the MSE-optimal
        # amplitude is sp = r * st, and the measured physical-MSE run duly
        # converged to amp 0.39 against ACC 0.34. Nothing else in the
        # objective forbids that hedging, which IS the intensity blurring this
        # project targets, so it needs its own term.
        if var_weight < 0:
            raise ValueError(f"var_weight must be >= 0, got {var_weight}")
        self.var_weight = float(var_weight)
        self.last_amp: float = float("nan")
        # Reference RMSE (mm/day) used to divide a physical-space MSE, e.g.
        # 9.3 puts it near 1.0 like the log-space loss so the tuned lr and
        # grad_clip_norm transfer. Pure loss rescaling: it cannot move the
        # optimum, and AdamW is largely scale-invariant anyway -- this mainly
        # keeps gradient clipping from binding differently.
        if scale_mm is not None and scale_mm <= 0:
            raise ValueError(f"scale_mm must be positive, got {scale_mm}")
        self.scale_mm = None if scale_mm is None else float(scale_mm)
        # Diagnostics from the last forward (detached, for logging).
        self.last_mse: float = float("nan")
        self.last_bias_mm: float = float("nan")
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
        pred: torch.Tensor,
        target_norm: torch.Tensor,
        target_mm: torch.Tensor,
    ) -> torch.Tensor:
        pred = _squeeze_channel(pred)
        t_norm = _squeeze_channel(target_norm)
        t_mm = _squeeze_channel(target_mm)
        if self.pred_space == "physical":
            # Unphysical negative rain is clipped before the log transform.
            pred_mm = pred.clamp(min=0.0)
            if self.space == "physical":
                pred, target = pred_mm, t_mm
            else:
                pred = normalize_precip(
                    pred_mm,
                    mean=self.precip_mean,
                    std=self.precip_std,
                    transform=self.precip_transform,
                )
                target = t_norm
        elif self.space == "physical":
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
        mse = _weighted_regional_mean(err, self.weights, finite)
        if self.space == "physical" and self.scale_mm is not None:
            mse = mse / self.scale_mm**2
        self.last_mse = float(mse.detach())
        if self.bias_weight <= 0 and self.var_weight <= 0:
            self.last_bias_mm = float("nan")
            self.last_amp = float("nan")
            return mse

        # Extra terms live in physical mm/day, whatever space the MSE used.
        if self.pred_space == "physical":
            p_mm = pred_mm
        else:
            p_mm = denormalize_precip(
                _squeeze_channel(pred) if self.space != "physical" else pred,
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
        finite_mm = torch.isfinite(t_mm)
        t_filled = torch.nan_to_num(t_mm)
        w = self.weights.unsqueeze(0) * finite_mm.to(p_mm.dtype)
        total = mse

        if self.bias_weight > 0:
            bias_mm = ((p_mm - t_filled) * w).sum() / w.sum().clamp(min=1e-12)
            total = total + self.bias_weight * bias_mm**2
            self.last_bias_mm = float(bias_mm.detach())
        else:
            self.last_bias_mm = float("nan")

        if self.var_weight > 0:
            wsum = w.sum(dim=(-2, -1)).clamp(min=1e-12)

            def _std(field):
                mu = (field * w).sum(dim=(-2, -1)) / wsum
                var = (
                    w * (field - mu[:, None, None]) ** 2
                ).sum(dim=(-2, -1)) / wsum
                return torch.sqrt(var.clamp(min=1e-12))

            ratio = _std(p_mm) / _std(t_filled).clamp(min=1e-6)
            total = total + self.var_weight * ((ratio - 1.0) ** 2).mean()
            self.last_amp = float(ratio.mean().detach())
        else:
            self.last_amp = float("nan")
        return total


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
        pred_space: str = "normalized",
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        if epsilon_mm <= 0:
            raise ValueError(f"epsilon_mm must be positive, got {epsilon_mm}")
        if pred_space not in ("normalized", "physical"):
            raise ValueError(
                f"pred_space must be normalized|physical, got {pred_space!r}"
            )
        self.pred_space = pred_space
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
        pred = _squeeze_channel(pred_norm)
        pred_mm = (
            pred
            if self.pred_space == "physical"
            else denormalize_precip(
                pred,
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
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
    pred_space: str = "normalized",
) -> nn.Module:
    """Dispatcher on ``cfg.loss.name`` (ai_rossby ``build_loss`` convention).

    ``pred_space`` is the space the mixture is formed in
    (``cfg.model.mix_space``), i.e. the space predictions arrive in.
    """
    name = str(cfg_loss.get("name", "regional_mse"))
    lat_weighted = bool(cfg_loss.get("lat_weighted", True))
    if name == "regional_mse":
        return RegionalPrecipMSE(
            lat,
            lon,
            box,
            space=str(cfg_loss.get("space", "normalized")),
            pred_space=pred_space,
            bias_weight=float(cfg_loss.get("bias_weight", 0.0)),
            var_weight=float(cfg_loss.get("var_weight", 0.0)),
            scale_mm=(
                float(cfg_loss["scale_mm"])
                if cfg_loss.get("scale_mm") is not None
                else None
            ),
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
            pred_space=pred_space,
            lat_weighted=lat_weighted,
            extra_mask=extra_mask,
        )
    raise ValueError(f"unknown loss name '{name}'")
