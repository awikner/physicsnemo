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

All losses take ``(pred, target_norm, target_mm)``:

* :class:`RegionalPrecipMSE` — weighted MSE in normalized space by default;
  ``space="physical"`` denormalizes with the shared IMERG stats first
  (linear, differentiable — the two differ only by a constant ``std^2``).
* :class:`RegionalPrecipLogMSE` — always physical:
  ``MSE(log1p(clamp(pred_mm, 0) / eps), log1p(target_mm / eps))``, which
  compresses the heavy tail so moderate/heavy intensity errors are not
  drowned out by the largest events.
* :class:`RegionalAlmostFairCRPS` — probabilistic: almost-fair CRPS (AIFS,
  arXiv:2412.15832) over an ensemble axis, for the FGN-style noise-conditioned
  gate (arXiv:2506.10772). ``pred`` may be ``(B, N, H, W)``; a deterministic
  ``(B, H, W)`` prediction scores as its regional weighted MAE (= the CRPS of
  a 1-member ensemble), so baselines stay directly comparable.
* :class:`RegionalPrecipFSS` — composite anchor + differentiable
  fractions-skill-score term (soft exceedance -> neighborhood fractions;
  Lagerquist & Ebert-Uphoff, arXiv:2203.11141) against exact-location
  double-penalty conditioning. Never used alone: an FSS-only objective is
  degenerate (mass can be rearranged freely within each window).
* :class:`RegionalPrecipAMSE` — drop-in MSE replacement implementing the
  double-penalty fix of Subich et al. (arXiv:2501.19374) on pooling-derived
  scale bands instead of spherical harmonics (the supervised region is a
  masked box, not the sphere): per band,
  ``(sigma_x - sigma_y)^2 + 2 max(sigma_x^2, sigma_y^2) (1 - rho)``,
  so shrinking small-scale amplitude no longer pays off where placement is
  unpredictable. Subsumes ``var_weight`` (its whole-region special case).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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


def pooled_fractions(
    field: torch.Tensor,
    weights: torch.Tensor,
    finite: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight-masked neighborhood mean: ``pool(w * field) / pool(w)``.

    ``field (B, H, W)``, ``weights (H, W)``, ``finite (B, H, W)`` bool. The
    ratio form makes zero-padding, the region mask, and NaN target cells all
    inert — cells with ``w = 0`` contribute nothing to either sum, so the
    padding mode cannot leak fake dry borders into the fractions (the failure
    mode of plain ``avg_pool`` on a 378-gridpoint region). Returns
    ``(f, w_pool)``: the pooled field (0 where the window holds no weight)
    and the pooled weight, the natural per-cell weight for scoring ``f``.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and >= 1, got {window}")
    w = (weights.unsqueeze(0) * finite.to(field.dtype)).to(field.dtype)
    pad = window // 2
    num = F.avg_pool2d(
        (w * field).unsqueeze(1), window, stride=1, padding=pad
    ).squeeze(1)
    den = F.avg_pool2d(w.unsqueeze(1), window, stride=1, padding=pad).squeeze(1)
    f = torch.where(den > 1e-12, num / den.clamp(min=1e-12), torch.zeros_like(num))
    return f, den


def almost_fair_crps(
    members: torch.Tensor, obs: torch.Tensor, alpha: float = 0.95
) -> torch.Tensor:
    """Pointwise almost-fair CRPS map (AIFS-CRPS, arXiv:2412.15832).

    ``members (B, N, H, W)``, ``obs (B, H, W)`` -> ``(B, H, W)``::

        afCRPS_a = mean_n |x_n - y|
                   - 0.5 * (a / (N (N-1)) + (1 - a) / N^2) * sum_{n,n'} |x_n - x_n'|

    ``alpha = 1`` is the fair (unbiased) estimator; ``alpha = 0`` the plain
    biased one. The 0.95 default re-anchors the one member that fair CRPS
    leaves unconstrained when all others equal the observation (a real risk
    at reduced precision). N = 1 degenerates to the MAE — the CRPS of a
    deterministic forecast — with no 0/0. O(N^2) pairwise on purpose: N is
    2–16 here, and this keeps the estimator autograd- and autocast-clean
    (``physicsnemo.metrics.general.crps.kcrps`` pins it in the tests).
    """
    if members.ndim != obs.ndim + 1:
        raise ValueError(
            f"members must carry one ensemble dim more than obs, got "
            f"{tuple(members.shape)} vs {tuple(obs.shape)}"
        )
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    n = members.shape[1]
    skill = (members - obs.unsqueeze(1)).abs().mean(dim=1)
    if n == 1:
        return skill
    spread = (members.unsqueeze(1) - members.unsqueeze(2)).abs().sum(dim=(1, 2))
    factor = float(alpha) / (n * (n - 1)) + (1.0 - float(alpha)) / (n * n)
    return skill - 0.5 * factor * spread


def scale_bands(
    field: torch.Tensor,
    weights: torch.Tensor,
    finite: torch.Tensor,
    windows: Sequence[int],
) -> list[torch.Tensor]:
    """Successive weight-masked smoothings -> band-pass components.

    With ``windows = (3, 7)``: ``[field - p3, p3 - p7, p7]`` — finer than the
    first window, between windows, and the remaining large-scale field. Used
    by both the AMSE loss and the validation-side banded-amplitude
    diagnostic so the two describe the same decomposition.
    """
    levels = [field]
    for k in windows:
        levels.append(pooled_fractions(field, weights, finite, int(k))[0])
    return [levels[i] - levels[i + 1] for i in range(len(windows))] + [levels[-1]]


def gate_smoothness_penalty(
    weights: torch.Tensor,
    biases: torch.Tensor,
    region_weights: torch.Tensor,
    *,
    bias_scale_mm: float = 9.3,
) -> torch.Tensor:
    """Weighted TV (squared finite differences) of the gate maps over the region.

    Smooths the MIXTURE PARAMETERS, not the rain: sharp structure carried by
    the experts passes through a spatially smooth convex combination
    unchanged, so this regularizes toward coherent expert-selection regimes
    (and suppresses patch-boundary blockiness from the per-patch detokenizer
    head) without re-blurring precipitation — a TV penalty on the mixed field
    itself would fight the amplitude objectives directly. ``weights`` /
    ``biases`` are gate outputs ``(B, [ens,] E, H, W)``; biases are mm/day and
    are divided by ``bias_scale_mm`` so one coefficient serves both terms. An
    edge only counts when both endpoints carry region weight.
    """
    w = region_weights
    wy = torch.minimum(w[1:, :], w[:-1, :])
    wx = torch.minimum(w[:, 1:], w[:, :-1])
    denom = (wy.sum() + wx.sum()).clamp(min=1e-12)

    def tv(f: torch.Tensor) -> torch.Tensor:
        dy = (f[..., 1:, :] - f[..., :-1, :]) ** 2
        dx = (f[..., :, 1:] - f[..., :, :-1]) ** 2
        return ((dy * wy).sum(dim=(-2, -1)) + (dx * wx).sum(dim=(-2, -1))).mean() / denom

    return tv(weights) + tv(biases / float(bias_scale_mm))


def load_precip_quantile_thresholds(
    store: str,
    quantiles: Sequence[float],
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    floor_mm: float = 1.0,
    coord_tol: float = 1e-3,
) -> torch.Tensor:
    """(Q, H, W) climatological threshold maps from a quantile store.

    The store (written by ``tools/compute_precip_quantiles.py``) holds
    ``precip_quantile_mm (quantile, lat, lon)``. Percentile thresholds remove
    frequency bias from exceedance terms so they isolate displacement error
    (Roberts & Lean 2008 recommendation). Grids are matched by value and a
    mismatch raises — never silently regridded. Thresholds are floored at
    ``floor_mm`` so near-dry climatology cells cannot produce degenerate
    sigmoid widths.
    """
    import xarray as xr

    with xr.open_zarr(store, consolidated=True) as ds:
        if "precip_quantile_mm" not in ds:
            raise ValueError(
                f"{store} has no precip_quantile_mm — write it with "
                "tools/compute_precip_quantiles.py"
            )
        have = np.asarray(ds["quantile"].values, dtype="float64")
        vals = ds["precip_quantile_mm"].values.astype("float32")
        s_lat = np.asarray(ds["lat"].values, dtype="float64")
        s_lon = np.asarray(ds["lon"].values, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    if s_lat.shape != lat.shape or not np.allclose(s_lat, lat, atol=coord_tol):
        raise ValueError(f"quantile store latitudes do not match the dataset grid ({store})")
    if s_lon.shape != lon.shape or not np.allclose(s_lon, lon, atol=coord_tol):
        raise ValueError(f"quantile store longitudes do not match the dataset grid ({store})")
    maps = []
    for want in quantiles:
        idx = np.nonzero(np.abs(have - float(want)) <= 1e-6)[0]
        if idx.size == 0:
            raise ValueError(
                f"quantile {want} not in store (has {have.tolist()}) — "
                "regenerate with tools/compute_precip_quantiles.py"
            )
        maps.append(vals[idx[0]])
    t = torch.from_numpy(np.stack(maps)).float()
    return torch.nan_to_num(t, nan=float(floor_mm)).clamp(min=float(floor_mm))


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


class RegionalAlmostFairCRPS(nn.Module):
    """Region-weighted almost-fair CRPS in physical mm/day.

    The training loss for the FGN-style noise-conditioned gate: ``pred`` is
    ``(B, N, H, W)`` (N ensemble members mixed from N noise draws) or
    ``(B, H, W)`` / ``(B, 1, H, W)``, which score as the regional weighted
    MAE — the CRPS of a deterministic forecast — so every deterministic
    baseline's ``{source}/loss`` stays directly comparable to the gate's.

    ``scale_mm`` divides LINEARLY: CRPS is first-order in the error (the MSE
    losses divide by ``scale_mm**2`` because they are quadratic).

    Diagnostics (detached, for logging): ``last_skill`` (regional ensemble
    MAE), ``last_spread`` (regional mean pairwise member distance),
    ``last_ens_std`` (regional mean member std in mm/day — the spread lever
    to watch against the convex-mixture amplitude cap; MOWE.md notes each
    member's amplitude is capped, so across-member variance is where
    calibrated spread must come from).
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        box: Sequence[float],
        *,
        alpha: float = 0.95,
        precip_mean: float = 0.0,
        precip_std: float = 1.0,
        precip_transform=None,
        pred_space: str = "physical",
        scale_mm: float | None = None,
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if pred_space not in ("normalized", "physical"):
            raise ValueError(
                f"pred_space must be normalized|physical, got {pred_space!r}"
            )
        if scale_mm is not None and scale_mm <= 0:
            raise ValueError(f"scale_mm must be positive, got {scale_mm}")
        self.alpha = float(alpha)
        self.pred_space = pred_space
        self.scale_mm = None if scale_mm is None else float(scale_mm)
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.precip_transform = precip_transform
        self.last_skill: float = float("nan")
        self.last_spread: float = float("nan")
        self.last_ens_std: float = float("nan")
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
        # (B, H, W) and (B, 1, H, W) are the same 1-member ensemble; the CRPS
        # of one member IS its MAE, so the two readings coincide numerically.
        members = pred if pred.ndim == 4 else pred.unsqueeze(1)
        if self.pred_space == "normalized":
            members = denormalize_precip(
                members,
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
        members = members.clamp(min=0.0)
        t_mm = _squeeze_channel(target_mm)
        finite = torch.isfinite(t_mm)
        obs = torch.nan_to_num(t_mm).clamp(min=0.0)
        crps_map = almost_fair_crps(members, obs, self.alpha)
        out = _weighted_regional_mean(crps_map, self.weights, finite)
        if self.scale_mm is not None:
            out = out / self.scale_mm
        with torch.no_grad():
            n = members.shape[1]
            skill = (members - obs.unsqueeze(1)).abs().mean(dim=1)
            self.last_skill = float(
                _weighted_regional_mean(skill, self.weights, finite)
            )
            if n > 1:
                spread = (
                    members.unsqueeze(1) - members.unsqueeze(2)
                ).abs().sum(dim=(1, 2)) / (n * (n - 1))
                self.last_spread = float(
                    _weighted_regional_mean(spread, self.weights, finite)
                )
                self.last_ens_std = float(
                    _weighted_regional_mean(
                        members.std(dim=1, unbiased=True), self.weights, finite
                    )
                )
            else:
                self.last_spread = float("nan")
                self.last_ens_std = float("nan")
        return out


class RegionalPrecipFSS(nn.Module):
    """Composite anchor loss + differentiable neighborhood-FSS term.

    The FSS term relaxes exact-location conditioning: both fields become soft
    exceedance probabilities ``sigmoid(sharpness * (x - T) / T)`` per
    threshold map ``T``, are neighborhood-averaged with weight-masked pooling
    (:func:`pooled_fractions`), and are scored as the fractions Brier score
    over its worst-case reference — ``1 - FSS`` per (window, threshold),
    averaged. Week-2 displacement error is O(several hundred km), so windows
    of 3–5 cells on the 1-degree grid are the intended scales.

    Only ever a composite (``total = anchor + fss_weight * term``): an
    FSS-only objective lets mass be rearranged arbitrarily within each
    window and degenerates (checkerboards; Lagerquist & Ebert-Uphoff 2022).
    For an ensemble ``pred (B, N, H, W)`` the exceedance is averaged over
    members BEFORE pooling — the probabilistic FSS (Necker et al. 2024),
    immune to member-noise inflation. Soft-thresholding BOTH fields makes the
    term exactly 0 for a perfect forecast (the obs side carries no gradient).

    ``ramp_epochs > 0`` anneals the weight 0 -> ``fss_weight`` over that many
    epochs — in training mode only, so the validation objective (which early
    stopping compares across epochs) stays fixed. Off by default: the anchor
    already prevents the degeneracy.
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        box: Sequence[float],
        *,
        anchor: nn.Module,
        threshold_maps: torch.Tensor,
        threshold_labels: Sequence[str],
        windows: Sequence[int] = (3, 5),
        sharpness: float = 10.0,
        fss_weight: float = 0.3,
        ramp_epochs: int = 0,
        precip_mean: float = 0.0,
        precip_std: float = 1.0,
        precip_transform=None,
        pred_space: str = "physical",
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        if threshold_maps.ndim != 3 or threshold_maps.shape[0] != len(
            threshold_labels
        ):
            raise ValueError(
                f"threshold_maps must be (Q, H, W) matching threshold_labels, "
                f"got {tuple(threshold_maps.shape)} vs {len(threshold_labels)}"
            )
        if (threshold_maps <= 0).any():
            raise ValueError("threshold maps must be strictly positive (mm/day)")
        for k in windows:
            if int(k) < 1 or int(k) % 2 == 0:
                raise ValueError(f"windows must be odd and >= 1, got {windows}")
        if sharpness <= 0:
            raise ValueError(f"sharpness must be positive, got {sharpness}")
        if fss_weight < 0:
            raise ValueError(f"fss_weight must be >= 0, got {fss_weight}")
        if ramp_epochs < 0:
            raise ValueError(f"ramp_epochs must be >= 0, got {ramp_epochs}")
        if pred_space not in ("normalized", "physical"):
            raise ValueError(
                f"pred_space must be normalized|physical, got {pred_space!r}"
            )
        self.anchor = anchor
        self.windows = [int(k) for k in windows]
        self.threshold_labels = [str(s) for s in threshold_labels]
        self.sharpness = float(sharpness)
        self.fss_weight = float(fss_weight)
        self.ramp_epochs = int(ramp_epochs)
        self.pred_space = pred_space
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.precip_transform = precip_transform
        self._epoch = 0
        self.last_anchor: float = float("nan")
        self.last_fss_term: float = float("nan")
        self.register_buffer("threshold_maps", threshold_maps.float())
        self.register_buffer(
            "weights",
            region_weights(
                lat, lon, box, lat_weighted=lat_weighted, extra_mask=extra_mask
            ),
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def forward(
        self,
        pred: torch.Tensor,
        target_norm: torch.Tensor,
        target_mm: torch.Tensor,
    ) -> torch.Tensor:
        total = self.anchor(pred, target_norm, target_mm)

        members = pred if pred.ndim == 4 else pred.unsqueeze(1)
        if self.pred_space == "normalized":
            members = denormalize_precip(
                members,
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
        members = members.clamp(min=0.0)
        t_mm = _squeeze_channel(target_mm)
        finite = torch.isfinite(t_mm)
        obs = torch.nan_to_num(t_mm).clamp(min=0.0)

        terms = []
        for qi in range(self.threshold_maps.shape[0]):
            thr = self.threshold_maps[qi]  # (H, W), > 0 by construction
            steep = self.sharpness / thr
            b_p = torch.sigmoid((members - thr) * steep).mean(dim=1)  # pFSS
            b_o = torch.sigmoid((obs - thr) * steep)
            for k in self.windows:
                f_p, w_pool = pooled_fractions(b_p, self.weights, finite, k)
                f_o, _ = pooled_fractions(b_o, self.weights, finite, k)
                num = (w_pool * (f_p - f_o) ** 2).sum()
                den = (w_pool * (f_p**2 + f_o**2)).sum()
                # An informationless (window, threshold) — e.g. an all-dry
                # winter batch at the 95th JJAS percentile — contributes 0,
                # never NaN.
                terms.append(
                    torch.where(
                        den > 1e-8, num / den.clamp(min=1e-8), num.new_zeros(())
                    )
                )
        fss_term = torch.stack(terms).mean()

        w_eff = self.fss_weight
        if self.training and self.ramp_epochs > 0:
            w_eff = w_eff * min(1.0, (self._epoch + 1) / self.ramp_epochs)
        self.last_anchor = float(total.detach())
        self.last_fss_term = float(fss_term.detach())
        return total + w_eff * fss_term


class RegionalPrecipAMSE(nn.Module):
    """Amplitude-preserving MSE (Subich et al., arXiv:2501.19374), regional.

    The double-penalty fix: standard MSE decomposes per scale as
    ``sx^2 + sy^2 - 2 sx sy rho``, so wherever placement is unpredictable
    (``rho ~ 0``) the model minimizes MSE by shrinking amplitude toward
    ``rho * sy`` — that shrinkage IS the intensity blurring this project
    targets. AMSE rewrites each scale as

        ``(sx - sy)^2 + 2 max(sx^2, sy^2) (1 - rho)``

    so shrinking below the observed amplitude no longer reduces the
    decorrelation penalty (the ``max`` floors it at the obs variance) and the
    only gradient at decorrelated scales pushes amplitude toward truth.
    Parameter-free at each scale; subsumes ``var_weight`` (its single-band,
    whole-region special case).

    The paper's spherical harmonics need the full sphere; the supervised
    region here is a masked 30x40-degree box, so scales come from
    weight-masked pooling instead: with ``windows = (3, 7)`` the bands are
    ``x - p3(x)`` (< ~3 deg), ``p3(x) - p7(x)`` (3–7 deg) and ``p7(x)``
    (> ~7 deg), each scored with region-weighted std and correlation, plus a
    separate ``(mu_x - mu_y)^2`` bias term (the k = 0 component). ``scale_mm``
    divides quadratically, like the MSE losses. An ensemble ``(B, N, H, W)``
    prediction is scored per member and averaged (AMSE protects per-member
    sharpness; only used in combination arms).
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        box: Sequence[float],
        *,
        windows: Sequence[int] = (3, 7),
        scale_mm: float | None = None,
        precip_mean: float = 0.0,
        precip_std: float = 1.0,
        precip_transform=None,
        pred_space: str = "physical",
        lat_weighted: bool = True,
        extra_mask=None,
    ) -> None:
        super().__init__()
        ws = [int(k) for k in windows]
        if ws != sorted(ws) or len(set(ws)) != len(ws):
            raise ValueError(f"windows must be strictly increasing, got {windows}")
        for k in ws:
            if k < 1 or k % 2 == 0:
                raise ValueError(f"windows must be odd and >= 1, got {windows}")
        if scale_mm is not None and scale_mm <= 0:
            raise ValueError(f"scale_mm must be positive, got {scale_mm}")
        if pred_space not in ("normalized", "physical"):
            raise ValueError(
                f"pred_space must be normalized|physical, got {pred_space!r}"
            )
        self.windows = ws
        self.scale_mm = None if scale_mm is None else float(scale_mm)
        self.pred_space = pred_space
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.precip_transform = precip_transform
        self.last_bias_mm: float = float("nan")
        # sigma_pred / sigma_obs per band, finest first (the regional
        # effective-resolution diagnostic; ~1 everywhere = amplitude kept).
        self.last_amp_bands: list[float] = []
        self.register_buffer(
            "weights",
            region_weights(
                lat, lon, box, lat_weighted=lat_weighted, extra_mask=extra_mask
            ),
        )

    def _bands(
        self, field: torch.Tensor, finite: torch.Tensor
    ) -> list[torch.Tensor]:
        return scale_bands(field, self.weights, finite, self.windows)

    def forward(
        self,
        pred: torch.Tensor,
        target_norm: torch.Tensor,
        target_mm: torch.Tensor,
    ) -> torch.Tensor:
        members = pred if pred.ndim == 4 else pred.unsqueeze(1)
        n_ens = members.shape[1]
        if self.pred_space == "normalized":
            members = denormalize_precip(
                members,
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
        members = members.clamp(min=0.0)
        t_mm = _squeeze_channel(target_mm)
        finite = torch.isfinite(t_mm)
        obs = torch.nan_to_num(t_mm).clamp(min=0.0)

        # Fold members into the batch; repeat the target alongside.
        b = members.shape[0]
        x = members.reshape(b * n_ens, *members.shape[2:])
        y = obs.repeat_interleave(n_ens, dim=0)
        fin = finite.repeat_interleave(n_ens, dim=0)
        w = self.weights.unsqueeze(0) * fin.to(x.dtype)  # (B*N, H, W)
        wsum = w.sum(dim=(-2, -1)).clamp(min=1e-12)

        def wmean(f: torch.Tensor) -> torch.Tensor:
            return (f * w).sum(dim=(-2, -1)) / wsum

        total = torch.zeros((), device=x.device, dtype=x.dtype)
        amp_bands: list[float] = []
        for bx, by in zip(self._bands(x, fin), self._bands(y, fin)):
            bx = bx - wmean(bx)[:, None, None]
            by = by - wmean(by)[:, None, None]
            var_x = wmean(bx**2)
            var_y = wmean(by**2)
            sx = torch.sqrt(var_x.clamp(min=1e-12))
            sy = torch.sqrt(var_y.clamp(min=1e-12))
            cov = wmean(bx * by)
            rho = (cov / (sx * sy).clamp(min=1e-12)).clamp(-1.0, 1.0)
            term = (sx - sy) ** 2 + 2.0 * torch.maximum(var_x, var_y) * (1.0 - rho)
            # A band empty on both sides (e.g. an all-dry sample) is
            # informationless, not an error.
            term = torch.where((sx < 1e-6) & (sy < 1e-6), torch.zeros_like(term), term)
            total = total + term.mean()
            amp_bands.append(float((sx / sy.clamp(min=1e-6)).mean().detach()))
        bias = wmean(x) - wmean(y)
        total = total + (bias**2).mean()

        self.last_bias_mm = float(bias.mean().detach())
        self.last_amp_bands = amp_bands
        if self.scale_mm is not None:
            total = total / self.scale_mm**2
        return total


def resolve_fss_thresholds(
    cfg_thr, lat: np.ndarray, lon: np.ndarray
) -> tuple[torch.Tensor, list[str]]:
    """``cfg.loss.thresholds`` -> ``((Q, H, W) maps, labels)``.

    ``kind: percentile`` reads per-gridpoint climatological quantile maps
    (labels ``p75`` ...); ``kind: fixed`` broadcasts constant mm/day values
    (labels ``5mm`` ..., matching the validator's threshold tags). Shared by
    the FSS loss and the validation-side hard-FSS metric so both use
    identical thresholds.
    """
    kind = str(cfg_thr.get("kind", "fixed"))
    if kind == "percentile":
        store = cfg_thr.get("store", None)
        values = list(cfg_thr.get("values", ()))
        if not store or not values:
            raise ValueError(
                "thresholds.kind=percentile needs thresholds.store and "
                "thresholds.values"
            )
        maps = load_precip_quantile_thresholds(
            str(store),
            [float(v) for v in values],
            lat,
            lon,
            floor_mm=float(cfg_thr.get("floor_mm", 1.0)),
        )
        labels = [f"p{float(v):g}" for v in values]
        return maps, labels
    if kind == "fixed":
        values = list(cfg_thr.get("values_mm", ()))
        if not values:
            raise ValueError("thresholds.kind=fixed needs thresholds.values_mm")
        h, w = np.asarray(lat).size, np.asarray(lon).size
        maps = torch.stack(
            [torch.full((h, w), float(v)) for v in values]
        )
        labels = [f"{float(v):g}".replace(".", "p") + "mm" for v in values]
        return maps, labels
    raise ValueError(f"thresholds.kind must be percentile|fixed, got {kind!r}")


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
    if name == "regional_crps":
        return RegionalAlmostFairCRPS(
            lat,
            lon,
            box,
            alpha=float(cfg_loss.get("alpha", 0.95)),
            scale_mm=(
                float(cfg_loss["scale_mm"])
                if cfg_loss.get("scale_mm") is not None
                else None
            ),
            precip_mean=precip_mean,
            precip_std=precip_std,
            precip_transform=precip_transform,
            pred_space=pred_space,
            lat_weighted=lat_weighted,
            extra_mask=extra_mask,
        )
    if name == "regional_amse":
        return RegionalPrecipAMSE(
            lat,
            lon,
            box,
            windows=[int(k) for k in cfg_loss.get("windows", (3, 7))],
            scale_mm=(
                float(cfg_loss["scale_mm"])
                if cfg_loss.get("scale_mm") is not None
                else None
            ),
            precip_mean=precip_mean,
            precip_std=precip_std,
            precip_transform=precip_transform,
            pred_space=pred_space,
            lat_weighted=lat_weighted,
            extra_mask=extra_mask,
        )
    if name == "regional_fss":
        anchor_cfg = cfg_loss.get("anchor", None)
        if anchor_cfg is None:
            raise ValueError(
                "regional_fss is a composite term and needs cfg.loss.anchor "
                "(an FSS-only objective is degenerate)"
            )
        anchor = build_loss(
            anchor_cfg,
            lat=lat,
            lon=lon,
            box=box,
            precip_mean=precip_mean,
            precip_std=precip_std,
            precip_transform=precip_transform,
            extra_mask=extra_mask,
            pred_space=pred_space,
        )
        thr_cfg = cfg_loss.get("thresholds", None)
        if thr_cfg is None:
            raise ValueError("regional_fss needs cfg.loss.thresholds")
        threshold_maps, threshold_labels = resolve_fss_thresholds(
            thr_cfg, lat, lon
        )
        return RegionalPrecipFSS(
            lat,
            lon,
            box,
            anchor=anchor,
            threshold_maps=threshold_maps,
            threshold_labels=threshold_labels,
            windows=[int(k) for k in cfg_loss.get("windows", (3, 5))],
            sharpness=float(cfg_loss.get("sharpness", 10.0)),
            fss_weight=float(cfg_loss.get("fss_weight", 0.3)),
            ramp_epochs=int(cfg_loss.get("ramp_epochs", 0)),
            precip_mean=precip_mean,
            precip_std=precip_std,
            precip_transform=precip_transform,
            pred_space=pred_space,
            lat_weighted=lat_weighted,
            extra_mask=extra_mask,
        )
    raise ValueError(f"unknown loss name '{name}'")
