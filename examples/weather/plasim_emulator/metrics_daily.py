# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Daily-mean precipitation metrics - the selection metric for round 2.

Round 1 selected on 6-hourly gridpoint RMSE. That is the wrong target: the
downstream use is rare-event sampling of daily precipitation, and 6-hourly RMSE
rewards a smooth conditional mean rather than a calibrated distribution.

Primary metric here is **CRPS on the daily mean**. Threshold-exceedance scores
are reported alongside but are deliberately NOT training targets and are capped
at moderate thresholds: a Brier score gets its signal only from the exceedances,
so at rare thresholds it is dominated by sampling noise. Every threshold is
reported with its empirical event count so an uninformative threshold is
obvious rather than silently trusted.
"""

import numpy as np
import torch

# Moderate by design. 20 mm/day is ~p99 of the daily-mean field; above that the
# event count collapses and the score stops being evidence.
DEFAULT_THRESHOLDS_MMDAY = (1.0, 5.0, 10.0, 20.0)
STEPS_PER_DAY = 4


def aggregate_daily(x: torch.Tensor, dim: int = -3) -> torch.Tensor:
    """Mean over each block of 4 consecutive 6-hourly steps.

    `x` has a time axis at `dim` whose length must be a multiple of 4; trailing
    partial days are dropped rather than silently averaged over fewer steps.
    """
    x = x.movedim(dim, 0)
    n = (x.shape[0] // STEPS_PER_DAY) * STEPS_PER_DAY
    if n == 0:
        raise ValueError(f"need >= {STEPS_PER_DAY} steps to form a day, got {x.shape[0]}")
    x = x[:n].reshape(n // STEPS_PER_DAY, STEPS_PER_DAY, *x.shape[1:]).mean(1)
    return x.movedim(0, dim)


def kcrps(ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Kernel CRPS with the ensemble on axis 0.

    Mirrors physicsnemo.metrics.general.crps.kcrps; kept local so the evaluator
    has no hard dependency on the physicsnemo import path, but the estimator is
    identical (biased/"fair" form with the 2M^2 denominator).
    A 1-member ensemble reduces to MAE, which is the correct limit.
    """
    M = ens.shape[0]
    if M == 1:
        return (ens[0] - obs).abs()
    t1 = (ens - obs.unsqueeze(0)).abs().mean(0)
    t2 = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum((0, 1)) / (2 * M * M)
    return t1 - t2


def exceedance_prob(ens: torch.Tensor, threshold: float) -> torch.Tensor:
    """Empirical P[Y > threshold] from an ensemble on axis 0.

    Sample-based rather than closed-form: the daily mean is an average of four
    correlated hurdle-Gamma steps and has no analytic CDF, so the honest route
    is to draw the ensemble, aggregate to daily, then count.
    """
    return (ens > threshold).to(ens.dtype).mean(0)


def brier(prob: torch.Tensor, obs: torch.Tensor, threshold: float, weights=None):
    """Brier score plus the event count that determines whether it means anything."""
    y = (obs > threshold).to(prob.dtype)
    se = (prob - y) ** 2
    if weights is not None:
        se = se * weights
        bs = float(se.mean())
    else:
        bs = float(se.mean())
    return {
        "brier": bs,
        "n_events": int(y.sum()),
        "n_total": int(y.numel()),
        "base_rate": float(y.mean()),
    }


def reliability(prob: torch.Tensor, obs: torch.Tensor, threshold: float, n_bins: int = 10):
    """Binned reliability curve: mean forecast probability vs observed frequency."""
    y = (obs > threshold).to(prob.dtype)
    p = prob.reshape(-1)
    y = y.reshape(-1)
    edges = torch.linspace(0, 1, n_bins + 1, device=p.device, dtype=p.dtype)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(m.sum())
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
            "mean_forecast": float(p[m].mean()) if n else None,
            "observed_freq": float(y[m].mean()) if n else None,
        })
    return out


def score_daily(ens_6h: torch.Tensor, truth_6h: torch.Tensor, lat_w: torch.Tensor,
                box=None, thresholds=DEFAULT_THRESHOLDS_MMDAY):
    """Full daily-mean score set for one forecast.

    ens_6h   [M, T, H, W]  ensemble of 6-hourly precip in mm/day
    truth_6h [T, H, W]
    lat_w    [H, 1] cos-latitude weights, mean 1
    box      optional (lat_idx, lon_idx) for a regional stencil
    """
    ens = aggregate_daily(ens_6h, dim=1)      # [M, D, H, W]
    obs = aggregate_daily(truth_6h, dim=0)    # [D, H, W]
    D = obs.shape[0]

    def rmean(x):
        if box is None:
            return (x * lat_w).mean(dim=(-2, -1))
        la, lo = box
        return x[..., la, :][..., lo].mean(dim=(-2, -1))

    res = {"n_days": D, "crps": [], "rmse": [], "thresholds": {}}
    for d in range(D):
        res["crps"].append(float(rmean(kcrps(ens[:, d], obs[d]))))
        res["rmse"].append(float(torch.sqrt(rmean((ens[:, d].mean(0) - obs[d]) ** 2))))
    for thr in thresholds:
        per_day = []
        for d in range(D):
            p = exceedance_prob(ens[:, d], thr)
            b = brier(p, obs[d], thr, weights=lat_w)
            b["reliability"] = reliability(p, obs[d], thr)
            per_day.append(b)
        res["thresholds"][str(thr)] = per_day
    return res
