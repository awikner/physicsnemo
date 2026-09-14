# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Losses for the PlaSim emulator.

State channels use a latitude-weighted L2 (area weighting matters on a lat/lon
grid: without it the poles, which carry a tiny fraction of the area, dominate
the gradient). Diagnostics use the hurdle-Gamma negative log likelihood, with an
explicit weight on precipitation.

v11 used ``channel_weights: 'constant'`` and a plain z-scored L2 for every
channel including precipitation. Both are fixed here.
"""

import numpy as np
import torch
import torch.nn as nn

from channels import DIAGNOSTIC_CHANNELS, PRECIP_CHANNEL
from distributions import HurdleGamma


def latitude_weights(lat_deg: np.ndarray) -> torch.Tensor:
    """cos(lat) weights, normalized to mean 1 over the grid."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))
    w = np.clip(w, 0.0, None)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class StateLoss(nn.Module):
    """Latitude-weighted, optionally per-channel-weighted L2 on z-scored state."""

    def __init__(self, lat_deg, channel_weights: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("w_lat", latitude_weights(lat_deg)[None, None, :, None])
        if channel_weights is None:
            self.register_buffer("w_chan", torch.ones(1))
        else:
            self.register_buffer("w_chan", channel_weights[None, :, None, None])

    def forward(self, pred, target):
        # pred/target: [B, C, H, W] or [B, T, C, H, W]
        if pred.dim() == 5:
            pred = pred.flatten(0, 1)
            target = target.flatten(0, 1)
        se = (pred - target) ** 2
        return (se * self.w_lat * self.w_chan).mean()


class DiagnosticLoss(nn.Module):
    """Hurdle-Gamma NLL over the diagnostics, latitude weighted.

    `scales` maps each diagnostic name to the scale its Gamma was fitted in
    (written by tools/compute_stats.py). `weights` lets precipitation be
    upweighted relative to the other diagnostics.
    """

    def __init__(self, lat_deg, scales: dict, weights: dict | None = None):
        super().__init__()
        self.register_buffer("w_lat", latitude_weights(lat_deg)[None, :, None])
        self.scales = [float(scales[n]["scale"] if isinstance(scales[n], dict)
                             else scales[n]) for n in DIAGNOSTIC_CHANNELS]
        w = weights or {}
        self.weights = [float(w.get(n, 1.0)) for n in DIAGNOSTIC_CHANNELS]

    def forward(self, diag_raw, diag_target):
        """
        diag_raw    : [B, T, 3*ndiag, H, W] raw params  (or [B, 3*ndiag, H, W])
        diag_target : same leading dims, [.., ndiag, H, W], PHYSICAL units >= 0
        """
        if diag_raw.dim() == 5:
            diag_raw = diag_raw.flatten(0, 1)
            diag_target = diag_target.flatten(0, 1)

        total = diag_raw.new_zeros(())
        parts = {}
        for i, name in enumerate(DIAGNOSTIC_CHANNELS):
            raw_i = diag_raw[:, 3 * i : 3 * (i + 1)]
            dist = HurdleGamma(raw_i, scale=self.scales[i])
            nll = dist.nll(diag_target[:, i])
            term = (nll * self.w_lat).mean()
            parts[name] = term.detach()
            total = total + self.weights[i] * term
        return total, parts


class DeterministicDiagnosticLoss(nn.Module):
    """Baseline: latitude-weighted L2 on log1p-transformed diagnostics.

    This is the like-for-like point-forecast comparison for the probabilistic
    head. log1p(y/eps) is PhysicsNeMo's PrecipNorm (Pathak et al. 2022) and is
    still much better than v11's plain z-score for a zero-inflated field.
    """

    def __init__(self, lat_deg, eps: float = 1e-5, weights: dict | None = None):
        super().__init__()
        self.register_buffer("w_lat", latitude_weights(lat_deg)[None, :, None])
        self.eps = eps
        w = weights or {}
        self.weights = [float(w.get(n, 1.0)) for n in DIAGNOSTIC_CHANNELS]

    def forward(self, diag_pred, diag_target):
        if diag_pred.dim() == 5:
            diag_pred = diag_pred.flatten(0, 1)
            diag_target = diag_target.flatten(0, 1)
        tgt = torch.log1p(torch.clamp(diag_target, min=0) / self.eps)
        total = diag_pred.new_zeros(())
        parts = {}
        for i, name in enumerate(DIAGNOSTIC_CHANNELS):
            se = (diag_pred[:, i] - tgt[:, i]) ** 2
            term = (se * self.w_lat).mean()
            parts[name] = term.detach()
            total = total + self.weights[i] * term
        return total, parts


def default_precip_weight(w: float = 5.0) -> dict:
    """Upweight precipitation relative to the other diagnostics."""
    return {n: (w if n == PRECIP_CHANNEL else 1.0) for n in DIAGNOSTIC_CHANNELS}


class EnsembleStateLoss(nn.Module):
    """Energy score (multivariate CRPS) over an ensemble of state predictions.

    Why this exists: a plain L2 state loss is minimised by the conditional mean,
    so a model given noise input channels is *rewarded* for ignoring them. That
    is exactly what happened here - after training, the encoder weights on the
    noise channels came out ~400x smaller than those on the state channels, the
    ensemble spread collapsed to 0.04% of the signal, and spread/skill fell to
    ~0.00 instead of ~1.

    The energy score is a strictly proper scoring rule for a distribution:

        ES = (1/M) sum_i ||x_i - y|| - 1/(2 M (M-1)) sum_{i!=j} ||x_i - x_j||

    The first term rewards accuracy, the second *rewards spread*, so an
    underdispersed ensemble is penalised. It is minimised only when the
    predictive distribution matches the truth, which makes the noise channels
    worth using. M = 2 already gives an unbiased estimate, so this costs 2
    forward passes per step rather than a full ensemble.
    """

    _is_ensemble = True

    def __init__(self, lat_deg):
        super().__init__()
        self.register_buffer("w_lat", latitude_weights(lat_deg)[None, None, :, None])

    def forward(self, ens, target):
        """ens [M, B, C, H, W] (or [M, B, T, C, H, W]); target without the M axis."""
        if ens.dim() == 6:
            ens = ens.flatten(1, 2)
            target = target.flatten(0, 1)
        M = ens.shape[0]
        if M < 2:
            raise ValueError("energy score needs at least 2 ensemble members")
        w = self.w_lat
        # Area-weighted L2 norm per member, averaged over the batch.
        d = torch.sqrt((((ens - target.unsqueeze(0)) ** 2) * w).sum(dim=(2, 3, 4)) + 1e-12)
        term1 = d.mean()
        # Pairwise spread term over distinct member pairs.
        diff = ens.unsqueeze(0) - ens.unsqueeze(1)              # [M,M,B,C,H,W]
        p = torch.sqrt(((diff ** 2) * w).sum(dim=(3, 4, 5)) + 1e-12)
        term2 = p.sum() / (2.0 * M * (M - 1) * ens.shape[1])
        return term1 - term2


class DailyAggregateCRPS(nn.Module):
    """CRPS on the DAILY MEAN of a diagnostic, over a rollout.

    Round 2's selection metric is daily-mean precipitation, not the 6-hourly
    field, and the two are not the same objective: a per-step NLL can be
    well-optimised while the daily aggregate is badly calibrated, because the
    errors across the four steps of a day are correlated.

    CRPS is used rather than a threshold/Brier term on purpose. A Brier term
    draws gradient only from the rare exceedances, so it is dominated by
    sampling noise; CRPS is smooth, strictly proper, and gets signal from every
    sample. Threshold scores stay a reporting metric only.

    Requires the rollout to cover at least one whole day (unroll_steps >= 4).
    """

    STEPS_PER_DAY = 4

    def __init__(self, lat_deg, scales: dict, weights: dict | None = None,
                 n_samples: int = 8, tau: float = 0.1):
        super().__init__()
        self.register_buffer("w_lat", latitude_weights(lat_deg)[None, :, None])
        self.scales = [float(scales[n]["scale"] if isinstance(scales[n], dict)
                             else scales[n]) for n in DIAGNOSTIC_CHANNELS]
        w = weights or {}
        self.weights = [float(w.get(n, 1.0)) for n in DIAGNOSTIC_CHANNELS]
        self.n_samples = n_samples
        self.tau = tau

    def forward(self, diag_raw, diag_target):
        """diag_raw [B, T, 3*ndiag, H, W]; diag_target [B, T, ndiag, H, W].

        Returns (total, parts). Silently returns zero if T < 4 so the caller can
        enable this unconditionally.
        """
        if diag_raw.dim() != 5:
            raise ValueError("DailyAggregateCRPS needs a rollout axis [B,T,...]")
        B, T = diag_raw.shape[0], diag_raw.shape[1]
        nday = T // self.STEPS_PER_DAY
        if nday == 0:
            z = diag_raw.new_zeros(())
            return z, {n: z.detach() for n in DIAGNOSTIC_CHANNELS}

        keep = nday * self.STEPS_PER_DAY
        total = diag_raw.new_zeros(())
        parts = {}
        for i, name in enumerate(DIAGNOSTIC_CHANNELS):
            raw = diag_raw[:, :keep, 3 * i:3 * (i + 1)]          # [B,keep,3,H,W]
            tgt = diag_target[:, :keep, i]                        # [B,keep,H,W]
            dist = HurdleGamma(raw.flatten(0, 1), scale=self.scales[i])
            # M reparameterised draws, reshaped back to [M,B,day,4,H,W] -> daily mean
            draws = torch.stack(
                [dist.rsample(tau=self.tau) for _ in range(self.n_samples)], 0
            ).reshape(self.n_samples, B, nday, self.STEPS_PER_DAY, *tgt.shape[-2:])
            ens = draws.mean(3)                                   # [M,B,day,H,W]
            obs = tgt.reshape(B, nday, self.STEPS_PER_DAY, *tgt.shape[-2:]).mean(2)

            M = self.n_samples
            t1 = (ens - obs.unsqueeze(0)).abs().mean(0)
            t2 = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum((0, 1)) / (2 * M * M)
            crps = ((t1 - t2) * self.w_lat).mean()
            parts[name] = crps.detach()
            total = total + self.weights[i] * crps
        return total, parts
