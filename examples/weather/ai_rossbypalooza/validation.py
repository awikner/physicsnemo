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

"""Validation driver: per-lead regional RMSE / bias / SEEPS for the gate
and its baselines (each expert alone, the equal-weight available-expert
mean). The baselines are the bar the gate must beat.

Streaming + DDP-safe (update/finalize with all-reduced sums), following
``examples/weather/ai_rossby/validate.py``. All scores are computed in
physical mm/day over the region box.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from losses import denormalize_precip
from mowe_precip import mix
from seeps import (
    SeepsClimatology,
    StreamingRegionalSEEPS,
    months_from_hours_since_1900,
    years_from_hours_since_1900,
)


def _all_reduce_sum(t: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


class StreamingRegionalScore:
    """Per-lead regional RMSE + bias (weighted, NaN-masked, DDP-safe)."""

    def __init__(
        self, *, n_leads: int, region_weights: torch.Tensor, device: torch.device
    ) -> None:
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.sq_sum = torch.zeros(n_leads, device=device)
        self.err_sum = torch.zeros(n_leads, device=device)
        self.w_sum = torch.zeros(n_leads, device=device)

    @torch.no_grad()
    def update(
        self, lead_index: int, pred_mm: torch.Tensor, target_mm: torch.Tensor
    ) -> None:
        finite = torch.isfinite(target_mm)
        w = self.weights.unsqueeze(0) * finite.float()
        err = pred_mm - torch.nan_to_num(target_mm)
        self.sq_sum[lead_index] += (w * err**2).sum()
        self.err_sum[lead_index] += (w * err).sum()
        self.w_sum[lead_index] += w.sum()

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        _all_reduce_sum(self.sq_sum)
        _all_reduce_sum(self.err_sum)
        _all_reduce_sum(self.w_sum)
        denom = self.w_sum.clamp(min=1e-12)
        return torch.sqrt(self.sq_sum / denom), self.err_sum / denom


class StreamingMonthlyRmseAcc:
    """Per-(year, month) lat-weighted RMSE + anomaly correlation (ACC) over
    a masked region (the IMD-coverage gridpoints), streaming + DDP-safe.

    Anomalies are relative to the monthly climatological mean
    (``clim_mean (12, H, W)`` from the climatology store). Bins are the
    calendar (year, month) of each sample's VALID day.
    """

    def __init__(
        self,
        *,
        bins: dict[tuple[int, int], int],
        clim_mean: torch.Tensor,
        region_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.bins = dict(bins)
        n = len(self.bins)
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.clim_mean = clim_mean.to(device=device, dtype=torch.float32)
        self.sq_sum = torch.zeros(n, device=device)
        self.s_pt = torch.zeros(n, device=device)
        self.s_pp = torch.zeros(n, device=device)
        self.s_tt = torch.zeros(n, device=device)
        self.w_sum = torch.zeros(n, device=device)

    @torch.no_grad()
    def update(
        self,
        bin_index: int,
        pred_mm: torch.Tensor,
        target_mm: torch.Tensor,
        months: torch.Tensor,
    ) -> None:
        finite = torch.isfinite(target_mm)
        w = self.weights.unsqueeze(0) * finite.float()
        t = torch.nan_to_num(target_mm)
        clim = self.clim_mean[months.long() - 1]  # (B, H, W)
        err = pred_mm - t
        p_anom = pred_mm - clim
        t_anom = t - clim
        self.sq_sum[bin_index] += (w * err**2).sum()
        self.s_pt[bin_index] += (w * p_anom * t_anom).sum()
        self.s_pp[bin_index] += (w * p_anom**2).sum()
        self.s_tt[bin_index] += (w * t_anom**2).sum()
        self.w_sum[bin_index] += w.sum()

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(rmse, acc, weight_total) per bin; empty bins are NaN."""
        for t in (self.sq_sum, self.s_pt, self.s_pp, self.s_tt, self.w_sum):
            _all_reduce_sum(t)
        denom = self.w_sum.clamp(min=1e-12)
        rmse = torch.sqrt(self.sq_sum / denom)
        acc = self.s_pt / torch.sqrt(
            self.s_pp.clamp(min=1e-12) * self.s_tt.clamp(min=1e-12)
        )
        empty = self.w_sum <= 0
        rmse[empty] = float("nan")
        acc[empty] = float("nan")
        return rmse, acc, self.w_sum


class MixtureValidator:
    """Scores the gate + baselines over a validation loader.

    Sources scored: ``"gate"``, ``"equal_weight"`` (mean of live experts'
    precip), and each expert by name (only on samples where it is live).
    Emitted metric keys: ``{source}/rmse_lead{tau}``, ``.../bias_lead{tau}``,
    ``.../seeps_lead{tau}``, plus ``.../{rmse,bias,seeps}_mean`` over leads.
    """

    def __init__(
        self,
        *,
        expert_names: list[str],
        lead_days: tuple[int, int],
        region_weights: torch.Tensor,
        seeps_climatology: SeepsClimatology,
        precip_mean: float,
        precip_std: float,
        precip_transform=None,
        device: torch.device,
        n_weight_map_samples: int = 2,
        monthly_region_weights: torch.Tensor | None = None,
        val_years: tuple[int, int] | None = None,
    ) -> None:
        self.expert_names = list(expert_names)
        self.lead_lo, self.lead_hi = int(lead_days[0]), int(lead_days[1])
        self.n_leads = self.lead_hi - self.lead_lo + 1
        self.region_weights = region_weights
        self.seeps_clim = seeps_climatology
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.precip_transform = precip_transform
        self.device = device
        self.n_weight_map_samples = int(n_weight_map_samples)
        # Monthly IMD-region RMSE/ACC: bins cover every (year, month) a
        # valid day can land in (leads spill past Dec into year+1).
        self.monthly_region_weights = monthly_region_weights
        self.month_bins: dict[tuple[int, int], int] = {}
        if monthly_region_weights is not None:
            if val_years is None:
                raise ValueError("monthly metrics need val_years")
            if seeps_climatology.clim_mean is None:
                raise ValueError(
                    "the climatology store lacks clim_mean — regenerate it "
                    "with tools/compute_seeps_climatology.py"
                )
            self.month_bins = {
                (y, m): i
                for i, (y, m) in enumerate(
                    (y, m)
                    for y in range(int(val_years[0]), int(val_years[1]) + 2)
                    for m in range(1, 13)
                )
            }

    def _sources(self) -> list[str]:
        return ["gate", "equal_weight", *self.expert_names]

    @torch.no_grad()
    def run(self, model, loader) -> tuple[dict[str, float], dict]:
        """Returns (metrics, extras); extras carries a few gate-weight maps
        (``(E, H, W)`` numpy arrays keyed by ``weights_lead{tau}``)."""
        scores = {
            s: StreamingRegionalScore(
                n_leads=self.n_leads,
                region_weights=self.region_weights,
                device=self.device,
            )
            for s in self._sources()
        }
        seeps = {
            s: StreamingRegionalSEEPS(
                n_leads=self.n_leads,
                climatology=self.seeps_clim,
                region_weights=self.region_weights,
                device=self.device,
            )
            for s in self._sources()
        }
        monthly = None
        if self.monthly_region_weights is not None:
            monthly = {
                s: StreamingMonthlyRmseAcc(
                    bins=self.month_bins,
                    clim_mean=self.seeps_clim.clim_mean,
                    region_weights=self.monthly_region_weights,
                    device=self.device,
                )
                for s in self._sources()
            }
        weight_maps: dict = {}

        was_training = model.training
        model.eval()
        for batch in loader:
            x = batch["expert_inputs"].to(self.device, non_blocking=True)
            mask = batch["expert_mask"].to(self.device, non_blocking=True)
            target_mm = batch["target_mm"].to(self.device, non_blocking=True)
            target_mm = target_mm.squeeze(1)
            taus = batch["lead_days"].to(self.device)
            months = months_from_hours_since_1900(batch["valid_time"]).to(
                self.device
            )
            years = years_from_hours_since_1900(batch["valid_time"]).to(
                self.device
            )

            weights, biases = model(x, mask, taus)
            pred_mm = denormalize_precip(
                mix(weights, biases, x[:, :, 0]),
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
            expert_mm = denormalize_precip(
                x[:, :, 0],
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
            live = mask > 0
            eq_mm = (expert_mm * live.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            eq_mm = eq_mm / live.sum(dim=1).clamp(min=1).unsqueeze(-1).unsqueeze(-1)

            # Bucket the batch by lead day (leads are mixed within a batch).
            for tau in taus.unique().tolist():
                li = int(tau) - self.lead_lo
                if not 0 <= li < self.n_leads:
                    continue
                sel = taus == tau
                t_mm = target_mm[sel]
                m_sel = months[sel]
                scores["gate"].update(li, pred_mm[sel], t_mm)
                seeps["gate"].update(li, pred_mm[sel], t_mm, m_sel)
                scores["equal_weight"].update(li, eq_mm[sel], t_mm)
                seeps["equal_weight"].update(li, eq_mm[sel], t_mm, m_sel)
                for ei, name in enumerate(self.expert_names):
                    esel = sel & live[:, ei]
                    if not esel.any():
                        continue
                    scores[name].update(li, expert_mm[esel, ei], target_mm[esel])
                    seeps[name].update(
                        li, expert_mm[esel, ei], target_mm[esel], months[esel]
                    )
                key = f"weights_lead{int(tau)}"
                if len(weight_maps) < self.n_weight_map_samples and key not in weight_maps:
                    weight_maps[key] = weights[sel][0].float().cpu().numpy()

            if monthly is not None:
                ym = years * 100 + months
                for code in ym.unique().tolist():
                    y, m = int(code) // 100, int(code) % 100
                    bi = self.month_bins.get((y, m))
                    if bi is None:
                        continue
                    sel = ym == code
                    t_mm = target_mm[sel]
                    m_sel = months[sel]
                    monthly["gate"].update(bi, pred_mm[sel], t_mm, m_sel)
                    monthly["equal_weight"].update(bi, eq_mm[sel], t_mm, m_sel)
                    for ei, name in enumerate(self.expert_names):
                        esel = sel & live[:, ei]
                        if not esel.any():
                            continue
                        monthly[name].update(
                            bi, expert_mm[esel, ei], target_mm[esel], months[esel]
                        )

        metrics: dict[str, float] = {}
        for s in self._sources():
            rmse, bias = scores[s].finalize()
            sp = seeps[s].finalize()
            for li in range(self.n_leads):
                tau = self.lead_lo + li
                metrics[f"{s}/rmse_lead{tau}"] = float(rmse[li])
                metrics[f"{s}/bias_lead{tau}"] = float(bias[li])
                metrics[f"{s}/seeps_lead{tau}"] = float(sp[li])
            metrics[f"{s}/rmse_mean"] = float(rmse.mean())
            metrics[f"{s}/bias_mean"] = float(bias.mean())
            metrics[f"{s}/seeps_mean"] = float(sp.mean())
        if monthly is not None:
            import math

            for s in self._sources():
                m_rmse, m_acc, m_w = monthly[s].finalize()
                vals_r, vals_a = [], []
                for (y, m), bi in self.month_bins.items():
                    r, a = float(m_rmse[bi]), float(m_acc[bi])
                    if math.isnan(r):
                        continue
                    metrics[f"{s}/imd_rmse_{y}-{m:02d}"] = r
                    metrics[f"{s}/imd_acc_{y}-{m:02d}"] = a
                    vals_r.append(r)
                    vals_a.append(a)
                if vals_r:
                    metrics[f"{s}/imd_rmse_mean"] = sum(vals_r) / len(vals_r)
                    metrics[f"{s}/imd_acc_mean"] = sum(vals_a) / len(vals_a)
        if was_training:
            model.train()
        return metrics, {"weight_maps": weight_maps}
