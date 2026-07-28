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

from mowe_precip import mix
from seeps import (
    SeepsClimatology,
    StreamingRegionalSEEPS,
    months_from_hours_since_1900,
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
        device: torch.device,
        n_weight_map_samples: int = 2,
    ) -> None:
        self.expert_names = list(expert_names)
        self.lead_lo, self.lead_hi = int(lead_days[0]), int(lead_days[1])
        self.n_leads = self.lead_hi - self.lead_lo + 1
        self.region_weights = region_weights
        self.seeps_clim = seeps_climatology
        self.precip_mean = float(precip_mean)
        self.precip_std = float(precip_std)
        self.device = device
        self.n_weight_map_samples = int(n_weight_map_samples)

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

            weights, biases = model(x, mask, taus)
            pred_mm = (
                mix(weights, biases, x[:, :, 0]) * self.precip_std
                + self.precip_mean
            )
            expert_mm = x[:, :, 0] * self.precip_std + self.precip_mean
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
        if was_training:
            model.train()
        return metrics, {"weight_maps": weight_maps}
