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
physical mm/day over the SAME region the training loss uses (monsoon box
intersected with IMD gauge coverage) — the gate is only supervised there.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from losses import denormalize_precip, normalize_precip
from mowe_precip import mix
from seeps import (
    P1_MAX,
    P1_MIN,
    SeepsClimatology,
    StreamingRegionalSEEPS,
    months_from_hours_since_1900,
    seeps_penalty,
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


class StreamingMonthlyScores:
    """Per-calendar-month lat-weighted RMSE, bias, anomaly correlation (ACC)
    and SEEPS over the scoring region, streaming + DDP-safe.

    Anomalies are relative to the monthly climatological mean
    (``clim_mean (12, H, W)`` from the climatology store). Bins are the
    calendar month of each sample's VALID day, pooled over all validation
    years (one score per month, from every sample of that month).

    SEEPS uses the same p1/t2 climatology and validity range as
    :class:`seeps.StreamingRegionalSEEPS`, so its gridpoint set is the
    region minus points whose climatological ``p1`` is outside
    ``[P1_MIN, P1_MAX]`` — hence a separate weight total.
    """

    def __init__(
        self,
        *,
        bins: dict[int, int],
        climatology: SeepsClimatology,
        region_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.bins = dict(bins)
        n = len(self.bins)
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.clim = climatology.to(device)
        self.clim_mean = self.clim.clim_mean.to(dtype=torch.float32)
        self.sq_sum = torch.zeros(n, device=device)
        self.err_sum = torch.zeros(n, device=device)
        self.s_pt = torch.zeros(n, device=device)
        self.s_pp = torch.zeros(n, device=device)
        self.s_tt = torch.zeros(n, device=device)
        self.w_sum = torch.zeros(n, device=device)
        self.seeps_sum = torch.zeros(n, device=device)
        self.seeps_w_sum = torch.zeros(n, device=device)
        self._p1_valid = (self.clim.p1 >= P1_MIN) & (self.clim.p1 <= P1_MAX)

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
        m = months.long() - 1
        clim = self.clim_mean[m]  # (B, H, W)
        err = pred_mm - t
        p_anom = pred_mm - clim
        t_anom = t - clim
        self.sq_sum[bin_index] += (w * err**2).sum()
        self.err_sum[bin_index] += (w * err).sum()
        self.s_pt[bin_index] += (w * p_anom * t_anom).sum()
        self.s_pp[bin_index] += (w * p_anom**2).sum()
        self.s_tt[bin_index] += (w * t_anom**2).sum()
        self.w_sum[bin_index] += w.sum()

        penalty = seeps_penalty(
            pred_mm,
            target_mm,
            self.clim.p1[m].clamp(1e-6, 1.0 - 1e-6),
            self.clim.t2[m],
            dry_threshold_mm=self.clim.dry_threshold_mm,
        )
        sw = w * self._p1_valid[m].float()
        self.seeps_sum[bin_index] += (penalty * sw).sum()
        self.seeps_w_sum[bin_index] += sw.sum()

    def finalize(self) -> dict[str, torch.Tensor]:
        """Per-bin ``rmse``/``bias``/``acc``/``seeps``; empty bins are NaN."""
        for t in (
            self.sq_sum,
            self.err_sum,
            self.s_pt,
            self.s_pp,
            self.s_tt,
            self.w_sum,
            self.seeps_sum,
            self.seeps_w_sum,
        ):
            _all_reduce_sum(t)
        denom = self.w_sum.clamp(min=1e-12)
        rmse = torch.sqrt(self.sq_sum / denom)
        bias = self.err_sum / denom
        acc = self.s_pt / torch.sqrt(
            self.s_pp.clamp(min=1e-12) * self.s_tt.clamp(min=1e-12)
        )
        seeps = self.seeps_sum / self.seeps_w_sum.clamp(min=1e-12)
        empty = self.w_sum <= 0
        for v in (rmse, bias, acc):
            v[empty] = float("nan")
        seeps[self.seeps_w_sum <= 0] = float("nan")
        return {"rmse": rmse, "bias": bias, "acc": acc, "seeps": seeps}


class MixtureValidator:
    """Scores the gate + baselines over a validation loader.

    Sources scored: ``"gate"``, ``"equal_weight"`` (mean of live experts'
    precip), and each expert by name (only on samples where it is live).

    Every metric uses the SAME ``region_weights`` as the training loss (the
    monsoon box intersected with the IMD-coverage mask) — the gate is only
    supervised there, so scoring anywhere else would measure untrained
    extrapolation. Emitted keys per source: ``rmse_lead{tau}``,
    ``bias_lead{tau}``, ``seeps_lead{tau}`` + ``{rmse,bias,seeps}_mean``
    over leads, and ``imd_{rmse,bias,acc,seeps}_{MM}`` per calendar month
    (pooled over all validation years) + ``imd_{...}_mean``. With
    ``loss_fn`` set, also ``{source}/loss`` and a bare ``loss`` (the gate's),
    the training criterion evaluated on the val split.
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
        monthly: bool | None = None,
        loss_fn=None,
        mix_space: str = "physical",
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
        # The training criterion, evaluated on the val split: emitted as
        # `loss` (the gate's, directly comparable to the logged train loss)
        # and `{source}/loss` for every baseline.
        self.loss_fn = loss_fn
        # Must match training: "physical" mixes experts' mm/day (arithmetic
        # mean), "log" mixes the standardized log channels (geometric mean).
        self.mix_space = str(mix_space)
        # Monthly scores: one bin per calendar month of the valid day,
        # pooled over all validation years.
        # monthly=None: enable when the climatology carries clim_mean.
        has_clim_mean = seeps_climatology.clim_mean is not None
        self.monthly = has_clim_mean if monthly is None else bool(monthly)
        self.month_bins: dict[int, int] = {}
        if self.monthly:
            if not has_clim_mean:
                raise ValueError(
                    "the climatology store lacks clim_mean — regenerate it "
                    "with tools/compute_seeps_climatology.py"
                )
            self.month_bins = {m: m - 1 for m in range(1, 13)}

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
        if self.monthly:
            monthly = {
                s: StreamingMonthlyScores(
                    bins=self.month_bins,
                    climatology=self.seeps_clim,
                    region_weights=self.region_weights,
                    device=self.device,
                )
                for s in self._sources()
            }
        weight_maps: dict = {}
        loss_sums = (
            {s: torch.zeros((), device=self.device) for s in self._sources()}
            if self.loss_fn is not None
            else None
        )
        loss_counts = (
            {s: torch.zeros((), device=self.device) for s in self._sources()}
            if self.loss_fn is not None
            else None
        )

        was_training = model.training
        model.eval()
        for batch in loader:
            x = batch["expert_inputs"].to(self.device, non_blocking=True)
            mask = batch["expert_mask"].to(self.device, non_blocking=True)
            target_mm = batch["target_mm"].to(self.device, non_blocking=True)
            target_mm = target_mm.squeeze(1)
            target_norm = batch["target"].to(self.device, non_blocking=True)
            target_norm = target_norm.squeeze(1)
            taus = batch["lead_days"].to(self.device)
            months = months_from_hours_since_1900(batch["valid_time"]).to(
                self.device
            )

            weights, biases = model(x, mask, taus)
            expert_mm = denormalize_precip(
                x[:, :, 0],
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
            if self.mix_space == "physical":
                pred_norm = None
                pred_mm = mix(weights, biases, expert_mm).clamp(min=0.0)
            else:
                pred_norm = mix(weights, biases, x[:, :, 0])
                pred_mm = denormalize_precip(
                    pred_norm,
                    mean=self.precip_mean,
                    std=self.precip_std,
                    transform=self.precip_transform,
                )
            live = mask > 0
            eq_mm = (expert_mm * live.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            eq_mm = eq_mm / live.sum(dim=1).clamp(min=1).unsqueeze(-1).unsqueeze(-1)

            if loss_sums is not None:
                # Feed every source in the mixture's space, which is the
                # space the loss expects (its pred_space) -- so each source's
                # loss and its RMSE describe the same forecast. The gate's
                # value is byte-for-byte what the training loss sees.
                if self.mix_space == "physical":
                    preds_for_loss = {"gate": pred_mm, "equal_weight": eq_mm}
                    for ei, name in enumerate(self.expert_names):
                        preds_for_loss[name] = expert_mm[:, ei]
                else:
                    preds_for_loss = {
                        "gate": pred_norm,
                        "equal_weight": normalize_precip(
                            eq_mm,
                            mean=self.precip_mean,
                            std=self.precip_std,
                            transform=self.precip_transform,
                        ),
                    }
                    for ei, name in enumerate(self.expert_names):
                        preds_for_loss[name] = x[:, ei, 0]
                for name, pn in preds_for_loss.items():
                    if name in self.expert_names:
                        sel = live[:, self.expert_names.index(name)]
                        if not sel.any():
                            continue
                        pn, tn, tm = pn[sel], target_norm[sel], target_mm[sel]
                    else:
                        tn, tm = target_norm, target_mm
                    n = float(pn.shape[0])
                    loss_sums[name] += self.loss_fn(pn.float(), tn, tm) * n
                    loss_counts[name] += n

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
                for code in months.unique().tolist():
                    bi = self.month_bins.get(int(code))
                    if bi is None:
                        continue
                    sel = months == code
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
                mvals = monthly[s].finalize()
                pooled: dict[str, list[float]] = {k: [] for k in mvals}
                for m, bi in self.month_bins.items():
                    if math.isnan(float(mvals["rmse"][bi])):
                        continue
                    for name, arr in mvals.items():
                        v = float(arr[bi])
                        metrics[f"{s}/imd_{name}_{m:02d}"] = v
                        if not math.isnan(v):
                            pooled[name].append(v)
                for name, vals in pooled.items():
                    if vals:
                        metrics[f"{s}/imd_{name}_mean"] = sum(vals) / len(vals)
        if loss_sums is not None:
            for s in self._sources():
                _all_reduce_sum(loss_sums[s])
                _all_reduce_sum(loss_counts[s])
                if float(loss_counts[s]) > 0:
                    v = float(loss_sums[s] / loss_counts[s])
                    metrics[f"{s}/loss"] = v
                    if s == "gate":
                        # Bare key pairs with the training loss in wandb.
                        metrics["loss"] = v

        if was_training:
            model.train()
        return metrics, {"weight_maps": weight_maps}
