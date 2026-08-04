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

import logging
import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

from losses import (
    almost_fair_crps,
    denormalize_precip,
    normalize_precip,
    pooled_fractions,
    scale_bands,
)
from mowe_precip import mix
from seeps import (
    P1_MAX,
    P1_MIN,
    SeepsClimatology,
    StreamingRegionalSEEPS,
    doy_from_hours_since_1900,
    months_from_hours_since_1900,
    seeps_penalty,
)


logger = logging.getLogger("mowe_validation")


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

    ACC anomalies are relative to the DAY-OF-YEAR climatological mean
    (``clim_mean_daily (366, H, W)``, a +/-7-day smoothed reference). A
    monthly 12-step reference would leave the monsoon onset/withdrawal signal
    in both the forecast and observed anomalies, inflating their correlation;
    it is used only as a fallback for older climatology stores. Bins are the
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
        daily = getattr(self.clim, "clim_mean_daily", None)
        self.clim_daily = None if daily is None else daily.to(dtype=torch.float32)
        self.clim_mean = self.clim.clim_mean.to(dtype=torch.float32)
        if self.clim_daily is None:
            logger.warning(
                "climatology store has no clim_mean_daily; ACC falls back to "
                "the monthly reference, which inflates it -- regenerate with "
                "tools/compute_seeps_climatology.py"
            )
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
        doys: torch.Tensor | None = None,
    ) -> None:
        finite = torch.isfinite(target_mm)
        w = self.weights.unsqueeze(0) * finite.float()
        t = torch.nan_to_num(target_mm)
        m = months.long() - 1
        if self.clim_daily is not None and doys is not None:
            clim = self.clim_daily[doys.long() - 1]  # (B, H, W)
        else:
            clim = self.clim_mean[m]
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
        """Per-bin ``rmse``/``bias``/``acc``/``seeps``/``amp``; empty bins NaN.

        ``amp`` is the anomaly-amplitude ratio sigma_pred / sigma_obs. It
        matters because MSE decomposes as
        ``bias^2 + (sp - st)^2 + 2*sp*st*(1 - r)``, so shrinking the forecast
        anomaly toward zero kills the decorrelation term outright: shrinking
        LOWERS MSE whenever r < 0.5, and the MSE-optimal amplitude is
        ``sp = r * st``. With ACC near 0.29 a pure-MSE objective is therefore
        rewarded for keeping only ~29% of observed variance -- exactly the
        intensity compression this project exists to fix. amp ~ 1 means
        intensity is preserved; amp ~ r means the loss is hedging.
        """
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
        amp = torch.sqrt(
            self.s_pp.clamp(min=0.0) / self.s_tt.clamp(min=1e-12)
        )
        empty = self.w_sum <= 0
        for v in (rmse, bias, acc, amp):
            v[empty] = float("nan")
        seeps[self.seeps_w_sum <= 0] = float("nan")
        return {
            "rmse": rmse,
            "bias": bias,
            "acc": acc,
            "seeps": seeps,
            "amp": amp,
        }


class StreamingThresholdScores:
    """Intensity-resolved scores at fixed daily-rain thresholds, DDP-safe.

    The project's success criterion is stated at moderate-to-heavy rain
    intensities specifically, which aggregate RMSE/ACC cannot show. Per
    threshold T (mm/day) this accumulates region-weighted contingency counts
    and reports

    * ``exc_bias_T`` -- frequency bias, P(pred > T) / P(obs > T). This is the
      direct read on intensity compression: < 1 means the forecast produces
      too few events at that intensity, and it should fall further as T rises
      if the field is over-smoothed.
    * ``csi_T`` -- critical success index hits / (hits + misses + false
      alarms), the usual skill measure at a threshold.
    """

    def __init__(
        self,
        *,
        thresholds: Sequence[float],
        region_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.thresholds = [float(t) for t in thresholds]
        n = len(self.thresholds)
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.hits = torch.zeros(n, device=device)
        self.misses = torch.zeros(n, device=device)
        self.false_alarms = torch.zeros(n, device=device)
        self.pred_yes = torch.zeros(n, device=device)
        self.obs_yes = torch.zeros(n, device=device)

    @torch.no_grad()
    def update(self, pred_mm: torch.Tensor, target_mm: torch.Tensor) -> None:
        finite = torch.isfinite(target_mm)
        w = self.weights.unsqueeze(0) * finite.float()
        t = torch.nan_to_num(target_mm)
        for i, thr in enumerate(self.thresholds):
            p_yes = (pred_mm > thr).float()
            o_yes = (t > thr).float()
            self.hits[i] += (w * p_yes * o_yes).sum()
            self.misses[i] += (w * (1 - p_yes) * o_yes).sum()
            self.false_alarms[i] += (w * p_yes * (1 - o_yes)).sum()
            self.pred_yes[i] += (w * p_yes).sum()
            self.obs_yes[i] += (w * o_yes).sum()

    def finalize(self) -> dict[str, torch.Tensor]:
        for t in (
            self.hits,
            self.misses,
            self.false_alarms,
            self.pred_yes,
            self.obs_yes,
        ):
            _all_reduce_sum(t)
        exc_bias = self.pred_yes / self.obs_yes.clamp(min=1e-12)
        denom = self.hits + self.misses + self.false_alarms
        csi = self.hits / denom.clamp(min=1e-12)
        # A threshold no observation ever reaches carries no information.
        empty = self.obs_yes <= 0
        exc_bias[empty] = float("nan")
        csi[denom <= 0] = float("nan")
        return {"exc_bias": exc_bias, "csi": csi}


class StreamingEnsembleScores:
    """Per-lead fair CRPS, spread/skill and rank histogram, DDP-safe.

    Verification-grade probabilistic scores for an ensemble prediction
    ``(B, N, H, W)``:

    * ``crps`` — the FAIR estimator (``almost_fair_crps`` at alpha = 1),
      unbiased in N, so values at the validation ensemble size are
      comparable across configs regardless of the training-time alpha/N.
    * ``spread_skill`` — ``sqrt((N+1)/N * mean ens variance) / RMSE(ens
      mean)`` (fair-corrected); ~1 for a calibrated ensemble.
    * rank histogram — rank of the observation among the members with
      seeded random jitter tie-breaking. Daily monsoon precip is
      zero-inflated, so exact ties at 0 mm are the NORM, not the exception;
      deterministic tie-breaking would fake a U-shape. The jitter generator
      lives on CPU with a fixed seed so runs are reproducible.
    """

    def __init__(
        self,
        *,
        n_leads: int,
        ens_size: int,
        region_weights: torch.Tensor,
        device: torch.device,
        jitter_seed: int = 0,
    ) -> None:
        if ens_size < 2:
            raise ValueError(f"ens_size must be >= 2, got {ens_size}")
        self.ens_size = int(ens_size)
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.crps_sum = torch.zeros(n_leads, device=device)
        self.var_sum = torch.zeros(n_leads, device=device)
        self.msqe_sum = torch.zeros(n_leads, device=device)
        self.w_sum = torch.zeros(n_leads, device=device)
        self.rank_counts = torch.zeros(n_leads, ens_size + 1, device=device)
        self._gen = torch.Generator()
        self._gen.manual_seed(int(jitter_seed))

    @torch.no_grad()
    def update(
        self, lead_index: int, pred_ens_mm: torch.Tensor, target_mm: torch.Tensor
    ) -> None:
        if pred_ens_mm.ndim != 4 or pred_ens_mm.shape[1] != self.ens_size:
            raise ValueError(
                f"expected (B, {self.ens_size}, H, W), got {tuple(pred_ens_mm.shape)}"
            )
        finite = torch.isfinite(target_mm)
        w = self.weights.unsqueeze(0) * finite.float()
        obs = torch.nan_to_num(target_mm)
        crps = almost_fair_crps(pred_ens_mm, obs, alpha=1.0)
        self.crps_sum[lead_index] += (w * crps).sum()
        self.w_sum[lead_index] += w.sum()
        mean = pred_ens_mm.mean(dim=1)
        var = pred_ens_mm.var(dim=1, unbiased=True)
        self.var_sum[lead_index] += (w * var).sum()
        self.msqe_sum[lead_index] += (w * (mean - obs) ** 2).sum()
        # 1e-4 mm/day jitter: far below any meteorological difference, far
        # above float noise -- randomizes only genuine ties (mostly 0 mm).
        jm = torch.rand(pred_ens_mm.shape, generator=self._gen)
        jo = torch.rand(obs.shape, generator=self._gen)
        m_j = pred_ens_mm + 1e-4 * jm.to(pred_ens_mm.device)
        o_j = (obs + 1e-4 * jo.to(obs.device)).unsqueeze(1)
        rank = (m_j < o_j).sum(dim=1)  # (B, H, W) in 0..N
        for r in range(self.ens_size + 1):
            self.rank_counts[lead_index, r] += (w * (rank == r).float()).sum()

    def finalize(self) -> dict[str, torch.Tensor]:
        for t in (
            self.crps_sum,
            self.var_sum,
            self.msqe_sum,
            self.w_sum,
            self.rank_counts,
        ):
            _all_reduce_sum(t)
        denom = self.w_sum.clamp(min=1e-12)
        crps = self.crps_sum / denom
        n = self.ens_size
        spread = torch.sqrt((n + 1) / n * self.var_sum / denom)
        skill = torch.sqrt(self.msqe_sum / denom)
        spread_skill = spread / skill.clamp(min=1e-12)
        empty = self.w_sum <= 0
        for v in (crps, spread_skill):
            v[empty] = float("nan")
        pooled = self.rank_counts.sum(dim=0)
        pooled = pooled / pooled.sum().clamp(min=1e-12)
        uniform = 1.0 / (n + 1)
        rank_hist_dev = 0.5 * (pooled - uniform).abs().sum()
        return {
            "crps": crps,
            "spread_skill": spread_skill,
            "rank_hist": pooled,
            "rank_hist_dev": rank_hist_dev,
        }


class StreamingFSS:
    """Hard-threshold fractions skill score, weight-masked pooling, DDP-safe.

    The verification-grade counterpart of the FSS loss term (same
    ``pooled_fractions`` and the same threshold maps, but a HARD exceedance
    indicator): per (window k, threshold q),
    ``FSS = 1 - sum w (f_p - f_o)^2 / sum w (f_p^2 + f_o^2)`` accumulated
    over all batches. NaN when the reference sum is 0 (threshold never
    reached by either field).
    """

    def __init__(
        self,
        *,
        windows: Sequence[int],
        threshold_maps: torch.Tensor,
        threshold_labels: Sequence[str],
        region_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.windows = [int(k) for k in windows]
        self.labels = [str(s) for s in threshold_labels]
        self.maps = threshold_maps.to(device=device, dtype=torch.float32)
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        k, q = len(self.windows), len(self.labels)
        self.fbs = torch.zeros(k, q, device=device)
        self.fbs_ref = torch.zeros(k, q, device=device)

    @torch.no_grad()
    def update(self, pred_mm: torch.Tensor, target_mm: torch.Tensor) -> None:
        finite = torch.isfinite(target_mm)
        obs = torch.nan_to_num(target_mm)
        for qi in range(self.maps.shape[0]):
            thr = self.maps[qi]
            b_p = (pred_mm > thr).float()
            b_o = (obs > thr).float()
            for ki, k in enumerate(self.windows):
                f_p, w_pool = pooled_fractions(b_p, self.weights, finite, k)
                f_o, _ = pooled_fractions(b_o, self.weights, finite, k)
                self.fbs[ki, qi] += (w_pool * (f_p - f_o) ** 2).sum()
                self.fbs_ref[ki, qi] += (w_pool * (f_p**2 + f_o**2)).sum()

    def finalize(self) -> dict[str, float]:
        _all_reduce_sum(self.fbs)
        _all_reduce_sum(self.fbs_ref)
        out: dict[str, float] = {}
        for ki, k in enumerate(self.windows):
            for qi, label in enumerate(self.labels):
                ref = float(self.fbs_ref[ki, qi])
                val = 1.0 - float(self.fbs[ki, qi]) / ref if ref > 0 else float("nan")
                out[f"fss_w{k}_{label}"] = val
        return out


class StreamingBandAmp:
    """Per-scale-band amplitude ratio sigma_pred / sigma_obs, DDP-safe.

    The regional effective-resolution diagnostic: the same pooling-band
    decomposition as the AMSE loss (``losses.scale_bands``), reporting how
    much of the observed variance the forecast keeps at each scale. ~1
    everywhere = intensity preserved; falling with band fineness = the
    classic MSE blur signature. Reported for every arm so loss ablations
    are comparable.
    """

    def __init__(
        self,
        *,
        windows: Sequence[int],
        region_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.windows = [int(k) for k in windows]
        n_bands = len(self.windows) + 1
        self.weights = region_weights.to(device=device, dtype=torch.float32)
        self.sp = torch.zeros(n_bands, device=device)
        self.st = torch.zeros(n_bands, device=device)
        self.w_sum = torch.zeros(n_bands, device=device)

    @torch.no_grad()
    def update(self, pred_mm: torch.Tensor, target_mm: torch.Tensor) -> None:
        finite = torch.isfinite(target_mm)
        obs = torch.nan_to_num(target_mm)
        w = self.weights.unsqueeze(0) * finite.float()
        wsum = w.sum(dim=(-2, -1)).clamp(min=1e-12)

        def wmean(f):
            return (f * w).sum(dim=(-2, -1)) / wsum

        p_bands = scale_bands(pred_mm, self.weights, finite, self.windows)
        o_bands = scale_bands(obs, self.weights, finite, self.windows)
        for bi, (bp, bo) in enumerate(zip(p_bands, o_bands)):
            bp = bp - wmean(bp)[:, None, None]
            bo = bo - wmean(bo)[:, None, None]
            self.sp[bi] += (w * bp**2).sum()
            self.st[bi] += (w * bo**2).sum()
            self.w_sum[bi] += w.sum()

    def finalize(self) -> torch.Tensor:
        for t in (self.sp, self.st, self.w_sum):
            _all_reduce_sum(t)
        amp = torch.sqrt(self.sp.clamp(min=0.0) / self.st.clamp(min=1e-12))
        amp[self.w_sum <= 0] = float("nan")
        return amp


class MixtureValidator:
    """Scores the gate + baselines over a validation loader.

    Sources scored: ``"gate"``, ``"equal_weight"`` (mean of live experts'
    precip), and each expert by name (only on samples where it is live).

    Every metric uses the SAME ``region_weights`` as the training loss (the
    monsoon box intersected with the IMD-coverage mask) — the gate is only
    supervised there, so scoring anywhere else would measure untrained
    extrapolation. Emitted keys per source: ``rmse_lead{tau}``,
    ``bias_lead{tau}``, ``seeps_lead{tau}`` + ``{rmse,bias,seeps}_mean``
    over leads, and ``imd_{rmse,bias,acc,seeps,amp}_{MM}`` per calendar month
    (pooled over all validation years) + ``imd_{...}_mean``. With
    ``loss_fn`` set, also ``{source}/loss`` and a bare ``loss`` (the gate's),
    the training criterion evaluated on the val split.

    Additive metrics (never renaming existing keys):
    ``{source}/amp_band{i}`` banded amplitude ratios always; with FSS
    threshold maps, ``{source}/fss_w{k}_{label}``; in ensemble mode
    (``ens_size`` + ``noise_dim`` set), ``gate/crps_lead{tau}`` /
    ``gate/crps_mean`` (fair), ``gate/spread_skill_*``,
    ``gate/rank_hist_{i}`` + ``gate/rank_hist_dev``, and
    ``expert_ensemble/crps_mean`` (the equal-weight all-live-expert
    ensemble — the CRPSS baseline).
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
        thresholds: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 50.0),
        ens_size: int = 0,
        noise_dim: int | None = None,
        noise_seed: int = 0,
        fss_windows: Sequence[int] | None = None,
        fss_threshold_maps: torch.Tensor | None = None,
        fss_threshold_labels: Sequence[str] | None = None,
        amp_band_windows: Sequence[int] = (3, 7),
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
        # Daily-rain thresholds (mm/day) for the intensity-resolved scores.
        self.thresholds = [float(t) for t in thresholds]
        # Ensemble mode: with the noise-conditioned gate, draw a fixed-seed
        # ensemble per batch, route the ENSEMBLE MEAN into every existing
        # deterministic accumulator (zero metric-key churn) and add the
        # probabilistic scores on top. The gate's monitored `loss` is the
        # training criterion on the FULL ensemble (afCRPS at this ens_size:
        # the fair estimator is unbiased in N, so the val value is comparable
        # across ensemble sizes, just lower-variance than the train one).
        self.ensemble = bool(ens_size and noise_dim)
        self.ens_size = int(ens_size)
        self.noise_dim = None if noise_dim is None else int(noise_dim)
        self.noise_seed = int(noise_seed)
        # Hard-threshold FSS metric (all arms when threshold maps are given).
        self.fss_windows = None if fss_windows is None else [int(k) for k in fss_windows]
        self.fss_threshold_maps = fss_threshold_maps
        self.fss_threshold_labels = (
            None if fss_threshold_labels is None else [str(s) for s in fss_threshold_labels]
        )
        self.amp_band_windows = [int(k) for k in amp_band_windows]
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
        thresh = (
            {
                s: StreamingThresholdScores(
                    thresholds=self.thresholds,
                    region_weights=self.region_weights,
                    device=self.device,
                )
                for s in self._sources()
            }
            if self.thresholds
            else None
        )
        fss = None
        if self.fss_windows and self.fss_threshold_maps is not None:
            fss = {
                s: StreamingFSS(
                    windows=self.fss_windows,
                    threshold_maps=self.fss_threshold_maps,
                    threshold_labels=self.fss_threshold_labels,
                    region_weights=self.region_weights,
                    device=self.device,
                )
                for s in self._sources()
            }
        band_amp = {
            s: StreamingBandAmp(
                windows=self.amp_band_windows,
                region_weights=self.region_weights,
                device=self.device,
            )
            for s in self._sources()
        }
        ens_scores = None
        expert_ens_scores = None
        if self.ensemble:
            ens_scores = StreamingEnsembleScores(
                n_leads=self.n_leads,
                ens_size=self.ens_size,
                region_weights=self.region_weights,
                device=self.device,
                jitter_seed=self.noise_seed,
            )
            # CRPSS baseline: the live experts as an equal-weight ensemble,
            # restricted to samples where all experts are live so the member
            # count is fixed (noted in MOWE.md).
            if len(self.expert_names) >= 2:
                expert_ens_scores = StreamingEnsembleScores(
                    n_leads=self.n_leads,
                    ens_size=len(self.expert_names),
                    region_weights=self.region_weights,
                    device=self.device,
                    jitter_seed=self.noise_seed,
                )
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
        batch_counter = 0
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
            doys = doy_from_hours_since_1900(batch["valid_time"]).to(self.device)

            if self.ensemble:
                # CPU generator, fixed seed per batch: the noise (and hence
                # every probabilistic score) is identical across epochs and
                # independent of device, so spread/CRPS curves are
                # comparable epoch to epoch.
                g = torch.Generator()
                g.manual_seed(self.noise_seed * 1_000_003 + batch_counter)
                noise = torch.randn(
                    x.shape[0], self.ens_size, self.noise_dim, generator=g
                ).to(self.device)
                weights, biases = model(x, mask, taus, noise)
            else:
                weights, biases = model(x, mask, taus)
            batch_counter += 1
            expert_mm = denormalize_precip(
                x[:, :, 0],
                mean=self.precip_mean,
                std=self.precip_std,
                transform=self.precip_transform,
            )
            # In ensemble mode pred_* carries a member axis (B, N, H, W);
            # the ENSEMBLE MEAN feeds every deterministic accumulator below,
            # so metric keys and semantics match the deterministic arms.
            if self.mix_space == "physical":
                pred_norm = None
                pred_full = mix(weights, biases, expert_mm, mask=mask).clamp(min=0.0)
            else:
                pred_norm = mix(weights, biases, x[:, :, 0], mask=mask)
                pred_full = denormalize_precip(
                    pred_norm,
                    mean=self.precip_mean,
                    std=self.precip_std,
                    transform=self.precip_transform,
                )
            if self.ensemble:
                pred_ens_mm = pred_full
                pred_mm = pred_full.mean(dim=1)
            else:
                pred_ens_mm = None
                pred_mm = pred_full
            live = mask > 0
            eq_mm = (expert_mm * live.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            eq_mm = eq_mm / live.sum(dim=1).clamp(min=1).unsqueeze(-1).unsqueeze(-1)

            if loss_sums is not None:
                # Feed every source in the mixture's space, which is the
                # space the loss expects (its pred_space) -- so each source's
                # loss and its RMSE describe the same forecast. The gate's
                # value is byte-for-byte what the training loss sees.
                # The gate's loss sees the FULL ensemble (the training
                # criterion); deterministic sources score as 1-member
                # ensembles under a CRPS loss (their regional MAE), which is
                # exactly the CRPS of a deterministic forecast.
                gate_pred = pred_ens_mm if self.ensemble else pred_mm
                if self.mix_space == "physical":
                    preds_for_loss = {"gate": gate_pred, "equal_weight": eq_mm}
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
                if ens_scores is not None:
                    ens_scores.update(li, pred_ens_mm[sel], t_mm)
                if expert_ens_scores is not None:
                    all_live = sel & (live.sum(dim=1) == len(self.expert_names))
                    if all_live.any():
                        expert_ens_scores.update(
                            li, expert_mm[all_live], target_mm[all_live]
                        )
                key = f"weights_lead{int(tau)}"
                if len(weight_maps) < self.n_weight_map_samples and key not in weight_maps:
                    wm = weights[sel][0]
                    if wm.ndim == 4:  # (ens, E, H, W) -> member 0
                        wm = wm[0]
                    weight_maps[key] = wm.float().cpu().numpy()

            if thresh is not None:
                thresh["gate"].update(pred_mm, target_mm)
                thresh["equal_weight"].update(eq_mm, target_mm)
                for ei, name in enumerate(self.expert_names):
                    esel = live[:, ei]
                    if esel.any():
                        thresh[name].update(expert_mm[esel, ei], target_mm[esel])

            for acc in (fss, band_amp):
                if acc is None:
                    continue
                acc["gate"].update(pred_mm, target_mm)
                acc["equal_weight"].update(eq_mm, target_mm)
                for ei, name in enumerate(self.expert_names):
                    esel = live[:, ei]
                    if esel.any():
                        acc[name].update(expert_mm[esel, ei], target_mm[esel])

            if monthly is not None:
                for code in months.unique().tolist():
                    bi = self.month_bins.get(int(code))
                    if bi is None:
                        continue
                    sel = months == code
                    t_mm = target_mm[sel]
                    m_sel = months[sel]
                    d_sel = doys[sel]
                    monthly["gate"].update(bi, pred_mm[sel], t_mm, m_sel, d_sel)
                    monthly["equal_weight"].update(bi, eq_mm[sel], t_mm, m_sel, d_sel)
                    for ei, name in enumerate(self.expert_names):
                        esel = sel & live[:, ei]
                        if not esel.any():
                            continue
                        monthly[name].update(
                            bi,
                            expert_mm[esel, ei],
                            target_mm[esel],
                            months[esel],
                            doys[esel],
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
        if thresh is not None:
            for s in self._sources():
                tv = thresh[s].finalize()
                for i, t in enumerate(self.thresholds):
                    tag = f"{t:g}".replace(".", "p")
                    for name, arr in tv.items():
                        val = float(arr[i])
                        if not math.isnan(val):
                            metrics[f"{s}/{name}_{tag}mm"] = val
        if fss is not None:
            for s in self._sources():
                for key, val in fss[s].finalize().items():
                    if not math.isnan(val):
                        metrics[f"{s}/{key}"] = val
        for s in self._sources():
            amp = band_amp[s].finalize()
            for bi in range(amp.numel()):
                val = float(amp[bi])
                if not math.isnan(val):
                    metrics[f"{s}/amp_band{bi}"] = val
        if ens_scores is not None:
            ev = ens_scores.finalize()
            for li in range(self.n_leads):
                tau = self.lead_lo + li
                metrics[f"gate/crps_lead{tau}"] = float(ev["crps"][li])
                metrics[f"gate/spread_skill_lead{tau}"] = float(
                    ev["spread_skill"][li]
                )
            # NaN-aware: a lead with no samples must not blank the mean the
            # success criteria (CRPSS, spread-skill window) are read from.
            metrics["gate/crps_mean"] = float(ev["crps"].nanmean())
            metrics["gate/spread_skill_mean"] = float(ev["spread_skill"].nanmean())
            metrics["gate/rank_hist_dev"] = float(ev["rank_hist_dev"])
            for i in range(ev["rank_hist"].numel()):
                metrics[f"gate/rank_hist_{i}"] = float(ev["rank_hist"][i])
        if expert_ens_scores is not None:
            ev = expert_ens_scores.finalize()
            metrics["expert_ensemble/crps_mean"] = float(ev["crps"].nanmean())
            metrics["expert_ensemble/spread_skill_mean"] = float(
                ev["spread_skill"].nanmean()
            )

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
