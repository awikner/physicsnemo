#!/usr/bin/env python
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

"""Numbers behind the week-2 ACC / RMSE panel figures. No plotting here.

Called in order by ``plot_week2_acc_panels.py``; figures live in
``acc_panels_plots.py``. Everything accumulates in float64.

See ``README.md`` in this folder for the workflow, the diagram and the
reasoning behind each step.
"""

from __future__ import annotations

import datetime
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

# The recipe root (examples/weather/ai_rossbypalooza) holds `datapipes` and
# `losses`. Same bootstrap the sibling tools use, repeated in each module of
# this folder so any one of them can be imported on its own.
_RECIPE_DIR = Path(__file__).resolve().parents[2]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

# the modules are imported here not before because the system path had to be set.
# "noqa: E402" is added because the reader complains about this choice of placement
from datapipes.sampler import MixturePairSampler  # noqa: E402
from losses import (  # noqa: E402
    denormalize_precip,
    imd_valid_mask,
    region_weights,
)

logger = logging.getLogger("week2_acc_panels")

# The two derived sources that are not one of the harmonized expert archives.
EQUAL_WEIGHT = "equal_weight"
GATE = "gate"

# Display names, used for log lines and panel titles alike so the figure and
# the log always call a source the same thing.
SOURCE_LABELS = {
    "pangu_s2s": "Pangu-S2S",
    "sfno_era5": "SFNO-S2S",
    "graphcast": "GraphCast",
    "aifs_single_v2": "AIFS",
    EQUAL_WEIGHT: "Equal weight",
    GATE: "MoWE gate",
}


def panel_source_order(expert_names: list[str]) -> list[str]:
    """The six sources in the order the panels are drawn.

    Experts first (in datapipe order), then the two blends, so a reader scans
    the individual models before the things built out of them.
    """
    return [*expert_names, EQUAL_WEIGHT, GATE]


# --------------------------------------------------------------------------
# 1. The scored region
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredRegion:
    """The gridpoints that get scored, plus the box that contains them.

    Attributes
    ----------
    rows, cols : slices cropping the global 180x360 grid to the bounding box.
    mask : ``(n_lat, n_lon)`` bool, True where the gridpoint is scored.
    lat, lon : the cropped coordinate values, for the map axes.
    """

    rows: slice
    cols: slice
    mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the cropped bounding box, ``(n_lat, n_lon)``."""
        return self.mask.shape

    @property
    def n_scored_points(self) -> int:
        """How many gridpoints inside the box are actually scored."""
        return int(self.mask.sum())


def build_region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    box: tuple[float, float, float, float],
    imd_store: str | None,
    min_finite_frac: float,
) -> np.ndarray:
    """Boolean ``(lat, lon)`` mask: inside the box and (optionally) IMD-gauged.

    Reuses ``losses.region_weights``/``imd_valid_mask`` so the scored region
    matches training exactly. Only the ``> 0`` pattern is used; the
    cos-latitude values are discarded.
    """
    extra = None
    if imd_store:
        extra = imd_valid_mask(imd_store, lat, lon, min_finite_frac=min_finite_frac)
    weights_2d = region_weights(lat, lon, box, extra_mask=extra)
    return (weights_2d > 0).numpy()


def select_scored_region(
    lat: np.ndarray,
    lon: np.ndarray,
    box: tuple[float, float, float, float],
    imd_store: str | None,
    min_finite_frac: float,
) -> ScoredRegion:
    """Build the region mask and the tightest bounding box that holds it."""
    mask = build_region_mask(lat, lon, box, imd_store, min_finite_frac)
    if not mask.any():
        raise ValueError(f"empty region mask for box {box} and IMD {imd_store}")

    rows_with_data = np.where(mask.any(axis=1))[0]
    cols_with_data = np.where(mask.any(axis=0))[0]
    rows = slice(rows_with_data.min(), rows_with_data.max() + 1)
    cols = slice(cols_with_data.min(), cols_with_data.max() + 1)

    region = ScoredRegion(
        rows=rows,
        cols=cols,
        mask=mask[rows, cols],
        lat=np.asarray(lat)[rows],
        lon=np.asarray(lon)[cols],
    )
    logger.info(
        "region: %d scored gridpoints inside a %dx%d box "
        "(lat %.1f..%.1f, lon %.1f..%.1f)",
        region.n_scored_points,
        region.shape[0],
        region.shape[1],
        region.lat.min(),
        region.lat.max(),
        region.lon.min(),
        region.lon.max(),
    )
    return region


# --------------------------------------------------------------------------
# 2. Climatology and 3. the gate forecast
# --------------------------------------------------------------------------


def load_daily_climatology(climatology_store: Path, region: ScoredRegion) -> np.ndarray:
    """Day-of-year mean precipitation, ``(366, n_lat, n_lon)`` in mm/day.

    Rejects a store carrying only the monthly ``clim_mean`` -- see README §1.
    """
    with xr.open_zarr(climatology_store, consolidated=True) as clim_ds:
        if "clim_mean_daily" not in clim_ds:
            raise ValueError(
                f"{climatology_store} has no clim_mean_daily; regenerate with "
                "tools/compute_seeps_climatology.py"
            )
        return (
            clim_ds["clim_mean_daily"]
            .isel(lat=region.rows, lon=region.cols)
            .values.astype("float64")
        )


def load_gate_weekly_totals(
    forecast_store: Path, precip_var: str, region: ScoredRegion
) -> tuple[dict[pd.Timestamp, np.ndarray], np.ndarray]:
    """Week-2 totals for the MoWE gate, keyed by initialization date.

    Straight off the inference zarr -- no checkpoint, no datapipe, no GPU.

    Returns
    -------
    (weekly_totals, lead_days)
        init date -> ``(n_lat, n_lon)`` mm/week, and the summed lead axis.
    """
    forecast_ds = xr.open_zarr(forecast_store, consolidated=True)
    lead_days = forecast_ds["lead_time"].values.astype(int)
    init_dates = pd.DatetimeIndex(forecast_ds["init_time"].values).normalize()
    weekly_totals = (
        forecast_ds[precip_var]
        .isel(lat=region.rows, lon=region.cols)
        # skipna=False: a NaN lead must poison the whole weekly total.
        .sum("lead_time", skipna=False)
        .values.astype("float64")
    )
    logger.info(
        "gate zarr: %d inits %s..%s, leads %d-%d summed to weekly totals",
        init_dates.size,
        init_dates[0].date(),
        init_dates[-1].date(),
        lead_days.min(),
        lead_days.max(),
    )
    return {date: weekly_totals[k] for k, date in enumerate(init_dates)}, lead_days


# --------------------------------------------------------------------------
# 4. Experts, equal-weight, truth and climatology through the datapipe
# --------------------------------------------------------------------------


@dataclass
class WeeklyTotals:
    """Week-2 accumulated precipitation, one row per initialization.

    Every array is indexed by a compact week index ``k``; ``init_dates[k]``
    is the date of week ``k``.

    Attributes
    ----------
    expert_names : the harmonized expert archives, in datapipe order.
    init_dates : ``(n_weeks,)`` init dates (None if never seen).
    forecast_sums : source -> ``(n_weeks, n_lat, n_lon)`` mm/week.
    live_lead_counts : source -> ``(n_weeks,)`` leads that source contributed;
        a source is comparable only where this equals ``n_leads_per_week``.
    observed_sum : ``(n_weeks, n_lat, n_lon)`` IMERG mm/week.
    observed_lead_count : ``(n_weeks,)`` leads of truth accumulated.
    climatology_sum : ``(n_weeks, n_lat, n_lon)`` day-of-year climatology
        summed over the same seven valid days.
    n_leads_per_week : leads in a complete week (7 for leads 8-14).
    """

    expert_names: list[str]
    init_dates: list[pd.Timestamp | None]
    forecast_sums: dict[str, np.ndarray]
    live_lead_counts: dict[str, np.ndarray]
    observed_sum: np.ndarray
    observed_lead_count: np.ndarray
    climatology_sum: np.ndarray
    n_leads_per_week: int

    @property
    def n_weeks(self) -> int:
        """Number of initializations in the table."""
        return len(self.init_dates)

    @property
    def datapipe_source_names(self) -> list[str]:
        """The sources this table carries (experts plus equal-weight)."""
        return [*self.expert_names, EQUAL_WEIGHT]


def _equal_weight_blend(
    expert_precip_mm: torch.Tensor, expert_is_live: torch.Tensor
) -> torch.Tensor:
    """Plain average over the LIVE experts only, ``(B, E, H, W) -> (B, H, W)``.

    Dividing by the live count, not by E -- see README §2. ``clamp(min=1)``
    only guards a division by zero the datapipe already prevents.
    """
    live_weight = expert_is_live.float()[..., None, None]
    live_count = expert_is_live.sum(dim=1).clamp(min=1)[:, None, None]
    return (expert_precip_mm * live_weight).sum(dim=1) / live_count


def _init_date_of_row(dataset, init_row: int) -> pd.Timestamp:
    """Calendar date of one initialization row of the datapipe index.

    ``init_keys`` entries are ``(year, month, day, hour)`` 4-tuples; inits are
    00Z, so the hour is dropped to match the zarr's normalized init axis.
    """
    year, month, day = (int(v) for v in dataset.index.init_keys[init_row][:3])
    return pd.Timestamp(year, month, day)


def accumulate_weekly_totals(
    dataset,
    region: ScoredRegion,
    climatology_daily: np.ndarray,
    n_leads_per_week: int,
    batch_size: int,
    num_workers: int,
) -> WeeklyTotals:
    """Sum every (init, lead) sample of the datapipe into weekly totals.

    One pass over the validation split, adding each source's field into the
    row for its init, once per lead. Experts come through the datapipe rather
    than the stores because it owns ``day_offset`` and the live mask --
    README §2.
    """
    expert_names = list(dataset.expert_names)
    datapipe_sources = [*expert_names, EQUAL_WEIGHT]

    # The pair table lists (init_row, lead) pairs, so the same init_row shows
    # up once per lead. Compress the sparse init_row values to 0..n_weeks-1.
    init_rows = np.unique(dataset.pairs["init_row"])
    init_row_to_week = {int(row): k for k, row in enumerate(init_rows)}
    n_weeks = len(init_rows)
    week_field_shape = (n_weeks, region.shape[0], region.shape[1])

    forecast_sums = {name: np.zeros(week_field_shape) for name in datapipe_sources}
    live_lead_counts = {name: np.zeros(n_weeks, dtype=int) for name in datapipe_sources}
    observed_sum = np.zeros(week_field_shape)
    observed_lead_count = np.zeros(n_weeks, dtype=int)
    climatology_sum = np.zeros(week_field_shape)
    init_dates: list[pd.Timestamp | None] = [None] * n_weeks

    logger.info(
        "accumulating %d datapipe samples over %d inits and %d experts",
        len(dataset),
        n_weeks,
        len(expert_names),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=False: a sum is order-independent, but a deterministic
        # sweep keeps the progress log comparable between runs.
        sampler=MixturePairSampler(len(dataset), shuffle=False),
        num_workers=num_workers,
    )

    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            # (B, E, 1 + C, H, W); channel 0 is precip, normalized in log
            # space, so undo that before anything is summed.
            expert_precip_mm = denormalize_precip(
                batch["expert_inputs"][:, :, 0],
                mean=dataset.precip_mean,
                std=dataset.precip_std,
                transform=dataset.precip_transform,
            )
            expert_is_live = batch["expert_mask"] > 0
            equal_weight_precip_mm = _equal_weight_blend(
                expert_precip_mm, expert_is_live
            )

            # Crop to the bounding box once, then stay in numpy.
            expert_precip_region = expert_precip_mm.numpy()[
                ..., region.rows, region.cols
            ]
            equal_weight_region = equal_weight_precip_mm.numpy()[
                ..., region.rows, region.cols
            ]
            observed_region = (
                batch["target_mm"].squeeze(1).numpy()[..., region.rows, region.cols]
            )
            expert_is_live_np = expert_is_live.numpy()

            for sample in range(expert_precip_region.shape[0]):
                pair_row = dataset.pairs[int(batch["pair_idx"][sample])]
                init_row = int(pair_row["init_row"])
                week = init_row_to_week[init_row]
                lead_day = int(batch["lead_days"][sample])

                if init_dates[week] is None:
                    init_dates[week] = _init_date_of_row(dataset, init_row)

                # An expert contributes only where live, so its count can
                # end below n_leads_per_week; equal-weight always counts.
                for expert_number, expert_name in enumerate(expert_names):
                    if expert_is_live_np[sample, expert_number]:
                        forecast_sums[expert_name][week] += expert_precip_region[
                            sample, expert_number
                        ]
                        live_lead_counts[expert_name][week] += 1

                forecast_sums[EQUAL_WEIGHT][week] += equal_weight_region[sample]
                live_lead_counts[EQUAL_WEIGHT][week] += 1

                observed_sum[week] += observed_region[sample]
                observed_lead_count[week] += 1

                # index.py: the record for lead tau sits at
                # date(init) + (tau - 1), NOT at valid_time -- README §2.
                record_date = init_dates[week] + datetime.timedelta(days=lead_day - 1)
                climatology_sum[week] += climatology_daily[record_date.dayofyear - 1]

            if batch_number % 100 == 0:
                logger.info("  batch %d", batch_number)

    return WeeklyTotals(
        expert_names=expert_names,
        init_dates=init_dates,
        forecast_sums=forecast_sums,
        live_lead_counts=live_lead_counts,
        observed_sum=observed_sum,
        observed_lead_count=observed_lead_count,
        climatology_sum=climatology_sum,
        n_leads_per_week=n_leads_per_week,
    )


# --------------------------------------------------------------------------
# 5. Which weeks are scored
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredWeeks:
    """Which initializations enter the score.

    Attributes
    ----------
    keep : ``(n_weeks,)`` bool; matched mode = weeks every source completes,
        unmatched mode = weeks with truth and a gate forecast.
    complete_per_source : source -> ``(n_weeks,)`` bool, all leads present.
    matched : the mode this selection was built in.
    """

    keep: np.ndarray
    complete_per_source: dict[str, np.ndarray]
    matched: bool

    @property
    def n_weeks(self) -> int:
        """How many weeks the matched selection keeps."""
        return int(self.keep.sum())

    def weeks_for(self, source: str) -> np.ndarray:
        """The weeks a given source is scored on.

        Matched mode: the same weeks for everyone. Unmatched: that source's
        own complete weeks.
        """
        if self.matched:
            return self.keep
        return self.keep & self.complete_per_source.get(source, np.ones_like(self.keep))


def find_scored_weeks(
    totals: WeeklyTotals,
    gate_weekly_totals: dict[pd.Timestamp, np.ndarray],
    matched: bool,
) -> ScoredWeeks:
    """Decide which initializations every source is scored on.

    Matched mode (the default) keeps only the weeks where EVERY source has
    all its leads, because expert coverage is badly uneven and scoring each
    on its own weeks flips the headline result -- README §4.
    """
    complete_per_source = {
        source: totals.live_lead_counts[source] == totals.n_leads_per_week
        for source in totals.datapipe_source_names
    }
    has_observations = totals.observed_lead_count == totals.n_leads_per_week
    has_gate = np.array(
        [
            date is not None
            and date in gate_weekly_totals
            and np.isfinite(gate_weekly_totals[date]).all()
            for date in totals.init_dates
        ]
    )
    complete_per_source[GATE] = has_gate

    for source in totals.datapipe_source_names:
        logger.info(
            "complete weeks  %-16s %d",
            SOURCE_LABELS[source],
            int(complete_per_source[source].sum()),
        )
    logger.info("complete weeks  %-16s %d", SOURCE_LABELS[GATE], int(has_gate.sum()))

    keep = has_observations & has_gate
    if matched:
        for source in totals.datapipe_source_names:
            keep = keep & complete_per_source[source]
        logger.info(
            "matched: %d of %d weeks have every source complete",
            int(keep.sum()),
            totals.n_weeks,
        )
    else:
        logger.info("unmatched: %d weeks with obs + gate", int(keep.sum()))

    if keep.sum() < 2:
        raise ValueError(
            f"only {int(keep.sum())} complete weeks; need at least 2 to correlate"
        )
    return ScoredWeeks(
        keep=keep, complete_per_source=complete_per_source, matched=matched
    )


# --------------------------------------------------------------------------
# 6. The two metrics, and scoring every source with them
# --------------------------------------------------------------------------


def anomaly_correlation_map(
    forecast_anomaly: np.ndarray, truth_anomaly: np.ndarray
) -> np.ndarray:
    """Per-gridpoint ACC over the sample (week) axis.

    Uncentred (classic) ACC, and deliberately unweighted by cos-latitude --
    README §1.
    """
    numerator = (forecast_anomaly * truth_anomaly).sum(axis=0)
    denominator = np.sqrt(
        (forecast_anomaly * forecast_anomaly).sum(axis=0)
        * (truth_anomaly * truth_anomaly).sum(axis=0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def root_mean_square_error_map(
    forecast: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Per-gridpoint RMSE of the weekly total, mm/week.

    On the RAW fields, not anomalies: the climatology cancels in a difference
    and is deliberately not applied. Unweighted, as for ACC.
    """
    error = forecast - observed
    is_finite = np.isfinite(error)
    n_samples = is_finite.sum(axis=0)
    sum_squared_error = np.where(is_finite, error * error, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            n_samples > 0,
            np.sqrt(sum_squared_error / np.maximum(n_samples, 1)),
            np.nan,
        )


def _gate_weekly_stack(
    totals: WeeklyTotals, gate_weekly_totals: dict[pd.Timestamp, np.ndarray]
) -> np.ndarray:
    """Gate totals as a ``(n_weeks, n_lat, n_lon)`` array aligned to ``totals``.

    NaN-filled, so an init missing from the zarr drops out via ``has_gate``
    rather than raising a KeyError here.
    """
    missing = np.full(totals.climatology_sum.shape[1:], np.nan)
    return np.stack(
        [gate_weekly_totals.get(date, missing) for date in totals.init_dates]
    )


def score_every_source(
    totals: WeeklyTotals,
    gate_weekly_totals: dict[pd.Timestamp, np.ndarray],
    region: ScoredRegion,
    scored_weeks: ScoredWeeks,
    metric: str,
) -> dict[str, np.ndarray]:
    """One ``(n_lat, n_lon)`` skill map per source, masked to the region.

    ``acc`` correlates anomalies against the weekly climatology; ``rmse``
    uses the raw weekly totals.
    """
    truth_anomaly_all_weeks = totals.observed_sum - totals.climatology_sum
    gate_weekly_stack = _gate_weekly_stack(totals, gate_weekly_totals)

    scores: dict[str, np.ndarray] = {}
    for source in panel_source_order(totals.expert_names):
        forecast_totals = (
            gate_weekly_stack if source == GATE else totals.forecast_sums[source]
        )
        weeks = scored_weeks.weeks_for(source)

        if metric == "acc":
            forecast_anomaly = (forecast_totals - totals.climatology_sum)[weeks]
            truth_anomaly = truth_anomaly_all_weeks[weeks]
            # Zeroing both sides drops the week without poisoning the
            # gridpoint -- README §2.
            both_finite = np.isfinite(forecast_anomaly) & np.isfinite(truth_anomaly)
            score_map = anomaly_correlation_map(
                np.where(both_finite, forecast_anomaly, 0.0),
                np.where(both_finite, truth_anomaly, 0.0),
            )
        else:
            score_map = root_mean_square_error_map(
                forecast_totals[weeks], totals.observed_sum[weeks]
            )

        scores[source] = np.where(region.mask, score_map, np.nan)
        logger.info(
            "%s %-16s median %.3f  n=%d",
            metric.upper(),
            SOURCE_LABELS[source],
            float(np.nanmedian(scores[source])),
            int(weeks.sum()),
        )
    return scores


def pick_reference_expert(
    scores: dict[str, np.ndarray], expert_names: list[str], metric: str
) -> str:
    """The single expert the gate is compared against in the difference figure.

    One fixed model by region-median score, not a pointwise best, and the
    winner is metric-dependent -- README §2 and §7.
    """
    choose_best = max if metric == "acc" else min
    best_expert = choose_best(
        expert_names, key=lambda name: float(np.nanmedian(scores[name]))
    )
    logger.info(
        "best single expert by %s: %s (median %.3f)",
        metric.upper(),
        SOURCE_LABELS[best_expert],
        float(np.nanmedian(scores[best_expert])),
    )
    return best_expert


# --------------------------------------------------------------------------
# 7. Saving the numbers behind the figures
# --------------------------------------------------------------------------


def save_score_maps(
    out_path: Path,
    scores: dict[str, np.ndarray],
    source_names: list[str],
    region: ScoredRegion,
    metric: str,
    matched: bool,
    n_weeks: int,
    best_expert: str,
    generator: str,
) -> Path:
    """Write the six score maps to netCDF next to the figure.

    One variable per source on the cropped region grid.
    """
    dataset = xr.Dataset(
        {
            source: (("lat", "lon"), scores[source].astype("float32"))
            for source in source_names
        },
        coords={"lat": region.lat, "lon": region.lon},
        attrs={
            "description": (
                "per-gridpoint week-2 accumulated-precip " + metric.upper()
            ),
            "units": "1" if metric == "acc" else "mm/week",
            "matched": str(bool(matched)),
            "n_weeks": int(n_weeks),
            "best_expert": best_expert,
            "generator": generator,
        },
    )
    dataset.to_netcdf(out_path)
    logger.info("wrote %s", out_path)
    return out_path
