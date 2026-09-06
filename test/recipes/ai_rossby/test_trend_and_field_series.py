# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Campaign B's artifacts: the t2m field series and the warming-trend fit.

Upstream ``amip_v2`` implements NO trend metric -- ``rollout_t2m.py`` saves the
daily global-mean series and the fit happens downstream, outside the repo. So
the fit is defined here, and the thing worth testing is that it recovers a
KNOWN slope and that the annual-mean variant (the one to quote) differs from
the raw daily fit in the way the seasonal cycle predicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TEST_DIR))
_AI_ROSSBY_DIR = _TEST_DIR.parents[2] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from climate_eval_suite import FieldSeriesScorer, fit_linear_trend  # noqa: E402

DAY = 24.0
YEAR = 365


# ---------------------------------------------------------------------------
# fit_linear_trend
# ---------------------------------------------------------------------------
def test_recovers_a_known_slope_in_kelvin_per_decade():
    """0.2 K/decade over 35 years of daily steps."""
    n = 35 * YEAR
    t_dec = np.arange(n) * DAY / (24.0 * 365.25 * 10.0)
    series = 288.0 + 0.2 * t_dec
    fit = fit_linear_trend(series, step_hours=DAY)
    assert fit["slope"] == pytest.approx(0.2, rel=1e-6)
    assert fit["intercept"] == pytest.approx(288.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["per"] == "decade"


def test_a_seasonal_cycle_BIASES_the_daily_fit_but_not_the_annual_one():
    """The reason the annual fit is the one to quote -- and it is a stronger
    reason than inflated error bars.

    A seasonal cycle is not merely noise: over an exact whole number of years
    its covariance with a linear ramp is NOT zero, so a naive daily OLS slope
    is *biased*. Closed form for amplitude A over M periods of N steps:

        shift = -A * (MN/2) * cot(pi/N) / sum((k - kbar)^2)

    which at N=365, M=35 is -0.0156 K/decade per K of amplitude. Global-mean
    t2m has a ~3-4 K seasonal cycle, so a daily fit is biased by roughly
    -0.05 to -0.06 K/decade -- comparable to the warming signal being measured.
    Block averaging to annual means removes it entirely.
    """
    n = 35 * YEAR
    t_dec = np.arange(n) * DAY / (24.0 * 365.25 * 10.0)
    A = 10.0
    series = 288.0 + 0.2 * t_dec + A * np.sin(2 * np.pi * np.arange(n) / YEAR)
    daily = fit_linear_trend(series, step_hours=DAY)
    annual = fit_linear_trend(series, step_hours=DAY, block=YEAR)

    # The annual fit recovers the true slope.
    assert annual["slope"] == pytest.approx(0.2, rel=1e-3)
    assert annual["n"] == 35 and annual["block"] == YEAR

    # The daily fit is biased, and by the predicted amount.
    k = np.arange(n)
    predicted = (
        ((k - k.mean()) * np.sin(2 * np.pi * k / YEAR)).sum() * A
        / ((k - k.mean()) ** 2).sum() * (24.0 * 365.25 * 10.0 / DAY)
    )
    assert daily["slope"] == pytest.approx(0.2 + predicted, rel=1e-3)
    assert abs(daily["slope"] - 0.2) > 0.1, "expected a large seasonal bias"

    # And its residual is dominated by the cycle, so its stderr is meaningless.
    assert annual["stderr"] < daily["stderr"] / 10
    assert annual["r2"] > daily["r2"]


def test_per_year_and_per_decade_differ_by_exactly_ten():
    series = 288.0 + 0.1 * np.arange(1000)
    dec = fit_linear_trend(series, step_hours=DAY, per="decade")["slope"]
    yr = fit_linear_trend(series, step_hours=DAY, per="year")["slope"]
    assert dec == pytest.approx(10.0 * yr, rel=1e-9)


def test_nans_from_a_killed_run_are_dropped():
    n = 10 * YEAR
    t_dec = np.arange(n) * DAY / (24.0 * 365.25 * 10.0)
    series = 288.0 + 0.3 * t_dec
    series[int(0.6 * n):] = np.nan          # a run killed 60% through
    fit = fit_linear_trend(series, step_hours=DAY)
    assert fit["slope"] == pytest.approx(0.3, rel=1e-6)
    assert fit["n"] < n


def test_rejects_a_too_short_series_and_a_bad_per():
    with pytest.raises(ValueError, match="cannot fit"):
        fit_linear_trend([1.0, 2.0], step_hours=DAY)
    with pytest.raises(ValueError, match="per must be"):
        fit_linear_trend(np.arange(100.0), step_hours=DAY, per="century")


def test_block_longer_than_the_series_raises():
    with pytest.raises(ValueError, match="shorter than one block"):
        fit_linear_trend(np.arange(10.0), step_hours=DAY, block=YEAR)


# ---------------------------------------------------------------------------
# FieldSeriesScorer
# ---------------------------------------------------------------------------
class _Cat:
    surface = ["skin_temperature", "surface_pressure", "2m_temperature"]
    upper_air: list[str] = []
    diagnostic: list[str] = []
    levels: list[float] = []


class _Drive:
    def __init__(self, horizon=5, grid=(4, 8), batch_size=1, world=1,
                 member_split=False):
        self.horizon = horizon
        self.scored_grid = grid
        self.batch_size = batch_size
        self._world_size = world
        self.member_split = member_split
        self.device = torch.device("cpu")


class _Ctx:
    def __init__(self, m_idx, pred_phys, kind="surface"):
        self.m_idx = m_idx
        self.pred_phys = pred_phys
        self.kind = kind


def test_field_series_records_the_named_channel_per_frame():
    sc = FieldSeriesScorer(catalog=_Cat(), variables=["2m_temperature"])
    sc.bind(_Drive(horizon=3))
    for k in range(3):
        pred = torch.zeros(1, 3, 4, 8)
        pred[0, 2] = float(k + 1)          # channel 2 == 2m_temperature
        sc.score_step(_Ctx(k, pred))
    field = sc.finalize()["field_series"]["2m_temperature"]
    assert field.shape == (3, 4, 8)
    for k in range(3):
        assert torch.allclose(field[k], torch.full((4, 8), float(k + 1)))


def test_unwritten_frames_stay_nan_so_a_partial_run_is_self_describing():
    sc = FieldSeriesScorer(catalog=_Cat(), variables=["2m_temperature"])
    sc.bind(_Drive(horizon=4))
    sc.score_step(_Ctx(0, torch.zeros(1, 3, 4, 8)))
    field = sc.finalize()["field_series"]["2m_temperature"]
    assert torch.isfinite(field[0]).all()
    assert torch.isnan(field[1:]).all()


def test_local_only_omits_the_tensor():
    """A 207 MB tensor must not ride along in every progress snapshot."""
    sc = FieldSeriesScorer(catalog=_Cat(), variables=["2m_temperature"])
    sc.bind(_Drive(horizon=4))
    sc.score_step(_Ctx(0, torch.zeros(1, 3, 4, 8)))
    out = sc.finalize(local_only=True)["field_series"]["2m_temperature"]
    assert out == {"n_written": 1}


def test_refuses_multiple_ics_and_ic_split_ddp():
    sc = FieldSeriesScorer(catalog=_Cat(), variables=["2m_temperature"])
    with pytest.raises(ValueError, match="batch_size=1"):
        sc.bind(_Drive(batch_size=2))
    sc2 = FieldSeriesScorer(catalog=_Cat(), variables=["2m_temperature"])
    with pytest.raises(ValueError, match="IC-split"):
        sc2.bind(_Drive(world=4, member_split=False))
    # member-split is fine: every rank holds the same ensemble-mean field
    sc3 = FieldSeriesScorer(catalog=_Cat(), variables=["2m_temperature"])
    sc3.bind(_Drive(world=4, member_split=True))


def test_unknown_variable_raises_at_construction():
    with pytest.raises(ValueError, match="not found in the catalog"):
        FieldSeriesScorer(catalog=_Cat(), variables=["nonexistent"])
