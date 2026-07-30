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

"""Tests for sample-index construction (datapipes/index.py).

Uses duck-typed fake experts/truth: ``build_sample_index`` only touches
``name`` / ``init_lookup()`` / ``lead_supported()`` and ``day_lookup()``,
so index logic is tested in isolation from zarr I/O (which the adapter and
dataset tests cover).
"""

from __future__ import annotations

import datetime

import cftime
import numpy as np
import pytest

from datapipes.index import build_sample_index


class FakeExpert:
    def __init__(self, name, init_keys, lead_range=(1, 100)):
        self.name = name
        self._table = {
            k: (k[0], i) for i, k in enumerate(sorted(init_keys))
        }
        self.lead_range = lead_range

    def init_lookup(self):
        return self._table

    def lead_supported(self, tau):
        return self.lead_range[0] <= tau <= self.lead_range[1]


class FakeTruth:
    def __init__(self, days):
        self._days = {d: (d[0], i) for i, d in enumerate(sorted(days))}

    def day_lookup(self):
        return self._days


def _daily_days(year, months):
    days = []
    for m in months:
        d = cftime.DatetimeGregorian(year, m, 1)
        while d.month == m:
            days.append((d.year, d.month, d.day))
            d = d + datetime.timedelta(days=1)
    return days


def test_union_masks_and_filters():
    e1 = FakeExpert("a", [(2001, 6, 1, 0), (2001, 6, 5, 0), (2001, 7, 1, 0)])
    e2 = FakeExpert("b", [(2001, 6, 5, 0), (2001, 6, 9, 0), (2002, 6, 1, 0)])
    truth = FakeTruth(_daily_days(2001, (6, 7)) + _daily_days(2002, (6,)))
    idx = build_sample_index(
        [e1, e2], truth,
        years=(2001, 2001), init_months=(6,), lead_days=(8, 9),
    )
    # Union of June-2001 inits: 6/1 (a only), 6/5 (both), 6/9 (b only).
    assert idx.init_keys == [
        (2001, 6, 1, 0), (2001, 6, 5, 0), (2001, 6, 9, 0)
    ]
    # 3 inits x 2 taus, all IMERG days present.
    assert len(idx.pairs) == 6
    masks = {
        (int(r["init_row"]), int(r["tau"])): idx.expert_mask(r).tolist()
        for r in idx.pairs
    }
    assert masks[(0, 8)] == [True, False]
    assert masks[(1, 8)] == [True, True]
    assert masks[(2, 9)] == [False, True]
    # init_locs: -1 where the expert lacks the init.
    assert idx.init_locs[0, 1, 0] == -1
    assert idx.init_locs[1, 0].tolist() == [2001, 1]  # a's 2nd sorted init


def test_lead_support_masks_expert():
    e1 = FakeExpert("a", [(2001, 6, 1, 0)], lead_range=(1, 8))
    e2 = FakeExpert("b", [(2001, 6, 1, 0)], lead_range=(9, 20))
    truth = FakeTruth(_daily_days(2001, (6,)))
    idx = build_sample_index(
        [e1, e2], truth,
        years=(2001, 2001), init_months=(6,), lead_days=(8, 9),
    )
    masks = {int(r["tau"]): idx.expert_mask(r).tolist() for r in idx.pairs}
    assert masks[8] == [True, False]
    assert masks[9] == [False, True]


def test_imerg_gap_drops_exactly_those_pairs():
    e = FakeExpert("a", [(2001, 6, 1, 0)])
    days = [d for d in _daily_days(2001, (6,)) if d != (2001, 6, 9)]
    idx = build_sample_index(
        [e], FakeTruth(days),
        years=(2001, 2001), init_months=(6,), lead_days=(8, 12),
    )
    # Valid-day stamp is init + (tau - 1): tau=10 -> June 10... the gap at
    # June 9 kills exactly tau=10's record stamped (2001, 6, 10)? No —
    # stamp = 1 Jun + (tau-1) days: tau=9 -> Jun 9 (gap), others survive.
    taus = sorted(int(r["tau"]) for r in idx.pairs)
    assert taus == [8, 10, 11, 12]


def test_cross_year_valid_day_and_feb29():
    e = FakeExpert("a", [(2001, 12, 28, 0), (2004, 2, 25, 0)])
    days = _daily_days(2002, (1,)) + _daily_days(2004, (2,))
    idx = build_sample_index(
        [e], FakeTruth(days),
        years=(2001, 2004), init_months=(12, 2), lead_days=(8, 8),
    )
    stamps = set()
    for r in idx.pairs:
        key = idx.init_keys[int(r["init_row"])]
        d = cftime.DatetimeGregorian(*key) + datetime.timedelta(days=7)
        stamps.add((d.year, d.month, d.day))
    # Dec 28 + 7 = Jan 4 of the NEXT year; Feb 25 2004 + 7 = Mar 3 (absent
    # from truth -> dropped), so only the cross-year pair survives...
    # Feb 29 2004 exists in the truth and would be hit by tau=5.
    assert (2002, 1, 4) in stamps
    idx29 = build_sample_index(
        [e], FakeTruth(days),
        years=(2004, 2004), init_months=(2,), lead_days=(5, 5),
    )
    d = cftime.DatetimeGregorian(2004, 2, 25) + datetime.timedelta(days=4)
    assert (d.year, d.month, d.day) == (2004, 2, 29)
    assert len(idx29.pairs) == 1


def test_min_experts_all_is_intersection():
    e1 = FakeExpert("a", [(2001, 6, 1, 0), (2001, 6, 5, 0)])
    e2 = FakeExpert("b", [(2001, 6, 5, 0), (2001, 6, 9, 0)])
    truth = FakeTruth(_daily_days(2001, (6,)))
    idx = build_sample_index(
        [e1, e2], truth,
        years=(2001, 2001), init_months=(6,), lead_days=(8, 8),
        min_experts="all",
    )
    assert len(idx.pairs) == 1
    (row,) = idx.pairs
    assert idx.init_keys[int(row["init_row"])] == (2001, 6, 5, 0)
    assert idx.expert_mask(row).all()


def test_empty_index_raises_with_counts():
    e = FakeExpert("a", [(2001, 6, 1, 0)])
    truth = FakeTruth(_daily_days(2003, (6,)))  # no overlap
    with pytest.raises(ValueError, match="empty sample index"):
        build_sample_index(
            [e], truth,
            years=(2001, 2001), init_months=(6,), lead_days=(8, 9),
        )


def test_bad_args_raise():
    e = FakeExpert("a", [(2001, 6, 1, 0)])
    truth = FakeTruth(_daily_days(2001, (6,)))
    with pytest.raises(ValueError, match="out of range"):
        build_sample_index(
            [e], truth, years=(2001, 2001), init_months=(6,),
            lead_days=(8, 8), min_experts=2,
        )
    with pytest.raises(ValueError, match="must be an int or 'all'"):
        build_sample_index(
            [e], truth, years=(2001, 2001), init_months=(6,),
            lead_days=(8, 8), min_experts="any",
        )
    with pytest.raises(ValueError, match="bad lead_days"):
        build_sample_index(
            [e], truth, years=(2001, 2001), init_months=(6,), lead_days=(0, 8),
        )


def test_exclude_years_drops_a_cv_fold():
    """exclude_years removes whole init years from inside the range, which is
    what k-fold CV needs: three of five folds have a gap in the middle that a
    plain (lo, hi) range cannot express."""
    e = FakeExpert(
        "a",
        [(2001, 6, 1, 0), (2002, 6, 1, 0), (2003, 6, 1, 0)],
    )
    truth = FakeTruth(
        _daily_days(2001, (6,)) + _daily_days(2002, (6,)) + _daily_days(2003, (6,))
    )
    common = dict(init_months=(6,), lead_days=(8, 9))
    full = build_sample_index([e], truth, years=(2001, 2003), **common)
    held = build_sample_index(
        [e], truth, years=(2001, 2003), exclude_years=(2002,), **common
    )
    assert {k[0] for k in full.init_keys} == {2001, 2002, 2003}
    assert {k[0] for k in held.init_keys} == {2001, 2003}
    assert len(held.pairs) < len(full.pairs)
    # Excluding a year outside the range changes nothing.
    same = build_sample_index(
        [e], truth, years=(2001, 2003), exclude_years=(1999,), **common
    )
    assert len(same.pairs) == len(full.pairs)
