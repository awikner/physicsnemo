# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Unit tests for ``inference.resolve_init_schedule`` (Phase B2 hindcast).

The resolver turns a calendar ``init_schedule`` block into sorted INTEGER
time indices into the (multi-year) store time coord. It is exercised here
against a synthetic 6-hourly cftime axis.

Why the resolver is imported by *source extraction* rather than
``import inference``: ``inference.py`` imports the full physicsnemo stack
(hydra / omegaconf / torch CUDA / warp) at module top, which is neither
installed nor safe to import in a lightweight test env. ``resolve_init_schedule``
(and its tiny helper ``_timestamp_ymdh``) are, by contrast, pure functions that
depend only on ``numpy`` + stdlib. We therefore parse ``inference.py`` with
``ast``, pull out *exactly those two function definitions verbatim* (so the
test exercises the real shipping code, not a drifting copy), and ``exec`` them
in an isolated namespace. This keeps ``inference.py``'s real imports intact
(constraint: do not break them) while making the resolver testable in
isolation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cftime
import numpy as np

_INFERENCE_PY = Path(__file__).resolve().parent / "inference.py"
_WANTED = ("_timestamp_ymdh", "resolve_init_schedule")


def _load_resolver():
    """Extract + exec the pure resolver functions from ``inference.py``."""
    source = _INFERENCE_PY.read_text()
    tree = ast.parse(source)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            seg = ast.get_source_segment(source, node)
            assert seg is not None, f"could not extract source for {node.name}"
            segments.append(seg)
    assert len(segments) == len(_WANTED), (
        f"expected to find {_WANTED} in inference.py, found {len(segments)}"
    )
    # ``from __future__ import annotations`` keeps the type hints as unevaluated
    # strings so we don't need typing symbols in the exec namespace.
    prelude = "from __future__ import annotations\nimport numpy as np\n"
    ns: dict = {}
    exec(compile(prelude + "\n\n".join(segments), str(_INFERENCE_PY), "exec"), ns)
    return ns["resolve_init_schedule"]


resolve_init_schedule = _load_resolver()

# Schedule from the task: every 4th day starting on the 1st, at 00Z.
DAYS = [1, 5, 9, 13, 17, 21, 25, 29]
MONTHS = list(range(1, 13))
HOURS = [0]


def _six_hourly_axis(years):
    """Build a contiguous 6-hourly ``cftime.DatetimeGregorian`` axis."""
    times = []
    for y in years:
        t = cftime.DatetimeGregorian(y, 1, 1, 0)
        end = cftime.DatetimeGregorian(y + 1, 1, 1, 0)
        while t < end:
            times.append(t)
            t = t + __import__("datetime").timedelta(hours=6)
    return np.array(times, dtype=object)


def test_count_per_year_excludes_feb29():
    """95 ICs/year: 11 months x 8 days + Feb (7 days; day-29 dropped/absent)."""
    # 2000 is a leap year (Feb 29 exists on the axis and must be dropped);
    # 2001 is not (Feb 29 simply never appears).
    times = _six_hourly_axis([2000, 2001])

    idx_2000 = resolve_init_schedule(times, MONTHS, DAYS, HOURS, years=[2000])
    idx_2001 = resolve_init_schedule(times, MONTHS, DAYS, HOURS, years=[2001])
    assert len(idx_2000) == 95, len(idx_2000)
    assert len(idx_2001) == 95, len(idx_2001)

    both = resolve_init_schedule(times, MONTHS, DAYS, HOURS)
    assert len(both) == 190, len(both)


def test_indices_sorted_and_valid():
    times = _six_hourly_axis([2000, 2001])
    idx = resolve_init_schedule(times, MONTHS, DAYS, HOURS)
    assert idx == sorted(idx)
    # strictly increasing (no duplicates)
    assert all(b > a for a, b in zip(idx, idx[1:]))
    assert all(0 <= i < len(times) for i in idx)


def test_feb29_dropped_in_leap_year():
    times = _six_hourly_axis([2000])  # leap year — Feb 29 present on the axis
    # Sanity: Feb 29 00Z really is on the raw axis.
    assert any(t.month == 2 and t.day == 29 for t in times)
    idx = resolve_init_schedule(times, MONTHS, DAYS, HOURS)
    for i in idx:
        t = times[i]
        assert not (t.month == 2 and t.day == 29), f"Feb 29 leaked at index {i}"


def test_hour_and_day_filters_applied():
    times = _six_hourly_axis([2001])
    idx = resolve_init_schedule(times, MONTHS, DAYS, HOURS)
    for i in idx:
        t = times[i]
        assert t.hour == 0
        assert t.day in DAYS
        assert t.month in MONTHS


def test_none_filters_match_any_but_still_drop_feb29():
    """No month/day/hour filter -> every step except Feb-29 steps."""
    times = _six_hourly_axis([2000])
    idx = resolve_init_schedule(times, None, None, None)
    n_feb29 = sum(1 for t in times if t.month == 2 and t.day == 29)
    assert n_feb29 == 4  # 4 x 6-hourly steps on Feb 29 (leap year)
    assert len(idx) == len(times) - n_feb29


def test_np_datetime64_axis():
    """The resolver also accepts a numpy datetime64 axis (ERA5/E3SM regime)."""
    times = np.arange(
        np.datetime64("2001-01-01T00", "h"),
        np.datetime64("2002-01-01T00", "h"),
        np.timedelta64(6, "h"),
    )
    idx = resolve_init_schedule(times, MONTHS, DAYS, HOURS, years=[2001])
    assert len(idx) == 95, len(idx)
    assert idx == sorted(idx)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
