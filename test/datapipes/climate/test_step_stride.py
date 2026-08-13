# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Model step vs store row spacing (2026-08-13 audit).

A store's rows are ``data_timedelta_hours`` apart, which is not always the
model's timestep: every upstream AMIP config (v1 *and* v2) and the ERA5 recipes
step 24 h over 6-hourly rows, i.e. 4 rows per step. The fork spells that at row
level as ``forecast_lead_times`` and, since this audit, optionally in hours as
``timedelta_hours``.

:func:`resolve_step_stride` is where the two meet, and the cross-check is the
point: a stride disagreement changes no shape and produces a perfectly healthy
loss, so it cannot be caught downstream. ``SequenceDataset`` then has to stride
*both* its window frames and its ``forcing_lag`` offset — "one step back" is one
model step, not one row.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.climate import (
        SequenceDataset,
        resolve_step_stride,
    )

_H, _W = 4, 8
_N_TIME = 60


class _Layout:
    def __init__(self, hours):
        self.data_timedelta_hours = hours


class _Store:
    """Frames whose values encode their own row index."""

    n_time = _N_TIME
    layout = None

    def __init__(self, hours=6):
        self.layout = _Layout(hours)

    def __getitem__(self, index):
        t, _lead = index if isinstance(index, tuple) else (index, 1)
        t = float(t)
        return {
            "surface_in": torch.full((2, _H, _W), t),
            "varying_boundary": torch.full((3, _H, _W), 100.0 + t),
            "constant_boundary": torch.full((2, _H, _W), 7.0),
            "calendar": torch.full((2,), 200.0 + t),
        }

    def __len__(self):
        return self.n_time


def _rows(seq):
    return [float(seq[i].flatten()[0]) for i in range(seq.shape[0])]


# ---------------------------------------------------------------------------
# resolve_step_stride
# ---------------------------------------------------------------------------


def test_lead_alone_is_the_stride():
    assert resolve_step_stride(_Store(), [4]) == 4
    assert resolve_step_stride(_Store(), [1]) == 1


def test_hours_are_divided_by_the_stores_own_spacing():
    # Upstream's formula: timedelta_hours // data_timedelta_hours.
    assert resolve_step_stride(_Store(6), None, 24) == 4
    assert resolve_step_stride(_Store(6), None, 6) == 1
    assert resolve_step_stride(_Store(24), None, 24) == 1


def test_the_two_spellings_are_cross_checked():
    assert resolve_step_stride(_Store(6), [4], 24) == 4
    with pytest.raises(ValueError, match="model step disagreement"):
        resolve_step_stride(_Store(6), [1], 24)
    with pytest.raises(ValueError, match="model step disagreement"):
        resolve_step_stride(_Store(6), [4], 6)


def test_hours_must_divide_the_row_spacing():
    with pytest.raises(ValueError, match="whole number"):
        resolve_step_stride(_Store(6), None, 9)


def test_hours_need_the_store_attribute():
    class _NoAttr:
        layout = _Layout(0)

    with pytest.raises(ValueError, match="data_timedelta_hours"):
        resolve_step_stride(_NoAttr(), None, 24)


def test_multiple_distinct_leads_are_refused():
    # Lead-conditioned training is single-step only; there is no one timestep
    # to stride a window by, and picking one silently would be worse.
    with pytest.raises(ValueError, match="several distinct"):
        resolve_step_stride(_Store(), [1, 4])
    # A repeated value is not ambiguous.
    assert resolve_step_stride(_Store(), [4, 4]) == 4


def test_empty_or_zero_leads_are_refused():
    with pytest.raises(ValueError, match="empty"):
        resolve_step_stride(_Store(), [])
    with pytest.raises(ValueError, match=">= 1"):
        resolve_step_stride(_Store(), [0])


def test_no_information_means_one_row_per_step():
    assert resolve_step_stride(_Store()) == 1


# ---------------------------------------------------------------------------
# SequenceDataset striding
# ---------------------------------------------------------------------------


def test_window_frames_are_a_model_step_apart():
    ds = SequenceDataset(_Store(), unroll_steps=2, step_stride=4)
    s = ds[8]
    assert _rows(s["surface_in_seq"]) == [8.0, 12.0, 16.0]
    assert _rows(s["varying_boundary_seq"]) == [108.0, 112.0, 116.0]
    assert _rows(s["calendar_seq"]) == [208.0, 212.0, 216.0]


def test_forcing_lag_is_one_model_step_not_one_row():
    """The lag must stride too.

    With ``forcing_lag=1`` and a 4-row step, slot j is the state at row
    ``t + 4(j+1)`` conditioned on the forcing at row ``t + 4j`` — one *step*
    behind. A lag of one raw row would condition a 24-hour forecast on a
    forcing 6 hours behind it, which is neither convention.
    """
    ds = SequenceDataset(
        _Store(), unroll_steps=2, forcing_lag=1, emit_boundary_next=True,
        step_stride=4,
    )
    s = ds[8]
    assert _rows(s["surface_in_seq"]) == [12.0, 16.0, 20.0]
    assert _rows(s["varying_boundary_seq"]) == [108.0, 112.0, 116.0]
    assert _rows(s["varying_boundary_next_seq"]) == [112.0, 116.0, 120.0]


def test_row_span_and_length_account_for_the_stride():
    ds = SequenceDataset(_Store(), unroll_steps=2, step_stride=4)
    assert ds.frames_per_sample == 3
    assert ds.row_span == 8               # (3 - 1) * 4
    assert len(ds) == _N_TIME - 8
    lagged = SequenceDataset(
        _Store(), unroll_steps=2, forcing_lag=1, step_stride=4
    )
    assert lagged.row_span == 12          # (4 - 1) * 4
    assert len(lagged) == _N_TIME - 12


def test_the_last_valid_index_does_not_read_past_the_archive():
    ds = SequenceDataset(_Store(), unroll_steps=2, forcing_lag=1, step_stride=4)
    last = len(ds) - 1
    frames = _rows(ds[last]["surface_in_seq"])
    assert max(frames) <= _N_TIME - 1
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_stride_one_is_the_previous_behavior():
    a = SequenceDataset(_Store(), unroll_steps=2)
    b = SequenceDataset(_Store(), unroll_steps=2, step_stride=1)
    assert a.row_span == b.row_span == 2
    assert len(a) == len(b) == _N_TIME - 2
    assert _rows(a[3]["surface_in_seq"]) == _rows(b[3]["surface_in_seq"]) == [
        3.0, 4.0, 5.0
    ]


def test_zero_stride_is_refused():
    with pytest.raises(ValueError, match="step_stride"):
        SequenceDataset(_Store(), unroll_steps=2, step_stride=0)


def test_constant_boundary_and_bookkeeping_survive_striding():
    ds = SequenceDataset(_Store(), unroll_steps=2, step_stride=4)
    s = ds[8]
    assert torch.all(s["constant_boundary"] == 7.0)
    assert int(s["start_idx"]) == 8


# ---------------------------------------------------------------------------
# The datapipe's multistep branch (on a real 6-hourly store)
# ---------------------------------------------------------------------------


def _write_store(path, n_time=40):
    """Tiny 6-hourly store whose surface field encodes its own row index."""
    import cftime
    import numpy as np
    import xarray as xr
    from datetime import timedelta

    base = cftime.DatetimeGregorian(2000, 1, 1)
    times = [base + timedelta(hours=6 * i) for i in range(n_time)]
    idx = np.arange(n_time, dtype="float32")[:, None, None]
    xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), np.broadcast_to(idx, (n_time, _H, _W)).copy()),
            "lsm": (("lat", "lon"), np.zeros((_H, _W), dtype="float32")),
            "sst": (("time", "lat", "lon"), np.broadcast_to(idx, (n_time, _H, _W)).copy()),
        },
        coords={
            "time": ("time", times),
            "lat": ("lat", np.linspace(87.5, -87.5, _H, dtype="float32")),
            "lon": ("lon", np.linspace(0, 360 * (_W - 1) / _W, _W, dtype="float32")),
        },
        attrs={
            "calendar": "standard",
            "data_timedelta_hours": 6,
            "surface_variables": ["t2m"],
            "constant_boundary_variables": ["lsm"],
            "varying_boundary_variables": ["sst"],
            "diagnostic_variables": [],
            "pressure_upper_air_variables": [],
            "sigma_upper_air_variables": [],
        },
    ).to_zarr(path, mode="w", consolidated=True, zarr_format=3)


def _pipe(tmp_path, **kw):
    from physicsnemo.experimental.datapipes.climate import ClimateDatapipe

    store = tmp_path / "store.zarr"
    if not store.exists():
        _write_store(store)
    return ClimateDatapipe(
        store,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        device=torch.device("cpu"),
        **kw,
    )


def test_datapipe_multistep_strides_by_the_configured_lead(tmp_path):
    pipe = _pipe(tmp_path, unroll_steps=3, forecast_lead_times=[4])
    assert isinstance(pipe.dataset, SequenceDataset)
    assert pipe.dataset.step_stride == 4
    # And the frames really are 4 rows apart end to end.
    batch = next(iter(pipe))
    assert _rows(batch["surface_in_seq"][0]) == [0.0, 4.0, 8.0, 12.0]


def test_datapipe_multistep_default_lead_is_unchanged(tmp_path):
    pipe = _pipe(tmp_path, unroll_steps=3, forecast_lead_times=[1])
    assert pipe.dataset.step_stride == 1
    batch = next(iter(pipe))
    assert _rows(batch["surface_in_seq"][0]) == [0.0, 1.0, 2.0, 3.0]


def test_datapipe_single_step_branch_is_untouched(tmp_path):
    # Single-step mode already honored the lead through the sampler, so it must
    # not be wrapped in a SequenceDataset.
    pipe = _pipe(tmp_path, unroll_steps=1, forecast_lead_times=[4])
    assert not isinstance(pipe.dataset, SequenceDataset)
