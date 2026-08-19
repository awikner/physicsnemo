# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12f — ``SequenceDataset`` forcing alignment + W+1 boundary emission.

Upstream amip aligns window slot ``w`` (the state at step ``w+1``) with the
forcing at step ``w``: *denoising the frame at time T uses the boundary
forcing from T-1*. The fork's original rolling path paired both at the same
time. The two differ by a whole-window shift, so every tensor keeps its shape
and a mismatch is silent — hence these tests pin the frame identities, not just
the shapes.

``emit_boundary_next`` then adds the second view of the same read window: the
boundary at each *state* frame's own time, which is the ocean-channel training
target. Asking for it at ``forcing_lag=0`` is refused, because there the two
views are the same tensor and the ocean task would be an identity copy of an
input channel.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.climate import SequenceDataset

_SURFACE, _UA, _VARY, _CONST = 2, 1, 3, 2
_LEVELS = 2
_H, _W = 4, 8
_N_TIME = 12


class _Base:
    """Base dataset whose every field encodes its own time index.

    Each frame is a constant field equal to ``t`` (offset per group) so an
    off-by-one in the window slicing shows up as a wrong number rather than a
    statistical difference.
    """

    n_time = _N_TIME
    layout = None

    def __getitem__(self, index):
        t, _lead = index if isinstance(index, tuple) else (index, 1)
        t = float(t)
        return {
            "surface_in": torch.full((_SURFACE, _H, _W), t),
            "upper_air_in": torch.full((_UA, _LEVELS, _H, _W), t),
            "diagnostic": torch.full((2, _H, _W), t),
            "varying_boundary": torch.full((_VARY, _H, _W), 100.0 + t),
            "constant_boundary": torch.full((_CONST, _H, _W), 7.0),
            "calendar": torch.full((2,), 200.0 + t),
        }

    def __len__(self):
        return self.n_time


def _frame_times(seq):
    """Recover the encoded time index of each frame in a ``(W, ...)`` stack."""
    return [float(seq[i].flatten()[0]) for i in range(seq.shape[0])]


# ---------------------------------------------------------------------------
# forcing_lag = 0 (the fork's historical behavior)
# ---------------------------------------------------------------------------


def test_lag_zero_pairs_state_and_forcing_at_the_same_time():
    ds = SequenceDataset(_Base(), unroll_steps=2)
    s = ds[3]
    assert _frame_times(s["surface_in_seq"]) == [3.0, 4.0, 5.0]
    assert _frame_times(s["varying_boundary_seq"]) == [103.0, 104.0, 105.0]
    assert _frame_times(s["calendar_seq"]) == [203.0, 204.0, 205.0]
    assert "varying_boundary_next_seq" not in s


def test_lag_zero_length_and_frame_count_unchanged():
    ds = SequenceDataset(_Base(), unroll_steps=2)
    assert ds.forcing_lag == 0
    assert ds.frames_per_sample == 3
    assert len(ds) == _N_TIME - 2


# ---------------------------------------------------------------------------
# forcing_lag = 1 (upstream / v1 / v2)
# ---------------------------------------------------------------------------


def test_lag_one_puts_the_state_one_step_ahead_of_its_forcing():
    ds = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1)
    s = ds[3]
    # Slot j: state at t+1+j, conditioned on the forcing at t+j.
    assert _frame_times(s["surface_in_seq"]) == [4.0, 5.0, 6.0]
    assert _frame_times(s["upper_air_in_seq"]) == [4.0, 5.0, 6.0]
    assert _frame_times(s["diagnostic_seq"]) == [4.0, 5.0, 6.0]
    assert _frame_times(s["varying_boundary_seq"]) == [103.0, 104.0, 105.0]
    assert _frame_times(s["calendar_seq"]) == [203.0, 204.0, 205.0]


def test_lag_one_reads_one_extra_frame_and_shortens_the_dataset():
    ds = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1)
    assert ds.frames_per_sample == 4
    assert len(ds) == _N_TIME - 3
    # The last valid index must not read past the end.
    _ = ds[len(ds) - 1]
    with pytest.raises(IndexError):
        _ = ds[_N_TIME - 3]


def test_lag_one_shapes_match_lag_zero():
    # The shift is invisible in the shapes — the reason the identities above
    # are what these tests assert.
    a = SequenceDataset(_Base(), unroll_steps=2)[3]
    b = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1)[3]
    for key in ("surface_in_seq", "varying_boundary_seq", "calendar_seq"):
        assert a[key].shape == b[key].shape


# ---------------------------------------------------------------------------
# emit_boundary_next (the ocean target)
# ---------------------------------------------------------------------------


def test_boundary_next_is_the_boundary_at_the_state_frames_own_time():
    ds = SequenceDataset(
        _Base(), unroll_steps=2, forcing_lag=1, emit_boundary_next=True
    )
    s = ds[3]
    assert _frame_times(s["surface_in_seq"]) == [4.0, 5.0, 6.0]
    assert _frame_times(s["varying_boundary_seq"]) == [103.0, 104.0, 105.0]
    assert _frame_times(s["varying_boundary_next_seq"]) == [104.0, 105.0, 106.0]
    # Same shape as the forcing window: nothing about the shapes distinguishes
    # them, which is exactly the failure mode this key exists to prevent.
    assert s["varying_boundary_next_seq"].shape == s["varying_boundary_seq"].shape


def test_boundary_next_costs_no_extra_reads():
    ds = SequenceDataset(
        _Base(), unroll_steps=2, forcing_lag=1, emit_boundary_next=True
    )
    assert ds.frames_per_sample == 4      # same as lag=1 without the key
    assert len(ds) == _N_TIME - 3


def test_boundary_next_at_lag_zero_is_refused():
    with pytest.raises(ValueError, match="identity task"):
        SequenceDataset(_Base(), unroll_steps=2, emit_boundary_next=True)


def test_negative_forcing_lag_is_refused():
    with pytest.raises(ValueError, match="forcing_lag"):
        SequenceDataset(_Base(), unroll_steps=2, forcing_lag=-1)


def test_window_of_one_still_shifts():
    # W=1 (unroll_steps=0) is a degenerate but legal window; the shift must
    # still apply or a single-frame stage would train on the wrong forcing.
    ds = SequenceDataset(
        _Base(), unroll_steps=0, forcing_lag=1, emit_boundary_next=True
    )
    s = ds[5]
    assert _frame_times(s["surface_in_seq"]) == [6.0]
    assert _frame_times(s["varying_boundary_seq"]) == [105.0]
    assert _frame_times(s["varying_boundary_next_seq"]) == [106.0]


def test_constant_boundary_and_bookkeeping_survive_the_shift():
    ds = SequenceDataset(
        _Base(), unroll_steps=2, forcing_lag=1, emit_boundary_next=True
    )
    s = ds[3]
    assert s["constant_boundary"].shape == (_CONST, _H, _W)
    assert torch.all(s["constant_boundary"] == 7.0)
    assert int(s["start_idx"]) == 3
    assert int(s["unroll_steps"]) == 2


# ---------------------------------------------------------------------------
# emit_anchor — the pre-window state frame a data-coupled scheduler needs
# ---------------------------------------------------------------------------


def test_emit_anchor_returns_the_frame_before_the_window():
    """Rolling Stochastic Interpolants anchor slot 1 on the frame before W.

    Slot j targets the state at ``t+1+j``; slot 0's anchor is therefore the
    state at ``t``, which at ``forcing_lag=1`` is already read (it is where
    slot 0's forcing comes from) and today thrown away.
    """
    ds = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1, emit_anchor=True)
    s = ds[3]
    assert _frame_times(s["surface_in_seq"]) == [4.0, 5.0, 6.0]
    # One frame earlier than the window's first target, and unstacked.
    assert s["surface_in_prev"].shape == (_SURFACE, _H, _W)
    assert float(s["surface_in_prev"].flatten()[0]) == 3.0
    assert float(s["upper_air_in_prev"].flatten()[0]) == 3.0
    assert float(s["diagnostic_prev"].flatten()[0]) == 3.0


def test_emit_anchor_costs_no_extra_read():
    """The anchor is already in the sample; the flag must not lengthen it."""
    plain = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1)
    anchored = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1,
                               emit_anchor=True)
    assert anchored.frames_per_sample == plain.frames_per_sample
    assert anchored.row_span == plain.row_span
    assert len(anchored) == len(plain)


def test_emit_anchor_leaves_every_existing_key_untouched():
    plain = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1,
                            emit_boundary_next=True)[3]
    anchored = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1,
                               emit_boundary_next=True, emit_anchor=True)[3]
    for k, v in plain.items():
        assert torch.equal(anchored[k], v), f"{k} changed under emit_anchor"
    assert set(anchored) - set(plain) == {
        "surface_in_prev", "upper_air_in_prev", "diagnostic_prev"
    }


def test_anchor_own_time_boundary_is_the_slot_one_forcing_frame():
    """The W+1 ocean-target stack the recipe builds must line up frame-wise.

    ``_train_step`` prepends ``varying_boundary_seq[:, :1]`` to
    ``varying_boundary_next_seq`` to get the boundary at each of the W+1 state
    frames' OWN time. That only works if the anchor's own-time boundary really
    is the slot-1 conditioning frame.
    """
    ds = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1,
                         emit_boundary_next=True, emit_anchor=True)
    s = ds[3]
    anchor_t = float(s["surface_in_prev"].flatten()[0])          # 3.0
    stack = torch.cat(
        [s["varying_boundary_seq"][:1], s["varying_boundary_next_seq"]], dim=0
    )
    states = [anchor_t] + _frame_times(s["surface_in_seq"])      # [3, 4, 5, 6]
    assert _frame_times(stack) == [100.0 + t for t in states]


def test_emit_anchor_is_refused_at_lag_zero():
    with pytest.raises(ValueError, match="emit_anchor"):
        SequenceDataset(_Base(), unroll_steps=2, forcing_lag=0, emit_anchor=True)


def test_emit_anchor_defaults_off():
    s = SequenceDataset(_Base(), unroll_steps=2, forcing_lag=1)[3]
    assert not any(k.endswith("_prev") for k in s)
