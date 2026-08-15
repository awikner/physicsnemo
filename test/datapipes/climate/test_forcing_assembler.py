# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12d unit tests: ForcingAssembler + masked-boundary smoothing.

The assembler is the fork's ``forcing_from_raw`` choke point (CO₂-style
scalar routing + calendar row); the smoothing is the port of upstream's
``smooth_masked_boundary`` coast fade. All synthetic — no data fixtures.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.datapipes.climate import (
    ForcingAssembler,
    NanFillTransform,
    smooth_masked_boundary,
)

_H, _W = 8, 16
_VARY = ["global_mean_co2", "DSWRFtoa_24h_lead", "sst", "sic"]
_CONST = ["geopotential_at_surface", "land_sea_mask"]


def _assembler(**kw) -> ForcingAssembler:
    kw.setdefault("scalar_routed_variables", ["global_mean_co2"])
    return ForcingAssembler(
        varying_boundary_variables=_VARY,
        constant_boundary_variables=_CONST,
        **kw,
    )


def _sample(co2: float = 1.25, *, calendar=(0.0, 12.0)) -> dict:
    torch.manual_seed(0)
    vb = torch.randn(len(_VARY), _H, _W)
    vb[0] = co2  # uniform map, as stored
    return {
        "varying_boundary": vb,
        "constant_boundary": torch.randn(len(_CONST), _H, _W),
        "calendar": torch.tensor(list(calendar)),
    }


# ---------------------------------------------------------------------------
# Derived contract
# ---------------------------------------------------------------------------


def test_dims_derived_from_variable_lists():
    a = _assembler()
    # Upstream ERDM_co2.yaml: c_grid_dim 5, scalar_dim 3.
    assert a.c_grid_dim == 5
    assert a.scalar_dim == 3
    assert a.varying_boundary_variables_out == _VARY[1:]
    assert a.active


def test_no_routing_is_identity_passthrough():
    a = _assembler(scalar_routed_variables=[])
    assert not a.active
    assert a.c_grid_dim == len(_CONST) + len(_VARY)
    assert a.scalar_dim == 2
    s = _sample()
    out = a(s)
    assert out is s  # untouched object, not just equal


def test_unknown_routed_variable_rejected():
    with pytest.raises(ValueError, match="not in varying_boundary_variables"):
        _assembler(scalar_routed_variables=["not_a_channel"])


# ---------------------------------------------------------------------------
# The routing itself
# ---------------------------------------------------------------------------


def test_pops_channel_and_appends_scalar():
    a = _assembler()
    s = _sample(co2=1.25)
    before = s["varying_boundary"].clone()
    out = a(s)
    assert out["varying_boundary"].shape == (len(_VARY) - 1, _H, _W)
    # Remaining channels are the non-routed ones, in stored order, unchanged.
    assert torch.equal(out["varying_boundary"], before[1:])
    assert out["calendar"].tolist() == pytest.approx([0.0, 12.0, 1.25])


def test_routes_multiple_channels_in_listed_order():
    a = _assembler(scalar_routed_variables=["sic", "global_mean_co2"])
    s = _sample(co2=0.5)
    s["varying_boundary"][3] = -2.0  # sic uniform
    out = a(s)
    assert out["varying_boundary"].shape == (2, _H, _W)
    assert out["calendar"].tolist() == pytest.approx([0.0, 12.0, -2.0, 0.5])
    assert a.varying_boundary_variables_out == ["DSWRFtoa_24h_lead", "sst"]


def test_non_uniform_routed_channel_raises_in_strict_mode():
    a = _assembler()
    s = _sample()
    s["varying_boundary"][0, 0, 0] += 5.0  # break uniformity
    with pytest.raises(ValueError, match="not spatially uniform"):
        a(s)


def test_non_uniform_tolerated_when_not_strict():
    a = _assembler(strict=False)
    s = _sample(co2=1.0)
    s["varying_boundary"][0, 0, 0] = 1.0 + 8.0
    out = a(s)  # mean of a mostly-1.0 map with one outlier
    assert out["calendar"].shape == (3,)
    assert out["calendar"][2] > 1.0


def test_reduce_first_matches_upstream_corner_read():
    a = _assembler(reduce="first", strict=False)
    s = _sample(co2=2.0)
    s["varying_boundary"][0, 0, 0] = 2.0
    s["varying_boundary"][0, 4, 4] = 99.0  # ignored by 'first'
    out = a(s)
    assert float(out["calendar"][2]) == pytest.approx(2.0)


def test_missing_calendar_raises_strict_and_passes_through_otherwise():
    s = _sample()
    del s["calendar"]
    with pytest.raises(KeyError, match="calendar"):
        _assembler()(dict(s))
    out = _assembler(strict=False)(dict(s))
    # Upstream: no calendar -> no scalar, CO2 stays in the grid.
    assert out["varying_boundary"].shape[0] == len(_VARY)


def test_leading_axis_is_preserved():
    """A (T, C, H, W) stack routes on the channel axis and keeps T per-frame.

    Rewritten 2026-08-14. It previously paired a leading-axis boundary with a
    1-D ``(2,)`` calendar and only asserted the boundary's shape — which passed
    because ``_scalar_of`` flattened *everything* and averaged the routed
    channel across all 3 frames. The calendar it produced was therefore a single
    shared CO2 value, and a genuinely batched caller (inference.py's
    ``_maybe_normalize``) died in the concat instead: ``cat`` of ``(B, 2)`` and
    ``(1,)``.
    """
    a = _assembler()
    vb = torch.randn(3, len(_VARY), _H, _W)
    # A DIFFERENT uniform CO2 per frame — the point is that each survives.
    per_frame = [0.75, 1.5, 2.25]
    for t, v in enumerate(per_frame):
        vb[t, 0] = v
    cal = torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]])
    out = a({"varying_boundary": vb, "calendar": cal})
    assert out["varying_boundary"].shape == (3, len(_VARY) - 1, _H, _W)
    assert out["calendar"].shape == (3, 3)
    # Calendar columns untouched, routed column per-frame.
    torch.testing.assert_close(out["calendar"][:, :2], cal)
    torch.testing.assert_close(
        out["calendar"][:, 2], torch.tensor(per_frame), rtol=0, atol=1e-6
    )


def test_a_batched_sample_routes_per_sample():
    """The inference.py path: (B, C, H, W) with a (B, 2) calendar."""
    a = _assembler()
    vb = torch.randn(2, len(_VARY), _H, _W)
    vb[0, 0] = 1.0
    vb[1, 0] = 400.0
    cal = torch.tensor([[0.0, 12.0], [0.0, 12.0]])
    out = a({"varying_boundary": vb, "calendar": cal})
    assert out["calendar"].shape == (2, 3)
    # Not averaged into a single shared value.
    torch.testing.assert_close(
        out["calendar"][:, 2], torch.tensor([1.0, 400.0]), rtol=0, atol=1e-6
    )


def test_non_uniformity_in_one_batch_element_is_not_excused_by_the_others():
    """Per-sample spread, then the worst — otherwise a bad frame hides."""
    a = _assembler(strict=True)
    vb = torch.randn(2, len(_VARY), _H, _W)
    vb[0, 0] = 1.0                      # uniform
    vb[1, 0] = torch.randn(_H, _W)      # not
    cal = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="not spatially uniform"):
        a({"varying_boundary": vb, "calendar": cal})


def test_sst_rescaler_hook_runs_before_the_pop():
    seen = {}

    def rescaler(sample):
        # 12g contract: SST is still a gridded channel here.
        seen["n_channels"] = sample["varying_boundary"].shape[-3]
        return sample

    a = _assembler(sst_rescaler=rescaler)
    a(_sample())
    assert seen["n_channels"] == len(_VARY)


# ---------------------------------------------------------------------------
# smooth_masked_boundary (12d.14)
# ---------------------------------------------------------------------------


def test_smoothing_preserves_interior_exactly_and_fades_outside():
    torch.manual_seed(0)
    data = torch.zeros(_H, _W)
    mask = torch.zeros(_H, _W)
    mask[:, 8:] = 1.0  # right half valid
    data[:, 8:] = torch.randn(_H, 8)
    out = smooth_masked_boundary(data, mask, sigma=1.5, kernel_size=5, n_iters=10)
    # Interior identical (Dirichlet reset each iteration).
    assert torch.allclose(out[:, 8:], data[:, 8:], atol=0)
    # Outside: non-zero near the seam, decaying away from it.
    assert out[:, 7].abs().mean() > 0
    assert out[:, 7].abs().mean() > out[:, 4].abs().mean()


def test_smoothing_is_longitude_circular():
    data = torch.zeros(4, 8)
    mask = torch.zeros(4, 8)
    mask[:, 0] = 1.0  # valid stripe at lon index 0
    data[:, 0] = 1.0
    out = smooth_masked_boundary(data, mask, sigma=1.0, kernel_size=3, n_iters=3)
    # Bleeds across the wrap seam into the LAST column, not just column 1.
    assert out[0, -1] > 0
    assert out[0, 1] > 0


def test_nan_fill_smooth_mode_fades_to_fill_and_removes_nan():
    vb = torch.full((1, _H, _W), 285.0)
    vb[:, :, :8] = float("nan")  # "land" half
    fill = NanFillTransform(
        varying_boundary_variables=["sst"],
        fill_values={"sst": 270.0},
        smooth_nan_boundaries=True,
        smooth_sigma=1.5,
        smooth_kernel_size=5,
        smooth_n_iters=10,
    )
    out = fill({"varying_boundary": vb.clone()})["varying_boundary"]
    assert torch.isfinite(out).all()
    # Ocean side preserved exactly.
    assert torch.allclose(out[:, :, 8:], vb[:, :, 8:], atol=0)
    # Coast column blends strictly between the fill and the ocean value.
    assert 270.0 < float(out[0, 0, 7]) < 285.0
    # Decay away from the seam, toward the fill. Longitude is CIRCULAR, so
    # this land strip is bounded by ocean on BOTH sides (col 8 and the wrap
    # from col 15): the profile is symmetric with its minimum mid-strip, not
    # at col 0. Assert the decay on the interior-ward direction and pin the
    # wrap symmetry that makes col 0 a coast column too.
    row = out[0, 0]
    assert float(row[3]) < float(row[5]) < float(row[6]) < float(row[7])
    assert float(row[0]) == pytest.approx(float(row[7]), abs=1e-4)
    assert 270.0 <= float(row[3]) < 285.0


def test_nan_fill_hard_mode_still_default_and_exact():
    vb = torch.full((1, _H, _W), 285.0)
    vb[:, :, :8] = float("nan")
    out = NanFillTransform(
        varying_boundary_variables=["sst"], fill_values={"sst": 270.0}
    )({"varying_boundary": vb.clone()})["varying_boundary"]
    assert float(out[0, 0, 0]) == pytest.approx(270.0)
    assert float(out[0, 0, 7]) == pytest.approx(270.0)


def test_nan_free_input_is_untouched_by_smooth_mode():
    vb = torch.randn(2, _H, _W)
    fill = NanFillTransform(
        varying_boundary_variables=["a", "b"],
        smooth_nan_boundaries=True,
    )
    out = fill({"varying_boundary": vb.clone()})["varying_boundary"]
    assert torch.equal(out, vb)
