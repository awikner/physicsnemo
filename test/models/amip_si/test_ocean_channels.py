# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12f — ``ocean_state_variables`` on :class:`RollingDiTWrapper`.

The wrapper owns the whole ocean contract: how wide the state axis gets, and
which ``c_grid`` channels the truth is read from. Both are *derived* from the
variable lists that build the pack, so a config cannot size the model
differently from the data. These tests pin the derivation and every way it is
allowed to fail loudly.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.amip_si import RollingDiTWrapper

_SURFACE = ["skin_temperature", "surface_pressure"]
_UA = ["temperature", "u_component_of_wind"]
_DIAG = ["PRATEsfc_24h"]
_CONST = ["geopotential_at_surface", "land_sea_mask"]
_VARY = [
    "global_mean_co2",
    "DSWRFtoa_24h_lead",
    "sea_surface_temperature_monthly_interp",
    "sea_ice_cover_monthly_interp",
]
_LEVELS = [500.0, 850.0]
_H, _W = 8, 16

_OCEAN = [
    "sea_surface_temperature_monthly_interp",
    "sea_ice_cover_monthly_interp",
]

# The Phase-12e projections; the ocean block is refused on the legacy ones.
_BUDGET_KWARGS = dict(
    dim=64,
    num_heads=2,
    num_blocks=1,
    window_size=3,
    input_embed={"mode": "budget", "d_boundary": 16, "d_calendar": 16},
    output_head={"mode": "mix", "num_experts": 2},
)


def _wrapper(*, ocean=(), routed=("global_mean_co2",), layout="v2", **kw):
    dit = dict(_BUDGET_KWARGS)
    dit.update(kw.pop("rolling_dit_kwargs", {}))
    return RollingDiTWrapper(
        surface_variables=_SURFACE,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        constant_boundary_variables=_CONST,
        varying_boundary_variables=_VARY,
        levels=_LEVELS,
        horizontal_resolution=(_H, _W),
        scalar_dim=2 + len(routed),
        channel_layout=layout,
        scalar_routed_boundary_variables=list(routed),
        ocean_state_variables=list(ocean),
        rolling_dit_kwargs=dit,
        **kw,
    )


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_ocean_channels_widen_the_state_axis():
    n_state = len(_SURFACE) + len(_DIAG) + len(_UA) * len(_LEVELS)
    plain = _wrapper()
    ocean = _wrapper(ocean=_OCEAN)
    assert plain.in_channels == n_state
    assert ocean.num_state_channels == n_state
    assert ocean.in_channels == n_state + 2
    assert ocean.backbone.in_channels == n_state + 2
    assert ocean.backbone.out_channels == n_state + 2


def test_c_grid_width_is_unaffected_by_ocean_prediction():
    # SST is still a forcing: predicting it adds a state channel, it does not
    # remove an input.
    assert _wrapper().c_grid_dim == _wrapper(ocean=_OCEAN).c_grid_dim


def test_ocean_grid_indices_skip_the_scalar_routed_channel():
    w = _wrapper(ocean=_OCEAN)
    # Stored varying order is [co2, dswrf, sst, ice]; co2 is popped into the
    # calendar row, so the stream that reaches c_grid is [dswrf, sst, ice].
    assert w.active_varying_boundary_variables == _VARY[1:]
    assert w.ocean_grid_indices == [1, 2]


def test_ocean_grid_indices_follow_the_requested_order():
    w = _wrapper(ocean=list(reversed(_OCEAN)))
    assert w.ocean_grid_indices == [2, 1]


def test_ocean_grid_indices_address_the_assembled_c_grid():
    """The indices must be valid against ``c_grid``, not just the varying stream.

    That only holds because the v1/v2 c_grid layout is ``[varying | constant]``
    — the varying channels are a prefix. This test is the one that would fail
    if the pack order were ever flipped.
    """
    w = _wrapper(ocean=_OCEAN)
    B, W = 2, 3
    n_active = len(_VARY) - 1
    varying = torch.arange(n_active, dtype=torch.float32)
    varying = varying.view(1, 1, n_active, 1, 1).expand(B, W, n_active, _H, _W)
    const = torch.full((B, W, len(_CONST), _H, _W), -1.0)
    c_grid = w.pack_window_c_grid(
        {
            "surface_in": torch.zeros(B, W, len(_SURFACE), _H, _W),
            "constant_boundary": const,
            "varying_boundary": varying,
        }
    )
    for want, idx in zip(w.ocean_state_variables, w.ocean_grid_indices):
        channel = c_grid[:, :, idx]
        expected = float(w.active_varying_boundary_variables.index(want))
        assert torch.all(channel == expected)


def test_state_layout_reports_nocean():
    assert _wrapper().state_layout()["nocean"] == 0
    layout = _wrapper(ocean=_OCEAN).state_layout()
    assert layout["nocean"] == 2
    # The state blocks keep their sizes: the ocean block is a tail, so a
    # no-ocean checkpoint's projections stay addressable.
    plain = _wrapper().state_layout()
    assert {k: v for k, v in layout.items() if k != "nocean"} == {
        k: v for k, v in plain.items() if k != "nocean"
    }


def test_projections_receive_nocean():
    w = _wrapper(ocean=_OCEAN)
    assert w.backbone.nocean == 2
    assert w.backbone.input_embed.nocean == 2
    assert w.backbone.input_embed.n_state == w.num_state_channels
    assert w.backbone.output_head.nocean == 2
    assert w.backbone.output_head.n_state_out == w.num_state_channels
    assert w.backbone.input_embed.ocean_embed is not None


def test_ocean_embed_is_zero_init_so_the_added_input_starts_inert():
    # Warm-starting from a no-ocean checkpoint must not perturb the forecast on
    # step one: the new input path contributes exactly nothing at init.
    embed = _wrapper(ocean=_OCEAN).backbone.input_embed.ocean_embed
    assert torch.all(embed.weight == 0)
    assert torch.all(embed.bias == 0)


# ---------------------------------------------------------------------------
# Forcing alignment
# ---------------------------------------------------------------------------


def test_forcing_lag_follows_the_channel_layout():
    assert _wrapper(layout="v2").forcing_lag == 1
    assert _wrapper(layout="v1").forcing_lag == 1
    assert _wrapper(layout="fork", routed=()).forcing_lag == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fork_layout_is_refused():
    with pytest.raises(ValueError, match="varying-first c_grid layout"):
        _wrapper(ocean=_OCEAN, layout="fork", routed=())


def test_scalar_routed_variable_cannot_be_predicted():
    with pytest.raises(ValueError, match="scalar-routed"):
        _wrapper(ocean=["global_mean_co2"])


def test_unknown_variable_is_refused():
    with pytest.raises(ValueError, match="not one of the gridded"):
        _wrapper(ocean=["not_a_variable"])


def test_duplicate_ocean_variables_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        _wrapper(ocean=_OCEAN[:1] * 2)


def test_ocean_without_varying_boundary_is_refused():
    with pytest.raises(ValueError, match="needs gridded forcings"):
        RollingDiTWrapper(
            surface_variables=_SURFACE,
            upper_air_variables=_UA,
            constant_boundary_variables=_CONST,
            varying_boundary_variables=[],
            levels=_LEVELS,
            horizontal_resolution=(_H, _W),
            channel_layout="v2",
            ocean_state_variables=_OCEAN,
            rolling_dit_kwargs=dict(_BUDGET_KWARGS),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"input_embed": None},
        {"output_head": None},
    ],
)
def test_legacy_projections_refuse_the_ocean_block(override):
    # Widening PatchEmbed / Unpatchify would silently discard the trained
    # projection instead of extending it, so this is a hard error.
    with pytest.raises(ValueError, match="legacy"):
        _wrapper(ocean=_OCEAN, rolling_dit_kwargs=override)


# ---------------------------------------------------------------------------
# Forward / pack
# ---------------------------------------------------------------------------


def test_forward_round_trips_the_widened_channel_axis():
    torch.manual_seed(0)
    w = _wrapper(ocean=_OCEAN).eval()
    B, W = 1, 3
    x = torch.randn(B, W, w.in_channels, _H, _W)
    c_grid = torch.randn(B, W, w.c_grid_dim, _H, _W)
    c_scalar = torch.randn(B, W, w.scalar_dim)
    with torch.no_grad():
        out = w(x, torch.zeros(B, W), c_grid=c_grid, c_scalar=c_scalar)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_unpack_window_state_drops_the_ocean_tail():
    w = _wrapper(ocean=_OCEAN)
    B, W = 2, 3
    x = torch.randn(B, W, w.in_channels, _H, _W)
    out = w.unpack_window_state(x)
    assert out["surface_in"].shape == (B, W, len(_SURFACE), _H, _W)
    assert out["diagnostic"].shape == (B, W, len(_DIAG), _H, _W)
    assert out["upper_air_in"].shape == (B, W, len(_UA), len(_LEVELS), _H, _W)
    total = sum(
        v.shape[2] * (v.shape[3] if v.dim() == 6 else 1) for v in out.values()
    )
    assert total == w.num_state_channels


def test_unpack_state_works_on_a_single_frame():
    """The name the rolling drivers actually call.

    ``validate_diffusion.py`` / ``inference.py`` score one emitted frame at a
    time via ``unpack_state``; the rolling wrappers only defined
    ``unpack_window_state``, so any rolling validation or inference run died
    with an AttributeError on its first scored frame. The two are the same
    code — every block is read at ``narrow(-3, ...)`` — which also means the
    ocean tail falls off here exactly as it does for a window.
    """
    w = _wrapper(ocean=_OCEAN)
    B = 3
    frame = torch.randn(B, w.in_channels, _H, _W)
    out = w.unpack_state(frame)
    assert out["surface_in"].shape == (B, len(_SURFACE), _H, _W)
    assert out["diagnostic"].shape == (B, len(_DIAG), _H, _W)
    assert out["upper_air_in"].shape == (B, len(_UA), len(_LEVELS), _H, _W)
    # And it agrees with the window path on the same data.
    win = w.unpack_window_state(frame.unsqueeze(1))
    for key in out:
        assert torch.equal(out[key], win[key].squeeze(1))


def test_pack_unpack_state_round_trip_ignores_ocean():
    w = _wrapper(ocean=_OCEAN)
    B, W = 2, 3
    sample = {
        "surface_in": torch.randn(B, W, len(_SURFACE), _H, _W),
        "diagnostic": torch.randn(B, W, len(_DIAG), _H, _W),
        "upper_air_in": torch.randn(B, W, len(_UA), len(_LEVELS), _H, _W),
    }
    packed = w.pack_window_state(sample)
    assert packed.shape[2] == w.num_state_channels
    back = w.unpack_window_state(packed)
    for key, value in sample.items():
        assert torch.allclose(back[key], value)


def test_muon_param_groups_still_cover_every_parameter():
    # The ocean block adds parameters to the input projection and the head; a
    # grouping that missed them would leave them permanently at init.
    w = _wrapper(ocean=_OCEAN)
    groups = w.muon_param_groups(lr=1e-4)
    grouped = {id(p) for g in groups for p in g["params"]}
    backbone = {id(p) for p in w.backbone.parameters()}
    assert backbone == grouped
    assert sum(len(g["params"]) for g in groups) == len(backbone)
