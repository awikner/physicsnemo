# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12e tests: RollingDiT backbone parity with amip_v2.

Covers the four features vendored in 12e — budget input projection, mixture
output head, forcing cross-attention, global conditioning — plus the
non-negotiable requirement that ``legacy`` modes leave the pre-12e module
tree and forward **bit-identical** so trained checkpoints still load.

All synthetic, all tiny. Note the small-width constraints the layers enforce
(they are real, not test artifacts): ``state_encoder='column'`` needs
``d_state >= 24`` (three 8-channel-floor blocks), and a ``scalar_dim >= 3``
budget needs ``d_calendar > 8`` so the trend scalar's reserved slice can be
strictly smaller. Upstream's ``dim=1024`` is far above both; these tests pass
explicit budgets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo.experimental.models.amip_si import RollingDiTWrapper
from physicsnemo.experimental.models.amip_si.layers.input_embed import (
    RollingDiTInputEmbed,
)
from physicsnemo.experimental.models.amip_si.layers.output_head import (
    RollingDiTOutputHead,
)
from physicsnemo.experimental.models.amip_si.rolling_dit import RollingDiT

_REF = Path(__file__).parent / "data" / "rolling_dit_legacy_v1.pt"

_COMMON = dict(
    in_channels=20,
    nlat=8,
    nlon=16,
    dim=64,
    num_heads=2,
    temporal_num_heads=2,
    num_blocks=2,
    c_grid_dim=3,
    scalar_dim=3,
    c_grid_downsample=2,
    window_size=3,
)
_LAYOUT = dict(nsurface=2, ndiagnostic=3, nlevels=3, n_upper_air=5)
_BUDGET = {"mode": "budget", "d_boundary": 16, "d_calendar": 16, "d_co2": 8}
_BUDGET_COL = {**_BUDGET, "state_encoder": "column"}
_MIX = {"mode": "mix", "num_experts": 2}
_B, _W = 2, 3


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        z=torch.randn(_B, _W, 20, 8, 16, generator=g),
        t=torch.rand(_B, _W, generator=g),
        c_grid=torch.randn(_B, _W, 3, 16, 32, generator=g),
        c_scalar=torch.randn(_B, _W, 3, generator=g),
    )


def _forward(model, inp):
    with torch.no_grad():
        return model(inp["z"], inp["t"], c_grid=inp["c_grid"], c_scalar=inp["c_scalar"])


def _make_responsive(model, seed: int = 11):
    """Fill every parameter with small deterministic noise.

    Necessary for any "changing an input changes the output" assertion: the
    output head is zero-init by design (``F == 0``, the intended soft start),
    so a freshly-constructed model emits exactly 0 for *every* input and such
    a test would compare zeros to zeros.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


# ---------------------------------------------------------------------------
# Legacy bit-identity — the checkpoint-compatibility guarantee
# ---------------------------------------------------------------------------


def _legacy_reference_model():
    """The exact construction + weight fill used to build the fixture."""
    kw = dict(
        in_channels=20, out_channels=20, nlat=8, nlon=16, dim=64, num_heads=2,
        temporal_num_heads=2, num_blocks=2, c_grid_dim=3, scalar_dim=3,
        c_grid_downsample=2, c_grid_embed_dim=8, c_scalar_embed_dim=8, dropout=0.0,
    )
    torch.manual_seed(1234)
    m = RollingDiT(**kw).eval()
    # Fill every parameter deterministically in sorted-key order: the output
    # head is zero-init, so a bare init would emit exactly 0 and the comparison
    # would be vacuous.
    g = torch.Generator().manual_seed(7)
    named = dict(m.named_parameters())
    with torch.no_grad():
        for k in sorted(m.state_dict().keys()):
            p = named.get(k)
            if p is not None:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return m


def test_legacy_module_tree_and_forward_are_bit_identical():
    """Default (all-legacy) kwargs must reproduce the pre-12e model exactly.

    Fixture generated from the commit before 12e landed
    (``data/rolling_dit_legacy_v1.pt``): state-dict key set, parameter count,
    and the forward output on fixed inputs.
    """
    ref = torch.load(_REF, weights_only=False)
    m = _legacy_reference_model()
    assert sorted(m.state_dict().keys()) == ref["keys"]
    assert sum(p.numel() for p in m.parameters()) == ref["n_params"]

    gi = torch.Generator().manual_seed(99)
    z = torch.randn(2, 3, 20, 8, 16, generator=gi)
    t = torch.rand(2, 3, generator=gi)
    cg = torch.randn(2, 3, 3, 16, 32, generator=gi)
    cs = torch.randn(2, 3, 3, generator=gi)
    with torch.no_grad():
        out = m(z, t, c_grid=cg, c_scalar=cs)
    assert torch.equal(out, ref["out"]), "legacy forward drifted from the pre-12e reference"


def test_legacy_builds_no_new_submodules():
    m = RollingDiT(**_COMMON)
    assert m.input_embed is None
    assert m.output_head is None
    assert m.patch_embed_main is not None  # legacy projection present
    assert m.unpatchify_layer is not None
    assert len(m.forcing_blocks) == 0
    assert m.global_cond is False and m.n_global_cond == 0


def test_budget_and_mix_drop_the_legacy_submodules():
    """The replaced modules must be absent, not dead weight in the state dict."""
    m = RollingDiT(**_COMMON, input_embed=_BUDGET, output_head=_MIX, state_layout=_LAYOUT)
    assert m.patch_embed_main is None
    assert m.c_grid_embed is None and m.scalar_embedder is None
    assert m.unpatchify_layer is None
    keys = " ".join(m.state_dict().keys())
    assert "patch_embed_main" not in keys
    assert "unpatchify_layer" not in keys


# ---------------------------------------------------------------------------
# Constructor / forward sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embed", [None, _BUDGET, _BUDGET_COL], ids=["e-legacy", "e-budget", "e-column"])
@pytest.mark.parametrize("head", [None, _MIX, {**_MIX, "decoder": "column"}], ids=["h-legacy", "h-mix", "h-column"])
@pytest.mark.parametrize("global_cond", [False, True], ids=["gc-off", "gc-on"])
@pytest.mark.parametrize("cross", [0, 2], ids=["x0", "x2"])
def test_feature_sweep_forward_shape_and_finite(embed, head, global_cond, cross):
    m = RollingDiT(
        **_COMMON,
        input_embed=embed,
        output_head=head,
        global_cond=global_cond,
        c_grid_cross_layers=cross,
        c_grid_cross_heads=2,
        state_layout=_LAYOUT,
    ).eval()
    out = _forward(m, _inputs())
    assert out.shape == (_B, _W, 20, 8, 16)
    assert torch.isfinite(out).all()


def test_budget_slice_widths_sum_to_dim():
    m = RollingDiT(**_COMMON, input_embed=_BUDGET_COL, state_layout=_LAYOUT)
    d = m.input_embed.describe()
    assert d["d_state"] + d["d_boundary"] + d["d_calendar"] == _COMMON["dim"]
    assert d["d_calendar_time"] + d["d_calendar_co2"] == d["d_calendar"]
    # Column encoder's internal split also sums to its slice.
    assert d["d_surface"] + d["d_diagnostic"] + d["d_upper_air"] == d["d_state"]
    assert m.input_embed.kv_dim == d["d_boundary"]


def test_budget_tolerates_absent_forcings_where_legacy_cannot():
    """Budget mode zero-fills a missing source; legacy's fixed-width
    projection cannot (pre-existing, documented behavior difference)."""
    inp = _inputs()
    budget = RollingDiT(**_COMMON, input_embed=_BUDGET, state_layout=_LAYOUT).eval()
    with torch.no_grad():
        out = budget(inp["z"], inp["t"])  # no c_grid / c_scalar
    assert torch.isfinite(out).all()

    legacy = RollingDiT(**_COMMON).eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            legacy(inp["z"], inp["t"])


# ---------------------------------------------------------------------------
# Forcing cross-attention
# ---------------------------------------------------------------------------


def test_cross_attention_is_identity_at_init():
    """Zero-init gate ⇒ adding the blocks to a trained model is a no-op.

    Same seed, same weights for every shared module; the only difference is
    the presence of the cross-attention blocks, whose gate is zero.
    """
    torch.manual_seed(5)
    base = RollingDiT(**_COMMON).eval()
    torch.manual_seed(5)
    withx = RollingDiT(**_COMMON, c_grid_cross_layers=2, c_grid_cross_heads=2).eval()
    # Copy the shared parameters so only the new blocks differ.
    withx.load_state_dict(base.state_dict(), strict=False)
    inp = _inputs(1)
    assert torch.allclose(_forward(base, inp), _forward(withx, inp), atol=0)


def test_cross_attention_is_causal_over_the_window():
    """Perturbing a LATER frame's forcing must not change an EARLIER output.

    The mask lets query frame w see forcing frames 0..w only.
    """
    torch.manual_seed(3)
    m = _make_responsive(
        RollingDiT(**_COMMON, c_grid_cross_layers=2, c_grid_cross_heads=2).eval()
    )
    inp = _inputs(2)
    out_a = _forward(m, inp)
    perturbed = {**inp, "c_grid": inp["c_grid"].clone()}
    perturbed["c_grid"][:, -1] += 10.0  # last frame's forcing only
    out_b = _forward(m, perturbed)
    # Frames before the last must be untouched...
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-6)
    # ...and the last frame must actually respond (else nothing was tested).
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-6)


def test_cross_attention_requires_a_boundary_stream():
    with pytest.raises(ValueError, match="c_grid_dim > 0"):
        RollingDiT(**{**_COMMON, "c_grid_dim": 0}, c_grid_cross_layers=1)
    with pytest.raises(ValueError, match="num_blocks"):
        RollingDiT(**_COMMON, c_grid_cross_layers=99)


# ---------------------------------------------------------------------------
# Global conditioning
# ---------------------------------------------------------------------------


def test_global_cond_widens_the_time_embedder_and_uses_doy():
    m = RollingDiT(**_COMMON, global_cond=True).eval()
    assert m.n_global_cond == 2  # doy + trend scalar (scalar_dim=3)
    assert m.t_embedder.num_conds == 3
    # Day-of-year enters the AdaLN vector, so changing it changes the output
    # even though the gridded calendar embedding is unchanged.
    _make_responsive(m)
    inp = _inputs(4)
    out_a = _forward(m, inp)
    other = {**inp, "c_scalar": inp["c_scalar"].clone()}
    other["c_scalar"][..., 1] += 40.0  # day-of-year
    assert not torch.allclose(out_a, _forward(m, other), atol=1e-6)


def test_global_cond_without_trend_scalar_is_one_column():
    m = RollingDiT(**{**_COMMON, "scalar_dim": 2}, global_cond=True)
    assert m.n_global_cond == 1
    assert m.t_embedder.num_conds == 2


def test_global_cond_requires_a_calendar():
    with pytest.raises(ValueError, match="scalar_dim >= 2"):
        RollingDiT(**{**_COMMON, "scalar_dim": 0}, global_cond=True)


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------


def test_mix_head_is_zero_at_init_and_gates_on_cond():
    head = RollingDiTOutputHead(
        dim=32, out_channels=7, nlat=4, nlon=8, cond_dim=32, num_experts=2
    )
    x = torch.randn(2, 32, 32)
    assert torch.equal(
        head(x, torch.randn(2, 32)), torch.zeros(2, 4, 8, 7)
    )  # F == 0 soft start
    # Non-zero experts + gate ⇒ σ (the cond) changes the readout.
    with torch.no_grad():
        for e in head.experts:
            e.weight.normal_(0, 0.1)
        head.gate[-1].weight.normal_(0, 0.1)
    c1, c2 = torch.zeros(2, 32), torch.ones(2, 32)
    assert not torch.allclose(head(x, c1), head(x, c2))


@pytest.mark.parametrize("num_experts", [1, 2, 3])
def test_mix_head_expert_count_shapes(num_experts):
    head = RollingDiTOutputHead(
        dim=16, out_channels=5, nlat=2, nlon=4, cond_dim=16, num_experts=num_experts
    )
    assert len(head.experts) == num_experts
    assert head.gate[-1].out_features == num_experts * 5
    assert head(torch.randn(1, 8, 16), torch.randn(1, 16)).shape == (1, 2, 4, 5)


def test_column_paths_reject_a_layout_that_does_not_add_up():
    bad = dict(nsurface=2, ndiagnostic=3, nlevels=3, n_upper_air=4)  # 2+3+12 != 20
    with pytest.raises(ValueError, match="does not add up"):
        RollingDiTInputEmbed(
            dim=64, in_channels=20, nlat=8, nlon=16, state_encoder="column", **bad
        )
    with pytest.raises(ValueError, match="does not add up"):
        RollingDiTOutputHead(
            dim=64, out_channels=20, nlat=8, nlon=16, cond_dim=64,
            decoder="column", **bad
        )


def test_column_paths_require_the_layout():
    with pytest.raises(ValueError, match="needs"):
        RollingDiTInputEmbed(dim=64, in_channels=20, nlat=8, nlon=16, state_encoder="column")
    with pytest.raises(ValueError, match="needs"):
        RollingDiTOutputHead(dim=64, out_channels=20, nlat=8, nlon=16, cond_dim=64, decoder="column")


def test_unknown_modes_rejected():
    with pytest.raises(ValueError, match="state_encoder"):
        RollingDiTInputEmbed(dim=64, in_channels=20, nlat=8, nlon=16, state_encoder="nope")
    with pytest.raises(ValueError, match="boundary_encoder"):
        RollingDiTInputEmbed(
            dim=64, in_channels=20, nlat=8, nlon=16, c_grid_dim=3, boundary_encoder="nope"
        )
    with pytest.raises(ValueError, match="decoder"):
        RollingDiTOutputHead(dim=64, out_channels=20, nlat=8, nlon=16, cond_dim=64, decoder="nope")


def test_boundary_pool_stats_inert_at_downsample_one():
    """Nothing to pool over when the forcings are already on the token grid."""
    e = RollingDiTInputEmbed(
        dim=64, in_channels=20, nlat=8, nlon=16, c_grid_dim=3,
        c_grid_downsample=1, boundary_pool_stats=True, d_boundary=16,
    )
    assert e.boundary_embed.pool_stats is False


# ---------------------------------------------------------------------------
# Wrapper integration + Muon coverage
# ---------------------------------------------------------------------------


def _wrapper(**rolling_kwargs):
    return RollingDiTWrapper(
        surface_variables=[f"s{i}" for i in range(6)],
        upper_air_variables=[f"u{i}" for i in range(5)],
        diagnostic_variables=[f"d{i}" for i in range(15)],
        constant_boundary_variables=["lsm", "zsfc"],
        varying_boundary_variables=["co2", "toa", "sst", "sic"],
        scalar_routed_boundary_variables=["co2"],
        levels=[float(i) for i in range(26)],
        horizontal_resolution=(8, 16),
        scalar_dim=3,
        channel_layout="v2",
        rolling_dit_kwargs=dict(
            dim=64, num_heads=2, temporal_num_heads=2, num_blocks=2,
            c_grid_downsample=2, window_size=3, **rolling_kwargs,
        ),
    )


def test_wrapper_derives_state_layout_for_the_column_paths():
    """A config never restates the state block sizes (the 12b lesson)."""
    w = _wrapper(
        input_embed=_BUDGET_COL, output_head={**_MIX, "decoder": "column"}
    )
    assert w.state_layout() == dict(
        nsurface=6, ndiagnostic=15, nlevels=26, n_upper_air=5, nocean=0
    )
    # The column encoder sized itself from that, not from a config echo.
    assert w.backbone.input_embed.state_embed.nsurface == 6
    assert w.backbone.input_embed.state_embed.nlevels == 26


@pytest.mark.parametrize(
    "kw",
    [
        {},
        dict(input_embed=_BUDGET_COL),
        dict(output_head={**_MIX, "decoder": "column"}),
        dict(global_cond=True, c_grid_cross_layers=2, c_grid_cross_heads=2),
        dict(
            input_embed=_BUDGET_COL,
            output_head={**_MIX, "decoder": "column"},
            global_cond=True,
            c_grid_cross_layers=2,
            c_grid_cross_heads=2,
        ),
    ],
    ids=["legacy", "budget", "mix", "gc+cross", "all-on"],
)
def test_muon_groups_cover_every_backbone_param_exactly_once(kw):
    w = _wrapper(**kw)
    groups = w.muon_param_groups(lr=1e-4)
    ids = [id(p) for g in groups for p in g["params"]]
    covered = sum(p.numel() for g in groups for p in g["params"])
    total = sum(p.numel() for p in w.backbone.parameters())
    assert len(ids) == len(set(ids)), "a parameter landed in two groups"
    assert covered == total, f"{total - covered} parameters missing from the optimizer"


def test_forcing_positional_tables_go_to_adamw_not_muon():
    """They are 2-D but position tables, so Muon's orthogonalisation is wrong."""
    w = _wrapper(c_grid_cross_layers=2, c_grid_cross_heads=2)
    groups = w.muon_param_groups(lr=1e-4)
    muon_ids = {
        id(p) for g in groups if g.get("use_muon") for p in g["params"]
    }
    for name, p in w.backbone.forcing_blocks.named_parameters():
        if name.endswith(("temporal_pos", "query_pos")):
            assert id(p) not in muon_ids, name
        elif name.endswith("weight") and p.ndim >= 2:
            assert id(p) in muon_ids, name


def test_wrapper_forward_with_all_features(tmp_path):
    w = _wrapper(
        input_embed=_BUDGET_COL,
        output_head={**_MIX, "decoder": "column"},
        global_cond=True,
        c_grid_cross_layers=2,
        c_grid_cross_heads=2,
    ).eval()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, 3, w.in_channels, 8, 16, generator=g)
    c_grid = torch.randn(1, 3, w.c_grid_dim, 16, 32, generator=g)
    c_scalar = torch.randn(1, 3, 3, generator=g)
    with torch.no_grad():
        out = w(z, torch.rand(1, 3, generator=g), c_grid=c_grid, c_scalar=c_scalar)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()

    # .mdlus round-trip keeps the feature configuration.
    path = tmp_path / "w.mdlus"
    w.save(str(path))
    from physicsnemo import Module

    loaded = Module.from_checkpoint(str(path))
    assert loaded.backbone.input_embed is not None
    assert loaded.backbone.output_head is not None
    assert len(loaded.backbone.forcing_blocks) == 2
    assert loaded.backbone.global_cond is True
