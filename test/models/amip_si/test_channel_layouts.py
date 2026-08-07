# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12b channel-contract tests (amip_v2 rebaseline).

Pins the three packing contracts (``fork`` / ``v1`` / ``v2``) against
vendored copies of upstream amip's ``assemble_input`` — no amip / amip_v2
import at test time — plus the layout round-trips, the c_grid ordering,
the fixed v1<->v2 permutation the translator's ``--source-contract``
relies on, and ``.mdlus`` layout provenance.

Upstream references:
* v1: ``amip@497827e common/utils.py`` —
  ``rearrange(multilevel, "b c l h w -> b (c l) h w")``,
  channel order ``[surface | diagnostic | upper_air]``,
  ``assemble_forcing(forcing, invariant)`` = ``[varying | constant]``.
* v2: ``amip_v2@e0b7b60 common/utils.py`` —
  ``rearrange(multilevel.flip(2), "b c l h w -> b (l c) h w")``,
  same group order.
"""

from __future__ import annotations

import pytest
import torch
from einops import rearrange

from physicsnemo.experimental.models.amip_si import (
    AmipDiTWrapper,
    ERDMWrapper,
    RollingDiTWrapper,
    XDDCWrapper,
)

_SURFACE, _UA_VARS, _DIAG = 3, 2, 4
_LEVELS = [100.0, 500.0, 850.0]  # config order (top -> surface, upstream-style)
_H, _W = 8, 16


# ---------------------------------------------------------------------------
# Vendored upstream references (single-frame, batched (b, ...) tensors).
# ---------------------------------------------------------------------------


def upstream_v1_assemble(surface, multilevel, diagnostic):
    multilevel = rearrange(multilevel, "b c l h w -> b (c l) h w")
    return torch.cat((surface, diagnostic, multilevel), dim=1)


def upstream_v2_assemble(surface, multilevel, diagnostic):
    multilevel = rearrange(multilevel.flip(2), "b c l h w -> b (l c) h w")
    return torch.cat((surface, diagnostic, multilevel), dim=1)


def upstream_assemble_forcing(forcing, invariant):
    return torch.cat((forcing, invariant), dim=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rolling_wrapper(layout: str) -> RollingDiTWrapper:
    return RollingDiTWrapper(
        surface_variables=[f"s{i}" for i in range(_SURFACE)],
        upper_air_variables=[f"u{i}" for i in range(_UA_VARS)],
        diagnostic_variables=[f"d{i}" for i in range(_DIAG)],
        constant_boundary_variables=["lsm", "zsfc"],
        varying_boundary_variables=["sst", "sic", "toa"],
        levels=list(_LEVELS),
        horizontal_resolution=(_H, _W),
        channel_layout=layout,
        rolling_dit_kwargs=dict(dim=16, num_heads=2, num_blocks=1),
    )


def _xddc_wrapper(layout: str) -> XDDCWrapper:
    return XDDCWrapper(
        surface_variables=[f"s{i}" for i in range(_SURFACE)],
        upper_air_variables=[f"u{i}" for i in range(_UA_VARS)],
        diagnostic_variables=[f"d{i}" for i in range(_DIAG)],
        levels=list(_LEVELS),
        horizontal_resolution=(_H, _W),
        channel_layout=layout,
        unet_kwargs=dict(model_channels=8, channel_mult=(1,), num_res_blocks=1, num_groups=4),
    )


def _window_sample(B=2, W=3):
    torch.manual_seed(0)
    return {
        "surface_in": torch.randn(B, W, _SURFACE, _H, _W),
        "upper_air_in": torch.randn(B, W, _UA_VARS, len(_LEVELS), _H, _W),
        "diagnostic": torch.randn(B, W, _DIAG, _H, _W),
        "constant_boundary": torch.randn(2, _H, _W),
        "varying_boundary": torch.randn(B, W, 3, _H, _W),
        "calendar": torch.randn(B, W, 2),
    }


def _single_sample(B=2):
    torch.manual_seed(1)
    return {
        "surface_in": torch.randn(B, _SURFACE, _H, _W),
        "upper_air_in": torch.randn(B, _UA_VARS, len(_LEVELS), _H, _W),
        "diagnostic": torch.randn(B, _DIAG, _H, _W),
    }


# ---------------------------------------------------------------------------
# Bit-parity against the vendored upstream references
# ---------------------------------------------------------------------------


def test_rolling_v2_pack_bitmatches_upstream_v2_assemble():
    w = _rolling_wrapper("v2")
    s = _window_sample()
    packed = w.pack_window_state(s)  # (B, W, C, H, W)
    for t in range(s["surface_in"].shape[1]):
        ref = upstream_v2_assemble(
            s["surface_in"][:, t], s["upper_air_in"][:, t], s["diagnostic"][:, t]
        )
        assert torch.equal(packed[:, t], ref)


def test_rolling_v1_pack_bitmatches_upstream_v1_assemble():
    w = _rolling_wrapper("v1")
    s = _window_sample()
    packed = w.pack_window_state(s)
    for t in range(s["surface_in"].shape[1]):
        ref = upstream_v1_assemble(
            s["surface_in"][:, t], s["upper_air_in"][:, t], s["diagnostic"][:, t]
        )
        assert torch.equal(packed[:, t], ref)


def test_xddc_pack_bitmatches_upstream_assemble():
    s = _single_sample()
    ref_v2 = upstream_v2_assemble(
        s["surface_in"], s["upper_air_in"], s["diagnostic"]
    )
    ref_v1 = upstream_v1_assemble(
        s["surface_in"], s["upper_air_in"], s["diagnostic"]
    )
    assert torch.equal(_xddc_wrapper("v2").pack_state(s), ref_v2)
    assert torch.equal(_xddc_wrapper("v1").pack_state(s), ref_v1)


def test_rolling_v2_c_grid_matches_upstream_forcing_order():
    # Upstream: assemble_forcing(forcing, invariant) = [varying | constant].
    w = _rolling_wrapper("v2")
    s = _window_sample()
    c_grid = w.pack_window_c_grid(s)
    B, W = s["surface_in"].shape[:2]
    const = s["constant_boundary"].expand(B, W, -1, -1, -1)
    for t in range(W):
        ref = upstream_assemble_forcing(s["varying_boundary"][:, t], const[:, t])
        assert torch.equal(c_grid[:, t], ref)


def test_rolling_fork_layout_is_preserved_bit_identical():
    # The frozen pre-12b fork behavior: [surface | ua(var-major) | diag],
    # c_grid [constant | varying]. Guards the frozen ERDMWrapper path too
    # (same mixin, channel_layout="fork").
    w = _rolling_wrapper("fork")
    s = _window_sample()
    packed = w.pack_window_state(s)
    B, W = s["surface_in"].shape[:2]
    ua_flat = s["upper_air_in"].reshape(B, W, _UA_VARS * len(_LEVELS), _H, _W)
    ref = torch.cat([s["surface_in"], ua_flat, s["diagnostic"]], dim=2)
    assert torch.equal(packed, ref)
    c_grid = w.pack_window_c_grid(s)
    const = s["constant_boundary"].expand(B, W, -1, -1, -1)
    ref_c = torch.cat([const, s["varying_boundary"]], dim=2)
    assert torch.equal(c_grid, ref_c)


def _erdm_wrapper(layout: str = "fork") -> ERDMWrapper:
    return ERDMWrapper(
        surface_variables=[f"s{i}" for i in range(_SURFACE)],
        upper_air_variables=[f"u{i}" for i in range(_UA_VARS)],
        diagnostic_variables=[f"d{i}" for i in range(_DIAG)],
        constant_boundary_variables=["lsm", "zsfc"],
        varying_boundary_variables=["sst", "sic", "toa"],
        levels=list(_LEVELS),
        horizontal_resolution=(_H, _W),
        channel_layout=layout,
        erdm_kwargs=dict(
            model_channels=8, channel_mult=(1,), num_res_blocks=1, num_groups=4
        ),
    )


def test_erdm_wrapper_defaults_to_fork_layout_and_rejects_v2():
    # Frozen family: default preserves the historical fork packing; the
    # "v1" option is the Phase-12b correctness fix; "v2" never applies.
    assert _erdm_wrapper().channel_layout == "fork"
    with pytest.raises(ValueError, match="channel_layout"):
        _erdm_wrapper("v2")


def test_erdm_wrapper_v1_pack_bitmatches_upstream_v1_assemble():
    w = _erdm_wrapper("v1")
    s = _window_sample()
    packed = w.pack_window_state(s)
    for t in range(s["surface_in"].shape[1]):
        ref = upstream_v1_assemble(
            s["surface_in"][:, t], s["upper_air_in"][:, t], s["diagnostic"][:, t]
        )
        assert torch.equal(packed[:, t], ref)


def _amip_dit_wrapper(layout: str = "fork") -> AmipDiTWrapper:
    return AmipDiTWrapper(
        surface_variables=[f"s{i}" for i in range(_SURFACE)],
        upper_air_variables=[f"u{i}" for i in range(_UA_VARS)],
        diagnostic_variables=[f"d{i}" for i in range(_DIAG)],
        constant_boundary_variables=["lsm", "zsfc"],
        varying_boundary_variables=["sst", "sic", "toa"],
        levels=list(_LEVELS),
        horizontal_resolution=(_H, _W),
        channel_layout=layout,
        dit_kwargs=dict(dim=16, num_heads=2, num_blocks=1, patch_size=2),
    )


def test_amip_dit_wrapper_defaults_to_fork_layout_and_rejects_v2():
    assert _amip_dit_wrapper().channel_layout == "fork"
    with pytest.raises(ValueError, match="channel_layout"):
        _amip_dit_wrapper("v2")


def test_amip_dit_v1_pack_bitmatches_upstream_v1_assemble():
    w = _amip_dit_wrapper("v1")
    s = _single_sample()
    ref = upstream_v1_assemble(s["surface_in"], s["upper_air_in"], s["diagnostic"])
    assert torch.equal(w.pack_state(s), ref)
    out = w.unpack_state(w.pack_state(s))
    assert torch.equal(out["upper_air_in"], s["upper_air_in"])
    assert torch.equal(out["diagnostic"], s["diagnostic"])


def test_amip_dit_v1_c_grid_matches_upstream_forcing_order():
    w = _amip_dit_wrapper("v1")
    torch.manual_seed(2)
    B = 2
    s = {
        "surface_in": torch.randn(B, _SURFACE, _H, _W),
        "constant_boundary": torch.randn(2, _H, _W),
        "varying_boundary": torch.randn(B, 3, _H, _W),
    }
    const = s["constant_boundary"].expand(B, -1, -1, -1)
    ref = upstream_assemble_forcing(s["varying_boundary"], const)
    assert torch.equal(w.pack_c_grid(s), ref)


def test_amip_dit_fork_layout_is_preserved_bit_identical():
    # The frozen pre-12b default: [surface | ua(var-major) | diag],
    # c_grid [constant | varying].
    w = _amip_dit_wrapper("fork")
    s = _single_sample()
    B = s["surface_in"].shape[0]
    ua_flat = s["upper_air_in"].reshape(B, _UA_VARS * len(_LEVELS), _H, _W)
    ref = torch.cat([s["surface_in"], ua_flat, s["diagnostic"]], dim=1)
    assert torch.equal(w.pack_state(s), ref)


# ---------------------------------------------------------------------------
# Round-trips + layout metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["fork", "v1", "v2"])
def test_rolling_pack_unpack_roundtrip(layout):
    w = _rolling_wrapper(layout)
    s = _window_sample()
    out = w.unpack_window_state(w.pack_window_state(s))
    assert torch.equal(out["surface_in"], s["surface_in"])
    assert torch.equal(out["upper_air_in"], s["upper_air_in"])
    assert torch.equal(out["diagnostic"], s["diagnostic"])


@pytest.mark.parametrize("layout", ["v1", "v2"])
def test_xddc_pack_unpack_roundtrip(layout):
    w = _xddc_wrapper(layout)
    s = _single_sample()
    out = w.unpack_state(w.pack_state(s))
    assert torch.equal(out["surface_in"], s["surface_in"])
    assert torch.equal(out["upper_air_in"], s["upper_air_in"])
    assert torch.equal(out["diagnostic"], s["diagnostic"])


def test_state_layout_reports_block_sizes():
    w = _rolling_wrapper("v2")
    assert w.state_layout() == {
        "nsurface": _SURFACE,
        "ndiagnostic": _DIAG,
        "nlevels": len(_LEVELS),
        "n_upper_air": _UA_VARS,
        "nocean": 0,
    }
    assert _xddc_wrapper("v2").state_layout()["nsurface"] == _SURFACE


def test_invalid_layout_rejected():
    with pytest.raises(ValueError, match="channel_layout"):
        _rolling_wrapper("v3")
    with pytest.raises(ValueError, match="channel_layout"):
        _xddc_wrapper("fork")  # x_DDC has no fork layout


def test_mdlus_roundtrip_preserves_channel_layout(tmp_path):
    w = _rolling_wrapper("v1")
    path = tmp_path / "rolling_v1.mdlus"
    w.save(str(path))
    from physicsnemo import Module

    loaded = Module.from_checkpoint(str(path))
    assert loaded.channel_layout == "v1"
    s = _window_sample()
    assert torch.equal(loaded.pack_window_state(s), w.pack_window_state(s))


# ---------------------------------------------------------------------------
# The v1 <-> v2 fixed permutation (what --source-contract relies on)
# ---------------------------------------------------------------------------


def test_v1_to_v2_is_a_fixed_channel_permutation():
    # For any sample, pack_v2(sample) == pack_v1(sample)[:, :, perm] where
    # perm depends only on the layout constants: identity on
    # [surface | diag], and ua slot (l, c) <- (c, L-1-l).
    L, C = len(_LEVELS), _UA_VARS
    n_head = _SURFACE + _DIAG
    perm = list(range(n_head))
    for l in range(L):
        for c in range(C):
            src_l = L - 1 - l  # v2 flips levels: slot l reads config level L-1-l
            perm.append(n_head + c * L + src_l)
    perm = torch.tensor(perm)

    s = _window_sample()
    x_v1 = _rolling_wrapper("v1").pack_window_state(s)
    x_v2 = _rolling_wrapper("v2").pack_window_state(s)
    assert torch.equal(x_v2, x_v1.index_select(2, perm))
