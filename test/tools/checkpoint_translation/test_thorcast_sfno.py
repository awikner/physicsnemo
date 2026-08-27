# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ThorCast SFNO -> PanguWeather-layout translator.

The translator's only real job is the input-channel permutation, so the central
test is an *equivalence* test rather than a shape check: build one base SFNO,
drive it with a ThorCast-ordered input vector, then drive a
:class:`SfnoPlasim` holding the permuted weights with the same fields routed
the ai-rossby way, and require the two to agree. That construction fails if the
permutation is wrong in either direction, and — importantly — fails if the
``big_skip`` block of ``decoder.0.weight`` is left unpermuted.
"""

from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "checkpoint_translation"
sys.path.insert(0, str(_TOOLS_DIR))

from thorcast_sfno import (  # noqa: E402
    PANGU_INPUT_GROUP_ORDER,
    THORCAST_DEFAULTS,
    THORCAST_INPUT_GROUP_ORDER,
    derive_permutation,
    expand_channels,
    load_thorcast_checkpoint,
    permute_state_dict,
    strip_wrap_prefixes,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.modulus_sfno import (
        SphericalFourierNeuralOperatorNet,
    )
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim


# Tiny stand-in for sfno_plasim_5412.yaml: same variable-group *structure* and
# the same group ordering, small enough to run on CPU in a test.
_TARGET_CFG = dict(
    surface_variables=["pl", "tas"],
    upper_air_variables=["ta", "ua", "va", "hus", "zg"],
    constant_boundary_variables=["lsm", "sg", "z0"],   # note: sg before z0
    varying_boundary_variables=["sst", "rsdt", "sic"],
    diagnostic_variables=["pr_6h"],
    levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    horizontal_resolution=[16, 32],
    embed_dim=32,
    num_layers=2,
    num_blocks=2,
    spectral_layers=2,
    encoder_layers=1,
    spectral_transform="sht",
    filter_type="linear",
    operator_type="dhconv",
    scale_factor=1,
    use_mlp=True,
    mlp_ratio=2.0,
    activation_function="gelu",
    pos_embed=True,
    normalization_layer="instance_norm",
    hard_thresholding_fraction=1.0,
    big_skip=True,
    data_grid="legendre-gauss",
)

_N_LEVELS = len(_TARGET_CFG["levels"])
_N_SURFACE = len(_TARGET_CFG["surface_variables"])
_N_UPPER = len(_TARGET_CFG["upper_air_variables"])
_N_CONST = len(_TARGET_CFG["constant_boundary_variables"])
_N_VARYING = len(_TARGET_CFG["varying_boundary_variables"])
_IN_CHANS = _N_SURFACE + _N_CONST + _N_VARYING + _N_UPPER * _N_LEVELS


def _target_groups() -> dict:
    return {
        "surface": _TARGET_CFG["surface_variables"],
        "upper_air": _TARGET_CFG["upper_air_variables"],
        "constant_boundary": _TARGET_CFG["constant_boundary_variables"],
        "varying_boundary": _TARGET_CFG["varying_boundary_variables"],
        "diagnostic": _TARGET_CFG["diagnostic_variables"],
    }


def _input_perm() -> list[int]:
    src = expand_channels(THORCAST_DEFAULTS, THORCAST_INPUT_GROUP_ORDER, _N_LEVELS)
    tgt = expand_channels(_target_groups(), PANGU_INPUT_GROUP_ORDER, _N_LEVELS)
    return derive_permutation(src, tgt)


# ---------------------------------------------------------------------------
# Permutation derivation
# ---------------------------------------------------------------------------


def test_permutation_matches_hand_derived_layout():
    """The derived gather must match the layout worked out from both sources."""
    # ThorCast: pl,tas | ta*L,ua*L,va*L,hus*L,zg*L | lsm,z0,sg | sst,rsdt,sic
    # Target:   pl,tas | lsm,sg,z0 | sst,rsdt,sic | ta*L,...,zg*L
    n_up = _N_UPPER * _N_LEVELS
    lsm, z0, sg = 2 + n_up, 3 + n_up, 4 + n_up
    sst, rsdt, sic = 5 + n_up, 6 + n_up, 7 + n_up
    expected = [0, 1, lsm, sg, z0, sst, rsdt, sic] + list(range(2, 2 + n_up))
    assert _input_perm() == expected


def test_permutation_is_a_bijection():
    perm = _input_perm()
    assert len(perm) == _IN_CHANS
    assert sorted(perm) == list(range(_IN_CHANS))


def test_constant_boundary_swap_is_captured():
    """z0/sg are swapped between ThorCast and the repo configs."""
    src = expand_channels(THORCAST_DEFAULTS, THORCAST_INPUT_GROUP_ORDER, _N_LEVELS)
    tgt = expand_channels(_target_groups(), PANGU_INPUT_GROUP_ORDER, _N_LEVELS)
    perm = derive_permutation(src, tgt)
    # Position 3 of the target is `sg`; it must pull ThorCast's `sg` slot.
    assert tgt[3] == "sg"
    assert src[perm[3]] == "sg"
    assert tgt[4] == "z0" and src[perm[4]] == "z0"


def test_output_order_is_identity():
    """surface|upper_air|diagnostic on both sides, so no decoder.2 rewrite."""
    from thorcast_sfno import (
        PANGU_OUTPUT_GROUP_ORDER,
        THORCAST_OUTPUT_GROUP_ORDER,
    )

    src = expand_channels(THORCAST_DEFAULTS, THORCAST_OUTPUT_GROUP_ORDER, _N_LEVELS)
    tgt = expand_channels(_target_groups(), PANGU_OUTPUT_GROUP_ORDER, _N_LEVELS)
    perm = derive_permutation(src, tgt)
    assert perm == list(range(len(perm)))


def test_mismatched_variable_sets_raise():
    """A wrong variable list must fail loudly, not produce silent garbage."""
    bad = dict(THORCAST_DEFAULTS, surface=["pl", "NOT_A_VAR"])
    src = expand_channels(bad, THORCAST_INPUT_GROUP_ORDER, _N_LEVELS)
    tgt = expand_channels(_target_groups(), PANGU_INPUT_GROUP_ORDER, _N_LEVELS)
    with pytest.raises(ValueError, match="different channel sets"):
        derive_permutation(src, tgt)


def test_level_count_mismatch_raises():
    src = expand_channels(THORCAST_DEFAULTS, THORCAST_INPUT_GROUP_ORDER, _N_LEVELS + 1)
    tgt = expand_channels(_target_groups(), PANGU_INPUT_GROUP_ORDER, _N_LEVELS)
    with pytest.raises(ValueError, match="channel count differs"):
        derive_permutation(src, tgt)


def test_strip_wrap_prefixes_is_idempotent():
    assert strip_wrap_prefixes("module._orig_mod.encoder.0.weight") == "encoder.0.weight"
    once = strip_wrap_prefixes("module.encoder.0.weight")
    assert strip_wrap_prefixes(once) == once


# ---------------------------------------------------------------------------
# Weight surgery
# ---------------------------------------------------------------------------


def _build_thorcast_style_sfno() -> SphericalFourierNeuralOperatorNet:
    """A base SFNO with ThorCast's flat (in_chans, out_chans) contract."""

    class _GridParams:
        data_grid = _TARGET_CFG["data_grid"]

    out_chans = _N_SURFACE + _N_UPPER * _N_LEVELS + len(
        _TARGET_CFG["diagnostic_variables"]
    )
    kwargs = {
        k: v
        for k, v in _TARGET_CFG.items()
        if k
        not in (
            "surface_variables",
            "upper_air_variables",
            "constant_boundary_variables",
            "varying_boundary_variables",
            "diagnostic_variables",
            "levels",
            "horizontal_resolution",
            "data_grid",
        )
    }
    return SphericalFourierNeuralOperatorNet(
        params=_GridParams(),
        img_shape=tuple(_TARGET_CFG["horizontal_resolution"]),
        in_chans=_IN_CHANS,
        out_chans=out_chans,
        **kwargs,
    )


def test_permute_state_dict_rewrites_only_the_expected_tensors():
    torch.manual_seed(0)
    model = _build_thorcast_style_sfno()
    sd = OrderedDict(model.state_dict())
    perm = _input_perm()
    out = permute_state_dict(sd, input_perm=perm, output_perm=None)

    changed = {k for k in sd if not torch.equal(sd[k], out[k])}
    assert changed == {"encoder.0.weight", "decoder.0.weight"}, changed
    # The latent half of the big-skip concat must be untouched.
    embed = _TARGET_CFG["embed_dim"]
    torch.testing.assert_close(
        out["decoder.0.weight"][:, :embed], sd["decoder.0.weight"][:, :embed]
    )


def test_permute_state_dict_rejects_wrong_perm_length():
    model = _build_thorcast_style_sfno()
    sd = OrderedDict(model.state_dict())
    with pytest.raises(ValueError, match="input_perm has"):
        permute_state_dict(sd, input_perm=list(range(_IN_CHANS - 1)))


def test_permute_state_dict_rejects_non_sfno_dict():
    with pytest.raises(KeyError, match="encoder.0.weight"):
        permute_state_dict(OrderedDict(foo=torch.zeros(1)), input_perm=[0])


# ---------------------------------------------------------------------------
# End-to-end numerical equivalence — the test that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
def test_permuted_weights_reproduce_thorcast_outputs(batch):
    """ThorCast-ordered base SFNO == SfnoPlasim on the permuted weights.

    Same physical fields, two different channel conventions, one permutation
    between the weight sets. If the permutation (or the big_skip half of it) is
    wrong, the two disagree.
    """
    torch.manual_seed(1234)
    nlat, nlon = _TARGET_CFG["horizontal_resolution"]

    surface = torch.randn(batch, _N_SURFACE, nlat, nlon)
    const = torch.randn(batch, _N_CONST, nlat, nlon)
    varying = torch.randn(batch, _N_VARYING, nlat, nlon)
    upper = torch.randn(batch, _N_UPPER, _N_LEVELS, nlat, nlon)
    upper_flat = upper.reshape(batch, _N_UPPER * _N_LEVELS, nlat, nlon)

    # `const` is supplied in the TARGET order (lsm, sg, z0) because that is what
    # the datapipe hands the model. ThorCast's vector wants (lsm, z0, sg).
    tc_const = const[:, [0, 2, 1]]
    thorcast_input = torch.cat((surface, upper_flat, tc_const, varying), dim=1)

    base = _build_thorcast_style_sfno().eval()
    with torch.no_grad():
        reference = base(thorcast_input)

    perm = _input_perm()
    permuted = permute_state_dict(
        OrderedDict(base.state_dict()), input_perm=perm, output_perm=None
    )

    wrapper = SfnoPlasim(**_TARGET_CFG).eval()
    incoming = wrapper.load_state_dict(
        OrderedDict((f"sfno.{k}", v) for k, v in permuted.items()), strict=False
    )
    assert not incoming.missing_keys, incoming.missing_keys
    assert not incoming.unexpected_keys, incoming.unexpected_keys

    with torch.no_grad():
        out_surface, out_upper, out_diag, *_ = wrapper(
            surface, const, varying, upper
        )

    n_up = _N_UPPER * _N_LEVELS
    torch.testing.assert_close(out_surface, reference[:, :_N_SURFACE], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        out_upper,
        reference[:, _N_SURFACE : _N_SURFACE + n_up].reshape(
            batch, _N_UPPER, _N_LEVELS, nlat, nlon
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        out_diag, reference[:, _N_SURFACE + n_up :], rtol=1e-5, atol=1e-6
    )


def test_unpermuted_weights_do_not_match():
    """Guard against a vacuous equivalence test.

    If loading the *raw* ThorCast weights into SfnoPlasim happened to agree with
    the reference, the test above would prove nothing. It must not agree.
    """
    torch.manual_seed(7)
    nlat, nlon = _TARGET_CFG["horizontal_resolution"]
    surface = torch.randn(1, _N_SURFACE, nlat, nlon)
    const = torch.randn(1, _N_CONST, nlat, nlon)
    varying = torch.randn(1, _N_VARYING, nlat, nlon)
    upper = torch.randn(1, _N_UPPER, _N_LEVELS, nlat, nlon)
    upper_flat = upper.reshape(1, _N_UPPER * _N_LEVELS, nlat, nlon)

    base = _build_thorcast_style_sfno().eval()
    with torch.no_grad():
        reference = base(
            torch.cat((surface, upper_flat, const[:, [0, 2, 1]], varying), dim=1)
        )

    wrapper = SfnoPlasim(**_TARGET_CFG).eval()
    wrapper.load_state_dict(
        OrderedDict((f"sfno.{k}", v) for k, v in base.state_dict().items()),
        strict=False,
    )
    with torch.no_grad():
        out_surface, *_ = wrapper(surface, const, varying, upper)

    assert not torch.allclose(
        out_surface, reference[:, :_N_SURFACE], rtol=1e-3, atol=1e-4
    ), "unpermuted weights matched — the equivalence test would be vacuous"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def test_load_thorcast_checkpoint_round_trip(tmp_path):
    """A ThorCast-shaped payload loads, strips `module.`, and drops _metadata."""
    model = _build_thorcast_style_sfno()
    ddp_sd = OrderedDict(
        (f"module.{k}", v) for k, v in model.state_dict().items()
    )
    ddp_sd["_metadata"] = {"": {"version": 1}}
    path = tmp_path / "thorcast.pth"
    torch.save(
        {
            "epoch": 50,
            "step": 228250,
            "model_state_dict": ddp_sd,
            "optimizer_state_dict": {"dummy": 1},
            "scheduler_state_dict": {"dummy": 2},
        },
        path,
    )

    sd, provenance = load_thorcast_checkpoint(path)
    assert provenance == {"epoch": 50, "step": 228250}
    assert "_metadata" not in sd
    assert all(not k.startswith("module.") for k in sd)
    assert set(sd) == set(model.state_dict())


def test_target_groups_read_from_real_config():
    """The shipped sfno_plasim_5412.yaml must yield the 58->53 contract."""
    from thorcast_sfno import load_target_groups

    cfg_path = (
        Path(__file__).resolve().parents[3]
        / "examples/weather/ai_rossby/conf/model/sfno_plasim_5412.yaml"
    )
    groups, n_levels = load_target_groups(cfg_path)
    assert n_levels == 10
    n_in = (
        len(groups["surface"])
        + len(groups["constant_boundary"])
        + len(groups["varying_boundary"])
        + len(groups["upper_air"]) * n_levels
    )
    n_out = (
        len(groups["surface"])
        + len(groups["diagnostic"])
        + len(groups["upper_air"]) * n_levels
    )
    assert (n_in, n_out) == (58, 53)

    # And the real config must agree with ThorCast on channel *content*.
    src = expand_channels(THORCAST_DEFAULTS, THORCAST_INPUT_GROUP_ORDER, n_levels)
    tgt = expand_channels(groups, PANGU_INPUT_GROUP_ORDER, n_levels)
    perm = derive_permutation(src, tgt)
    assert sorted(perm) == list(range(58))
    assert perm != list(range(58)), "expected a non-trivial permutation"
