# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit + live-validation tests for the upstream amip Lightning → ai-rossby translator.

Pure unit tests exercise the prefix-strip / re-prefix logic, the
source ``model_name`` detection (including hard errors on out-of-scope
families), and the auto-derive-from-hparams path on a synthetic
Lightning blob.

A parameterized live-validation test sweeps over every ``last.ckpt``
inside ``/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/``, runs
the full translation, loads the resulting ``.mdlus`` back via
``physicsnemo.Module.from_checkpoint``, and asserts the wrapper does
a finite forward pass on synthetic inputs at the upstream's
two-resolution convention (x_noised / cond at backbone working res,
c_grid at data res). The live test auto-skips when the checkpoint
tree isn't mounted (so CI on non-Delta hosts still passes).
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

from amip_si import (  # noqa: E402
    _MODEL_NAME_TO_WRAPPER,
    _UNSUPPORTED_MODEL_NAMES,
    _strip_wrap_prefixes,
    build_target_wrapper,
    cross_check_compatibility,
    detect_model_name,
    pick_source_state_dict,
    translate_state_dict,
    wrapper_kwargs_from_hparams,
)


# ---------------------------------------------------------------------------
# Synthetic Lightning blob fixture.
# ---------------------------------------------------------------------------


def _make_state_dict(prefix: str = "model.") -> OrderedDict:
    """Build a backbone-shaped state dict for the synthetic AmipDiT.

    Sized to match the tiny configuration in ``_make_hparams`` so the
    end-to-end load test produces zero missing/unexpected keys.
    """
    sd: OrderedDict = OrderedDict()
    # Backbone weights with the right channel-group shapes for
    # in_channels=12 (2*6), out_channels=6, dim=16, c_grid_dim=2.
    sd[prefix + "c_grid_embed.weight"] = torch.zeros(8, 2, 2, 2)
    sd[prefix + "c_grid_embed.bias"] = torch.zeros(8)
    # Scheduler buffer the translator must drop.
    sd["scheduler.noise_scales"] = torch.zeros(4)
    # DDP-wrapped key to exercise the prefix-strip path.
    sd["module._orig_mod." + prefix + "extra.weight"] = torch.zeros(3)
    return sd


def _make_hparams(model_name: str = "SI_X") -> dict:
    """Minimal Lightning hparams blob matching the synthetic state dict."""
    return {
        "config": {
            "data": {
                "surface_variables": ["t2m", "msl"],
                "upper_air_variables": ["ta"],
                "diagnostic_variables": ["rsds"],
                "constant_boundary_variables": ["lsm"],
                "varying_boundary_variables": ["sst"],
                "diagnostic_input": True,
                "levels": [1000.0, 850.0, 500.0],
                "horizontal_resolution": [16, 32],
            },
            "model": {
                "model_name": model_name,
                model_name: {
                    "model": {
                        "dim": 16,
                        "num_heads": 4,
                        "num_blocks": 1,
                        "patch_size": 2,
                        "scalar_dim": 2,
                        "c_grid_embed_dim": 8,
                        "c_scalar_embed_dim": 8,
                        "c_grid_downsample": 1,
                        "nlat": 16,
                        "nlon": 32,
                        "num_ca_blocks": 0,
                        "num_output_blocks": 0,
                        # Backbone-derived keys the translator should drop.
                        "in_channels": 6,
                        "out_channels": 6,
                        "c_grid_dim": 2,
                    },
                    "scheduler": {"num_steps": 10},
                },
            },
            "training": {"ema_decay": 0.99, "max_epochs": 50},
        }
    }


def _make_blob(model_name: str = "SI_X") -> dict:
    return {
        "epoch": 5,
        "global_step": 1000,
        "state_dict": _make_state_dict(),
        "current_model_state": _make_state_dict(),
        "averaging_state": {"n_averaged": torch.tensor(1000)},
        "hyper_parameters": _make_hparams(model_name),
    }


# ---------------------------------------------------------------------------
# Prefix strip + state-dict translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,expected",
    [
        ("module.foo", "foo"),
        ("_orig_mod.foo", "foo"),
        ("module._orig_mod.foo", "foo"),
        ("_orig_mod.module.foo", "foo"),
        ("foo.bar", "foo.bar"),
        ("module.module._orig_mod.foo", "foo"),
    ],
)
def test_strip_wrap_prefixes_idempotent(src, expected):
    assert _strip_wrap_prefixes(src) == expected
    assert _strip_wrap_prefixes(_strip_wrap_prefixes(src)) == expected


def test_translate_state_dict_reprefixes_and_drops_scheduler():
    sd = _make_state_dict()
    out, stats = translate_state_dict(sd)
    # model.c_grid_embed.weight → backbone.c_grid_embed.weight
    assert "backbone.c_grid_embed.weight" in out
    assert "backbone.c_grid_embed.bias" in out
    # DDP-wrapped key got peeled to model.extra.weight → backbone.extra.weight.
    assert "backbone.extra.weight" in out
    # scheduler.* dropped.
    assert not any(k.startswith("scheduler.") for k in out)
    # Sanity on accounting.
    assert stats["kept"] == 3
    assert stats["dropped_scheduler"] == 1
    assert stats["dropped_unknown"] == 0


# ---------------------------------------------------------------------------
# Model-name detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["SI", "SI_X", "ERDM", "RFM", "EDM"])
def test_detect_model_name_supported(name):
    blob = _make_blob(model_name=name)
    assert detect_model_name(blob) == name


@pytest.mark.parametrize("name", _UNSUPPORTED_MODEL_NAMES)
def test_detect_model_name_unsupported_raises(name):
    blob = _make_blob(model_name=name)
    with pytest.raises(NotImplementedError, match="not supported"):
        detect_model_name(blob)


def test_detect_model_name_unknown_raises():
    blob = _make_blob(model_name="HypothesisNet")
    with pytest.raises(ValueError, match="unrecognized"):
        detect_model_name(blob)


def test_detect_model_name_missing_hparams_raises():
    blob = {"state_dict": OrderedDict()}
    with pytest.raises(KeyError, match="hyper_parameters"):
        detect_model_name(blob)


# ---------------------------------------------------------------------------
# Source state-dict picker
# ---------------------------------------------------------------------------


def test_pick_state_dict_prefers_ema_by_default():
    blob = _make_blob()
    sd = pick_source_state_dict(blob, prefer_live=False)
    assert isinstance(sd, OrderedDict)
    assert "model.c_grid_embed.weight" in sd


def test_pick_state_dict_prefer_live_picks_current_model_state():
    blob = _make_blob()
    blob["state_dict"]["model.c_grid_embed.weight"] = torch.ones(8, 2, 2, 2)
    blob["current_model_state"]["model.c_grid_embed.weight"] = torch.full(
        (8, 2, 2, 2), 5.0
    )
    sd_ema = pick_source_state_dict(blob, prefer_live=False)
    sd_live = pick_source_state_dict(blob, prefer_live=True)
    assert sd_ema["model.c_grid_embed.weight"][0, 0, 0, 0].item() == 1.0
    assert sd_live["model.c_grid_embed.weight"][0, 0, 0, 0].item() == 5.0


def test_pick_state_dict_missing_state_dict_raises():
    blob = {"current_model_state": OrderedDict()}
    with pytest.raises(KeyError, match="state_dict"):
        pick_source_state_dict(blob, prefer_live=False)


def test_pick_state_dict_missing_current_model_state_raises():
    blob = {"state_dict": OrderedDict()}
    with pytest.raises(KeyError, match="current_model_state"):
        pick_source_state_dict(blob, prefer_live=True)


# ---------------------------------------------------------------------------
# Cross-check compatibility
# ---------------------------------------------------------------------------


def test_cross_check_warns_on_family_mismatch(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="amip_si"):
        cross_check_compatibility("ERDM", "AmipDiTWrapper")
    msgs = [r.message for r in caplog.records]
    assert any("source model_name='ERDM' expects" in m for m in msgs)


def test_cross_check_silent_on_match(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="amip_si"):
        cross_check_compatibility("SI_X", "AmipDiTWrapper")
    assert not any("expects" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# wrapper_kwargs_from_hparams
# ---------------------------------------------------------------------------


def test_wrapper_kwargs_from_hparams_drops_derived_keys():
    blob = _make_blob("SI_X")
    kwargs = wrapper_kwargs_from_hparams(blob, "AmipDiTWrapper")
    assert kwargs["surface_variables"] == ["t2m", "msl"]
    assert kwargs["upper_air_variables"] == ["ta"]
    assert kwargs["diagnostic_variables"] == ["rsds"]
    assert kwargs["levels"] == [1000.0, 850.0, 500.0]
    dit = kwargs["dit_kwargs"]
    # in_channels / out_channels / c_grid_dim are stripped.
    assert "in_channels" not in dit
    assert "out_channels" not in dit
    assert "c_grid_dim" not in dit
    # nlat / nlon are preserved (legacy two-res layout).
    assert dit["nlat"] == 16
    assert dit["nlon"] == 32
    # scalar_dim is hoisted to the top-level wrapper kwarg, not dit_kwargs.
    assert "scalar_dim" not in dit
    assert kwargs["scalar_dim"] == 2


def test_wrapper_kwargs_from_hparams_diagnostic_input_false_empties_diagnostic():
    blob = _make_blob("SI_X")
    blob["hyper_parameters"]["config"]["data"]["diagnostic_input"] = False
    kwargs = wrapper_kwargs_from_hparams(blob, "AmipDiTWrapper")
    assert kwargs["diagnostic_variables"] == []


def test_wrapper_kwargs_from_hparams_rolling_uses_rolling_dit_kwargs():
    blob = _make_blob("RFM")
    kwargs = wrapper_kwargs_from_hparams(blob, "RollingDiTWrapper")
    assert "rolling_dit_kwargs" in kwargs
    assert "dit_kwargs" not in kwargs


# ---------------------------------------------------------------------------
# Class mapping completeness
# ---------------------------------------------------------------------------


def test_model_name_to_wrapper_mapping_covers_all_supported():
    assert set(_MODEL_NAME_TO_WRAPPER) == {"SI", "SI_X", "ERDM", "RFM", "EDM"}
    assert set(_MODEL_NAME_TO_WRAPPER.values()) == {
        "AmipDiTWrapper",
        "ERDMWrapper",
        "RollingDiTWrapper",
    }


# ---------------------------------------------------------------------------
# Live validation against Midway3-transferred ckpts.
# ---------------------------------------------------------------------------

_MIDWAY_CKPT_ROOT = Path("/work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs")

# Run dirs whose state_dict's backbone shape can't be loaded into the
# vendored ``AmipDiT`` even after auto-derive — used to mark xfail
# cases in the parametrized live test. The reason is recorded so
# ``pytest -ra`` lists it; the source ckpts are otherwise intact.
_KNOWN_ARCHITECTURE_MISMATCHES = {
    # Predates the vendored ``CalendarEmbedding`` (May 27 amip commit
    # 497827e). Its ``scalar_embedder.out_proj`` is shape (32, 3) —
    # a plain ``Linear(scalar_dim → c_scalar_embed_dim)``. The
    # vendored code wraps that in sinusoidal embeddings and ends up
    # at shape (32, 192). Would need either a re-vendor of the older
    # backbone or a hand-written shim — out of scope for Phase 8e.
    "SI_V_new_42_2026-05-20T20-47-08": (
        "older ScalarEmbedder variant, predates vendored commit 497827e"
    ),
}


def _collect_midway_ckpts() -> list[Path]:
    if not _MIDWAY_CKPT_ROOT.exists():
        return []
    # Translator MVP supports the SI / SI_X families; x_DDC is deferred
    # to Phase 8f and intentionally errors out in detect_model_name.
    candidates: list[Path] = []
    for d in sorted(_MIDWAY_CKPT_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("x_DDC") or name.startswith("Combined"):
            continue
        last = d / "last.ckpt"
        if last.exists():
            candidates.append(last)
    return candidates


def _ckpt_id(p) -> str:
    return p.parent.name if isinstance(p, Path) else str(p)


@pytest.mark.slow
@pytest.mark.parametrize(
    "ckpt_path",
    _collect_midway_ckpts(),
    ids=_ckpt_id,
)
def test_live_translation_round_trips(ckpt_path, tmp_path, request):
    name = _ckpt_id(ckpt_path)
    if name in _KNOWN_ARCHITECTURE_MISMATCHES:
        request.applymarker(
            pytest.mark.xfail(
                reason=_KNOWN_ARCHITECTURE_MISMATCHES[name],
                strict=True,
            )
        )
    """Translate → load → forward, asserting finite output.

    Skips when the Midway3 checkpoint tree isn't mounted (CI on
    non-Delta hosts) — the parametrize collection comes back empty in
    that case, so pytest reports a no-tests-collected for this case
    without erroring.
    """
    import physicsnemo
    from amip_si import (
        build_target_wrapper,
        detect_model_name,
        load_lightning_ckpt,
        pick_source_state_dict,
        translate_state_dict,
    )

    blob = load_lightning_ckpt(ckpt_path)
    src_model_name = detect_model_name(blob)
    model = build_target_wrapper(blob=blob, source_model_name=src_model_name)
    src_sd = pick_source_state_dict(blob)
    tgt_sd, stats = translate_state_dict(src_sd)
    incoming = model.load_state_dict(tgt_sd, strict=False)
    # The translator MUST produce a state dict that the wrapper accepts
    # with zero unexpected keys. Missing keys are tolerated (pos_embed is
    # computed at forward time, not stored as a parameter).
    assert not incoming.unexpected_keys, (
        f"unexpected keys for {ckpt_path.parent.name}: "
        f"{incoming.unexpected_keys[:5]}"
    )

    out_path = tmp_path / f"{ckpt_path.parent.name}.mdlus"
    model.save(str(out_path))
    loaded = physicsnemo.Module.from_checkpoint(str(out_path))
    loaded.eval()

    # Two-resolution forward: x_noised / cond at backbone working res,
    # c_grid at data res (upstream amip's legacy layout).
    B = 1
    nlat_b, nlon_b = loaded.backbone.nlat, loaded.backbone.nlon
    nlat_d, nlon_d = loaded.horizontal_resolution
    xn = torch.randn(B, loaded.in_channels, nlat_b, nlon_b)
    cond = torch.randn(B, loaded.in_channels, nlat_b, nlon_b)
    t = torch.tensor([0.5])
    c_grid = torch.randn(B, loaded.c_grid_dim, nlat_d, nlon_d)
    c_scalar = torch.randn(B, loaded.scalar_dim)
    with torch.no_grad():
        out = loaded(xn, cond, t, c_grid=c_grid, c_scalar=c_scalar)
    assert out.shape == (B, loaded.in_channels, nlat_b, nlon_b)
    assert torch.isfinite(out).all(), (
        f"non-finite output for {ckpt_path.parent.name}"
    )
