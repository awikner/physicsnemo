# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PanguWeather PANGU_PLASIM → ai-rossby translator.

Covers both target classes (:class:`PanguPlasim` VAE +
:class:`PanguPlasimLegacy` deterministic) and the DDP / torch.compile
prefix-stripping robustness — the user-flagged production foot-gun.
The end-to-end test confirms the translated model's forward output is
**bit-equivalent** to the source model on identical input, the
"actually reasonable predictions" guarantee.
"""

from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import yaml

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "checkpoint_translation"
sys.path.insert(0, str(_TOOLS_DIR))

from pangu_plasim import (  # noqa: E402
    _strip_wrap_prefixes,
    build_target_model_from_yaml,
    load_panguweather_state_dict,
    translate_state_dict,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.pangu_plasim import (
        PanguPlasim,
        PanguPlasimLegacy,
    )


# Tiny model config — fits CPU, exercises the standard Pangu submodule layout.
_TINY_CFG_LEGACY = dict(
    surface_variables=["pl", "tas"],
    upper_air_variables=["ta", "ua", "va", "hus", "zg"],
    constant_boundary_variables=["lsm", "sg", "z0"],
    varying_boundary_variables=["rsdt", "sst", "sic"],
    diagnostic_variables=["pr_6h"],
    land_variables=[],
    ocean_variables=[],
    levels=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    horizontal_resolution=[16, 32],
    patch_size=[2, 4, 4],
    window_size=[1, 2, 2],
    depths=[1, 1, 1, 1],
    num_heads=[2, 4, 4, 2],
    embed_dim=32,
    updown_scale_factor=2,
    predict_delta=False,
    mask_output=False,
    upper_air_boundary=False,
    vertical_windowing=False,
    subpixel_deconv=True,
    polar_pad=False,
    grid_has_poles=False,
    recovery_head=True,
    diagnostic_head=False,
    has_diagnostic=True,
    drop_rate=0.0,
    checkpointing=0,
    use_reentrant=False,
)

_TINY_CFG_VAE = dict(_TINY_CFG_LEGACY)
# PanguPlasim adds a couple of VAE-specific knobs; everything else mirrors
# the legacy variant so the same fixture sample exercises both paths.


def _write_yaml(path: Path, cfg: dict, target_class: str) -> None:
    blob = dict(cfg, name=target_class)
    with open(path, "w") as fh:
        yaml.safe_dump(blob, fh)


# ---------------------------------------------------------------------------
# Prefix stripper — direct unit tests
# ---------------------------------------------------------------------------


def test_strip_wrap_prefixes_no_prefix():
    assert _strip_wrap_prefixes("layer1.weight") == "layer1.weight"


def test_strip_wrap_prefixes_single_module():
    assert _strip_wrap_prefixes("module.layer1.weight") == "layer1.weight"


def test_strip_wrap_prefixes_single_compile():
    assert _strip_wrap_prefixes("_orig_mod.layer1.weight") == "layer1.weight"


def test_strip_wrap_prefixes_stacked_module_compile():
    assert _strip_wrap_prefixes("module._orig_mod.layer1.weight") == "layer1.weight"
    assert _strip_wrap_prefixes("_orig_mod.module.layer1.weight") == "layer1.weight"


def test_strip_wrap_prefixes_repeated_module():
    """Multi-rank DDP nested in DDP somehow → both prefixes peeled."""
    assert _strip_wrap_prefixes("module.module.layer1.weight") == "layer1.weight"


def test_strip_wrap_prefixes_idempotent():
    once = _strip_wrap_prefixes("module._orig_mod.layer1.weight")
    twice = _strip_wrap_prefixes(once)
    assert once == twice == "layer1.weight"


# ---------------------------------------------------------------------------
# translate_state_dict
# ---------------------------------------------------------------------------


def test_translate_state_dict_strips_module_no_reprefix():
    """Pangu submodule names already align — no `sfno.` re-prefix here."""
    src = OrderedDict(
        {
            "module.layer1.weight": torch.zeros(2),
            "layer2.blocks.0.attn.weight": torch.zeros(2),
        }
    )
    out = translate_state_dict(src)
    assert "layer1.weight" in out
    assert "layer2.blocks.0.attn.weight" in out
    for k in out:
        assert not k.startswith("module.")
        assert not k.startswith("_orig_mod.")


def test_translate_state_dict_handles_stacked_wrapper_prefixes():
    src = OrderedDict(
        {
            "module._orig_mod.layer1.weight": torch.zeros(2),
            "_orig_mod.module.patchembed3d.proj.weight": torch.zeros(2),
        }
    )
    out = translate_state_dict(src)
    assert "layer1.weight" in out
    assert "patchembed3d.proj.weight" in out


# ---------------------------------------------------------------------------
# load_panguweather_state_dict
# ---------------------------------------------------------------------------


def test_load_state_dict_prefers_ema(tmp_path):
    blob = {
        "iters": 1,
        "epoch": 1,
        "model_state": OrderedDict({"a": torch.tensor([1.0])}),
        "ema_state": OrderedDict({"a": torch.tensor([2.0])}),
    }
    p = tmp_path / "ckpt.pt"
    torch.save(blob, p)
    assert float(load_panguweather_state_dict(p, prefer_ema=True)["a"].item()) == 2.0
    assert float(load_panguweather_state_dict(p, prefer_ema=False)["a"].item()) == 1.0


def test_load_state_dict_no_ema_falls_back_to_model_state(tmp_path):
    blob = {"model_state": OrderedDict({"a": torch.tensor([7.0])}), "ema_state": None}
    p = tmp_path / "ckpt.pt"
    torch.save(blob, p)
    assert float(load_panguweather_state_dict(p)["a"].item()) == 7.0


def test_load_state_dict_bare_ordered_dict_round_trip(tmp_path):
    raw = OrderedDict({"layer.weight": torch.zeros(4)})
    p = tmp_path / "ckpt.pt"
    torch.save(raw, p)
    out = load_panguweather_state_dict(p)
    assert "layer.weight" in out


def test_load_state_dict_handles_dot_tar_extension(tmp_path):
    """PanguWeather writes ``ckpt_epoch_*.tar`` / ``best_ckpt.tar`` /
    ``ckpt_latest.tar`` via ``torch.save`` (not actual tarballs).
    ``torch.load`` reads them transparently regardless of suffix —
    confirm the translator's load path accepts them.
    """
    blob = {
        "iters": 1,
        "epoch": 1,
        "model_state": OrderedDict({"module.layer1.weight": torch.tensor([3.14])}),
        "ema_state": OrderedDict({"module.layer1.weight": torch.tensor([2.71])}),
    }
    p = tmp_path / "ckpt_latest.tar"  # PanguWeather's actual filename pattern
    torch.save(blob, p)
    # Default prefers EMA.
    sd = load_panguweather_state_dict(p)
    translated = translate_state_dict(sd)
    # Stripped `module.` prefix, retained the value.
    assert "layer1.weight" in translated
    assert float(translated["layer1.weight"].item()) == pytest.approx(2.71)


# ---------------------------------------------------------------------------
# build_target_model_from_yaml — class resolution
# ---------------------------------------------------------------------------


def test_build_from_yaml_resolves_legacy(tmp_path):
    yaml_path = tmp_path / "legacy.yaml"
    _write_yaml(yaml_path, _TINY_CFG_LEGACY, "PanguPlasimLegacy")
    m = build_target_model_from_yaml(yaml_path)
    assert isinstance(m, PanguPlasimLegacy)


def test_build_from_yaml_resolves_vae(tmp_path):
    yaml_path = tmp_path / "vae.yaml"
    _write_yaml(yaml_path, _TINY_CFG_VAE, "PanguPlasim")
    m = build_target_model_from_yaml(yaml_path)
    assert isinstance(m, PanguPlasim)


def test_build_from_yaml_target_class_override(tmp_path):
    # YAML says Legacy, override to VAE via the CLI flag.
    yaml_path = tmp_path / "amb.yaml"
    _write_yaml(yaml_path, _TINY_CFG_VAE, "PanguPlasimLegacy")
    m = build_target_model_from_yaml(yaml_path, target_class="PanguPlasim")
    assert isinstance(m, PanguPlasim)


def test_build_from_yaml_rejects_unknown_name(tmp_path):
    yaml_path = tmp_path / "bad.yaml"
    _write_yaml(yaml_path, _TINY_CFG_LEGACY, "NotARealClass")
    with pytest.raises(ValueError, match=r"target class"):
        build_target_model_from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# End-to-end: source model → save → translate → load → forward bit-match
# ---------------------------------------------------------------------------


def _build_source_legacy():
    torch.manual_seed(0)
    return PanguPlasimLegacy(**_TINY_CFG_LEGACY)


def _make_inputs(cfg):
    H, W = cfg["horizontal_resolution"]
    n_surface = len(cfg["surface_variables"])
    n_const = len(cfg["constant_boundary_variables"])
    n_varying = len(cfg["varying_boundary_variables"])
    n_upper = len(cfg["upper_air_variables"])
    n_levels = len(cfg["levels"])
    return {
        "surface": torch.randn(1, n_surface, H, W),
        "constant_boundary": torch.randn(1, n_const, H, W),
        "varying_boundary": torch.randn(1, n_varying, H, W),
        "upper_air": torch.randn(1, n_upper, n_levels, H, W),
    }


def test_end_to_end_round_trip_legacy(tmp_path):
    """Source PanguPlasimLegacy → state_dict → translator → fresh PanguPlasimLegacy
    → forward output bit-matches the source for identical inputs."""
    src = _build_source_legacy().eval()
    src_state = OrderedDict(src.state_dict())

    blob = {"model_state": src_state, "ema_state": src_state}
    ckpt = tmp_path / "src.pt"
    torch.save(blob, ckpt)
    yaml_path = tmp_path / "legacy.yaml"
    _write_yaml(yaml_path, _TINY_CFG_LEGACY, "PanguPlasimLegacy")

    loaded_sd = load_panguweather_state_dict(ckpt)
    translated = translate_state_dict(loaded_sd)
    target = build_target_model_from_yaml(yaml_path).eval()
    incoming = target.load_state_dict(translated, strict=False)
    # Pangu submodules align exactly between source and target.
    assert incoming.missing_keys == [], (
        f"unexpected missing keys: {incoming.missing_keys[:5]}"
    )
    assert incoming.unexpected_keys == [], (
        f"unexpected extra keys: {incoming.unexpected_keys[:5]}"
    )

    # Same input → same output, byte-for-byte (modulo float-noise).
    inp = _make_inputs(_TINY_CFG_LEGACY)
    with torch.no_grad():
        out_src = src(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
        out_tgt = target(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
    # The model emits a 6/7-tuple; the first 2 (or 3 with diagnostic) entries
    # are real tensors. The latent placeholders are zero-tensors / ints.
    for i, (a, b) in enumerate(zip(out_src[:3], out_tgt[:3])):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.numel() > 0:
            assert torch.allclose(a, b, atol=1e-6), (
                f"tuple[{i}] differs between source and translated: "
                f"max_diff={(a - b).abs().max().item():.3e}"
            )
            # The output also needs to be *non-degenerate*: not all zeros,
            # not all NaN/inf. (Without this, a translator that zeroed
            # every weight would still pass the bit-match.)
            assert torch.isfinite(a).all(), f"output[{i}] has non-finite values"
            assert (a.abs().max() > 0).item(), f"output[{i}] is all zeros"


def test_end_to_end_round_trip_legacy_with_ddp_wrapped_checkpoint(tmp_path):
    """The same round-trip but the source state_dict carries `module.` on
    every key — the user-flagged production foot-gun."""
    src = _build_source_legacy().eval()
    raw = src.state_dict()
    ddp = OrderedDict((f"module.{k}", v) for k, v in raw.items())
    blob = {"model_state": ddp, "ema_state": ddp}
    ckpt = tmp_path / "src_ddp.pt"
    torch.save(blob, ckpt)
    yaml_path = tmp_path / "legacy.yaml"
    _write_yaml(yaml_path, _TINY_CFG_LEGACY, "PanguPlasimLegacy")

    translated = translate_state_dict(load_panguweather_state_dict(ckpt))
    target = build_target_model_from_yaml(yaml_path).eval()
    incoming = target.load_state_dict(translated, strict=False)
    assert incoming.missing_keys == []
    assert incoming.unexpected_keys == []

    inp = _make_inputs(_TINY_CFG_LEGACY)
    with torch.no_grad():
        out_src = src(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
        out_tgt = target(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
    assert torch.allclose(out_src[0], out_tgt[0], atol=1e-6)
    assert torch.allclose(out_src[1], out_tgt[1], atol=1e-6)


def test_end_to_end_round_trip_legacy_with_stacked_wrappers(tmp_path):
    """`module._orig_mod.` (DDP + torch.compile) round-trip."""
    src = _build_source_legacy().eval()
    raw = src.state_dict()
    stacked = OrderedDict((f"module._orig_mod.{k}", v) for k, v in raw.items())
    blob = {"model_state": stacked, "ema_state": stacked}
    ckpt = tmp_path / "src_stacked.pt"
    torch.save(blob, ckpt)
    yaml_path = tmp_path / "legacy.yaml"
    _write_yaml(yaml_path, _TINY_CFG_LEGACY, "PanguPlasimLegacy")

    translated = translate_state_dict(load_panguweather_state_dict(ckpt))
    target = build_target_model_from_yaml(yaml_path).eval()
    incoming = target.load_state_dict(translated, strict=False)
    assert incoming.missing_keys == []
    assert incoming.unexpected_keys == []

    inp = _make_inputs(_TINY_CFG_LEGACY)
    with torch.no_grad():
        out_src = src(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
        out_tgt = target(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
    assert torch.allclose(out_src[0], out_tgt[0], atol=1e-6)


def test_end_to_end_round_trip_vae(tmp_path):
    """PanguPlasim (VAE) round-trip: same submodule alignment as Legacy,
    plus the VAE-only branches (layer*_e2, mu / sigma heads, etc.)."""
    torch.manual_seed(2)
    src = PanguPlasim(**_TINY_CFG_VAE).eval()
    src_state = OrderedDict(src.state_dict())
    blob = {"model_state": src_state, "ema_state": src_state}
    ckpt = tmp_path / "vae.pt"
    torch.save(blob, ckpt)
    yaml_path = tmp_path / "vae.yaml"
    _write_yaml(yaml_path, _TINY_CFG_VAE, "PanguPlasim")

    translated = translate_state_dict(load_panguweather_state_dict(ckpt))
    target = build_target_model_from_yaml(yaml_path).eval()
    incoming = target.load_state_dict(translated, strict=False)
    assert incoming.missing_keys == []
    assert incoming.unexpected_keys == []

    inp = _make_inputs(_TINY_CFG_VAE)
    # PanguPlasim calls torch.randn_like in its reparameterization
    # regardless of self.training, so back-to-back forwards consume
    # different noise. Re-seed before each call so we're comparing
    # the deterministic mean-field paths bit-for-bit.
    with torch.no_grad():
        torch.manual_seed(99)
        out_src = src(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
        torch.manual_seed(99)
        out_tgt = target(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
    assert torch.allclose(out_src[0], out_tgt[0], atol=1e-6)
    assert torch.allclose(out_src[1], out_tgt[1], atol=1e-6)
    # And the predictions are non-degenerate (real values, not all zero/NaN).
    assert torch.isfinite(out_tgt[0]).all() and (out_tgt[0].abs().max() > 0).item()
    assert torch.isfinite(out_tgt[1]).all() and (out_tgt[1].abs().max() > 0).item()
