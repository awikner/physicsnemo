# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PanguWeather SFNO → ai-rossby SfnoPlasim translator.

Synthesizes a PanguWeather-shaped ``.pt`` blob (dict with ``model_state`` and
``ema_state`` OrderedDicts) by training a tiny PanguWeather-shaped SFNO_v2 in
the test, dumping its state dict, then running the translator end-to-end and
verifying the loaded SfnoPlasim produces matching outputs.
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

from sfno_plasim import (  # noqa: E402
    build_target_model_from_yaml,
    load_panguweather_state_dict,
    translate_state_dict,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.modulus_sfno import (
        SphericalFourierNeuralOperatorNet,
    )
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim


# Tiny model config — must match both the PanguWeather SFNO_v2 base class and
# the ai-rossby SfnoPlasim constructor.
_TINY_CFG = dict(
    surface_variables=["pl", "tas"],
    upper_air_variables=["ta", "ua", "va", "hus", "zg"],
    constant_boundary_variables=["lsm", "sg", "z0"],
    varying_boundary_variables=["rsdt", "sst", "sic"],
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
    normalization_layer="instance_norm",
)


class _GridParams:
    def __init__(self, grid: str) -> None:
        self.data_grid = grid


def _build_panguweather_shaped_base():
    """Instantiate the bare SphericalFourierNeuralOperatorNet w/ the same kwargs
    SfnoPlasim uses internally. Mirrors PanguWeather's SFNO_v2.super().__init__().
    """
    n_surface = len(_TINY_CFG["surface_variables"])
    n_const = len(_TINY_CFG["constant_boundary_variables"])
    n_varying = len(_TINY_CFG["varying_boundary_variables"])
    n_diag = len(_TINY_CFG["diagnostic_variables"])
    n_upper = len(_TINY_CFG["upper_air_variables"])
    n_levels = len(_TINY_CFG["levels"])
    return SphericalFourierNeuralOperatorNet(
        params=_GridParams("equiangular"),
        spectral_transform=_TINY_CFG["spectral_transform"],
        filter_type=_TINY_CFG["filter_type"],
        operator_type=_TINY_CFG["operator_type"],
        img_shape=tuple(_TINY_CFG["horizontal_resolution"]),
        scale_factor=1,
        in_chans=n_surface + n_const + n_varying + n_upper * n_levels,
        out_chans=n_surface + n_diag + n_upper * n_levels,
        embed_dim=_TINY_CFG["embed_dim"],
        num_layers=_TINY_CFG["num_layers"],
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        encoder_layers=_TINY_CFG["encoder_layers"],
        pos_embed=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=_TINY_CFG["num_blocks"],
        sparsity_threshold=0.0,
        normalization_layer=_TINY_CFG["normalization_layer"],
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=True,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=_TINY_CFG["spectral_layers"],
        checkpointing=0,
    )


def _write_yaml(path: Path) -> None:
    """Write the tiny test config as the format the translator's YAML loader expects."""
    cfg = dict(_TINY_CFG, name="sfno_plasim_tiny", target="...", model_type="SfnoPlasim")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)


def test_translate_state_dict_strips_module_prefix_and_adds_sfno():
    """`module.X` (DDP) → `sfno.X`; plain `X` → `sfno.X`."""
    src = OrderedDict(
        {"module.encoder.weight": torch.zeros(2), "blocks.0.norm.bias": torch.zeros(2)}
    )
    out = translate_state_dict(src)
    assert "sfno.encoder.weight" in out
    assert "sfno.blocks.0.norm.bias" in out
    assert "module." not in next(iter(out.keys()))


def test_load_panguweather_state_dict_prefers_ema(tmp_path):
    """When both `model_state` and `ema_state` are present, EMA wins by default."""
    blob = {
        "iters": 100,
        "epoch": 5,
        "model_state": OrderedDict({"a": torch.tensor([1.0])}),
        "ema_state": OrderedDict({"a": torch.tensor([2.0])}),
    }
    p = tmp_path / "ckpt.pt"
    torch.save(blob, p)
    sd = load_panguweather_state_dict(p, prefer_ema=True)
    assert float(sd["a"].item()) == 2.0  # EMA path
    sd = load_panguweather_state_dict(p, prefer_ema=False)
    assert float(sd["a"].item()) == 1.0  # model_state path


def test_load_panguweather_state_dict_falls_back_to_model_state(tmp_path):
    """Older checkpoints have model_state only (no ema_state)."""
    blob = {
        "iters": 100,
        "epoch": 5,
        "model_state": OrderedDict({"a": torch.tensor([7.0])}),
        "ema_state": None,
    }
    p = tmp_path / "ckpt.pt"
    torch.save(blob, p)
    sd = load_panguweather_state_dict(p)
    assert float(sd["a"].item()) == 7.0


def test_load_panguweather_state_dict_treats_raw_dict(tmp_path):
    """Some pre-PanguWeather checkpoints save the state dict directly."""
    blob = OrderedDict({"layer.weight": torch.zeros(4)})
    p = tmp_path / "ckpt.pt"
    torch.save(blob, p)
    sd = load_panguweather_state_dict(p)
    assert "layer.weight" in sd


def test_end_to_end_translation_preserves_outputs(tmp_path):
    """Synthesize a PanguWeather-shaped checkpoint, translate, and verify the
    translated SfnoPlasim's forward output matches the source base SFNO's
    output (after the wrapper's reshape contract)."""
    torch.manual_seed(0)
    base = _build_panguweather_shaped_base().eval()
    # PanguWeather-style checkpoint dict.
    blob = {
        "iters": 1,
        "epoch": 1,
        "model_state": OrderedDict(base.state_dict()),
        "ema_state": OrderedDict(base.state_dict()),  # tied for the test
    }
    ckpt = tmp_path / "panguweather.pt"
    torch.save(blob, ckpt)

    yaml_path = tmp_path / "model.yaml"
    _write_yaml(yaml_path)

    # Translate.
    src_sd = load_panguweather_state_dict(ckpt)
    tgt_sd = translate_state_dict(src_sd)
    target_model = build_target_model_from_yaml(yaml_path)
    incoming = target_model.load_state_dict(tgt_sd, strict=False)
    # The base SFNO + SfnoPlasim wrapper share the same parameters, so all
    # base-SFNO keys land cleanly with the `sfno.` prefix.
    assert incoming.missing_keys == []
    assert incoming.unexpected_keys == []

    # Verify the forward output of the SfnoPlasim matches what the base SFNO
    # would produce for the same flattened input — modulo the wrapper's
    # input/output reshape (which the test reconstructs manually).
    target_model.eval()
    B = 1
    H, W = _TINY_CFG["horizontal_resolution"]
    n_surface = len(_TINY_CFG["surface_variables"])
    n_const = len(_TINY_CFG["constant_boundary_variables"])
    n_varying = len(_TINY_CFG["varying_boundary_variables"])
    n_upper = len(_TINY_CFG["upper_air_variables"])
    n_levels = len(_TINY_CFG["levels"])

    surface = torch.randn(B, n_surface, H, W)
    c_bound = torch.randn(B, n_const, H, W)
    v_bound = torch.randn(B, n_varying, H, W)
    upper = torch.randn(B, n_upper, n_levels, H, W)

    with torch.no_grad():
        out_target = target_model(surface, c_bound, v_bound, upper)
        # Recreate what the bare SFNO would see.
        upper_flat = upper.view(B, n_upper * n_levels, H, W)
        x = torch.cat((surface, c_bound, v_bound, upper_flat), dim=1)
        out_base_flat = base(x)
        n_diag = len(_TINY_CFG["diagnostic_variables"])
        # Slice the bare-SFNO output back into the wrapper's shape contract.
        expected_surface = out_base_flat[:, :n_surface]
        expected_upper = out_base_flat[:, n_surface : n_surface + n_upper * n_levels].view(
            B, n_upper, n_levels, H, W
        )
        expected_diag = out_base_flat[:, n_surface + n_upper * n_levels :]
    assert torch.allclose(out_target[0], expected_surface, atol=1e-6)
    assert torch.allclose(out_target[1], expected_upper, atol=1e-6)
    if n_diag:
        assert torch.allclose(out_target[2], expected_diag, atol=1e-6)
