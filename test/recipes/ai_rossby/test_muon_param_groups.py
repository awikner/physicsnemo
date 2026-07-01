# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8f (F1) unit tests for the Muon param-group split.

Covers the ``muon_param_groups()`` method on each amip_si wrapper
(:class:`AmipDiTWrapper`, :class:`RollingDiTWrapper`, :class:`ERDMWrapper`)
plus the ``optimizer_type="Muon"`` branch in
:func:`train_loop.make_optimizer`'s error paths (the ``muon`` package
itself is not required to run these — only the partitioning logic and
the guard rails are exercised).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.amip_si import (
        AmipDiTWrapper,
        ERDMWrapper,
        RollingDiTWrapper,
    )

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from train_loop import make_optimizer  # noqa: E402


def _kwargs() -> dict:
    return dict(
        surface_variables=["a", "b", "c"],
        upper_air_variables=["ta", "ua"],
        diagnostic_variables=["p1"],
        constant_boundary_variables=["lsm"],
        varying_boundary_variables=["sst", "co2"],
        levels=[100.0, 500.0, 1000.0],
        horizontal_resolution=(32, 64),
    )


def _amip_dit_wrapper() -> AmipDiTWrapper:
    torch.manual_seed(0)
    return AmipDiTWrapper(
        **_kwargs(),
        dit_kwargs=dict(
            dim=32,
            num_heads=4,
            num_blocks=1,
            patch_size=2,
            c_grid_downsample=1,
            num_ca_blocks=1,  # exercise the optional ca_embed branch
        ),
    )


def _rolling_dit_wrapper() -> RollingDiTWrapper:
    torch.manual_seed(0)
    return RollingDiTWrapper(
        **_kwargs(),
        rolling_dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1),
    )


def _erdm_wrapper() -> ERDMWrapper:
    torch.manual_seed(0)
    return ERDMWrapper(
        **_kwargs(),
        erdm_kwargs=dict(
            model_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attn_levels=(1,),
            num_groups=4,
        ),
    )


def _assert_partition_covers_all_params(model: torch.nn.Module, groups: list[dict]) -> None:
    assert len(groups) == 2
    muon_group, adam_group = groups
    assert muon_group["use_muon"] is True
    assert adam_group["use_muon"] is False

    all_params = list(model.parameters())
    total_numel = sum(p.numel() for p in all_params)
    group_numel = sum(p.numel() for g in groups for p in g["params"])
    assert group_numel == total_numel

    muon_ids = {id(p) for p in muon_group["params"]}
    adam_ids = {id(p) for p in adam_group["params"]}
    assert muon_ids.isdisjoint(adam_ids)
    assert muon_ids | adam_ids == {id(p) for p in all_params}

    # Muon group must only contain >=2D matmul weights.
    assert all(p.ndim >= 2 for p in muon_group["params"])


@pytest.mark.parametrize(
    "make_wrapper",
    [_amip_dit_wrapper, _rolling_dit_wrapper, _erdm_wrapper],
    ids=["AmipDiTWrapper", "RollingDiTWrapper", "ERDMWrapper"],
)
def test_muon_param_groups_partition_all_params(make_wrapper):
    model = make_wrapper()
    groups = model.muon_param_groups(lr=1e-4, weight_decay=0.01)
    _assert_partition_covers_all_params(model, groups)


def test_muon_param_groups_lr_and_weight_decay():
    model = _amip_dit_wrapper()
    groups = model.muon_param_groups(lr=1e-4, weight_decay=0.01, muon_lr_multiplier=10.0)
    muon_group, adam_group = groups
    assert muon_group["lr"] == pytest.approx(1e-3)
    assert adam_group["lr"] == pytest.approx(1e-4)
    assert muon_group["weight_decay"] == pytest.approx(0.01)
    assert adam_group["weight_decay"] == pytest.approx(0.01)
    assert adam_group["betas"] == (0.9, 0.95)


def test_make_optimizer_muon_requires_muon_param_groups_method():
    model = torch.nn.Linear(4, 2)  # plain module — no muon_param_groups()
    cfg = OmegaConf.create({"optimizer_type": "Muon", "lr": 1e-4})
    with pytest.raises(ValueError, match="muon_param_groups"):
        make_optimizer(model, cfg)


def test_make_optimizer_muon_missing_package_raises_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "muon":
            raise ImportError("no module named muon")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    model = _amip_dit_wrapper()
    cfg = OmegaConf.create({"optimizer_type": "Muon", "lr": 1e-4})
    with pytest.raises(ImportError, match="muon"):
        make_optimizer(model, cfg)
