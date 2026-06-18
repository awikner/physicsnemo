# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`ModelEMA`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "pangu_plasim"
sys.path.insert(0, str(_RECIPE_DIR))

from ema import ModelEMA  # noqa: E402


def _make_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))


def test_init_shadow_matches_initial_params():
    model = _make_model()
    ema = ModelEMA(model, decay=0.99, warmup_epochs=0)
    for name, p in model.named_parameters():
        assert torch.equal(ema.shadow[name], p.detach())


def test_warmup_clamps_effective_decay():
    """At ``epoch=0`` with ``warmup_epochs=6`` the effective decay is 1/7 ≈ 0.143,
    not 0.999. The shadow should move sharply toward the new parameter values."""
    model = _make_model()
    ema = ModelEMA(model, decay=0.999, warmup_epochs=6)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Mutate params by a known amount and update.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, epoch=0)
    # At epoch 0, decay = 1/7. shadow = (1/7)*initial + (6/7)*new = initial + 6/7.
    expected_decay = 1.0 / 7.0
    for n, p in model.named_parameters():
        expected = expected_decay * initial[n] + (1.0 - expected_decay) * p.detach()
        assert torch.allclose(ema.shadow[n], expected, atol=1e-6)


def test_post_warmup_uses_configured_decay():
    """After warmup the effective decay equals ``decay``."""
    model = _make_model()
    decay = 0.9
    ema = ModelEMA(model, decay=decay, warmup_epochs=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    # epoch=10 well past warmup_epochs=1.
    ema.update(model, epoch=10)
    initial = 0.0  # original zeros via _make_model + add_(1.0): shadow had 0-init values
    # Actually verify by computing what the shadow would look like.
    saved = {n: ema.shadow[n].clone() for n, _ in model.named_parameters()}
    # Apply another update and verify decay holds.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, epoch=11)
    for n, p in model.named_parameters():
        expected = decay * saved[n] + (1.0 - decay) * p.detach()
        assert torch.allclose(ema.shadow[n], expected, atol=1e-6)


def test_apply_and_restore_round_trip():
    model = _make_model()
    ema = ModelEMA(model, decay=0.5, warmup_epochs=0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.update(model, epoch=10)

    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    shadow_copy = {n: ema.shadow[n].clone() for n in ema.shadow}

    ema.apply_to(model)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), shadow_copy[n])

    ema.restore(model)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), original[n])


def test_apply_twice_without_restore_raises():
    model = _make_model()
    ema = ModelEMA(model, decay=0.5, warmup_epochs=0)
    ema.apply_to(model)
    with pytest.raises(RuntimeError):
        ema.apply_to(model)
    ema.restore(model)


def test_state_dict_round_trip():
    model = _make_model()
    ema = ModelEMA(model, decay=0.8, warmup_epochs=3)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    ema.update(model, epoch=5)
    state = ema.state_dict()

    # Build a fresh EMA on a fresh (zero-ed) model and load the state.
    fresh_model = _make_model(seed=1)  # different initial weights
    fresh_ema = ModelEMA(fresh_model, decay=0.1, warmup_epochs=0)
    fresh_ema.load_state_dict(state)
    assert fresh_ema.decay == pytest.approx(0.8)
    assert fresh_ema.warmup_epochs == 3
    for n in state["shadow"]:
        assert torch.equal(fresh_ema.shadow[n], state["shadow"][n])
