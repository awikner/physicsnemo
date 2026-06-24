# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`ModelEMA` (swa_utils-backed implementation).

The bespoke ``ModelEMA`` was rewritten on top of
:class:`torch.optim.swa_utils.AveragedModel` in Phase 8-pre-1. Its public
surface (``update`` / ``apply_to`` / ``restore`` / ``state_dict`` /
``load_state_dict``) is preserved; these tests pin the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from ema import ModelEMA  # noqa: E402


def _make_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))


def _shadow_state(ema: ModelEMA) -> dict[str, torch.Tensor]:
    """Snapshot of the EMA's current shadow parameters."""
    return {k: v.detach().clone() for k, v in ema.avg_model.module.state_dict().items()}


# ---------------------------------------------------------------------------
# Initial state + schedule
# ---------------------------------------------------------------------------


def test_init_shadow_matches_initial_params():
    """At construction the shadow is a clone of the model's current params."""
    model = _make_model()
    ema = ModelEMA(model, decay=0.99, warmup_epochs=0, steps_per_epoch=1)
    shadow = _shadow_state(ema)
    for name, p in model.named_parameters():
        assert torch.equal(shadow[name], p.detach())


def test_first_update_initializes_shadow_to_current_params():
    """The first call to ``update`` copies current params into the shadow
    (matches ``torch.optim.swa_utils.AveragedModel``'s initialization step;
    documented behavior change from the bespoke implementation)."""
    model = _make_model()
    ema = ModelEMA(model, decay=0.999, warmup_epochs=6, steps_per_epoch=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    shadow = _shadow_state(ema)
    for name, p in model.named_parameters():
        assert torch.allclose(shadow[name], p.detach(), atol=1e-6)


def test_second_update_applies_warmup_decay():
    """After the first update (which just copies params) the second update
    runs the avg_fn. At step=1 with warmup_epochs=6, steps_per_epoch=1
    the schedule gives ``epoch = 1 // 1 = 1`` ⇒ ``eff = min(0.999, 2/7) = 2/7``.
    """
    model = _make_model()
    ema = ModelEMA(model, decay=0.999, warmup_epochs=6, steps_per_epoch=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    saved = _shadow_state(ema)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    expected_decay = 2.0 / 7.0
    shadow = _shadow_state(ema)
    for name, p in model.named_parameters():
        if name in saved:
            expected = expected_decay * saved[name] + (1.0 - expected_decay) * p.detach()
            assert torch.allclose(shadow[name], expected, atol=1e-6)


def test_post_warmup_uses_configured_decay():
    """Once the step counter clears ``warmup_epochs * steps_per_epoch``,
    the effective decay clamps to ``decay``."""
    model = _make_model()
    decay = 0.9
    ema = ModelEMA(model, decay=decay, warmup_epochs=1, steps_per_epoch=1)
    # Drive past warmup. With warmup_epochs=1, steps_per_epoch=1, warmup_steps=1.
    # Step 0: copy. Step 1: epoch=1 ⇒ eff=2/2=1, clamped to 0.9. Step 2+: same.
    for k in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)
    saved = _shadow_state(ema)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    shadow = _shadow_state(ema)
    for name, p in model.named_parameters():
        if name in saved:
            expected = decay * saved[name] + (1.0 - decay) * p.detach()
            assert torch.allclose(shadow[name], expected, atol=1e-6)


def test_steps_per_epoch_affects_warmup_pace():
    """With ``steps_per_epoch=4``, the schedule treats every 4 updates as
    one epoch — so at step=3 we should still be at ``epoch=0``."""
    model = _make_model()
    # warmup_epochs=2, steps_per_epoch=4 ⇒ warmup_steps=8.
    ema = ModelEMA(model, decay=0.999, warmup_epochs=2, steps_per_epoch=4)
    # Step 0 just copies. Steps 1..3 run avg_fn with epoch = (1..3) // 4 = 0.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    saved = _shadow_state(ema)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    expected_decay = 1.0 / 3.0  # epoch=0 ⇒ (1+0)/(2+1) = 1/3
    shadow = _shadow_state(ema)
    for name, p in model.named_parameters():
        if name in saved:
            expected = expected_decay * saved[name] + (1.0 - expected_decay) * p.detach()
            assert torch.allclose(shadow[name], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# apply_to / restore
# ---------------------------------------------------------------------------


def test_apply_and_restore_round_trip():
    model = _make_model()
    ema = ModelEMA(model, decay=0.5, warmup_epochs=0, steps_per_epoch=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.update(model)  # first update: just copies current params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.update(model)  # second update applies the avg_fn

    shadow_copy = _shadow_state(ema)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}

    ema.apply_to(model)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), shadow_copy[name])

    ema.restore(model)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), original[name])


def test_apply_twice_without_restore_raises():
    model = _make_model()
    ema = ModelEMA(model, decay=0.5, warmup_epochs=0, steps_per_epoch=1)
    ema.apply_to(model)
    with pytest.raises(RuntimeError):
        ema.apply_to(model)
    ema.restore(model)


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    model = _make_model()
    ema = ModelEMA(model, decay=0.8, warmup_epochs=3, steps_per_epoch=2)
    # Drive a few updates so shadow ≠ initial.
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        ema.update(model)
    state = ema.state_dict()
    shadow_before = _shadow_state(ema)

    # Fresh EMA, different decay/warmup; load_state_dict should overwrite.
    fresh_model = _make_model(seed=1)
    fresh_ema = ModelEMA(fresh_model, decay=0.1, warmup_epochs=0, steps_per_epoch=1)
    fresh_ema.load_state_dict(state)

    assert fresh_ema.decay == pytest.approx(0.8)
    assert fresh_ema.warmup_epochs == 3
    assert fresh_ema.steps_per_epoch == 2

    shadow_after = _shadow_state(fresh_ema)
    for name in shadow_before:
        assert torch.equal(shadow_after[name], shadow_before[name])


# ---------------------------------------------------------------------------
# Integration: stable-plateau parity vs hand-computed schedule.
# ---------------------------------------------------------------------------


def test_post_warmup_matches_handrolled_recurrence_within_tolerance():
    """For steps well past warmup, the shadow's evolution matches the
    closed-form recurrence ``s_k = d * s_{k-1} + (1-d) * p_k`` exactly."""
    model = _make_model()
    decay = 0.8
    warmup = 2  # warmup_epochs=2, steps_per_epoch=1 ⇒ warmup_steps=2.
    ema = ModelEMA(model, decay=decay, warmup_epochs=warmup, steps_per_epoch=1)
    # First copy + first warmup step.
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        ema.update(model)
    # We're now at step=3 with epoch=3 ⇒ effective_decay = min(0.8, 4/3) = 0.8.
    # Verify the recurrence on the next 5 steps.
    handrolled = _shadow_state(ema)
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05)
        new_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        ema.update(model)
        for name in handrolled:
            handrolled[name] = decay * handrolled[name] + (1.0 - decay) * new_params[name]
        shadow = _shadow_state(ema)
        for name in handrolled:
            assert torch.allclose(shadow[name], handrolled[name], atol=1e-6), name
