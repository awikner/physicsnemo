# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 v3 unit tests: bf16 AMP autocast wiring + fused AdamW.

The trainer wraps the forward+loss block in
``torch.amp.autocast(device_type=..., dtype=cfg.amp)`` when ``cfg.amp`` is
truthy. bf16 is the documented default (matches PanguWeather v2.0); fp16
additionally uses a :class:`torch.amp.GradScaler` for underflow protection.

Fused AdamW (``torch.optim.AdamW(..., fused=True)``) is opt-in via
``cfg.fused: True`` in the scheduler/optimizer group. Falls back to eager
AdamW with a warning on CPU.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from loss import PanguPlasimLoss  # noqa: E402
from train_loop import (  # noqa: E402
    _resolve_amp_dtype,
    make_optimizer,
    make_scheduler,
    train_step,
)


# ---------------------------------------------------------------------------
# _resolve_amp_dtype
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (False, None),
        ("none", None),
        ("off", None),
        (True, torch.bfloat16),  # True → default bf16 (PanguWeather convention)
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp16", torch.float16),
        ("float16", torch.float16),
    ],
)
def test_resolve_amp_dtype(value, expected):
    assert _resolve_amp_dtype(value) is expected


# ---------------------------------------------------------------------------
# Fused AdamW
# ---------------------------------------------------------------------------
def test_make_optimizer_fused_true_on_cpu_warns_and_falls_back():
    """Fused AdamW requires CUDA. On CPU we warn + return non-fused AdamW."""
    model = torch.nn.Linear(4, 2)
    cfg = OmegaConf.create({"optimizer_type": "AdamW", "lr": 1e-3, "fused": True})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opt = make_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    if not torch.cuda.is_available():
        assert any("CUDA" in str(x.message) for x in w), (
            f"expected a CUDA-fallback RuntimeWarning, got {[str(x.message) for x in w]}"
        )


def test_make_optimizer_fused_false_default():
    """No `fused` key → plain AdamW; no warning."""
    model = torch.nn.Linear(4, 2)
    cfg = OmegaConf.create({"optimizer_type": "AdamW", "lr": 1e-3})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opt = make_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    # No CUDA-fallback warning should fire when fused isn't requested.
    assert not any("fused" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# train_step with amp_dtype on CPU (autocast(device_type="cpu") is supported)
# ---------------------------------------------------------------------------
class _TinyLegacy(torch.nn.Module):
    """PanguPlasimLegacy-shaped model returning zero placeholders for latents."""

    def __init__(self):
        super().__init__()
        self.lin_s = torch.nn.Conv2d(2, 2, 1)
        self.lin_u = torch.nn.Conv3d(3, 3, 1)

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in):
        return self.lin_s(surface_in), self.lin_u(upper_air_in), 0, 0, 0, 0


def _toy_batch():
    return {
        "surface_in": torch.randn(2, 2, 4, 8),
        "constant_boundary": torch.zeros(2, 1, 4, 8),
        "varying_boundary": torch.zeros(2, 1, 4, 8),
        "upper_air_in": torch.randn(2, 3, 2, 4, 8),
        "target_surface": torch.ones(2, 2, 4, 8),
        "target_upper_air": torch.ones(2, 3, 2, 4, 8),
    }


def _make_loss():
    return PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=[],
        num_lat=4,
        loss_type="l2",
    )


def _make_optim_and_sched(model, total_steps=50):
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-2,
            "scheduler": "OneCycleLR",
            "oc_pct_start": 0.1,
            "oc_div_factor": 1e5,
            "oc_final_div_factor": 0.00025,
        }
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, total_steps=total_steps)
    return optimizer, scheduler


def test_train_step_amp_dtype_none_unchanged():
    """amp_dtype=None matches the existing eager-mode behavior."""
    torch.manual_seed(0)
    model = _TinyLegacy()
    optimizer, scheduler = _make_optim_and_sched(model)
    losses = train_step(
        model=model,
        loss_fn=_make_loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        batch=_toy_batch(),
        has_diagnostic=False,
        amp_dtype=None,
    )
    assert torch.isfinite(losses["loss"]).all()


def test_train_step_amp_dtype_bf16_runs_on_cpu():
    """bf16 autocast on CPU should run cleanly and emit a finite loss."""
    torch.manual_seed(0)
    model = _TinyLegacy()
    optimizer, scheduler = _make_optim_and_sched(model)
    losses = train_step(
        model=model,
        loss_fn=_make_loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        batch=_toy_batch(),
        has_diagnostic=False,
        amp_dtype=torch.bfloat16,
    )
    assert torch.isfinite(losses["loss"]).all()


def test_train_step_loss_decreases_under_bf16_amp():
    """bf16 AMP should still drive the loss down over a handful of iterations."""
    torch.manual_seed(0)
    model = _TinyLegacy()
    optimizer, scheduler = _make_optim_and_sched(model, total_steps=30)
    batch = _toy_batch()
    initial = None
    for _ in range(20):
        out = train_step(
            model=model,
            loss_fn=_make_loss(),
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            has_diagnostic=False,
            amp_dtype=torch.bfloat16,
        )
        if initial is None:
            initial = float(out["loss"].detach())
    final = float(out["loss"].detach())
    assert final < initial
