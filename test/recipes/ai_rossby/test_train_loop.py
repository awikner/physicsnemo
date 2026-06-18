# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optimizer/scheduler factories + train_step in
``examples/weather/ai_rossby/train_loop.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from loss import PanguPlasimLoss  # noqa: E402
from train_loop import make_optimizer, make_scheduler, train_step  # noqa: E402


def _toy_model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def test_make_optimizer_adamw():
    model = _toy_model()
    cfg = OmegaConf.create({"optimizer_type": "AdamW", "lr": 1e-3, "weight_decay": 1e-5})
    opt = make_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-5)


def test_make_optimizer_rejects_unknown_type():
    model = _toy_model()
    cfg = OmegaConf.create({"optimizer_type": "FusedAdam", "lr": 1e-3})
    with pytest.raises(ValueError, match="optimizer_type"):
        make_optimizer(model, cfg)


def test_make_scheduler_onecycle_smokes_through_total_steps():
    model = _toy_model()
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-3,
            "scheduler": "OneCycleLR",
            "oc_pct_start": 0.1,
            "oc_div_factor": 1e5,
            "oc_final_div_factor": 0.00025,
        }
    )
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg, total_steps=20)
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)
    for _ in range(20):
        opt.step()
        sched.step()


def test_make_scheduler_cosine_warmup_composes():
    model = _toy_model()
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-3,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "num_warmup_steps": 5,
            "warmup_start_lr": 1e-8,
            "eta_min": 0.0,
        }
    )
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg, total_steps=20)
    lrs = []
    for _ in range(20):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # LR should rise during the linear warmup then anneal down.
    assert lrs[0] < lrs[4]
    assert lrs[-1] < lrs[5]


def test_make_scheduler_cosine_no_warmup_falls_through_to_cosine():
    model = _toy_model()
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-3,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "num_warmup_steps": 0,
            "eta_min": 0.0,
        }
    )
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg, total_steps=10)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_make_scheduler_rejects_unknown_name():
    model = _toy_model()
    cfg = OmegaConf.create({"optimizer_type": "AdamW", "lr": 1e-3, "scheduler": "Nope"})
    opt = make_optimizer(model, cfg)
    with pytest.raises(ValueError, match="Unknown scheduler"):
        make_scheduler(opt, cfg, total_steps=10)


def test_train_step_reduces_loss_on_toy_model():
    """Minimal model that mimics PanguPlasim's call signature; verifies the
    forward → backward → step → scheduler.step plumbing works end-to-end."""

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_s = torch.nn.Conv2d(2, 2, 1)
            self.lin_u = torch.nn.Conv3d(3, 3, 1)

        def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in):
            return self.lin_s(surface_in), self.lin_u(upper_air_in), 0, 0, 0, 0

    model = _Tiny()
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-1,
            "scheduler": "OneCycleLR",
            "oc_pct_start": 0.1,
            "oc_div_factor": 1e5,
            "oc_final_div_factor": 0.00025,
        }
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, total_steps=50)
    loss_fn = PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=[],
        num_lat=4,
        loss_type="l2",
    )
    batch = {
        "surface_in": torch.randn(2, 2, 4, 8),
        "constant_boundary": torch.zeros(2, 1, 4, 8),
        "varying_boundary": torch.zeros(2, 1, 4, 8),
        "upper_air_in": torch.randn(2, 3, 2, 4, 8),
        "target_surface": torch.ones(2, 2, 4, 8),
        "target_upper_air": torch.ones(2, 3, 2, 4, 8),
    }
    initial_loss = None
    for _ in range(20):
        out = train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            has_diagnostic=False,
        )
        if initial_loss is None:
            initial_loss = float(out["loss"].detach())
    final_loss = float(out["loss"].detach())
    assert final_loss < initial_loss
