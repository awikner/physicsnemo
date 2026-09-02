# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""StepLR must reproduce upstream amip_v2's per-EPOCH lr decay.

Upstream trains the ERDM baseline with
``StepLR(optimizer, step_size=1, gamma=0.95)`` stepped once per epoch by
Lightning. Our loop steps the scheduler every OPTIMIZER step, so the epoch
cadence has to be converted through ``steps_per_epoch`` — get that wrong by a
factor of ``steps_per_epoch`` (13,143 here) and the lr either never decays or
collapses within one epoch, in both cases silently. Hence these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_loop import make_scheduler  # noqa: E402

LR = 5.0e-5
SPE = 100  # steps per epoch (small stand-in for the real 13,143)


def _opt():
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=LR)


def _cfg(**kw):
    return OmegaConf.create({"scheduler": "StepLR", **kw})


def test_decays_once_per_epoch_by_gamma():
    opt = _opt()
    sch = make_scheduler(opt, _cfg(sl_gamma=0.95), total_steps=SPE * 5,
                         steps_per_epoch=SPE)
    assert opt.param_groups[0]["lr"] == pytest.approx(LR)
    # within the first epoch: unchanged
    for _ in range(SPE - 1):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LR)
    # crossing the epoch boundary: one gamma
    sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LR * 0.95)


def test_matches_upstream_over_24_epochs():
    """The production budget: 24 epochs at gamma 0.95 -> 0.95**24 of base."""
    opt = _opt()
    sch = make_scheduler(opt, _cfg(sl_gamma=0.95), total_steps=SPE * 24,
                         steps_per_epoch=SPE)
    for _ in range(SPE * 24):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LR * 0.95**24, rel=1e-6)
    # sanity on the magnitude: a mild decay, not a collapse
    assert 0.25 < 0.95**24 < 0.32


def test_step_epochs_knob():
    opt = _opt()
    sch = make_scheduler(opt, _cfg(sl_gamma=0.5, sl_step_epochs=2),
                         total_steps=SPE * 6, steps_per_epoch=SPE)
    for _ in range(2 * SPE - 1):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LR)
    sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LR * 0.5)


def test_missing_steps_per_epoch_raises_loudly():
    with pytest.raises(ValueError, match="steps_per_epoch"):
        make_scheduler(_opt(), _cfg(), total_steps=1000)


def test_resume_fast_forward_lands_on_the_right_epoch():
    """train_diffusion fast-forwards done_in_stage*steps_per_epoch steps on
    resume; StepLR must then sit exactly where the interrupted run left it."""
    done = 3
    opt = _opt()
    sch = make_scheduler(opt, _cfg(sl_gamma=0.95), total_steps=SPE * 24,
                         steps_per_epoch=SPE)
    for _ in range(done * SPE):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(LR * 0.95**done)


def test_other_schedulers_unaffected_by_the_new_kwarg():
    opt = _opt()
    # CosineToFloor derives its floor ratio from cfg.lr, so it must be present
    cfg = OmegaConf.create({"scheduler": "CosineToFloor", "lr": LR,
                            "ct_decay_steps": 50, "ct_floor_lr": 1e-6})
    sch = make_scheduler(opt, cfg, total_steps=200, steps_per_epoch=SPE)
    for _ in range(60):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-3)
