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


# ---------------------------------------------------------------------------
# Resume fast-forward must be IDEMPOTENT across a chain of resumes.
# ---------------------------------------------------------------------------
# The bug this pins (found in production 2026-09-05, jobs 7591784-91): PyTorch's
# StepLR computes the next lr from the CURRENT ``param_group['lr']``, not from a
# fixed base, and ``load_checkpoint`` -> ``set_optimizer_state_dict`` restores
# ``param_groups`` — so a resumed run arrives carrying the DECAYED lr and the
# fast-forward applies ``done`` decays on top of it. Over a chain the decay
# count compounds as k^2+k-2: by epoch 14 of a 12-link chain the lr was
# 4.361e-05 against an intended 7.711e-04 (70 decays instead of 14), and the
# remaining links would have run at ~5.8e-07, i.e. not trained at all.
#
# ``test_resume_fast_forward_lands_on_the_right_epoch`` above did NOT catch it,
# because it fast-forwards a FRESH optimizer once. The compounding only appears
# when the optimizer's lr is already decayed, so that is what these construct.

BASE = 1.5811e-3


def _resume_link(done_epochs, opt_state, *, reset_first):
    """One chain link. ``reset_first`` is the fix: restore the configured lr
    before building the scheduler so the fast-forward cannot double-count."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=BASE)
    base_group_lrs = [g["lr"] for g in opt.param_groups]
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    if reset_first and done_epochs > 0:
        for g, b in zip(opt.param_groups, base_group_lrs):
            g["lr"] = b
            g.pop("initial_lr", None)
    sch = make_scheduler(opt, _cfg(sl_gamma=0.95), total_steps=SPE * 24,
                         steps_per_epoch=SPE)
    for _ in range(done_epochs * SPE):
        sch.step()
    ff_lr = opt.param_groups[0]["lr"]
    for _ in range(2 * SPE):          # this link's two epochs
        sch.step()
    return ff_lr, opt.state_dict()


def test_chained_resumes_do_not_compound_the_decay():
    """Eight links, 2 epochs each: lr must track base * 0.95**done exactly."""
    state = None
    for k in range(1, 9):
        done = 2 * (k - 1)
        ff, state = _resume_link(done, state, reset_first=True)
        assert ff == pytest.approx(BASE * 0.95**done, rel=1e-9), (
            f"link {k}: lr {ff:.4e} != base*0.95**{done} "
            f"({BASE * 0.95**done:.4e})"
        )


def test_without_the_reset_the_decay_compounds():
    """The bug, stated as a test, so the fix cannot be silently reverted.

    Exact values observed on Polaris at epochs 2/14 of the batch-40 chain.
    """
    state = None
    seen = []
    for k in range(1, 9):
        ff, state = _resume_link(2 * (k - 1), state, reset_first=False)
        seen.append(ff)
    # link 2 (2 epochs done) and link 8 (14 epochs done)
    assert seen[1] == pytest.approx(1.2878e-3, rel=1e-3), seen[1]
    assert seen[7] == pytest.approx(4.3613e-5, rel=1e-3), seen[7]
    # ...i.e. 18x below the intended value at epoch 14.
    assert seen[7] < BASE * 0.95**14 / 15


def test_a_single_fresh_resume_hides_the_bug():
    """Why the original test passed: one resume of a FRESH optimizer is
    correct either way, so only a chain exposes the compounding."""
    ff_fixed, _ = _resume_link(3, None, reset_first=True)
    ff_buggy, _ = _resume_link(3, None, reset_first=False)
    assert ff_fixed == pytest.approx(ff_buggy, rel=1e-9)
