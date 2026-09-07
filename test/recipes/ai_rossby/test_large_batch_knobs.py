# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The two large-batch knobs: Muon momentum, and a fixed cosine horizon.

Both exist because global batch 40 sits ~7x past the measured gradient noise
scale (B_noise ~ 6). Momentum: the package default 0.95 is a ~20-step window
that at batch 40 averages 800 already-clean samples and only adds lag.
Horizon: a chained run sets num_epochs per LINK, so without a pinned horizon
the cosine would change shape at every resume boundary.
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
from physicsnemo.experimental.models.amip_si.wrappers import _muon_groups  # noqa: E402

LR, SPE = 5.0e-4, 100   # peak lr; a small stand-in for the real 1315 steps/epoch


# ---------------------------------------------------------------------------
# Momentum plumbing
# ---------------------------------------------------------------------------
def _params():
    return [torch.nn.Parameter(torch.zeros(4, 4))], [torch.nn.Parameter(torch.zeros(4))]


def test_momentum_lands_in_the_muon_group_only():
    mw, aw = _params()
    groups = _muon_groups(mw, aw, lr=LR, weight_decay=0.01,
                          muon_lr_multiplier=10.0, adam_betas=(0.9, 0.95),
                          muon_momentum=0.85)
    muon = next(g for g in groups if g["use_muon"])
    adam = next(g for g in groups if not g["use_muon"])
    assert muon["momentum"] == pytest.approx(0.85)
    assert "momentum" not in adam, "AdamW group must not get a Muon key"


def test_momentum_default_is_the_package_default():
    """Unset must be bit-identical to before: every existing run used 0.95."""
    mw, aw = _params()
    groups = _muon_groups(mw, aw, lr=LR, weight_decay=0.01,
                          muon_lr_multiplier=10.0, adam_betas=(0.9, 0.95))
    assert next(g for g in groups if g["use_muon"])["momentum"] == pytest.approx(0.95)


def test_package_optimizer_accepts_the_group():
    """MuonWithAuxAdam asserts the EXACT key set of a muon group; an extra or
    misnamed key raises at construction. Pin that ours is accepted."""
    muon = pytest.importorskip("muon")
    mw, aw = _params()
    groups = _muon_groups(mw, aw, lr=LR, weight_decay=0.01,
                          muon_lr_multiplier=10.0, adam_betas=(0.9, 0.95),
                          muon_momentum=0.85)
    opt = muon.MuonWithAuxAdam(groups)
    assert next(g for g in opt.param_groups if g["use_muon"])["momentum"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Fixed cosine horizon
# ---------------------------------------------------------------------------
def _cfg(**kw):
    return OmegaConf.create({"scheduler": "LinearWarmupCosineAnnealingLR",
                             "lr": LR, "num_warmup_steps": SPE, "eta_min": 0.1 * LR,
                             **kw})


def _opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR)


def _run(opt, sch, n):
    for _ in range(n):
        opt.step(); sch.step()
    return opt.param_groups[0]["lr"]


def test_horizon_is_pinned_regardless_of_stage_length():
    """A 4-epoch TEST of a 24-epoch recipe must sample the first 4 epochs of
    the 24-epoch curve, not anneal to the floor by epoch 4."""
    short = make_scheduler(_opt(), _cfg(cosine_total_epochs=24),
                           total_steps=4 * SPE, steps_per_epoch=SPE)
    long_ = make_scheduler(_opt(), _cfg(cosine_total_epochs=24),
                           total_steps=24 * SPE, steps_per_epoch=SPE)
    # identical curve at epoch 4 despite different stage lengths
    a = _run(short.optimizer, short, 4 * SPE)
    b = _run(long_.optimizer, long_, 4 * SPE)
    assert a == pytest.approx(b, rel=1e-9)
    # ...and still near the peak, not at the floor
    assert a > 0.9 * LR, f"epoch-4 lr {a:.3e} should be ~93% of peak on a 24-epoch cosine"


def test_without_the_knob_the_horizon_follows_stage_length():
    """The pre-existing behaviour, so nothing else moves."""
    s = make_scheduler(_opt(), _cfg(), total_steps=4 * SPE, steps_per_epoch=SPE)
    lr4 = _run(s.optimizer, s, 4 * SPE)
    assert lr4 == pytest.approx(0.1 * LR, rel=1e-6), "should have annealed to the floor"


def test_chained_resume_reproduces_the_continuous_curve():
    """What the 2-link chain relies on: link 2 rebuilds the scheduler from
    scratch and fast-forwards done*SPE steps (after resetting lr to base, per
    the compounding-lr fix). That must land exactly where a continuous run is."""
    cont = make_scheduler(_opt(), _cfg(cosine_total_epochs=24),
                          total_steps=24 * SPE, steps_per_epoch=SPE)
    lr_cont = _run(cont.optimizer, cont, 2 * SPE)          # end of link 1

    opt2 = _opt()
    for g in opt2.param_groups:                             # the reset
        g["lr"] = LR; g.pop("initial_lr", None)
    link2 = make_scheduler(opt2, _cfg(cosine_total_epochs=24),
                           total_steps=24 * SPE, steps_per_epoch=SPE)
    for _ in range(2 * SPE):                                # the fast-forward
        link2.step()
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr_cont, rel=1e-6)
    # past the 1-epoch warmup, inside the cosine
    assert opt2.param_groups[0]["lr"] < LR


def test_horizon_shorter_than_warmup_is_refused():
    with pytest.raises(ValueError, match="not more than"):
        make_scheduler(_opt(), _cfg(cosine_total_epochs=0.5),
                       total_steps=4 * SPE, steps_per_epoch=SPE)
