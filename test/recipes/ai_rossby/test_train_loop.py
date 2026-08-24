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


def test_make_scheduler_cosine_to_floor():
    """CosineToFloor: cosine lr -> ct_floor_lr over ct_decay_steps, then HOLD.

    The hold is the point — CosineAnnealingLR past T_max is periodic (lr
    climbs back up), which is exactly wrong for an objective that keeps
    sharpening as the fit improves (the RSI A2 story,
    docs/dev/context/rsi-h1-precond-instability.md). Also pins that the
    multiplicative lambda preserves per-group lr ratios, which is what keeps
    Muon's hidden-weight multiplier intact under the schedule.
    """
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(
        [
            {"params": [model.weight], "lr": 1e-3},
            {"params": [model.bias], "lr": 3e-3},  # a "multiplied" group
        ]
    )
    cfg = OmegaConf.create(
        {"scheduler": "CosineToFloor", "lr": 1e-3,
         "ct_decay_steps": 100, "ct_floor_lr": 2e-4}
    )
    sched = make_scheduler(opt, cfg, total_steps=10_000)
    lrs = {}
    for t in range(301):
        if t in (0, 50, 100, 300):
            lrs[t] = [g["lr"] for g in opt.param_groups]
        opt.step()
        sched.step()
    assert lrs[0][0] == pytest.approx(1e-3)
    # halfway: floor + half the gap
    assert lrs[50][0] == pytest.approx(2e-4 + 0.5 * 8e-4, rel=1e-6)
    # at and past the decay end: pinned to the floor, never climbing back
    assert lrs[100][0] == pytest.approx(2e-4, rel=1e-6)
    assert lrs[300][0] == pytest.approx(2e-4, rel=1e-6)
    # group ratio (3x) preserved throughout
    for t, v in lrs.items():
        assert v[1] == pytest.approx(3.0 * v[0], rel=1e-6)


def test_make_scheduler_cosine_to_floor_rejects_bad_floor():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = OmegaConf.create(
        {"scheduler": "CosineToFloor", "lr": 1e-3,
         "ct_decay_steps": 100, "ct_floor_lr": 2e-3}  # floor above lr
    )
    with pytest.raises(ValueError, match="ct_floor_lr"):
        make_scheduler(opt, cfg, total_steps=1000)


def test_flatten_optimizer_cfg_muon_lr_multiplier_passthrough():
    """`training.optimizer.muon_lr_multiplier` reaches make_optimizer's cfg.

    Unset, the key must be ABSENT (the wrapper's own 10x default applies);
    set, it must arrive as a float. Muon's hidden-weight lr is `lr *
    multiplier`, and the multiplier is the surgical stability knob for
    trunk-led sharpening (docs/dev/context/rsi-h1-precond-instability.md).
    """
    from train import _flatten_optimizer_cfg

    base = {"type": "Muon", "lr": 5e-5, "weight_decay": 3e-6, "fused": None}
    flat = _flatten_optimizer_cfg(OmegaConf.create(base))
    assert "muon_lr_multiplier" not in flat
    flat = _flatten_optimizer_cfg(
        OmegaConf.create({**base, "muon_lr_multiplier": 3})
    )
    assert flat.muon_lr_multiplier == pytest.approx(3.0)


def _gov_setup(lr=1e-3, mult=3.0):
    from train_loop import GnormLrGovernor

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(
        [{"params": [model.weight], "lr": lr},
         {"params": [model.bias], "lr": mult * lr}]
    )
    cfg = OmegaConf.create(
        {"scheduler": "CosineToFloor", "lr": lr,
         "ct_decay_steps": 1000, "ct_floor_lr": lr / 5}
    )
    sched = make_scheduler(opt, cfg, total_steps=10_000)
    gov = GnormLrGovernor(factor=8.0, drop=0.5, cooldown=50, warmup=10)
    return opt, sched, gov


def test_gnorm_governor_drops_on_spike_and_survives_scheduler():
    """A spike halves lr permanently — including through scheduler.step().

    The subtle failure this pins: a governor that edits only
    ``param_groups[i]['lr']`` is silently undone by the next
    ``scheduler.step()`` (schedulers recompute from ``base_lrs``). The RSI A2
    excursions are exactly when that mistake would go unnoticed.
    """
    opt, sched, gov = _gov_setup()
    for _ in range(20):                       # healthy band: ema ~ 1e4
        assert not gov.update(1e4, opt, sched)
    lr_before = opt.param_groups[0]["lr"]
    assert gov.update(1e6, opt, sched)        # 100x the band: drop
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * lr_before)
    opt.step()
    sched.step()                              # must NOT restore the old lr
    assert opt.param_groups[0]["lr"] < 0.51 * lr_before
    # group ratio preserved
    g = opt.param_groups
    assert g[1]["lr"] == pytest.approx(3.0 * g[0]["lr"], rel=1e-6)


def test_gnorm_governor_cooldown_warmup_and_ema_hygiene():
    opt, sched, gov = _gov_setup()
    # warmup: no drops before `warmup` samples, however large the spike
    assert not gov.update(1e4, opt, sched)
    assert not gov.update(1e9, opt, sched)
    for _ in range(20):
        gov.update(1e4, opt, sched)
    ema_before = gov.ema
    assert gov.update(1e6, opt, sched)        # first drop
    # spike must NOT be folded into the healthy EMA
    assert gov.ema == pytest.approx(ema_before)
    assert not gov.update(1e6, opt, sched)    # cooldown blocks the second
    assert gov.drops == 1
    # non-finite gnorms are ignored, not counted
    assert not gov.update(float("nan"), opt, sched)


def test_gnorm_governor_respects_min_lr():
    from train_loop import GnormLrGovernor

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-7)
    gov = GnormLrGovernor(factor=8.0, drop=0.5, cooldown=0, warmup=0,
                          min_lr=1e-7)
    gov.ema = 1e4
    gov.seen = 100
    assert not gov.update(1e6, opt)           # 0.75e-7 < min_lr: refuse
    assert opt.param_groups[0]["lr"] == pytest.approx(1.5e-7)


def test_gnorm_governor_band_freezes_during_sustained_excursion():
    """A sustained sub-trigger excursion must NOT inflate the healthy band.

    v1 clipped each EMA update at the trigger threshold, but a long stretch of
    elevated-yet-sub-trigger gnorms still ratcheted the band up 48x (measured,
    Midway job 54457716), so the trigger chased the excursion and fired ~1.5k
    batches too late. v2 freezes the band at freeze_factor x ema.
    """
    from train_loop import GnormLrGovernor

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gov = GnormLrGovernor(factor=4.0, drop=0.5, cooldown=0, warmup=10,
                          freeze_factor=2.0)
    for _ in range(50):
        gov.update(1e4, opt)
    band = gov.ema
    # 3x the band: above freeze (2x), below trigger (4x) — the dangerous zone
    for _ in range(500):
        gov.update(3e4, opt)
    assert gov.ema == pytest.approx(band), "band chased a sustained excursion"
    assert gov.update(4.5e4, opt), "trigger must still fire against the frozen band"


def test_rewind_buffer_roundtrip_restores_model_and_optimizer():
    from train_loop import RewindBuffer

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    buf = RewindBuffer(every=10, keep=2)

    def _train_some(n):
        for _ in range(n):
            opt.zero_grad()
            model(torch.randn(8, 4)).pow(2).mean().backward()
            opt.step()

    _train_some(9)
    assert buf.maybe_snapshot(10, model, opt, healthy=True)
    w10 = model.weight.detach().clone()
    _train_some(10)
    assert buf.maybe_snapshot(20, model, opt, healthy=True)
    _train_some(17)                      # "excursion" region
    assert not torch.allclose(model.weight, w10)
    step = buf.restore(model, opt)
    assert step == 10, "must restore the OLDEST snapshot, not the newest"
    torch.testing.assert_close(model.weight, w10)
    # optimizer momenta rewound too: exp_avg matches a fresh post-restore step
    assert buf.restore(model, opt) is None, "snapshots are consumed on restore"


def test_rewind_buffer_skips_unhealthy_snapshots():
    from train_loop import RewindBuffer

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    buf = RewindBuffer(every=10, keep=2)
    assert not buf.maybe_snapshot(10, model, opt, healthy=False)
    assert not buf.maybe_snapshot(15, model, opt, healthy=True)  # off-cadence
    assert buf.restore(model, opt) is None
