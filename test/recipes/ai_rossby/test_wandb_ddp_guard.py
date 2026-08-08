# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""W1 unit tests for the wandb-under-DDP auto-disable guard.

wandb's background threads can stall NCCL and hang DDP init (Phase 12b,
jobs 20918380/20916803) — ``_maybe_init_wandb`` must refuse to start
wandb when ``world_size > 1`` unless ``wandb.allow_multigpu`` is set.
See ``docs/dev/wandb_ddp_hang_fix_plan.md``.

The guard sits BEFORE the ``initialize_wandb`` import/call, so these
tests never touch a real wandb run: for the guard cases no wandb code
runs at all, and for the pass-through cases ``initialize_wandb`` is
monkeypatched out.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

import train  # noqa: E402


@dataclass
class _StubDist:
    rank: int = 0
    world_size: int = 1


def _cfg(**wandb_overrides):
    wandb_cfg = {
        "enabled": True,
        "allow_multigpu": False,
        "project": "test",
        "entity": None,
        "name": "test-run",
        "mode": "offline",
        "init_timeout": 300,
    }
    wandb_cfg.update(wandb_overrides)
    return OmegaConf.create({"run_name": "test-run", "wandb": wandb_cfg})


def _patch_initialize(monkeypatch, calls: list):
    """Route initialize_wandb to a recorder (guard-pass cases only)."""

    def _fake_initialize(**kwargs):
        calls.append(kwargs)

    import physicsnemo.utils.logging.wandb as pw

    monkeypatch.setattr(pw, "initialize_wandb", _fake_initialize)


def test_guard_disables_wandb_under_ddp(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert train._maybe_init_wandb(_cfg(), dist=_StubDist(rank=0, world_size=2)) is False
    assert any("auto-disabled under DDP" in r.message for r in caplog.records)


def test_guard_disables_on_every_rank_silently_off_rank0(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert train._maybe_init_wandb(_cfg(), dist=_StubDist(rank=1, world_size=2)) is False
    # Only rank 0 logs the warning — non-zero ranks return quietly.
    assert not any("auto-disabled" in r.message for r in caplog.records)


def test_guard_respects_allow_multigpu_escape_hatch(monkeypatch):
    calls: list = []
    _patch_initialize(monkeypatch, calls)
    out = train._maybe_init_wandb(
        _cfg(allow_multigpu=True), dist=_StubDist(rank=0, world_size=2)
    )
    assert out is True  # rank 0 drives LaunchLogger
    assert len(calls) == 1


def test_guard_inactive_single_gpu(monkeypatch):
    calls: list = []
    _patch_initialize(monkeypatch, calls)
    out = train._maybe_init_wandb(_cfg(), dist=_StubDist(rank=0, world_size=1))
    assert out is True
    assert len(calls) == 1


def test_enabled_false_short_circuits_before_guard():
    # No warning, no wandb — plain disabled path unchanged.
    assert (
        train._maybe_init_wandb(_cfg(enabled=False), dist=_StubDist(world_size=2))
        is False
    )


def test_nonzero_rank_mode_default_offline(monkeypatch):
    calls: list = []
    _patch_initialize(monkeypatch, calls)
    out = train._maybe_init_wandb(
        _cfg(allow_multigpu=True, mode="online"),
        dist=_StubDist(rank=1, world_size=2),
    )
    assert out is False  # non-zero ranks never drive LaunchLogger
    assert calls[0]["mode"] == "offline"


def test_nonzero_rank_mode_disabled_passthrough(monkeypatch):
    # W2 cell M5: non-zero ranks thread-free.
    calls: list = []
    _patch_initialize(monkeypatch, calls)
    train._maybe_init_wandb(
        _cfg(allow_multigpu=True, mode="online", nonzero_rank_mode="disabled"),
        dist=_StubDist(rank=1, world_size=2),
    )
    assert calls[0]["mode"] == "disabled"
    # Rank 0 keeps the configured mode regardless of the knob.
    calls.clear()
    train._maybe_init_wandb(
        _cfg(allow_multigpu=True, mode="online", nonzero_rank_mode="disabled"),
        dist=_StubDist(rank=0, world_size=2),
    )
    assert calls[0]["mode"] == "online"


def test_missing_allow_multigpu_key_defaults_to_guarded():
    # Configs without the key must stay safe (guard active) — the W3
    # default flip lives in conf/config.yaml, not in the code fallback.
    cfg = OmegaConf.create(
        {"run_name": "t", "wandb": {"enabled": True, "mode": "offline"}}
    )
    assert train._maybe_init_wandb(cfg, dist=_StubDist(world_size=4)) is False


def test_repo_config_default_allows_multigpu():
    # W3 decision pin (2026-08-07): after the every-rank call-site fix was
    # validated (3/3 short + 93-min full-epoch wandb-on runs), the shipped
    # config default is allow_multigpu=true. Flipping it back should be a
    # deliberate act — this test makes it show up in review.
    from hydra import compose, initialize_config_dir

    cfg_dir = str((_AI_ROSSBY_DIR / "conf").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
        cfg = compose(config_name="config")
    assert bool(cfg.wandb.allow_multigpu) is True