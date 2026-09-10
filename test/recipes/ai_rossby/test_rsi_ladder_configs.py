# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The RSI ladder's shipped configs instantiate and the eval samplers mirror
their training schedulers.

A sampler yaml that drifts from the loss yaml it was written for silently
evaluates a checkpoint under a different interpolant (the profiles, the lag
and the per-channel scales define the process the heads were regressed
against), so the mirror is pinned here for every (loss, sampler) pair of the
ladder.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.diffusion import RSIScheduler

_CONF = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby" / "conf"

#: everything a sampler must copy from its training scheduler
_MIRRORED = (
    "W", "anchor_lag", "parameterization", "h1_precond", "beta_mode",
    "beta_floor", "gamma_0", "gamma_1", "gamma_mode", "delta_std",
    "time_eps", "label_mode", "fresh_noise_scale", "reduce_to_erdm",
    "init_frames", "anchor_frames", "nocean",
)
_PAIRS = [
    ("rsi_a1", "rsi_a1_sstpred"),
    ("rsi_a2l", "rsi_a2l_sstpred"),
]


def _loss(name, **overrides):
    ov = [f"loss={name}", "loss.noise_scale_path=null"] + [
        f"loss.{k}={v}" for k, v in overrides.items()]
    with initialize_config_dir(config_dir=str(_CONF), version_base="1.2"):
        cfg = compose(config_name="config", overrides=ov)
    return instantiate(cfg.loss)


def _sampler(name):
    cfg = OmegaConf.load(_CONF / "sampler" / f"{name}.yaml")
    cfg.noise_scale_path = None
    return instantiate(cfg)


def test_a2l_loss_is_the_lag_w_sample_anchor():
    s = _loss("rsi_a2l")
    assert isinstance(s, RSIScheduler)
    assert s.W == 6 and s.anchor_lag == 6
    assert s.init_frames == 12 and s.anchor_frames == 6 and s.history_frames == 5
    assert s.parameterization == "state" and s.h1_precond == "edm"
    assert s.gamma_0 == pytest.approx(1.0) and s.gamma_1 == pytest.approx(0.04)
    assert s.fresh_noise_scale == pytest.approx(1.0)
    assert not s.reduce_to_erdm and s.pushforward_rolls == 0


def test_a2l_yaml_points_at_the_lag_six_artifact():
    text = (_CONF / "loss" / "rsi_a2l.yaml").read_text()
    assert "sigma_c_sstpred153_lag6.pt" in text
    assert "sigma_c_sstpred153.pt" not in text.replace("sigma_c_sstpred153_lag6.pt", "")
    text = (_CONF / "sampler" / "rsi_a2l_sstpred.yaml").read_text()
    assert "sigma_c_sstpred153_lag6.pt" in text


def test_a2_default_stays_at_lag_one():
    s = _loss("rsi")
    assert s.anchor_lag == 1 and s.init_frames == 7 and s.history_frames == 0


def test_a1_reduction_is_lag_independent_in_config():
    """rsi_a1 keeps lag 1; asking for lag W leaves the reduction intact."""
    assert _loss("rsi_a1").anchor_lag == 1
    s = _loss("rsi_a1", anchor_lag=6)
    assert s.reduce_to_erdm and s.anchor_lag == 6 and s.init_frames == 12


@pytest.mark.parametrize("loss_name,sampler_name", _PAIRS)
def test_eval_sampler_mirrors_its_training_scheduler(loss_name, sampler_name):
    a, b = _loss(loss_name), _sampler(sampler_name)
    for k in _MIRRORED:
        assert getattr(a, k) == getattr(b, k), (k, getattr(a, k), getattr(b, k))
