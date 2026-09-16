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
import torch
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
    # A2-L-HN trains against a reweighted loss but is SAMPLED by the same
    # process, so it shares A2-L's sampler: the weighting is training-only.
    ("rsi_a2l_hn", "rsi_a2l_sstpred"),
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


# ---------------------------------------------------------------------------
# The corrected batch-40 pair (2026-09-15): the high-noise loss boost.
# ---------------------------------------------------------------------------

_HN_ERDM = dict(hn_sigma=10.0, hn_power=2.0, hn_clip=30.0)
_HN_RSI = dict(hn_sigma=3.5, hn_power=3.0, hn_clip=30.0)


def test_a0hn_is_erdm_v2_plus_the_boost():
    """erdm_v2_hn must differ from erdm_v2 in the three hn_* keys and NOTHING
    else -- it is the corrected arm of a controlled pair."""
    base, hn = _loss("erdm_v2"), _loss("erdm_v2_hn")
    for k, v in _HN_ERDM.items():
        assert getattr(hn, k) == pytest.approx(v), k
        assert getattr(base, k) == 0.0, f"the control must carry no boost ({k})"
    for k in ("W", "num_steps", "sigma_min", "sigma_max", "rho", "sigma_data",
              "P_mean", "P_std", "solver", "S_churn", "S_noise", "S_tmin",
              "S_tmax", "alpha", "ocean_loss_weight"):
        assert getattr(hn, k) == getattr(base, k), k


def test_a2lhn_is_rsi_a2l_plus_the_boost():
    base, hn = _loss("rsi_a2l"), _loss("rsi_a2l_hn")
    for k, v in _HN_RSI.items():
        assert getattr(hn, k) == pytest.approx(v), k
        assert getattr(base, k) == 0.0, f"the control must carry no boost ({k})"
    for k in _MIRRORED + ("weighting", "P_mean", "P_std", "w_1", "w_z",
                          "pushforward_rolls", "anchor_noise", "eps_scale"):
        assert getattr(hn, k) == getattr(base, k), k


def test_the_boost_actually_reweights_the_back_slot_in_both_configs():
    """The pair's claim: both configs multiply the TOP slot's mean loss weight
    by ~3 and leave the lower slots untouched, so a0hn vs a2lhn is a
    controlled pair even though the knob values live in different coordinates
    (ERDM sigma vs RSI sigma_eff).

    This is the mean WEIGHT ratio over a uniform-t grid -- directly computable
    from the configs. The change in each slot's share of the realized weighted
    LOSS also depends on how the model's error varies with sigma inside the
    slot; measured for A0-HN on the epoch-17 weights (job 7626106, 104 paired
    samples): slot 6 x11.19, share 1.44% -> 13.9%, total loss x1.158."""
    for name, base_name, expect in (("erdm_v2_hn", "erdm_v2", 8.14),
                                    ("rsi_a2l_hn", "rsi_a2l", 8.52)):
        base, hn = _loss(base_name), _loss(name)
        t = torch.linspace(0.0, 1.0, 2001)[:-1]
        if name.startswith("erdm"):
            w0 = base.loss_weight(base.sigma_schedule(t))
            w1 = hn.loss_weight(hn.sigma_schedule(t))
        else:
            tau = base.local_time(t)
            w0, w1 = base.loss_weight(tau), hn.loss_weight(tau)
        per_slot = [float(w1[:, w].mean() / w0[:, w].mean()) for w in range(6)]
        assert per_slot[5] == pytest.approx(expect, rel=0.05), (name, per_slot)
        for w in range(4):
            assert per_slot[w] == pytest.approx(1.0, abs=1e-3), (name, w, per_slot)


def test_hn_is_training_only_and_absent_from_the_samplers():
    """A sampler yaml carrying hn_* would be a silent claim that the weighting
    changes the sampled process. It does not."""
    for f in sorted((_CONF / "sampler").glob("*.yaml")):
        assert "hn_sigma" not in f.read_text(), f.name
        assert "hn_power" not in f.read_text(), f.name
    assert "hn_" not in "".join(
        k for k in _MIRRORED), "hn_* must stay out of the mirror list"


# ---------------------------------------------------------------------------
# Whitelisted sampler overrides: the post-hoc sweep surface (rung A5).
# ---------------------------------------------------------------------------

def _rollout_module():
    import sys
    sys.path.insert(0, str(_CONF.parent))
    import rollout
    return rollout


def test_sampler_overrides_apply_sampling_knobs():
    r = _rollout_module()
    cfg = OmegaConf.load(_CONF / "sampler" / "rsi_a2l_sstpred.yaml")
    out = r.merge_sampler_overrides(
        cfg, OmegaConf.create({"eps_mode": "gamma2", "eps_scale": 0.02,
                               "num_steps": 4}))
    assert out.eps_mode == "gamma2"
    assert out.eps_scale == pytest.approx(0.02)
    assert out.num_steps == 4
    # the original is not mutated, and the interpolant is untouched
    assert cfg.eps_scale == pytest.approx(0.0)
    assert out.anchor_lag == cfg.anchor_lag and out.gamma_0 == cfg.gamma_0


@pytest.mark.parametrize("key,val", [
    ("anchor_lag", 1), ("gamma_0", 0.5), ("h1_precond", "none"),
    ("noise_scale_path", "/tmp/other.pt"), ("parameterization", "residual"),
])
def test_sampler_overrides_reject_interpolant_keys(key, val):
    """These define the process the heads were regressed against; overriding
    one from the CLI evaluates the checkpoint under a different model and the
    shapes all still line up."""
    r = _rollout_module()
    cfg = OmegaConf.load(_CONF / "sampler" / "rsi_a2l_sstpred.yaml")
    with pytest.raises(ValueError, match="sampling knobs"):
        r.merge_sampler_overrides(cfg, OmegaConf.create({key: val}))


def test_sampler_overrides_none_is_a_passthrough():
    r = _rollout_module()
    cfg = OmegaConf.load(_CONF / "sampler" / "rsi_a2l_sstpred.yaml")
    assert r.merge_sampler_overrides(cfg, None) is cfg
    assert r.merge_sampler_overrides(cfg, OmegaConf.create({})) is cfg


def test_the_overridden_sampler_still_instantiates():
    """An eps sweep value must produce a working scheduler, not just a dict."""
    r = _rollout_module()
    cfg = OmegaConf.load(_CONF / "sampler" / "rsi_a2l_sstpred.yaml")
    cfg.noise_scale_path = None
    out = r.merge_sampler_overrides(
        cfg, OmegaConf.create({"eps_mode": "gamma2", "eps_scale": 0.05}))
    sched = instantiate(out)
    assert sched.eps_scale == pytest.approx(0.05) and sched.eps_mode == "gamma2"
    assert sched.anchor_lag == 6
