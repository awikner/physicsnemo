# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""RSI under bf16 autocast on a real GPU — the part CPU tests cannot cover.

Training runs this recipe under ``torch.autocast(bf16)``, and two pieces of RSI
are numerically hostile to that:

* **the score**, ``-Gamma^{-1} zhat``. ``gamma_1`` is a small emission floor, so
  in bf16 (max ~3.4e38 but only ~3 significant decimal digits) the reciprocal
  both overflows and quantizes. The scheduler computes it in fp32 for exactly
  this reason; these tests are what would notice if that were dropped.
* **the spherical transform** behind the spectral ``Gamma``. A bf16 SHT round
  trip loses far more than the transform costs, so it is also pinned to fp32.

Everything here is tiny and takes seconds; the point is the dtype path, not
scale.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    from physicsnemo.experimental.diffusion import RSIScheduler

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

_B, _W, _C, _H, _WD = 2, 3, 4, 16, 32


class _TwoHeadStub(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h1 = nn.Conv2d(channels, channels, 1)
        self.hz = nn.Conv2d(channels, channels, 1)

    def forward(self, x, label, c_grid, c_scalar):
        b, W = x.shape[0], x.shape[1]
        flat = x.flatten(0, 1)
        return torch.cat([self.h1(flat), self.hz(flat)], dim=1).unflatten(0, (b, W))


def _assert_bounded(out, limit=50.0, what="rollout"):
    """Assert a trajectory stayed on-scale, not merely that it is a number.

    ``torch.isfinite`` passes at 1e30. It passed throughout the preconditioning
    bug, where the frame emitted at lead W reached 4.65e5 against unit-variance
    data. See test_rsi_scheduler.py for the full note.
    """
    m = float(out.abs().max())
    assert torch.isfinite(out).all(), f"{what} produced non-finite values"
    assert m < limit, (
        f"{what} reached |{m:.3e}| against unit-variance data (limit {limit})"
    )


def _dev():
    return torch.device("cuda")


@pytest.mark.parametrize("param", ["state", "residual"])
def test_training_step_is_finite_under_bf16_autocast(param):
    torch.manual_seed(0)
    dev = _dev()
    s = RSIScheduler(window_size=_W, num_steps=2, parameterization=param,
                     gamma_0=0.5, gamma_1=0.02).to(dev)
    model = _TwoHeadStub(_C).to(dev)
    y = torch.randn(_B, _W + 1, _C, _H, _WD, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = s.compute_loss(model, None, None, y)
    assert torch.isfinite(loss), loss
    loss.backward()
    for p in model.parameters():
        assert torch.isfinite(p.grad).all()


def test_the_score_survives_a_small_noise_floor_in_bf16():
    """Gamma^{-1} with a tiny gamma_1 is the overflow case F5 called out."""
    dev = _dev()
    s = RSIScheduler(window_size=_W, gamma_0=0.5, gamma_1=1e-4).to(dev)
    zhat = torch.randn(_B, _W, _C, _H, _WD, device=dev, dtype=torch.bfloat16)
    tau = torch.ones(_B, _W, device=dev)          # tau = 1 => Gamma = gamma_1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        score = s.score(tau, zhat)
    assert torch.isfinite(score).all()
    assert score.dtype == torch.float32, "the score must be computed in fp32"
    # 1/1e-4 = 1e4 — representable, but only if the divide happened in fp32.
    assert score.abs().max() > 1e3


def test_rollout_is_finite_under_bf16_autocast():
    torch.manual_seed(0)
    dev = _dev()
    s = RSIScheduler(window_size=_W, num_steps=2, gamma_0=0.4).to(dev)
    model = _TwoHeadStub(_C).to(dev).eval()
    init = torch.randn(_B, _W + 1, _C, _H, _WD, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out = s.sample_rollout(model, init, None, None, horizon=4)
    assert out.shape == (_B, 4, _C, _H, _WD)
    _assert_bounded(out, what="bf16 rollout")


def test_sde_sampling_is_finite_under_bf16_autocast():
    """eps > 0 puts the fp32 score directly into the update."""
    torch.manual_seed(0)
    dev = _dev()
    s = RSIScheduler(window_size=_W, num_steps=2, eps_scale=0.05).to(dev)
    model = _TwoHeadStub(_C).to(dev).eval()
    init = torch.randn(_B, _W + 1, _C, _H, _WD, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out = s.sample_rollout(model, init, None, None, horizon=3)
    _assert_bounded(out, what="bf16 SDE rollout")


def test_spectral_gamma_is_finite_under_bf16_autocast():
    pytest.importorskip("torch_harmonics")
    torch.manual_seed(0)
    dev = _dev()
    s = RSIScheduler(window_size=_W, num_steps=2, spectral_sharpness=2.0,
                     grid=(_H, _WD)).to(dev)
    model = _TwoHeadStub(_C).to(dev)
    y = torch.randn(_B, _W + 1, _C, _H, _WD, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = s.compute_loss(model, None, None, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.hz.weight.grad).all()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out = s.sample_rollout(model, y, None, None, horizon=3)
    _assert_bounded(out, what="bf16 spectral rollout")


def test_fp32_and_bf16_rollouts_agree_to_bf16_tolerance():
    """A dtype path that silently diverges is worse than one that overflows."""
    torch.manual_seed(0)
    dev = _dev()
    model = _TwoHeadStub(_C).to(dev).eval()
    init = torch.randn(_B, _W + 1, _C, _H, _WD, device=dev)
    mk = lambda: RSIScheduler(window_size=_W, num_steps=2, gamma_0=0.4).to(dev)
    torch.manual_seed(3)
    with torch.no_grad():
        ref = mk().sample_rollout(model, init, None, None, horizon=3)
    torch.manual_seed(3)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        got = mk().sample_rollout(model, init, None, None, horizon=3)
    rel = (got.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
    assert rel < 0.1, f"bf16 rollout drifted {rel:.3f} from fp32"
