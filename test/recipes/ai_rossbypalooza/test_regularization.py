# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anti-overfitting machinery: stochastic depth, the mix() mask guard, EMA."""

from __future__ import annotations

import pytest
import torch

from ema import ModelEMA
from mowe_precip import MoWEPrecipGate, mix


def _gate(**kw):
    kw.setdefault("patch_size", (4, 4))
    kw.setdefault("hidden_size", 32)
    kw.setdefault("depth", 4)
    kw.setdefault("num_heads", 2)
    return MoWEPrecipGate(input_size=(8, 16), in_channels=3, n_experts=3, **kw)


def test_drop_path_expands_to_per_block_schedule():
    """DiT wants one rate per block; the scalar becomes 0 -> drop_path."""
    g = _gate(drop_path=0.3)
    rates = [
        float(getattr(b.drop_path, "drop_prob", 0.0)) for b in g.blocks
    ]
    assert rates == pytest.approx([0.0, 0.1, 0.2, 0.3], abs=1e-6)
    # Default is off everywhere.
    off = [float(getattr(b.drop_path, "drop_prob", 0.0)) for b in _gate().blocks]
    assert off == [0.0, 0.0, 0.0, 0.0]


def test_drop_path_rejects_out_of_range():
    with pytest.raises(ValueError, match="drop_path"):
        _gate(drop_path=1.0)
    with pytest.raises(ValueError, match="drop_path"):
        _gate(drop_path=-0.1)


def test_drop_path_is_inactive_in_eval():
    """Stochastic depth must not perturb validation."""
    g = _gate(drop_path=0.5).eval()
    x = torch.randn(2, 3, 3, 8, 16)
    m = torch.ones(2, 3)
    t = torch.full((2,), 8.0)
    with torch.no_grad():
        a = g(x, m, t)[0]
        b = g(x, m, t)[0]
    torch.testing.assert_close(a, b)


def test_mix_mask_guard_blocks_phantom_rain():
    """A masked expert's channel is non-zero in mm/day (~0.7 phantom rain);
    passing the mask must stop it contributing even with a non-zero weight."""
    p = torch.tensor([[[[4.0]], [[100.0]]]]).squeeze(-1)   # (1, 2, 1)
    w = torch.tensor([[[[0.5]], [[0.5]]]]).squeeze(-1)     # deliberately not masked-softmax
    b = torch.zeros_like(p)
    mask = torch.tensor([[1.0, 0.0]])                      # expert 1 missing
    torch.testing.assert_close(mix(w, b, p).squeeze(), torch.tensor(52.0))
    torch.testing.assert_close(
        mix(w, b, p, mask=mask).squeeze(), torch.tensor(2.0)
    )


def test_mix_mask_guard_is_noop_when_all_live():
    p = torch.rand(2, 3, 4, 5) * 10
    w = torch.full((2, 3, 4, 5), 1 / 3)
    b = torch.zeros_like(p)
    torch.testing.assert_close(mix(w, b, p), mix(w, b, p, mask=torch.ones(2, 3)))


def test_ema_tracks_then_restores_weights():
    """apply_to swaps in the averaged weights; restore puts the live ones back."""
    g = _gate()
    ema = ModelEMA(g, decay=0.5, warmup_epochs=0, steps_per_epoch=1)
    before = [p.detach().clone() for p in g.parameters()]
    ema.update(g)
    with torch.no_grad():
        for p in g.parameters():
            p.add_(1.0)
    moved = [p.detach().clone() for p in g.parameters()]

    ema.apply_to(g)
    applied = [p.detach().clone() for p in g.parameters()]
    # EMA weights sit between the first snapshot and the perturbed ones.
    assert not any(torch.allclose(a, m) for a, m in zip(applied, moved))
    assert any(
        ((a - b0).abs() < (m - b0).abs()).any()
        for a, b0, m in zip(applied, before, moved)
    )

    ema.restore(g)
    for p, m in zip(g.parameters(), moved):
        torch.testing.assert_close(p, m)


def test_ema_state_dict_round_trips():
    g = _gate()
    ema = ModelEMA(g, decay=0.9, warmup_epochs=0, steps_per_epoch=1)
    ema.update(g)
    state = ema.state_dict()
    fresh = ModelEMA(_gate(), decay=0.9, warmup_epochs=0, steps_per_epoch=1)
    fresh.load_state_dict(state)
    ema.apply_to(g)
    want = [p.detach().clone() for p in g.parameters()]
    ema.restore(g)
    fresh.apply_to(g)
    for p, w in zip(g.parameters(), want):
        torch.testing.assert_close(p, w)


def test_shrunk_gate_matches_the_configured_capacity():
    """The production gate is 192/4; guard against silently regrowing it."""
    big = MoWEPrecipGate(
        input_size=(180, 360), in_channels=9, n_experts=4,
        patch_size=(4, 4), hidden_size=384, depth=8, num_heads=8,
    )
    small = MoWEPrecipGate(
        input_size=(180, 360), in_channels=9, n_experts=4,
        patch_size=(4, 4), hidden_size=192, depth=4, num_heads=6,
    )
    n_big = sum(p.numel() for p in big.parameters())
    n_small = sum(p.numel() for p in small.parameters())
    assert 3.5e6 < n_small < 4.2e6, n_small
    assert n_big / n_small > 5.5


def test_single_expert_mask_yields_the_debiased_expert():
    """Masking to one expert must give it weight exactly 1.0, so the mixture
    reduces to P_i + b_i -- the premise of the debiased-expert experiment
    (tools/plot_week2_acc.py +debias_experts=true)."""
    g = _gate().eval()
    b, e, c, h, w = 2, 3, 3, 8, 16
    x = torch.randn(b, e, c, h, w)
    t = torch.full((b,), 10.0)

    for keep in range(e):
        xi = torch.zeros_like(x)
        xi[:, keep] = x[:, keep]
        mi = torch.zeros(b, e)
        mi[:, keep] = 1.0
        with torch.no_grad():
            weights, biases = g(xi, mi, t)
        # Exactly one live expert -> softmax over a single unmasked logit = 1.
        torch.testing.assert_close(
            weights[:, keep], torch.ones(b, h, w), rtol=0, atol=1e-6
        )
        others = [i for i in range(e) if i != keep]
        torch.testing.assert_close(
            weights[:, others], torch.zeros(b, len(others), h, w), rtol=0, atol=0
        )
        # Therefore mix() == P_keep + b_keep, exactly.
        precip = torch.rand(b, e, h, w) * 20.0
        got = mix(weights, biases, precip, mask=mi)
        torch.testing.assert_close(got, precip[:, keep] + biases[:, keep])
