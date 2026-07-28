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

"""Tests for the MoWE precip gate (mowe_precip.py)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("physicsnemo", reason="model tests need physicsnemo (DiT)")

from mowe_precip import MoWEPrecipGate, expert_dropout, mix  # noqa: E402


def _tiny_gate(n_experts=3, in_channels=4):
    return MoWEPrecipGate(
        input_size=(8, 16),
        in_channels=in_channels,
        n_experts=n_experts,
        patch_size=(2, 2),
        hidden_size=32,
        depth=1,
        num_heads=2,
        attention_backend="timm",
    )


def test_forward_shapes_and_weight_sum():
    torch.manual_seed(0)
    model = _tiny_gate()
    x = torch.randn(2, 3, 4, 8, 16)
    mask = torch.ones(2, 3)
    t = torch.tensor([8, 14])
    weights, biases = model(x, mask, t)
    assert weights.shape == (2, 3, 8, 16)
    assert biases.shape == (2, 3, 8, 16)
    torch.testing.assert_close(
        weights.sum(dim=1), torch.ones(2, 8, 16), atol=1e-5, rtol=0
    )
    assert (weights >= 0).all()


def test_masked_expert_gets_exactly_zero_weight():
    torch.manual_seed(0)
    model = _tiny_gate()
    x = torch.randn(2, 3, 4, 8, 16)
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    weights, _ = model(x, mask, torch.tensor([8, 8]))
    assert (weights[0, 1] == 0).all()
    torch.testing.assert_close(
        weights.sum(dim=1), torch.ones(2, 8, 16), atol=1e-5, rtol=0
    )
    # Fully-masked sample is rejected.
    with pytest.raises(ValueError, match="at least one live expert"):
        model(x, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), torch.tensor([8, 8]))


def test_mix_formula_and_masked_bias_vanishes():
    b, e, h, w = 2, 3, 4, 6
    weights = torch.zeros(b, e, h, w)
    weights[:, 0] = 0.25
    weights[:, 1] = 0.75
    # expert 2 masked: weight 0 -> its (P + b) must not leak.
    precip = torch.randn(b, e, h, w)
    biases = torch.randn(b, e, h, w)
    out = mix(weights, biases, precip)
    expected = 0.25 * (precip[:, 0] + biases[:, 0]) + 0.75 * (
        precip[:, 1] + biases[:, 1]
    )
    torch.testing.assert_close(out, expected)
    biases2 = biases.clone()
    biases2[:, 2] += 1000.0
    torch.testing.assert_close(out, mix(weights, biases2, precip))


def test_gradients_flow_to_gate_only():
    torch.manual_seed(0)
    model = _tiny_gate(n_experts=2, in_channels=3)
    x = torch.randn(1, 2, 3, 8, 16, requires_grad=True)
    mask = torch.ones(1, 2)
    weights, biases = model(x, mask, torch.tensor([10]))
    out = mix(weights, biases, x[:, :, 0])
    out.mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_ensemble_noise_path():
    torch.manual_seed(0)
    model = MoWEPrecipGate(
        input_size=(8, 16),
        in_channels=4,
        n_experts=3,
        patch_size=(2, 2),
        hidden_size=32,
        depth=1,
        num_heads=2,
        attention_backend="timm",
        noise_dim=8,
    )
    x = torch.randn(2, 3, 4, 8, 16)
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    noise = torch.randn(2, 5, 8)
    weights, biases = model(x, mask, torch.tensor([8, 8]), noise)
    assert weights.shape == (2, 5, 3, 8, 16)
    assert biases.shape == (2, 5, 3, 8, 16)
    assert (weights[0, :, 1] == 0).all()
    out = mix(weights, biases, x[:, :, 0])
    assert out.shape == (2, 5, 8, 16)


def test_input_validation():
    with pytest.raises(ValueError, match="divisible"):
        MoWEPrecipGate(
            input_size=(9, 16), in_channels=2, n_experts=2, patch_size=(2, 2),
            hidden_size=32, depth=1, num_heads=2, attention_backend="timm",
        )
    model = _tiny_gate()
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(1, 2, 4, 8, 16), torch.ones(1, 2), torch.tensor([8]))


class TestExpertDropout:
    def test_noop_at_zero_p(self):
        x = torch.randn(2, 3, 4, 8, 16)
        mask = torch.ones(2, 3)
        x2, m2 = expert_dropout(x, mask, 0.0)
        assert x2 is x and m2 is mask

    def test_keeps_at_least_one_and_zeroes_inputs(self):
        torch.manual_seed(0)
        x = torch.randn(64, 3, 2, 4, 4).abs() + 1.0  # strictly nonzero
        mask = torch.ones(64, 3)
        g = torch.Generator().manual_seed(0)
        x2, m2 = expert_dropout(x, mask, 0.9, generator=g)
        assert (m2.sum(dim=1) >= 1).all()
        # Dropped experts have zeroed inputs; kept experts untouched.
        for i in range(64):
            for e in range(3):
                if m2[i, e] == 0:
                    assert x2[i, e].abs().max() == 0
                else:
                    torch.testing.assert_close(x2[i, e], x[i, e])

    def test_never_revives_missing_expert(self):
        torch.manual_seed(1)
        mask = torch.tensor([[1.0, 0.0, 1.0]]).repeat(128, 1)
        x = torch.randn(128, 3, 2, 4, 4)
        g = torch.Generator().manual_seed(1)
        _, m2 = expert_dropout(x, mask, 0.95, generator=g)
        assert (m2[:, 1] == 0).all()
        assert (m2.sum(dim=1) >= 1).all()

    def test_deterministic_with_generator(self):
        x = torch.randn(8, 3, 2, 4, 4)
        mask = torch.ones(8, 3)
        a = expert_dropout(x, mask, 0.5, generator=torch.Generator().manual_seed(5))
        b = expert_dropout(x, mask, 0.5, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(a[1], b[1])
