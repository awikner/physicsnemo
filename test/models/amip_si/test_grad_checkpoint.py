# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""RollingDiT activation checkpointing must be mathematically invisible.

Checkpointing exists to fit fp32 training on 40 GB cards (Polaris/Delta
A100+A40), which RSI needs because bf16 is what destabilises it (campaign 2,
arm9c). A recompute wrapper that perturbs the forward or the gradients would
silently change every training run, so equivalence is pinned here rather than
inferred from "it looks like a no-op".
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.amip_si.rolling_dit import RollingDiT

# Deliberately tiny but structurally complete: forcing cross-attention on the
# last 2 of 4 blocks, so all three block types are exercised.
KW = dict(
    in_channels=6, out_channels=6, dim=64, num_heads=4, temporal_num_heads=4,
    num_blocks=4, nlat=8, nlon=16, scalar_dim=2, c_grid_dim=3,
    c_grid_downsample=1, window_size=3, c_grid_cross_layers=2,
    c_grid_cross_heads=4,
)
B, W = 2, 3


def _inputs(model, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(B, W, KW["in_channels"], KW["nlat"], KW["nlon"])
    t = torch.rand(B, W)
    c_grid = torch.randn(B, W, KW["c_grid_dim"], KW["nlat"], KW["nlon"])
    c_scalar = torch.randn(B, W, KW["scalar_dim"])
    return z, t, c_grid, c_scalar


def _build(grad_checkpoint):
    torch.manual_seed(1234)
    m = RollingDiT(**KW, grad_checkpoint=grad_checkpoint)
    m.train()
    return m


def test_flag_defaults_off_and_is_settable():
    m = _build(False)
    assert m.grad_checkpoint is False
    m.grad_checkpoint = True          # the training-config switch path
    assert m.grad_checkpoint is True
    assert _build(True).grad_checkpoint is True


def test_forward_is_bitwise_identical():
    off, on = _build(False), _build(True)
    on.load_state_dict(off.state_dict())
    args = _inputs(off)
    with torch.no_grad():
        a = off(*args)
    with torch.no_grad():
        b = on(*args)
    # Under no_grad the checkpoint path is skipped entirely, so this must be
    # exact, not merely close.
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_gradients_match_within_recompute_tolerance():
    off, on = _build(False), _build(True)
    on.load_state_dict(off.state_dict())
    args = _inputs(off)

    off(*args).square().mean().backward()
    on(*args).square().mean().backward()

    g_off = {k: p.grad for k, p in off.named_parameters() if p.grad is not None}
    g_on = {k: p.grad for k, p in on.named_parameters() if p.grad is not None}
    assert set(g_off) == set(g_on)
    assert g_off, "no gradients produced"
    worst, worst_k = 0.0, None
    for k in g_off:
        denom = max(g_off[k].abs().max().item(), 1e-12)
        rel = (g_off[k] - g_on[k]).abs().max().item() / denom
        if rel > worst:
            worst, worst_k = rel, k
    # Recompute reorders float ops, so exact equality is not guaranteed; the
    # bar is "indistinguishable from run-to-run noise".
    assert worst < 1e-4, f"{worst_k}: relative grad diff {worst:.3e}"


def test_checkpointing_engages_only_when_grad_is_enabled(monkeypatch):
    """The recompute wrapper must not fire under no_grad (pure overhead)."""
    import physicsnemo.experimental.models.amip_si.rolling_dit as rd

    calls = []
    real = rd.torch_checkpoint

    def counting(fn, *a, **kw):
        calls.append(1)
        return real(fn, *a, **kw)

    monkeypatch.setattr(rd, "torch_checkpoint", counting)
    m = _build(True)
    args = _inputs(m)

    with torch.no_grad():
        m(*args)
    assert not calls, "checkpoint used under no_grad"

    m(*args)
    # 4 spatial + 4 temporal + 2 forcing blocks
    assert len(calls) == 10, len(calls)


def test_no_op_when_flag_off_even_with_grad():
    import physicsnemo.experimental.models.amip_si.rolling_dit as rd

    calls = []
    real = rd.torch_checkpoint
    try:
        rd.torch_checkpoint = lambda fn, *a, **kw: (calls.append(1), real(fn, *a, **kw))[1]
        m = _build(False)
        m(*_inputs(m))
        assert not calls
    finally:
        rd.torch_checkpoint = real
