# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in reduced-precision attention (2026-08-18).

Nsight Systems on a GH200, at the shipped v2 geometries: with TF32 enabled, ~70%
of a training step is fp32 *mem-efficient* attention running
``fmha_cutlass*_f32_aligned_..._sm80`` kernels. TF32 cannot touch them — they are
hand-written CUTLASS, not cuBLAS — so ERDM's attention backward measured 3488.8 ms
in fp32 and 3510.7 ms with TF32 (unchanged) while its four biggest GEMMs fell from
6416 ms to 705 ms. Attention went from 43.6% of the step to 69.7% by standing
still.

``set_attention_dtype("bf16")`` runs only the attention in bf16, which reaches the
sm90 flash kernels, and returns the result in the query's dtype so the rest of the
graph stays fp32.

These tests pin the contract rather than the speed: off by default, dtype restored
on the way out, bool masks NOT cast (a bool mask says where to attend; 0.0/1.0
would mean a bias), and the numerics close enough that this is a precision knob and
not a different operator.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from physicsnemo.experimental.models.amip_si import (
        attention_dtype,
        get_attention_dtype,
        set_attention_dtype,
    )
    from physicsnemo.experimental.models.amip_si._attention import sdpa


@pytest.fixture(autouse=True)
def _reset():
    set_attention_dtype(None)
    yield
    set_attention_dtype(None)


def _qkv(b=2, h=4, n=16, d=32, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, h, n, d, generator=g, dtype=dtype) for _ in range(3)]


def test_off_by_default_is_bit_identical_to_calling_torch_directly():
    """The default path must not perturb existing runs at all."""
    q, k, v = _qkv()
    assert get_attention_dtype() is None
    assert torch.equal(sdpa(q, k, v), F.scaled_dot_product_attention(q, k, v))


def test_the_output_dtype_follows_the_query_not_the_setting():
    """Otherwise the reduced dtype leaks into the rest of the block."""
    q, k, v = _qkv()
    with attention_dtype("bf16"):
        out = sdpa(q, k, v)
    assert out.dtype == torch.float32


def test_reduced_precision_stays_close_to_fp32():
    """A precision knob, not a different operator. bf16 has ~3 decimal digits, so
    the tolerance is set by the dtype rather than by taste."""
    q, k, v = _qkv()
    exact = F.scaled_dot_product_attention(q, k, v)
    with attention_dtype("bf16"):
        got = sdpa(q, k, v)
    assert (got - exact).abs().max() < 0.05 * exact.abs().max()


def test_already_reduced_inputs_are_left_alone():
    """Under bf16 autocast there is nothing to gain, and casting the OUTPUT back to
    fp32 would silently upcast the remainder of the block."""
    q, k, v = _qkv(dtype=torch.bfloat16)
    with attention_dtype("bf16"):
        out = sdpa(q, k, v)
    assert out.dtype == torch.bfloat16


def test_a_float_mask_follows_the_dtype():
    q, k, v = _qkv()
    mask = torch.zeros(16, 16)
    mask[:, 8:] = float("-inf")
    with attention_dtype("bf16"):
        out = sdpa(q, k, v, attn_mask=mask)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_a_bool_mask_is_not_cast():
    """Casting bool to bf16 would turn "attend here" into an additive 1.0 bias."""
    q, k, v = _qkv()
    mask = torch.ones(16, 16, dtype=torch.bool).tril()
    with attention_dtype("bf16"):
        got = sdpa(q, k, v, attn_mask=mask)
    exact = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert (got - exact).abs().max() < 0.05 * exact.abs().max()


def test_is_causal_passes_through():
    q, k, v = _qkv()
    with attention_dtype("bf16"):
        got = sdpa(q, k, v, is_causal=True)
    exact = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (got - exact).abs().max() < 0.05 * exact.abs().max()


def test_fp32_and_none_both_mean_off():
    """`fp32` reads naturally in a config and must not add a pointless cast."""
    for value in (None, "fp32", torch.float32, "none"):
        assert set_attention_dtype(value) is None


def test_an_unknown_dtype_is_refused_by_name():
    with pytest.raises(ValueError, match="attention dtype 'int8'"):
        set_attention_dtype("int8")


def test_the_context_manager_restores_even_on_error():
    set_attention_dtype("bf16")
    with pytest.raises(RuntimeError), attention_dtype("fp16"):
        raise RuntimeError("boom")
    assert get_attention_dtype() is torch.bfloat16


@pytest.mark.parametrize("model_dtype", ["bf16", "fp16"])
def test_a_real_block_runs_and_keeps_its_dtype(model_dtype):
    """End to end through a DiTBlock, since that is what both v2 backbones use:
    DiTAE composes DiTBlock and RollingDiT's spatial blocks share the same call."""
    from physicsnemo.experimental.models.amip_si import DiTBlock

    dim, heads, n = 64, 4, 16
    block = DiTBlock(dim=dim, num_heads=heads)
    x = torch.randn(2, n, dim)
    t_emb = torch.randn(2, dim)
    # RoPE tables are [1, n, head_dim // 2], per the forward's own docstring.
    half = (dim // heads) // 2
    rope = [torch.randn(1, n, half) for _ in range(4)]
    with torch.no_grad():
        ref = block(x, t_emb, *rope)
        with attention_dtype(model_dtype):
            got = block(x, t_emb, *rope)
    assert got.dtype == torch.float32
    assert (got - ref).abs().max() < 0.05 * ref.abs().max().clamp(min=1e-3)
