# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`SfnoPlasim`.

Validates:

* Constructor signature parity with :class:`PanguPlasimLegacy` for the
  shared kwargs (variable groups + level geometry).
* Forward output tuple shape: ``(out_surface, out_upper_air[, out_diag],
  0, 0, 0, 0)`` matching PanguPlasimLegacy.forward so the trainer treats
  all model families uniformly.
* The `has_diagnostic` switch toggles the diagnostic slot correctly.
* Constant boundary tensor accepts both ``(C, H, W)`` and ``(B, C, H, W)``.
* MOD-008c checkpoint roundtrip via :func:`Module.save` /
  :meth:`Module.from_checkpoint`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    import physicsnemo
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim


# Tiny config that finishes a forward pass in ≤ 1 s on CPU; mirrors the
# pattern in test/models/pangu_plasim/test_pangu_plasim.py.
_SMOKE_KWARGS = dict(
    surface_variables=["pl", "tas"],
    upper_air_variables=["ta", "ua", "va", "hus", "zg"],
    constant_boundary_variables=["lsm"],
    varying_boundary_variables=["rsdt", "sst", "sic"],
    levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    horizontal_resolution=[16, 32],
    embed_dim=32,
    num_layers=2,
    num_blocks=2,
    spectral_layers=2,
)


def _make_inputs(*, has_diagnostic=False, batch_size=1):
    H, W = 16, 32
    surface = torch.randn(batch_size, 2, H, W)
    const_b = torch.randn(1, H, W)  # unbatched — wrapper should handle
    vary_b = torch.randn(batch_size, 3, H, W)
    upper = torch.randn(batch_size, 5, 5, H, W)
    return surface, const_b, vary_b, upper


def test_constructor_no_diagnostic_outputs_six_tuple():
    model = SfnoPlasim(**_SMOKE_KWARGS).eval()
    out = model(*_make_inputs(batch_size=2))
    assert len(out) == 6
    assert out[0].shape == (2, 2, 16, 32)
    assert out[1].shape == (2, 5, 5, 16, 32)
    # Trailing 4 zero-tensor placeholders for VAE compatibility.
    for t in out[2:]:
        assert torch.is_tensor(t)
        assert t.numel() <= 1
        assert float(t.item()) == 0.0


def test_constructor_with_diagnostic_outputs_seven_tuple():
    kw = dict(_SMOKE_KWARGS, diagnostic_variables=["pr_6h"])
    model = SfnoPlasim(**kw).eval()
    out = model(*_make_inputs(batch_size=1))
    assert len(out) == 7
    assert out[0].shape == (1, 2, 16, 32)
    assert out[1].shape == (1, 5, 5, 16, 32)
    assert out[2].shape == (1, 1, 16, 32)


def test_constant_boundary_accepts_unbatched_and_batched():
    """The wrapper should broadcast (C, H, W) and slice (B, C, H, W) cleanly."""
    model = SfnoPlasim(**_SMOKE_KWARGS).eval()
    surface, _, vary_b, upper = _make_inputs(batch_size=2)
    const_b_unbatched = torch.randn(1, 16, 32)
    const_b_batched = torch.randn(2, 1, 16, 32)
    out_u = model(surface, const_b_unbatched, vary_b, upper)
    out_b = model(surface, const_b_batched, vary_b, upper)
    assert out_u[0].shape == out_b[0].shape


def test_train_kwarg_and_targets_are_accepted_and_ignored():
    """SFNO accepts the PanguPlasim signature but ignores the VAE-only kwargs."""
    model = SfnoPlasim(**_SMOKE_KWARGS).eval()
    surface, c_b, v_b, upper = _make_inputs(batch_size=1)
    out_a = model(surface, c_b, v_b, upper)
    out_b = model(
        surface, c_b, v_b, upper,
        target_surface=torch.randn_like(surface),
        target_upper_air=torch.randn_like(upper),
        train=True,
    )
    # SFNO is deterministic — same inputs (modulo unused targets) → same outputs.
    assert torch.allclose(out_a[0], out_b[0])


def test_return_latent_appends_bottleneck():
    model = SfnoPlasim(**_SMOKE_KWARGS).eval()
    inputs = _make_inputs(batch_size=1)
    out = model(*inputs, return_latent=True)
    # Tuple length = base (6 or 7) + 1 latent.
    assert len(out) == 7
    assert out[-1].ndim >= 2  # the latent has spatial dims; exact shape is SFNO-internal


def test_backward_flow_through_all_outputs():
    """Gradients reach the model from a simple sum-loss over surface + upper_air."""
    model = SfnoPlasim(**_SMOKE_KWARGS).train()
    surface, c_b, v_b, upper = _make_inputs(batch_size=1)
    out = model(surface, c_b, v_b, upper)
    loss = out[0].pow(2).mean() + out[1].pow(2).mean()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            # At least one param has nonzero gradient (model isn't dead).
    nonzero_grads = sum(
        1 for p in model.parameters() if p.grad is not None and (p.grad != 0).any()
    )
    assert nonzero_grads > 0


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    """save → Module.from_checkpoint → forward matches the pre-save output."""
    torch.manual_seed(0)
    model = SfnoPlasim(**_SMOKE_KWARGS).eval()
    surface, c_b, v_b, upper = _make_inputs(batch_size=1)
    with torch.no_grad():
        out_pre = model(surface, c_b, v_b, upper)

    ckpt = tmp_path / "sfno_plasim.mdlus"
    model.save(str(ckpt))

    loaded = physicsnemo.Module.from_checkpoint(str(ckpt)).eval()
    with torch.no_grad():
        out_post = loaded(surface, c_b, v_b, upper)

    assert torch.allclose(out_pre[0], out_post[0], atol=1e-6)
    assert torch.allclose(out_pre[1], out_post[1], atol=1e-6)
