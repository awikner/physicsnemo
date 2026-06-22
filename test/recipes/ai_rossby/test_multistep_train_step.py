# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the multi-step rollout train step.

CPU-only. Builds a tiny identity-ish stub model + a sequence batch and
verifies (1) keyed inputs are required, (2) per-step loss accumulates and
averages, (3) gradients flow through the rollout chain, (4) the
single-step case (unroll_steps=1) collapses to the same value as
:func:`train_step` on the first frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_loop import multistep_train_step, train_step  # noqa: E402


class _StubModel(nn.Module):
    """Tiny model: small Conv + per-step residual on the state.

    Returns the input + a learned perturbation so backward + step actually
    update params; ``upper_air_in`` is reshaped to (B, C*L, H, W) for the
    conv, then back. Diagnostic head returns zeros sized like one surface
    channel when ``has_diagnostic=True``.
    """

    def __init__(self, n_surface=2, n_upper=3, n_levels=4, has_diagnostic=False):
        super().__init__()
        self.n_surface = n_surface
        self.n_upper = n_upper
        self.n_levels = n_levels
        self.has_diagnostic = has_diagnostic
        in_chans = n_surface + n_upper * n_levels
        out_chans = n_surface + n_upper * n_levels + (1 if has_diagnostic else 0)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in, **_):
        b, cu, l, h, w = upper_air_in.shape
        u_flat = upper_air_in.reshape(b, cu * l, h, w)
        x = torch.cat((surface_in, u_flat), dim=1)
        delta = self.conv(x)
        if self.has_diagnostic:
            d_surf = delta[:, : self.n_surface]
            d_upper = delta[:, self.n_surface : self.n_surface + cu * l]
            d_diag = delta[:, self.n_surface + cu * l :]
            next_surface = surface_in + d_surf
            next_upper = (u_flat + d_upper).reshape(b, cu, l, h, w)
            next_diag = d_diag
            return next_surface, next_upper, next_diag, 0, 0, 0, 0
        d_surf = delta[:, : self.n_surface]
        d_upper = delta[:, self.n_surface :]
        next_surface = surface_in + d_surf
        next_upper = (u_flat + d_upper).reshape(b, cu, l, h, w)
        return next_surface, next_upper, 0, 0, 0, 0


def _make_sequence_batch(B=2, T=3, Cs=2, Cu=3, L=4, H=8, W=8, has_diagnostic=False):
    torch.manual_seed(0)
    batch = {
        "surface_in_seq": torch.randn(B, T + 1, Cs, H, W),
        "upper_air_in_seq": torch.randn(B, T + 1, Cu, L, H, W),
        "varying_boundary_seq": torch.randn(B, T + 1, 2, H, W),
        "constant_boundary": torch.randn(B, 1, H, W),
    }
    if has_diagnostic:
        batch["diagnostic_seq"] = torch.randn(B, T + 1, 1, H, W)
    return batch


class _FakeLoss(nn.Module):
    """Lightweight MSE loss matching :class:`PanguPlasimLoss`'s output dict."""

    def forward(
        self,
        out_surface,
        out_upper_air,
        target_surface,
        target_upper_air,
        *,
        out_diagnostic=None,
        target_diagnostic=None,
    ):
        s = ((out_surface - target_surface) ** 2).mean()
        u = ((out_upper_air - target_upper_air) ** 2).mean()
        d = torch.zeros((), device=s.device, dtype=s.dtype)
        if out_diagnostic is not None and target_diagnostic is not None:
            d = ((out_diagnostic - target_diagnostic) ** 2).mean()
        total = s + u + d
        return {"loss": total, "surface": s.detach(), "upper_air": u.detach(), "diagnostic": d.detach()}


def test_multistep_requires_seq_keys():
    model = _StubModel()
    loss_fn = _FakeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Build a NON-sequence batch (single-step keys).
    bad_batch = {
        "surface_in": torch.zeros(1, 2, 8, 8),
        "upper_air_in": torch.zeros(1, 3, 4, 8, 8),
        "varying_boundary": torch.zeros(1, 2, 8, 8),
        "constant_boundary": torch.zeros(1, 1, 8, 8),
        "target_surface": torch.zeros(1, 2, 8, 8),
        "target_upper_air": torch.zeros(1, 3, 4, 8, 8),
    }
    with pytest.raises(KeyError):
        multistep_train_step(
            model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
            batch=bad_batch, has_diagnostic=False, unroll_steps=2,
        )


def test_multistep_loss_is_finite_and_optimizer_steps():
    torch.manual_seed(42)
    model = _StubModel()
    loss_fn = _FakeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    batch = _make_sequence_batch(B=2, T=3)
    losses = multistep_train_step(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
        batch=batch, has_diagnostic=False, unroll_steps=3,
    )
    for k in ("loss", "surface", "upper_air", "diagnostic", "vae_kl"):
        assert k in losses
        assert torch.isfinite(losses[k]).all(), (k, losses[k])
    # Optimizer actually stepped — gradients accumulated and weights moved
    # away from zero-init.
    assert any(p.abs().sum().item() > 0 for p in model.parameters())


def test_multistep_loss_decreases_after_training():
    """Sanity: model trained on a fixed batch should reduce its rollout loss."""
    torch.manual_seed(7)
    model = _StubModel()
    loss_fn = _FakeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)

    batch = _make_sequence_batch(B=2, T=2)
    losses_before = multistep_train_step(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
        batch=batch, has_diagnostic=False, unroll_steps=2,
    )
    initial = float(losses_before["loss"])
    for _ in range(8):
        multistep_train_step(
            model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
            batch=batch, has_diagnostic=False, unroll_steps=2,
        )
    losses_after = multistep_train_step(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
        batch=batch, has_diagnostic=False, unroll_steps=2,
    )
    final = float(losses_after["loss"])
    assert final < initial, f"loss did not decrease: {initial} -> {final}"


def test_unroll_steps_one_matches_singlestep_first_frame():
    """unroll_steps=1 path should produce the same loss as single-step train_step
    on the corresponding first frame + first target."""
    torch.manual_seed(11)
    model_a = _StubModel()
    model_b = _StubModel()
    # Sync weights so the two models predict identically.
    model_b.load_state_dict(model_a.state_dict())

    batch = _make_sequence_batch(B=2, T=1)
    loss_fn = _FakeLoss()

    opt_a = torch.optim.AdamW(model_a.parameters(), lr=0.0)  # no step
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.0)

    losses_a = multistep_train_step(
        model=model_a, loss_fn=loss_fn, optimizer=opt_a, scheduler=None,
        batch=batch, has_diagnostic=False, unroll_steps=1,
    )

    # Pull the equivalent single-step batch out of the sequence.
    single_batch = {
        "surface_in": batch["surface_in_seq"][:, 0],
        "upper_air_in": batch["upper_air_in_seq"][:, 0],
        "varying_boundary": batch["varying_boundary_seq"][:, 0],
        "constant_boundary": batch["constant_boundary"],
        "target_surface": batch["surface_in_seq"][:, 1],
        "target_upper_air": batch["upper_air_in_seq"][:, 1],
    }
    losses_b = train_step(
        model=model_b, loss_fn=loss_fn, optimizer=opt_b, scheduler=None,
        batch=single_batch, has_diagnostic=False, vae_kl_weight=0.0,
        amp_dtype=None, grad_scaler=None,
    )
    assert torch.allclose(losses_a["loss"], losses_b["loss"], atol=1e-6)


def test_multistep_diagnostic_path():
    """Diagnostic-head models should produce a non-zero diagnostic loss component."""
    torch.manual_seed(0)
    model = _StubModel(has_diagnostic=True)
    loss_fn = _FakeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    batch = _make_sequence_batch(B=2, T=2, has_diagnostic=True)
    losses = multistep_train_step(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
        batch=batch, has_diagnostic=True, unroll_steps=2,
    )
    assert losses["diagnostic"] > 0
