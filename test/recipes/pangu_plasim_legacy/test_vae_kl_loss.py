# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :func:`vae_kl_loss` and its integration into ``train_step``.

The Pangu_Plasim recipe ships one task loss (`PanguPlasimLoss`) and one
auxiliary VAE-KL loss (``vae_kl_loss``). The trainer's ``train_step`` adds the
KL when ``vae_kl_weight > 0`` and the model emits real ``(mu, logvar, mu_e2,
logvar_e2)`` tuples; for the deterministic PanguPlasimLegacy variant the model
returns zero placeholders so the KL evaluates to zero and the task loss is
unchanged regardless of the weight.

These tests pin both behaviors so a future ``train_step`` simplification can't
silently drop the VAE path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "pangu_plasim"
sys.path.insert(0, str(_RECIPE_DIR))

from loss import PanguPlasimLoss, vae_kl_loss  # noqa: E402
from train_loop import make_optimizer, make_scheduler, train_step  # noqa: E402


# ---------------------------------------------------------------------------
# Analytic properties of vae_kl_loss.
# ---------------------------------------------------------------------------
def test_kl_zero_when_q_equals_p():
    """If q == p exactly, KL(q || p) = 0."""
    torch.manual_seed(0)
    mu = torch.randn(4, 8)
    logvar = torch.randn(4, 8)
    kl = vae_kl_loss(mu, logvar, mu, logvar)
    assert kl.abs().item() < 1e-6


def test_kl_nonneg_against_standard_normal():
    """KL(q || N(0,1)) >= 0 for any q."""
    torch.manual_seed(0)
    mu = torch.randn(4, 8)
    logvar = torch.randn(4, 8)
    kl = vae_kl_loss(mu, logvar)  # defaults to N(0,1)
    assert kl.item() >= -1e-6


def test_kl_standard_normal_against_standard_normal_is_zero():
    """KL(N(0,1) || N(0,1)) = 0."""
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    kl = vae_kl_loss(mu, logvar)
    assert kl.abs().item() < 1e-6


def test_kl_increases_when_mu_diverges():
    """KL(N(mu, 1) || N(0, 1)) increases monotonically with |mu|."""
    base = vae_kl_loss(torch.zeros(4, 8), torch.zeros(4, 8))
    big_mu = vae_kl_loss(torch.full((4, 8), 2.0), torch.zeros(4, 8))
    assert big_mu.item() > base.item()


def test_kl_supports_backward():
    """Gradients flow through the KL into mu/logvar."""
    mu = torch.randn(4, 8, requires_grad=True)
    logvar = torch.randn(4, 8, requires_grad=True)
    kl = vae_kl_loss(mu, logvar)
    kl.backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    assert logvar.grad is not None


# ---------------------------------------------------------------------------
# train_step wiring — VAE-on and VAE-off paths.
# ---------------------------------------------------------------------------
class _TinyLegacy(torch.nn.Module):
    """PanguPlasimLegacy-shaped model: returns zero placeholders for latents."""

    def __init__(self):
        super().__init__()
        self.lin_s = torch.nn.Conv2d(2, 2, 1)
        self.lin_u = torch.nn.Conv3d(3, 3, 1)

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in):
        return self.lin_s(surface_in), self.lin_u(upper_air_in), 0, 0, 0, 0


class _TinyVae(torch.nn.Module):
    """PanguPlasim-shaped (VAE) model: emits real (mu, logvar, mu_e2, logvar_e2)."""

    has_vae = True

    def __init__(self):
        super().__init__()
        self.lin_s = torch.nn.Conv2d(2, 2, 1)
        self.lin_u = torch.nn.Conv3d(3, 3, 1)
        # Tiny latent heads — emit a (B, latent_dim) mean + logvar for each encoder.
        self.encoder_q = torch.nn.Linear(2 * 4 * 8, 32)
        self.encoder_p = torch.nn.Linear(2 * 4 * 8, 32)

    def forward(
        self,
        surface_in,
        constant_boundary,
        varying_boundary,
        upper_air_in,
        target_surface=None,
        target_upper_air=None,
        train=False,
    ):
        s = self.lin_s(surface_in)
        u = self.lin_u(upper_air_in)
        flat = surface_in.flatten(start_dim=1)
        mu = self.encoder_q(flat)[:, :16]
        logvar = self.encoder_q(flat)[:, 16:]
        if train and target_surface is not None:
            t_flat = target_surface.flatten(start_dim=1)
            mu_e2 = self.encoder_p(t_flat)[:, :16]
            logvar_e2 = self.encoder_p(t_flat)[:, 16:]
        else:
            mu_e2 = torch.zeros_like(mu)
            logvar_e2 = torch.zeros_like(logvar)
        return s, u, mu, logvar, mu_e2, logvar_e2


def _toy_batch():
    return {
        "surface_in": torch.randn(2, 2, 4, 8),
        "constant_boundary": torch.zeros(2, 1, 4, 8),
        "varying_boundary": torch.zeros(2, 1, 4, 8),
        "upper_air_in": torch.randn(2, 3, 2, 4, 8),
        "target_surface": torch.ones(2, 2, 4, 8),
        "target_upper_air": torch.ones(2, 3, 2, 4, 8),
    }


def _make_loss():
    return PanguPlasimLoss(
        surface_variables=["a", "b"],
        upper_air_variable_names=["x", "y", "z"],
        diagnostic_variables=[],
        num_lat=4,
        loss_type="l2",
    )


def _make_optim_and_sched(model, total_steps=20):
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-2,
            "scheduler": "OneCycleLR",
            "oc_pct_start": 0.1,
            "oc_div_factor": 1e5,
            "oc_final_div_factor": 0.00025,
        }
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, total_steps=total_steps)
    return optimizer, scheduler


def test_train_step_legacy_path_emits_zero_vae_kl():
    """Legacy model returns zero placeholders → KL key is 0 even at vae_kl_weight=1."""
    model = _TinyLegacy()
    optimizer, scheduler = _make_optim_and_sched(model)
    losses = train_step(
        model=model,
        loss_fn=_make_loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        batch=_toy_batch(),
        has_diagnostic=False,
        vae_kl_weight=1.0,
    )
    assert "vae_kl" in losses
    assert float(losses["vae_kl"]) == pytest.approx(0.0)


def test_train_step_vae_path_emits_nonzero_kl():
    """VAE-shaped model with random latent outputs → KL > 0 when weight > 0."""
    torch.manual_seed(42)
    model = _TinyVae()
    optimizer, scheduler = _make_optim_and_sched(model)
    losses = train_step(
        model=model,
        loss_fn=_make_loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        batch=_toy_batch(),
        has_diagnostic=False,
        vae_kl_weight=1.0,
    )
    assert float(losses["vae_kl"]) > 0.0


def test_train_step_vae_weight_zero_skips_kl_term():
    """When vae_kl_weight=0, the KL slot is the zero tensor and task loss stands."""
    model = _TinyVae()
    optimizer, scheduler = _make_optim_and_sched(model)
    losses = train_step(
        model=model,
        loss_fn=_make_loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        batch=_toy_batch(),
        has_diagnostic=False,
        vae_kl_weight=0.0,
    )
    # vae_kl_weight=0 short-circuits the KL term; the slot stays zero.
    assert float(losses["vae_kl"]) == pytest.approx(0.0)


def test_train_step_vae_loss_reduces_over_iterations():
    """Sanity: with VAE-KL on, the combined loss still trends down on a toy batch."""
    torch.manual_seed(0)
    model = _TinyVae()
    optimizer, scheduler = _make_optim_and_sched(model, total_steps=40)
    batch = _toy_batch()
    initial = None
    for _ in range(30):
        out = train_step(
            model=model,
            loss_fn=_make_loss(),
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            has_diagnostic=False,
            vae_kl_weight=1e-3,
        )
        if initial is None:
            initial = float(out["loss"].detach())
    final = float(out["loss"].detach())
    assert final < initial
