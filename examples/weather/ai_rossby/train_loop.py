# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-epoch training step + optimizer/scheduler factories for Pangu_Plasim.

Phase 3 v1: PanguPlasimLegacy (deterministic, no VAE-KL). The optimizer +
scheduler choices come from PanguWeather v2.0 config conventions
(AdamW + OneCycleLR for the legacy variant; AdamW + LinearWarmupCosineAnnealingLR
for the future PanguPlasim with VAE).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)


def make_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build an optimizer from a config dict-like.

    Recognized keys: ``optimizer_type`` (``"AdamW"`` only for v1), ``lr``,
    ``weight_decay``.
    """
    name = getattr(cfg, "optimizer_type", "AdamW")
    if name != "AdamW":
        raise ValueError(
            f"Phase 3 v1 only supports optimizer_type='AdamW' (got {name!r}). "
            "Wire other optimizers (e.g. FusedAdam, Muon) as the recipe matures."
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a scheduler from a config dict-like.

    Supported ``scheduler``:

    * ``"OneCycleLR"`` — uses ``oc_pct_start``, ``oc_div_factor``,
      ``oc_final_div_factor`` (PanguWeather PANGU_PLASIM_H5_DERECHO_0514 keys).
      ``max_lr`` defaults to ``lr``.
    * ``"LinearWarmupCosineAnnealingLR"`` — composes a linear warmup
      (``num_warmup_steps``) with cosine annealing to ``eta_min``.
    """
    name = getattr(cfg, "scheduler", "OneCycleLR")
    if name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=float(cfg.lr),
            total_steps=total_steps,
            pct_start=float(getattr(cfg, "oc_pct_start", 0.1)),
            div_factor=float(getattr(cfg, "oc_div_factor", 1e5)),
            final_div_factor=float(getattr(cfg, "oc_final_div_factor", 0.00025)),
            anneal_strategy="cos",
        )
    if name == "LinearWarmupCosineAnnealingLR":
        warmup_steps = int(getattr(cfg, "num_warmup_steps", 0) or 0)
        warmup_start_lr = float(getattr(cfg, "warmup_start_lr", 1e-8))
        eta_min = float(getattr(cfg, "eta_min", 0.0))
        if warmup_steps <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_lr / float(cfg.lr),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    raise ValueError(f"Unknown scheduler {name!r}")


def train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    vae_kl_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """One optimizer step: forward + backward + step + scheduler tick.

    Returns the loss dict from :class:`PanguPlasimLoss` plus a ``"vae_kl"``
    entry. Compatible with both PanguPlasimLegacy (5- or 7-tuple output with
    zero latent placeholders — `vae_kl` stays ~0) and PanguPlasim with VAE
    (6- or 7-tuple with real ``mu``/``logvar``).

    When ``vae_kl_weight > 0`` and the model emits real ``(mu, logvar,
    mu_e2, logvar_e2)`` tuples, the KL divergence between the two encoder
    posteriors is computed and added: ``total = task_loss + vae_kl_weight * kl``.
    For PanguPlasimLegacy the model returns zero placeholders for the latent
    fields, so the KL evaluates to 0 and the task loss is unchanged
    regardless of ``vae_kl_weight``.
    """
    from loss import vae_kl_loss  # local import keeps train_loop / loss decoupled at import time

    optimizer.zero_grad(set_to_none=True)
    out = model(
        batch["surface_in"],
        batch["constant_boundary"],
        batch["varying_boundary"],
        batch["upper_air_in"],
        target_surface=batch.get("target_surface"),
        target_upper_air=batch.get("target_upper_air"),
        train=True,
    ) if _model_accepts_train_kwarg(model) else model(
        batch["surface_in"],
        batch["constant_boundary"],
        batch["varying_boundary"],
        batch["upper_air_in"],
    )

    # Output tuple layout:
    # * PanguPlasimLegacy (no diag): (surface, upper_air, 0, 0, 0, 0)
    # * PanguPlasimLegacy (diag):    (surface, upper_air, diag, 0, 0, 0, 0)
    # * PanguPlasim (no diag, train=True): (surface, upper_air, mu, logvar, mu_e2, logvar_e2)
    # * PanguPlasim (diag, train=True):    (surface, upper_air, diag, mu, logvar, mu_e2, logvar_e2)
    if has_diagnostic:
        out_surface, out_upper_air, out_diag = out[0], out[1], out[2]
        latent_offset = 3
    else:
        out_surface, out_upper_air = out[0], out[1]
        out_diag = None
        latent_offset = 2

    losses = loss_fn(
        out_surface,
        out_upper_air,
        batch["target_surface"],
        batch["target_upper_air"],
        out_diagnostic=out_diag,
        target_diagnostic=batch.get("diagnostic") if has_diagnostic else None,
    )

    # The VAE-KL branch fires only when (a) KL weight > 0, (b) the model
    # returned at least four latent slots, AND (c) those slots are torch
    # Tensors (the legacy port emits Python int `0` placeholders, not
    # tensors — easy sentinel for "no VAE here").
    latent_slots = out[latent_offset : latent_offset + 4] if len(out) >= latent_offset + 4 else ()
    has_real_latents = (
        len(latent_slots) == 4
        and all(isinstance(x, torch.Tensor) and x.numel() > 0 for x in latent_slots)
    )
    if vae_kl_weight > 0.0 and has_real_latents:
        mu, logvar, mu_e2, logvar_e2 = latent_slots
        kl = vae_kl_loss(mu, logvar, mu_e2, logvar_e2)
        losses["vae_kl"] = kl.detach()
        losses["loss"] = losses["loss"] + vae_kl_weight * kl
    else:
        # VAE disabled or model emits placeholders. Keep the key for logger uniformity.
        losses["vae_kl"] = torch.zeros((), device=out_surface.device, dtype=out_surface.dtype)

    losses["loss"].backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses


def _model_accepts_train_kwarg(model: torch.nn.Module) -> bool:
    """Detect whether the model's forward signature accepts ``train=`` + targets.

    The faithful PanguPlasim port takes a ``train`` flag plus optional
    ``target_*`` kwargs (it routes them through the VAE's second encoder when
    ``train=True``). PanguPlasimLegacy doesn't — its forward only takes the
    four input tensors.
    """
    inner = model.module if hasattr(model, "module") else model
    return getattr(inner, "has_vae", False) or "train" in getattr(
        inner.forward, "__code__", type("_x", (), {"co_varnames": ()})()
    ).co_varnames
