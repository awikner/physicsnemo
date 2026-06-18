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
) -> dict[str, torch.Tensor]:
    """One optimizer step: forward + backward + step + scheduler tick.

    Returns the loss dict from :class:`PanguPlasimLoss` for the logger.
    Compatible with PanguPlasimLegacy (5- or 7-tuple output with zero latent
    placeholders) — the trailing four zero-tensors are ignored.
    """
    optimizer.zero_grad(set_to_none=True)
    out = model(
        batch["surface_in"],
        batch["constant_boundary"],
        batch["varying_boundary"],
        batch["upper_air_in"],
    )
    # Output tuple: (surface, upper_air[, diag], mu, sigma, mu_e2, sigma_e2).
    if has_diagnostic:
        out_surface, out_upper_air, out_diag = out[0], out[1], out[2]
    else:
        out_surface, out_upper_air = out[0], out[1]
        out_diag = None

    losses = loss_fn(
        out_surface,
        out_upper_air,
        batch["target_surface"],
        batch["target_upper_air"],
        out_diagnostic=out_diag,
        target_diagnostic=batch.get("diagnostic") if has_diagnostic else None,
    )
    losses["loss"].backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses
