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

import contextlib
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

    Recognized keys:

    * ``optimizer_type`` — ``"AdamW"`` (only supported variant so far).
    * ``lr`` — base learning rate.
    * ``weight_decay`` — default 0.
    * ``fused`` — when True, requests the fused CUDA kernel for AdamW
      (``torch.optim.AdamW(..., fused=True)``). Requires CUDA; falls back to
      the eager AdamW with a warning if the runtime can't honor it. Defaults
      to True on CUDA (matches PanguWeather's reference SFNO trainer), False
      otherwise.
    """
    name = getattr(cfg, "optimizer_type", "AdamW")
    if name != "AdamW":
        raise ValueError(
            f"Phase 3 only supports optimizer_type='AdamW' (got {name!r}). "
            "Wire other optimizers (e.g. Muon) as the recipe matures."
        )
    fused = bool(getattr(cfg, "fused", torch.cuda.is_available()))
    kwargs = dict(
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
    )
    if fused:
        if not torch.cuda.is_available():
            import warnings as _warnings

            _warnings.warn(
                "cfg.fused=True requested but CUDA is not available; falling "
                "back to eager AdamW.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


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
    if name == "CosineAnnealingLR":
        # Plain CosineAnnealingLR — used by the AMIP diffusion recipe.
        # ``T_max`` defaults to ``total_steps`` (the per-stage budget the
        # caller supplies); ``cosine_eta_min`` mirrors the yaml key name
        # used in conf/training/amip_diffusion.yaml. ``eta_min`` is
        # accepted as a synonym so the Phase 3 LinearWarmupCosineAnnealingLR
        # config keys also work here.
        T_max = int(getattr(cfg, "T_max", total_steps))
        eta_min = float(
            getattr(cfg, "cosine_eta_min", getattr(cfg, "eta_min", 0.0))
        )
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler {name!r}")


_AMP_DTYPES = {
    "none": None,
    "off": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _resolve_amp_dtype(amp: str | bool | None) -> Optional[torch.dtype]:
    """Map ``cfg.amp`` (string or bool) to a torch dtype or ``None`` for off."""
    if amp is None or amp is False:
        return None
    if amp is True:
        return torch.bfloat16  # default-on AMP picks bf16 (matches PanguWeather)
    return _AMP_DTYPES.get(str(amp).lower())


def train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    vae_kl_weight: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
) -> dict[str, torch.Tensor]:
    """One optimizer step: forward + backward + step + scheduler tick.

    Returns the loss dict from :class:`PanguPlasimLoss` plus a ``"vae_kl"``
    entry. Compatible with both PanguPlasimLegacy (5- or 7-tuple output with
    zero latent placeholders — `vae_kl` stays ~0) and PanguPlasim with VAE
    (6- or 7-tuple with real ``mu``/``logvar``).

    Mixed-precision support
    -----------------------
    When ``amp_dtype`` is not ``None``, the forward + loss computation runs
    under ``torch.amp.autocast(device_type="cuda", dtype=amp_dtype)``. For
    ``bf16`` (matches PanguWeather v2.0's default for SFNO_PLASIM) no
    :class:`GradScaler` is needed. For ``fp16`` pass an externally-managed
    ``grad_scaler`` so the trainer can also persist its state across
    checkpoints. The optimizer step is wrapped in
    ``grad_scaler.step`` + ``grad_scaler.update`` when present.

    When ``vae_kl_weight > 0`` and the model emits real ``(mu, logvar,
    mu_e2, logvar_e2)`` tuples, the KL divergence between the two encoder
    posteriors is computed and added: ``total = task_loss + vae_kl_weight * kl``.
    For PanguPlasimLegacy the model returns zero placeholders for the latent
    fields, so the KL evaluates to 0 and the task loss is unchanged
    regardless of ``vae_kl_weight``.
    """
    from loss import vae_kl_loss  # local import keeps train_loop / loss decoupled at import time

    optimizer.zero_grad(set_to_none=True)

    # Autocast context — no-op when amp_dtype is None.
    if amp_dtype is None:
        amp_ctx = contextlib.nullcontext()
    else:
        device_type = "cuda" if batch["surface_in"].is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    with amp_ctx:
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

    # Backward + step. GradScaler is required for fp16 (underflow protection);
    # bf16 retains enough dynamic range that no scaling is needed.
    if grad_scaler is not None:
        grad_scaler.scale(losses["loss"]).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        losses["loss"].backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses


def multistep_train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    unroll_steps: int,
    vae_kl_weight: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
) -> dict[str, torch.Tensor]:
    r"""K-step rollout training with per-step loss accumulation.

    Expects ``batch`` to carry sequence keys produced by
    :class:`physicsnemo.experimental.datapipes.plasim.SequenceDataset`:

    * ``surface_in_seq``:        ``(B, T+1, C_s, H, W)``
    * ``upper_air_in_seq``:      ``(B, T+1, C_u, L, H, W)``
    * ``varying_boundary_seq``:  ``(B, T+1, C_b, H, W)``
    * ``diagnostic_seq``:        ``(B, T+1, C_d, H, W)`` (when has_diagnostic)
    * ``constant_boundary``:     ``(C_b^c, H, W)`` or ``(B, C_b^c, H, W)``

    The model is unrolled K times (K = ``unroll_steps``); the prediction
    at step k is fed back as the input state for step k+1. Per-step
    losses are summed then divided by K — the resulting scalar is in the
    same scale as the single-step loss for direct LR/EMA comparability.

    VAE-KL is not supported in this code path (the multi-step rollout
    averages predictions away from the latent encoder semantics); pass
    ``vae_kl_weight=0`` (default) or use single-step
    :func:`train_step` for the VAE variant.
    """
    if "surface_in_seq" not in batch or "upper_air_in_seq" not in batch:
        raise KeyError(
            "multistep_train_step requires sequence batch keys "
            "(`*_seq`). Use the datapipe in unroll_steps>1 mode."
        )
    if int(unroll_steps) < 1:
        raise ValueError(f"unroll_steps must be ≥ 1, got {unroll_steps}")

    optimizer.zero_grad(set_to_none=True)

    if amp_dtype is None:
        amp_ctx = contextlib.nullcontext()
    else:
        device_type = "cuda" if batch["surface_in_seq"].is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    surface_seq = batch["surface_in_seq"]               # (B, T+1, C_s, H, W)
    upper_seq = batch["upper_air_in_seq"]               # (B, T+1, C_u, L, H, W)
    varying_seq = batch["varying_boundary_seq"]         # (B, T+1, C_b, H, W)
    diag_seq = batch.get("diagnostic_seq") if has_diagnostic else None
    const_boundary = batch.get("constant_boundary")     # (C, H, W) or (B, C, H, W)

    # Initial state = first frame.
    state_surface = surface_seq[:, 0]
    state_upper = upper_seq[:, 0]

    accum_components = {
        "surface": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
        "upper_air": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
        "diagnostic": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
    }
    accum_loss = torch.zeros((), device=state_surface.device, dtype=state_surface.dtype)

    with amp_ctx:
        for k in range(int(unroll_steps)):
            boundary_in = varying_seq[:, k]
            out = model(
                state_surface,
                const_boundary,
                boundary_in,
                state_upper,
            )
            if has_diagnostic:
                next_surface, next_upper, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper = out[0], out[1]
                next_diag = None

            target_surface_k = surface_seq[:, k + 1]
            target_upper_k = upper_seq[:, k + 1]
            target_diag_k = diag_seq[:, k + 1] if diag_seq is not None else None

            losses_k = loss_fn(
                next_surface,
                next_upper,
                target_surface_k,
                target_upper_k,
                out_diagnostic=next_diag,
                target_diagnostic=target_diag_k,
            )
            accum_loss = accum_loss + losses_k["loss"]
            for comp in ("surface", "upper_air", "diagnostic"):
                if comp in losses_k:
                    val = losses_k[comp]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(float(val), device=accum_loss.device, dtype=accum_loss.dtype)
                    accum_components[comp] = accum_components[comp] + val

            # Detach the boundary path (no grad through it) but keep the
            # state path so per-step gradients flow back through the rollout.
            state_surface = next_surface
            state_upper = next_upper

    total = accum_loss / float(unroll_steps)
    avg_components = {
        k: (v / float(unroll_steps)).detach() for k, v in accum_components.items()
    }
    losses_out: dict[str, torch.Tensor] = {
        "loss": total,
        "surface": avg_components["surface"],
        "upper_air": avg_components["upper_air"],
        "diagnostic": avg_components["diagnostic"],
        "vae_kl": torch.zeros((), device=total.device, dtype=total.dtype),
    }

    if grad_scaler is not None:
        grad_scaler.scale(total).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        total.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses_out


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
