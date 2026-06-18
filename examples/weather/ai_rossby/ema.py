# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exponential Moving Average (EMA) wrapper for model parameters.

Matches PanguWeather v2.0's training recipe (decay=0.999, warmup_epochs=6).
"""

from __future__ import annotations

from copy import deepcopy

import torch


class ModelEMA:
    r"""Track an exponential moving average of model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The training model. EMA state mirrors ``model.parameters()``.
    decay : float, optional, default=0.999
        EMA decay (PanguWeather convention: 0.999).
    warmup_epochs : int, optional, default=6
        Number of training epochs over which the effective decay ramps up:
        ``effective_decay = min(decay, (1 + epoch) / (warmup_epochs + 1))``.

    Methods
    -------
    update(model, epoch)
        Call after every optimizer.step() with the current model and epoch.
    apply_to(model)
        Copy EMA weights into ``model.parameters()`` (e.g., at validation).
    restore(model)
        Restore the original (non-EMA) weights that ``apply_to`` saved.
    state_dict() / load_state_dict()
        Standard checkpoint protocol.
    """

    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999, warmup_epochs: int = 6) -> None:
        self.decay = float(decay)
        self.warmup_epochs = int(warmup_epochs)
        # Snapshot of model params on the same device.
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    def _effective_decay(self, epoch: int) -> float:
        return min(self.decay, (1 + epoch) / (self.warmup_epochs + 1))

    @torch.no_grad()
    def update(self, model: torch.nn.Module, epoch: int) -> None:
        d = self._effective_decay(epoch)
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """Swap the model's parameters with the EMA values; back up the originals."""
        if self._backup:
            raise RuntimeError("apply_to called twice without intervening restore()")
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"shadow": dict(self.shadow), "decay": self.decay, "warmup_epochs": self.warmup_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = dict(state["shadow"])
        self.decay = float(state.get("decay", self.decay))
        self.warmup_epochs = int(state.get("warmup_epochs", self.warmup_epochs))
