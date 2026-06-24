# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exponential Moving Average (EMA) wrapper backed by
:class:`torch.optim.swa_utils.AveragedModel`.

The public surface (``update`` / ``apply_to`` / ``restore`` / ``state_dict``
/ ``load_state_dict``) is preserved verbatim from the pre-Phase 8 bespoke
``ModelEMA`` implementation so the four call sites in
``examples/weather/ai_rossby/train.py`` don't need to move. Internally the
shadow weights are now stored inside an ``AveragedModel`` clone of the
training model, with a custom ``avg_fn`` that mirrors the historical
per-epoch warmup schedule:

    effective_decay = min(decay, (1 + epoch) / (warmup_epochs + 1))

where ``epoch`` is recovered from ``swa_utils``'s internal step counter
``num_averaged`` via ``epoch = num_averaged // steps_per_epoch``.

Notes
-----
There is one intentional behavior difference from the bespoke
implementation: when ``update`` is called for the *very first time*,
``AveragedModel`` copies the model's current parameters into the
averaged shadow rather than blending them with the construction-time
weights. The bespoke implementation blended (``shadow = d * initial + (1-d) * current``
on call 1). The difference washes out within a handful of optimizer
steps and the tests in
``test/recipes/ai_rossby/test_ema.py`` verify agreement to ``rtol=1e-5``
after warmup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel


class ModelEMA:
    r"""Track an exponential moving average of model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The training model. EMA state mirrors ``model.parameters()``.
    decay : float, optional, default=0.999
        EMA decay (PanguWeather convention: 0.999).
    warmup_epochs : int, optional, default=6
        Number of training epochs over which the effective decay ramps up.
    steps_per_epoch : int, optional, default=1
        Number of ``update`` calls per training epoch. Required to
        translate the step-counted ``swa_utils.AveragedModel.n_averaged``
        back into an epoch index for the warmup schedule. Pass
        ``len(datapipe)`` at construction.

    Methods
    -------
    update(model, epoch=0)
        Call after every ``optimizer.step()``. The ``epoch`` argument is
        retained for backward compatibility but is ignored — warmup is
        driven by the internal step counter.
    apply_to(model)
        Copy EMA weights into ``model.parameters()`` (e.g., at validation).
    restore(model)
        Restore the original (non-EMA) weights that ``apply_to`` saved.
    state_dict() / load_state_dict()
        Standard checkpoint protocol.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.999,
        warmup_epochs: int = 6,
        steps_per_epoch: int = 1,
    ) -> None:
        self.decay = float(decay)
        self.warmup_epochs = int(warmup_epochs)
        self.steps_per_epoch = max(1, int(steps_per_epoch))

        # Closure captured by AveragedModel. ``num_averaged`` is the step
        # counter immediately BEFORE this avg_fn is invoked. The first call to
        # update_parameters bypasses avg_fn entirely (copies model into the
        # shadow); calls 2..K invoke avg_fn with num_averaged = 1..K-1.
        decay_value = self.decay
        warmup_e = self.warmup_epochs
        spe = self.steps_per_epoch

        def _avg_fn(averaged_param, current_param, num_averaged):
            step = (
                int(num_averaged.item())
                if hasattr(num_averaged, "item")
                else int(num_averaged)
            )
            epoch = step // spe
            eff = min(decay_value, (1 + epoch) / (warmup_e + 1))
            return averaged_param * eff + current_param * (1.0 - eff)

        self.avg_model = AveragedModel(
            model,
            avg_fn=_avg_fn,
            use_buffers=False,  # only average params, like the bespoke impl
        )
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module, epoch: int = 0) -> None:
        """Call after each ``optimizer.step()``.

        The ``epoch`` kwarg is accepted for back-compat but ignored — the
        warmup schedule is driven entirely by the step counter inside the
        wrapped :class:`AveragedModel`.
        """
        self.avg_model.update_parameters(model)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Swap ``model.parameters()`` with the EMA values; back up the originals."""
        if self._backup:
            raise RuntimeError("apply_to called twice without intervening restore()")
        ema_state = self.avg_model.module.state_dict()
        for name, p in model.named_parameters():
            if name in ema_state:
                self._backup[name] = p.data.clone()
                p.data.copy_(ema_state[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict:
        return {
            "avg_model_state": self.avg_model.state_dict(),
            "decay": self.decay,
            "warmup_epochs": self.warmup_epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self.avg_model.load_state_dict(state["avg_model_state"])
        self.decay = float(state.get("decay", self.decay))
        self.warmup_epochs = int(state.get("warmup_epochs", self.warmup_epochs))
        self.steps_per_epoch = int(state.get("steps_per_epoch", self.steps_per_epoch))
