# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that let deterministic models ride the diffusion eval machinery.

The whole gap between the two families is one seam: the diffusion validators
step a model via ``scheduler.sample(model, x, c_grid, c_scalar, num_steps)``
on packed flat tensors, while the deterministic families (SfnoPlasim,
PanguPlasim/Legacy/Native, ArchesWeather) expose one shared positional
forward::

    model(surface_in, constant_boundary, varying_boundary, upper_air_in,
          **{surface_prev_in, upper_air_prev_in, calendar})   # extras iff named
    -> (out_surface, out_upper_air[, out_diagnostic], ...)

and have no ``pack_state``/``unpack_state``/``pack_c_grid``. Two shims close
it:

* :class:`DeterministicPackShim` — wraps the model with the pack/unpack
  surface the validators drive, plus the variable-name attributes
  (``surface_variables`` etc.) the QBO/flux scorers read. Names come from a
  :class:`~climate_eval_suite.VariableCatalog` built from ``cfg.model`` —
  NOT from the model class, because the Pangu families expose only counts.
* :class:`DeterministicStepAdapter` — quacks like a single-step scheduler:
  ``.sample()`` performs one deterministic forward. It deliberately does NOT
  define ``sample_rollout``, so ``DiffusionRolloutValidator`` takes its
  single-step autoregressive path.

CAUTION: the shim intentionally quacks like a diffusion wrapper
(``pack_state`` is exactly the attribute ``inference.py._is_diffusion_model``
dispatches on). Never hand a shim to inference.py — the eval suite performs
its own family dispatch on the UNWRAPPED model before shimming.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeterministicPackShim(torch.nn.Module):
    """Pack/unpack + variable-name surface for a deterministic model.

    Channel layout of the packed state:
    ``[surface | upper_air flattened var-major (var0 all levels, var1 ...) |
    diagnostic?]`` — the layout is PRIVATE to the eval drive (packed by
    ``pack_state``, unpacked by ``unpack_state``, never touching a
    checkpoint), so it only has to be self-consistent.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        catalog,
        n_constant: int,
        n_varying: int,
        has_diagnostic: bool,
    ):
        super().__init__()
        self.model = model
        self.surface_variables = list(catalog.surface)
        self.upper_air_variables = list(catalog.upper_air)
        self.diagnostic_variables = list(catalog.diagnostic)
        self.levels = list(catalog.levels)
        self.n_constant = int(n_constant)
        self.n_varying = int(n_varying)
        self.has_diagnostic = bool(has_diagnostic)
        self._n_surface = len(self.surface_variables)
        self._n_upper_flat = len(self.upper_air_variables) * len(self.levels)
        self._n_diag = len(self.diagnostic_variables) if has_diagnostic else 0

    # -- state ---------------------------------------------------------- #
    def pack_state(self, sample: dict) -> torch.Tensor:
        parts = [sample["surface_in"]]
        ua = sample["upper_air_in"]
        b_shape = ua.shape[:-4]
        parts.append(
            ua.reshape(*b_shape, self._n_upper_flat, *ua.shape[-2:])
        )
        if self._n_diag:
            diag = sample.get("diagnostic")
            if diag is None:
                # First step of a rollout may have no diagnostic yet (it is
                # an OUTPUT of the model, not part of the IC) — pad zeros so
                # the packed width is constant across the rollout.
                diag = ua.new_zeros(*b_shape, self._n_diag, *ua.shape[-2:])
            parts.append(diag)
        return torch.cat(parts, dim=-3)

    def unpack_state(self, x: torch.Tensor) -> dict:
        n_s, n_u = self._n_surface, self._n_upper_flat
        out = {"surface_in": x.narrow(-3, 0, n_s)}
        ua_flat = x.narrow(-3, n_s, n_u)
        b_shape = ua_flat.shape[:-3]
        out["upper_air_in"] = ua_flat.reshape(
            *b_shape,
            len(self.upper_air_variables),
            len(self.levels),
            *ua_flat.shape[-2:],
        )
        if self._n_diag:
            out["diagnostic"] = x.narrow(-3, n_s + n_u, self._n_diag)
        return out

    # -- forcings -------------------------------------------------------- #
    def pack_c_grid(self, sample: dict) -> torch.Tensor:
        const = sample["constant_boundary"]
        surface = sample["surface_in"]
        while const.dim() < surface.dim():
            const = const.unsqueeze(0)
        const = const.expand(*surface.shape[:-3], -1, -1, -1)
        return torch.cat([const, sample["varying_boundary"]], dim=-3)

    def split_c_grid(self, c_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Undo :meth:`pack_c_grid` — the deterministic forward takes the
        constant and varying boundaries as two separate positional args."""
        return torch.split(c_grid, [self.n_constant, self.n_varying], dim=-3)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class DeterministicStepAdapter:
    """Single-step 'scheduler' whose sample() is one deterministic forward.

    Holds the previous input state so ArchesWeather-style models receive
    ``surface_prev_in`` / ``upper_air_prev_in``; :meth:`on_rollout_start`
    (called by the drive at the top of every IC batch) resets it. At k=1
    the extras are omitted and the model's own persistence fallback applies
    — a deliberate one-step transient vs inference.py, which fetches the
    true archive frame ``prev_state_steps`` back.
    """

    #: advertised for parity with real schedulers; a deterministic forward
    #: has exactly one "step".
    num_steps = 1

    def __init__(self, shim: DeterministicPackShim, *, optional_kwargs: set):
        self.shim = shim
        self.optional_kwargs = set(optional_kwargs)
        self._prev: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._warned_num_steps = False

    def on_rollout_start(self, state: dict) -> None:
        self._prev = None

    @torch.no_grad()
    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        if num_steps not in (None, 1) and not self._warned_num_steps:
            logger.info(
                "sampler_num_steps=%s is meaningless for a deterministic "
                "model (one forward per frame); ignoring.", num_steps,
            )
            self._warned_num_steps = True
        shim = self.shim
        unpacked = shim.unpack_state(x)   # diagnostic slice ignored as input
        surface = unpacked["surface_in"]
        upper = unpacked["upper_air_in"]
        const, varying = shim.split_c_grid(c_grid)
        extras = {}
        if "surface_prev_in" in self.optional_kwargs and self._prev is not None:
            extras["surface_prev_in"] = self._prev[0]
            extras["upper_air_prev_in"] = self._prev[1]
        if "calendar" in self.optional_kwargs and c_scalar is not None:
            extras["calendar"] = c_scalar
        out = shim.model(surface, const, varying, upper, **extras)
        self._prev = (surface, upper)
        next_state = {"surface_in": out[0], "upper_air_in": out[1]}
        if shim.has_diagnostic and len(out) > 2 and torch.is_tensor(out[2]):
            next_state["diagnostic"] = out[2]
        return shim.pack_state(next_state)
