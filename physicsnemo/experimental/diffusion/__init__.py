# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMIP diffusion / stochastic-interpolant schedulers (Phase 8a).

Five schedulers vendored from /work/nvme/bdiu/awikner/amip @ commit
497827e with imports rewritten to a local ``_utils`` module:

* :class:`DriftScheduler` — stochastic interpolant baseline (SI variant).
* :class:`DynamicInterpolant` — x-prediction variant (SI_X variant).
* :class:`ERDMScheduler` — Elucidated Rolling Diffusion Model.
* :class:`RFMScheduler` — Rolling Flow Matching.
* :class:`EDMScheduler` — Elucidated Diffusion (Karras et al.) — single-step
  baseline.
* :class:`DataDependentInterpolant` — the x_DDC super-resolution cascade's
  scheduler (Phase 8f, F6). Interpolates between a data-dependent
  low-res coupling and the high-res target; conditions the denoiser on
  the low-res field directly (``model(y, x_lowres, t)``) rather than a
  ``c_grid`` / ``c_scalar`` split.

Each scheduler exposes both the train and inference halves of the
diffusion contract on the same class:

* ``compute_loss(model, …) → loss`` is called by the training recipe.
* ``sample(model, …)`` / ``sample_rollout(model, …)`` is called by the
  inference recipe.

Recipes wire a *training* scheduler under the Hydra ``loss=`` group and
an *inference* scheduler under ``inference.sampler=``; the two can be
distinct instances (e.g., train ERDM, sample EDM for fast wall-time at
inference).

EDMScheduler is the only scheduler the upstream defines as a plain
class (not an :class:`nn.Module`). For API consistency this module also
exposes :class:`EDMSchedulerModule`, a thin :class:`nn.Module` adapter
that delegates to it — recipes that want the to-device / state-dict
contract can use the wrapper directly.

EDM has **no** Phase 8c wrapper / Hydra recipe wiring (Phase 8c
follow-up Q9 = b). Upstream amip's ``train_module.py`` doesn't wire
EDM either despite shipping a ``configs/EDM.yaml`` — it's vendored
here for ad-hoc use only (e.g. as a fast-sampling alternative at
inference) but doesn't participate in the train_diffusion recipe.
SI / SI_X / ERDM / RFM are the supported diffusion training paths.
"""

import torch.nn as nn

from .dynamic_interpolant import DriftScheduler
from .edm import EDMScheduler
from .erdm import ERDMScheduler
from .rfm import RFMScheduler
from .x_ddc_interpolant import DataDependentInterpolant
from .x_interpolant import DynamicInterpolant


class EDMSchedulerModule(nn.Module):
    r"""``nn.Module`` adapter over the plain :class:`EDMScheduler`.

    EDMScheduler has no learnable parameters or registered buffers — it
    only stores a handful of hyperparameters. The adapter exists so the
    recipe can call ``scheduler.to(device)`` / ``scheduler.state_dict()``
    uniformly across all 5 schedulers in this package without a special
    case for EDM.

    Parameters
    ----------
    **kwargs
        Forwarded verbatim to :class:`EDMScheduler`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._inner = EDMScheduler(**kwargs)

    def compute_loss(self, x, y, model, **kwargs):
        return self._inner.compute_loss(x, y, model, **kwargs)

    def sample(self, initial_cond, model, edm_solver="euler", **kwargs):
        return self._inner.sample(initial_cond, model, edm_solver=edm_solver, **kwargs)


__all__ = [
    "DataDependentInterpolant",
    "DriftScheduler",
    "DynamicInterpolant",
    "EDMScheduler",
    "EDMSchedulerModule",
    "ERDMScheduler",
    "RFMScheduler",
]
