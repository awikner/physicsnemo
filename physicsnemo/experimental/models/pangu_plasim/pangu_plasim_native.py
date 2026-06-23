# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Native PhysicsNeMo variants of the Pangu_Plasim emulators.

After Phase 6 Track A the faithful Pangu_Plasim classes are themselves
backed by upstream :mod:`physicsnemo.nn` blocks. The *native* classes in
this module are thin subclasses that differ from the faithful pair only
in their :class:`ModelMetaData`:

* :class:`PanguPlasimLegacyNative` — same architecture, same forward, same
  state-dict layout as :class:`PanguPlasimLegacy`, but advertises
  ``cuda_graphs=True`` + ``auto_grad=True`` so the model can be wrapped in
  :class:`physicsnemo.utils.capture.StaticCaptureTraining` with no Module
  introspection complaining about an unsupported optimization path.
* :class:`PanguPlasimNative` — same relationship to :class:`PanguPlasim`
  (the VAE variant).

Trainer / inference code routes to these via the standard
``cfg.model.name`` switch; the native variants are intended for *fresh*
training runs that exercise the AMP + CUDA-graph fast path. The
:mod:`tools.checkpoint_translation.pangu_plasim` translator continues to
target the faithful classes only — translation into native variants is
not supported because the design intent is to evolve the native
architectures independently of the PanguWeather .tar contract.

CUDA-graph caveat
-----------------
The faithful classes' forward includes activation checkpointing
controlled by the ``checkpointing`` constructor kwarg. CUDA graphs are
incompatible with ``torch.utils.checkpoint.checkpoint``, so users must
set ``checkpointing=0`` (the default) when wrapping a native variant in
``StaticCaptureTraining`` with CUDA graphs enabled. AMP without CUDA
graphs is unaffected.
"""

from dataclasses import dataclass

from physicsnemo.core.meta import ModelMetaData

from .pangu_plasim import PanguPlasim
from .pangu_plasim_legacy import PanguPlasimLegacy


@dataclass
class _NativeMetaData(ModelMetaData):
    """ModelMetaData advertising the StaticCapture-friendly fast paths.

    Defaults to ``cuda_graphs=True`` + ``auto_grad=True``. The user is
    responsible for disabling activation checkpointing
    (``checkpointing=0``) when wrapping with CUDA graphs.
    """

    jit: bool = False  # activation checkpoints + dynamic-shape masks aren't jit-traceable
    cuda_graphs: bool = True
    amp: bool = True
    amp_gpu: bool = True
    bf16: bool = True
    auto_grad: bool = True
    onnx: bool = False


class PanguPlasimLegacyNative(PanguPlasimLegacy):
    r"""Native PhysicsNeMo build of :class:`PanguPlasimLegacy`.

    Identical layer composition, forward, and state-dict layout as the
    faithful class — but registered with a CUDA-graph-friendly
    :class:`ModelMetaData`. Intended for fresh training runs (no
    PanguWeather .tar checkpoint translation path).

    Signature note
    --------------
    Uses **kwargs-only forwarding** rather than ``*args, **kwargs``. The
    parent :class:`physicsnemo.Module` introspects ``inspect.signature``
    at construction to persist constructor args for checkpoint
    round-trip — a ``*args`` parameter named ``args`` would be saved as
    the literal kwarg ``args=...`` and break the load path. All callers
    (Module.instantiate, build_model, tests) pass kwargs, so this is no
    practical restriction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override the faithful class' MetaData with the native one.
        self.meta = _NativeMetaData()


class PanguPlasimNative(PanguPlasim):
    r"""Native PhysicsNeMo build of :class:`PanguPlasim` (VAE variant).

    Same relationship to :class:`PanguPlasim` as :class:`PanguPlasimLegacyNative`
    has to :class:`PanguPlasimLegacy`. The VAE branch is unchanged —
    only the ModelMetaData differs.

    See :class:`PanguPlasimLegacyNative` for the kwargs-only signature
    rationale.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta = _NativeMetaData()


__all__ = ["PanguPlasimLegacyNative", "PanguPlasimNative"]
