# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored copy of NVIDIA Modulus's Spherical Fourier Neural Operator.

This is a byte-for-byte vendor of the SFNO building blocks from
https://github.com/ai2cm/modulus at commit 22df4a9427f5f12ff6ac891083220e7f2f54d229
— the same lineage PanguWeather v2.0 uses at ``networks/modulus_sfno/``.

Vendor rationale (Phase 7 of the ai-rossby implementation plan):

* ``makani`` (the upstream NVIDIA project that maintains an evolved version of
  this code) is **not currently installable as a hard dep** for physicsnemo —
  its pyproject pins ``nvidia-physicsnemo>=1.3.0`` while this fork tracks
  physicsnemo 2.0. The circular dep is documented at
  ``test/models/sfno/test_sfno.py:27``.
* Even when that resolves, makani's current ``SFNO`` constructor has 15+
  signature mismatches with the PanguWeather wrapper — we'd need a shim
  layer regardless.
* Pinning to the commit PanguWeather already pins gives exact
  weight-compatibility with PanguWeather's ``.pt`` SFNO checkpoints.

When upstream's circular dep clears and the constructor APIs converge, this
vendored copy can be replaced with ``import makani``. For now, the wrapper at
:mod:`physicsnemo.experimental.models.sfno_plasim` consumes the classes
exposed here.
"""

from .sfnonet import (  # noqa: F401
    SphericalFourierNeuralOperatorNet,
    SphericalFourierNeuralOperatorNet_v2,
)

__all__ = [
    "SphericalFourierNeuralOperatorNet",
    "SphericalFourierNeuralOperatorNet_v2",
]
