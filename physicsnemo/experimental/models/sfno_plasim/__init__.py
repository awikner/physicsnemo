# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""PLASIM-routed Spherical Fourier Neural Operator.

Wraps the vendored Modulus SFNO at
:mod:`physicsnemo.experimental.models.modulus_sfno` with the PLASIM-style
variable-routing contract (separate surface / constant boundary / varying
boundary / upper-air inputs, returns a tuple ``(out_surface, out_upper_air[,
out_diag], 0, 0, 0, 0)`` matching :class:`PanguPlasimLegacy`). The trainer
at ``examples/weather/ai_rossby/train.py`` picks between PanguPlasim,
PanguPlasimLegacy, and SfnoPlasim via ``cfg.model.model_type``.

See :class:`SfnoPlasim` for full documentation.
"""

from .sfno_plasim import SfnoPlasim

__all__ = ["SfnoPlasim"]
