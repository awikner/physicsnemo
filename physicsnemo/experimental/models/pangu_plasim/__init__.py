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

"""Pangu_Plasim weather emulators ported from PanguWeather v2.0.

Two architectures, each (eventually) in a faithful (weight-compatible) and a
native (rebuilt-on-PhysicsNeMo) flavor:

* :class:`PanguPlasim` — the current model with the training-only VAE dual-encoder.
* ``PanguPlasimLegacy`` — the predecessor model without the VAE (added next).
"""

from .pangu_plasim import PanguPlasim
from .pangu_plasim_legacy import PanguPlasimLegacy

__all__ = ["PanguPlasim", "PanguPlasimLegacy"]
