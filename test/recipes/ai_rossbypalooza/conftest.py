# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures for the ai_rossbypalooza recipe tests.

The recipe modules live under ``examples/weather/ai_rossbypalooza/`` (examples
don't ship as an installable package), so that directory is inserted on
``sys.path`` here once for all test modules. Synthetic tiny-grid zarr fixtures
for both hindcast schemas, IMERG truth, and normalization stats are added in
this file as the datapipe tests grow.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_RECIPE_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossbypalooza"
)
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
)

#: Module names both recipe directories define. ``examples/`` is not an
#: installable package, so each recipe's modules are imported by bare name off
#: ``sys.path`` — and these two names resolve to whichever recipe got imported
#: first in the session. Under ``pytest test/recipes`` that was the ai_rossby
#: suite, so the palooza tests were calling into the wrong ``train`` module and
#: failing with "module 'train' has no attribute 'run'".
_COLLIDING = ("train", "ema")


def load_recipe_module(name: str):
    """Import ``name`` from the palooza recipe, whatever else is on sys.path.

    Clears the colliding names from ``sys.modules`` and puts this recipe's
    directory first, so the fresh import — *and every sibling it pulls in
    transitively* — resolves here. That transitivity is the reason this is not
    just an ``importlib`` alias for ``train``: palooza's ``train`` does
    ``from ema import ModelEMA``, and ai_rossby defines ``ema`` too.

    Restores whatever was displaced, so the ai_rossby suite is unaffected
    regardless of which order the two run in.
    """
    saved_modules = {n: sys.modules.pop(n, None) for n in _COLLIDING}
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, str(_RECIPE_DIR))
        module = importlib.import_module(name)
        # Cheap proof we got the right copy: a silent wrong-module import is the
        # entire failure mode being fixed here.
        got = Path(getattr(module, "__file__", "")).resolve()
        if _RECIPE_DIR not in got.parents:
            raise AssertionError(
                f"imported {name!r} from {got}, expected it under {_RECIPE_DIR}"
            )
        return module
    finally:
        sys.path[:] = saved_path
        for n, mod in saved_modules.items():
            if mod is not None:
                sys.modules[n] = mod
            else:
                sys.modules.pop(n, None)


@pytest.fixture
def palooza_train():
    """The palooza recipe's ``train`` module, isolated from ai_rossby's."""
    return load_recipe_module("train")
