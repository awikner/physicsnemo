# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — canonical home moved to
:mod:`physicsnemo.experimental.datapipes.climate.datapipe` in Phase 8b.
"""

from ..climate.datapipe import *  # noqa: F401, F403
from ..climate.datapipe import ClimateDatapipe  # noqa: F401

PlasimClimateDatapipe = ClimateDatapipe
