# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — canonical home moved to
:mod:`physicsnemo.experimental.datapipes.climate.multiyear` in Phase 8b.
"""

from ..climate.multiyear import *  # noqa: F401, F403
from ..climate.multiyear import ClimateZarrMultiYearDataset  # noqa: F401

PlasimMultiYearDataset = ClimateZarrMultiYearDataset
