# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — canonical home moved to
:mod:`physicsnemo.experimental.datapipes.climate.dataset` in Phase 8b.
"""

from ..climate.dataset import *  # noqa: F401, F403
from ..climate.dataset import (  # noqa: F401  — re-export non-* names too
    CLIMATE_ZARR_SCHEMA_VERSION,
    ClimateZarrDataset,
    ClimateZarrStoreLayout,
    PlasimStoreLayout,
)

# Legacy aliases for the dataset class + schema-version constant.
PlasimClimateDataset = ClimateZarrDataset
PLASIM_ZARR_SCHEMA_VERSION = CLIMATE_ZARR_SCHEMA_VERSION
