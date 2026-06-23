# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Back-compat shim for the renamed :mod:`...datapipes.climate` package.

The PLASIM climate datapipe was renamed to ``climate`` in Phase 8b so the
same metadata-driven loader serves PLASIM, ERA5, E3SM, and AMIP. Every
public name from the old ``.plasim`` sub-package is now re-exported here
under both the canonical ``Climate*`` names and the legacy ``Plasim*``
aliases (``PlasimNormalizer`` → :class:`ClimateNormalizer`,
``PlasimClimateDataset`` → :class:`ClimateZarrDataset`, etc.).

Prefer the canonical names in new code. Existing code that imports from
``physicsnemo.experimental.datapipes.plasim`` continues to work
unchanged.
"""

from ..climate import (
    CLIMATE_ZARR_SCHEMA_VERSION,
    ClimateDatapipe,
    ClimateNormalizer,
    ClimateZarrDataset,
    ClimateZarrMultiYearDataset,
    ClimateZarrStoreLayout,
    ComposeTransform,
    IntSampler,
    LeadTimePairSampler,
    NanFillTransform,
    SequenceDataset,
)

# Legacy ``Plasim*`` aliases — every name the old sub-package exported.
PlasimClimateDataset = ClimateZarrDataset
PlasimClimateDatapipe = ClimateDatapipe
PlasimNormalizer = ClimateNormalizer
PlasimMultiYearDataset = ClimateZarrMultiYearDataset
PLASIM_ZARR_SCHEMA_VERSION = CLIMATE_ZARR_SCHEMA_VERSION

__all__ = [
    # Canonical climate-Zarr names — same set the ``climate`` sub-package exports.
    "CLIMATE_ZARR_SCHEMA_VERSION",
    "ClimateDatapipe",
    "ClimateNormalizer",
    "ClimateZarrDataset",
    "ClimateZarrMultiYearDataset",
    "ClimateZarrStoreLayout",
    "ComposeTransform",
    "IntSampler",
    "LeadTimePairSampler",
    "NanFillTransform",
    "SequenceDataset",
    # Legacy PLASIM-prefixed aliases (kept indefinitely as no-op rebrands).
    "PLASIM_ZARR_SCHEMA_VERSION",
    "PlasimClimateDatapipe",
    "PlasimClimateDataset",
    "PlasimMultiYearDataset",
    "PlasimNormalizer",
]
