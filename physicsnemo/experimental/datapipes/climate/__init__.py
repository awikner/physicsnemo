# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Climate-data Zarr datapipe — PLASIM, ERA5, E3SM, and AMIP all read
through the same metadata-driven :class:`ClimateZarrDataset`.

The Zarr store schema is identical across the four datasets — channel
groups, calendar, and level coords live in store ``attrs``. The dataset
class introspects the attrs and stacks per-sample tensors into the
``surface_in`` / ``upper_air_in`` / ``constant_boundary`` /
``varying_boundary`` / ``diagnostic`` groups documented at
:class:`ClimateZarrDataset`.

Companion converters at ``tools/data/{plasim,era5,e3sm,amip}/*.py``
materialize each upstream archive into this Zarr layout.

Boundary substitution (varying boundaries from a separate store)
supports three modes via the dataset constructor kwargs:

* **Inline** (default): varying boundaries read from the prognostic store.
* **Single-year static**: pass ``boundary_zarr_path=<store>``.
* **Yearly-repeating cycle**: pass ``yearly_repeating_boundary=True`` +
  ``leap_boundary_zarr_path`` + ``non_leap_boundary_zarr_path``; the
  dataset routes by calendar-aware leap-year + day-of-year mapping.

The historical sub-package
:mod:`physicsnemo.experimental.datapipes.plasim` is now a thin
re-export shim — its ``Plasim*`` names alias to the canonical classes
below for back-compat. Prefer the canonical names in new code.
"""

from .datapipe import ClimateDatapipe
from .dataset import (
    CLIMATE_ZARR_SCHEMA_VERSION,
    ClimateZarrDataset,
    ClimateZarrStoreLayout,
)
from .multiyear import ClimateZarrMultiYearDataset
from .samplers import LeadTimePairSampler
from .sequence import IntSampler, SequenceDataset
from .transforms import ClimateNormalizer, ComposeTransform, NanFillTransform

__all__ = [
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
]
