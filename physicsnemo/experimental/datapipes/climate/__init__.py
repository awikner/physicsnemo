# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Shared climate-data Zarr datapipe — PLASIM, ERA5, and E3SM all read
through the same metadata-driven loader here.

The Zarr store schema is identical across the three datasets (channel groups,
calendar, and level coords live in store ``attrs``); a single
:class:`ClimateZarrDataset` (and its multi-year composite
:class:`ClimateZarrMultiYearDataset`) handles all three. The
PLASIM-flavored aliases in
:mod:`physicsnemo.experimental.datapipes.plasim` remain valid for backward
compatibility.

See [`tools/data/`](tools/data/) for the per-dataset converters that write
into this Zarr schema.

Boundary substitution (varying boundaries from a separate store) supports
three modes via the dataset constructor kwargs:

* **Inline** (default): varying boundaries read from the prognostic store.
* **Single-year static**: pass ``boundary_zarr_path=<store>``.
* **Yearly-repeating cycle**: pass ``yearly_repeating_boundary=True`` +
  ``leap_boundary_zarr_path`` + ``non_leap_boundary_zarr_path``; the
  dataset routes by calendar-aware leap-year + day-of-year mapping.
"""

from ..plasim.dataset import (
    CLIMATE_ZARR_SCHEMA_VERSION,
    ClimateZarrDataset,
    ClimateZarrStoreLayout,
)
from ..plasim.multiyear import ClimateZarrMultiYearDataset

__all__ = [
    "CLIMATE_ZARR_SCHEMA_VERSION",
    "ClimateZarrDataset",
    "ClimateZarrMultiYearDataset",
    "ClimateZarrStoreLayout",
]
