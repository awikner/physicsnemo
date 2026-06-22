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

r"""PLASIM climate datapipe — Zarr-backed lat/lon dataset with dual sigma + pressure
level systems, designed to feed
:class:`physicsnemo.experimental.models.pangu_plasim.PanguPlasim` and
:class:`physicsnemo.experimental.models.pangu_plasim.PanguPlasimLegacy`.

Companion tooling at ``tools/data/plasim/pangu_h5_to_zarr.py`` converts
PanguWeather v2.0 per-timestep HDF5 archives into the Zarr layout this
sub-package reads.

The Zarr store schema (see also
:attr:`physicsnemo.experimental.datapipes.plasim.dataset.PLASIM_ZARR_SCHEMA_VERSION`):

* Coordinates: ``time`` (cftime), ``lat`` (deg N), ``lon`` (deg E),
  optional ``pressure_level`` (Pa), optional ``sigma_level`` (unitless).
* Data variables: separate arrays per atmospheric variable, dimensioned by
  ``(time, [{level dim}], lat, lon)``; constant boundaries are
  ``(lat, lon)``.
* Store ``attrs`` carry the channel-group bookkeeping
  (``surface_variables``, ``constant_boundary_variables``,
  ``varying_boundary_variables``, ``diagnostic_variables``,
  ``pressure_upper_air_variables``, ``sigma_upper_air_variables``), the
  calendar string, and the inter-sample timedelta in hours.

See :doc:`../../../../../pangu_plasim_reuse_plan` for the design context.
"""

from .datapipe import PlasimClimateDatapipe
from .dataset import (
    CLIMATE_ZARR_SCHEMA_VERSION,
    PLASIM_ZARR_SCHEMA_VERSION,
    ClimateZarrDataset,
    ClimateZarrStoreLayout,
    PlasimClimateDataset,
)
from .multiyear import ClimateZarrMultiYearDataset, PlasimMultiYearDataset
from .samplers import LeadTimePairSampler
from .sequence import IntSampler, SequenceDataset
from .transforms import ComposeTransform, NanFillTransform, PlasimNormalizer

__all__ = [
    # Canonical climate-Zarr names (shared across PLASIM/ERA5/E3SM).
    "CLIMATE_ZARR_SCHEMA_VERSION",
    "ClimateZarrDataset",
    "ClimateZarrMultiYearDataset",
    "ClimateZarrStoreLayout",
    # PLASIM-flavored aliases (backward compatibility).
    "PLASIM_ZARR_SCHEMA_VERSION",
    "PlasimClimateDataset",
    "PlasimMultiYearDataset",
    # Other PLASIM datapipe pieces.
    "ComposeTransform",
    "IntSampler",
    "LeadTimePairSampler",
    "NanFillTransform",
    "PlasimClimateDatapipe",
    "PlasimNormalizer",
    "SequenceDataset",
]
