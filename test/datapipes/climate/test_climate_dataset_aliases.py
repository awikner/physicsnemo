# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the backward-compat alias between the new canonical climate-Zarr names
and the older PLASIM-flavored names.

The shared loader lives at
:class:`physicsnemo.experimental.datapipes.climate.ClimateZarrDataset` and is
re-exported under the old PLASIM names for backward compatibility. If anyone
later drops the alias, these tests fail loudly so the breakage is intentional
rather than silent.
"""

from __future__ import annotations

import warnings


def test_plasim_alias_is_climate_dataset():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.datapipes.climate import (
            ClimateZarrDataset,
            ClimateZarrMultiYearDataset,
            ClimateZarrStoreLayout,
        )
        from physicsnemo.experimental.datapipes.plasim import (
            PlasimClimateDataset,
            PlasimMultiYearDataset,
        )
        from physicsnemo.experimental.datapipes.plasim.dataset import PlasimStoreLayout

    assert ClimateZarrDataset is PlasimClimateDataset
    assert ClimateZarrMultiYearDataset is PlasimMultiYearDataset
    assert ClimateZarrStoreLayout is PlasimStoreLayout


def test_schema_version_alias():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.datapipes.climate import (
            CLIMATE_ZARR_SCHEMA_VERSION,
        )
        from physicsnemo.experimental.datapipes.plasim import (
            PLASIM_ZARR_SCHEMA_VERSION,
        )

    assert CLIMATE_ZARR_SCHEMA_VERSION == PLASIM_ZARR_SCHEMA_VERSION


def test_climate_package_exports():
    """Canonical names are all importable from the climate sub-package."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        import physicsnemo.experimental.datapipes.climate as climate

    for name in (
        "CLIMATE_ZARR_SCHEMA_VERSION",
        "ClimateZarrDataset",
        "ClimateZarrMultiYearDataset",
        "ClimateZarrStoreLayout",
    ):
        assert hasattr(climate, name), f"climate.{name} missing"
