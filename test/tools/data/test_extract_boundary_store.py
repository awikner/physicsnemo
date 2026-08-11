# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12d option (b): boundary-only native-resolution store extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "data" / "amip"
sys.path.insert(0, str(_TOOLS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_boundary_store import extract_boundary_store  # noqa: E402

# Reuse the coarsen suite's synthetic store fixture (same converter schema).
from test_coarsen_zarr import _make_store  # noqa: E402

_SST = "sea_surface_temperature_monthly_interp"


def test_extracts_only_boundary_vars_at_native_resolution(tmp_path):
    src_path = tmp_path / "src.zarr"
    src = _make_store(src_path)
    out = extract_boundary_store(src_path, tmp_path / "bnd.zarr")

    # Only the role-listed boundary variables (constant + varying).
    assert sorted(out.data_vars) == ["land_sea_mask", _SST]
    # Native resolution + bit-identical values.
    assert out[_SST].shape == src[_SST].shape
    assert np.array_equal(
        np.nan_to_num(out[_SST].values, nan=-9.0),
        np.nan_to_num(src[_SST].values, nan=-9.0),
    )
    # NaN PRESERVED: the runtime NanFillTransform does the coast fade, which is
    # upstream's behavior and the reason not to pre-fill here.
    assert bool(np.isnan(out[_SST].values).any())
    # Role lists carried through so the pairing self-documents.
    assert out.attrs["varying_boundary_variables"] == [_SST]
    assert out.attrs["boundary_only"] is True
    assert "c_grid_downsample=4" in out.attrs["boundary_note"]


def test_time_chunking_matches_the_window_read(tmp_path):
    src_path = tmp_path / "src.zarr"
    _make_store(src_path)
    out = extract_boundary_store(src_path, tmp_path / "bnd.zarr", time_chunk=2)
    assert out[_SST].encoding["chunks"][0] == 2


def test_rejects_a_store_with_no_boundary_role_lists(tmp_path):
    import xarray as xr

    src_path = tmp_path / "empty.zarr"
    xr.Dataset(
        {"t2m": (("lat", "lon"), np.zeros((2, 2), dtype="float32"))},
        attrs={"constant_boundary_variables": [], "varying_boundary_variables": []},
    ).to_zarr(src_path)
    with pytest.raises(ValueError, match="no boundary variables"):
        extract_boundary_store(src_path, tmp_path / "out.zarr")
