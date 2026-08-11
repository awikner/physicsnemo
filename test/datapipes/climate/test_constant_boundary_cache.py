# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12d.15 — the constant-boundary cache question, settled by test.

Upstream amip_v2 keeps constant boundaries in a hand-built ``.npz``
(``scripts/make_constant_boundary.py``) because its per-timestep HDF5 layout
would otherwise re-read those time-invariant maps out of **every** file.

The fork needs no such artifact, and not merely because Zarr is fast: the
store holds each constant field once (no time axis) and
``ClimateZarrDataset._eager_load_constants`` reads them **once at init**,
handing every sample a reference to the same cached tensor. These tests pin
both halves of that claim — zero per-sample reads, and (the real hazard of a
shared cached tensor) that the 12d transform chain never mutates it.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import cftime
import numpy as np
import torch
import xarray as xr

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.climate import (
        ClimateZarrDataset,
        ForcingAssembler,
        NanFillTransform,
    )

_H, _W, _N_TIME = 4, 8, 6


def _write_store(path: Path) -> None:
    """Tiny AMIP-shaped store whose constant boundary carries land NaN."""
    base = cftime.DatetimeGregorian(2000, 1, 1)
    times = [base + timedelta(hours=6 * i) for i in range(_N_TIME)]
    rng = np.random.default_rng(0)
    lsm = rng.standard_normal((_H, _W)).astype("float32")
    lsm[0, 0] = np.nan  # a masked constant gridpoint
    co2 = np.full((_N_TIME, _H, _W), 380.0, dtype="float32")
    xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), rng.standard_normal((_N_TIME, _H, _W)).astype("float32")),
            "lsm": (("lat", "lon"), lsm),
            "global_mean_co2": (("time", "lat", "lon"), co2),
            "sst": (("time", "lat", "lon"), rng.standard_normal((_N_TIME, _H, _W)).astype("float32")),
            "ta": (
                ("time", "pressure_level", "lat", "lon"),
                rng.standard_normal((_N_TIME, 2, _H, _W)).astype("float32"),
            ),
        },
        coords={
            "time": ("time", times),
            "lat": ("lat", np.linspace(87.5, -87.5, _H, dtype="float32")),
            "lon": ("lon", np.linspace(0, 360 * (_W - 1) / _W, _W, dtype="float32")),
            "pressure_level": ("pressure_level", np.array([500.0, 850.0], dtype="float32")),
        },
        attrs={
            "calendar": "standard",
            "data_timedelta_hours": 6,
            "surface_variables": ["t2m"],
            "constant_boundary_variables": ["lsm"],
            "varying_boundary_variables": ["global_mean_co2", "sst"],
            "diagnostic_variables": [],
            "pressure_upper_air_variables": ["ta"],
            "sigma_upper_air_variables": [],
        },
    ).to_zarr(path, mode="w", consolidated=True, zarr_format=3)


def test_constants_are_read_once_and_shared_by_reference(tmp_path):
    path = tmp_path / "s.zarr"
    _write_store(path)
    ds = ClimateZarrDataset(path)
    # Same tensor object for every sample => no per-sample read, no copy.
    # This is what makes upstream's .npz cache redundant here.
    assert ds[(0, 1)]["constant_boundary"] is ds._constants_tensor
    assert ds[(3, 1)]["constant_boundary"] is ds._constants_tensor


def test_transform_chain_does_not_mutate_the_shared_cache(tmp_path):
    """The hazard of an eagerly cached tensor: an in-place transform would
    corrupt every later sample. The 12d chain must stay copy-on-write."""
    path = tmp_path / "s.zarr"
    _write_store(path)
    ds = ClimateZarrDataset(path, emit_calendar=True)
    cached_before = ds._constants_tensor.clone()

    ds.transform = lambda sample: ForcingAssembler(
        varying_boundary_variables=["global_mean_co2", "sst"],
        constant_boundary_variables=["lsm"],
        scalar_routed_variables=["global_mean_co2"],
    )(
        NanFillTransform(
            constant_boundary_variables=["lsm"],
            varying_boundary_variables=["global_mean_co2", "sst"],
            fill_values={"lsm": 0.0},
        )(sample)
    )

    first = ds[(0, 1)]
    # The fill removed NaN in the *returned* sample...
    assert torch.isfinite(first["constant_boundary"]).all()
    # ...while the dataset's cache still holds the original NaN, so a second
    # sample sees the same input as the first (no accumulated mutation).
    assert torch.equal(
        torch.nan_to_num(ds._constants_tensor, nan=-999.0),
        torch.nan_to_num(cached_before, nan=-999.0),
    )
    assert bool(torch.isnan(ds._constants_tensor).any())
    second = ds[(1, 1)]
    assert torch.equal(first["constant_boundary"], second["constant_boundary"])


def test_full_chain_produces_the_routed_contract(tmp_path):
    """End-to-end through a real dataset: CO₂ leaves the grid, joins calendar."""
    path = tmp_path / "s.zarr"
    _write_store(path)
    ds = ClimateZarrDataset(path, emit_calendar=True)
    asm = ForcingAssembler(
        varying_boundary_variables=["global_mean_co2", "sst"],
        constant_boundary_variables=["lsm"],
        scalar_routed_variables=["global_mean_co2"],
    )
    ds.transform = asm
    sample = ds[(0, 1)]
    assert sample["varying_boundary"].shape == (1, _H, _W)  # sst only
    assert sample["calendar"].shape == (3,)
    assert float(sample["calendar"][2]) == 380.0
    assert asm.c_grid_dim == 2  # lsm + sst
