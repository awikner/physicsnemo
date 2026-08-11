# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12c.10 unit tests for tools/data/amip/coarsen_zarr.py.

Small synthetic per-year stores (converter attrs schema) — no real data.
The load-bearing assertion is bit-parity with upstream amip_v2's
``F.interpolate(mode='bilinear', align_corners=False)`` call, because the
x_DDC downscaler's corruption operator is *defined* by that blur.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import xarray as xr

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "data" / "amip"
sys.path.insert(0, str(_TOOLS_DIR))

from coarsen_zarr import coarsen_field, coarsen_store  # noqa: E402

_T, _L, _H, _W = 6, 3, 16, 24
_FACTOR = 4


def _make_store(path: Path, *, sst_nan: bool = True, state_nan: bool = False):
    rng = np.random.default_rng(0)
    time = xr.date_range(
        "2000-01-01", periods=_T, freq="6h", calendar="standard", use_cftime=True
    )
    lat = np.linspace(89.5, -89.5, _H).astype("float64")
    lon = (np.arange(_W) + 0.5).astype("float64")
    surf = rng.normal(size=(_T, _H, _W)).astype("float32")
    ua = rng.normal(size=(_T, _L, _H, _W)).astype("float32")
    if state_nan:
        surf[0, 0, 0] = np.nan
    diag = rng.normal(size=(_T, _H, _W)).astype("float32")
    sst = rng.normal(loc=290.0, size=(_T, _H, _W)).astype("float32")
    if sst_nan:
        sst[:, : _H // 2, :] = np.nan  # "land" half
    lsm = (rng.random(size=(_H, _W)) > 0.5).astype("float32")
    extra = rng.normal(size=(_T, _H, _W)).astype("float32")

    ds = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), surf),
            "temperature": (("time", "pressure_level", "lat", "lon"), ua),
            "PRATEsfc_24h": (("time", "lat", "lon"), diag),
            "sea_surface_temperature_monthly_interp": (("time", "lat", "lon"), sst),
            "land_sea_mask": (("lat", "lon"), lsm),
            "snow_depth": (("time", "lat", "lon"), extra),
        },
        coords={
            "time": time,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "pressure_level": ("pressure_level", np.array([100.0, 500.0, 850.0], dtype="float32")),
        },
    )
    ds.attrs = {
        "climate_zarr_schema_version": "1.1",
        "calendar": "standard",
        "data_timedelta_hours": 6,
        "surface_variables": ["t2m"],
        "pressure_upper_air_variables": ["temperature"],
        "sigma_upper_air_variables": [],
        "diagnostic_variables": ["PRATEsfc_24h"],
        "constant_boundary_variables": ["land_sea_mask"],
        "varying_boundary_variables": ["sea_surface_temperature_monthly_interp"],
        "extra_surface_variables": ["snow_depth"],
        "extra_pressure_upper_air_variables": [],
        "year_index": 2000,
    }
    ds.to_zarr(path)
    return ds


def _upstream_reference(x: np.ndarray, factor: int = _FACTOR) -> np.ndarray:
    """Literal transcription of amip_v2 modules/layers/bilinear.py."""
    t = torch.from_numpy(x)
    if t.ndim == 3:  # (T, H, W) -> treat T as batch, 1 channel
        t = t.unsqueeze(1)
        out = F.interpolate(
            t, scale_factor=1 / factor, mode="bilinear", align_corners=False
        )
        return out.squeeze(1).numpy()
    if t.ndim == 4:  # (T, L, H, W) — fold levels into channels like upstream
        out = F.interpolate(
            t, scale_factor=1 / factor, mode="bilinear", align_corners=False
        )
        return out.numpy()
    if t.ndim == 2:  # (H, W)
        out = F.interpolate(
            t[None, None],
            scale_factor=1 / factor,
            mode="bilinear",
            align_corners=False,
        )
        return out[0, 0].numpy()
    raise AssertionError(t.ndim)


def test_coarsen_field_bitmatches_upstream_interpolate():
    rng = np.random.default_rng(1)
    for shape in [(_H, _W), (_T, _H, _W), (_T, _L, _H, _W)]:
        x = rng.normal(size=shape).astype("float32")
        assert np.array_equal(coarsen_field(x, _FACTOR), _upstream_reference(x))


def test_coarsen_store_state_boundary_and_coords(tmp_path):
    src_path = tmp_path / "src.zarr"
    src = _make_store(src_path)
    out = coarsen_store(src_path, tmp_path / "out.zarr", factor=_FACTOR)

    # Shapes: state + boundary coarsened; extras skipped by default.
    assert out["t2m"].shape == (_T, _H // _FACTOR, _W // _FACTOR)
    assert out["temperature"].shape == (_T, _L, _H // _FACTOR, _W // _FACTOR)
    assert out["land_sea_mask"].shape == (_H // _FACTOR, _W // _FACTOR)
    assert "snow_depth" not in out.data_vars

    # Bit-parity with the upstream call on NaN-free state.
    assert np.array_equal(
        out["t2m"].values, _upstream_reference(src["t2m"].values)
    )
    assert np.array_equal(
        out["temperature"].values, _upstream_reference(src["temperature"].values)
    )

    # Boundary: NaN-filled with the upstream default (270 K) BEFORE the blur.
    sst_filled = np.nan_to_num(
        src["sea_surface_temperature_monthly_interp"].values, nan=270.0
    ).astype("float32")
    assert np.array_equal(
        out["sea_surface_temperature_monthly_interp"].values,
        _upstream_reference(sst_filled),
    )
    assert np.isfinite(out["sea_surface_temperature_monthly_interp"].values).all()

    # Coords: block means (1-deg pixel centers -> factor-4 centers).
    assert np.allclose(
        out["lat"].values, src["lat"].values.reshape(-1, _FACTOR).mean(axis=1)
    )
    assert np.allclose(
        out["lon"].values, src["lon"].values.reshape(-1, _FACTOR).mean(axis=1)
    )
    # Time + attrs survive; provenance recorded.
    assert out.sizes["time"] == _T
    assert out.attrs["surface_variables"] == ["t2m"]
    assert out.attrs["coarsen_factor"] == _FACTOR
    assert "align_corners=False" in out.attrs["coarsen_interpolation"]


def test_coarsen_store_small_time_blocks_match_single_pass(tmp_path):
    src_path = tmp_path / "src.zarr"
    _make_store(src_path)
    a = coarsen_store(
        src_path, tmp_path / "a.zarr", factor=_FACTOR, time_block=2, time_chunk=2
    )
    b = coarsen_store(
        src_path, tmp_path / "b.zarr", factor=_FACTOR, time_block=64, time_chunk=2
    )
    for name in a.data_vars:
        assert np.array_equal(a[name].values, b[name].values), name


def test_coarsen_store_time_chunking(tmp_path):
    # 12c benchmark finding: chunk length must serve random window reads,
    # not the processing block size (64-step chunks caused a 64x over-read
    # per sample). Pin: output chunks == time_chunk, and time_block must be
    # a chunk multiple so region writes stay chunk-aligned.
    src_path = tmp_path / "src.zarr"
    _make_store(src_path)
    out = coarsen_store(
        src_path, tmp_path / "out.zarr", factor=_FACTOR, time_block=4, time_chunk=2
    )
    assert out["t2m"].encoding["chunks"][0] == 2
    assert out.attrs["coarsen_time_chunk"] == 2
    with pytest.raises(ValueError, match="multiple"):
        coarsen_store(
            src_path, tmp_path / "bad.zarr", factor=_FACTOR, time_block=3, time_chunk=2
        )


def test_coarsen_store_include_extras(tmp_path):
    src_path = tmp_path / "src.zarr"
    src = _make_store(src_path)
    out = coarsen_store(
        src_path, tmp_path / "out.zarr", factor=_FACTOR, include_extras=True
    )
    assert np.array_equal(
        out["snow_depth"].values, _upstream_reference(src["snow_depth"].values)
    )


def test_coarsen_store_rejects_nan_state(tmp_path):
    src_path = tmp_path / "src.zarr"
    _make_store(src_path, state_nan=True)
    with pytest.raises(ValueError, match="NaN-free"):
        coarsen_store(src_path, tmp_path / "out.zarr", factor=_FACTOR)


def test_coarsen_store_rejects_unfilled_boundary_nan(tmp_path):
    src_path = tmp_path / "src.zarr"
    _make_store(src_path)
    with pytest.raises(ValueError, match="mask-fill"):
        coarsen_store(
            src_path, tmp_path / "out.zarr", factor=_FACTOR, mask_fill={}
        )


def test_coarsen_store_rejects_indivisible_grid(tmp_path):
    src_path = tmp_path / "src.zarr"
    _make_store(src_path)
    with pytest.raises(ValueError, match="divisible"):
        coarsen_store(src_path, tmp_path / "out.zarr", factor=5)

def test_smooth_boundaries_changes_coastal_coarse_cells(tmp_path):
    """Phase 12d follow-up (recommendation (a)): smooth-filling BEFORE
    coarsening is not cosmetic — it changes the value of every coarse cell
    that straddles a coastline, because a 4x4 block averages real ocean
    values with whatever fills the land side."""
    src_path = tmp_path / "src.zarr"
    src = _make_store(src_path)
    hard = coarsen_store(src_path, tmp_path / "hard.zarr", factor=_FACTOR)
    soft = coarsen_store(
        src_path, tmp_path / "soft.zarr", factor=_FACTOR, smooth_boundaries=True
    )
    name = "sea_surface_temperature_monthly_interp"
    h = hard[name].isel(time=0).values
    s = soft[name].isel(time=0).values

    # Both are NaN-free and same shape.
    assert np.isfinite(h).all() and np.isfinite(s).all()
    assert h.shape == s.shape

    # The fixture masks the northern half (rows 0..3 of 8 -> coarse row 0),
    # so coarse row 1 is the coast-adjacent band: smoothing pulls it AWAY
    # from the 270 K fill and toward the real ocean values.
    assert not np.allclose(h, s)
    assert s[1].mean() > h[1].mean()
    # Fully-masked coarse cells still relax toward the fill under smoothing,
    # so the fade is bounded by the two extremes rather than inventing data.
    raw = src[name].isel(time=0).values
    ocean_mean = float(np.nanmean(raw))
    assert 270.0 <= float(s[0].mean()) <= ocean_mean

    # Provenance records which fill built the store.
    assert hard.attrs["coarsen_boundary_fill"] == "hard"
    assert soft.attrs["coarsen_boundary_fill"] == "smooth"
    assert json.loads(soft.attrs["coarsen_smooth_params"])["n_iters"] == 10


def test_smooth_boundaries_leaves_state_and_nan_free_channels_untouched(tmp_path):
    src_path = tmp_path / "src.zarr"
    _make_store(src_path)
    hard = coarsen_store(src_path, tmp_path / "hard.zarr", factor=_FACTOR)
    soft = coarsen_store(
        src_path, tmp_path / "soft.zarr", factor=_FACTOR, smooth_boundaries=True
    )
    # State groups have no NaN, so the flag must not perturb them at all.
    for name in ("t2m", "temperature", "PRATEsfc_24h", "land_sea_mask"):
        assert np.array_equal(hard[name].values, soft[name].values), name
