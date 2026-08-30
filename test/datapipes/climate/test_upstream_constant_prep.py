# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Upstream-parity constant-boundary preparation (root-caused 2026-08-29).

Upstream amip_v2 prepares constant boundaries as: fill -> coast-fade the
land-sea mask (kernel 3, hardcoded) -> per-field spatial z-score with
call-time stats. The fork feeding RAW constants to upstream-trained
checkpoints crushed every varying forcing through the model's source_norm
(the ERDM bias-gap failure mode: no seasonal cycle, -117 hPa pressure
collapse in days, intact day-1 skill). These tests pin the two new knobs:
``NanFillTransform(smooth_constant_lsm=...)`` and
``ClimateNormalizer(constant_stats="spatial")`` semantics.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.climate import (
        NanFillTransform,
        smooth_masked_boundary,
    )

CONST_VARS = ["geopotential_at_surface", "land_sea_mask"]


def _sample(H=24, W=48):
    torch.manual_seed(0)
    z_sfc = torch.rand(H, W) * 5.0e4
    lsm = (torch.rand(H, W) > 0.6).to(torch.float32)
    return {
        "constant_boundary": torch.stack([z_sfc, lsm], dim=0),
        "surface_in": torch.randn(3, H, W),
    }


def _fill(**kw):
    return NanFillTransform(
        constant_boundary_variables=CONST_VARS,
        varying_boundary_variables=["sst"],
        smooth_nan_boundaries=True,
        smooth_sigma=1.5,
        smooth_kernel_size=5,
        smooth_n_iters=10,
        **kw,
    )


def test_lsm_smoothing_matches_direct_call_and_hardcodes_kernel_3():
    s = _sample()
    out = _fill(smooth_constant_lsm=True)(dict(s))
    lsm = s["constant_boundary"][1]
    ref = smooth_masked_boundary(
        lsm, (lsm > 0.5).to(lsm.dtype), sigma=1.5, kernel_size=3, n_iters=10,
        lon_circular=True,
    )
    torch.testing.assert_close(out["constant_boundary"][1], ref)
    # NOT the config kernel (5): a k=5 fade must differ somewhere off-coast
    ref5 = smooth_masked_boundary(
        lsm, (lsm > 0.5).to(lsm.dtype), sigma=1.5, kernel_size=5, n_iters=10,
        lon_circular=True,
    )
    assert not torch.equal(ref, ref5)


def test_lsm_smoothing_leaves_other_channels_and_input_untouched():
    s = _sample()
    before = s["constant_boundary"].clone()
    out = _fill(smooth_constant_lsm=True)(dict(s))
    # z_sfc channel passes through exactly
    torch.testing.assert_close(out["constant_boundary"][0], before[0])
    # the shared input tensor is NOT mutated in place
    torch.testing.assert_close(s["constant_boundary"], before)
    # land interior of the mask is preserved exactly (Dirichlet interior)
    land = before[1] > 0.5
    torch.testing.assert_close(out["constant_boundary"][1][land], before[1][land])


def test_flag_off_is_the_previous_behavior():
    s = _sample()
    out = _fill(smooth_constant_lsm=False)(dict(s))
    torch.testing.assert_close(out["constant_boundary"], s["constant_boundary"])


def test_flag_without_lsm_variable_is_a_noop():
    fill = NanFillTransform(
        constant_boundary_variables=["geopotential_at_surface"],
        varying_boundary_variables=["sst"],
        smooth_nan_boundaries=True,
        smooth_constant_lsm=True,
    )
    s = {"constant_boundary": torch.rand(1, 8, 16) * 1e4}
    out = fill(dict(s))
    torch.testing.assert_close(out["constant_boundary"], s["constant_boundary"])


# ---------------------------------------------------------------------------
# ClimateNormalizer constant_stats="spatial"
# ---------------------------------------------------------------------------


def _make_normalizer(tmp_path, constant_stats):
    import xarray as xr

    ds = xr.Dataset(
        {
            "t2m": ((), 280.0),
            "geopotential_at_surface": ((), 3709.2),
            "land_sea_mask": ((), 0.3357),
        }
    )
    std = xr.Dataset(
        {
            "t2m": ((), 21.0),
            "geopotential_at_surface": ((), 8272.3),
            "land_sea_mask": ((), 0.4537),
        }
    )
    mp, sp = tmp_path / "mean.nc", tmp_path / "std.nc"
    ds.to_netcdf(mp)
    std.to_netcdf(sp)
    from physicsnemo.experimental.datapipes.climate import ClimateNormalizer

    return ClimateNormalizer(
        mp,
        sp,
        surface_variables=["t2m"],
        varying_boundary_variables=[],
        sigma_upper_air_variables=[],
        pressure_upper_air_variables=[],
        sigma_levels=[],
        pressure_levels=[],
        constant_boundary_variables=CONST_VARS,
        normalize_constant_boundary=True,
        constant_stats=constant_stats,
    )


def test_spatial_constant_stats_zscore_matches_hand_computed(tmp_path):
    norm = _make_normalizer(tmp_path, "spatial")
    cb = _sample()["constant_boundary"]
    out = norm({"constant_boundary": cb.clone(), "surface_in": torch.randn(1, 24, 48)})
    got = out["constant_boundary"]
    for c in range(2):
        ref = (cb[c] - cb[c].mean()) / cb[c].std()  # unbiased, upstream's torch.std
        torch.testing.assert_close(got[c], ref)
    # z-scored fields: ~0 mean, ~1 std
    assert abs(float(got[0].mean())) < 1e-5
    assert abs(float(got[0].std()) - 1.0) < 1e-4


def test_file_constant_stats_uses_the_nc_entries(tmp_path):
    norm = _make_normalizer(tmp_path, "file")
    cb = _sample()["constant_boundary"]
    out = norm({"constant_boundary": cb.clone(), "surface_in": torch.randn(1, 24, 48)})
    ref0 = (cb[0] - 3709.2) / 8272.3
    torch.testing.assert_close(out["constant_boundary"][0], ref0)


def test_invalid_constant_stats_raises(tmp_path):
    with pytest.raises(ValueError, match="constant_stats"):
        _make_normalizer(tmp_path, "bogus")


def test_spatial_stats_after_lsm_smoothing_matches_upstream_composition(tmp_path):
    """The full upstream pipeline: fill -> smooth lsm -> spatial z-score."""
    s = _sample()
    fill = _fill(smooth_constant_lsm=True)
    norm = _make_normalizer(tmp_path, "spatial")
    out = norm(fill(dict(s)))["constant_boundary"]
    # reference: upstream's _load_constant_boundary_data on the same fields
    lsm = s["constant_boundary"][1]
    sm = smooth_masked_boundary(
        lsm, (lsm > 0.5).to(lsm.dtype), sigma=1.5, kernel_size=3, n_iters=10,
        lon_circular=True,
    )
    raw = torch.stack([s["constant_boundary"][0], sm], dim=0)
    mean = raw.mean(dim=(1, 2))
    std = raw.std(dim=(1, 2))
    ref = (raw - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
    torch.testing.assert_close(out, ref)
