# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Multi-year archives in the diffusion training recipe (2026-08-17).

``train_diffusion._build_dataset`` opened ``ClimateZarrDataset`` unconditionally,
so the recipe could only ever train on ONE year — while the upstream SI and ERDM
runs trained 1979-2015. ``train.py`` and ``inference.py`` had routed a directory
to :class:`ClimateZarrMultiYearDataset` since the multi-year port; this pins the
same routing here, and pins the two properties that make it usable rather than
merely present:

* pairs that straddle a year boundary read their target from the NEXT year, and
* the whole forcing pipeline (fill -> normalize -> route, plus the
  varying-boundary subset) still applies through the multi-year dataset.

Note what is deliberately NOT asserted: that the cross-year branch produces a
*different* ``diagnostic`` from the single-year path. It does not, and should not
— both take it from the start frame (measured). An earlier plan of mine called
that a bug; "fixing" it would have made the two paths disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_diffusion import _build_dataset  # noqa: E402

_H, _W = 2, 4
_NT = 12                      # rows per year store
_LEVELS = [500.0]
_SURFACE = ["skin_temperature"]
_UPPER = ["temperature"]
_DIAG = ["PRATEsfc_24h"]
_CONST = ["land_sea_mask"]
_VARY = ["global_mean_co2", "sea_ice_cover_monthly_interp"]


def _write_year(path: Path, offset: int) -> None:
    """A year store whose every field encodes its own GLOBAL row index.

    That makes "which frame did this come from" readable straight off the value,
    which is the only way to check the year seams without trusting the loader.
    """
    t = np.arange(_NT) + offset
    f3 = lambda: (  # noqa: E731
        ("time", "lat", "lon"),
        (t[:, None, None] * np.ones((_NT, _H, _W))).astype("float32"),
    )
    data = {n: f3() for n in _SURFACE + _DIAG + _VARY}
    data[_UPPER[0]] = (
        ("time", "pressure_level", "lat", "lon"),
        (t[:, None, None, None] * np.ones((_NT, len(_LEVELS), _H, _W))).astype("float32"),
    )
    data[_CONST[0]] = (("lat", "lon"), np.zeros((_H, _W), "float32"))
    # Real datetimes, 6 h apart: _build_dataset opens with emit_calendar=True,
    # which reads `.year` off the coord. The GLOBAL row index lives in the field
    # VALUES, not here, so the seam checks below are unaffected.
    times = [
        cftime.num2date(6 * (offset + i), "hours since 1981-01-01", calendar="standard")
        for i in range(_NT)
    ]
    xr.Dataset(
        data,
        coords={
            "time": ("time", times),
            "pressure_level": ("pressure_level", np.array(_LEVELS, "float32")),
            "lat": ("lat", np.linspace(-45, 45, _H).astype("float32")),
            "lon": ("lon", np.linspace(0, 270, _W).astype("float32")),
        },
        attrs={
            "data_timedelta_hours": 6,
            "surface_variables": _SURFACE,
            "diagnostic_variables": _DIAG,
            "pressure_upper_air_variables": _UPPER,
            "constant_boundary_variables": _CONST,
            "varying_boundary_variables": _VARY,
            "climate_zarr_schema_version": 1,
        },
    ).to_zarr(path, mode="w", consolidated=True)


def _write_norm(path: Path, fill: float) -> None:
    """Mean and std go in SEPARATE files: mean 0 / std 1 makes normalization the
    identity, so a field's value still reads as its global row index below."""
    data = {n: ((), np.float32(fill)) for n in _SURFACE + _DIAG + _VARY + _CONST}
    data[_UPPER[0]] = (("pressure_level",), np.full(len(_LEVELS), fill, "float32"))
    xr.Dataset(
        data,
        coords={"pressure_level": ("pressure_level", np.array(_LEVELS, "float32"))},
    ).to_netcdf(path)


def _cfg(tmp_path: Path, *, multiyear: bool):
    archive = tmp_path / "arch"
    archive.mkdir(exist_ok=True)
    for i, year in enumerate((1981, 1982)):
        p = archive / f"{year}.zarr"
        if not p.exists():
            _write_year(p, i * _NT)
    mean_nc, std_nc = tmp_path / "mean.nc", tmp_path / "std.nc"
    if not mean_nc.exists():
        _write_norm(mean_nc, 0.0)
        _write_norm(std_nc, 1.0)
    zarr_path = str(archive) if multiyear else str(archive / "1981.zarr")
    return OmegaConf.create({
        "model": {
            "surface_variables": _SURFACE,
            "upper_air_variables": _UPPER,
            "diagnostic_variables": _DIAG,
            "constant_boundary_variables": _CONST,
            # The model consumes only the gridded one: exercises the subset path
            # through the multi-year dataset too.
            "varying_boundary_variables": ["sea_ice_cover_monthly_interp"],
            "scalar_routed_boundary_variables": [],
            "levels": _LEVELS,
            "timedelta_hours": 24,
        },
        "dataset": {
            "zarr_path": zarr_path,
            "boundary_zarr_path": None,
            "yearly_repeating_boundary": False,
            "leap_boundary_zarr_path": None,
            "non_leap_boundary_zarr_path": None,
            "mean_path": str(mean_nc),
            "std_path": str(std_nc),
            "forecast_lead_times": [4],
            "nan_fill_default": 0.0,
            "nan_fill_values": {},
            "normalize_constant_boundary": False,
            "normalize_diagnostic": False,
        },
    })


def test_a_directory_routes_to_the_multiyear_dataset(tmp_path):
    from physicsnemo.experimental.datapipes.climate import ClimateZarrMultiYearDataset

    ds = _build_dataset(_cfg(tmp_path, multiyear=True))
    assert isinstance(ds, ClimateZarrMultiYearDataset)
    # Both years visible as one contiguous timeline — the point of the change.
    assert ds.n_time == 2 * _NT


def test_a_single_zarr_still_routes_to_the_one_year_dataset(tmp_path):
    from physicsnemo.experimental.datapipes.climate import (
        ClimateZarrDataset,
        ClimateZarrMultiYearDataset,
    )

    ds = _build_dataset(_cfg(tmp_path, multiyear=False))
    assert isinstance(ds, ClimateZarrDataset)
    assert not isinstance(ds, ClimateZarrMultiYearDataset)
    assert ds.n_time == _NT


def test_a_pair_straddling_the_year_boundary_reads_the_next_year(tmp_path):
    """global 10 -> 14 crosses the seam at 12. Values encode the row index."""
    ds = _build_dataset(_cfg(tmp_path, multiyear=True))
    sample = ds[(10, 4)]
    assert float(sample["surface_in"].reshape(-1)[0]) == pytest.approx(10.0)
    assert float(sample["target_surface"].reshape(-1)[0]) == pytest.approx(14.0)
    # Second year's rows are reachable at all, i.e. the seam is not a wall.
    late = ds[(20, 2)]
    assert float(late["surface_in"].reshape(-1)[0]) == pytest.approx(20.0)


def test_the_cross_year_diagnostic_matches_the_single_year_convention(tmp_path):
    """Both take `diagnostic` from the START frame — pinned so it stays that way.

    ``_pack_single_step`` builds the SI target's diagnostic block from
    ``sample["diagnostic"]``, so if these two paths ever disagreed, pairs at the
    seams would train against a different frame than pairs in mid-year, silently.
    """
    multi = _build_dataset(_cfg(tmp_path, multiyear=True))
    single = _build_dataset(_cfg(tmp_path, multiyear=False))
    mid_multi = float(multi[(2, 4)]["diagnostic"].reshape(-1)[0])
    mid_single = float(single[(2, 4)]["diagnostic"].reshape(-1)[0])
    cross = float(multi[(10, 4)]["diagnostic"].reshape(-1)[0])
    assert mid_multi == pytest.approx(mid_single) == pytest.approx(2.0)
    assert cross == pytest.approx(10.0), "cross-year diagnostic left the start frame"


def test_the_forcing_pipeline_applies_through_the_multiyear_dataset(tmp_path):
    """Subset + fill + normalize must survive the routing change.

    The model lists 1 of the store's 2 varying channels, so a sample that comes
    back 2 wide means the pipeline was bypassed — which is how the store's CO2
    channel would silently become the model's sea ice.
    """
    ds = _build_dataset(_cfg(tmp_path, multiyear=True))
    assert getattr(ds, "forcing_pipeline", None) is not None
    sample = ds[(4, 4)]
    assert sample["varying_boundary"].shape[-3] == 1, sample["varying_boundary"].shape
    assert "calendar" in sample, "emit_calendar did not reach the multi-year dataset"
    assert torch.isfinite(sample["varying_boundary"]).all()
