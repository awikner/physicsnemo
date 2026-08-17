# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Per-year boundary pairing and the cross-year seam (2026-08-17).

Two defects blocked training a diffusion model over a multi-year archive, both
found by driving the loader rather than reading it:

* ``boundary_zarr_path`` was passed VERBATIM to every per-year sub-dataset, so a
  directory of per-year boundary stores — the AMIP pairing, coarse state plus a
  1-degree boundary archive — died with ``GroupNotFoundError``. Pointing it at one
  year's store instead would have fed every year that year's SST. Boundary reads
  are indexed by day-of-year, so each state year needs the boundary store for the
  SAME calendar year, which is what pairing by file name gives.
* the cross-year branch indexed ``target_sample["upper_air_in"]``
  unconditionally, where the single-year path guards it and also carries the
  sigma/pressure variants. A surface-only store raised ``KeyError`` on any pair
  straddling a year boundary, and a mixed sigma/pressure store silently lost its
  targets there.

Fixtures use REAL distinct calendar years on purpose: two stores inside the same
January look like two years to the index map but not to the day-of-year boundary
lookup, which is how the first version of this test fooled itself.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from physicsnemo.experimental.datapipes.climate import ClimateZarrMultiYearDataset

_H, _W, _NT = 2, 4, 8          # 8 six-hourly rows = the first 2 days of a year


def _write(path: Path, year: int, tag: float, *, boundary: bool, upper: bool = True,
           self_contained: bool = False):
    times = [
        cftime.DatetimeGregorian(year, 1, 1 + i // 4, 6 * (i % 4)) for i in range(_NT)
    ]
    vals = (np.full(_NT, tag)[:, None, None] * np.ones((_NT, _H, _W))).astype("float32")
    attrs = {
        "data_timedelta_hours": 6,
        "constant_boundary_variables": ["land_sea_mask"],
        "varying_boundary_variables": ["sea_surface_temperature_monthly_interp"],
        "climate_zarr_schema_version": 1,
    }
    data = {"land_sea_mask": (("lat", "lon"), np.zeros((_H, _W), "float32"))}
    if boundary:
        data["sea_surface_temperature_monthly_interp"] = (
            ("time", "lat", "lon"), vals
        )
        attrs["surface_variables"] = []
    else:
        data["skin_temperature"] = (("time", "lat", "lon"), vals)
        attrs["surface_variables"] = ["skin_temperature"]
        if self_contained:
            # No separate boundary store: the state store must actually carry the
            # varying variable its layout declares (as the real coarse store does).
            data["sea_surface_temperature_monthly_interp"] = (
                ("time", "lat", "lon"), vals
            )
        if upper:
            data["temperature"] = (
                ("time", "pressure_level", "lat", "lon"),
                vals[:, None].repeat(2, axis=1),
            )
            attrs["pressure_upper_air_variables"] = ["temperature"]
    coords = {
        "time": ("time", times),
        "lat": ("lat", np.linspace(-45, 45, _H).astype("float32")),
        "lon": ("lon", np.linspace(0, 270, _W).astype("float32")),
    }
    if not boundary and upper:
        coords["pressure_level"] = ("pressure_level", np.array([500.0, 850.0], "float32"))
    xr.Dataset(data, coords=coords, attrs=attrs).to_zarr(
        path, mode="w", consolidated=True
    )


def _archive(tmp_path: Path, *, with_boundary_dir: bool, upper: bool = True):
    state, bnd = tmp_path / "state", tmp_path / "bnd"
    state.mkdir(exist_ok=True)
    bnd.mkdir(exist_ok=True)
    # State tagged 10/20 per year, boundary 100/200, so each read reveals which
    # store and which year it came from.
    for year, stag, btag in ((1981, 10.0, 100.0), (1982, 20.0, 200.0)):
        _write(state / f"{year}.zarr", year, stag, boundary=False, upper=upper)
        _write(bnd / f"{year}.zarr", year, btag, boundary=True)
    return state, (bnd if with_boundary_dir else None)


def test_a_directory_of_boundary_stores_pairs_by_year(tmp_path):
    state, bnd = _archive(tmp_path, with_boundary_dir=True)
    ds = ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd)
    assert ds.n_time == 2 * _NT
    # Each year reads ITS OWN boundary store, not the first one for everything.
    for global_idx, want_state, want_bnd in (
        (0, 10.0, 100.0), (5, 10.0, 100.0), (8, 20.0, 200.0), (11, 20.0, 200.0)
    ):
        s = ds[(global_idx, 1)]
        assert float(s["surface_in"].reshape(-1)[0]) == pytest.approx(want_state)
        assert float(s["varying_boundary"].reshape(-1)[0]) == pytest.approx(want_bnd)


def test_a_missing_boundary_year_raises_rather_than_substituting(tmp_path):
    state, bnd = _archive(tmp_path, with_boundary_dir=True)
    (bnd / "1982.zarr").rename(bnd / "1999.zarr")
    with pytest.raises(ValueError, match="no boundary store for"):
        ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd)


def test_an_empty_boundary_directory_is_refused(tmp_path):
    state, bnd = _archive(tmp_path, with_boundary_dir=True)
    for p in bnd.glob("*.zarr"):
        p.rename(p.with_suffix(".notzarr"))
    with pytest.raises(ValueError, match="no \\*.zarr sub-stores"):
        ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd)


def test_a_single_boundary_store_still_serves_every_year(tmp_path):
    """Allowed for a genuine climatology; the caller asserts it, not us."""
    state, bnd = _archive(tmp_path, with_boundary_dir=True)
    ds = ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd / "1981.zarr")
    assert float(ds[(0, 1)]["varying_boundary"].reshape(-1)[0]) == pytest.approx(100.0)
    # 1982's rows read 1981's boundary — the documented consequence.
    assert float(ds[(8, 1)]["varying_boundary"].reshape(-1)[0]) == pytest.approx(100.0)


def test_no_boundary_store_is_unaffected(tmp_path):
    state = tmp_path / "self_contained"
    state.mkdir()
    for year, tag in ((1981, 10.0), (1982, 20.0)):
        _write(state / f"{year}.zarr", year, tag, boundary=False, self_contained=True)
    ds = ClimateZarrMultiYearDataset(state)
    assert ds.n_time == 2 * _NT
    assert float(ds[(0, 1)]["surface_in"].reshape(-1)[0]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# The cross-year seam
# ---------------------------------------------------------------------------


def test_a_cross_year_pair_reads_its_target_from_the_next_year(tmp_path):
    state, bnd = _archive(tmp_path, with_boundary_dir=True)
    ds = ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd)
    s = ds[(7, 1)]                     # last row of 1981 -> first row of 1982
    assert float(s["surface_in"].reshape(-1)[0]) == pytest.approx(10.0)
    assert float(s["target_surface"].reshape(-1)[0]) == pytest.approx(20.0)
    assert float(s["target_upper_air"].reshape(-1)[0]) == pytest.approx(20.0)
    # Conditioning boundary stays the START year's, as for any other pair.
    assert float(s["varying_boundary"].reshape(-1)[0]) == pytest.approx(100.0)


def test_a_surface_only_store_survives_the_seam(tmp_path):
    """No upper air at all: the cross-year branch used to raise KeyError here."""
    state, bnd = _archive(tmp_path, with_boundary_dir=True, upper=False)
    ds = ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd)
    s = ds[(7, 1)]
    assert float(s["target_surface"].reshape(-1)[0]) == pytest.approx(20.0)
    assert "target_upper_air" not in s


def test_the_seam_and_mid_year_agree_on_which_keys_appear(tmp_path):
    """A pair at the boundary must look like every other pair to the pack step."""
    state, bnd = _archive(tmp_path, with_boundary_dir=True)
    ds = ClimateZarrMultiYearDataset(state, boundary_zarr_path=bnd)
    mid = set(ds[(2, 1)].keys())
    seam = set(ds[(7, 1)].keys())
    assert mid == seam, mid.symmetric_difference(seam)
