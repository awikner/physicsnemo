#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/data/check_lat_orientation.py.

Synthesizes tiny ClimateZarr-like stores on a coarse global grid with physically
sensible fields (Antarctic land ring, Antarctic ice-sheet geopotential, analytic
solstice insolation, a seasonal temperature cycle, and a humidity field sharing
the same continental structure), then checks that the auditor calls each planted
defect correctly:

* a correct N->S store                      -> every array OK, exit 0
* the same store with all data reversed     -> FLIPPED, exit 1
* one single array reversed                 -> only that array FLIPPED
* one array reversed over a RANGE of steps  -> MIXED-IN-TIME (the half-repair case)
* a correct S->N-LABELLED store             -> OK, not "flipped" (the E3SM trap)

Runnable directly (``python test_check_lat_orientation.py``) or via pytest. Uses
only numpy / xarray / zarr / cftime -- no physicsnemo import, no cluster data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

# Moved into the collected test tree 2026-08-17: at tools/data/ it was never
# run by `pytest test`, so the auditor it covers was effectively untested in
# every full-suite run. The import target is three levels up.
_TOOLS_DATA = Path(__file__).resolve().parents[3] / "tools" / "data"
sys.path.insert(0, str(_TOOLS_DATA))
import check_lat_orientation as clo  # noqa: E402

N_LAT, N_LON, N_TIME = 45, 90, 365
LEVELS = [850.0, 500.0, 200.0]


def _grid(descending: bool = True):
    """Coarse 4-degree global grid; lat N->S by default."""
    lat = np.linspace(88.0, -88.0, N_LAT)
    if not descending:
        lat = lat[::-1]
    lon = np.linspace(0.0, 356.0, N_LON)
    return lat.astype("float32"), lon.astype("float32")


def _continents(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A fixed, hemisphere-asymmetric pattern standing in for continents.

    Cross-array and temporal checks need real longitudinal structure, and the
    lat-reversal test needs that structure to be ASYMMETRIC about the equator --
    which on Earth it emphatically is. Every lobe below is a Gaussian centred at
    a latitude with no mirror-image partner, so reversing lat genuinely changes
    the pattern. (A term like ``cos(lat) * sin(3*lon)`` would be latitude-even and
    survive a flip almost unchanged, which is exactly what makes a symmetric
    fixture useless for this test.)
    """
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    return (
        0.9 * np.exp(-((LAT - 45.0) ** 2) / 400.0) * np.cos(np.deg2rad(2 * LON))
        + 0.7 * np.exp(-((LAT - 8.0) ** 2) / 200.0) * np.sin(np.deg2rad(3 * LON))
        - 0.5 * np.exp(-((LAT + 32.0) ** 2) / 300.0) * np.cos(np.deg2rad(LON))
    )


def _build(path: Path, *, descending: bool = True) -> None:
    """Write a physically self-consistent store whose data matches its labels."""
    lat, lon = _grid(descending)
    LAT, _LON = np.meshgrid(lat, lon, indexing="ij")
    cont = _continents(lat, lon)

    # Antarctica: solid land and a high ice sheet. The Arctic: ocean at sea level.
    lsm = np.where(LAT < -70.0, 1.0, 0.05 * np.ones_like(LAT))
    lsm = np.where((LAT > 20.0) & (LAT < 60.0) & (_LON < 120.0), 1.0, lsm)
    orog = np.where(LAT < -70.0, 20000.0, 200.0) + 800.0 * np.clip(cont, 0, None)

    times = [
        cftime.num2date(i, "days since 2001-01-01", calendar="standard")
        for i in range(N_TIME)
    ]
    doy = np.arange(N_TIME) + 1
    decl = 23.44 * np.sin(2 * np.pi * (doy - 80.0) / 365.0)  # solar declination

    insol = np.empty((N_TIME, N_LAT, N_LON), "float32")
    t2m = np.empty((N_TIME, N_LAT, N_LON), "float32")
    q = np.empty((N_TIME, N_LAT, N_LON), "float32")
    ta = np.empty((N_TIME, len(LEVELS), N_LAT, N_LON), "float32")
    for i in range(N_TIME):
        # Analytic daily-mean insolation: zero through the winter polar night.
        insol[i] = 500.0 * np.clip(np.cos(np.deg2rad(LAT - decl[i])), 0.0, None)
        base = (
            280.0
            - 45.0 * np.abs(LAT) / 90.0
            + 25.0 * np.sin(np.deg2rad(decl[i])) * np.sin(np.deg2rad(LAT))
            + 4.0 * cont
        )
        t2m[i] = base
        q[i] = 0.01 * np.exp((base - 250.0) / 40.0) + 0.002 * cont
        for li, lev in enumerate(LEVELS):
            ta[i, li] = base - 0.02 * (1000.0 - lev)

    ds = xr.Dataset(
        {
            "land_sea_mask": (("lat", "lon"), lsm.astype("float32")),
            "geopotential_at_surface": (("lat", "lon"), orog.astype("float32")),
            "toa_incident_solar_radiation": (("time", "lat", "lon"), insol),
            "2m_temperature": (("time", "lat", "lon"), t2m),
            "specific_humidity_2m": (("time", "lat", "lon"), q),
            "temperature": (("time", "pressure_level", "lat", "lon"), ta),
        },
        coords={
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "time": ("time", times),
            "pressure_level": ("pressure_level", np.array(LEVELS, "float32")),
        },
    )
    ds.to_zarr(path, mode="w", consolidated=True)


def _reverse_lat(path: Path, arrays: list[str], t_slice: slice | None = None) -> None:
    """Reverse arrays along lat IN PLACE, leaving the lat coordinate alone."""
    import zarr

    g = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    for name in arrays:
        arr = g[name]
        axis = arr.ndim - 2  # (..., lat, lon)
        if arr.ndim == 2 or t_slice is None:
            arr[:] = np.flip(np.asarray(arr[:]), axis=axis)
        else:
            block = np.asarray(arr[t_slice])
            arr[t_slice] = np.flip(block, axis=axis)
    zarr.consolidate_metadata(g.store)


def _verdicts(path: Path) -> dict[str, str]:
    rep = clo.audit_store(str(path), band=70.0)
    assert "error" not in rep, rep.get("error")
    return {k: v["verdict"] for k, v in rep["arrays"].items()}


# --------------------------------------------------------------------------- #
def test_correct_store_is_clean(tmp_path):
    p = tmp_path / "good.zarr"
    _build(p)
    v = _verdicts(p)
    assert v["land_sea_mask"] == "OK", v
    assert v["geopotential_at_surface"] == "OK", v
    assert v["toa_incident_solar_radiation"] == "OK", v
    assert v["2m_temperature"] == "OK", v
    assert all(x in ("OK", "UNKNOWN") for x in v.values()), v
    assert clo.main(["--stores", str(p)]) == 0


def test_upside_down_store_is_caught(tmp_path):
    p = tmp_path / "flipped.zarr"
    _build(p)
    _reverse_lat(p, ["land_sea_mask", "geopotential_at_surface",
                     "toa_incident_solar_radiation", "2m_temperature",
                     "specific_humidity_2m", "temperature"])
    v = _verdicts(p)
    for name in ("land_sea_mask", "geopotential_at_surface",
                 "toa_incident_solar_radiation", "2m_temperature"):
        assert v[name] == "FLIPPED", (name, v)
    assert clo.main(["--stores", str(p)]) == 1


def test_single_reversed_array_is_isolated(tmp_path):
    """The 1987-SST case: one array upside-down in an otherwise correct store."""
    p = tmp_path / "one_bad.zarr"
    _build(p)
    _reverse_lat(p, ["specific_humidity_2m"])
    v = _verdicts(p)
    assert v["specific_humidity_2m"] == "FLIPPED", v
    assert v["2m_temperature"] == "OK", v
    assert v["land_sea_mask"] == "OK", v


def test_partially_reversed_array_reports_a_seam(tmp_path):
    """The 1988/1989 case: an array left half-flipped by an interrupted repair."""
    p = tmp_path / "half_bad.zarr"
    _build(p)
    _reverse_lat(p, ["2m_temperature"], t_slice=slice(180, N_TIME))
    v = _verdicts(p)
    assert v["2m_temperature"] == "MIXED-IN-TIME", v


def test_ascending_labels_with_matching_data_are_ok(tmp_path):
    """The E3SM trap: S->N labels are fine when the data really is S->N."""
    p = tmp_path / "ascending.zarr"
    _build(p, descending=False)
    rep = clo.audit_store(str(p), band=70.0)
    assert rep["lat_order"] == "S->N", rep["lat_order"]
    v = {k: x["verdict"] for k, x in rep["arrays"].items()}
    assert v["land_sea_mask"] == "OK", v
    assert v["toa_incident_solar_radiation"] == "OK", v
    assert v["2m_temperature"] == "OK", v


def test_subset_store_does_not_invent_a_solstice(tmp_path):
    """A store covering only part of the year (an AMIP quarter) has no solstice.

    Without a tolerance on the date match, "nearest to 21 Jun" and "nearest to
    21 Dec" both land on the same edge timestep and return opposite verdicts for
    identical data, which reads as a defect in a perfectly good store.
    """
    import zarr

    p = tmp_path / "q1.zarr"
    _build(p)
    full = zarr.open_group(str(p), mode="r")
    ds = xr.open_zarr(str(p), consolidated=None, decode_timedelta=False)
    q1 = ds.isel(time=slice(0, 91))  # January through late March only
    q1_path = tmp_path / "quarter.zarr"
    q1.to_zarr(q1_path, mode="w", consolidated=True)
    assert full["lat"][0] > full["lat"][-1]

    rep = clo.audit_store(str(q1_path), band=70.0, anchors_only=True)
    seasonal = [e for e in rep["parts"]["A"]
                if e["test"].startswith(("insolation", "temperature"))]
    assert all(e["verdict"] != "FLIPPED" for e in seasonal), seasonal
    v = {k: x["verdict"] for k, x in rep["arrays"].items()}
    assert v["land_sea_mask"] == "OK", v
    assert rep["store_verdict"] in ("OK", "OK-WITH-UNKNOWNS"), rep["store_verdict"]


def test_anchors_only_mode_screens_whole_stores(tmp_path):
    p = tmp_path / "anchors.zarr"
    _build(p)
    _reverse_lat(p, ["land_sea_mask", "geopotential_at_surface",
                     "toa_incident_solar_radiation", "2m_temperature"])
    rep = clo.audit_store(str(p), band=70.0, anchors_only=True)
    v = {k: x["verdict"] for k, x in rep["arrays"].items()}
    assert v["land_sea_mask"] == "FLIPPED", v
    assert rep["parts"]["B"] == [] and rep["parts"]["C"] == []


if __name__ == "__main__":
    import tempfile

    failures = 0
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
                print(f"PASS {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("all passed" if not failures else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)


# ---------------------------------------------------------------------------
# Anchor margins (2026-08-17). The land-mask and orography votes had no
# minimum contrast, so a near-tie produced a confident answer — which is how a
# synthetic archive with a RANDOM land mask voted `S->N` off nothing and failed
# an unrelated converter test. The real anchors are emphatic; a near-tie is an
# absent signal, not a weak one.
# ---------------------------------------------------------------------------


def _lat_module():
    sys.path.insert(0, str(_TOOLS_DATA))
    from _common import lat_orientation

    return lat_orientation


@pytest.mark.parametrize("nrows", [45, 180])
def test_a_random_land_mask_abstains_on_a_real_grid(nrows):
    """At real resolutions the floor is far above what noise can reach.

    Measured: over a polar band of 5x90 (45 rows) or 20x360 (180 rows) cells,
    random noise tops out at 0.061 and 0.015 contrast respectively, against a
    0.15 floor and a real land contrast of 0.4-0.7. Under the old thresholdless
    vote every one of these returned a confident order.
    """
    lo = _lat_module()
    rng = np.random.default_rng(0)
    votes = [
        lo.vote_land_mask(rng.random((nrows, nrows * 2)).astype("float32"))
        for _ in range(20)
    ]
    assert set(votes) == {None}, votes


def test_the_margin_does_not_rescue_a_toy_grid():
    """Documents the limit, so nobody trusts the floor where it cannot help.

    On an 8-row grid the 20-degree band is a SINGLE row of 16 cells, and a
    16-sample mean difference clears 0.15 about 16% of the time. No threshold
    fixes that without also rejecting real anchors — which is why the converter's
    synthetic test archive plants an actual Antarctic land ring rather than
    relying on this margin (test_amip_h5_to_zarr_extras.py).
    """
    lo = _lat_module()
    rng = np.random.default_rng(0)
    votes = [
        lo.vote_land_mask(rng.random((8, 16)).astype("float32")) for _ in range(200)
    ]
    assert None in votes and len(set(votes) - {None}) > 0, (
        "expected a toy grid to be decidable-by-accident sometimes; if this now "
        "always abstains the floor may have been raised past real anchors"
    )


def test_a_real_antarctic_land_ring_still_votes():
    lo = _lat_module()
    field = np.zeros((8, 16), dtype="float32")
    field[0] = 1.0                     # row 0 = South Pole, all land
    assert lo.vote_land_mask(field) == lo.SOUTH_FIRST
    assert lo.vote_land_mask(field[::-1].copy()) == lo.NORTH_FIRST


def test_orography_needs_more_than_a_ripple():
    lo = _lat_module()
    flat = np.full((8, 16), 500.0, dtype="float32")
    flat[0] += 50.0                    # 50 m: below the 200 m floor
    assert lo.vote_orography(flat) is None
    ice_sheet = np.zeros((8, 16), dtype="float32")
    ice_sheet[0] = 2500.0              # Antarctic plateau
    assert lo.vote_orography(ice_sheet) == lo.SOUTH_FIRST


def test_an_exact_tie_still_abstains():
    """The pre-existing behaviour, kept: equality is not evidence."""
    lo = _lat_module()
    assert lo.vote_land_mask(np.full((8, 16), 0.5, dtype="float32")) is None
