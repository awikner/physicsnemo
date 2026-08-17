# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide a latitude row order from DATA, not from what a coordinate claims.

The raw per-timestep H5 archives this project converts carry no latitude
coordinate, so a converter that assumes one can be wrong without anything
failing. That is exactly how the AMIP stores ended up upside-down relative to
their labels (see ``docs/dev/context/lat-orientation-audit.md``). This module is
the shared source of truth for both the ingest-time assertion in the converters
and the standalone audit tool ``tools/data/check_lat_orientation.py``.

Every test here rests on a fact about the Earth that cannot be argued with:

``land_sea_mask``
    Antarctica is a solid land ring at 70-90S; the Arctic is ocean. Land
    fraction in the two polar bands differs by ~0.87 vs ~0.18.
``geopotential_at_surface`` / orography
    The Antarctic ice sheet is ~2 km high; the Arctic is at sea level
    (~18,700 vs ~1,450 m2/s2).
``toa_incident_solar_radiation`` / insolation
    Polar day and night are pure astronomy: at the June solstice the north pole
    is sunlit 24 h and the south pole is dark. Needs no external reference.
``2m_temperature`` / near-surface temperature
    The summer pole is the warm one (January vs July).

Anything asymmetric about the two hemispheres works; these four are the ones
present in the archives here. A test only votes when the two polar bands differ.
"""
from __future__ import annotations

import numpy as np

#: Recognized names for each anchor field, across the ERA5 / AMIP / E3SM /
#: PLASIM naming conventions in this project.
LAND_KEYS = ("land_sea_mask", "lsm", "PFTDATA_MASK", "land_binary_mask", "landmask")
OROG_KEYS = ("geopotential_at_surface", "sg", "TOPO", "orography")
INSOL_KEYS = (
    "toa_incident_solar_radiation",
    "DSWRFtoa",
    "rsdt",
    "sol_in",
    "DSWRFtoa_24h_lead",
    "DSWRFtoa_24h",
)
TEMP_KEYS = ("2m_temperature", "tas", "TREFHT", "skin_temperature", "ts", "2m_temperature_24h")

NORTH_FIRST = "N->S"
SOUTH_FIRST = "S->N"

#: Polar band half-width in degrees: |lat| > 90 - BAND_DEG.
BAND_DEG = 20.0


def band_rows(n_rows: int, band_deg: float = BAND_DEG) -> int:
    """Number of grid rows spanning ``band_deg`` degrees of latitude."""
    return max(1, int(round(band_deg / 180.0 * n_rows)))


def _vote(
    first: float, last: float, north_first_is: str, min_contrast: float = 0.0
) -> str | None:
    """Translate a (first band, last band) pair into a row-order vote.

    ``north_first_is`` says what the FIRST band would look like if row 0 were the
    north pole: ``"smaller"`` or ``"larger"``.

    ``min_contrast`` is the smallest |first - last| that counts as evidence.
    Abstaining below it matters because the real anchors are *emphatic* — an
    Antarctic land ring against an Arctic ocean, a 2 km ice sheet against sea
    level — so a near-tie is not a weak signal but an absent one, and answering
    anyway turns noise into a confident orientation. (Found 2026-08-17: a
    synthetic test archive with a random land mask voted ``S->N`` off nothing,
    and would have voted either way with a different seed.)
    """
    if not (np.isfinite(first) and np.isfinite(last)):
        return None
    if abs(first - last) <= min_contrast:
        return None
    first_smaller = first < last
    if north_first_is == "smaller":
        return NORTH_FIRST if first_smaller else SOUTH_FIRST
    return SOUTH_FIRST if first_smaller else NORTH_FIRST


#: Minimum polar-band land-fraction contrast for the land-mask test to vote.
#: The real contrast is huge — the 60-90S band is very nearly all land, the
#: 60-90N band mostly ocean — so anything under ~0.15 is not a faint anchor but
#: a missing one (a fractional mask, a masked-out band, or synthetic noise).
MIN_LAND_FRACTION_CONTRAST = 0.15

#: Minimum polar-band orography contrast, in metres of geopotential height. The
#: Antarctic plateau is ~2-3 km against an Arctic ocean at sea level, so 200 m
#: is a floor with two orders of magnitude of headroom. Fields in geopotential
#: (m^2/s^2) have an even larger contrast, so the same floor is conservative.
MIN_OROGRAPHY_CONTRAST = 200.0


def vote_land_mask(field: np.ndarray, band_deg: float = BAND_DEG) -> str | None:
    """Antarctica is land, the Arctic is ocean -> the land-heavy end is south."""
    nb = band_rows(field.shape[0], band_deg)
    return _vote(
        float(np.nanmean(field[:nb])),
        float(np.nanmean(field[-nb:])),
        "smaller",
        min_contrast=MIN_LAND_FRACTION_CONTRAST,
    )


def vote_orography(field: np.ndarray, band_deg: float = BAND_DEG) -> str | None:
    """The Antarctic ice sheet is high, the Arctic ocean is not."""
    nb = band_rows(field.shape[0], band_deg)
    return _vote(
        float(np.nanmean(field[:nb])),
        float(np.nanmean(field[-nb:])),
        "smaller",
        min_contrast=MIN_OROGRAPHY_CONTRAST,
    )


def vote_insolation(field: np.ndarray, month: int, band_deg: float = BAND_DEG) -> str | None:
    """Solstice insolation: in June the sunlit pole is north, in December south."""
    if month not in (6, 12):
        return None
    nb = band_rows(field.shape[0], band_deg)
    return _vote(
        float(np.nanmean(field[:nb])),
        float(np.nanmean(field[-nb:])),
        "larger" if month == 6 else "smaller",
    )


#: Minimum polar-band temperature contrast (K) for the seasonal test to vote.
#: In the seasonal MEAN the contrast is 20-45 K, but on a single day it can be a
#: few K and occasionally the wrong sign -- a warm Arctic intrusion in early
#: January, say. Below this the test abstains rather than outvoting the
#: unambiguous, time-invariant land-mask and orography anchors.
MIN_TEMP_CONTRAST_K = 8.0


def vote_temperature(
    field: np.ndarray,
    month: int,
    band_deg: float = BAND_DEG,
    min_contrast: float = MIN_TEMP_CONTRAST_K,
) -> str | None:
    """The summer pole is the warm one: July favours north, January south."""
    if month not in (1, 7):
        return None
    nb = band_rows(field.shape[0], band_deg)
    north = float(np.nanmean(field[:nb]))
    south = float(np.nanmean(field[-nb:]))
    if not (np.isfinite(north) and np.isfinite(south)):
        return None
    if abs(north - south) < min_contrast:
        return None  # within day-to-day noise; say nothing
    return _vote(north, south, "larger" if month == 7 else "smaller")


def combine(votes: dict[str, str | None]) -> tuple[str | None, dict[str, str]]:
    """Reduce named votes to one row order.

    Returns ``(order, cast)`` where ``order`` is ``"N->S"``/``"S->N"``, or ``None``
    when nothing voted or the votes disagree. Disagreement is deliberately not
    resolved by majority: these tests are independent and unambiguous, so a
    conflict means an assumption is wrong and a human should look.
    """
    cast = {k: v for k, v in votes.items() if v is not None}
    if not cast:
        return None, cast
    distinct = set(cast.values())
    if len(distinct) != 1:
        return None, cast
    return distinct.pop(), cast


def order_of_coord(lat: np.ndarray) -> str:
    """Row order that a latitude COORDINATE claims (says nothing about the data)."""
    lat = np.asarray(lat)
    return NORTH_FIRST if float(lat[0]) > float(lat[-1]) else SOUTH_FIRST


def infer_h5_row_order(
    group,
    *,
    month: int | None = None,
    band_deg: float = BAND_DEG,
) -> tuple[str | None, dict[str, str]]:
    """Infer the row order of one raw H5 sample's ``input`` group.

    ``group`` is an open ``h5py`` group. ``month`` (1-12), when given, unlocks the
    seasonal tests for that sample; the land-mask and orography tests need no date.
    Returns ``(order, votes_cast)``.
    """
    votes: dict[str, str | None] = {}

    def first_present(names):
        return next((n for n in names if n in group), None)

    k = first_present(LAND_KEYS)
    if k is not None:
        votes[f"land_mask:{k}"] = vote_land_mask(
            np.asarray(group[k][:], dtype="float64"), band_deg
        )
    k = first_present(OROG_KEYS)
    if k is not None:
        votes[f"orography:{k}"] = vote_orography(
            np.asarray(group[k][:], dtype="float64"), band_deg
        )
    if month is not None:
        k = first_present(INSOL_KEYS)
        if k is not None:
            votes[f"insolation:{k}"] = vote_insolation(
                np.asarray(group[k][:], dtype="float64"), month, band_deg
            )
        k = first_present(TEMP_KEYS)
        if k is not None:
            votes[f"temperature:{k}"] = vote_temperature(
                np.asarray(group[k][:], dtype="float64"), month, band_deg
            )
    return combine(votes)


def assert_coord_matches_data(
    lat: np.ndarray,
    data_row_order: str | None,
    *,
    votes: dict[str, str] | None = None,
    context: str = "",
    strict: bool = True,
) -> str:
    """Check a lat coordinate against the row order the DATA actually has.

    Returns a human-readable summary. Raises ``ValueError`` when they disagree and
    ``strict``; when ``data_row_order`` is ``None`` nothing could be decided, which
    is reported but never fatal (some grids have no usable anchor).
    """
    claimed = order_of_coord(lat)
    where = f" [{context}]" if context else ""
    cast = votes or {}
    if data_row_order is None:
        return (
            f"lat orientation UNVERIFIED{where}: coordinate claims {claimed}, "
            f"no anchor field voted (votes={cast})"
        )
    if data_row_order != claimed:
        msg = (
            f"lat orientation MISMATCH{where}: the coordinate says {claimed} but the "
            f"data rows are {data_row_order} (votes={cast}). Writing this store would "
            f"put every field upside-down relative to its label. Either reverse the "
            f"data on ingest or declare the lat coordinate in {data_row_order} order."
        )
        if strict:
            raise ValueError(msg)
        return msg
    return f"lat orientation OK{where}: coordinate and data rows both {claimed} (votes={cast})"
