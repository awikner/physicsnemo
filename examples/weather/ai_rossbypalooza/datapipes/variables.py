# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical variable identity for the MoWE hindcast mixture datapipe.

Every expert archive names variables differently: the DSI stores use flat
level-baked names (``z_500``, ``2t``, ``tp``) while the consolidated
pangu/sfno stores use canonical ERA5 long names with a 3-D
``pressure_level`` coordinate.  The invariant this module enforces is that
*every* native name — from either schema — is normalized to a canonical
``(variable, level_hPa)`` key **before** any channel lookup, so the same
physical variable from different models always occupies the same master
channel slot (and receives the same normalization statistics).

The master channel layout is fixed by config order; channel 0 of every
assembled expert block is always daily precipitation in mm/day and is not
part of the configurable list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# --------------------------------------------------------------------------- #
# Alias tables: canonical ERA5 name -> accepted aliases (matched lowercase).
# Copied from tools/data/hindcast/consolidate_hindcasts.py (examples/ must not
# import tools/); keep the two in sync when adding aliases.
# --------------------------------------------------------------------------- #
SURFACE_ALIASES: dict[str, list[str]] = {
    "2m_temperature": ["2m_temperature", "t2m", "2t", "tas", "temperature_2m", "air_temperature_2m"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10", "10u", "uas", "10m_u_wind"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10", "10v", "vas", "10m_v_wind"],
    "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl", "mslp", "air_pressure_at_mean_sea_level"],
    "surface_pressure": ["surface_pressure", "sp", "ps", "surface_air_pressure", "pres"],
}
DIAGNOSTIC_ALIASES: dict[str, list[str]] = {
    "total_precipitation_24hr": [
        "total_precipitation_24hr", "total_precipitation", "tp", "precip", "pr", "precipitation",
    ],
    "mean_top_net_long_wave_radiation_flux": [
        "mean_top_net_long_wave_radiation_flux", "mtnlwrf", "ttr", "olr",
        "top_net_long_wave_radiation", "mean_top_net_lw_radiation_flux",
    ],
}
UPPER_AIR_ALIASES: dict[str, list[str]] = {
    "temperature": ["temperature", "t", "ta", "air_temperature"],
    "u_component_of_wind": ["u_component_of_wind", "u", "ua", "eastward_wind"],
    "v_component_of_wind": ["v_component_of_wind", "v", "va", "northward_wind"],
    "specific_humidity": ["specific_humidity", "q", "hus", "humidity"],
    "geopotential": ["geopotential", "z", "zg", "gh", "geopotential_height"],
}


def _invert(aliases: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for canon, names in aliases.items():
        for n in names:
            out[n.lower()] = canon
    return out


SURFACE_LOOKUP = _invert(SURFACE_ALIASES)
DIAGNOSTIC_LOOKUP = _invert(DIAGNOSTIC_ALIASES)
UPPER_AIR_LOOKUP = _invert(UPPER_AIR_ALIASES)
# Scalar (level-free) names first so e.g. "t2m" never parses as upper-air.
SCALAR_LOOKUP = {**SURFACE_LOOKUP, **DIAGNOSTIC_LOOKUP}

#: Relative/absolute tolerance for matching pressure levels by value —
#: mirrors ClimateNormalizer._nearest_indices (transforms.py).
LEVEL_TOL = 1e-3


def canonicalize_scalar(name: str) -> Optional[str]:
    """Canonical name for a level-free (surface/diagnostic) variable, else None."""
    return SCALAR_LOOKUP.get(name.lower())


def canonicalize_upper(name: str) -> Optional[str]:
    """Canonical name for an upper-air variable, else None."""
    return UPPER_AIR_LOOKUP.get(name.lower())


def _normalize_level(value: float) -> float:
    """Levels are carried in hPa; values > 2000 are assumed Pa and converted."""
    return value / 100.0 if value > 2000.0 else value


def parse_flat_name(name: str) -> Optional[tuple[str, Optional[float]]]:
    """Parse a flat (possibly level-baked) native name to a canonical key.

    ``"z_500"`` -> ``("geopotential", 500.0)``; ``"2t"`` ->
    ``("2m_temperature", None)``; ``"tp"`` ->
    ``("total_precipitation_24hr", None)``.  Names that resolve to nothing
    (e.g. ``"swvl1"``) return None and are ignored by callers.
    """
    scalar = canonicalize_scalar(name)
    if scalar is not None:
        return scalar, None
    head, sep, tail = name.rpartition("_")
    if sep:
        try:
            level = float(tail)
        except ValueError:
            level = None
        if level is not None:
            upper = canonicalize_upper(head)
            if upper is not None:
                return upper, _normalize_level(level)
    upper = canonicalize_upper(name)
    if upper is not None:
        # Upper-air name with no level baked in (3-D variable) — level comes
        # from the store's pressure_level coordinate, not the name.
        return upper, None
    return None


def levels_match(a: float, b: float) -> bool:
    """By-value level comparison with the ClimateNormalizer tolerance rule."""
    tol = max(LEVEL_TOL, LEVEL_TOL * abs(b))
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


@dataclass(frozen=True)
class Channel:
    """One master-layout slot: a canonical variable, optionally at a level."""

    canonical: str
    level_hpa: Optional[float] = None

    @property
    def name(self) -> str:
        if self.level_hpa is None:
            return self.canonical
        return f"{self.canonical}/{self.level_hpa:g}"


#: Index of the daily-precip channel in every assembled expert block.
PRECIP_INDEX = 0
PRECIP_CHANNEL_NAME = "precip_mm_day"


class ChannelLayout:
    """Master canonical channel list shared by all experts.

    ``master`` entries are ``"<name>[/<level_hPa>]"`` where ``<name>`` may be
    a canonical ERA5 name or any known alias (``"z/500"`` == ``"geopotential/500"``).
    Precipitation is *not* listed here; it is always channel ``PRECIP_INDEX``
    of the assembled ``(1 + C, H, W)`` block.
    """

    def __init__(self, master: Sequence[str]) -> None:
        channels: list[Channel] = []
        for entry in master:
            head, sep, tail = str(entry).partition("/")
            head = head.strip()
            if sep:
                canonical = canonicalize_upper(head)
                if canonical is None:
                    raise ValueError(
                        f"master channel '{entry}': '{head}' is not a known "
                        f"upper-air variable (known: {sorted(UPPER_AIR_ALIASES)})"
                    )
                channel = Channel(canonical, _normalize_level(float(tail)))
            else:
                canonical = canonicalize_scalar(head)
                if canonical is None:
                    raise ValueError(
                        f"master channel '{entry}': '{head}' is not a known "
                        f"surface/diagnostic variable "
                        f"(known: {sorted(SCALAR_LOOKUP.values())})"
                    )
                channel = Channel(canonical, None)
            if channel in channels:
                raise ValueError(f"duplicate master channel '{channel.name}'")
            channels.append(channel)
        self.channels: tuple[Channel, ...] = tuple(channels)

    @property
    def num_channels(self) -> int:
        """Total assembled channels per expert, including precip at index 0."""
        return 1 + len(self.channels)

    @property
    def channel_names(self) -> list[str]:
        return [PRECIP_CHANNEL_NAME] + [c.name for c in self.channels]

    def index_of(self, canonical: str, level_hpa: Optional[float]) -> Optional[int]:
        """Index in the full ``(1 + C)`` layout, or None if not a master channel."""
        for i, c in enumerate(self.channels):
            if c.canonical != canonical:
                continue
            if c.level_hpa is None and level_hpa is None:
                return 1 + i
            if (
                c.level_hpa is not None
                and level_hpa is not None
                and levels_match(level_hpa, c.level_hpa)
            ):
                return 1 + i
        return None

    def upper_air_levels(self, canonical: str) -> list[float]:
        """All master levels requested for one upper-air variable."""
        return [
            c.level_hpa
            for c in self.channels
            if c.canonical == canonical and c.level_hpa is not None
        ]
