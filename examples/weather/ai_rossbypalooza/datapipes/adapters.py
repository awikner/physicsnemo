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

"""Per-schema expert adapters for the hindcast mixture datapipe.

An adapter knows one archive schema. Cold path (main process, at dataset
init): ``discover()`` reads coords + attrs from every yearly store and builds
the init lookup, lead-index maps, and the native-name -> master-channel
mapping. Hot path (DataLoader worker): ``plan()`` emits ``ReadRequest``s the
dataset resolves against its per-worker zarr handle cache, and ``assemble()``
turns the fetched arrays into one ``(1 + C, H, W)`` float32 block in physical
units, canonical channel order, channel 0 = daily precip in mm/day, missing
channels 0.

Two implementations:

* :class:`SchemaAAdapter` — DSI stores (``dsi_hindcast_to_formats.py``
  Format 2), *pre-regridded to 1 degree* by ``tools/regrid_dsi_to_1deg.py``:
  flat level-baked names on two lead axes
  ``(init_time, prediction_timedelta[h] | prediction_timedelta_daily[d], lat, lon)``.
* :class:`SchemaBAdapter` — consolidated stores
  (``consolidate_hindcasts.py``): canonical ERA5 names,
  ``(init_time, lead_time[day index, 0 = IC], [pressure_level,] lat, lon)``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from .precip import PrecipSpec
from .regrid import grids_equal
from .variables import (
    ChannelLayout,
    canonicalize_scalar,
    canonicalize_upper,
    levels_match,
    parse_flat_name,
)

logger = logging.getLogger(__name__)

InitKey = tuple[int, int, int, int]  # (year, month, day, hour) at init


@dataclass(frozen=True)
class ReadRequest:
    """One zarr array read: resolved by the dataset's per-worker handle cache.

    ``array_key = (owner_name, store_year, var_name)``; ``index`` is a
    plain numpy-style index tuple applied to the array.
    """

    array_key: tuple[str, int, str]
    index: tuple


def _open_meta(store: Path) -> xr.Dataset:
    """Open a store for coords/attrs only (cftime-decoded init times)."""
    return xr.open_zarr(
        store,
        consolidated=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    )


def _init_keys(times: np.ndarray) -> list[InitKey]:
    return [
        (int(t.year), int(t.month), int(t.day), int(t.hour)) for t in times
    ]


class ExpertAdapter(ABC):
    """Common interface over one expert's yearly zarr archive."""

    def __init__(
        self,
        name: str,
        root: str | Path,
        layout: ChannelLayout,
        precip: PrecipSpec,
        *,
        exclude_variables: Sequence[str] = (),
    ) -> None:
        self.name = name
        self.root = Path(root)
        self.layout = layout
        self.precip = precip
        self.exclude_variables = {v.lower() for v in exclude_variables}
        self._discovered = False
        self._init_lookup: dict[InitKey, tuple[int, int]] = {}
        self._years: list[int] = []
        # var -> master channel index (dynamical predictors, fixed order)
        self._dyn_channels: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # cold path
    # ------------------------------------------------------------------ #
    def discover(
        self,
        expected_lat: Optional[np.ndarray] = None,
        expected_lon: Optional[np.ndarray] = None,
    ) -> None:
        stores = sorted(self.root.glob("*.zarr"))
        if not stores:
            raise ValueError(
                f"expert '{self.name}': no *.zarr stores under {self.root}"
            )
        for store in stores:
            try:
                year = int(store.stem)
            except ValueError:
                logger.debug("%s: skipping non-year store %s", self.name, store)
                continue
            ds = _open_meta(store)
            try:
                if expected_lat is not None and not (
                    grids_equal(ds["lat"].values, expected_lat)
                    and grids_equal(ds["lon"].values, expected_lon)
                ):
                    raise ValueError(self._grid_error(store, ds))
                self._discover_store(year, ds)
                for local_idx, key in enumerate(
                    _init_keys(ds["init_time"].values)
                ):
                    if key in self._init_lookup:
                        raise ValueError(
                            f"expert '{self.name}': duplicate init {key} "
                            f"in {store}"
                        )
                    self._init_lookup[key] = (year, local_idx)
            finally:
                ds.close()
            self._years.append(year)
        self._discovered = True
        self._validate_discovery()

    def init_lookup(self) -> dict[InitKey, tuple[int, int]]:
        self._require_discovered()
        return self._init_lookup

    @property
    def channel_mask(self) -> np.ndarray:
        """(1 + C,) bool: which master channels this expert supplies."""
        self._require_discovered()
        mask = np.zeros(self.layout.num_channels, dtype=bool)
        mask[0] = True  # precip is mandatory (validated at discover)
        for idx in self._dyn_channels.values():
            mask[idx] = True
        return mask

    def _require_discovered(self) -> None:
        if not self._discovered:
            raise RuntimeError(f"expert '{self.name}': call discover() first")

    def _grid_error(self, store: Path, ds: xr.Dataset) -> str:
        return (
            f"expert '{self.name}': {store} grid "
            f"({ds.sizes['lat']}x{ds.sizes['lon']}) does not match the "
            f"target 1-degree grid"
        )

    # ------------------------------------------------------------------ #
    # schema-specific hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _discover_store(self, year: int, ds: xr.Dataset) -> None: ...

    @abstractmethod
    def _validate_discovery(self) -> None: ...

    @abstractmethod
    def lead_supported(self, tau_days: int) -> bool: ...

    @abstractmethod
    def plan(self, year: int, init_idx: int, tau_days: int) -> list[ReadRequest]: ...

    @abstractmethod
    def assemble(self, arrays: list[np.ndarray], tau_days: int) -> np.ndarray: ...

    # ------------------------------------------------------------------ #
    # shared assembly helper
    # ------------------------------------------------------------------ #
    def _empty_block(self, shape_hw: tuple[int, int]) -> np.ndarray:
        return np.zeros(
            (self.layout.num_channels, *shape_hw), dtype=np.float32
        )


class SchemaAAdapter(ExpertAdapter):
    """DSI-schema stores, pre-regridded to the common 1-degree grid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hours_index: dict[int, int] = {}  # lead hour -> position
        self._days_index: dict[int, int] = {}  # lead day -> position
        self._lead_signature: Optional[tuple] = None
        self._var_signature: Optional[tuple] = None

    def _grid_error(self, store: Path, ds: xr.Dataset) -> str:
        return (
            f"expert '{self.name}': {store} grid "
            f"({ds.sizes['lat']}x{ds.sizes['lon']}) does not match the "
            f"target 1-degree grid — run "
            f"tools/regrid_dsi_to_1deg.py on this archive first"
        )

    def _discover_store(self, year: int, ds: xr.Dataset) -> None:
        vars_6h = [str(v) for v in ds.attrs.get("channel_variables_6h", [])]
        vars_daily = [str(v) for v in ds.attrs.get("channel_variables_daily", [])]
        hours = (
            [int(v) for v in ds["prediction_timedelta"].values]
            if "prediction_timedelta" in ds.coords
            else []
        )
        days = (
            [int(v) for v in ds["prediction_timedelta_daily"].values]
            if "prediction_timedelta_daily" in ds.coords
            else []
        )
        sig_lead = (tuple(hours), tuple(days))
        sig_vars = (tuple(sorted(vars_6h)), tuple(sorted(vars_daily)))
        if self._lead_signature is None:
            self._lead_signature = sig_lead
            self._var_signature = sig_vars
            self._hours_index = {h: i for i, h in enumerate(hours)}
            self._days_index = {d: i for i, d in enumerate(days)}
            self._resolve_channels(vars_6h, vars_daily)
        else:
            if sig_lead != self._lead_signature or sig_vars != self._var_signature:
                raise ValueError(
                    f"expert '{self.name}': {year}.zarr lead axes or variable "
                    f"set differ from the first year's — archives must be "
                    f"homogeneous across years"
                )

    def _resolve_channels(
        self, vars_6h: list[str], vars_daily: list[str]
    ) -> None:
        precip_pool = vars_6h if self.precip.axis == "6h" else vars_daily
        if self.precip.var not in precip_pool:
            raise ValueError(
                f"expert '{self.name}': precip var '{self.precip.var}' not in "
                f"the {self.precip.axis}-axis variables {sorted(precip_pool)}"
            )
        for v in vars_6h:
            if v == self.precip.var or v.lower() in self.exclude_variables:
                continue
            parsed = parse_flat_name(v)
            if parsed is None:
                logger.debug("%s: unmapped native variable '%s'", self.name, v)
                continue
            idx = self.layout.index_of(*parsed)
            if idx is None:
                logger.debug(
                    "%s: '%s' (%s) not in the master layout", self.name, v, parsed
                )
                continue
            self._dyn_channels[v] = idx
        # Daily-axis variables other than precip are precip-like diagnostics
        # in these archives; deliberately ignored as gate predictors.

    def _validate_discovery(self) -> None:
        pass

    def lead_supported(self, tau_days: int) -> bool:
        self._require_discovered()
        if self._dyn_channels and (tau_days * 24) not in self._hours_index:
            return False
        pool = (
            self._hours_index if self.precip.axis == "6h" else self._days_index
        )
        return all(v in pool for v in self.precip.lead_values(tau_days))

    def plan(self, year: int, init_idx: int, tau_days: int) -> list[ReadRequest]:
        pool = (
            self._hours_index if self.precip.axis == "6h" else self._days_index
        )
        lead_idx = [pool[v] for v in self.precip.lead_values(tau_days)]
        reqs = [
            ReadRequest((self.name, year, self.precip.var), (init_idx, lead_idx))
        ]
        h_idx = None
        if self._dyn_channels:
            h_idx = self._hours_index[tau_days * 24]
        for v in self._dyn_channels:
            reqs.append(ReadRequest((self.name, year, v), (init_idx, h_idx)))
        return reqs

    def assemble(self, arrays: list[np.ndarray], tau_days: int) -> np.ndarray:
        precip_slabs = np.asarray(arrays[0], dtype=np.float32)
        block = self._empty_block(precip_slabs.shape[-2:])
        block[0] = self.precip.to_mm_per_day(precip_slabs)
        for slab, idx in zip(arrays[1:], self._dyn_channels.values()):
            block[idx] = slab
        return block


class SchemaBAdapter(ExpertAdapter):
    """Consolidated-schema stores (pangu_s2s / sfno_era5 / archesweather)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._n_lead: Optional[int] = None
        # upper var -> (level positions in the store, master channel indices)
        self._upper_reads: dict[str, tuple[list[int], list[int]]] = {}
        self._var_signature: Optional[tuple] = None

    def _discover_store(self, year: int, ds: xr.Dataset) -> None:
        surface = [str(v) for v in ds.attrs.get("surface_variables", [])]
        diagnostic = [str(v) for v in ds.attrs.get("diagnostic_variables", [])]
        upper = [str(v) for v in ds.attrs.get("pressure_upper_air_variables", [])]
        if not upper:
            upper = [str(v) for v in ds.attrs.get("upper_air_variables", [])]
        levels = (
            [float(v) for v in ds["pressure_level"].values]
            if "pressure_level" in ds.coords
            else []
        )
        n_lead = int(ds.sizes["lead_time"])
        sig = (tuple(sorted(surface)), tuple(sorted(diagnostic)),
               tuple(sorted(upper)), tuple(levels))
        if self._var_signature is None:
            self._var_signature = sig
            self._n_lead = n_lead
            self._resolve_channels(surface, diagnostic, upper, levels)
        else:
            if sig != self._var_signature:
                raise ValueError(
                    f"expert '{self.name}': {year}.zarr variable set or levels "
                    f"differ from the first year's — archives must be "
                    f"homogeneous across years"
                )
            # Guard against a short year: availability uses the minimum.
            self._n_lead = min(self._n_lead, n_lead)

    def _resolve_channels(
        self,
        surface: list[str],
        diagnostic: list[str],
        upper: list[str],
        levels: list[float],
    ) -> None:
        scalars = [*surface, *diagnostic]
        if self.precip.var not in scalars:
            raise ValueError(
                f"expert '{self.name}': precip var '{self.precip.var}' not in "
                f"the store's surface/diagnostic variables {sorted(scalars)}"
            )
        if self.precip.axis != "daily":
            raise ValueError(
                f"expert '{self.name}': consolidated stores are daily "
                f"(lead_time_hours=24); precip axis must be 'daily'"
            )
        for v in scalars:
            if v == self.precip.var or v.lower() in self.exclude_variables:
                continue
            canonical = canonicalize_scalar(v)
            if canonical is None:
                logger.debug("%s: unmapped scalar variable '%s'", self.name, v)
                continue
            idx = self.layout.index_of(canonical, None)
            if idx is not None:
                self._dyn_channels[v] = idx
        for v in upper:
            if v.lower() in self.exclude_variables:
                continue
            canonical = canonicalize_upper(v)
            if canonical is None:
                logger.debug("%s: unmapped upper variable '%s'", self.name, v)
                continue
            wanted = self.layout.upper_air_levels(canonical)
            if not wanted:
                continue
            level_pos: list[int] = []
            chan_idx: list[int] = []
            for lv in wanted:
                matches = [
                    j for j, stored in enumerate(levels) if levels_match(stored, lv)
                ]
                if not matches:
                    raise ValueError(
                        f"expert '{self.name}': master level {lv} hPa for "
                        f"'{canonical}' not in the store's pressure_level "
                        f"coord {levels} — add '{v}' to exclude_variables to "
                        f"opt out explicitly"
                    )
                level_pos.append(matches[0])
                chan_idx.append(self.layout.index_of(canonical, lv))
            self._upper_reads[v] = (level_pos, chan_idx)

    def _validate_discovery(self) -> None:
        pass

    @property
    def channel_mask(self) -> np.ndarray:
        mask = super().channel_mask
        for _, chan_idx in self._upper_reads.values():
            mask[chan_idx] = True
        return mask

    def lead_supported(self, tau_days: int) -> bool:
        self._require_discovered()
        leads = self.precip.lead_values(tau_days)
        lo, hi = min(*leads, tau_days), max(*leads, tau_days)
        # lead 0 is the IC, not a forecast — require forecasts only.
        return lo >= 1 and hi <= self._n_lead - 1

    def plan(self, year: int, init_idx: int, tau_days: int) -> list[ReadRequest]:
        leads = self.precip.lead_values(tau_days)
        reqs = [
            ReadRequest(
                (self.name, year, self.precip.var),
                (init_idx, leads if len(leads) > 1 else leads[0]),
            )
        ]
        for v in self._dyn_channels:
            reqs.append(ReadRequest((self.name, year, v), (init_idx, tau_days)))
        for v, (level_pos, _) in self._upper_reads.items():
            reqs.append(
                ReadRequest((self.name, year, v), (init_idx, tau_days, level_pos))
            )
        return reqs

    def assemble(self, arrays: list[np.ndarray], tau_days: int) -> np.ndarray:
        precip_slabs = np.asarray(arrays[0], dtype=np.float32)
        if precip_slabs.ndim == 2:
            precip_slabs = precip_slabs[np.newaxis]
        block = self._empty_block(precip_slabs.shape[-2:])
        block[0] = self.precip.to_mm_per_day(precip_slabs)
        pos = 1
        for idx in self._dyn_channels.values():
            block[idx] = arrays[pos]
            pos += 1
        for _, chan_idx in self._upper_reads.values():
            slab = np.asarray(arrays[pos], dtype=np.float32)
            for row, idx in enumerate(chan_idx):
                block[idx] = slab[row]
            pos += 1
        return block


def build_adapter(
    name: str,
    schema: str,
    root: str | Path,
    layout: ChannelLayout,
    precip: PrecipSpec,
    *,
    exclude_variables: Sequence[str] = (),
) -> ExpertAdapter:
    """Config-string factory: ``schema`` is ``"dsi"`` or ``"consolidated"``."""
    if schema == "dsi":
        cls = SchemaAAdapter
    elif schema == "consolidated":
        cls = SchemaBAdapter
    else:
        raise ValueError(
            f"expert '{name}': unknown schema '{schema}' "
            f"(expected 'dsi' or 'consolidated')"
        )
    return cls(name, root, layout, precip, exclude_variables=exclude_variables)
