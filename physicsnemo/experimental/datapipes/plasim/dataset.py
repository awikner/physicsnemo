# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""PLASIM Zarr dataset.

Lazy ``xarray + zarr`` reader that produces tensors shaped exactly the way
:class:`physicsnemo.experimental.models.pangu_plasim.PanguPlasim` expects in
``forward``. The dataset is responsible for IO and channel routing only;
normalization, ``predict_delta`` tendency computation, and lead-time pairing
live in :mod:`.samplers` and (later) :mod:`.transforms`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import xarray as xr
import zarr
from torch.utils.data import Dataset
from zarr.api.asynchronous import open_group as _zarr_open_group_async
from zarr.core.sync import sync as _zarr_sync

PLASIM_ZARR_SCHEMA_VERSION = "1.0"


# Raise zarr's async concurrency cap to a generous default. The library default
# (10) bottlenecks batched reads under high worker counts; the Earthmover blog on
# zarr-python 3 shows headline wins from raising this. Free at module import.
# See ``benchmarks/.../plasim/RESULTS.md`` for context on why this matters.
try:
    # NOTE: dotted-key form merges; the full-dict form would replace the
    # entire ``async`` sub-config and strip ``async.timeout``.
    zarr.config.set({"async.concurrency": 100})
except Exception:  # pragma: no cover — defensive; older zarr layouts
    pass


@dataclass
class PlasimStoreLayout:
    """Channel-group bookkeeping read from the Zarr store's ``attrs``."""

    surface_variables: list[str] = field(default_factory=list)
    constant_boundary_variables: list[str] = field(default_factory=list)
    varying_boundary_variables: list[str] = field(default_factory=list)
    diagnostic_variables: list[str] = field(default_factory=list)
    pressure_upper_air_variables: list[str] = field(default_factory=list)
    sigma_upper_air_variables: list[str] = field(default_factory=list)
    calendar: str = "proleptic_gregorian"
    data_timedelta_hours: int = 6
    pressure_levels: list[float] = field(default_factory=list)
    sigma_levels: list[float] = field(default_factory=list)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "PlasimStoreLayout":
        a = ds.attrs

        def _strlist(key: str) -> list[str]:
            v = a.get(key, [])
            if isinstance(v, str):
                # xarray sometimes serializes attr lists as strings — be defensive.
                v = v.strip("[]").replace("'", "").replace('"', "")
                v = [x.strip() for x in v.split(",") if x.strip()]
            return list(v)

        layout = cls(
            surface_variables=_strlist("surface_variables"),
            constant_boundary_variables=_strlist("constant_boundary_variables"),
            varying_boundary_variables=_strlist("varying_boundary_variables"),
            diagnostic_variables=_strlist("diagnostic_variables"),
            pressure_upper_air_variables=_strlist("pressure_upper_air_variables"),
            sigma_upper_air_variables=_strlist("sigma_upper_air_variables"),
            calendar=str(a.get("calendar", "proleptic_gregorian")),
            data_timedelta_hours=int(a.get("data_timedelta_hours", 6)),
        )
        if "pressure_level" in ds.coords:
            layout.pressure_levels = ds["pressure_level"].values.astype("float32").tolist()
        if "sigma_level" in ds.coords:
            layout.sigma_levels = ds["sigma_level"].values.astype("float32").tolist()
        return layout


class PlasimClimateDataset(Dataset):
    r"""Random-access PLASIM climate dataset backed by a Zarr store.

    Yields per-sample dicts containing the four tensors
    :class:`PanguPlasim.forward` consumes (``surface_in``,
    ``constant_boundary``, ``varying_boundary``, ``upper_air_in``) plus the
    same shapes for ``target_surface`` and ``target_upper_air`` (paired
    by lead time). The companion :class:`.samplers.LeadTimePairSampler`
    drives the (``t``, ``lead``) iteration.

    Channel ordering exactly follows the source-config lists carried in
    the store ``attrs`` (see :class:`PlasimStoreLayout`). Sigma- and
    pressure-level upper-air variables are concatenated **along the
    variable dim** (sigma vars first, pressure vars second) — the model
    treats both as opaque ``(n_upper, n_levels, H, W)`` channels. The
    sigma and pressure level coordinate arrays themselves are exposed on
    the dataset (:attr:`sigma_levels`, :attr:`pressure_levels`) for
    transforms / metric code that needs the actual level values.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        Path to a Zarr store produced by ``tools/data/plasim/pangu_h5_to_zarr.py``.
    consolidated : bool, optional, default=True
        Whether to read consolidated Zarr metadata (faster open). Ignored if
        the store has no consolidated metadata.
    pin_memory_dtype : torch.dtype, optional, default=torch.float32
        Tensor dtype produced by ``__getitem__``.

    Forward
    -------
    index : int
        Position in the dataset's sequence of (start time, lead time) pairs as
        produced by an associated sampler. With a default sampler, ``index``
        addresses a single (start, lead) pair.

    Outputs
    -------
    dict
        ``{"surface_in": Tensor(C_s, H, W),
        "constant_boundary": Tensor(C_b^c, H, W),
        "varying_boundary": Tensor(C_b^v, H, W),
        "upper_air_in": Tensor(C_u, L, H, W),
        "target_surface": Tensor(C_s, H, W),
        "target_upper_air": Tensor(C_u, L, H, W),
        "diagnostic": Tensor(C_d, H, W),
        "lead_time": Tensor(int),
        "time_idx": Tensor(int)}``
        — the diagnostic tensor is present iff ``diagnostic_variables`` is
        non-empty; ``upper_air_in`` and ``target_upper_air`` stack sigma vars
        first then pressure vars along the variable axis.

    Notes
    -----
    The Zarr store is opened lazily; per-sample reads are slice operations
    on xarray's lazy view. This is fast on fast filesystems (Delta /work/nvme)
    and degrades gracefully on shared filesystems (chunked reads are cheap
    relative to the open).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.datapipes.plasim import PlasimClimateDataset
    >>> ds = PlasimClimateDataset("/path/to/smoke_month.zarr")
    >>> sample = ds[(0, 1)]
    >>> sample["surface_in"].shape, sample["upper_air_in"].shape
    (torch.Size([2, 64, 128]), torch.Size([5, 10, 64, 128]))
    """

    def __init__(
        self,
        zarr_path: str | Path,
        consolidated: bool = True,
        pin_memory_dtype: torch.dtype = torch.float32,
        transform=None,
        *,
        boundary_zarr_path: Optional[str | Path] = None,
        yearly_repeating_boundary: bool = False,
        leap_boundary_zarr_path: Optional[str | Path] = None,
        non_leap_boundary_zarr_path: Optional[str | Path] = None,
    ) -> None:
        self.zarr_path = str(zarr_path)
        self.dtype = pin_memory_dtype
        self.transform = transform
        # Boundary-substitution configuration (Phase-2 follow-up; see
        # implementation_plan.md). When set, varying boundary variables are
        # read from a SEPARATE Zarr store (single-year, time-indexed) instead
        # of the prognostic Zarr at the same time index.
        self._boundary_zarr_path = (
            str(boundary_zarr_path) if boundary_zarr_path is not None else None
        )
        self._yearly_repeating_boundary = bool(yearly_repeating_boundary)
        self._leap_boundary_zarr_path = (
            str(leap_boundary_zarr_path) if leap_boundary_zarr_path is not None else None
        )
        self._non_leap_boundary_zarr_path = (
            str(non_leap_boundary_zarr_path)
            if non_leap_boundary_zarr_path is not None
            else None
        )
        if self._yearly_repeating_boundary:
            if not (self._leap_boundary_zarr_path and self._non_leap_boundary_zarr_path):
                raise ValueError(
                    "yearly_repeating_boundary=True requires both "
                    "leap_boundary_zarr_path and non_leap_boundary_zarr_path."
                )
        # `xr.open_zarr` is lazy; the actual slice reads happen in __getitem__.
        # consolidated=True is the fast path when the store has consolidated
        # metadata; xarray transparently falls back if it doesn't.
        self._ds = xr.open_zarr(self.zarr_path, consolidated=consolidated)
        self.layout = PlasimStoreLayout.from_dataset(self._ds)

        # Sanity: when both level systems are used, the model concat assumes
        # equal level counts. (Pad-to-max could be added later; v1 = strict.)
        if (
            self.layout.sigma_upper_air_variables
            and self.layout.pressure_upper_air_variables
        ):
            if len(self.layout.sigma_levels) != len(self.layout.pressure_levels):
                raise ValueError(
                    f"Mismatched level counts: sigma={len(self.layout.sigma_levels)}, "
                    f"pressure={len(self.layout.pressure_levels)}. v1 requires equal "
                    "counts to concatenate dual-system upper-air variables; pad-to-max "
                    "support TBD."
                )

        self.n_time = int(self._ds.sizes["time"])
        self.n_lat = int(self._ds.sizes["lat"])
        self.n_lon = int(self._ds.sizes["lon"])

        # Hot-path reads bypass xarray. Per-sample we issue one ``asyncio.gather``
        # across all per-variable AsyncArray.getitem calls — a single ``zarr_sync``
        # round-trip replaces N separate event-loop spins. Pattern follows
        # ``physicsnemo.experimental.datapipes.healda.loaders.zarr_loader.ZarrLoader._get``.
        # Profiling showed ~78 % of single-process __getitem__ time was in
        # zarr.core.sync per-call asyncio bookkeeping; batched gather collapses
        # that to one call.
        #
        # Known issue (zarr-python 3): the synchronous read API has substantial
        # per-call overhead. See
        # https://github.com/zarr-developers/zarr-python/issues/3524 and
        # https://github.com/zarr-developers/zarr-python/issues/2084. When that
        # is fixed upstream, this hand-rolled batching is no longer needed and
        # can be replaced with naive per-variable reads against the sync API
        # (or even xarray) — re-benchmark before deleting.
        self._async_group = _zarr_sync(
            _zarr_open_group_async(self.zarr_path, mode="r")
        )
        self._async_arrays: dict[str, object] = {}
        for v in (
            *self.layout.surface_variables,
            *self.layout.varying_boundary_variables,
            *self.layout.diagnostic_variables,
            *self.layout.sigma_upper_air_variables,
            *self.layout.pressure_upper_air_variables,
        ):
            self._async_arrays[v] = _zarr_sync(self._async_group.get(v))

        # Constant boundaries don't vary over time — read once at init so the
        # per-sample async batch doesn't include them.
        self._constants_tensor = self._eager_load_constants()

        # Boundary-substitution: open the separate boundary Zarr group(s) and
        # cache async-array handles for the varying-boundary variables only.
        # Per-sample varying-boundary reads then route through these handles.
        self._boundary_async_groups: dict[str, object] = {}
        self._boundary_async_arrays: dict[str, dict[str, object]] = {}
        if self._yearly_repeating_boundary:
            self._open_boundary_store("leap", self._leap_boundary_zarr_path)
            self._open_boundary_store("non_leap", self._non_leap_boundary_zarr_path)
        elif self._boundary_zarr_path:
            self._open_boundary_store("single", self._boundary_zarr_path)
        # Cache the prognostic time coord for boundary-time-index lookup.
        # Only needed when actually using a separate boundary store.
        self._prog_times = (
            self._ds["time"].values if self._boundary_async_groups else None
        )
        self._steps_per_day = (
            24 // self.layout.data_timedelta_hours
            if self.layout.data_timedelta_hours
            else 4
        )

    def _open_boundary_store(self, key: str, path: str) -> None:
        group = _zarr_sync(_zarr_open_group_async(path, mode="r"))
        self._boundary_async_groups[key] = group
        self._boundary_async_arrays[key] = {
            v: _zarr_sync(group.get(v))
            for v in self.layout.varying_boundary_variables
        }

    def _boundary_store_key(self, time_idx: int) -> str:
        """Pick which boundary store to read from for a given prognostic time index."""
        if self._yearly_repeating_boundary:
            import cftime

            t = self._prog_times[time_idx]
            year = t.year
            try:
                is_leap = cftime.is_leap_year(year, self.layout.calendar)
            except Exception:
                # cftime API variance across versions; fall back to per-cls method.
                is_leap = hasattr(t, "daysinmonth") and t.daysinmonth[1] == 29
            return "leap" if is_leap else "non_leap"
        return "single"

    def _boundary_time_index(self, time_idx: int) -> int:
        """Map the prognostic time index to the boundary-store time index."""
        if not self._boundary_async_groups:
            return time_idx
        t = self._prog_times[time_idx]
        # Day-of-year (1-indexed in cftime); convert to 0-indexed.
        try:
            doy = t.dayofyr - 1
        except AttributeError:
            doy = t.timetuple().tm_yday - 1
        hour = t.hour
        return doy * self._steps_per_day + hour // self.layout.data_timedelta_hours

    # ------------------------------------------------------------------ #
    # Public read-only attributes
    # ------------------------------------------------------------------ #
    @property
    def pressure_levels(self) -> list[float]:
        return list(self.layout.pressure_levels)

    @property
    def sigma_levels(self) -> list[float]:
        return list(self.layout.sigma_levels)

    @property
    def horizontal_resolution(self) -> tuple[int, int]:
        return (self.n_lat, self.n_lon)

    @property
    def num_levels(self) -> int:
        # Sigma and pressure (when both present) are required equal-length above.
        if self.layout.sigma_levels:
            return len(self.layout.sigma_levels)
        return len(self.layout.pressure_levels)

    @property
    def num_upper_air_channels(self) -> int:
        return len(self.layout.sigma_upper_air_variables) + len(
            self.layout.pressure_upper_air_variables
        )

    @property
    def upper_air_variable_names(self) -> list[str]:
        """Concatenation order used by ``upper_air_in`` (sigma first, then pressure)."""
        return [
            *self.layout.sigma_upper_air_variables,
            *self.layout.pressure_upper_air_variables,
        ]

    def __len__(self) -> int:
        return self.n_time

    # ------------------------------------------------------------------ #
    # Sample assembly — async-batched zarr reads (see __init__ note for why).
    # ------------------------------------------------------------------ #
    def _eager_load_constants(self) -> Optional[torch.Tensor]:
        """Read constant boundary fields once at init (they don't vary in time)."""
        names = self.layout.constant_boundary_variables
        if not names:
            return None

        async def _read_one(n: str):
            arr = await self._async_group.get(n)
            return await arr.getitem(slice(None))

        async def _batch():
            return await asyncio.gather(*(_read_one(n) for n in names))

        arrays = [np.asarray(a, dtype="float32") for a in _zarr_sync(_batch())]
        return torch.from_numpy(np.stack(arrays, axis=0)).to(self.dtype)

    def _read_all_async(self, time_idx: int) -> dict[str, np.ndarray]:
        """Issue ALL per-sample variable reads in a single asyncio.gather call.

        Returns a dict mapping variable name to the freshly-read numpy array.
        """
        return self._read_many_async([time_idx])[time_idx]

    def _read_many_async(
        self, time_indices: Sequence[int]
    ) -> dict[int, dict[str, np.ndarray]]:
        """Coalesce reads for MULTIPLE time indices into one ``asyncio.gather``.

        Used by ``__getitem__`` to batch the start + target sample reads.
        Returns ``{time_idx: {var_name: ndarray}}``. When a separate boundary
        store is configured, the varying-boundary variables route through that
        store (with day-of-year + leap-aware time-index translation) instead
        of the prognostic store.
        """
        # Variables read from the PROGNOSTIC store (skip varying-boundary when
        # a boundary store is configured — those go through the boundary path).
        prog_names: list[str] = [
            *self.layout.surface_variables,
            *self.layout.diagnostic_variables,
            *self.layout.sigma_upper_air_variables,
            *self.layout.pressure_upper_air_variables,
        ]
        use_boundary_store = bool(self._boundary_async_groups)
        if not use_boundary_store:
            prog_names = (
                self.layout.surface_variables
                + self.layout.varying_boundary_variables
                + self.layout.diagnostic_variables
                + self.layout.sigma_upper_air_variables
                + self.layout.pressure_upper_air_variables
            )

        async def _batch():
            tasks = []
            # Prognostic-store reads.
            for t in time_indices:
                for n in prog_names:
                    tasks.append(self._async_arrays[n].getitem(t))
            # Boundary-store reads (separate physical store + index translation).
            if use_boundary_store:
                for t in time_indices:
                    key = self._boundary_store_key(t)
                    bt = self._boundary_time_index(t)
                    for n in self.layout.varying_boundary_variables:
                        tasks.append(
                            self._boundary_async_arrays[key][n].getitem(bt)
                        )
            return await asyncio.gather(*tasks)

        arrays = _zarr_sync(_batch())
        out: dict[int, dict[str, np.ndarray]] = {t: {} for t in time_indices}
        k = 0
        for t in time_indices:
            for n in prog_names:
                out[t][n] = np.asarray(arrays[k], dtype="float32")
                k += 1
        if use_boundary_store:
            for t in time_indices:
                for n in self.layout.varying_boundary_variables:
                    out[t][n] = np.asarray(arrays[k], dtype="float32")
                    k += 1
        return out

    def _stack_named(
        self, raw: dict[str, np.ndarray], names: Sequence[str]
    ) -> Optional[torch.Tensor]:
        if not names:
            return None
        stacked = np.stack([raw[n] for n in names], axis=0)
        return torch.from_numpy(stacked).to(self.dtype)

    def _build_sample(self, raw: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Assemble a sample dict from a per-variable ``{name: ndarray}`` mapping."""
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = self._stack_named(raw, self.layout.surface_variables)
        if self._constants_tensor is not None:
            out["constant_boundary"] = self._constants_tensor
        out["varying_boundary"] = self._stack_named(
            raw, self.layout.varying_boundary_variables
        )

        # Upper-air: sigma vars first, then pressure vars, concatenated along var.
        parts: list[torch.Tensor] = []
        sigma = self._stack_named(raw, self.layout.sigma_upper_air_variables)
        if sigma is not None:
            parts.append(sigma)
        pressure = self._stack_named(raw, self.layout.pressure_upper_air_variables)
        if pressure is not None:
            parts.append(pressure)
        out["upper_air_in"] = (
            parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        )

        if self.layout.diagnostic_variables:
            out["diagnostic"] = self._stack_named(raw, self.layout.diagnostic_variables)
        return out

    def _sample_at(self, time_idx: int) -> dict[str, torch.Tensor]:
        return self._build_sample(self._read_all_async(time_idx))

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        r"""Index is a single int or a ``(start_time_idx, lead_time_steps)`` pair.

        Parameters
        ----------
        index : int or (int, int)
            ``int`` is shorthand for ``(index, 1)`` (next-step prediction). Tuples
            yield a paired sample with the target at ``start + lead``.

        Returns
        -------
        dict of torch.Tensor
            See class docstring.
        """
        if isinstance(index, tuple):
            start_idx, lead = index
        else:
            start_idx, lead = int(index), 1
        target_idx = start_idx + lead
        if not (0 <= start_idx < self.n_time and 0 <= target_idx < self.n_time):
            raise IndexError(
                f"index ({start_idx}, {lead}) -> ({start_idx}, {target_idx}) out of "
                f"range [0, {self.n_time})"
            )

        # Coalesce start + target reads into ONE asyncio.gather (halves the
        # per-sample asyncio sync-bookkeeping cost vs two _sample_at calls).
        raw = self._read_many_async([start_idx, target_idx])
        sample = self._build_sample(raw[start_idx])
        target = self._build_sample(raw[target_idx])
        sample["target_surface"] = target["surface_in"]
        sample["target_upper_air"] = target["upper_air_in"]
        sample["lead_time"] = torch.tensor(lead, dtype=torch.long)
        sample["time_idx"] = torch.tensor(start_idx, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
