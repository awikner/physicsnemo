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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import xarray as xr
from jaxtyping import Float
from torch.utils.data import Dataset

PLASIM_ZARR_SCHEMA_VERSION = "1.0"


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
    ) -> None:
        self.zarr_path = str(zarr_path)
        self.dtype = pin_memory_dtype
        self.transform = transform
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
    # Sample assembly
    # ------------------------------------------------------------------ #
    def _stack_along_var(
        self,
        names: Sequence[str],
        time_idx: int,
        *,
        with_levels: bool,
    ) -> Optional[Float[torch.Tensor, "..."]]:
        """Read and stack a list of variables for a single time index."""
        if not names:
            return None
        arrays: list[np.ndarray] = []
        for v in names:
            arr = self._ds[v].isel(time=time_idx).values.astype("float32", copy=False)
            arrays.append(arr)
        stacked = np.stack(arrays, axis=0)  # (n_vars, [L,] H, W)
        return torch.from_numpy(stacked).to(self.dtype)

    def _stack_upper_air(self, time_idx: int) -> torch.Tensor:
        """Sigma vars first (n_sigma, L, H, W); pressure vars second; concat along var."""
        parts: list[torch.Tensor] = []
        sigma = self._stack_along_var(
            self.layout.sigma_upper_air_variables, time_idx, with_levels=True
        )
        if sigma is not None:
            parts.append(sigma)
        pressure = self._stack_along_var(
            self.layout.pressure_upper_air_variables, time_idx, with_levels=True
        )
        if pressure is not None:
            parts.append(pressure)
        return torch.cat(parts, dim=0)

    def _read_constant_boundary(self) -> Optional[torch.Tensor]:
        if not self.layout.constant_boundary_variables:
            return None
        arrays = [
            self._ds[v].values.astype("float32", copy=False)
            for v in self.layout.constant_boundary_variables
        ]
        return torch.from_numpy(np.stack(arrays, axis=0)).to(self.dtype)

    def _sample_at(self, time_idx: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        out["surface_in"] = self._stack_along_var(
            self.layout.surface_variables, time_idx, with_levels=False
        )
        const = self._read_constant_boundary()
        if const is not None:
            out["constant_boundary"] = const
        out["varying_boundary"] = self._stack_along_var(
            self.layout.varying_boundary_variables, time_idx, with_levels=False
        )
        out["upper_air_in"] = self._stack_upper_air(time_idx)
        if self.layout.diagnostic_variables:
            out["diagnostic"] = self._stack_along_var(
                self.layout.diagnostic_variables, time_idx, with_levels=False
            )
        return out

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

        sample = self._sample_at(start_idx)
        target = self._sample_at(target_idx)
        sample["target_surface"] = target["surface_in"]
        sample["target_upper_air"] = target["upper_air_in"]
        sample["lead_time"] = torch.tensor(lead, dtype=torch.long)
        sample["time_idx"] = torch.tensor(start_idx, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
