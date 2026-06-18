# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Composite Dataset that glues a directory of per-year PLASIM Zarr stores.

The new format produced by ``tools/data/plasim/pangu_h5_to_zarr.py`` writes one
Zarr per year. For multi-year training, :class:`PlasimMultiYearDataset` wraps
many such per-year stores into a single ``torch.utils.data.Dataset`` with one
contiguous global time index — start (and target) sample reads dispatch to the
sub-dataset for whichever year covers the global index.

This composition layer reuses :class:`PlasimClimateDataset` unchanged for each
year. Channel layout (surface vars, level systems, calendar) must match across
years; constant boundaries are taken from the first year (they're static).

Typical layout::

    <root>/
        100.zarr/    # PanguPlasim year-100 (or arbitrarily named)
        101.zarr/
        102.zarr/
        ...

Sub-stores are sorted by their **start time** (read from the Zarr's ``time``
coord) so the global index aligns chronologically regardless of filename order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import PlasimClimateDataset, PlasimStoreLayout


class PlasimMultiYearDataset(Dataset):
    r"""Composite PLASIM dataset across a directory of per-year Zarr stores.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing per-year ``*.zarr`` sub-stores.
    pin_memory_dtype : torch.dtype, optional, default=torch.float32
        Tensor dtype produced by ``__getitem__``.
    transform : callable or None, optional
        Per-sample transform, applied AFTER the per-year sub-dataset's
        ``__getitem__`` returns. Composition order matches the single-store
        path.
    boundary_zarr_path : str or pathlib.Path, optional
        Single-year boundary store. Forwarded to each per-year sub-dataset.
    yearly_repeating_boundary : bool, optional
        Forwarded to each per-year sub-dataset.
    leap_boundary_zarr_path, non_leap_boundary_zarr_path : str or pathlib.Path, optional
        Forwarded to each per-year sub-dataset when
        ``yearly_repeating_boundary=True``.
    consolidated : bool, optional, default=True
        Forwarded to each per-year sub-dataset.

    Notes
    -----
    The dataset returns the SAME per-sample dict shape as
    :class:`PlasimClimateDataset` (no additional fields), with one
    addition: ``time_idx`` is the GLOBAL time index, not the per-year index.
    This matches the contract :class:`LeadTimePairSampler` already uses.

    Forecast lead-time pairs that would cross a year boundary are handled by
    reading the target sample from the next year's sub-dataset. The composite
    transparently dispatches the start and the target to the correct year
    based on the global index map.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        pin_memory_dtype: torch.dtype = torch.float32,
        transform=None,
        boundary_zarr_path: Optional[Union[str, Path]] = None,
        yearly_repeating_boundary: bool = False,
        leap_boundary_zarr_path: Optional[Union[str, Path]] = None,
        non_leap_boundary_zarr_path: Optional[Union[str, Path]] = None,
        consolidated: bool = True,
    ) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise ValueError(
                f"root {root_path} is not a directory; "
                "PlasimMultiYearDataset expects a directory of *.zarr sub-stores"
            )
        store_paths = sorted(root_path.glob("*.zarr"))
        if not store_paths:
            raise ValueError(f"no *.zarr sub-stores found in {root_path}")

        self.root = root_path
        self.dtype = pin_memory_dtype
        self.transform = transform

        sub_kwargs = dict(
            consolidated=consolidated,
            pin_memory_dtype=pin_memory_dtype,
            transform=None,  # per-year sub-datasets stay raw; we apply ours below
            boundary_zarr_path=boundary_zarr_path,
            yearly_repeating_boundary=yearly_repeating_boundary,
            leap_boundary_zarr_path=leap_boundary_zarr_path,
            non_leap_boundary_zarr_path=non_leap_boundary_zarr_path,
        )
        sub_datasets: list[PlasimClimateDataset] = [
            PlasimClimateDataset(p, **sub_kwargs) for p in store_paths
        ]

        # Sort sub-datasets by start time of each store, then validate layout match.
        keyed = sorted(
            enumerate(sub_datasets),
            key=lambda iv: iv[1]._ds["time"].values[0],
        )
        self._sub_paths = [store_paths[i] for i, _ in keyed]
        self.sub_datasets = [d for _, d in keyed]

        ref_layout = self.sub_datasets[0].layout
        for sub_idx, sub in enumerate(self.sub_datasets[1:], start=1):
            _assert_layouts_match(ref_layout, sub.layout, self._sub_paths[sub_idx])

        # Global index map: cumulative time index → (sub_idx, local_idx)
        sub_lengths = np.asarray([len(d) for d in self.sub_datasets], dtype=np.int64)
        self._cum_lengths = np.concatenate(([0], np.cumsum(sub_lengths)))
        self.layout: PlasimStoreLayout = ref_layout

    # ------------------------------------------------------------------ #
    # Read-only attributes mirroring PlasimClimateDataset's public surface
    # ------------------------------------------------------------------ #
    @property
    def pressure_levels(self) -> list[float]:
        return self.sub_datasets[0].pressure_levels

    @property
    def sigma_levels(self) -> list[float]:
        return self.sub_datasets[0].sigma_levels

    @property
    def horizontal_resolution(self) -> tuple[int, int]:
        return self.sub_datasets[0].horizontal_resolution

    @property
    def num_levels(self) -> int:
        return self.sub_datasets[0].num_levels

    @property
    def num_upper_air_channels(self) -> int:
        return self.sub_datasets[0].num_upper_air_channels

    @property
    def upper_air_variable_names(self) -> list[str]:
        return self.sub_datasets[0].upper_air_variable_names

    def __len__(self) -> int:
        return int(self._cum_lengths[-1])

    # ------------------------------------------------------------------ #
    # Global-index dispatch
    # ------------------------------------------------------------------ #
    def _global_to_local(self, global_idx: int) -> tuple[int, int]:
        """Return ``(sub_dataset_index, local_time_index)`` for a global index."""
        if global_idx < 0:
            global_idx += int(self._cum_lengths[-1])
        if not 0 <= global_idx < int(self._cum_lengths[-1]):
            raise IndexError(
                f"global index {global_idx} out of range [0, {int(self._cum_lengths[-1])})"
            )
        sub_idx = int(np.searchsorted(self._cum_lengths, global_idx, side="right") - 1)
        local_idx = int(global_idx - self._cum_lengths[sub_idx])
        return sub_idx, local_idx

    def __getitem__(
        self, idx: Union[int, tuple[int, int]]
    ) -> dict[str, torch.Tensor]:
        """Dispatch a ``(start, lead)`` global pair to the correct sub-dataset(s).

        The start sample is read from the sub-dataset that owns its global
        index; if the target (start + lead) crosses into the next year, the
        target sample is read from that next sub-dataset. The output dict
        carries the GLOBAL ``time_idx`` (matches what ``LeadTimePairSampler``
        emits).
        """
        if isinstance(idx, tuple):
            start_g, lead = idx
        else:
            start_g, lead = int(idx), 1
        target_g = start_g + lead

        start_sub_idx, start_local = self._global_to_local(start_g)
        target_sub_idx, target_local = self._global_to_local(target_g)

        if start_sub_idx == target_sub_idx:
            # Common path: same year — single sub-dataset call.
            sample = self.sub_datasets[start_sub_idx][(start_local, lead)]
        else:
            # Cross-year lead: assemble manually from two sub-datasets.
            sub_start = self.sub_datasets[start_sub_idx]
            sub_target = self.sub_datasets[target_sub_idx]
            start_sample = sub_start[(start_local, 0)]  # lead=0 → no target read
            target_sample = sub_target[(target_local, 0)]
            sample = dict(start_sample)
            sample["target_surface"] = target_sample["surface_in"]
            sample["target_upper_air"] = target_sample["upper_air_in"]
            sample["lead_time"] = torch.tensor(int(lead), dtype=torch.int64)

        # Rewrite time_idx to the GLOBAL index so downstream code (samplers,
        # logging, metrics) sees a contiguous timeline across the whole archive.
        sample["time_idx"] = torch.tensor(int(start_g), dtype=torch.int64)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def _assert_layouts_match(ref: PlasimStoreLayout, other: PlasimStoreLayout, path: Path) -> None:
    """Raise if a per-year sub-store's layout disagrees with the reference."""
    for field in (
        "surface_variables",
        "constant_boundary_variables",
        "varying_boundary_variables",
        "diagnostic_variables",
        "pressure_upper_air_variables",
        "sigma_upper_air_variables",
        "calendar",
        "data_timedelta_hours",
        "pressure_levels",
        "sigma_levels",
    ):
        if getattr(ref, field) != getattr(other, field):
            raise ValueError(
                f"sub-store {path} layout field {field!r} disagrees with the "
                f"reference: ref={getattr(ref, field)!r}, "
                f"this={getattr(other, field)!r}"
            )
