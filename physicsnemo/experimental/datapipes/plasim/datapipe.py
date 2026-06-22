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

r"""High-level orchestrator that bundles
:class:`PlasimClimateDataset` + :class:`LeadTimePairSampler` +
:class:`PlasimNormalizer` + :class:`torch.utils.data.DataLoader` (with
multi-worker prefetch + persistent workers) + DDP integration via
:class:`physicsnemo.distributed.DistributedManager`. Phase 3's training
recipe consumes this rather than assembling the pieces by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

from .dataset import PlasimClimateDataset
from .samplers import LeadTimePairSampler
from .transforms import PlasimNormalizer


class PlasimClimateDatapipe(Datapipe):
    r"""Training-ready datapipe for PLASIM climate data.

    Wraps :class:`PlasimClimateDataset` in a :class:`torch.utils.data.DataLoader`
    with multi-worker prefetch + persistent workers, drives sampling with
    :class:`LeadTimePairSampler`, optionally applies a :class:`PlasimNormalizer`
    and a NaN-fill on the GPU side of the transfer, and (optionally) wires the
    sampler's rank/world_size to :class:`physicsnemo.distributed.DistributedManager`.

    Iteration yields batched dicts with the keys
    :class:`PlasimClimateDataset` produces (``surface_in``,
    ``constant_boundary``, ``varying_boundary``, ``upper_air_in``,
    ``target_surface``, ``target_upper_air``, ``diagnostic``, ``lead_time``,
    ``time_idx``) plus a leading batch dim. Tensors are moved to ``device``
    (default: ``cuda`` when available else ``cpu``) before yielding.

    Parameters
    ----------
    zarr_path : str or pathlib.Path
        PLASIM Zarr store path (see ``tools/data/plasim/pangu_h5_to_zarr.py``).
    forecast_lead_times : sequence of int, optional, default=(1,)
        Lead times in units of the store's ``data_timedelta_hours``.
    normalizer : PlasimNormalizer or None, optional, default=None
        Optional GPU-side normalizer (applied after device transfer for
        kernel-fused throughput). Pass ``None`` to leave raw values through.
    nan_fill : float or None, optional, default=0.0
        Value to substitute for NaN in ``constant_boundary`` and
        ``varying_boundary`` (PLASIM's ``lsm`` / ``sst`` carry NaN at poles
        / over land respectively by convention). ``None`` disables.
    batch_size : int, optional, default=1
        DataLoader batch size.
    num_samples_per_epoch : int or None, optional, default=None
        Pairs drawn per epoch (before DDP per-rank slicing). ``None`` defaults
        to ``len(dataset)``.
    shuffle : bool, optional, default=True
        Whether the sampler shuffles each epoch.
    num_workers : int, optional, default=4
        DataLoader worker processes.
    prefetch_factor : int, optional, default=2
        DataLoader prefetch factor (per worker). Ignored if ``num_workers == 0``.
    persistent_workers : bool, optional, default=True
        Keep worker processes alive across epochs. Ignored if
        ``num_workers == 0``.
    pin_memory : bool, optional, default=True
        Pin DataLoader output for faster H2D transfer.
    device : torch.device or str or None, optional, default=None
        Target device for the yielded batch. ``None`` picks ``"cuda"`` if
        :func:`torch.cuda.is_available`, else ``"cpu"``.
    seed : int, optional, default=0
        RNG seed for the sampler.
    distributed : bool, optional, default=False
        Pull rank / world_size from :class:`~physicsnemo.distributed.DistributedManager`
        for DDP-friendly per-rank sampling. ``False`` runs single-rank.

    Forward
    -------
    epoch : int, optional
        Set via :meth:`set_epoch` between epochs to advance the sampler's RNG.

    Outputs
    -------
    Iterator of dict of torch.Tensor
        Each yielded dict carries the per-sample tensors with a leading batch
        dim, all on ``device``.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        forecast_lead_times: Sequence[int] = (1,),
        *,
        normalizer: Optional[PlasimNormalizer] = None,
        nan_fill: Optional[float] = 0.0,
        batch_size: int = 1,
        num_samples_per_epoch: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        device: Optional[torch.device | str] = None,
        seed: int = 0,
        distributed: bool = False,
        boundary_zarr_path: Optional[str | Path] = None,
        yearly_repeating_boundary: bool = False,
        leap_boundary_zarr_path: Optional[str | Path] = None,
        non_leap_boundary_zarr_path: Optional[str | Path] = None,
        unroll_steps: int = 1,
    ) -> None:
        super().__init__(meta=DatapipeMetaData(name="plasim_climate"))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.nan_fill = nan_fill

        # Dataset stays IO-only (no transform); the normalizer + NaN-fill run
        # GPU-side after the H2D transfer to fuse into one batched op.
        base_dataset = PlasimClimateDataset(
            zarr_path,
            boundary_zarr_path=boundary_zarr_path,
            yearly_repeating_boundary=yearly_repeating_boundary,
            leap_boundary_zarr_path=leap_boundary_zarr_path,
            non_leap_boundary_zarr_path=non_leap_boundary_zarr_path,
        )

        if distributed:
            from physicsnemo.distributed import DistributedManager

            dm = DistributedManager()
            rank = dm.rank
            world_size = dm.world_size
        else:
            rank, world_size = 0, 1

        self.unroll_steps = int(unroll_steps)
        if self.unroll_steps <= 1:
            # Single-step mode — emit (init, target_at_t+lead) pairs.
            self.dataset = base_dataset
            self.sampler = LeadTimePairSampler(
                dataset_length=len(self.dataset),
                forecast_lead_times=list(forecast_lead_times),
                num_samples=num_samples_per_epoch,
                shuffle=shuffle,
                seed=seed,
                rank=rank,
                world_size=world_size,
            )
        else:
            # Multi-step rollout — wrap with SequenceDataset and use a plain
            # int sampler. Each batch carries a leading time axis of size
            # unroll_steps+1; the trainer consumes the ``_seq`` keys.
            from .sequence import IntSampler, SequenceDataset

            self.dataset = SequenceDataset(base_dataset, unroll_steps=self.unroll_steps)
            self.sampler = IntSampler(
                dataset_length=len(self.dataset),
                num_samples=num_samples_per_epoch,
                shuffle=shuffle,
                seed=seed,
                rank=rank,
                world_size=world_size,
            )

        if normalizer is not None:
            normalizer = normalizer.to(self.device)
        self.normalizer = normalizer

        # `LeadTimePairSampler` yields (start, lead) tuples; the DataLoader's
        # default `Sampler` interface expects an iterable of items that the
        # dataset's __getitem__ accepts — tuples work transparently.
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory=pin_memory,
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the sampler's RNG state. Call between epochs in training."""
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.dataloader:
            batch = self._postprocess(batch)
            yield batch

    def _postprocess(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        if self.nan_fill is not None:
            for key in (
                "constant_boundary",
                "varying_boundary",
                "varying_boundary_seq",
            ):
                if key in moved and torch.is_tensor(moved[key]):
                    moved[key] = torch.nan_to_num(moved[key], nan=float(self.nan_fill))
        if self.normalizer is not None:
            # Normalizer was built for un-batched samples (no leading batch dim);
            # its broadcast-shaped stats are (C, ...) so they apply per-batch
            # via PyTorch broadcasting against (B, C, ...) tensors.
            moved = self.normalizer(moved)
        return moved
