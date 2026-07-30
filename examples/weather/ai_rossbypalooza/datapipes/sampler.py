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

"""DDP-aware sampler over the precomputed (init, tau) pair table.

Mechanics copied from ``physicsnemo/experimental/datapipes/climate/
samplers.py::LeadTimePairSampler`` (``seed + epoch`` generator,
``[rank::world_size][:per_rank_len]`` sharding, ``set_epoch``); it yields
plain ints because the (init, tau) pairing already lives in the dataset's
index — uniform pair sampling is uniform tau sampling up to the
(correctly) dropped IMERG-gap pairs.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler


class MixturePairSampler(Sampler[int]):
    """Random (or sequential) pair indices, sharded across DDP ranks.

    Parameters
    ----------
    num_pairs : size of the dataset's pair table.
    num_samples : draws per epoch across all ranks (default: ``num_pairs``).
        May exceed ``num_pairs`` (sampling with replacement via randint).
    shuffle : random draws when True; a deterministic cycle when False.
    seed, rank, world_size : the usual; reseeded per epoch via
        :meth:`set_epoch`.
    """

    def __init__(
        self,
        num_pairs: int,
        *,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"bad rank/world_size: {rank}/{world_size}")
        self.num_pairs = int(num_pairs)
        self.num_samples = int(num_samples) if num_samples else self.num_pairs
        if self.num_samples < world_size:
            raise ValueError(
                f"num_samples={self.num_samples} < world_size={world_size}"
            )
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples // self.world_size

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            if self.num_samples <= self.num_pairs:
                idx = torch.randperm(self.num_pairs, generator=g)[
                    : self.num_samples
                ]
            else:
                idx = torch.randint(
                    self.num_pairs, (self.num_samples,), generator=g
                )
        else:
            idx = torch.arange(self.num_samples) % self.num_pairs
        shard = idx[self.rank :: self.world_size][: len(self)]
        yield from shard.tolist()
