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

"""Tests for MixturePairSampler (datapipes/sampler.py)."""

from __future__ import annotations

import pytest

from datapipes.sampler import MixturePairSampler


def test_seed_epoch_determinism():
    s = MixturePairSampler(100, seed=7)
    s.set_epoch(3)
    a = list(s)
    s.set_epoch(3)
    assert list(s) == a
    s.set_epoch(4)
    assert list(s) != a


def test_full_permutation_when_num_samples_default():
    s = MixturePairSampler(50, seed=0)
    idx = list(s)
    assert len(idx) == 50
    assert sorted(idx) == list(range(50))


def test_ddp_shards_disjoint_and_equal_length():
    world = 4
    shards = []
    for rank in range(world):
        s = MixturePairSampler(
            103, num_samples=100, seed=1, rank=rank, world_size=world
        )
        s.set_epoch(2)
        shards.append(list(s))
    assert all(len(sh) == 25 for sh in shards)
    flat = [i for sh in shards for i in sh]
    assert len(set(flat)) == len(flat)  # disjoint across ranks


def test_oversampling_with_replacement():
    s = MixturePairSampler(10, num_samples=40, seed=0)
    idx = list(s)
    assert len(idx) == 40
    assert set(idx) <= set(range(10))


def test_sequential_walk():
    s = MixturePairSampler(5, num_samples=8, shuffle=False)
    assert list(s) == [0, 1, 2, 3, 4, 0, 1, 2]


def test_validation():
    with pytest.raises(ValueError, match="must be positive"):
        MixturePairSampler(0)
    with pytest.raises(ValueError, match="bad rank"):
        MixturePairSampler(10, rank=2, world_size=2)
    with pytest.raises(ValueError, match="< world_size"):
        MixturePairSampler(10, num_samples=2, world_size=4)
