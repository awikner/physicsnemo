# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Each epoch must get its own shuffle under DDP.

``DistributedSampler`` derives its permutation from ``(seed, epoch)`` and its
``epoch`` only moves when ``set_epoch`` is called. The diffusion trainer ran
for months without that call, which silently made an N-epoch run N passes over
one frozen partition — same order, same rank->sample assignment, every epoch.
There is no error and no log line for this, so it gets a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_loop import advance_sampler_epoch  # noqa: E402

N = 64
WORLD = 4
SEED = 0


def _sampler(rank: int) -> DistributedSampler:
    return DistributedSampler(
        torch.arange(N), num_replicas=WORLD, rank=rank, shuffle=True, seed=SEED
    )


def _loader(rank: int) -> DataLoader:
    s = _sampler(rank)
    return DataLoader(torch.arange(N), batch_size=1, sampler=s)


def test_epoch_changes_the_permutation():
    loader = _loader(rank=0)
    advance_sampler_epoch(loader, 1)
    first = list(loader.sampler)
    advance_sampler_epoch(loader, 2)
    second = list(loader.sampler)
    assert first != second, "epoch 2 replayed epoch 1's order"


def test_without_the_call_every_epoch_is_identical():
    """The bug this guards against, stated as a test."""
    loader = _loader(rank=0)
    assert list(loader.sampler) == list(loader.sampler)


def test_same_epoch_is_reproducible():
    """Needed for resume: re-entering epoch k must rebuild epoch k's order."""
    a, b = _loader(rank=0), _loader(rank=0)
    advance_sampler_epoch(a, 7)
    advance_sampler_epoch(b, 7)
    assert list(a.sampler) == list(b.sampler)


def test_ranks_stay_disjoint_and_cover_the_dataset():
    """Reshuffling must not make two ranks train on the same sample."""
    for epoch in (1, 2, 5):
        seen: list[int] = []
        for rank in range(WORLD):
            loader = _loader(rank)
            advance_sampler_epoch(loader, epoch)
            seen.extend(list(loader.sampler))
        assert len(seen) == N, f"epoch {epoch}: {len(seen)} indices, want {N}"
        assert sorted(seen) == list(range(N)), f"epoch {epoch}: not a partition"


def test_returns_false_and_does_not_raise_without_a_distributed_sampler():
    # Single-GPU path: train_diffusion builds sampler=None.
    plain = DataLoader(torch.arange(N), batch_size=1, shuffle=True)
    assert advance_sampler_epoch(plain, 3) is False
    assert advance_sampler_epoch(object(), 3) is False


def test_returns_true_when_it_advanced_one():
    assert advance_sampler_epoch(_loader(rank=0), 3) is True
