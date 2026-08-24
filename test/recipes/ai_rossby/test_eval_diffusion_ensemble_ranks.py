# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU ensemble validation: members split evenly across ranks.

The eval suite's validators can split a TOTAL ensemble across the
distributed ranks (``split_ensemble_across_ranks``): every rank rolls the
same initial conditions with ``ensemble_size // world_size`` members, and
the per-step ensemble mean/variance are completed by cross-rank reductions.
These tests pin the distributed math against single-process references
using 2-process gloo groups on CPU — the first CPU multi-process pattern in
this test tree (the existing DDP tests are GPU/torchrun-gated). It works
because the validators and streaming accumulators use raw
``torch.distributed`` (never DistributedManager), so a plain
``init_process_group("gloo", init_method="file://...")`` inside a spawned
worker is a complete environment.

Everything the spawned workers touch (worker fns, stubs) must be
module-level — ``torch.multiprocessing.spawn`` pickles them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from eval_diffusion import BiasValidator, EnsembleEnvelopeValidator  # noqa: E402
from validate import Deterministic, ReplicateOnly  # noqa: E402
from validate_diffusion import DiffusionRolloutValidator  # noqa: E402

# Reuse the eval-suite stubs — same synthetic dataset in every process
# (internally seeded), which is what makes cross-process references exact.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_eval_diffusion import (  # noqa: E402
    _StubDataset,
    _StubWrapper,
)

pytestmark = pytest.mark.skipif(
    not dist.is_gloo_available(), reason="gloo backend unavailable"
)

_WORLD = 2
#: per-member additive offsets — the "ensemble" the stub scheduler generates.
_MEMBER_TABLE = [0.05, -0.15, 0.25, -0.35]


class _MemberOffsetScheduler:
    """Single-step scheduler whose members are distinct by GLOBAL index.

    ``sample`` adds ``table[rank * local_E + j]`` to member ``j`` of every
    IC. The union of members across ranks therefore reproduces exactly the
    members a single process generates with ``rank=0, local_E=E`` — the
    property the cross-rank mean/variance tests lean on. Fully
    deterministic: no RNG anywhere.
    """

    def __init__(self, rank: int, local_e: int, table):
        self.num_steps = 2
        self.rank = rank
        self.local_e = local_e
        self.table = list(table)

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        n_ic = x.shape[0] // self.local_e
        xm = x.view(n_ic, self.local_e, *x.shape[1:]).clone()
        for j in range(self.local_e):
            xm[:, j] += self.table[self.rank * self.local_e + j]
        return xm.view_as(x)


def _bias_kwargs(scheduler, ensemble_size, split, horizon=6):
    return dict(
        wrapper=_StubWrapper(),
        inference_scheduler=scheduler,
        horizon=horizon,
        device=torch.device("cpu"),
        max_initial_conditions=2,
        batch_size=2,
        ic_stride=1,
        ensemble_size=ensemble_size,
        perturber=ReplicateOnly(),
        split_ensemble_across_ranks=split,
    )


def _run_bias(rank: int, local_e: int, ensemble_size: int, split: bool):
    sched = _MemberOffsetScheduler(rank, local_e, _MEMBER_TABLE)
    v = BiasValidator(
        _StubDataset(), n_bins=3, steps_per_bin=2,
        **_bias_kwargs(sched, ensemble_size, split),
    )
    return v.run(nn.Identity(), epoch=0)


def _run_envelope(rank: int, local_e: int, ensemble_size: int, split: bool):
    sched = _MemberOffsetScheduler(rank, local_e, _MEMBER_TABLE)
    kwargs = _bias_kwargs(sched, ensemble_size, split, horizon=4)
    v = EnsembleEnvelopeValidator(_StubDataset(), **kwargs)
    return v.run(nn.Identity(), epoch=0)


# ---------------------------------------------------------------------------
# Spawned workers (module-level: spawn pickles them)
# ---------------------------------------------------------------------------


def _init_pg(rank: int, pg_file: str):
    dist.init_process_group(
        "gloo", init_method=f"file://{pg_file}", rank=rank, world_size=_WORLD
    )


def _worker_bias(rank: int, pg_file: str, out_dir: str, ensemble_size: int):
    _init_pg(rank, pg_file)
    try:
        result = _run_bias(
            rank, ensemble_size // _WORLD, ensemble_size, split=True
        )
        torch.save(result, Path(out_dir) / f"bias_rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _worker_envelope(rank: int, pg_file: str, out_dir: str, ensemble_size: int):
    _init_pg(rank, pg_file)
    try:
        result = _run_envelope(
            rank, ensemble_size // _WORLD, ensemble_size, split=True
        )
        torch.save(result, Path(out_dir) / f"env_rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _worker_divisibility(rank: int, pg_file: str, out_dir: str):
    _init_pg(rank, pg_file)
    try:
        with pytest.raises(ValueError, match="evenly"):
            _run_bias(rank, 1, ensemble_size=3, split=True)
        (Path(out_dir) / f"div_rank{rank}.ok").touch()
    finally:
        dist.destroy_process_group()


def _spawn(worker, tmp_path, *args):
    pg_file = str(tmp_path / "pg")
    mp.start_processes(
        worker,
        args=(pg_file, str(tmp_path), *args),
        nprocs=_WORLD,
        join=True,
        start_method="spawn",
    )


def _assert_results_close(split_result: dict, ref_result: dict):
    assert set(split_result) == set(ref_result)
    ref_rmse = ref_result["rmse_acc"]
    for k, v in split_result["rmse_acc"].items():
        assert v == pytest.approx(ref_rmse[k], rel=1e-5, abs=1e-6), k
    for key, ref_field in ref_result["climatology"].items():
        torch.testing.assert_close(
            split_result["climatology"][key], ref_field, rtol=1e-5, atol=1e-6
        )
    for group, ref_scalars in ref_result["global_bias"].items():
        torch.testing.assert_close(
            split_result["global_bias"][group], ref_scalars, rtol=1e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Single-process units (no process group)
# ---------------------------------------------------------------------------


def test_deterministic_perturber_with_real_ensemble_raises_at_ctor():
    """Silent zero-spread trap: under member-split with E == world_size each
    rank replicates by local_E == 1, so Deterministic's call-time guard would
    never fire and every 'member' would be identical."""
    with pytest.raises(ValueError, match="Deterministic"):
        BiasValidator(
            _StubDataset(), n_bins=3, steps_per_bin=2,
            **{**_bias_kwargs(_MemberOffsetScheduler(0, 2, _MEMBER_TABLE), 2, False),
               "perturber": Deterministic()},
        )


def test_ic_selection_not_rank_split_in_member_mode():
    v = BiasValidator(
        _StubDataset(), n_bins=3, steps_per_bin=2,
        **_bias_kwargs(_MemberOffsetScheduler(0, 2, _MEMBER_TABLE), 4, False),
    )
    # Simulate member-split bookkeeping without a process group.
    v.member_split = True
    assert v._select_ic_indices(0, 2) == v._select_ic_indices(1, 2)
    v.member_split = False
    r0 = v._select_ic_indices(0, 2)
    r1 = v._select_ic_indices(1, 2)
    assert not set(r0) & set(r1), "rank-modulo IC split must be preserved"


def test_generator_seed_rank_offset_only_when_member_split():
    v = BiasValidator(
        _StubDataset(), n_bins=3, steps_per_bin=2,
        **_bias_kwargs(_MemberOffsetScheduler(0, 2, _MEMBER_TABLE), 4, False),
    )
    v.member_split = False
    v._rank = 1
    base = v._generator_seed(epoch=3)
    v._rank = 0
    assert v._generator_seed(epoch=3) == base, "IC-split ranks share the seed"
    v.member_split = True
    v._rank = 1
    assert v._generator_seed(epoch=3) != base, "member-split ranks must differ"


def test_cross_rank_mean_is_identity_single_process():
    """world_size == 1: the shared method must be the old local reshape-mean."""
    v = DiffusionRolloutValidator(
        _StubDataset(),
        **_bias_kwargs(_MemberOffsetScheduler(0, 2, _MEMBER_TABLE), 4, False),
        log_steps=[1],
    )
    x = torch.randn(8, 5, 4, 4)          # 2 ICs x 4 members
    expected = x.view(2, 4, 5, 4, 4).mean(dim=1)
    torch.testing.assert_close(v._cross_rank_ensemble_mean(x), expected)
    v1 = DiffusionRolloutValidator(
        _StubDataset(),
        **_bias_kwargs(_MemberOffsetScheduler(0, 1, _MEMBER_TABLE), 1, False),
        log_steps=[1],
    )
    y = torch.randn(3, 5, 4, 4)
    assert v1._cross_rank_ensemble_mean(y) is y, "E=1 non-split is the identity"


# ---------------------------------------------------------------------------
# 2-process gloo tests
# ---------------------------------------------------------------------------


def test_member_split_bias_matches_single_process_reference(tmp_path):
    """E=4 over 2 ranks == E=4 on one process, for every bias-suite output.

    The union of the ranks' members reproduces the single-process member set
    exactly (offset table indexed by global member id), so every metric —
    per-step RMSE, bias maps, lat-weighted global bias — must agree to float
    tolerance (mean-of-means reassociates the reduction order).
    """
    _spawn(_worker_bias, tmp_path, 4)
    split = torch.load(tmp_path / "bias_rank0.pt", weights_only=False)
    ref = _run_bias(rank=0, local_e=4, ensemble_size=4, split=False)
    _assert_results_close(split, ref)
    # Both ranks scored identical (cross-rank-reduced) statistics.
    split1 = torch.load(tmp_path / "bias_rank1.pt", weights_only=False)
    _assert_results_close(split1, ref)


@pytest.mark.parametrize("ensemble_size", [4, 2])
def test_member_split_envelope_matches_single_process_reference(
    tmp_path, ensemble_size
):
    """Spread over the member UNION — including E=2/world=2 (local_E == 1),
    which exercises the clone-before-all-reduce path and the variance path
    with no local member axis at all."""
    _spawn(_worker_envelope, tmp_path, ensemble_size)
    split = torch.load(tmp_path / "env_rank0.pt", weights_only=False)
    ref = _run_envelope(
        rank=0, local_e=ensemble_size, ensemble_size=ensemble_size, split=False
    )
    spread_keys = [k for k in ref if k.startswith(("spread_", "spread_skill_"))]
    assert spread_keys, "envelope reference produced no spread metrics"
    for k in spread_keys:
        assert split[k] == pytest.approx(ref[k], rel=1e-5, abs=1e-6), k


def test_member_split_requires_even_division(tmp_path):
    _spawn(_worker_divisibility, tmp_path)
    for r in range(_WORLD):
        assert (tmp_path / f"div_rank{r}.ok").exists()
