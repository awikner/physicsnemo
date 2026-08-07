# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12b regression tests for train_diffusion's rolling-window path.

The window branch of ``_build_loader`` / ``_pack_window`` had never been
executed end-to-end before the Phase 12b Delta smoke (job 20917614):
``SequenceDataset`` was constructed with kwargs it doesn't accept, and
``_pack_window`` read bare keys where the dataset emits ``{key}_seq``
stacks (with ``calendar`` missing from the stack list entirely). These
tests drive the real loader + pack path on a synthetic base dataset so
the wiring can never silently regress again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_diffusion import _build_loader, _pack_window  # noqa: E402

from physicsnemo.experimental.diffusion import ERDMScheduler  # noqa: E402
from physicsnemo.experimental.models.amip_si import RollingDiTWrapper  # noqa: E402

_SURFACE, _UA, _DIAG, _VARY, _CONST = 3, 2, 2, 3, 2
_LEVELS = [100.0, 500.0, 850.0]
_H, _W = 8, 16
_N_TIME = 16


class _SyntheticBase:
    """Minimal ClimateZarrDataset stand-in for SequenceDataset."""

    n_time = _N_TIME
    layout = None

    def __getitem__(self, index):
        t, _lead = index if isinstance(index, tuple) else (index, 1)
        g = torch.Generator().manual_seed(int(t))
        return {
            "surface_in": torch.randn(_SURFACE, _H, _W, generator=g),
            "upper_air_in": torch.randn(_UA, len(_LEVELS), _H, _W, generator=g),
            "diagnostic": torch.randn(_DIAG, _H, _W, generator=g),
            "varying_boundary": torch.randn(_VARY, _H, _W, generator=g),
            "constant_boundary": torch.full((_CONST, _H, _W), 7.0),
            "calendar": torch.randn(2, generator=g),
        }

    def __len__(self):
        return self.n_time


def _cfg(batch_size=2, shuffle=False, num_workers=0):
    return OmegaConf.create(
        {
            "seed": 0,
            "dataset": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": 2,
                "persistent_workers": False,
                "pin_memory": False,
                "shuffle": shuffle,
                "forecast_lead_times": [1],
            },
        }
    )


def _wrapper() -> RollingDiTWrapper:
    return RollingDiTWrapper(
        surface_variables=[f"s{i}" for i in range(_SURFACE)],
        upper_air_variables=[f"u{i}" for i in range(_UA)],
        diagnostic_variables=[f"d{i}" for i in range(_DIAG)],
        constant_boundary_variables=[f"c{i}" for i in range(_CONST)],
        varying_boundary_variables=[f"v{i}" for i in range(_VARY)],
        levels=list(_LEVELS),
        horizontal_resolution=(_H, _W),
        channel_layout="v2",
        rolling_dit_kwargs=dict(dim=16, num_heads=2, num_blocks=1),
    )


def test_build_loader_window_mode_constructs_and_batches():
    W = 3
    loader, window_mode = _build_loader(
        _cfg(), _SyntheticBase(), window_size=W, rank=0
    )
    assert window_mode
    batch = next(iter(loader))
    B = 2
    assert batch["surface_in_seq"].shape == (B, W, _SURFACE, _H, _W)
    assert batch["upper_air_in_seq"].shape == (B, W, _UA, len(_LEVELS), _H, _W)
    assert batch["varying_boundary_seq"].shape == (B, W, _VARY, _H, _W)
    assert batch["diagnostic_seq"].shape == (B, W, _DIAG, _H, _W)
    assert batch["calendar_seq"].shape == (B, W, 2)
    # Constant boundary stays unstacked (one frame; expanded in _pack_window).
    assert batch["constant_boundary"].shape == (B, _CONST, _H, _W)


def test_build_loader_single_step_mode_unchanged():
    loader, window_mode = _build_loader(
        _cfg(), _SyntheticBase(), window_size=1, rank=0
    )
    assert not window_mode
    batch = next(iter(loader))
    assert "surface_in" in batch and "surface_in_seq" not in batch


def test_build_loader_ddp_window_sampler_partitions_disjointly():
    W = 3
    cfg = _cfg(batch_size=1)
    seen: list[set] = []
    for rank in range(2):
        loader, _ = _build_loader(
            cfg, _SyntheticBase(), window_size=W, rank=rank, world_size=2
        )
        assert isinstance(loader.sampler, torch.utils.data.DistributedSampler)
        seen.append({int(b["start_idx"]) for b in loader})
    assert seen[0].isdisjoint(seen[1])
    assert len(seen[0]) == len(seen[1]) > 0


def test_pack_window_feeds_scheduler_finite_loss():
    W = 3
    loader, _ = _build_loader(_cfg(), _SyntheticBase(), window_size=W, rank=0)
    batch = next(iter(loader))
    model = _wrapper().eval()
    y, c_grid, c_scalar = _pack_window(model, batch)
    B = 2
    n_state = _SURFACE + _DIAG + _UA * len(_LEVELS)
    assert y.shape == (B, W, n_state, _H, _W)
    assert c_grid.shape == (B, W, _CONST + _VARY, _H, _W)
    assert c_scalar.shape == (B, W, 2)
    # Constant boundary must be identical across the window axis.
    # v2 c_grid order = [varying | constant] -> constants are the tail.
    assert torch.equal(c_grid[:, 0, _VARY:], c_grid[:, -1, _VARY:])
    sched = ERDMScheduler(window_size=W, num_steps=2, noise="gaussian", sigma_data=1.0)
    torch.manual_seed(0)
    loss = sched.compute_loss(model, c_grid, c_scalar, y)
    assert torch.isfinite(loss)


def test_pack_window_rejects_missing_calendar():
    # Guard against the pre-12b silent contract: the dataset must emit
    # calendar for the rolling path (emit_calendar=True).
    W = 3
    loader, _ = _build_loader(_cfg(), _SyntheticBase(), window_size=W, rank=0)
    batch = next(iter(loader))
    del batch["calendar_seq"]
    with pytest.raises(KeyError, match="calendar_seq"):
        _pack_window(_wrapper(), batch)
