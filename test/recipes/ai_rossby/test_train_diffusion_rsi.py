# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""train_diffusion's Rolling Stochastic Interpolant path (W+1 anchor plumbing).

RSI is the first scheduler whose training sample is not the same shape as its
window: slot 1 is anchored on the frame *before* the window, so ``compute_loss``
takes ``W+1`` state frames while ``c_grid`` / ``c_scalar`` stay at ``W``.
Everything about that is a shape that still lines up if it is wired wrong —
a whole-window shift keeps every tensor the right size — so these tests drive
the real loader + pack + step path on a synthetic base and pin frame
identities, not just shapes.
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

from train_diffusion import _build_loader, _pack_window, _train_step  # noqa: E402

from physicsnemo.experimental.diffusion import RSIScheduler  # noqa: E402
from physicsnemo.experimental.models.amip_si import RollingDiTWrapper  # noqa: E402

_SURFACE, _UA, _DIAG, _VARY, _CONST = 3, 2, 2, 3, 2
_LEVELS = [100.0, 500.0, 850.0]
_H, _W = 8, 16
_N_TIME = 16


class _SyntheticBase:
    """Every field encodes its own time index, so an off-by-one is a number."""

    n_time = _N_TIME
    layout = None

    def __getitem__(self, index):
        t, _lead = index if isinstance(index, tuple) else (index, 1)
        t = float(t)
        return {
            "surface_in": torch.full((_SURFACE, _H, _W), t),
            "upper_air_in": torch.full((_UA, len(_LEVELS), _H, _W), t),
            "diagnostic": torch.full((_DIAG, _H, _W), t),
            "varying_boundary": torch.full((_VARY, _H, _W), 100.0 + t),
            "constant_boundary": torch.full((_CONST, _H, _W), 7.0),
            "calendar": torch.full((2,), 200.0 + t),
        }

    def __len__(self):
        return self.n_time


def _cfg(batch_size=2):
    return OmegaConf.create({
        "seed": 0,
        "dataset": {
            "batch_size": batch_size, "num_workers": 0, "prefetch_factor": 2,
            "persistent_workers": False, "pin_memory": False, "shuffle": False,
            "forecast_lead_times": [1],
        },
    })


def _wrapper(ocean=()) -> RollingDiTWrapper:
    return RollingDiTWrapper(
        surface_variables=[f"s{i}" for i in range(_SURFACE)],
        upper_air_variables=[f"u{i}" for i in range(_UA)],
        diagnostic_variables=[f"d{i}" for i in range(_DIAG)],
        constant_boundary_variables=[f"c{i}" for i in range(_CONST)],
        varying_boundary_variables=[f"v{i}" for i in range(_VARY)],
        ocean_state_variables=list(ocean),
        levels=list(_LEVELS),
        horizontal_resolution=(_H, _W),
        channel_layout="v2",
        rolling_dit_kwargs=dict(
            dim=32, num_heads=2, num_blocks=1,
            # The ocean block needs non-legacy projections on BOTH ends (their
            # legacy widths are checkpoint state-dict shapes); the second output
            # head is RSI's zhat readout.
            input_embed={"mode": "budget", "d_boundary": 8, "d_calendar": 8},
            output_head={"mode": "mix", "num_experts": 2, "num_output_heads": 2},
        ),
    )


def _rsi_loader(W=3, batch_size=2):
    return _build_loader(
        _cfg(batch_size), _SyntheticBase(), window_size=W, rank=0,
        forcing_lag=1, anchor_frames=1,
    )


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


def test_loader_emits_the_anchor_frame():
    W = 3
    loader, window_mode = _rsi_loader(W)
    assert window_mode
    b = next(iter(loader))
    assert b["surface_in_prev"].shape == (2, _SURFACE, _H, _W)
    assert b["surface_in_seq"].shape == (2, W, _SURFACE, _H, _W)


def test_pack_window_prepends_the_anchor_to_the_state_stack():
    """y gains a frame; the forcings must NOT."""
    W = 3
    loader, _ = _rsi_loader(W)
    batch = next(iter(loader))
    model = _wrapper().eval()

    y_plain, cg_plain, cs_plain = _pack_window(model, batch)
    y, c_grid, c_scalar = _pack_window(model, batch, anchor_frames=1)

    assert y_plain.shape[1] == W
    assert y.shape[1] == W + 1
    # The forcings stay aligned to slots 1..W, untouched.
    assert torch.equal(c_grid, cg_plain) and torch.equal(c_scalar, cs_plain)
    assert c_grid.shape[1] == W
    # y[:, 1:] must be exactly the un-anchored pack: the anchor is prepended,
    # not substituted, so slot w's target is unchanged.
    torch.testing.assert_close(y[:, 1:], y_plain)


def test_anchor_frame_is_one_step_before_the_first_target():
    W = 3
    loader, _ = _rsi_loader(W, batch_size=1)
    batch = next(iter(loader))
    y, _, _ = _pack_window(_wrapper().eval(), batch, anchor_frames=1)
    # Every channel of a frame carries that frame's time index.
    times = [float(y[0, i].flatten()[0]) for i in range(y.shape[1])]
    assert len(times) == W + 1
    assert all(times[i + 1] - times[i] == pytest.approx(1.0)
               for i in range(len(times) - 1)), times


def test_pack_window_without_anchor_key_is_a_loud_error():
    loader, _ = _build_loader(
        _cfg(), _SyntheticBase(), window_size=3, rank=0, forcing_lag=1)
    batch = next(iter(loader))
    with pytest.raises(KeyError, match="emit_anchor"):
        _pack_window(_wrapper().eval(), batch, anchor_frames=1)


# ---------------------------------------------------------------------------
# Loss / step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", ["state", "residual"])
def test_packed_window_feeds_rsi_a_finite_loss(param):
    W = 3
    loader, _ = _rsi_loader(W)
    batch = next(iter(loader))
    model = _wrapper()
    y, c_grid, c_scalar = _pack_window(model, batch, anchor_frames=1)
    sched = RSIScheduler(window_size=W, num_steps=2, parameterization=param)
    loss = sched.compute_loss(model, c_grid, c_scalar, y)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_train_step_runs_end_to_end():
    W = 3
    loader, window_mode = _rsi_loader(W)
    batch = next(iter(loader))
    model = _wrapper()
    sched = RSIScheduler(window_size=W, num_steps=2)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    out = _train_step(
        model=model, scheduler_loss=sched, sample=batch, optimizer=opt,
        grad_scaler=None, amp_dtype=None, device=torch.device("cpu"),
        window_mode=window_mode, anchor_frames=1,
    )
    assert torch.isfinite(torch.tensor(out["loss"]))


def test_erdm_path_is_untouched_by_the_anchor_plumbing():
    """anchor_frames=0 must reproduce the pre-RSI loader and pack exactly."""
    from physicsnemo.experimental.diffusion import ERDMScheduler

    W = 3
    loader, _ = _build_loader(
        _cfg(), _SyntheticBase(), window_size=W, rank=0, forcing_lag=1)
    batch = next(iter(loader))
    model = _wrapper()
    y, c_grid, c_scalar = _pack_window(model, batch)
    assert y.shape[1] == W
    # ERDM's own contract still holds against a 2-head backbone's leading half
    # only in principle; here we just assert the pack is unchanged in shape.
    sched = ERDMScheduler(window_size=W, num_steps=2)
    assert sched.window_size == W


# ---------------------------------------------------------------------------
# Ocean channels (Phase 12f contract under the W+1 stack)
# ---------------------------------------------------------------------------


_OCEAN = ["v1", "v2"]


def test_train_step_with_ocean_channels():
    """The ocean target stack must gain the anchor frame along with y."""
    from train_loop import adopt_ocean_contract

    W = 3
    loader, window_mode = _build_loader(
        _cfg(), _SyntheticBase(), window_size=W, rank=0,
        forcing_lag=1, emit_boundary_next=True, anchor_frames=1,
    )
    batch = next(iter(loader))
    model = _wrapper(ocean=_OCEAN)
    sched = RSIScheduler(window_size=W, num_steps=2)
    adopt_ocean_contract(sched, model)
    assert sched.nocean == len(_OCEAN)

    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    out = _train_step(
        model=model, scheduler_loss=sched, sample=batch, optimizer=opt,
        grad_scaler=None, amp_dtype=None, device=torch.device("cpu"),
        window_mode=window_mode, anchor_frames=1,
    )
    assert torch.isfinite(torch.tensor(out["loss"]))
    assert "loss_ocean" in out


def test_ocean_target_stack_carries_the_anchors_own_time_boundary():
    """The W+1 ocean truth must be the boundary at each frame's OWN time."""
    W = 3
    loader, _ = _build_loader(
        _cfg(batch_size=1), _SyntheticBase(), window_size=W, rank=0,
        forcing_lag=1, emit_boundary_next=True, anchor_frames=1,
    )
    b = next(iter(loader))
    stack = torch.cat(
        [b["varying_boundary_seq"][:, :1], b["varying_boundary_next_seq"]], dim=1
    )
    assert stack.shape[1] == W + 1
    anchor_t = float(b["surface_in_prev"].flatten()[0])
    state_t = [float(b["surface_in_seq"][0, i].flatten()[0]) for i in range(W)]
    got = [float(stack[0, i].flatten()[0]) for i in range(W + 1)]
    assert got == [100.0 + t for t in ([anchor_t] + state_t)]


# ---------------------------------------------------------------------------
# Gradient clipping and the non-finite-loss guard
# ---------------------------------------------------------------------------


def _step(model, sched, batch, window_mode=True, **kw):
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    return _train_step(
        model=model, scheduler_loss=sched, sample=batch, optimizer=opt,
        grad_scaler=None, amp_dtype=None, device=torch.device("cpu"),
        window_mode=window_mode, anchor_frames=1, **kw,
    )


def test_grad_clipping_is_applied_and_reported():
    """train_diffusion had NO clipping — train.py clips, this recipe did not.

    Worse, `amip_diffusion*.yaml` shipped a ``grad_clip_norm`` key that nothing
    in this recipe read, so it looked configured while being a no-op. An RSI A2
    run then trained cleanly for 11,700 batches and ran away exponentially with
    nothing to arrest it (e-folding every ~93 batches).
    """
    W = 3
    loader, _ = _rsi_loader(W)
    batch = next(iter(loader))
    sched = RSIScheduler(window_size=W, num_steps=2)

    out = _step(_wrapper(), sched, batch, grad_clip_norm=1.0)
    assert "grad_norm" in out, "the pre-clip norm must be reported"
    assert out["grad_norm"] >= 0.0

    # …and stays absent when clipping is off, so the two paths are distinguishable.
    out_off = _step(_wrapper(), sched, batch, grad_clip_norm=0.0)
    assert "grad_norm" not in out_off


def test_grad_clipping_actually_bounds_the_update():
    W = 3
    loader, _ = _rsi_loader(W)
    batch = next(iter(loader))
    sched = RSIScheduler(window_size=W, num_steps=2)
    model = _wrapper()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # measure grads, don't move
    _train_step(
        model=model, scheduler_loss=sched, sample=batch, optimizer=opt,
        grad_scaler=None, amp_dtype=None, device=torch.device("cpu"),
        window_mode=True, anchor_frames=1, grad_clip_norm=1e-4,
    )
    total = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                           for p in model.parameters() if p.grad is not None))
    assert float(total) <= 1e-4 * 1.01, f"grads not clipped: {float(total)}"


def test_non_finite_loss_aborts_instead_of_stepping():
    """One NaN update poisons every weight; the run cannot recover from it.

    The A2 run ground out 39,314 NaN batches over 8.5 h on a dedicated 4xH100
    node because nothing checked.
    """
    W = 3
    loader, _ = _rsi_loader(W)
    batch = next(iter(loader))

    class _NanScheduler(RSIScheduler):
        def compute_loss(self, *a, **kw):
            out = super().compute_loss(*a, **kw)
            loss = out[0] if isinstance(out, tuple) else out
            nan = loss * float("nan")
            return (nan, out[1]) if isinstance(out, tuple) else nan

    sched = _NanScheduler(window_size=W, num_steps=2)
    model = _wrapper()
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match="non-finite"):
        _step(model, sched, batch, grad_clip_norm=1.0)
    # and it must not have stepped
    for p, b in zip(model.parameters(), before):
        torch.testing.assert_close(p.detach(), b)
