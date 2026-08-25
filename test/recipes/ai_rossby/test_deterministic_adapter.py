# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic-model adapters for the fused climate eval suite.

The seam between the families is one call: the validators step models via
``scheduler.sample(model, x, c_grid, c_scalar, num_steps)`` on packed flat
tensors; the deterministic train.py families share one positional forward
``model(surface, const, varying, upper, **extras) -> tuple``. These tests pin
the shim/adapter that closes it, with a stub model honoring exactly that
contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from climate_eval_suite import VariableCatalog  # noqa: E402
from deterministic_adapter import (  # noqa: E402
    DeterministicPackShim,
    DeterministicStepAdapter,
)
from validate_diffusion import DiffusionRolloutValidator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_eval_diffusion import (  # noqa: E402
    _LEVELS,
    _SURFACE_VARS,
    _UPPER_VARS,
    _H,
    _W,
    _StubDataset,
)

_N_CONST, _N_VARY = 1, 1


class _PositionalTupleModel(nn.Module):
    """The deterministic families' contract: positional args, tuple out.

    Records every call's kwargs so tests can assert prev-state / calendar
    threading. Output = input + 0.5 (surface), + 0.25 (upper air), plus an
    optional diagnostic head.
    """

    def __init__(self, *, emit_diagnostic=False, optional_names=()):
        super().__init__()
        self.emit_diagnostic = emit_diagnostic
        self.calls: list[dict] = []
        # Build a forward whose co_varnames advertise the requested optional
        # kwargs — _model_optional_kwarg_names inspects the code object.
        self._optional = set(optional_names)

    def forward(self, surface_in, constant_boundary, varying_boundary,
                upper_air_in, surface_prev_in=None, upper_air_prev_in=None,
                calendar=None):
        self.calls.append(
            dict(
                surface_prev_in=surface_prev_in,
                upper_air_prev_in=upper_air_prev_in,
                calendar=calendar,
                const_shape=tuple(constant_boundary.shape),
                varying_shape=tuple(varying_boundary.shape),
            )
        )
        out = [surface_in + 0.5, upper_air_in + 0.25]
        if self.emit_diagnostic:
            out.append(surface_in.mean(dim=-3, keepdim=True).expand(
                *surface_in.shape[:-3], 1, *surface_in.shape[-2:]
            ))
        out += [torch.zeros(1)] * 2
        return tuple(out)


def _catalog(with_diag=False):
    return VariableCatalog(
        surface=list(_SURFACE_VARS),
        upper_air=list(_UPPER_VARS),
        diagnostic=["diag0"] if with_diag else [],
        levels=list(_LEVELS),
    )


def _shim(model=None, with_diag=False):
    return DeterministicPackShim(
        model or _PositionalTupleModel(emit_diagnostic=with_diag),
        catalog=_catalog(with_diag),
        n_constant=_N_CONST,
        n_varying=_N_VARY,
        has_diagnostic=with_diag,
    )


def test_shim_pack_unpack_roundtrip_including_diagnostic():
    shim = _shim(with_diag=True)
    sample = {
        "surface_in": torch.randn(3, len(_SURFACE_VARS), _H, _W),
        "upper_air_in": torch.randn(3, len(_UPPER_VARS), len(_LEVELS), _H, _W),
        "diagnostic": torch.randn(3, 1, _H, _W),
    }
    x = shim.pack_state(sample)
    assert x.shape == (3, len(_SURFACE_VARS) + len(_UPPER_VARS) * len(_LEVELS) + 1, _H, _W)
    out = shim.unpack_state(x)
    torch.testing.assert_close(out["surface_in"], sample["surface_in"])
    torch.testing.assert_close(out["upper_air_in"], sample["upper_air_in"])
    torch.testing.assert_close(out["diagnostic"], sample["diagnostic"])


def test_shim_pads_missing_diagnostic_with_zeros():
    """The IC carries no diagnostic (it is a model OUTPUT); the packed width
    must still be constant across the rollout."""
    shim = _shim(with_diag=True)
    sample = {
        "surface_in": torch.randn(2, len(_SURFACE_VARS), _H, _W),
        "upper_air_in": torch.randn(2, len(_UPPER_VARS), len(_LEVELS), _H, _W),
    }
    x = shim.pack_state(sample)
    assert (shim.unpack_state(x)["diagnostic"] == 0).all()


def test_shim_split_c_grid_widths():
    shim = _shim()
    sample = {
        "surface_in": torch.randn(2, len(_SURFACE_VARS), _H, _W),
        "constant_boundary": torch.randn(_N_CONST, _H, _W),
        "varying_boundary": torch.randn(2, _N_VARY, _H, _W),
    }
    c_grid = shim.pack_c_grid(sample)
    const, varying = shim.split_c_grid(c_grid)
    assert const.shape == (2, _N_CONST, _H, _W)
    assert varying.shape == (2, _N_VARY, _H, _W)
    torch.testing.assert_close(varying, sample["varying_boundary"])


def test_adapter_prev_state_semantics_and_reset():
    """k=1: no prev kwargs (the model's own persistence fallback); k>=2: the
    TRUE k-1 input; on_rollout_start resets across IC batches."""
    model = _PositionalTupleModel()
    shim = _shim(model)
    adapter = DeterministicStepAdapter(
        shim, optional_kwargs={"surface_prev_in", "upper_air_prev_in"}
    )
    s0 = torch.randn(2, len(_SURFACE_VARS), _H, _W)
    u0 = torch.randn(2, len(_UPPER_VARS), len(_LEVELS), _H, _W)
    x = shim.pack_state({"surface_in": s0, "upper_air_in": u0})
    c_grid = torch.randn(2, _N_CONST + _N_VARY, _H, _W)

    adapter.on_rollout_start({})
    x1 = adapter.sample(model, x, c_grid, None)
    assert model.calls[-1]["surface_prev_in"] is None
    adapter.sample(model, x1, c_grid, None)
    torch.testing.assert_close(model.calls[-1]["surface_prev_in"], s0)
    torch.testing.assert_close(model.calls[-1]["upper_air_prev_in"], u0)

    adapter.on_rollout_start({})          # new IC batch: memory must clear
    adapter.sample(model, x, c_grid, None)
    assert model.calls[-1]["surface_prev_in"] is None


def test_adapter_threads_calendar_iff_named():
    model = _PositionalTupleModel()
    shim = _shim(model)
    cal = torch.randn(2, 2)
    x = shim.pack_state({
        "surface_in": torch.randn(2, len(_SURFACE_VARS), _H, _W),
        "upper_air_in": torch.randn(2, len(_UPPER_VARS), len(_LEVELS), _H, _W),
    })
    c_grid = torch.randn(2, _N_CONST + _N_VARY, _H, _W)

    with_cal = DeterministicStepAdapter(shim, optional_kwargs={"calendar"})
    with_cal.sample(model, x, c_grid, cal)
    torch.testing.assert_close(model.calls[-1]["calendar"], cal)

    without = DeterministicStepAdapter(shim, optional_kwargs=set())
    without.sample(model, x, c_grid, cal)
    assert model.calls[-1]["calendar"] is None


def test_adapter_has_no_sample_rollout():
    """The drive dispatches window-vs-single-step on this attribute — the
    adapter must take the single-step autoregressive path."""
    adapter = DeterministicStepAdapter(_shim(), optional_kwargs=set())
    assert not hasattr(adapter, "sample_rollout")


def test_deterministic_model_end_to_end_through_the_drive():
    """A deterministic model rides the full fused-eval machinery: rollout,
    scoring, and the per-step forward count (one forward per frame)."""
    model = _PositionalTupleModel()
    shim = _shim(model)
    adapter = DeterministicStepAdapter(shim, optional_kwargs=set())
    drive = DiffusionRolloutValidator(
        _StubDataset(),
        wrapper=shim,
        inference_scheduler=adapter,
        log_steps=[1, 2, 3, 4],
        horizon=4,
        device=torch.device("cpu"),
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
    )
    assert drive.window_mode is False
    rmse_acc = drive.run(shim, epoch=0)
    assert len(model.calls) == 4          # one deterministic forward per frame
    rmse_keys = [k for k in rmse_acc if k.startswith("rmse_step")]
    assert len(rmse_keys) == 8            # 4 steps x {surface, upper_air}
    assert all(v > 0 for k, v in rmse_acc.items() if k.startswith("rmse_step"))


def test_family_dispatch_signal_covers_rolling_wrappers():
    """The dispatch must treat BOTH diffusion pack surfaces as diffusion:
    single-step wrappers expose pack_state, but the rolling family
    (RollingDiTWrapper/ERDMWrapper) exposes only pack_window_state —
    inference.py's pack_state-only signal misclassified the RSI fancy
    checkpoint as deterministic (Midway job 54834641)."""

    class _RollingLike:
        def pack_window_state(self):
            ...

    class _SingleStepLike:
        def pack_state(self):
            ...

    class _DeterministicLike:
        def forward(self):
            ...

    def is_diffusion(obj):
        return hasattr(obj, "pack_state") or hasattr(obj, "pack_window_state")

    assert is_diffusion(_RollingLike())
    assert is_diffusion(_SingleStepLike())
    assert not is_diffusion(_DeterministicLike())
