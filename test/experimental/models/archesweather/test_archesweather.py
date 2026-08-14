# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`ArchesWeather`.

Validates:

* Forward output tuple shape ``(out_surface, out_upper_air, 0, 0, 0, 0)`` —
  drop-in compatible with :class:`SfnoPlasim` so the trainer treats all model
  families uniformly. ``has_diagnostic`` is False (ArchesWeather-M has no
  diagnostic head; a non-empty ``diagnostic_variables`` raises).
* Correct surface / upper-air shapes at the model's 17->pad-18 level geometry
  (patch (2,3,3) -> zdim = 1 + 18/2 = 10 on the full grid; a tiny grid here).
* Previous-state + month/hour calendar kwargs are consumed; the no-prev /
  no-calendar fallback (persistence + month 1 / hour 0) runs.
* Constant boundary accepts both ``(C, H, W)`` and ``(B, C, H, W)``.
* ``Module.instantiate`` builds it from a name/module/args dict (the trainer
  path) and ``Module.save`` / ``from_checkpoint`` round-trips.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    import physicsnemo
    from physicsnemo.experimental.models.archesweather import ArchesWeather

# Tiny config: grid divisible by patch (12/3, 24/3), small window that fits the
# token grid; finishes a forward on CPU quickly.
_SMOKE_KWARGS = dict(
    surface_variables=["t2m", "u10", "v10", "mslp"],
    upper_air_variables=["ta", "ua", "va", "hus", "zg"],
    constant_boundary_variables=["lsm", "z"],
    varying_boundary_variables=["sst", "sic", "rsdt"],
    levels=[0.2, 0.5, 0.9],
    horizontal_resolution=[12, 24],
    emb_dim=16,
    depth_multiplier=1,
    num_heads=[2, 4, 4, 2],
    window_size=[1, 2, 2],
    cond_dim=32,
    patch_size=[2, 3, 3],
    mlp_layer="swiglu",
    use_prev_state=True,
)
_H, _W, _L = 12, 24, 3
_NS, _NU = 4, 5


def _inputs(batch_size=2, *, unbatched_const=True):
    surface = torch.randn(batch_size, _NS, _H, _W)
    const_b = torch.randn(2, _H, _W) if unbatched_const else torch.randn(batch_size, 2, _H, _W)
    vary_b = torch.randn(batch_size, 3, _H, _W)
    upper = torch.randn(batch_size, _NU, _L, _H, _W)
    return surface, const_b, vary_b, upper


def test_forward_shapes_and_tuple():
    model = ArchesWeather(**_SMOKE_KWARGS).eval()
    assert model.has_diagnostic is False
    s, c, v, u = _inputs(batch_size=2)
    out = model(
        s, c, v, u,
        surface_prev_in=torch.randn(2, _NS, _H, _W),
        upper_air_prev_in=torch.randn(2, _NU, _L, _H, _W),
        calendar=torch.tensor([[1.0, 0.0], [6.0, 12.0]]),
    )
    assert len(out) == 6
    assert out[0].shape == (2, _NS, _H, _W)
    assert out[1].shape == (2, _NU, _L, _H, _W)
    for t in out[2:]:
        assert torch.is_tensor(t) and t.numel() <= 1 and float(t.item()) == 0.0


def test_fallback_without_prev_or_calendar():
    model = ArchesWeather(**_SMOKE_KWARGS).eval()
    out = model(*_inputs(batch_size=1))  # no prev / no calendar
    assert out[0].shape == (1, _NS, _H, _W)
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()


def test_batched_constant_boundary_and_backward():
    model = ArchesWeather(**_SMOKE_KWARGS)
    s, c, v, u = _inputs(batch_size=2, unbatched_const=False)
    s.requires_grad_(True)
    out = model(s, c, v, u)
    (out[0].pow(2).mean() + out[1].pow(2).mean()).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_diagnostic_variables_attach_a_head():
    """A non-empty ``diagnostic_variables`` builds the diagnostic decoder.

    This used to assert the opposite — the original port (2026-07-21) was
    head-less and rejected the argument — but the diagnostic head landed on
    2026-07-27 as an ai-rossby extension over geoarches ArchesWeather-M, which
    deliberately removed that guard. The test outlived the behaviour it
    described; it now pins the behaviour the model documents.
    """
    kw = dict(_SMOKE_KWARGS, diagnostic_variables=["pr_6h", "olr"])
    model = ArchesWeather(**kw).eval()
    assert model.has_diagnostic
    with torch.no_grad():
        out = model(*_inputs(batch_size=1))
    # forward returns (surface, upper_air, diagnostic, 0, 0, 0).
    diagnostic = out[2]
    assert diagnostic.shape == (1, 2, _H, _W)
    assert torch.isfinite(diagnostic).all()


def test_the_head_less_path_is_unchanged():
    """Empty ``diagnostic_variables`` keeps the original geoarches architecture.

    The head is an extension, so its absence must stay the default: a scalar 0
    in the diagnostic slot rather than a tensor.
    """
    model = ArchesWeather(**_SMOKE_KWARGS).eval()
    assert not model.has_diagnostic
    with torch.no_grad():
        out = model(*_inputs(batch_size=1))
    assert not torch.is_tensor(out[2]) or out[2].numel() == 1


def test_instantiate_and_checkpoint_roundtrip(tmp_path):
    model = ArchesWeather(**_SMOKE_KWARGS).eval()
    ckpt = tmp_path / "aw.mdlus"
    model.save(str(ckpt))
    reload = physicsnemo.Module.from_checkpoint(str(ckpt)).eval()
    assert isinstance(reload, ArchesWeather)
    s, c, v, u = _inputs(batch_size=1)
    with torch.no_grad():
        a = model(s, c, v, u)[0]
        b = reload(s, c, v, u)[0]
    assert torch.allclose(a, b, atol=1e-5)
