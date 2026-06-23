# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8a backbone unit tests — forward-shape contract for DiT,
RollingDiT, and ERDM.

These are CPU-runnable smoke tests on tiny tensors (≤32×64 grid, ≤8
channels). They verify:

* Each backbone instantiates under :class:`physicsnemo.Module`.
* Forward returns a tensor with the documented shape.
* Output is finite (no NaN / inf) under random init.
* ``Module.from_checkpoint`` round-trips the constructor kwargs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    import physicsnemo
    from physicsnemo.experimental.models.amip_si import AmipDiT, ERDM, RollingDiT


# ---------------------------------------------------------------------------
# DiT (single-step, [B, C, H, W] forward)
# ---------------------------------------------------------------------------


def _dit_kwargs() -> dict:
    return dict(
        in_channels=4,
        out_channels=4,
        dim=64,
        num_heads=4,
        num_blocks=2,
        patch_size=2,
        nlat=16,
        nlon=32,
        scalar_dim=2,
        c_grid_dim=0,
    )


def test_dit_forward_shape_finite():
    torch.manual_seed(0)
    model = AmipDiT(**_dit_kwargs()).eval()
    b, c, h, w = 2, 4, 16, 32
    x_noised = torch.randn(b, c, h, w)
    cond = torch.randn(b, c, h, w)
    t = torch.rand(b)  # [b]
    c_scalar = torch.randn(b, 2)
    with torch.no_grad():
        out = model(x_noised, cond, t, c_grid=None, c_scalar=c_scalar)
    assert out.shape == (b, c, h, w), out.shape
    assert torch.isfinite(out).all()


def test_dit_from_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = AmipDiT(**_dit_kwargs())
    p = tmp_path / "dit.mdlus"
    model.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, AmipDiT)
    assert loaded.meta.cuda_graphs is False  # Phase 8a default; flips in 8f


# ---------------------------------------------------------------------------
# RollingDiT (rolling-window, [B, W, C, H, W'] forward)
# ---------------------------------------------------------------------------


def _rolling_dit_kwargs() -> dict:
    return dict(
        in_channels=4,
        dim=64,
        num_heads=4,
        temporal_num_heads=4,
        num_blocks=2,
        nlat=16,
        nlon=32,
        scalar_dim=2,
        c_grid_dim=0,
    )


def test_rolling_dit_forward_shape_finite():
    torch.manual_seed(0)
    model = RollingDiT(**_rolling_dit_kwargs()).eval()
    b, W, c, h, w = 1, 3, 4, 16, 32
    z = torch.randn(b, W, c, h, w)
    t = torch.rand(b, W)
    c_scalar = torch.randn(b, W, 2)
    with torch.no_grad():
        out = model(z, t, c_grid=None, c_scalar=c_scalar)
    assert out.shape == (b, W, c, h, w), out.shape
    assert torch.isfinite(out).all()


def test_rolling_dit_from_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = RollingDiT(**_rolling_dit_kwargs())
    p = tmp_path / "rolling_dit.mdlus"
    model.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, RollingDiT)


# ---------------------------------------------------------------------------
# ERDM UNet ([B, W, C, H, W'] with bilinear interpolation to working grid)
# ---------------------------------------------------------------------------


def _erdm_kwargs() -> dict:
    return dict(
        in_channels=4,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_levels=(1,),
        num_heads=4,
        temporal_num_heads=4,
        nlat=16,
        nlon=32,
        nlat_work=16,
        nlon_work=32,
        num_groups=8,  # 32 % 8 == 0
        scalar_dim=0,
        c_grid_dim=0,
    )


def test_erdm_forward_shape_finite():
    torch.manual_seed(0)
    model = ERDM(**_erdm_kwargs()).eval()
    b, W, c, h, w = 1, 3, 4, 16, 32
    x_noised = torch.randn(b, W, c, h, w)
    c_noise = torch.randn(b, W)
    with torch.no_grad():
        out = model(x_noised, c_noise, c_grid=None, c_scalar=None)
    assert out.shape == (b, W, c, h, w), out.shape
    assert torch.isfinite(out).all()


def test_erdm_from_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = ERDM(**_erdm_kwargs())
    p = tmp_path / "erdm.mdlus"
    model.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, ERDM)
