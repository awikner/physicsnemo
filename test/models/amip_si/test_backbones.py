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
    from physicsnemo.experimental.models.amip_si import AmipDiT, ERDM, RollingDiT, XDDCUNet


# ---------------------------------------------------------------------------
# DiT (single-step, [B, C, H, W] forward)
# ---------------------------------------------------------------------------


def _dit_kwargs() -> dict:
    # in_channels is the PatchEmbed channel count and bakes in the
    # [x_noised, cond] concat assumption (2 * state channels) — see
    # wrappers.py's AmipDiTWrapper.dit_kwargs.setdefault("in_channels", ...).
    # out_channels is the bare (undoubled) state channel count.
    return dict(
        in_channels=8,
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


def test_dit_c_grid_dim_zero_with_downsample_positive_forward_ok():
    """Regression test for a latent shape-mismatch bug: c_grid_embed was
    previously allocated (and its channels baked into patch_in_channels)
    whenever ``c_grid_downsample > 0``, regardless of ``c_grid_dim`` — so
    a c_grid_dim=0 config (no c_grid conditioning at all) crashed at
    forward time as soon as c_grid_downsample defaulted/was set to a
    positive value (the wrapper's own default). RollingDiT / ERDM already
    gated this correctly on ``c_grid_dim > 0``; AmipDiT didn't.
    """
    torch.manual_seed(0)
    kwargs = _dit_kwargs()
    kwargs["c_grid_downsample"] = 2  # nonzero — previously always allocated c_grid_embed
    model = AmipDiT(**kwargs).eval()
    assert model.c_grid_embed is None
    b, h, w = 2, 16, 32
    x_noised = torch.randn(b, 4, h, w)
    cond = torch.randn(b, 4, h, w)
    t = torch.rand(b)
    c_scalar = torch.randn(b, 2)
    with torch.no_grad():
        out = model(x_noised, cond, t, c_grid=None, c_scalar=c_scalar)
    assert out.shape == (b, 4, h, w)
    assert torch.isfinite(out).all()


def test_dit_from_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = AmipDiT(**_dit_kwargs())
    p = tmp_path / "dit.mdlus"
    model.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, AmipDiT)
    assert loaded.meta.cuda_graphs is False  # permanent: iterative sample() isn't graph-friendly


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


# ---------------------------------------------------------------------------
# XDDCUNet (x_DDC super-resolution cascade denoiser, [B, C, H, W] forward,
# no c_grid/c_scalar conditioning — Phase 8f)
# ---------------------------------------------------------------------------


def _xddc_unet_kwargs() -> dict:
    return dict(
        in_channels=8,
        out_channels=4,
        model_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_levels=(1,),
        num_heads=4,
        num_groups=4,
    )


def test_xddc_unet_forward_shape_finite():
    torch.manual_seed(0)
    model = XDDCUNet(**_xddc_unet_kwargs()).eval()
    b, c, h, w = 2, 4, 16, 32
    x_noised = torch.randn(b, c, h, w)
    cond = torch.randn(b, c, h, w)
    t = torch.rand(b, 1)
    with torch.no_grad():
        out = model(x_noised, cond, t)
    assert out.shape == (b, c, h, w), out.shape
    assert torch.isfinite(out).all()


def test_xddc_unet_from_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = XDDCUNet(**_xddc_unet_kwargs())
    p = tmp_path / "xddc_unet.mdlus"
    model.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, XDDCUNet)
    assert loaded.meta.cuda_graphs is False
