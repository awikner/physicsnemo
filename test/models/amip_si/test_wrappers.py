# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8c unit tests for the AMIP diffusion model wrappers.

Verifies:

* Each wrapper (AmipDiTWrapper, RollingDiTWrapper, ERDMWrapper) sizes its
  backbone correctly from the structured channel-group inputs.
* ``pack_state`` ↔ ``unpack_state`` round-trips.
* ``pack_c_grid`` produces the expected shape.
* End-to-end: scheduler.compute_loss(wrapper, …) returns a finite scalar
  for the matching scheduler family.
* ``Module.save`` / ``from_checkpoint`` round-trips the wrapper.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    import physicsnemo
    from physicsnemo.experimental.diffusion import (
        DriftScheduler,
        DynamicInterpolant,
        ERDMScheduler,
        RFMScheduler,
    )
    from physicsnemo.experimental.models.amip_si import (
        AmipDiTWrapper,
        ERDMWrapper,
        RollingDiTWrapper,
    )


# ---------------------------------------------------------------------------
# Common channel-group configuration shared across all wrappers.
# ---------------------------------------------------------------------------


def _kwargs() -> dict:
    return dict(
        surface_variables=["a", "b", "c"],
        upper_air_variables=["ta", "ua"],
        diagnostic_variables=["p1"],
        constant_boundary_variables=["lsm"],
        varying_boundary_variables=["sst", "co2"],
        levels=[100.0, 500.0, 1000.0],
        horizontal_resolution=(32, 64),
    )


def _single_step_sample(batch_size: int = 1) -> dict:
    nlat, nlon = 32, 64
    return {
        "surface_in": torch.randn(batch_size, 3, nlat, nlon),
        "upper_air_in": torch.randn(batch_size, 2, 3, nlat, nlon),
        "diagnostic": torch.randn(batch_size, 1, nlat, nlon),
        "constant_boundary": torch.randn(1, nlat, nlon),  # no batch dim
        "varying_boundary": torch.randn(batch_size, 2, nlat, nlon),
        "calendar": torch.randn(batch_size, 2),
    }


def _window_sample(batch_size: int = 1, W: int = 3) -> dict:
    nlat, nlon = 32, 64
    return {
        "surface_in": torch.randn(batch_size, W, 3, nlat, nlon),
        "upper_air_in": torch.randn(batch_size, W, 2, 3, nlat, nlon),
        "diagnostic": torch.randn(batch_size, W, 1, nlat, nlon),
        "constant_boundary": torch.randn(1, nlat, nlon),
        "varying_boundary": torch.randn(batch_size, W, 2, nlat, nlon),
        "calendar": torch.randn(batch_size, W, 2),
    }


# ---------------------------------------------------------------------------
# AmipDiTWrapper — single-step
# ---------------------------------------------------------------------------


def test_amip_dit_wrapper_packs_and_unpacks():
    torch.manual_seed(0)
    w = AmipDiTWrapper(
        **_kwargs(),
        dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
    )
    assert w.in_channels == 3 + 2 * 3 + 1  # surf + ua*L + diag = 10
    assert w.c_grid_dim == 1 + 2          # const + varying = 3
    sample = _single_step_sample()
    x = w.pack_state(sample)
    assert x.shape == (1, 10, 32, 64)
    unpacked = w.unpack_state(x)
    assert unpacked["surface_in"].shape == (1, 3, 32, 64)
    assert unpacked["upper_air_in"].shape == (1, 2, 3, 32, 64)
    assert unpacked["diagnostic"].shape == (1, 1, 32, 64)
    # Round-trip check.
    assert torch.allclose(unpacked["surface_in"], sample["surface_in"], atol=0)
    assert torch.allclose(unpacked["upper_air_in"], sample["upper_air_in"], atol=0)
    assert torch.allclose(unpacked["diagnostic"], sample["diagnostic"], atol=0)
    c_grid = w.pack_c_grid(sample)
    assert c_grid.shape == (1, 3, 32, 64)


def test_amip_dit_wrapper_with_drift_scheduler():
    torch.manual_seed(0)
    w = AmipDiTWrapper(
        **_kwargs(),
        dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
    ).eval()
    sample = _single_step_sample()
    target = _single_step_sample()
    x = w.pack_state(sample)
    y = w.pack_state(target)
    c_grid = w.pack_c_grid(sample)
    c_scalar = sample["calendar"]
    sched = DriftScheduler(num_steps=2, noise="gaussian")
    loss = sched.compute_loss(w, x, c_grid, c_scalar, y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_amip_dit_wrapper_with_dynamic_interpolant():
    torch.manual_seed(0)
    w = AmipDiTWrapper(
        **_kwargs(),
        dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
    ).eval()
    sample = _single_step_sample()
    target = _single_step_sample()
    x = w.pack_state(sample)
    y = w.pack_state(target)
    c_grid = w.pack_c_grid(sample)
    c_scalar = sample["calendar"]
    sched = DynamicInterpolant(num_steps=2, noise="gaussian")
    loss = sched.compute_loss(w, x, c_grid, c_scalar, y)
    assert torch.isfinite(loss)


def test_amip_dit_wrapper_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    w = AmipDiTWrapper(
        **_kwargs(),
        dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
    )
    p = tmp_path / "amip_dit_wrapper.mdlus"
    w.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, AmipDiTWrapper)
    assert loaded.in_channels == w.in_channels


# ---------------------------------------------------------------------------
# RollingDiTWrapper — rolling window
# ---------------------------------------------------------------------------


def test_rolling_dit_wrapper_packs_window():
    torch.manual_seed(0)
    w = RollingDiTWrapper(
        **_kwargs(),
        rolling_dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1),
    ).eval()
    sample = _window_sample(batch_size=1, W=3)
    y = w.pack_window_state(sample)
    assert y.shape == (1, 3, 10, 32, 64)
    unpacked = w.unpack_window_state(y)
    assert unpacked["surface_in"].shape == (1, 3, 3, 32, 64)
    assert unpacked["upper_air_in"].shape == (1, 3, 2, 3, 32, 64)
    c_grid = w.pack_window_c_grid(sample)
    assert c_grid.shape == (1, 3, 3, 32, 64)


def test_rolling_dit_wrapper_with_rfm_scheduler():
    torch.manual_seed(0)
    w = RollingDiTWrapper(
        **_kwargs(),
        rolling_dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1),
    ).eval()
    W = 3
    sample = _window_sample(batch_size=1, W=W)
    y = w.pack_window_state(sample)
    c_grid = w.pack_window_c_grid(sample)
    c_scalar = sample["calendar"]
    sched = RFMScheduler(window_size=W, num_steps=2, noise="gaussian")
    loss = sched.compute_loss(w, c_grid, c_scalar, y)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ERDMWrapper — rolling window, UNet backbone
# ---------------------------------------------------------------------------


def test_erdm_wrapper_with_erdm_scheduler():
    torch.manual_seed(0)
    w = ERDMWrapper(
        **_kwargs(),
        erdm_kwargs=dict(
            model_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attn_levels=(1,),
            num_groups=4,
        ),
    ).eval()
    W = 3
    sample = _window_sample(batch_size=1, W=W)
    y = w.pack_window_state(sample)
    c_grid = w.pack_window_c_grid(sample)
    c_scalar = sample["calendar"]
    sched = ERDMScheduler(window_size=W, num_steps=2, noise="gaussian")
    loss = sched.compute_loss(w, c_grid, c_scalar, y)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Hydra config instantiation
# ---------------------------------------------------------------------------


def test_hydra_diffusion_configs_instantiate():
    """All four (model, loss, training) combos compose + instantiate cleanly."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "weather"
        / "ai_rossby"
        / "conf"
    )

    pairs = [
        ("amip_si", "si", "DriftScheduler"),
        ("amip_si_x", "si_x", "DynamicInterpolant"),
        ("amip_rfm", "rfm", "RFMScheduler"),
        ("amip_erdm", "erdm", "ERDMScheduler"),
    ]
    for model_name, loss_name, expected_sched in pairs:
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.2"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"model={model_name}",
                    f"loss={loss_name}",
                    "training=amip_diffusion",
                ],
            )
        sched = instantiate(cfg.loss)
        assert type(sched).__name__ == expected_sched
