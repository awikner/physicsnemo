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
        DataDependentInterpolant,
        DriftScheduler,
        DynamicInterpolant,
        ERDMScheduler,
        RFMScheduler,
    )
    from physicsnemo.experimental.models.amip_si import (
        AmipDiTWrapper,
        CombinedModule,
        ERDMWrapper,
        RollingDiTWrapper,
        XDDCWrapper,
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
# F3: bf16-native MetaData (Phase 8f).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_wrapper",
    [
        lambda: AmipDiTWrapper(
            **_kwargs(),
            dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
        ),
        lambda: RollingDiTWrapper(
            **_kwargs(), rolling_dit_kwargs=dict(dim=32, num_heads=4, num_blocks=1)
        ),
        lambda: ERDMWrapper(
            **_kwargs(),
            erdm_kwargs=dict(
                model_channels=16,
                channel_mult=(1, 2),
                num_res_blocks=1,
                attn_levels=(1,),
                num_groups=4,
            ),
        ),
    ],
    ids=["AmipDiTWrapper", "RollingDiTWrapper", "ERDMWrapper"],
)
def test_wrapper_meta_is_bf16_native(make_wrapper):
    torch.manual_seed(0)
    w = make_wrapper()
    assert w.meta.amp is True
    assert w.meta.bf16 is True
    # Derived from ``amp`` via ModelMetaData.__post_init__, not hardcoded off.
    assert w.meta.amp_gpu is True
    assert w.meta.amp_cpu is True
    # Iterative diffusion sampling is not CUDA-graph friendly — stays off.
    assert w.meta.cuda_graphs is False


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
# XDDCWrapper — x_DDC super-resolution cascade (Phase 8f, F6)
# ---------------------------------------------------------------------------


def _xddc_kwargs() -> dict:
    return dict(
        surface_variables=["a", "b", "c"],
        upper_air_variables=["ta", "ua"],
        diagnostic_variables=["p1"],
        levels=[100.0, 500.0, 1000.0],
        horizontal_resolution=(16, 32),
        downsample_factor=4,
    )


def _xddc_sample(batch_size: int = 1) -> dict:
    nlat, nlon = 16, 32
    return {
        "surface_in": torch.randn(batch_size, 3, nlat, nlon),
        "upper_air_in": torch.randn(batch_size, 2, 3, nlat, nlon),
        "diagnostic": torch.randn(batch_size, 1, nlat, nlon),
    }


def test_xddc_wrapper_packs_and_unpacks_surface_diag_upper_air_order():
    torch.manual_seed(0)
    w = XDDCWrapper(
        **_xddc_kwargs(),
        unet_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1, num_groups=4),
    )
    assert w.in_channels == 3 + 2 * 3 + 1  # surf + ua*L + diag = 10
    sample = _xddc_sample()
    x = w.pack_state(sample)
    assert x.shape == (1, 10, 16, 32)
    # Channel order is (surface, diagnostic, upper_air) — NOT the
    # (surface, upper_air, diagnostic) order AmipDiTWrapper uses.
    assert torch.allclose(x[:, :3], sample["surface_in"])
    assert torch.allclose(x[:, 3:4], sample["diagnostic"])
    unpacked = w.unpack_state(x)
    assert torch.allclose(unpacked["surface_in"], sample["surface_in"])
    assert torch.allclose(unpacked["diagnostic"], sample["diagnostic"])
    assert torch.allclose(unpacked["upper_air_in"], sample["upper_air_in"])


def test_xddc_wrapper_downsample_then_upsample_shape_roundtrips():
    torch.manual_seed(0)
    w = XDDCWrapper(
        **_xddc_kwargs(),
        unet_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1, num_groups=4),
    )
    sample = _xddc_sample()
    cond = w.downsample_then_upsample(sample)
    assert cond.shape == (1, 10, 16, 32)
    assert torch.isfinite(cond).all()


def test_xddc_wrapper_with_data_dependent_interpolant():
    torch.manual_seed(0)
    w = XDDCWrapper(
        **_xddc_kwargs(),
        unet_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1, num_groups=4),
    ).eval()
    sample = _xddc_sample()
    x = w.pack_state(sample)
    cond = w.downsample_then_upsample(sample)
    sched = DataDependentInterpolant(num_steps=2, noise="gaussian", integrator="exponential")
    loss = sched.compute_loss(w, cond, x)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    out = sched.sample(w, cond, num_steps=2)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_xddc_wrapper_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    w = XDDCWrapper(
        **_xddc_kwargs(),
        unet_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1, num_groups=4),
    )
    p = tmp_path / "xddc_wrapper.mdlus"
    w.save(str(p))
    loaded = physicsnemo.Module.from_checkpoint(str(p))
    assert isinstance(loaded, XDDCWrapper)
    assert loaded.in_channels == w.in_channels


# ---------------------------------------------------------------------------
# CombinedModule — forecaster + x_DDC downscaler composition (Phase 8f, F6)
# ---------------------------------------------------------------------------


def test_combined_module_forward_composes_forecaster_and_downscaler():
    torch.manual_seed(0)
    forecaster = AmipDiTWrapper(
        surface_variables=["a", "b", "c"],
        upper_air_variables=["ta", "ua"],
        diagnostic_variables=["p1"],
        constant_boundary_variables=["lsm"],
        varying_boundary_variables=["sst"],
        levels=[100.0, 500.0, 1000.0],
        horizontal_resolution=(4, 8),
        dit_kwargs=dict(dim=16, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
    ).eval()
    downscaler = XDDCWrapper(
        surface_variables=["a", "b", "c"],
        upper_air_variables=["ta", "ua"],
        diagnostic_variables=["p1"],
        levels=[100.0, 500.0, 1000.0],
        horizontal_resolution=(16, 32),
        downsample_factor=4,
        unet_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1, num_groups=4),
    ).eval()
    combined = CombinedModule(
        forecaster=forecaster,
        forecaster_scheduler=DynamicInterpolant(num_steps=2, noise="gaussian"),
        downscaler=downscaler,
        downscaler_scheduler=DataDependentInterpolant(
            num_steps=2, noise="gaussian", integrator="exponential"
        ),
    )
    sample = {
        "surface_in": torch.randn(2, 3, 4, 8),
        "upper_air_in": torch.randn(2, 2, 3, 4, 8),
        "diagnostic": torch.randn(2, 1, 4, 8),
        "constant_boundary": torch.randn(1, 4, 8),
        "varying_boundary": torch.randn(2, 1, 4, 8),
        "calendar": torch.randn(2, 2),
    }
    out = combined(sample, forecaster_num_steps=2, downscaler_num_steps=2)
    assert out["surface_in"].shape == (2, 3, 16, 32)
    assert out["upper_air_in"].shape == (2, 2, 3, 16, 32)
    assert out["diagnostic"].shape == (2, 1, 16, 32)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_combined_module_handles_driftscheduler_forecaster_no_tuple_return():
    """DriftScheduler.sample() returns a plain tensor (no return_model_last) —
    CombinedModule's tuple-unpacking guard must be a no-op here."""
    torch.manual_seed(0)
    # Gives the forecaster a real (non-empty) c_grid — see
    # test_backbones.py::test_dit_c_grid_dim_zero_with_downsample_positive_forward_ok
    # for the dedicated c_grid_dim=0 regression coverage; this test's
    # focus is the tuple-vs-tensor scheduler handling, not that edge case.
    forecaster = AmipDiTWrapper(
        surface_variables=["a", "b"],
        upper_air_variables=["ta"],
        constant_boundary_variables=["lsm"],
        levels=[500.0],
        horizontal_resolution=(4, 8),
        dit_kwargs=dict(dim=16, num_heads=4, num_blocks=1, patch_size=2, c_grid_downsample=1),
    ).eval()
    downscaler = XDDCWrapper(
        surface_variables=["a", "b"],
        upper_air_variables=["ta"],
        levels=[500.0],
        horizontal_resolution=(16, 32),
        downsample_factor=4,
        unet_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1, num_groups=4),
    ).eval()
    combined = CombinedModule(
        forecaster=forecaster,
        forecaster_scheduler=DriftScheduler(num_steps=2, noise="gaussian"),
        downscaler=downscaler,
        downscaler_scheduler=DataDependentInterpolant(
            num_steps=2, noise="gaussian", integrator="exponential"
        ),
    )
    sample = {
        "surface_in": torch.randn(2, 2, 4, 8),
        "upper_air_in": torch.randn(2, 1, 1, 4, 8),
        "constant_boundary": torch.randn(1, 4, 8),
        "calendar": torch.randn(2, 2),
    }
    out = combined(sample, forecaster_num_steps=2, downscaler_num_steps=2)
    assert out["surface_in"].shape == (2, 2, 16, 32)
    assert out["upper_air_in"].shape == (2, 1, 1, 16, 32)
    assert torch.isfinite(out["surface_in"]).all()


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
        ("amip_x_ddc", "x_ddc", "DataDependentInterpolant"),
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


def test_amip_x_ddc_model_config_builds_xddc_wrapper():
    """conf/model/amip_x_ddc.yaml builds via the shared build_model() helper."""
    import sys
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    ai_rossby_dir = (
        Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
    )
    sys.path.insert(0, str(ai_rossby_dir))
    from train import build_model  # noqa: E402

    cfg_dir = ai_rossby_dir / "conf"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.2"):
        cfg = compose(config_name="config", overrides=["model=amip_x_ddc"])
    model = build_model(cfg.model)
    assert isinstance(model, XDDCWrapper)
    assert model.downsample_factor == 4
