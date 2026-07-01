# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 8d diffusion-aware inference path.

Mirrors :mod:`test_inference` (deterministic path) but exercises the
diffusion dispatch added in inference.py:

* ``_is_diffusion_model`` correctly classifies wrapper-shaped models.
* ``_build_per_ic_dataset(ensemble_save_mode="summary")`` produces
  ``*_mean`` / ``*_std`` variables and drops the ensemble axis.
* ``run_diffusion_inference_streaming_per_ic`` rolls a single-step
  scheduler stub autoregressively, writes one file per IC, and the
  per-IC layout reflects the chosen ``ensemble_save_mode``.
* The same function rolls a window-mode (``sample_rollout``)
  scheduler stub correctly.

All tests use synthetic stubs — no real backbones, no Hydra compose,
no real data — so they finish in milliseconds and can run in CPU CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent
    / "examples"
    / "weather"
    / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from async_writer import AsyncForecastWriter  # noqa: E402
from inference import (  # noqa: E402
    _build_per_ic_dataset,
    _is_diffusion_model,
    _write_per_ic_frame,
    run_diffusion_inference_streaming_per_ic,
)
from validate import Deterministic, ReplicateOnly  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset + wrapper + scheduler stubs.
# ---------------------------------------------------------------------------


class _StubLayout:
    def __init__(self):
        self.surface_variables = ["t2m", "msl"]
        self.upper_air_variables = ["ta", "ua"]
        self.sigma_upper_air_variables = []
        self.pressure_upper_air_variables = ["ta", "ua"]
        self.constant_boundary_variables = ["lsm"]
        self.varying_boundary_variables = ["sst"]
        self.diagnostic_variables = ["rsds"]


class _StubDiffusionDataset:
    """Mimics :class:`ClimateZarrDataset(emit_calendar=True)`."""

    def __init__(self, n_time=12, Cs=2, Cu=2, L=3, H=8, W=8, Cd=1):
        self.n_time = n_time
        self.Cs, self.Cu, self.L, self.H, self.W, self.Cd = Cs, Cu, L, H, W, Cd
        self.layout = _StubLayout()
        self.sigma_levels = []
        self.pressure_levels = [85000.0, 92500.0, 100000.0][:L]
        torch.manual_seed(7)
        self._surface = torch.randn(n_time, Cs, H, W)
        self._upper = torch.randn(n_time, Cu, L, H, W)
        self._const = torch.randn(1, H, W)
        self._varying = torch.randn(n_time, 1, H, W)
        self._diag = torch.randn(n_time, Cd, H, W)
        # Calendar is (second_of_day, day_of_year).
        self._calendar = torch.randn(n_time, 2)
        self.transform = None
        self._ds = xr.Dataset(
            coords={
                "lat": ("lat", np.linspace(-90, 90, H, dtype=np.float32)),
                "lon": (
                    "lon",
                    np.linspace(0, 360, W, endpoint=False, dtype=np.float32),
                ),
            }
        )

    def __len__(self):
        return self.n_time

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            t, _ = idx
        else:
            t = int(idx)
        return {
            "surface_in": self._surface[t],
            "upper_air_in": self._upper[t],
            "constant_boundary": self._const,
            "varying_boundary": self._varying[t],
            "diagnostic": self._diag[t],
            "calendar": self._calendar[t],
        }


class _StubDiffusionWrapper(nn.Module):
    """Minimum pack/unpack contract from the real AmipDiTWrapper."""

    def __init__(self, num_surface=2, num_upper=2, num_levels=3, num_diag=1):
        super().__init__()
        self.num_surface = num_surface
        self.num_upper_air_vars = num_upper
        self.num_levels = num_levels
        self.num_diagnostic = num_diag
        self.num_constant_boundary = 1
        self.num_varying_boundary = 1
        self.in_channels = num_surface + num_upper * num_levels + num_diag
        self.c_grid_dim = 2

    # -- pack/unpack on flat tensors --
    def pack_state(self, sample):
        parts = [sample["surface_in"]]
        ua = sample["upper_air_in"]
        b_shape = ua.shape[:-4]
        parts.append(
            ua.reshape(*b_shape, self.num_upper_air_vars * self.num_levels, *ua.shape[-2:])
        )
        if "diagnostic" in sample and sample["diagnostic"] is not None:
            parts.append(sample["diagnostic"])
        return torch.cat(parts, dim=-3)

    def unpack_state(self, x):
        idx = 0
        out = {}
        out["surface_in"] = x.narrow(-3, idx, self.num_surface)
        idx += self.num_surface
        ua_flat = x.narrow(
            -3, idx, self.num_upper_air_vars * self.num_levels
        )
        b_shape = ua_flat.shape[:-3]
        out["upper_air_in"] = ua_flat.reshape(
            *b_shape, self.num_upper_air_vars, self.num_levels, *ua_flat.shape[-2:]
        )
        idx += self.num_upper_air_vars * self.num_levels
        out["diagnostic"] = x.narrow(-3, idx, self.num_diagnostic)
        return out

    def pack_c_grid(self, sample):
        const = sample["constant_boundary"]
        if const.dim() == 3:
            const = const.unsqueeze(0).expand(
                sample["surface_in"].shape[0], -1, -1, -1
            )
        return torch.cat([const, sample["varying_boundary"]], dim=-3)

    def pack_window_state(self, window):
        # window shapes: surface_in (B, W, Cs, H, W), upper_air_in (B, W, Cu, L, H, W)
        s = window["surface_in"]
        ua = window["upper_air_in"]
        ua_flat = ua.reshape(
            ua.shape[0], ua.shape[1], self.num_upper_air_vars * self.num_levels,
            *ua.shape[-2:],
        )
        if "diagnostic" in window and window["diagnostic"] is not None:
            return torch.cat([s, ua_flat, window["diagnostic"]], dim=-3)
        return torch.cat([s, ua_flat], dim=-3)

    def pack_window_c_grid(self, window):
        # Mirror the real wrapper: broadcast const to (B, W, Cb, H, W) from
        # whatever leading shape it arrives with (3D unbatched or 4D batched).
        var_b = window["varying_boundary"]
        target_shape = var_b.shape  # (B, W, Cb, H, W)
        const = window["constant_boundary"]
        while const.dim() < var_b.dim():
            const = const.unsqueeze(0)
        const = const.expand(*target_shape[:-3], -1, -1, -1)
        return torch.cat([const, var_b], dim=-3)


class _StubSingleStepScheduler:
    """Records the calls and returns ``x + step_bias`` so we can check shapes."""

    def __init__(self):
        self.num_steps = 4
        self.calls = 0

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        self.calls += 1
        return x + 0.1


class _StubRollingScheduler(nn.Module):
    """``sample_rollout`` returns ``(B, horizon, C, H, W)`` zeros."""

    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.num_steps = 2
        self.calls = 0

    def sample_rollout(
        self, model, init_window, c_grid_traj, c_scalar_traj, horizon, num_steps=None
    ):
        self.calls += 1
        B, W, C, H, Wd = init_window.shape
        # Return a deterministic stand-in: each emitted frame is the
        # last input window frame plus a per-step bias. The actual
        # values don't matter for the shape/wiring tests.
        out = torch.zeros(B, horizon, C, H, Wd)
        for k in range(horizon):
            out[:, k] = init_window[:, -1] + 0.1 * (k + 1)
        return out


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_is_diffusion_model_true_for_wrapper():
    assert _is_diffusion_model(_StubDiffusionWrapper())


def test_is_diffusion_model_false_for_plain_module():
    class _Plain(nn.Module):
        def forward(self, *a, **kw):
            return a[0]

    assert not _is_diffusion_model(_Plain())


# ---------------------------------------------------------------------------
# Per-IC dataset — summary mode shape + vars
# ---------------------------------------------------------------------------


def test_build_per_ic_dataset_summary_mode_drops_ensemble_axis():
    ds = _build_per_ic_dataset(
        ic_index=0,
        n_frames=4,
        ensemble_size=3,
        lat=np.linspace(-90, 90, 6, dtype=np.float32),
        lon=np.linspace(0, 360, 8, endpoint=False, dtype=np.float32),
        surface_variables=["t2m", "msl"],
        upper_air_variables=["ta"],
        diagnostic_variables=["rsds"],
        levels=[1000.0, 850.0],
        n_levels=2,
        has_diagnostic=True,
        ensemble_save_mode="summary",
    )
    # Summary mode has mean+std variants, no ensemble axis.
    assert {"pred_surface_mean", "pred_surface_std"} <= set(ds.data_vars)
    assert {"pred_upper_air_mean", "pred_upper_air_std"} <= set(ds.data_vars)
    assert {"pred_diagnostic_mean", "pred_diagnostic_std"} <= set(ds.data_vars)
    assert "ensemble" not in ds.dims
    assert ds["pred_surface_mean"].shape == (4, 2, 6, 8)
    assert ds["pred_upper_air_mean"].shape == (4, 1, 2, 6, 8)
    assert ds.attrs["ensemble_save_mode"] == "summary"
    assert ds.attrs["ensemble_size"] == 3


def test_build_per_ic_dataset_members_mode_keeps_ensemble_axis():
    ds = _build_per_ic_dataset(
        ic_index=0,
        n_frames=4,
        ensemble_size=3,
        lat=np.linspace(-90, 90, 6, dtype=np.float32),
        lon=np.linspace(0, 360, 8, endpoint=False, dtype=np.float32),
        surface_variables=["t2m", "msl"],
        upper_air_variables=["ta"],
        diagnostic_variables=[],
        levels=[1000.0, 850.0],
        n_levels=2,
        has_diagnostic=False,
        ensemble_save_mode="members",
    )
    assert "ensemble" in ds.dims
    assert ds["pred_surface"].shape == (3, 4, 2, 6, 8)
    assert "pred_surface_mean" not in ds.data_vars
    assert ds.attrs["ensemble_save_mode"] == "members"


def test_build_per_ic_dataset_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ensemble_save_mode"):
        _build_per_ic_dataset(
            ic_index=0,
            n_frames=4,
            ensemble_size=1,
            lat=np.zeros(2, dtype=np.float32),
            lon=np.zeros(2, dtype=np.float32),
            surface_variables=["t2m"],
            upper_air_variables=["ta"],
            diagnostic_variables=[],
            levels=[1000.0],
            n_levels=1,
            has_diagnostic=False,
            ensemble_save_mode="middens",
        )


# ---------------------------------------------------------------------------
# _write_per_ic_frame round-trip
# ---------------------------------------------------------------------------


def test_write_per_ic_frame_summary_mean_std():
    ds = _build_per_ic_dataset(
        ic_index=0,
        n_frames=2,
        ensemble_size=4,
        lat=np.linspace(-90, 90, 4, dtype=np.float32),
        lon=np.linspace(0, 360, 4, endpoint=False, dtype=np.float32),
        surface_variables=["t2m"],
        upper_air_variables=["ta"],
        diagnostic_variables=[],
        levels=[1000.0],
        n_levels=1,
        has_diagnostic=False,
        ensemble_save_mode="summary",
    )
    # Build an ensemble where each member is a constant offset.
    members_surface = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1, 1, 1)
    members_upper = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1, 1, 1, 1)
    surface = members_surface.expand(4, 1, 4, 4).contiguous()
    upper_air = members_upper.expand(4, 1, 1, 4, 4).contiguous()
    _write_per_ic_frame(
        ds,
        frame=1,
        ensemble_save_mode="summary",
        ensemble_size=4,
        surface=surface,
        upper_air=upper_air,
    )
    assert np.allclose(ds["pred_surface_mean"][1].values, 2.5)
    assert np.allclose(
        ds["pred_surface_std"][1].values, np.std([1.0, 2.0, 3.0, 4.0])
    )


# ---------------------------------------------------------------------------
# Single-step diffusion rollout
# ---------------------------------------------------------------------------


def test_run_diffusion_single_step_writes_per_ic_summary(tmp_path):
    dataset = _StubDiffusionDataset()
    model = _StubDiffusionWrapper()
    scheduler = _StubSingleStepScheduler()
    with AsyncForecastWriter(max_in_flight=2, num_workers=1) as writer:
        paths = run_diffusion_inference_streaming_per_ic(
            model,
            dataset,
            scheduler=scheduler,
            normalizer=None,
            device=torch.device("cpu"),
            ic_indices=[0, 4],
            max_step=3,
            writer=writer,
            output_dir=str(tmp_path),
            model_name="amip_si_test",
            run_name="diff-test",
            output_format="nc",
            ensemble_size=2,
            perturber=ReplicateOnly(),
            has_diagnostic=True,
            sampler_num_steps=None,
            seed=0,
            ensemble_save_mode="summary",
        )
    assert len(paths) == 2
    assert scheduler.calls == 2 * 3  # 2 ICs × 3 rollout steps each
    for p in paths:
        ds = xr.open_dataset(p)
        # Summary layout — _mean/_std only.
        assert "pred_surface_mean" in ds.data_vars
        assert "pred_surface_std" in ds.data_vars
        assert "ensemble" not in ds.dims
        # 4 frames = IC + 3 prediction steps.
        assert ds.dims["frame"] == 4
        ds.close()


def test_run_diffusion_single_step_writes_per_ic_members(tmp_path):
    dataset = _StubDiffusionDataset()
    model = _StubDiffusionWrapper()
    scheduler = _StubSingleStepScheduler()
    with AsyncForecastWriter(max_in_flight=2, num_workers=1) as writer:
        paths = run_diffusion_inference_streaming_per_ic(
            model,
            dataset,
            scheduler=scheduler,
            normalizer=None,
            device=torch.device("cpu"),
            ic_indices=[0],
            max_step=2,
            writer=writer,
            output_dir=str(tmp_path),
            model_name="amip_si_test",
            run_name="diff-test",
            output_format="nc",
            ensemble_size=3,
            perturber=ReplicateOnly(),
            has_diagnostic=True,
            seed=0,
            ensemble_save_mode="members",
        )
    assert len(paths) == 1
    ds = xr.open_dataset(paths[0])
    assert "ensemble" in ds.dims
    assert ds.dims["ensemble"] == 3
    assert ds.dims["frame"] == 3
    # Frame 0 should be identical across ensemble members (the shared IC).
    f0 = ds["pred_surface"].isel(frame=0).values
    assert np.allclose(f0[0], f0[1])
    assert np.allclose(f0[0], f0[2])
    ds.close()


# ---------------------------------------------------------------------------
# Window-mode diffusion rollout
# ---------------------------------------------------------------------------


def test_run_diffusion_window_mode_writes_per_ic_summary(tmp_path):
    dataset = _StubDiffusionDataset(n_time=20)
    model = _StubDiffusionWrapper(num_diag=0)
    scheduler = _StubRollingScheduler(window_size=3)
    with AsyncForecastWriter(max_in_flight=2, num_workers=1) as writer:
        paths = run_diffusion_inference_streaming_per_ic(
            model,
            dataset,
            scheduler=scheduler,
            normalizer=None,
            device=torch.device("cpu"),
            ic_indices=[5],
            max_step=4,
            writer=writer,
            output_dir=str(tmp_path),
            model_name="amip_rfm_test",
            run_name="diff-rolling",
            output_format="nc",
            ensemble_size=2,
            perturber=ReplicateOnly(),
            has_diagnostic=False,
            seed=0,
            ensemble_save_mode="summary",
        )
    # Window mode → one sample_rollout call per IC, not per step.
    assert scheduler.calls == 1
    ds = xr.open_dataset(paths[0])
    assert ds.dims["frame"] == 5  # IC + 4 forecast frames
    assert "pred_surface_mean" in ds.data_vars
    ds.close()


def test_run_diffusion_deterministic_ensemble_size_one(tmp_path):
    """ensemble_size=1 with Deterministic perturber: no ensemble axis fuss."""
    dataset = _StubDiffusionDataset()
    model = _StubDiffusionWrapper()
    scheduler = _StubSingleStepScheduler()
    with AsyncForecastWriter(max_in_flight=2, num_workers=1) as writer:
        paths = run_diffusion_inference_streaming_per_ic(
            model,
            dataset,
            scheduler=scheduler,
            normalizer=None,
            device=torch.device("cpu"),
            ic_indices=[0],
            max_step=2,
            writer=writer,
            output_dir=str(tmp_path),
            model_name="amip_si_test",
            run_name="diff-det",
            output_format="nc",
            ensemble_size=1,
            perturber=Deterministic(),
            has_diagnostic=True,
            seed=0,
            ensemble_save_mode="members",
        )
    ds = xr.open_dataset(paths[0])
    assert ds.dims["ensemble"] == 1
    ds.close()
