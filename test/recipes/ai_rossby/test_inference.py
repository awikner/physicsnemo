# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the after-the-fact inference + validate CLIs (Phase 4b).

The CLIs assemble many pieces (checkpoint load, datapipe build, xarray
write, hydra) that are exercised end-to-end on Delta. These unit tests
focus on the pure logic: shape correctness, the rollout loop's output
allocations, and that the ensemble axis flows through correctly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from inference import _build_xr_dataset, run_inference  # noqa: E402
from validate import GaussianIC  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny dataset shim
# ---------------------------------------------------------------------------


class _StubLayout:
    def __init__(self):
        self.surface_variables = ["pl", "tas"]
        self.upper_air_variables = ["ta", "ua", "va", "hus", "zg"]
        self.sigma_upper_air_variables = ["ta", "ua", "va", "hus"]
        self.pressure_upper_air_variables = ["zg"]
        self.constant_boundary_variables = ["lsm", "sg"]
        self.varying_boundary_variables = ["sst", "rsdt"]
        self.diagnostic_variables = ["pr_6h"]


class _StubDataset:
    """A PlasimClimateDataset-shaped stub with synthetic frames."""

    def __init__(self, n_time=10, Cs=2, Cu=5, L=4, H=8, W=8):
        self.n_time = n_time
        self.Cs, self.Cu, self.L, self.H, self.W = Cs, Cu, L, H, W
        self.layout = _StubLayout()
        self.sigma_levels = [0.1, 0.3, 0.5, 0.7][:L]
        self.pressure_levels = [85000.0, 92500.0, 100000.0, 105000.0][:L]
        torch.manual_seed(0)
        self._surface = torch.randn(n_time, Cs, H, W)
        self._upper = torch.randn(n_time, Cu, L, H, W)
        self._const = torch.randn(2, H, W)
        self._varying = torch.randn(n_time, 2, H, W)
        self._diag = torch.randn(n_time, 1, H, W)
        self.transform = None
        # xarray-style proxy for lat/lon access used by inference.run_inference
        self._ds = xr.Dataset(
            coords={
                "lat": ("lat", np.linspace(-90, 90, H, dtype=np.float32)),
                "lon": ("lon", np.linspace(0, 360, W, endpoint=False, dtype=np.float32)),
            }
        )

    def __len__(self):
        return self.n_time

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            t, lead = idx
        else:
            t, lead = int(idx), 1
        return {
            "surface_in": self._surface[t],
            "upper_air_in": self._upper[t],
            "constant_boundary": self._const,
            "varying_boundary": self._varying[t],
            "target_surface": self._surface[t + lead],
            "target_upper_air": self._upper[t + lead],
            "diagnostic": self._diag[t + lead],
        }


class _StubModel(nn.Module):
    """Identity-ish model used to exercise the rollout shape plumbing."""

    def __init__(self, n_surface, n_upper, n_levels, has_diagnostic=False):
        super().__init__()
        self.has_diagnostic = has_diagnostic
        self.n_surface = n_surface
        self.n_upper = n_upper
        self.n_levels = n_levels

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in, **_):
        # Persistence: identity passthrough so we can verify the
        # rollout writes match the input states across steps.
        if self.has_diagnostic:
            diag = torch.zeros_like(surface_in[:, :1])
            return surface_in, upper_air_in, diag, 0, 0, 0, 0
        return surface_in, upper_air_in, 0, 0, 0, 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_xr_dataset_shapes():
    ds = _build_xr_dataset(
        ic_indices=[0, 5],
        max_step=4,
        ensemble_size=3,
        lat=np.linspace(-90, 90, 8, dtype=np.float32),
        lon=np.linspace(0, 360, 16, endpoint=False, dtype=np.float32),
        surface_variables=["pl", "tas"],
        upper_air_variables=["ta", "ua"],
        diagnostic_variables=["pr_6h"],
        levels=[0.1, 0.5],
        n_levels=2,
        has_diagnostic=True,
    )
    assert ds["pred_surface"].shape == (2, 3, 4, 2, 8, 16)
    assert ds["pred_upper_air"].shape == (2, 3, 4, 2, 2, 8, 16)
    assert ds["pred_diagnostic"].shape == (2, 3, 4, 1, 8, 16)


def test_run_inference_deterministic_shapes():
    ds = _StubDataset(n_time=10)
    model = _StubModel(n_surface=2, n_upper=5, n_levels=4)
    out = run_inference(
        model,
        ds,
        normalizer=None,
        device=torch.device("cpu"),
        ic_indices=[0, 4],
        max_step=3,
        ensemble_size=1,
        perturber=None,
        batch_size=1,
        has_diagnostic=False,
        seed=0,
    )
    # 2 ICs × 1 member × 3 steps
    assert out["pred_surface"].shape == (2, 1, 3, 2, 8, 8)
    assert out["pred_upper_air"].shape == (2, 1, 3, 5, 4, 8, 8)
    assert "pred_diagnostic" not in out.data_vars


def test_run_inference_ensemble_shapes_and_perturbed_values():
    ds = _StubDataset(n_time=10)
    model = _StubModel(n_surface=2, n_upper=5, n_levels=4)
    out = run_inference(
        model,
        ds,
        normalizer=None,
        device=torch.device("cpu"),
        ic_indices=[0],
        max_step=2,
        ensemble_size=4,
        perturber=GaussianIC(scales={"surface_in": 0.1}),
        batch_size=1,
        has_diagnostic=False,
        seed=1,
    )
    assert out["pred_surface"].shape == (1, 4, 2, 2, 8, 8)
    # With small noise + persistence model, the 4 ensemble members should
    # differ from each other at step 1.
    arr = out["pred_surface"].values
    member0 = arr[0, 0, 0]
    member1 = arr[0, 1, 0]
    assert not np.allclose(member0, member1, atol=1e-6)


def test_run_inference_persistence_matches_initial_state():
    """A persistence model at unroll_steps=1 should write back the IC frame."""
    ds = _StubDataset(n_time=10)
    model = _StubModel(n_surface=2, n_upper=5, n_levels=4)
    out = run_inference(
        model,
        ds,
        normalizer=None,
        device=torch.device("cpu"),
        ic_indices=[2],
        max_step=1,
        ensemble_size=1,
        perturber=None,
        batch_size=1,
        has_diagnostic=False,
        seed=0,
    )
    # The first prediction step should equal the IC's surface_in
    # (persistence: model(state)=state).
    pred = out["pred_surface"].values[0, 0, 0]  # (Cs, H, W)
    expected = ds._surface[2].numpy()
    np.testing.assert_allclose(pred, expected, atol=1e-6)


def test_run_inference_writes_netcdf_roundtrip(tmp_path):
    """End-to-end: run inference, save NetCDF, reload, verify shape."""
    ds = _StubDataset(n_time=10)
    model = _StubModel(n_surface=2, n_upper=5, n_levels=4)
    out = run_inference(
        model,
        ds,
        normalizer=None,
        device=torch.device("cpu"),
        ic_indices=[0, 5],
        max_step=2,
        ensemble_size=2,
        perturber=GaussianIC(scales={"surface_in": 0.01}),
        batch_size=1,
        has_diagnostic=False,
        seed=42,
    )
    out_path = tmp_path / "preds.nc"
    out.to_netcdf(out_path)
    reread = xr.open_dataset(out_path)
    assert dict(reread.sizes) == dict(out.sizes)
    for v in out.data_vars:
        np.testing.assert_allclose(reread[v].values, out[v].values, atol=1e-6)
