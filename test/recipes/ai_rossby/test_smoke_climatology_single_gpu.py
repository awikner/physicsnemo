# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU smoke test for the Phase 4c climatology CLI.

End-to-end on Delta `gpuA40x4-interactive`:
  PLASIM smoke fixture → tiny PanguPlasimLegacy → run_climatology with
  chunked forecast dumping → verify (a) the aggregator state is sane,
  (b) the chunked forecast files materialize via AsyncForecastWriter,
  (c) no two chunk files share a step.

Skipped when the Delta fixture isn't staged or the PLASIM stats are
missing — same gating as the other ai_rossby smoke tests.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from async_writer import AsyncForecastWriter  # noqa: E402
from climatology_cli import run_climatology  # noqa: E402

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.plasim import (
        NanFillTransform,
        PlasimClimateDataset,
        PlasimNormalizer,
    )
    from physicsnemo.experimental.models.pangu_plasim import PanguPlasimLegacy


_STATS_DIR = Path(
    "/work/nvme/bdiu/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data"
)
_MEAN_PATH = _STATS_DIR / "data_12-132_mean_sigma.nc"
_STD_PATH = _STATS_DIR / "data_12-132_std_sigma.nc"


def _fixture_path() -> Path | None:
    root = os.environ.get("AI_ROSSBY_TEST_DATA")
    if not root:
        return None
    p = Path(root) / "plasim" / "smoke_month.zarr"
    return p if p.exists() else None


_HAS_FIXTURE = _fixture_path() is not None
_HAS_STATS = _MEAN_PATH.exists() and _STD_PATH.exists()

_skip_no_fixture = pytest.mark.skipif(
    not _HAS_FIXTURE,
    reason="$AI_ROSSBY_TEST_DATA/plasim/smoke_month.zarr missing",
)
_skip_no_stats = pytest.mark.skipif(
    not _HAS_STATS, reason="PLASIM stats files missing"
)


# Tiny PanguPlasimLegacy that fits an A40 and runs a few rollout steps fast.
_BACKBONE_KWARGS = dict(
    patch_size=[2, 4, 4],
    depths=[1, 1, 1, 1],
    num_heads=[2, 4, 4, 2],
    embed_dim=64,
    updown_scale_factor=2,
    window_size=[1, 2, 2],
    drop_rate=0.0,
    checkpointing=0,
    use_reentrant=False,
)


def _build_tiny_model_and_dataset(device):
    ds = PlasimClimateDataset(_fixture_path())
    normalizer = PlasimNormalizer.from_dataset(
        ds, mean_path=_MEAN_PATH, std_path=_STD_PATH
    ).to(device)
    nan_fill = NanFillTransform(
        constant_boundary_variables=ds.layout.constant_boundary_variables,
        varying_boundary_variables=ds.layout.varying_boundary_variables,
        default=0.0,
    )
    ds.transform = nan_fill

    model = PanguPlasimLegacy(
        surface_variables=ds.layout.surface_variables,
        upper_air_variables=ds.upper_air_variable_names,
        constant_boundary_variables=ds.layout.constant_boundary_variables,
        varying_boundary_variables=ds.layout.varying_boundary_variables,
        diagnostic_variables=ds.layout.diagnostic_variables,
        levels=ds.sigma_levels,
        horizontal_resolution=list(ds.horizontal_resolution),
        **_BACKBONE_KWARGS,
    ).to(device)
    model.eval()
    return ds, normalizer, model


@pytest.mark.smoke
@pytest.mark.cuda
@_skip_no_fixture
@_skip_no_stats
def test_climatology_cli_smoke_with_chunked_forecast(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = torch.device("cuda:0")
    ds, normalizer, model = _build_tiny_model_and_dataset(device)

    max_step = 6
    chunk_steps = 3

    with AsyncForecastWriter(max_in_flight=2, num_workers=2) as writer:
        agg = run_climatology(
            model,
            ds,
            normalizer=normalizer,
            device=device,
            ic_indices=[0],
            max_step=max_step,
            n_bins=2,
            steps_per_bin=1,
            ensemble_size=1,
            perturber=None,
            has_diagnostic=model.has_diagnostic,
            seed=0,
            track_bins=True,
            track_binned_variance=True,
            forecast_chunk_steps=chunk_steps,
            forecast_writer=writer,
            forecast_output_dir=str(tmp_path),
            model_name="PanguPlasimLegacy",
            run_name="smoke",
            forecast_extension="zarr",
            include_ic_in_forecast=True,
        )

    # --- Aggregator sanity --------------------------------------------------
    for grp in ("surface", "upper_air"):
        pred_mean = agg["pred"][grp]["mean"].numpy()
        truth_mean = agg["truth"][grp]["mean"].numpy()
        assert np.isfinite(pred_mean).all(), f"{grp} pred_mean not finite"
        assert np.isfinite(truth_mean).all(), f"{grp} truth_mean not finite"
        pred_var = agg["pred"][grp]["var"].numpy()
        assert np.isfinite(pred_var).all(), f"{grp} pred_var not finite"
        assert (pred_var >= 0).all(), f"{grp} pred_var has negative entries"
        # Binned mean shape: (n_bins=2, *group_shape)
        assert agg["pred"][grp]["binned"].shape[0] == 2
        # Binned variance same shape as binned mean.
        assert agg["pred"][grp]["binned_var"].shape[0] == 2

    # --- Chunked forecast files --------------------------------------------
    forecast_paths = agg["forecast_paths"]
    assert len(forecast_paths) >= 1, "no forecast chunk files written"
    # Every chunk file should materialize at the expected path.
    for p in forecast_paths:
        assert Path(p).exists(), f"chunk file missing: {p}"
        # Filename carries the model + run + ic + chunk index.
        bn = Path(p).name
        assert "PanguPlasimLegacy__" in bn
        assert "smoke__ic0__" in bn
        assert "chunk" in bn
        assert bn.endswith(".zarr")

    # --- "No repeated dates" invariant --------------------------------------
    # Aggregate the step indices across all chunk files; verify each
    # step appears in exactly one file. Steps 0..max_step should all
    # appear (IC at step 0 in chunk 0, predictions 1..max_step in
    # chunks 0..N).
    seen_steps: dict[int, str] = {}
    for p in forecast_paths:
        out = xr.open_zarr(p)
        for s in out["step"].values.tolist():
            assert s not in seen_steps, (
                f"step {s} appears in both {seen_steps[s]} and {p}"
            )
            seen_steps[s] = p
    assert set(seen_steps.keys()) == set(range(0, max_step + 1)), (
        f"missing steps; got {sorted(seen_steps.keys())}"
    )

    # --- IC at frame 0 of chunk 0 ------------------------------------------
    sorted_paths = sorted(forecast_paths, key=lambda p: Path(p).name)
    chunk0 = xr.open_zarr(sorted_paths[0])
    assert int(chunk0["step"].values[0]) == 0, (
        f"chunk 0 first step should be 0 (IC), got {chunk0['step'].values[0]}"
    )
