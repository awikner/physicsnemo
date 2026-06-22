# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU smoke test for the Phase 4a rollout validator.

End-to-end on Delta `gpuA40x4-interactive`: PLASIM fixture → tiny
PanguPlasimLegacy → :class:`RolloutValidator` runs a 2-step rollout for
two ICs (deterministic + a 3-member IC-perturbation ensemble). Verifies
the validator produces a finite metric dict and that the ensemble path
exercises the GPU correctly.

Skipped when the Delta fixture isn't staged or the PLASIM stats files
aren't accessible — same gating as the training smoke tests.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest
import torch

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from validate import (  # noqa: E402
    GaussianIC,
    RolloutValidator,
)

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


# A4-fitting tiny model. Same shape rationale as test_smoke_single_gpu.py.
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
def test_rollout_validator_smoke_deterministic_one_gpu():
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = torch.device("cuda:0")
    ds, normalizer, model = _build_tiny_model_and_dataset(device)

    rv = RolloutValidator(
        dataset=ds,
        log_steps=[1, 2],
        device=device,
        ensemble_size=1,
        has_diagnostic=model.has_diagnostic,
        batch_size=1,
        max_initial_conditions=2,
        ic_stride=1,
        normalizer=normalizer,
        seed=0,
    )
    out = rv.run(model, epoch=0)

    # Must have RMSE entries for surface + upper_air at both log steps.
    must_have = {
        "rmse_step1_surface",
        "rmse_step2_surface",
        "rmse_step1_upper_air",
        "rmse_step2_upper_air",
    }
    assert must_have.issubset(out.keys()), (out.keys(), must_have)
    for k, v in out.items():
        assert isinstance(v, float)
        assert torch.isfinite(torch.tensor(v)), f"non-finite metric {k}={v}"
        assert v >= 0.0, f"negative RMSE/ACC magnitude not meaningful: {k}={v}"


@pytest.mark.smoke
@pytest.mark.cuda
@_skip_no_fixture
@_skip_no_stats
def test_rollout_validator_smoke_ensemble_perturbation_one_gpu():
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = torch.device("cuda:0")
    ds, normalizer, model = _build_tiny_model_and_dataset(device)

    # 3-member ensemble via small Gaussian noise on the surface initial
    # condition only. Per-rank effective batch = 1 IC × 3 members = 3.
    perturber = GaussianIC(scales={"surface_in": 0.01})
    rv = RolloutValidator(
        dataset=ds,
        log_steps=[1, 2],
        device=device,
        ensemble_size=3,
        perturber=perturber,
        has_diagnostic=model.has_diagnostic,
        batch_size=1,
        max_initial_conditions=1,
        ic_stride=1,
        normalizer=normalizer,
        seed=0,
    )
    out = rv.run(model, epoch=0)

    assert {"rmse_step1_surface", "rmse_step2_surface"}.issubset(out.keys())
    for v in out.values():
        assert torch.isfinite(torch.tensor(v))
        assert v >= 0.0
