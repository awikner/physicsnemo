# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU smoke test for the PanguPlasimLegacy training recipe.

End-to-end: PLASIM Zarr fixture → datapipe with normalizer + NaN fill →
PanguPlasimLegacy.forward → PanguPlasimLoss → AdamW + OneCycleLR → EMA update
→ checkpoint roundtrip. Skipped when the Delta fixture isn't staged.

Per the smoke-test contract in ``hpc/delta.md`` this runs on real disk-backed
data on the GPU; it is the recipe's primary integration sentinel.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from ema import ModelEMA  # noqa: E402
from loss import PanguPlasimLoss  # noqa: E402
from train_loop import make_optimizer, make_scheduler, train_step  # noqa: E402

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.plasim import (
        NanFillTransform,
        PlasimClimateDatapipe,
        PlasimClimateDataset,
        PlasimNormalizer,
    )
    from physicsnemo.experimental.models.pangu_plasim import PanguPlasimLegacy

import physicsnemo

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
    not _HAS_FIXTURE, reason="$AI_ROSSBY_TEST_DATA/plasim/smoke_month.zarr missing"
)
_skip_no_stats = pytest.mark.skipif(
    not _HAS_STATS, reason="PLASIM stats files missing"
)


# Tiny PanguPlasimLegacy that fits an A40 and finishes a couple of steps in
# under a minute. Mirrors the test_pangu_plasim.py smoke config but uses the
# fixture's actual channel groups (read from the Zarr layout at runtime).
_BACKBONE_KWARGS = dict(
    patch_size=[2, 4, 4],
    depths=[1, 1, 1, 1],
    num_heads=[2, 4, 4, 2],
    embed_dim=64,
    window_size=[2, 4, 8],
    checkpointing=0,
)


@pytest.mark.smoke
@pytest.mark.cuda
@_skip_no_fixture
@_skip_no_stats
def test_pangu_plasim_legacy_smoke_train_steps(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = torch.device("cuda:0")

    # --- Build dataset + transforms ----------------------------------------
    ds_for_layout = PlasimClimateDataset(_fixture_path())
    normalizer = PlasimNormalizer.from_dataset(
        ds_for_layout, mean_path=_MEAN_PATH, std_path=_STD_PATH
    )
    nan_fill = NanFillTransform(
        constant_boundary_variables=ds_for_layout.layout.constant_boundary_variables,
        varying_boundary_variables=ds_for_layout.layout.varying_boundary_variables,
        default=0.0,
    )
    pipe = PlasimClimateDatapipe(
        _fixture_path(),
        forecast_lead_times=[1],
        normalizer=normalizer,
        nan_fill=None,
        batch_size=1,
        num_samples_per_epoch=2,
        shuffle=True,
        num_workers=0,
        device=device,
        seed=0,
    )
    pipe.dataset.transform = nan_fill

    # --- Build model --------------------------------------------------------
    model = PanguPlasimLegacy(
        surface_variables=ds_for_layout.layout.surface_variables,
        upper_air_variables=ds_for_layout.upper_air_variable_names,
        constant_boundary_variables=ds_for_layout.layout.constant_boundary_variables,
        varying_boundary_variables=ds_for_layout.layout.varying_boundary_variables,
        diagnostic_variables=ds_for_layout.layout.diagnostic_variables,
        levels=ds_for_layout.sigma_levels,
        horizontal_resolution=list(ds_for_layout.horizontal_resolution),
        **_BACKBONE_KWARGS,
    ).to(device)

    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 3e-6,
            "scheduler": "OneCycleLR",
            "oc_pct_start": 0.1,
            "oc_div_factor": 1e5,
            "oc_final_div_factor": 0.00025,
        }
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, total_steps=4)
    loss_fn = PanguPlasimLoss(
        surface_variables=ds_for_layout.layout.surface_variables,
        upper_air_variable_names=ds_for_layout.upper_air_variable_names,
        diagnostic_variables=ds_for_layout.layout.diagnostic_variables,
        num_lat=ds_for_layout.horizontal_resolution[0],
        loss_type="l1",
    ).to(device)
    ema = ModelEMA(model, decay=0.999, warmup_epochs=1)

    # --- Take 2 train steps -------------------------------------------------
    model.train()
    n_steps = 0
    for batch in pipe:
        out = train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            has_diagnostic=model.has_diagnostic,
        )
        assert torch.isfinite(out["loss"]).all(), "NaN/inf loss on real data"
        ema.update(model, epoch=0)
        n_steps += 1
    assert n_steps == 2

    # --- Checkpoint roundtrip ----------------------------------------------
    # Use the Module .mdlus path (MOD-008c) rather than physicsnemo.utils.save_checkpoint,
    # which expects DistributedManager to be initialized. The recipe's train.py
    # initializes it explicitly; the smoke test exercises the model-only roundtrip.
    ckpt_path = tmp_path / "pangu_plasim_legacy_smoke.mdlus"
    model.save(str(ckpt_path))
    loaded = (
        physicsnemo.Module.from_checkpoint(str(ckpt_path)).to(device).eval()
    )
    src = dict(model.named_parameters())
    dst = dict(loaded.named_parameters())
    a_name = next(iter(src))
    assert torch.equal(src[a_name].detach(), dst[a_name].detach())

    del model, loaded, normalizer, pipe
    torch.cuda.empty_cache()
