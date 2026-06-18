# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU smoke test for the SfnoPlasim training recipe.

Mirrors test_smoke_single_gpu (PanguPlasimLegacy) and test_smoke_vae_single_gpu
(PanguPlasim with VAE-KL), but for the SFNO model + raw_l2 loss combination.
Validates the recipe's three-way model_type switch end-to-end.

End-to-end: PLASIM Zarr fixture → datapipe with normalizer + NaN fill →
SfnoPlasim.forward → PanguPlasimLoss(latitude_weighted=False) →
AdamW + LinearWarmupCosineAnnealingLR.
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
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim


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


# Tiny SfnoPlasim that finishes a forward+backward in well under a minute.
_BACKBONE_KWARGS = dict(
    spectral_transform="sht",
    filter_type="linear",
    operator_type="dhconv",
    embed_dim=32,
    num_layers=2,
    num_blocks=2,
    spectral_layers=2,
    encoder_layers=1,
    normalization_layer="instance_norm",
)


@pytest.mark.smoke
@pytest.mark.cuda
@_skip_no_fixture
@_skip_no_stats
def test_sfno_plasim_smoke_train_steps(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = torch.device("cuda:0")

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

    model = SfnoPlasim(
        surface_variables=ds_for_layout.layout.surface_variables,
        upper_air_variables=ds_for_layout.upper_air_variable_names,
        constant_boundary_variables=ds_for_layout.layout.constant_boundary_variables,
        varying_boundary_variables=ds_for_layout.layout.varying_boundary_variables,
        diagnostic_variables=ds_for_layout.layout.diagnostic_variables,
        levels=ds_for_layout.sigma_levels,
        horizontal_resolution=list(ds_for_layout.horizontal_resolution),
        **_BACKBONE_KWARGS,
    ).to(device)

    # SFNO + raw_l2 (no cos-lat weighting) + cosine warmup — the
    # PanguWeather SFNO_PLASIM_H5_DERECHO_5412 default combo.
    cfg = OmegaConf.create(
        {
            "optimizer_type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 3e-6,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "num_warmup_steps": 1,
            "warmup_start_lr": 1e-8,
            "eta_min": 1e-8,
        }
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, total_steps=4)
    loss_fn = PanguPlasimLoss(
        surface_variables=ds_for_layout.layout.surface_variables,
        upper_air_variable_names=ds_for_layout.upper_air_variable_names,
        diagnostic_variables=ds_for_layout.layout.diagnostic_variables,
        num_lat=ds_for_layout.horizontal_resolution[0],
        loss_type="l2",
        latitude_weighted=False,  # raw_l2
    ).to(device)
    ema = ModelEMA(model, decay=0.999, warmup_epochs=1)

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
            # SFNO is deterministic; KL stays at 0 regardless of weight.
            vae_kl_weight=0.0,
        )
        assert torch.isfinite(out["loss"]).all(), "NaN/inf loss on real data"
        # SFNO returns zero-tensor placeholders → KL slot stays at 0.
        assert float(out["vae_kl"]) == 0.0
        ema.update(model, epoch=0)
        n_steps += 1
    assert n_steps == 2

    del model, normalizer, pipe
    torch.cuda.empty_cache()
