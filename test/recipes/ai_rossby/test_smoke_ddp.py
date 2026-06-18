# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-GPU DDP smoke test for PanguPlasimLegacy training.

Run with::

    torchrun --standalone --nproc-per-node=2 -m pytest --multigpu-static \\
        test/recipes/ai_rossby/test_smoke_ddp.py -x

When invoked WITHOUT torchrun the test is skipped (single-rank run would
defeat the point). The test asserts that:

* Both ranks build the model + datapipe + optimizer without error.
* A forward + backward + step + AllReduce succeeds on real PLASIM data.
* The optimizer's parameter values remain bit-identical across ranks after
  one step (DDP's all-reduce on grads + identical lr_schedule).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

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

from physicsnemo.distributed import DistributedManager

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
_IN_TORCHRUN = "TORCHELASTIC_RUN_ID" in os.environ or "RANK" in os.environ

_skip_no_fixture = pytest.mark.skipif(
    not _HAS_FIXTURE, reason="$AI_ROSSBY_TEST_DATA/plasim/smoke_month.zarr missing"
)
_skip_no_stats = pytest.mark.skipif(
    not _HAS_STATS, reason="PLASIM stats files missing"
)
_skip_not_torchrun = pytest.mark.skipif(
    not _IN_TORCHRUN,
    reason="DDP smoke test must be launched via torchrun --nproc-per-node>=2",
)


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
@pytest.mark.multigpu
@_skip_not_torchrun
@_skip_no_fixture
@_skip_no_stats
def test_pangu_plasim_legacy_ddp_smoke():
    if not torch.cuda.is_available():
        pytest.skip("DDP smoke test requires CUDA")

    DistributedManager.initialize()
    dm = DistributedManager()
    assert dm.world_size >= 2, "DDP smoke test requires world_size>=2"

    device = dm.device

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
        num_samples_per_epoch=4,  # 2 per rank
        shuffle=True,
        num_workers=0,
        device=device,
        seed=0,
        distributed=True,
    )
    pipe.dataset.transform = nan_fill

    # --- Build model + DDP --------------------------------------------------
    torch.manual_seed(0)  # identical init across ranks
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
    ddp = DistributedDataParallel(
        model,
        device_ids=[dm.local_rank],
        output_device=device,
        broadcast_buffers=dm.broadcast_buffers,
        find_unused_parameters=dm.find_unused_parameters,
    )

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

    # --- Take one step on each rank ----------------------------------------
    ddp.train()
    n_steps = 0
    for batch in pipe:
        out = train_step(
            model=ddp,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            has_diagnostic=model.has_diagnostic,
        )
        assert torch.isfinite(out["loss"]).all()
        n_steps += 1
        if n_steps >= 2:
            break

    # --- Verify parameter sync after the step -----------------------------
    # DDP all-reduces gradients each step; with identical init + identical
    # scheduler step, params should be byte-identical across ranks.
    name = next(iter(dict(model.named_parameters()).keys()))
    local = dict(model.named_parameters())[name].detach()
    gathered = [torch.empty_like(local) for _ in range(dm.world_size)]
    dist.all_gather(gathered, local)
    for r, g in enumerate(gathered):
        assert torch.equal(g, local), (
            f"rank {dm.rank} param {name} diverged from rank {r}"
        )

    dist.barrier()
    del ddp, model, normalizer, pipe
    torch.cuda.empty_cache()
