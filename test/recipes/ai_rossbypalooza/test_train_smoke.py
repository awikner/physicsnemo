# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end CPU smoke test: two fake experts (one per schema) + fake
IMERG, a tiny gate, a few epochs through train.run(), checkpoint resume."""

from __future__ import annotations

import numpy as np
import xarray as xr
import pytest
import torch

pytest.importorskip("physicsnemo", reason="smoke test needs physicsnemo")
pytest.importorskip("hydra", reason="smoke test needs hydra/omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from datapipes.testing import (  # noqa: E402
    write_imerg_store,
    write_schema_a_store,
    write_schema_b_store,
    write_stats_store,
)


@pytest.fixture()
def smoke_cfg(tmp_path, monkeypatch):
    """A full training config over synthetic stores, run inside tmp_path."""
    a_root = tmp_path / "experts" / "model_a"
    b_root = tmp_path / "experts" / "model_b"
    write_schema_a_store(
        a_root / "2001.zarr",
        year=2001,
        init_dates=[(6, d) for d in (1, 4, 8, 11, 15, 18)],
        vars_6h=("2t", "z_500"),
        vars_daily=("tp",),
        lead_hours=range(168, 361, 6),
        lead_days=range(7, 16),
    )
    write_schema_b_store(
        b_root / "2001.zarr",
        year=2001,
        init_dates=[(6, d) for d in (1, 5, 9, 13, 17)],
        pressure_levels=(850.0, 500.0),
        n_lead=17,
    )
    write_imerg_store(tmp_path / "imerg" / "2001.zarr", year=2001, months=(6, 7))
    era5 = write_stats_store(
        tmp_path / "era5_stats.zarr",
        surface={"2m_temperature": (280.0, 15.0)},
        upper={"geopotential": {500.0: (54000.0, 3000.0), 850.0: (14000.0, 1500.0)}},
    )
    precip = write_stats_store(
        tmp_path / "imerg_stats.zarr",
        surface={"total_precipitation_24hr": (5.0, 10.0)},
    )
    # SEEPS climatology on the tiny grid.
    import xarray as xr

    from datapipes.testing import GRID_LAT, GRID_LON

    clim = xr.Dataset(
        {
            "p1": (("month", "lat", "lon"), np.full((12, 8, 8), 0.5, "f4")),
            "t2": (("month", "lat", "lon"), np.full((12, 8, 8), 5.0, "f4")),
            "clim_mean": (
                ("month", "lat", "lon"),
                np.full((12, 8, 8), 3.0, "f4"),
            ),
        },
        coords={
            "month": np.arange(1, 13),
            "lat": GRID_LAT,
            "lon": GRID_LON,
        },
        attrs={"dry_threshold_mm": 0.25},
    )
    clim.to_zarr(tmp_path / "seeps_clim.zarr", mode="w", zarr_format=3,
                 consolidated=True)

    cfg = OmegaConf.create(
        {
            "run_name": "smoke",
            "seed": 0,
            "start_epoch": 0,
            "checkpoint_save_interval": 2,
            "region": {"lat": [-4.0, 4.0], "lon": [0.0, 360.0]},
            "wandb": {"enabled": False},
            "dataset": {
                "master_channels": ["z/500", "2t"],
                "truth": {"root": str(tmp_path / "imerg")},
                "normalization": {
                    "dynamical_mean": str(era5),
                    "dynamical_std": str(era5),
                    "precip_stats": str(precip),
                },
                "experts": [
                    {
                        "name": "model_a",
                        "schema": "dsi",
                        "root": str(a_root),
                        "precip": {
                            "var": "tp", "axis": "daily",
                            "kind": "accum", "units": "mm",
                        },
                    },
                    {
                        "name": "model_b",
                        "schema": "consolidated",
                        "root": str(b_root),
                        "precip": {
                            "var": "total_precipitation_24hr", "axis": "daily",
                            "kind": "accum", "units": "mm",
                        },
                    },
                ],
                "train": {
                    "years": [2001, 2001],
                    "init_months": [6],
                    "lead_days": [8, 9],
                    "min_experts": 1,
                },
                "val": {
                    "years": [2001, 2001],
                    "init_months": [6],
                    "lead_days": [8, 9],
                    "min_experts": 1,
                },
                "loader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                    "shuffle": True,
                    "num_samples_per_epoch": None,
                    "zarr_concurrency": 2,
                },
            },
            "model": {
                "name": "mowe_precip",
                "params": {
                    "patch_size": [2, 2],
                    "hidden_size": 32,
                    "depth": 1,
                    "num_heads": 2,
                    "mlp_ratio": 2.0,
                    "attention_backend": "timm",
                    "noise_dim": None,
                },
            },
            "loss": {"name": "regional_mse", "space": "normalized",
                     "lat_weighted": True},
            "training": {
                "max_epochs": 3,
                "warmup_epochs": 1,
                "min_lr_ratio": 0.02,
                "amp": "none",
                "grad_clip_norm": 1.0,
                "expert_dropout": 0.2,
                "optimizer": {"lr": 3.0e-3, "betas": [0.9, 0.999],
                              "weight_decay": 0.01},
            },
            "validation": {
                "enabled": True,
                "every_n_epochs": 3,
                "seeps_climatology": str(tmp_path / "seeps_clim.zarr"),
            },
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    return cfg


@pytest.mark.slow
def test_train_smoke_and_resume(smoke_cfg, capsys):
    import train as train_mod

    torch.manual_seed(0)
    train_mod.run(smoke_cfg)

    from pathlib import Path

    ckpts = list(Path("checkpoints").glob("*"))
    assert ckpts, "no checkpoint written"
    npz = list(Path(".").glob("weight_maps_epoch*.npz"))
    assert npz, "no validation weight maps written"
    maps = np.load(npz[0])
    key = list(maps.keys())[0]
    assert maps[key].shape == (2, 8, 8)  # (E, H, W) weight maps

    # Resume: bump epochs and run again from the checkpoint.
    cfg2 = OmegaConf.merge(
        smoke_cfg, {"training": {"max_epochs": 4}}
    )
    train_mod.run(cfg2)


@pytest.mark.slow
def test_gate_learns_on_synthetic_signal(smoke_cfg):
    """Loss decreases over a few epochs on the synthetic data."""
    import train as train_mod
    from datapipes.factory import build_dataset
    from losses import build_loss
    from mowe_precip import MoWEPrecipGate, mix

    ds = build_dataset(smoke_cfg.dataset, "train")
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = MoWEPrecipGate(
        input_size=(8, 8),
        in_channels=3,
        n_experts=2,
        patch_size=(2, 2),
        hidden_size=32,
        depth=1,
        num_heads=2,
        attention_backend="timm",
    )
    box = (-4.0, 4.0, 0.0, 360.0)
    loss_fn = build_loss(
        {"name": "regional_mse"},
        lat=ds.lat, lon=ds.lon, box=box,
        precip_mean=ds.precip_mean, precip_std=ds.precip_std,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    torch.manual_seed(0)
    for _ in range(6):
        epoch_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            w, b = model(
                batch["expert_inputs"], batch["expert_mask"], batch["lead_days"]
            )
            pred = mix(w, b, batch["expert_inputs"][:, :, 0])
            loss = loss_fn(pred, batch["target"], batch["target_mm"])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    del train_mod  # imported to assert the module loads alongside


@pytest.mark.slow
def test_best_checkpoint_and_early_stopping(smoke_cfg, monkeypatch):
    """Best weights land in their own directory, and a validation loss that
    stops improving ends the run before max_epochs.

    The validator is stubbed to report a worsening loss -- on the synthetic
    fixture the real loss improves every epoch, so early stopping would
    (correctly) never fire.
    """
    import train as train_mod
    import validation as validation_mod
    from pathlib import Path

    calls = {"n": 0}

    def fake_run(self, model, loader):
        calls["n"] += 1
        # 1.0, then strictly worse every epoch.
        return {"loss": 1.0 + 0.1 * (calls["n"] - 1)}, {"weight_maps": {}}

    monkeypatch.setattr(validation_mod.MixtureValidator, "run", fake_run)

    torch.manual_seed(0)
    cfg = OmegaConf.merge(
        smoke_cfg,
        {
            "training": {
                "max_epochs": 8,
                "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0},
                "ema": {
                    "enabled": True,
                    "decay": 0.9,
                    "warmup_epochs": 0,
                    "validate_with_ema": True,
                },
            },
            "validation": {"every_n_epochs": 1},
        },
    )
    train_mod.run(cfg)

    # epoch 0 sets the best; epochs 1 and 2 do not improve -> stop at epoch 2.
    assert calls["n"] == 3, f"expected 3 validations before stopping, got {calls['n']}"
    assert list(Path("checkpoints_best").glob("*")), "no best-weights checkpoint"
    assert list(Path("checkpoints").glob("*")), "no periodic/final checkpoint"


@pytest.mark.slow
def test_ema_disabled_path_still_trains(smoke_cfg):
    """EMA off + early stopping off is the plain path and must still work."""
    import train as train_mod
    from pathlib import Path

    torch.manual_seed(0)
    cfg = OmegaConf.merge(
        smoke_cfg,
        {
            "training": {
                "max_epochs": 2,
                "early_stopping": {"enabled": False},
                "ema": {"enabled": False},
            }
        },
    )
    train_mod.run(cfg)
    assert list(Path("checkpoints").glob("*")), "no checkpoint written"


@pytest.mark.slow
def test_inference_writes_gate_forecasts(smoke_cfg, monkeypatch):
    """infer_mowe replays a split and writes a dense (init, lead, lat, lon)
    zarr of the mixture in mm/day, leaving pairs absent from the index NaN."""
    from pathlib import Path

    import train as train_mod

    torch.manual_seed(0)
    # every_n_epochs must be 1: the best checkpoint is only written when a
    # validation pass runs, so the default (3) with max_epochs=1 writes none.
    cfg = OmegaConf.merge(
        smoke_cfg,
        {"training": {"max_epochs": 2}, "validation": {"every_n_epochs": 1}},
    )
    train_mod.run(cfg)
    best = Path("checkpoints_best")
    assert list(best.glob("*")), "training produced no best checkpoint"

    out = Path("forecasts.zarr")
    import tools.infer_mowe as infer

    icfg = OmegaConf.merge(
        smoke_cfg,
        {"checkpoint": str(best.resolve()), "out": str(out.resolve()),
         "split": "val", "save_gate": True},
    )
    infer.main.__wrapped__(icfg)          # bypass the hydra decorator

    ds = xr.open_zarr(out)
    for v in ("total_precipitation_24hr", "gate_weights", "gate_biases"):
        assert v in ds, v
    p = ds["total_precipitation_24hr"]
    assert p.dims == ("init_time", "lead_time", "lat", "lon")
    assert p.sizes["lat"] == 8 and p.sizes["lon"] == 8
    assert p.attrs["units"] == "mm/day"
    finite = np.isfinite(p.values)
    assert finite.any(), "no forecasts written"
    assert (p.values[finite] >= 0).all(), "negative rainfall written"
    # Weights over live experts sum to 1 wherever a forecast exists.
    w = ds["gate_weights"].values
    idx = np.isfinite(w).all(axis=2)
    np.testing.assert_allclose(np.nansum(w, axis=2)[idx], 1.0, rtol=1e-4)
    # Biases share the grid and are written for exactly the same pairs.
    b = ds["gate_biases"].values
    assert b.shape == w.shape
    np.testing.assert_array_equal(np.isfinite(w), np.isfinite(b))
    assert ds["gate_biases"].attrs["units"] == "mm/day"   # physical mixing
    assert np.abs(b[np.isfinite(b)]).max() < 1e4          # finite, sane scale
