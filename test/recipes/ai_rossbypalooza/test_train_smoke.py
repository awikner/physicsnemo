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
    # The store must say where the gate was actually supervised: outside that
    # region the weights and biases are untrained extrapolation.
    assert "supervised_region_box" in ds.attrs
    assert "untrained extrapolation" in ds.attrs["supervised_region_note"]
    assert ds.attrs["mix_space"] == "physical"
    assert ds.attrs["split"] == "val"
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


# --------------------------------------------------------------------------- #
# Probabilistic (noise + CRPS) arm
# --------------------------------------------------------------------------- #


def _crps_overrides(max_epochs=2):
    return {
        "model": {"params": {"noise_dim": 4}},
        "loss": {"name": "regional_crps", "alpha": 0.95, "scale_mm": 4.0},
        "training": {"max_epochs": max_epochs, "ens_size": 2},
        "validation": {"every_n_epochs": 1, "ens_size": 3, "noise_seed": 0},
    }


@pytest.mark.slow
def test_train_smoke_crps_and_resume(smoke_cfg):
    """The noise-conditioned CRPS arm trains end to end, validates with the
    ensemble metrics, checkpoints, and resumes."""
    import train as train_mod
    from pathlib import Path

    torch.manual_seed(0)
    cfg = OmegaConf.merge(smoke_cfg, _crps_overrides(max_epochs=2))
    train_mod.run(cfg)
    assert list(Path("checkpoints").glob("*")), "no checkpoint written"
    cfg2 = OmegaConf.merge(smoke_cfg, _crps_overrides(max_epochs=3))
    train_mod.run(cfg2)


@pytest.mark.slow
def test_crps_config_guards(smoke_cfg):
    """The three ensemble knobs must agree; each mismatch fails actionably."""
    import train as train_mod

    with pytest.raises(ValueError, match="noise_dim"):
        train_mod.run(
            OmegaConf.merge(
                smoke_cfg,
                {"loss": {"name": "regional_crps", "alpha": 0.95}},
            )
        )
    with pytest.raises(ValueError, match="deterministic"):
        train_mod.run(
            OmegaConf.merge(smoke_cfg, {"model": {"params": {"noise_dim": 4}}})
        )
    with pytest.raises(ValueError, match="ens_size"):
        train_mod.run(
            OmegaConf.merge(
                smoke_cfg,
                {
                    "model": {"params": {"noise_dim": 4}},
                    "loss": {"name": "regional_crps"},
                    "training": {"ens_size": 1},
                },
            )
        )


@pytest.mark.slow
def test_warm_start_from_deterministic(smoke_cfg, tmp_path):
    """A deterministic checkpoint warm-starts the noise_dim gate; the zero-
    initialized condition embedder makes the first forward's members
    identical. A patch-size change is refused, not silently part-loaded."""
    import train as train_mod
    from pathlib import Path

    from mowe_precip import MoWEPrecipGate

    torch.manual_seed(0)
    cfg = OmegaConf.merge(
        smoke_cfg,
        {"training": {"max_epochs": 1}, "validation": {"every_n_epochs": 1}},
    )
    train_mod.run(cfg)
    det_ckpts = Path("checkpoints").resolve()
    assert list(det_ckpts.glob("*.mdlus"))

    # The zero-init property, via the same loader train.py uses.
    gate = MoWEPrecipGate(
        input_size=(8, 8),
        in_channels=3,
        n_experts=2,
        patch_size=(2, 2),
        hidden_size=32,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,  # must match smoke_cfg's model params
        attention_backend="timm",
        noise_dim=4,
    )
    src = train_mod._latest_mdlus(det_ckpts)
    gate.load(str(src), strict=False)
    x = torch.randn(2, 2, 3, 8, 8)
    mask = torch.ones(2, 2)
    noise = torch.randn(2, 3, 4) * 10.0
    with torch.no_grad():
        w, b = gate(x, mask, torch.tensor([8, 8]), noise)
    torch.testing.assert_close(w[:, 0], w[:, 1])
    torch.testing.assert_close(b[:, 0], b[:, 2])

    # End-to-end warm start in a fresh run dir.
    run2 = tmp_path / "run2"
    run2.mkdir()
    import os

    cwd = os.getcwd()
    os.chdir(run2)
    try:
        cfg2 = OmegaConf.merge(smoke_cfg, _crps_overrides(max_epochs=1))
        cfg2 = OmegaConf.merge(cfg2, {"training": {"init_from": str(det_ckpts)}})
        train_mod.run(cfg2)
        assert list(Path("checkpoints").glob("*")), "warm-started run saved nothing"
    finally:
        os.chdir(cwd)

    # Across patch sizes the load must refuse loudly.
    run3 = tmp_path / "run3"
    run3.mkdir()
    os.chdir(run3)
    try:
        cfg3 = OmegaConf.merge(smoke_cfg, _crps_overrides(max_epochs=1))
        cfg3 = OmegaConf.merge(
            cfg3,
            {
                "model": {"params": {"patch_size": [4, 4]}},
                "training": {"init_from": str(det_ckpts)},
            },
        )
        with pytest.raises(RuntimeError, match="patch_size"):
            train_mod.run(cfg3)
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_train_smoke_fss_amse_and_gate_tv(smoke_cfg, tmp_path):
    """The FSS composite (fixed thresholds), the AMSE loss, and the gate-map
    TV penalty all train end to end on the synthetic fixture."""
    import os

    import train as train_mod

    torch.manual_seed(0)
    fss_cfg = OmegaConf.merge(
        smoke_cfg,
        {
            "loss": {
                "name": "regional_fss",
                "anchor": {"name": "regional_mse", "space": "normalized",
                           "lat_weighted": True},
                "thresholds": {"kind": "fixed", "values_mm": [1.0, 5.0]},
                "windows": [3],
                "fss_weight": 0.3,
                "ramp_epochs": 2,
            },
            "training": {"max_epochs": 2, "gate_tv_weight": 0.01},
            "validation": {"every_n_epochs": 1},
        },
    )
    train_mod.run(fss_cfg)

    # Fresh run dir: the AMSE arm must not resume the FSS arm's checkpoints.
    amse_dir = tmp_path / "run_amse"
    amse_dir.mkdir()
    cwd = os.getcwd()
    os.chdir(amse_dir)
    try:
        amse_cfg = OmegaConf.merge(
            smoke_cfg,
            {
                "run_name": "smoke_amse",
                "loss": {"name": "regional_amse", "windows": [3], "scale_mm": 9.3},
                "training": {"max_epochs": 2},
            },
        )
        train_mod.run(amse_cfg)
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_ensemble_inference_writes_quantiles(smoke_cfg):
    """infer_mowe on a noise_dim checkpoint writes the ensemble mean into the
    usual variable plus member quantiles (and members when asked)."""
    from pathlib import Path

    import train as train_mod

    torch.manual_seed(0)
    cfg = OmegaConf.merge(smoke_cfg, _crps_overrides(max_epochs=2))
    train_mod.run(cfg)
    best = Path("checkpoints_best")
    assert list(best.glob("*")), "no best checkpoint from the CRPS smoke run"

    out = Path("forecasts_ens.zarr")
    import tools.infer_mowe as infer

    icfg = OmegaConf.merge(
        cfg,
        {"checkpoint": str(best.resolve()), "out": str(out.resolve()),
         "split": "val", "save_gate": True,
         "ens_size": 3, "noise_seed": 1, "save_members": True},
    )
    infer.main.__wrapped__(icfg)

    ds = xr.open_zarr(out)
    assert ds.attrs["ens_size"] == 3
    for v in ("total_precipitation_24hr", "total_precipitation_24hr_q10",
              "total_precipitation_24hr_q50", "total_precipitation_24hr_q90",
              "total_precipitation_24hr_members", "gate_weights"):
        assert v in ds, v
    assert ds["total_precipitation_24hr_members"].sizes["member"] == 3
    mean = ds["total_precipitation_24hr"].values
    members = ds["total_precipitation_24hr_members"].values
    finite = np.isfinite(mean)
    assert finite.any()
    # The written mean IS the member mean.
    np.testing.assert_allclose(
        mean[finite], members.mean(axis=2)[finite], rtol=1e-5, atol=1e-6
    )
    q10 = ds["total_precipitation_24hr_q10"].values
    q90 = ds["total_precipitation_24hr_q90"].values
    assert (q10[finite] <= q90[finite] + 1e-6).all()
