# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12f — the recipe's ocean-channel + warm-start wiring.

Three seams get exercised here, all of them the kind that fail silently:

* ``_build_loader`` must take the alignment and the extra boundary view from
  the *model*, so the loader can't be shifted differently from the pack.
* ``_build_scheduler_loss`` must inject ``nocean`` / ``ocean_grid_indices``
  from the model rather than from ``cfg.loss``.
* ``_train_step`` must feed the scheduler the ``[1:]`` boundary view and
  report the ocean loss separately.

Plus ``load_partial_weights``, whose whole point is the report: a no-ocean →
ocean warm start is expected to skip *zero* keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_diffusion import (  # noqa: E402
    _build_loader,
    _build_scheduler_loss,
    _pack_window,
    _train_step,
)
from train_loop import (  # noqa: E402
    adopt_ocean_contract,
    load_partial_weights,
)

from physicsnemo.experimental.models.amip_si import RollingDiTWrapper  # noqa: E402

_SURFACE = ["skin_temperature", "surface_pressure"]
_UA = ["temperature"]
_DIAG = ["PRATEsfc_24h"]
_CONST = ["geopotential_at_surface", "land_sea_mask"]
_VARY = [
    "global_mean_co2",
    "DSWRFtoa_24h_lead",
    "sea_surface_temperature_monthly_interp",
    "sea_ice_cover_monthly_interp",
]
_OCEAN = _VARY[2:]
_LEVELS = [500.0, 850.0]
_H, _W = 8, 16
_N_TIME = 16
_W_WINDOW = 3


class _Base:
    """Base dataset whose fields encode their own time index."""

    n_time = _N_TIME
    layout = None

    def __getitem__(self, index):
        t, _lead = index if isinstance(index, tuple) else (index, 1)
        t = float(t)
        return {
            "surface_in": torch.full((len(_SURFACE), _H, _W), t),
            "upper_air_in": torch.full((len(_UA), len(_LEVELS), _H, _W), t),
            "diagnostic": torch.full((len(_DIAG), _H, _W), t),
            # Channel c of the boundary at time t reads t + c/10 — so a frame
            # and a channel are both identifiable in one number.
            "varying_boundary": torch.stack(
                [torch.full((_H, _W), t + c / 10.0) for c in range(len(_VARY) - 1)]
            ),
            "constant_boundary": torch.full((len(_CONST), _H, _W), 7.0),
            "calendar": torch.full((3,), 200.0 + t),
        }

    def __len__(self):
        return self.n_time


def _cfg():
    return OmegaConf.create(
        {
            "seed": 0,
            "dataset": {
                "batch_size": 2,
                "num_workers": 0,
                "prefetch_factor": 2,
                "persistent_workers": False,
                "pin_memory": False,
                "shuffle": False,
                "forecast_lead_times": [1],
            },
            "loss": {
                "_target_": "physicsnemo.experimental.diffusion.ERDMScheduler",
                "window_size": _W_WINDOW,
                "num_steps": 2,
                "noise": "gaussian",
                "sigma_data": 1.0,
                "ocean_loss_weight": 1.0,
            },
        }
    )


def _wrapper(*, ocean=()):
    return RollingDiTWrapper(
        surface_variables=_SURFACE,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        constant_boundary_variables=_CONST,
        varying_boundary_variables=_VARY,
        levels=_LEVELS,
        horizontal_resolution=(_H, _W),
        scalar_dim=3,
        channel_layout="v2",
        scalar_routed_boundary_variables=["global_mean_co2"],
        ocean_state_variables=list(ocean),
        rolling_dit_kwargs=dict(
            dim=64,
            num_heads=2,
            num_blocks=1,
            window_size=_W_WINDOW,
            input_embed={"mode": "budget", "d_boundary": 16, "d_calendar": 16},
            output_head={"mode": "mix", "num_experts": 2},
        ),
    )


def _loader_for(model):
    return _build_loader(
        _cfg(),
        _Base(),
        window_size=_W_WINDOW,
        rank=0,
        forcing_lag=int(getattr(model, "forcing_lag", 0) or 0),
        emit_boundary_next=bool(getattr(model, "num_ocean", 0) or 0),
    )[0]


# ---------------------------------------------------------------------------
# Loader / scheduler wiring
# ---------------------------------------------------------------------------


def test_v2_model_gets_the_shifted_forcing_window():
    batch = next(iter(_loader_for(_wrapper())))
    # start_idx 0: state frames 1..3, forcings 0..2.
    assert [float(batch["surface_in_seq"][0, j].flatten()[0]) for j in range(3)] == [
        1.0, 2.0, 3.0
    ]
    assert [
        float(batch["varying_boundary_seq"][0, j, 0].flatten()[0]) for j in range(3)
    ] == [0.0, 1.0, 2.0]
    assert "varying_boundary_next_seq" not in batch


def test_ocean_model_also_gets_the_own_time_boundary_view():
    batch = next(iter(_loader_for(_wrapper(ocean=_OCEAN))))
    assert [
        float(batch["varying_boundary_seq"][0, j, 0].flatten()[0]) for j in range(3)
    ] == [0.0, 1.0, 2.0]
    assert [
        float(batch["varying_boundary_next_seq"][0, j, 0].flatten()[0])
        for j in range(3)
    ] == [1.0, 2.0, 3.0]


def test_scheduler_loss_gets_nocean_from_the_model():
    cfg = _cfg()
    stage = OmegaConf.create({"name": "s", "num_epochs": 1})
    device = torch.device("cpu")

    plain = _build_scheduler_loss(cfg, stage, device, model=_wrapper())
    assert plain.nocean == 0

    model = _wrapper(ocean=_OCEAN)
    sched = _build_scheduler_loss(cfg, stage, device, model=model)
    assert sched.nocean == 2
    assert sched.ocean_grid_indices == model.ocean_grid_indices == [1, 2]
    assert sched.ocean_loss_weight == 1.0


def test_scheduler_loss_without_a_model_is_unchanged():
    sched = _build_scheduler_loss(
        _cfg(), OmegaConf.create({"name": "s", "num_epochs": 1}), torch.device("cpu")
    )
    assert sched.nocean == 0


# ---------------------------------------------------------------------------
# adopt_ocean_contract — the single injection point
# ---------------------------------------------------------------------------


class _SchedStub:
    nocean = 0
    ocean_grid_indices: list = []


class _NoOceanSupport:
    """A single-step scheduler family (SI / SI_X) — no ocean attributes."""


def test_adopt_is_a_noop_for_a_model_without_ocean_channels():
    sched = _SchedStub()
    adopt_ocean_contract(sched, _wrapper())
    assert sched.nocean == 0
    assert sched.ocean_grid_indices == []


def test_adopt_is_a_noop_for_a_scheduler_without_ocean_support():
    # An ocean model paired with a single-step scheduler must not grow
    # attributes that family never reads.
    sched = _NoOceanSupport()
    adopt_ocean_contract(sched, _wrapper(ocean=_OCEAN))
    assert not hasattr(sched, "nocean")


def test_adopt_copies_both_halves_of_the_contract():
    sched = _SchedStub()
    model = _wrapper(ocean=_OCEAN)
    adopt_ocean_contract(sched, model)
    assert sched.nocean == 2
    assert sched.ocean_grid_indices == model.ocean_grid_indices


def test_adopt_refuses_a_self_inconsistent_model():
    class _Broken:
        num_ocean = 2
        ocean_grid_indices = [1]

    with pytest.raises(ValueError, match="derived together"):
        adopt_ocean_contract(_SchedStub(), _Broken())


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def _train_step_once(model, scheduler_loss, batch):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    return _train_step(
        model=model,
        scheduler_loss=scheduler_loss,
        sample=batch,
        optimizer=optimizer,
        grad_scaler=None,
        amp_dtype=None,
        device=torch.device("cpu"),
        window_mode=True,
    )


def test_train_step_reports_the_ocean_loss_separately():
    torch.manual_seed(0)
    model = _wrapper(ocean=_OCEAN)
    sched = _build_scheduler_loss(
        _cfg(), OmegaConf.create({"name": "s", "num_epochs": 1}),
        torch.device("cpu"), model=model,
    )
    batch = next(iter(_loader_for(model)))
    losses = _train_step_once(model, sched, batch)
    assert "loss" in losses and "loss_ocean" in losses
    assert losses["loss"] > 0.0
    assert 0.0 < losses["loss_ocean"] < losses["loss"]


def test_train_step_without_ocean_reports_only_the_total():
    torch.manual_seed(0)
    model = _wrapper()
    sched = _build_scheduler_loss(
        _cfg(), OmegaConf.create({"name": "s", "num_epochs": 1}),
        torch.device("cpu"), model=model,
    )
    losses = _train_step_once(model, sched, next(iter(_loader_for(model))))
    assert set(losses) == {"loss"}


def test_train_step_refuses_a_batch_without_the_ocean_target():
    torch.manual_seed(0)
    model = _wrapper(ocean=_OCEAN)
    sched = _build_scheduler_loss(
        _cfg(), OmegaConf.create({"name": "s", "num_epochs": 1}),
        torch.device("cpu"), model=model,
    )
    batch = next(iter(_loader_for(model)))
    del batch["varying_boundary_next_seq"]
    with pytest.raises(KeyError, match="emit_boundary_next"):
        _train_step_once(model, sched, batch)


def test_appended_target_is_the_own_time_boundary_not_the_forcing():
    """The identity-task guard, end to end on the real pack.

    Both boundary views have the same shape, so the only thing that catches a
    swap is checking *which* frame landed in the target.
    """
    model = _wrapper(ocean=_OCEAN)
    sched = _build_scheduler_loss(
        _cfg(), OmegaConf.create({"name": "s", "num_epochs": 1}),
        torch.device("cpu"), model=model,
    )
    batch = next(iter(_loader_for(model)))
    y, c_grid, _ = _pack_window(model, batch)
    y = sched.append_ocean_target(y, batch["varying_boundary_next_seq"])
    n_state = model.num_state_channels
    # SST is active-varying channel 1 -> boundary value t + 0.1; the target
    # frames are 1..3 while the c_grid the model sees holds 0..2.
    sst_target = [float(y[0, j, n_state].flatten()[0]) for j in range(3)]
    sst_forcing = [float(c_grid[0, j, 1].flatten()[0]) for j in range(3)]
    assert sst_target == pytest.approx([1.1, 2.1, 3.1])
    assert sst_forcing == pytest.approx([0.1, 1.1, 2.1])


# ---------------------------------------------------------------------------
# Partial-checkpoint warm start
# ---------------------------------------------------------------------------


def _save_state(model, path: Path):
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def test_warm_start_from_a_no_ocean_run_skips_zero_keys(tmp_path):
    """The upstream guarantee, and the reason the report exists.

    Ocean support only *adds* parameters, so every key of the source
    checkpoint must land. A nonzero skip count means the two configs differ
    somewhere else too — worth seeing before a long run.
    """
    torch.manual_seed(0)
    src = _wrapper()
    ckpt = _save_state(src, tmp_path / "plain.pt")

    torch.manual_seed(1)
    dst = _wrapper(ocean=_OCEAN)
    report = load_partial_weights(dst, ckpt)

    assert report["skipped"] == []
    assert len(report["loaded"]) == len(src.state_dict())
    # And the loaded tensors really are the source's.
    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    for k in src_sd:
        assert torch.equal(dst_sd[k], src_sd[k])
    # The only fresh keys are the ocean additions.
    assert report["fresh"]
    assert all("ocean" in k for k in report["fresh"]), report["fresh"]


def test_warm_start_reports_shape_mismatches(tmp_path):
    torch.manual_seed(0)
    src = _wrapper()
    sd = src.state_dict()
    key = next(k for k, v in sd.items() if v.dim() >= 2)
    sd[key] = torch.zeros(sd[key].shape[0] + 1, *sd[key].shape[1:])
    ckpt = tmp_path / "mismatch.pt"
    torch.save({"model_state_dict": sd}, ckpt)

    report = load_partial_weights(_wrapper(), ckpt)
    assert len(report["skipped"]) == 1
    assert key in report["skipped"][0] and "shape" in report["skipped"][0]


def test_warm_start_strips_ddp_and_compile_prefixes(tmp_path):
    torch.manual_seed(0)
    src = _wrapper()
    wrapped = {f"module._orig_mod.{k}": v for k, v in src.state_dict().items()}
    ckpt = tmp_path / "wrapped.pt"
    torch.save({"state_dict": wrapped}, ckpt)

    torch.manual_seed(1)
    dst = _wrapper()
    report = load_partial_weights(dst, ckpt)
    assert report["skipped"] == []
    assert report["fresh"] == []
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_warm_start_accepts_a_bare_state_dict(tmp_path):
    torch.manual_seed(0)
    src = _wrapper()
    ckpt = tmp_path / "bare.pt"
    torch.save(src.state_dict(), ckpt)
    report = load_partial_weights(_wrapper(), ckpt)
    assert report["skipped"] == [] and report["fresh"] == []


def test_warm_start_reads_a_mdlus_archive(tmp_path):
    # The format physicsnemo's save_checkpoint writes for our wrappers.
    torch.manual_seed(0)
    src = _wrapper()
    path = tmp_path / "model.mdlus"
    src.save(str(path))

    torch.manual_seed(1)
    dst = _wrapper(ocean=_OCEAN)
    report = load_partial_weights(dst, path)
    assert report["skipped"] == []
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_warm_start_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_partial_weights(_wrapper(), tmp_path / "nope.pt")
