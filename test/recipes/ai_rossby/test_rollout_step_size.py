# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""``step_size`` on the rollout drivers (2026-08-13 stride audit).

``inference.py``'s deterministic driver has always had ``step_size`` ("rollout
step size in dataset indices per model step… 24 h = 4 x 6 h"). Its three
siblings did not: :class:`RolloutValidator`, :class:`DiffusionRolloutValidator`
and the diffusion inference driver all marched one *store row* per model step.
On a 6-hourly archive under a 24-hour model step that scores a one-step forecast
against truth 6 hours out and walks the forcings 4x too fast — with no shape
error anywhere.

These tests pin the index arithmetic by recording which rows each driver
touches, since that is the only observable that distinguishes right from wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from validate import RolloutValidator  # noqa: E402
from validate_diffusion import DiffusionRolloutValidator  # noqa: E402

_H, _W = 4, 8
_N_TIME = 80


class _RecordingDataset:
    """Records every row index read, and encodes the row in the data."""

    n_time = _N_TIME

    def __init__(self):
        self.reads: list[int] = []

    def __getitem__(self, index):
        t, _lead = index if isinstance(index, tuple) else (index, 1)
        t = int(t)
        if not (0 <= t < self.n_time):
            raise IndexError(t)
        self.reads.append(t)
        f = float(t)
        return {
            "surface_in": torch.full((2, _H, _W), f),
            "upper_air_in": torch.full((1, 2, _H, _W), f),
            "constant_boundary": torch.zeros(2, _H, _W),
            "varying_boundary": torch.full((3, _H, _W), 100.0 + f),
            "calendar": torch.full((2,), 200.0 + f),
            "target_surface": torch.full((2, _H, _W), f + 1),
            "target_upper_air": torch.full((1, 2, _H, _W), f + 1),
        }

    def __len__(self):
        return self.n_time


def _validator(cls, **kw):
    base = dict(
        dataset=_RecordingDataset(),
        device=torch.device("cpu"),
        has_diagnostic=False,
        max_initial_conditions=1,
        ic_stride=1,
        normalizer=None,
        seed=0,
    )
    base.update(kw)
    return cls(**base)


# ---------------------------------------------------------------------------
# Deterministic RolloutValidator
# ---------------------------------------------------------------------------


def test_deterministic_ic_bound_reserves_whole_model_steps():
    # With max_step=5 at 4 rows/step the last IC must leave 20 rows of truth,
    # not 5 — otherwise the final steps score against clamped/absent frames.
    v1 = _validator(RolloutValidator, log_steps=[1, 5], step_size=1)
    v4 = _validator(RolloutValidator, log_steps=[1, 5], step_size=4)
    ics1 = v1._select_ic_indices(rank=0, world_size=1)
    ics4 = v4._select_ic_indices(rank=0, world_size=1)
    assert ics1 and ics4
    assert max(ics1) + 5 <= _N_TIME - 1
    assert max(ics4) + 5 * 4 <= _N_TIME - 1


def test_deterministic_step_size_defaults_to_one_row():
    assert _validator(RolloutValidator, log_steps=[1]).step_size == 1


# ---------------------------------------------------------------------------
# DiffusionRolloutValidator — single-step and window paths
# ---------------------------------------------------------------------------


class _SingleStepScheduler:
    """Has ``sample`` but no ``sample_rollout`` -> single-step dispatch."""

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        return x


class _WindowScheduler:
    window_size = 3

    def sample_rollout(
        self, model, init_window, c_grid_traj, c_scalar_traj, horizon,
        num_steps=None,
    ):
        b = init_window.shape[0]
        return init_window[:, :1].expand(b, horizon, *init_window.shape[2:])


def _diff_validator(scheduler, **kw):
    return _validator(
        DiffusionRolloutValidator,
        wrapper=None,
        inference_scheduler=scheduler,
        horizon=4,
        log_steps=[1, 4],
        **kw,
    )


def test_diffusion_single_step_ic_bound_scales_with_the_step():
    v1 = _diff_validator(_SingleStepScheduler(), step_size=1)
    v4 = _diff_validator(_SingleStepScheduler(), step_size=4)
    assert max(v1._select_ic_indices(0, 1)) + 4 <= _N_TIME - 1
    assert max(v4._select_ic_indices(0, 1)) + 4 * 4 <= _N_TIME - 1


def test_diffusion_window_ic_bound_reserves_past_and_future_in_steps():
    # Window mode needs (W-1) past frames AND horizon future frames, both in
    # model steps — at stride 4 that is 8 rows back and 16 rows forward.
    v = _diff_validator(_WindowScheduler(), step_size=4)
    ics = v._select_ic_indices(0, 1)
    assert ics
    assert min(ics) >= (3 - 1) * 4
    assert max(ics) + 4 * 4 <= _N_TIME - 1


def test_oracle_window_frames_are_a_model_step_apart():
    v = _diff_validator(_WindowScheduler(), step_size=4)
    v.dataset.reads.clear()
    window = v._stack_window([20], w_offset=0)
    # W=3 frames ending at the IC, 4 rows apart: rows 12, 16, 20.
    assert sorted(set(v.dataset.reads)) == [12, 16, 20]
    assert [
        float(window["surface_in"][0, i].flatten()[0]) for i in range(3)
    ] == [12.0, 16.0, 20.0]


def test_oracle_window_at_stride_one_is_unchanged():
    v = _diff_validator(_WindowScheduler(), step_size=1)
    v.dataset.reads.clear()
    v._stack_window([20], w_offset=0)
    assert sorted(set(v.dataset.reads)) == [18, 19, 20]


def test_diffusion_step_size_defaults_to_one_row():
    assert _diff_validator(_WindowScheduler()).step_size == 1


# ---------------------------------------------------------------------------
# Diffusion inference driver
# ---------------------------------------------------------------------------


def test_inference_window_initial_stack_strides():
    from inference import _stack_window_initial

    ds = _RecordingDataset()
    out = _stack_window_initial(ds, ic=20, W=3, device=torch.device("cpu"), step_size=4)
    assert sorted(set(ds.reads)) == [12, 16, 20]
    assert out["surface_in"].shape[:2] == (1, 3)
    assert [float(out["surface_in"][0, i].flatten()[0]) for i in range(3)] == [
        12.0, 16.0, 20.0
    ]


def test_inference_window_initial_stack_defaults_to_consecutive_rows():
    from inference import _stack_window_initial

    ds = _RecordingDataset()
    _stack_window_initial(ds, ic=20, W=3, device=torch.device("cpu"))
    assert sorted(set(ds.reads)) == [18, 19, 20]


def test_inference_driver_exposes_step_size():
    # The deterministic driver has had it since the ArchesWeather port; the
    # diffusion one was missing it entirely.
    import inspect

    from inference import (
        run_diffusion_inference_streaming_per_ic,
        run_inference_streaming_per_ic,
    )

    for fn in (run_inference_streaming_per_ic, run_diffusion_inference_streaming_per_ic):
        assert "step_size" in inspect.signature(fn).parameters, fn.__name__


# ---------------------------------------------------------------------------
# The AMIP configs the audit corrected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["amip_1981", "amip_dailyavg", "amip_dailyavg_coarse"]
)
def test_amip_configs_carry_the_upstream_24_hour_step(name):
    """Every upstream AMIP config, v1 and v2, is 6-hourly data / 24-hour step.

    That includes the frozen v1 families (v1 ``SI_NCAR_AIMIP.yaml``,
    ``EDM.yaml``, ``RFM.yaml``, ``ERDM_Unet.yaml``, ``DDC_NCAR_AIMIP.yaml`` all
    say ``timedelta_hours: 24``), so these configs must say 4 rows — they said
    1 from Phase 8b until this audit.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(_AI_ROSSBY_DIR / "conf" / "dataset" / f"{name}.yaml")
    assert list(cfg.forecast_lead_times) == [4]
    assert int(cfg.timedelta_hours) == 24


@pytest.mark.parametrize(
    "name",
    [
        "era5_multiyear",
        "era5_archesweather",
        "plasim_sim52_year12",
        "plasim_sim52_train_val",
    ],
)
def test_hours_and_rows_agree_in_every_config_that_states_both(name):
    """The cross-check must actually pass for the configs that opt into it.

    All these stores are 6-hourly: the ERA5 pair steps 24 h (4 rows), PLASIM
    steps 6 h (1 row).
    """
    from omegaconf import OmegaConf

    from physicsnemo.experimental.datapipes.climate import resolve_step_stride

    cfg = OmegaConf.load(_AI_ROSSBY_DIR / "conf" / "dataset" / f"{name}.yaml")

    class _Store:
        class layout:
            data_timedelta_hours = 6

    stride = resolve_step_stride(
        _Store(),
        forecast_lead_times=list(cfg.forecast_lead_times),
        timedelta_hours=cfg.timedelta_hours,
    )
    assert stride == int(cfg.timedelta_hours) // 6
