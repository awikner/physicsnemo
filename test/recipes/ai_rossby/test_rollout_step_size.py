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


def test_diffusion_window_ic_bound_is_all_future_no_past():
    # Window mode reads NOTHING before the IC (the oracle window is the
    # future y_{1:W}) but the forcing trajectory reaches
    # (W + horizon - 2) model steps forward — at W=3, horizon=4, stride 4
    # that is 20 rows, dominating the horizon's 16.
    v = _diff_validator(
        _WindowScheduler(), step_size=4, max_initial_conditions=1000
    )
    ics = v._select_ic_indices(0, 1)
    assert ics
    assert min(ics) == 0
    assert max(ics) + (3 + 4 - 2) * 4 <= _N_TIME - 1
    assert max(ics) == _N_TIME - 1 - (3 + 4 - 2) * 4


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
    out = _stack_window_initial(
        ds, ic=20, W=3, device=torch.device("cpu"), step_size=4, end_offset=3
    )
    # Future oracle window y_{1:W} ending at ic + W model steps: at stride 4
    # that is rows 24, 28, 32 — never a row at or before the IC.
    assert sorted(set(ds.reads)) == [24, 28, 32]
    assert out["surface_in"].shape[:2] == (1, 3)
    assert [float(out["surface_in"][0, i].flatten()[0]) for i in range(3)] == [
        24.0, 28.0, 32.0
    ]


def test_inference_window_initial_stack_defaults_to_consecutive_rows():
    from inference import _stack_window_initial

    ds = _RecordingDataset()
    _stack_window_initial(ds, ic=20, W=3, device=torch.device("cpu"), end_offset=3)
    assert sorted(set(ds.reads)) == [21, 22, 23]


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
    # The hours live in the model config now (the step is a model property);
    # every AMIP model config states 24.
    for model_name in ("amip_si", "amip_erdm_v2", "amip_rfm", "amip_x_ddc"):
        model = OmegaConf.load(
            _AI_ROSSBY_DIR / "conf" / "model" / f"{model_name}.yaml"
        )
        assert int(model.timedelta_hours) == 24, model_name


# (model config, dataset config, expected rows/step) — every pairing the repo
# ships, each verified against the model's named upstream config on 2026-08-13.
_PAIRINGS = [
    ("amip_si", "amip_1981", 4),
    ("amip_erdm_v2", "amip_dailyavg_coarse", 4),
    ("amip_erdm_v2_ocean", "amip_dailyavg_coarse", 4),
    ("amip_rfm", "amip_1981", 4),
    ("sfno_era5", "era5_multiyear", 4),
    ("archesweather_era5", "era5_archesweather", 4),
    ("sfno_plasim_s2s", "era5_sfno_s2s_1981", 4),
    ("sfno_e3sm", "e3sm", 1),
    # The shared store: same rows, two different steps.
    ("pangu_plasim_legacy", "plasim_sim52_year12", 4),
    ("sfno_plasim_5412", "plasim_sim52_year12", 1),
]


@pytest.mark.parametrize("model_name,data_name,expected", _PAIRINGS)
def test_every_shipped_pairing_resolves_to_its_upstream_step(
    model_name, data_name, expected
):
    """The model owns the step; the dataset's lead is a cross-check.

    PLASIM is the case that forced this: ``pangu_plasim_legacy`` (24 h, upstream
    ``PANGU_PLASIM_H5_DERECHO_0514.yaml``) and ``sfno_plasim_5412`` (6 h,
    ``SFNO_PLASIM_H5_DERECHO_5412_test.yaml``) read the *same* 6-hourly store,
    so no dataset-level number can serve both — its lead is ``null`` and the
    model supplies it.
    """
    from omegaconf import OmegaConf

    from train_loop import lead_times_for_sampler, model_step_rows

    cfg = OmegaConf.create(
        {
            "model": OmegaConf.load(
                _AI_ROSSBY_DIR / "conf" / "model" / f"{model_name}.yaml"
            ),
            "dataset": OmegaConf.load(
                _AI_ROSSBY_DIR / "conf" / "dataset" / f"{data_name}.yaml"
            ),
        }
    )

    class _Store:
        class layout:
            data_timedelta_hours = 6

    stride = model_step_rows(cfg, _Store())
    assert stride == expected
    # And the single-step sampler gets the same number in rows.
    assert lead_times_for_sampler(cfg, stride) == [expected]


def test_a_mismatched_pairing_fails_loudly():
    # Pointing a 24-hour model at a dataset config that insists on 1 row must
    # raise, not train quietly at the wrong step.
    from omegaconf import OmegaConf

    from train_loop import model_step_rows

    cfg = OmegaConf.create(
        {
            "model": {"timedelta_hours": 24},
            "dataset": {"forecast_lead_times": [1]},
        }
    )

    class _Store:
        class layout:
            data_timedelta_hours = 6

    with pytest.raises(ValueError, match="model step disagreement"):
        model_step_rows(cfg, _Store())


def test_the_step_is_not_passed_to_model_constructors():
    """``timedelta_hours`` is recipe metadata, not a wrapper argument.

    Model configs are forwarded to the constructor key-by-key, so a new
    top-level key becomes a keyword argument unless it is excluded — this is how
    the audit first broke ``amip_x_ddc``.
    """
    from train import _MODEL_CONFIG_ONLY_KEYS

    assert "timedelta_hours" in _MODEL_CONFIG_ONLY_KEYS


def test_every_model_config_declares_its_step():
    """No model config may leave the step implicit.

    A missing ``timedelta_hours`` falls back to the dataset's lead, which is
    exactly the ambiguity this audit removed.
    """
    from omegaconf import OmegaConf

    skip = {"amip_combined"}  # composes two checkpoints; states it separately
    for path in sorted((_AI_ROSSBY_DIR / "conf" / "model").glob("*.yaml")):
        cfg = OmegaConf.load(path)
        assert "timedelta_hours" in cfg or path.stem in skip, path.name
        if "timedelta_hours" in cfg:
            assert int(cfg.timedelta_hours) in (6, 24), path.name


@pytest.mark.parametrize(
    "name",
    [
        "era5_multiyear",
        "era5_archesweather",
        "era5_sfno_s2s_1981",
        "e3sm",
    ],
)
def test_single_family_stores_keep_an_explicit_row_lead(name):
    """Stores used by one family keep their lead as a redundant cross-check.

    Only the shared PLASIM configs are ``null``; everything else states the row
    count so a model/dataset mismatch is caught rather than assumed.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(_AI_ROSSBY_DIR / "conf" / "dataset" / f"{name}.yaml")
    assert cfg.forecast_lead_times, name
    assert "timedelta_hours" not in cfg, (
        f"{name} restates the step in hours; the model config owns that"
    )


# ---------------------------------------------------------------------------
# Durations -> step counts (2026-08-14). Bin widths are physical quantities, so
# the eval suite derives them from the model step instead of hard-coding a
# cadence: `steps_per_bin: 120` labelled "≈ 1 month" was true only for a
# 6-hourly model and made every AMIP "monthly" bin four months wide.
# ---------------------------------------------------------------------------


class _SixHourlyStore:
    class layout:
        data_timedelta_hours = 6


@pytest.mark.parametrize(
    "model_timedelta_hours, leads, expected_hours, expected_per_month",
    [
        (24, [4], 24.0, 30),   # every AMIP config, v1 and v2
        (6, [1], 6.0, 122),    # e.g. sfno_plasim_5412 — the old hard-coded 120
    ],
)
def test_steps_per_month_follows_the_model_step(
    model_timedelta_hours, leads, expected_hours, expected_per_month
):
    from omegaconf import OmegaConf
    from train_loop import model_step_hours, steps_per_month

    cfg = OmegaConf.create(
        {
            "model": {"timedelta_hours": model_timedelta_hours},
            "dataset": {"forecast_lead_times": leads},
        }
    )
    store = _SixHourlyStore()
    assert model_step_hours(cfg, store) == expected_hours
    assert steps_per_month(cfg, store) == expected_per_month


def test_a_store_with_no_declared_cadence_refuses_to_guess():
    """A row stride can be known while the wall-clock duration is not.

    With ``model.timedelta_hours`` absent, the stride comes from
    ``forecast_lead_times`` alone and never needs the store's cadence — so
    ``model_step_rows`` succeeds and there is still no way to say how long a
    month is. Guessing would silently mis-bin, so it raises.
    """
    from omegaconf import OmegaConf
    from train_loop import model_step_rows, steps_per_month

    cfg = OmegaConf.create({"model": {}, "dataset": {"forecast_lead_times": [4]}})

    class _Store:
        class layout:
            data_timedelta_hours = 0

    assert model_step_rows(cfg, _Store()) == 4
    with pytest.raises(ValueError, match="no data_timedelta_hours"):
        steps_per_month(cfg, _Store())


def test_the_shipped_eval_suite_no_longer_hard_codes_a_cadence():
    """Regression pin for the config itself, not just the helper."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(_AI_ROSSBY_DIR / "conf" / "validation" / "eval_suite.yaml")
    # "bias" dropped: its bin keys are deprecated in the fused suite (the
    # climatology block's values apply; the bias block is an enable alias).
    for block in ("climatology", "qbo"):
        assert cfg[block].steps_per_bin is None, (
            f"{block}.steps_per_bin is pinned to {cfg[block].steps_per_bin}; the "
            f"width should be derived from months_per_bin + the model step"
        )
        assert float(cfg[block].months_per_bin) > 0
    # And the two names that used to match no model config.
    assert cfg.qbo.u_variable_name == "u_component_of_wind"
    assert "DSWRFtoa" not in list(cfg.global_mean.flux_variables)
