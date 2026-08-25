# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fused climate eval suite (climate_eval_suite.py).

All tests use synthetic stubs — no real backbones, no Hydra compose,
no real data — so they finish in milliseconds on CPU. The scorers ride ONE
:class:`~validate_diffusion.DiffusionRolloutValidator` rollout (the fusion
this suite exists for); ``eval_diffusion`` is kept as a thin alias module and
several tests deliberately import through it to pin the re-exports.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from climate_eval_suite import (  # noqa: E402
    ClimatologyScorer,
    FluxSeriesScorer,
    QBOScorer,
    VariableCatalog,
    check_deterministic_ensemble,
    derive_spread_skill,
    _estimate_period_months,
    _tropical_band_mask_and_weights,
)
from validate import Deterministic, ReplicateOnly  # noqa: E402
from validate_diffusion import DiffusionRolloutValidator  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs: dataset (surface + upper_air), wrapper, single-step scheduler.
# ---------------------------------------------------------------------------

_SURFACE_VARS = ["t2m", "DSWRFtoa"]
_UPPER_VARS = ["ua"]
_LEVELS = [10.0, 30.0, 50.0]
_H, _W = 8, 8


class _StubDataset:
    def __init__(self, n_time=60):
        self.n_time = n_time
        torch.manual_seed(0)
        self._surface = torch.randn(n_time, len(_SURFACE_VARS), _H, _W)
        self._upper = torch.randn(n_time, len(_UPPER_VARS), len(_LEVELS), _H, _W)
        self._const = torch.randn(1, _H, _W)
        self._varying = torch.randn(n_time, 1, _H, _W)
        self._calendar = torch.randn(n_time, 2)

    def __len__(self):
        return self.n_time

    def __getitem__(self, idx):
        t = idx[0] if isinstance(idx, tuple) else int(idx)
        return {
            "surface_in": self._surface[t],
            "upper_air_in": self._upper[t],
            "constant_boundary": self._const,
            "varying_boundary": self._varying[t],
            "calendar": self._calendar[t],
        }


class _StubWrapper(nn.Module):
    surface_variables = list(_SURFACE_VARS)
    upper_air_variables = list(_UPPER_VARS)
    diagnostic_variables: list = []
    levels = list(_LEVELS)

    def pack_state(self, sample):
        s = sample["surface_in"]
        ua = sample["upper_air_in"]
        b_shape = ua.shape[:-4]
        ua_flat = ua.reshape(*b_shape, len(_UPPER_VARS) * len(_LEVELS), *ua.shape[-2:])
        return torch.cat([s, ua_flat], dim=-3)

    def unpack_state(self, x):
        n_s = len(_SURFACE_VARS)
        n_ul = len(_UPPER_VARS) * len(_LEVELS)
        surface = x.narrow(-3, 0, n_s)
        ua_flat = x.narrow(-3, n_s, n_ul)
        b_shape = ua_flat.shape[:-3]
        upper = ua_flat.reshape(*b_shape, len(_UPPER_VARS), len(_LEVELS), *ua_flat.shape[-2:])
        return {"surface_in": surface, "upper_air_in": upper}

    def pack_c_grid(self, sample):
        const = sample["constant_boundary"]
        surface = sample["surface_in"]
        while const.dim() < surface.dim():
            const = const.unsqueeze(0)
        const = const.expand(*surface.shape[:-3], -1, -1, -1)
        return torch.cat([const, sample["varying_boundary"]], dim=-3)


class _RecordingSingleStepScheduler:
    def __init__(self):
        self.num_steps = 4

    def sample(self, model, x, c_grid, c_scalar, num_steps=None):
        return x + 0.1


def _stub_catalog() -> VariableCatalog:
    return VariableCatalog.from_wrapper(_StubWrapper())


def _make_kwargs(horizon=6, **overrides):
    kwargs = dict(
        wrapper=_StubWrapper(),
        inference_scheduler=_RecordingSingleStepScheduler(),
        horizon=horizon,
        device=torch.device("cpu"),
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
    )
    kwargs.update(overrides)
    return kwargs


def _run_suite(scorers, horizon=6, **overrides):
    """One fused rollout with the given scorers -> (drive, rmse_acc, blocks)."""
    kwargs = _make_kwargs(horizon=horizon, **overrides)
    drive = DiffusionRolloutValidator(
        _StubDataset(),
        log_steps=list(range(1, horizon + 1)),
        scorers=list(scorers),
        **kwargs,
    )
    rmse_acc = drive.run(nn.Identity(), epoch=0)
    blocks: dict = {}
    for s in scorers:
        blocks.update(s.finalize())
    return drive, rmse_acc, blocks


# ---------------------------------------------------------------------------
# ClimatologyScorer (absorbs the old BiasValidator)
# ---------------------------------------------------------------------------


def test_climatology_scorer_returns_expected_keys_and_shapes():
    scorer = ClimatologyScorer(n_bins=3, steps_per_bin=2)
    _, rmse_acc, blocks = _run_suite([scorer])
    clim = blocks["climatology"]
    surf_shape = (len(_SURFACE_VARS), _H, _W)
    assert clim["surface_pred_mean"].shape == surf_shape
    assert clim["surface_truth_mean"].shape == surf_shape
    assert torch.allclose(
        clim["surface_bias"], clim["surface_pred_mean"] - clim["surface_truth_mean"]
    )
    assert clim["surface_pred_binned"].shape == (3, *surf_shape)
    upper_shape = (len(_UPPER_VARS), len(_LEVELS), _H, _W)
    assert clim["upper_air_pred_mean"].shape == upper_shape
    assert any(k.startswith("rmse_step") for k in rmse_acc)


def test_climatology_scorer_global_bias_matches_lat_weighted_reduction():
    from climatology import lat_weighted_global_scalars

    scorer = ClimatologyScorer(n_bins=3, steps_per_bin=2)
    _, _, blocks = _run_suite([scorer])
    expected = lat_weighted_global_scalars(blocks["climatology"]["surface_bias"])
    assert torch.allclose(blocks["global_bias"]["surface"], expected)
    assert blocks["global_bias"]["surface"].shape == (len(_SURFACE_VARS),)


def test_climatology_scorer_track_bins_false_skips_binned_maps():
    """The bias-only configuration: means/bias/global_bias, no per-bin maps."""
    scorer = ClimatologyScorer(n_bins=3, steps_per_bin=2, track_bins=False)
    _, _, blocks = _run_suite([scorer])
    clim = blocks["climatology"]
    assert "surface_bias" in clim and "surface_pred_binned" not in clim
    assert "global_bias" in blocks


def test_fused_equals_sequential_single_scorer_runs():
    """THE fusion pin: one rollout with all scorers == per-scorer rollouts.

    The stub scheduler is deterministic (x + 0.1, no RNG), so separate drives
    roll the identical trajectory and every block must agree exactly.
    """
    catalog = _stub_catalog()

    def _fresh_scorers():
        return [
            ClimatologyScorer(n_bins=3, steps_per_bin=2),
            QBOScorer(catalog=catalog, u_variable_name="ua",
                      qbo_levels=(10.0, 30.0), steps_per_bin=2),
            FluxSeriesScorer(catalog=catalog, flux_variables=["DSWRFtoa"]),
        ]

    fused = _fresh_scorers()
    _, fused_rmse, fused_blocks = _run_suite(fused, horizon=6)

    for i in range(3):
        solo = _fresh_scorers()[i]
        _, solo_rmse, solo_blocks = _run_suite([solo], horizon=6)
        assert solo_rmse == fused_rmse, "rmse_acc must not depend on scorers"
        for block_name, payload in solo_blocks.items():
            fused_payload = fused_blocks[block_name]
            for key, val in payload.items():
                if torch.is_tensor(val):
                    torch.testing.assert_close(fused_payload[key], val)
                elif isinstance(val, dict):
                    for k2, v2 in val.items():
                        torch.testing.assert_close(fused_payload[key][k2], v2)
                elif isinstance(val, float) and math.isnan(val):
                    assert math.isnan(fused_payload[key])
                else:
                    assert fused_payload[key] == val


def test_context_reductions_computed_once_per_step_kind():
    """N scorers must not multiply the (collective) ensemble reductions."""
    calls = {"n": 0}
    scorers = [ClimatologyScorer(n_bins=3, steps_per_bin=2),
               FluxSeriesScorer(catalog=_stub_catalog(), flux_variables=["DSWRFtoa"])]
    kwargs = _make_kwargs(horizon=4)
    drive = DiffusionRolloutValidator(
        _StubDataset(), log_steps=[1, 2, 3, 4], scorers=scorers, **kwargs
    )
    orig = drive._cross_rank_ensemble_mean

    def _spy(x):
        calls["n"] += 1
        return orig(x)

    drive._cross_rank_ensemble_mean = _spy
    drive.run(nn.Identity(), epoch=0)
    # horizon=4 frames x 2 kinds (surface, upper_air) x 1 IC batch
    assert calls["n"] == 8


# ---------------------------------------------------------------------------
# QBOScorer
# ---------------------------------------------------------------------------


def test_tropical_band_mask_and_weights_sum_to_one():
    mask, weights = _tropical_band_mask_and_weights(
        _H, 30.0, torch.device("cpu"), torch.float32
    )
    assert mask.dtype == torch.bool
    assert mask.sum().item() > 0
    assert weights.numel() == mask.sum().item()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_estimate_period_months_pure_sine():
    # 12 bins spanning one full period (matches how a clean periodic
    # signal composites into a climatological bin timeseries). A small
    # phase offset keeps the zero crossings off the array boundary.
    n_bins = 12
    ts = torch.tensor(
        [math.sin(2 * math.pi * i / n_bins + 0.3) for i in range(n_bins)]
    )
    period = _estimate_period_months(ts, months_per_bin=1.0)
    assert period == pytest.approx(12.0, abs=2.0)


def test_estimate_period_months_returns_nan_for_constant_series():
    # No oscillation at all (not even under the circular wrap-around
    # treatment) -> no crossings -> nan.
    ts = torch.full((10,), 3.0)
    assert math.isnan(_estimate_period_months(ts, months_per_bin=1.0))


def test_qbo_scorer_smoke_returns_expected_keys_and_shapes():
    scorer = QBOScorer(
        catalog=_stub_catalog(), u_variable_name="ua",
        qbo_levels=(10.0, 30.0, 50.0), steps_per_bin=2, months_per_bin=1.0,
    )
    _, _, blocks = _run_suite([scorer], horizon=12)
    qbo = blocks["qbo"]
    assert qbo["qbo_pred_timeseries"].shape == (scorer.n_bins, 3)
    assert qbo["qbo_truth_timeseries"].shape == (scorer.n_bins, 3)
    for lvl in (10, 30, 50):
        assert f"qbo_period_months_pred_hPa{lvl}" in qbo
        assert f"qbo_period_months_truth_hPa{lvl}" in qbo


def test_qbo_scorer_rejects_unknown_level():
    with pytest.raises(ValueError, match="qbo_levels"):
        QBOScorer(catalog=_stub_catalog(), u_variable_name="ua",
                  qbo_levels=(999.0,))


def test_qbo_scorer_resolves_level_values_to_channel_indices():
    # Phase 12a.1 (amip_v2 bug-parity audit): upstream amip v1 selected its
    # headline plot levels by hardcoded *position* into the 26-level list,
    # which was off by one slot (z500 was actually z600, u250 was u300,
    # t850 was t875) — fixed upstream in amip_v2 by resolving indices from
    # the level values. Pin that our selection is by value: request levels
    # out of storage order and feed a field whose level slots each carry
    # their own pressure value, then assert the band mean returns exactly
    # the requested pressures in the requested order.
    scorer = QBOScorer(
        catalog=_stub_catalog(), u_variable_name="ua",
        qbo_levels=(50.0, 10.0),  # reversed vs. storage order [10, 30, 50]
    )
    assert scorer.level_indices == [2, 0]
    # bind through a real drive so the band mask exists
    _run_suite([scorer], horizon=2)

    field = torch.zeros(2, len(_UPPER_VARS), len(_LEVELS), _H, _W)
    for li, lvl in enumerate(_LEVELS):
        field[:, :, li] = lvl
    band = scorer._band_mean(field)
    assert band.shape == (2, 2)
    expected = torch.tensor([[50.0, 10.0], [50.0, 10.0]])
    assert torch.allclose(band, expected)


def test_qbo_scorer_rejects_unknown_u_variable():
    with pytest.raises(ValueError, match="u_variable_name"):
        QBOScorer(catalog=_stub_catalog(), u_variable_name="nonexistent")


def test_variable_catalog_from_cfg_model_serves_count_only_families():
    """Pangu classes expose only channel counts — the catalog must come from
    the model CONFIG, and this pins that a plain OmegaConf model node with
    variable lists is enough for QBO/flux name resolution."""
    from omegaconf import OmegaConf

    cfg_model = OmegaConf.create(
        {
            "surface_variables": list(_SURFACE_VARS),
            "upper_air_variables": ["u_component_of_wind"],
            "diagnostic_variables": None,
            "levels": [10, 30, 50],
        }
    )
    catalog = VariableCatalog.from_cfg_model(cfg_model)
    scorer = QBOScorer(catalog=catalog, u_variable_name="u_component_of_wind",
                       qbo_levels=(50.0, 10.0))
    assert scorer.level_indices == [2, 0]
    flux = FluxSeriesScorer(catalog=catalog, flux_variables=["DSWRFtoa"])
    assert flux._flux_index["DSWRFtoa"] == ("surface", 1)


# ---------------------------------------------------------------------------
# FluxSeriesScorer
# ---------------------------------------------------------------------------


def test_flux_series_scorer_tracks_requested_flux_variables():
    scorer = FluxSeriesScorer(catalog=_stub_catalog(), flux_variables=["DSWRFtoa"])
    _, _, blocks = _run_suite([scorer], horizon=6)
    gm = blocks["global_mean"]
    assert gm["flux_pred_series"]["DSWRFtoa"].shape == (6,)
    assert gm["flux_truth_series"]["DSWRFtoa"].shape == (6,)


def test_flux_series_scorer_rejects_unknown_flux_variable():
    with pytest.raises(ValueError, match="flux variable"):
        FluxSeriesScorer(
            catalog=_stub_catalog(), flux_variables=["not_a_real_channel"]
        )


def test_flux_series_length_is_horizon_with_multiple_ic_batches():
    """Pins the fixed bug: the old validator APPENDED per (IC batch, step),
    so 2 IC batches yielded a 2*horizon series. The fused scorer keys by
    m_idx and averages, so the length is the horizon regardless."""
    scorer = FluxSeriesScorer(catalog=_stub_catalog(), flux_variables=["DSWRFtoa"])
    _, _, blocks = _run_suite(
        [scorer], horizon=4, max_initial_conditions=2, batch_size=1, ic_stride=3
    )
    series = blocks["global_mean"]["flux_pred_series"]["DSWRFtoa"]
    assert series.shape == (4,)
    assert torch.isfinite(series).all()


# ---------------------------------------------------------------------------
# spread / spread-skill (the old EnsembleEnvelopeValidator, dissolved)
# ---------------------------------------------------------------------------


def test_spread_skill_emitted_whenever_ensemble_gt_1():
    drive, rmse_acc, _ = _run_suite(
        [], horizon=3, ensemble_size=3, perturber=ReplicateOnly()
    )
    ratios = derive_spread_skill(rmse_acc, drive.log_steps)
    ratio_keys = [k for k in ratios if k.startswith("spread_skill_ratio_")]
    assert len(ratio_keys) > 0
    for k in ratio_keys:
        assert isinstance(ratios[k], float)


def test_spread_skill_empty_for_single_member():
    drive, rmse_acc, _ = _run_suite([], horizon=3)
    assert derive_spread_skill(rmse_acc, drive.log_steps) == {}


def test_deterministic_ensemble_guard():
    from validate import GaussianIC

    check_deterministic_ensemble(None, 1)                       # E=1: fine
    check_deterministic_ensemble(GaussianIC(scales={}), 4)      # real noise: fine
    for bad in (None, ReplicateOnly(), Deterministic()):
        with pytest.raises(ValueError, match="gaussian_ic"):
            check_deterministic_ensemble(bad, 4)


# ---------------------------------------------------------------------------
# eval_diffusion is now an alias module — pin the re-exports.
# ---------------------------------------------------------------------------


def test_eval_diffusion_alias_reexports():
    import climate_eval_suite
    import eval_diffusion

    assert eval_diffusion.main is climate_eval_suite.main
    assert eval_diffusion.scan_rmse_trace is climate_eval_suite.scan_rmse_trace
    assert eval_diffusion.resolve_steps_per_bin is climate_eval_suite.resolve_steps_per_bin
    assert eval_diffusion.ClimatologyScorer is climate_eval_suite.ClimatologyScorer


# ---------------------------------------------------------------------------
# Bin widths are derived from the model step (2026-08-14).
# ---------------------------------------------------------------------------


class TestResolveStepsPerBin:
    """The one function deciding what the aggregators bin by."""

    @staticmethod
    def _fn():
        from eval_diffusion import resolve_steps_per_bin

        return resolve_steps_per_bin

    def test_derives_from_months_when_unset(self):
        fn = self._fn()
        # 30 steps/month (24-hour step); a quarterly bin is 3x that.
        assert fn({"months_per_bin": 1.0, "steps_per_bin": None}, 30, name="c") == 30
        assert fn({"months_per_bin": 3.0, "steps_per_bin": None}, 30, name="c") == 90
        # 6-hour step: the number the config used to hard-code.
        assert fn({"months_per_bin": 1.0, "steps_per_bin": None}, 122, name="c") == 122

    def test_a_missing_months_key_defaults_to_one_month(self):
        assert self._fn()({}, 30, name="c") == 30

    def test_never_returns_zero(self):
        """A sub-step bin width would make an empty aggregator."""
        assert self._fn()({"months_per_bin": 0.001}, 30, name="c") == 1

    def test_an_explicit_value_wins(self):
        assert self._fn()({"months_per_bin": 1.0, "steps_per_bin": 7}, 30, name="c") == 7

    def test_a_disagreeing_explicit_value_warns(self, caplog):
        fn = self._fn()
        with caplog.at_level("WARNING"):
            assert fn({"months_per_bin": 1.0, "steps_per_bin": 120}, 30, name="qbo") == 120
        assert "qbo.steps_per_bin=120" in caplog.text
        assert "NOT 1.0 month" in caplog.text

    def test_an_agreeing_explicit_value_is_silent(self, caplog):
        fn = self._fn()
        with caplog.at_level("WARNING"):
            assert fn({"months_per_bin": 1.0, "steps_per_bin": 30}, 30, name="c") == 30
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# scan_rmse_trace: instability scan over the stored per-step RMSE traces
# ---------------------------------------------------------------------------
def _trace(group, values, start=1):
    return {f"rmse_step{start + i}_{group}": v for i, v in enumerate(values)}


def test_scan_rmse_trace_clean_series_reports_nothing():
    from eval_diffusion import scan_rmse_trace

    acc = _trace("surface", [100.0 + (i % 7) for i in range(200)])
    out = scan_rmse_trace(acc)
    s = out["surface"]
    assert s["n_steps"] == 200
    assert s["n_nonfinite"] == 0
    assert s["first_nonfinite_step"] is None
    assert s["jumps"] == []


def test_scan_rmse_trace_flags_a_jump_at_the_right_step():
    from eval_diffusion import scan_rmse_trace

    vals = [100.0] * 120
    vals[80] = 450.0                      # 4.5x the trailing median of 100
    out = scan_rmse_trace(_trace("upper_air", vals), jump_factor=3.0)
    jumps = out["upper_air"]["jumps"]
    assert len(jumps) == 1
    step, value, med = jumps[0]
    assert step == 81                     # steps are 1-indexed in the keys
    assert value == pytest.approx(450.0)
    assert med == pytest.approx(100.0)


def test_scan_rmse_trace_counts_nonfinite_and_survives_them():
    from eval_diffusion import scan_rmse_trace

    vals = [50.0] * 60
    vals[10] = float("nan")
    vals[30] = float("inf")
    out = scan_rmse_trace(_trace("diagnostic", vals))
    s = out["diagnostic"]
    assert s["n_nonfinite"] == 2
    assert s["first_nonfinite_step"] == 11
    # non-finite steps are excluded from the trailing median, not tripped over
    assert all(pytest.approx(50.0) == m for (_, _, m) in s["jumps"]) or s["jumps"] == []


def test_scan_rmse_trace_handles_multiple_groups_and_ignores_other_keys():
    from eval_diffusion import scan_rmse_trace

    acc = {**_trace("surface", [10.0] * 30), **_trace("upper_air", [20.0] * 30),
           "some_other_metric": 5.0}
    out = scan_rmse_trace(acc)
    assert set(out) == {"surface", "upper_air"}
    assert out["surface"]["n_steps"] == 30


def test_perturber_scales_accepts_empty_and_omegaconf_nodes():
    """An empty DictConfig is FALSY: `to_container(node or {})` passed a plain
    dict to to_container and crashed every eval that set a perturber with the
    default empty scales (Midway job 54702336)."""
    from omegaconf import OmegaConf

    from eval_diffusion import _perturber_scales

    assert _perturber_scales(None) == {}
    assert _perturber_scales(OmegaConf.create({})) == {}
    assert _perturber_scales(OmegaConf.create({"t2m": 0.1})) == {"t2m": 0.1}
    assert _perturber_scales({"t2m": 0.2}) == {"t2m": 0.2}


def test_alias_cli_resolves_hydra_config_from_any_entrypoint():
    """Invoking the suite THROUGH the alias must keep hydra on file-based
    config resolution. With a relative config_path, hydra resolves against
    main()'s defining module — an imported module under the alias, which
    flips hydra to module-based resolution and fails with "Primary config
    module 'conf' not found" (Midway job 54953980). --help forces the full
    config-module resolution without running anything."""
    import subprocess

    for script in ("eval_diffusion.py", "climate_eval_suite.py"):
        proc = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=str(_AI_ROSSBY_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"{script} --help failed:\n{proc.stderr[-2000:]}"
        assert "powered by Hydra" in proc.stdout, script
