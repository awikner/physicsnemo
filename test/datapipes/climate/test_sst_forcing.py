# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12g — SST anomaly channel + global-mean-SST trend scalar.

The point of the suite is the *scale*: the whole feature exists because a
+0.40 K ocean warming divided by the 12.3 K absolute-SST std arrives as 0.03
sigma and is illegible, while divided by the ~0.6 K residual std it arrives as
~0.7 sigma. So the tests check the arithmetic that produces that ratio, plus the
two ways it can be silently wrong: the day-of-year phase (this fork's calendar is
0-indexed, upstream's is 1-indexed) and the coastline (an anomaly that stamps the
land-sea mask into an ocean-variability channel).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.climate.sst_forcing import (
        SST_ANOMALY_CHANNEL_NAME,
        SSTForcing,
        harmonic_design_row,
        year_fraction,
        year_fraction_from_calendar,
    )

_H, _W = 8, 16
_VARYING = [
    "global_mean_co2",
    "DSWRFtoa_24h_lead",
    "sea_surface_temperature_monthly_interp",
    "sea_ice_cover_monthly_interp",
]
_SST_IDX_IN_STREAM = 2


def _write_artifact(path, *, n_harmonics=2, anom_std=0.6, gm_std=0.12,
                    gm_mean=0.0, land_cols=4, a0=300.0, seasonal=5.0):
    """A hand-built climatology: constant + one cosine, ocean = right-hand cols."""
    n_coef = 1 + 2 * n_harmonics
    coeffs = np.zeros((n_coef, _H, _W), dtype=np.float32)
    coeffs[0] = a0
    coeffs[1] = seasonal                      # cos(2 pi t)
    ocean = np.zeros((_H, _W), dtype=bool)
    ocean[:, land_cols:] = True
    lat = 90.0 - (np.arange(_H) + 0.5) * (180.0 / _H)
    weight = np.repeat(np.cos(np.deg2rad(lat))[:, None], _W, axis=1) * ocean
    weight = (weight / weight.sum()).astype(np.float32)
    np.savez(
        path,
        harmonic_coeffs=coeffs,
        n_harmonics=np.int32(n_harmonics),
        anom_std=np.float32(anom_std),
        gm_mean=np.float32(gm_mean),
        gm_std=np.float32(gm_std),
        ocean_weight=weight,
        fit_year_start=np.int32(1979),
        fit_year_end=np.int32(2015),
    )
    return path


@pytest.fixture
def artifact(tmp_path):
    return _write_artifact(tmp_path / "sst_climatology.npz")


def _forcing(artifact, **kw):
    return SSTForcing(str(artifact), **kw)


# ---------------------------------------------------------------------------
# The seasonal phase — where the two repos disagree
# ---------------------------------------------------------------------------


def test_upstream_year_fraction_is_one_indexed():
    # Bit-identical to upstream so an artifact from either repo evaluates the
    # same: Jan 1 midnight (doy=1) is phase 0.
    assert year_fraction(0, 1) == 0.0
    assert year_fraction(43200, 1) == pytest.approx(0.5 / 365.25)


def test_this_forks_calendar_row_is_converted():
    """The fork's calendar is 0-indexed; a raw pass-through is a one-day shift.

    ``ClimateZarrDataset._decompose_time`` returns ``dayofyr - 1``, so Jan 1
    arrives as ``doy=0``. Feeding that straight to ``year_fraction`` gives a
    *negative* phase — an error no shape or loss would reveal.
    """
    assert year_fraction_from_calendar([0.0, 0.0]) == 0.0
    assert year_fraction(0, 0) < 0                      # the bug this prevents
    # A whole year later in the fork's units is one full cycle.
    assert year_fraction_from_calendar([0.0, 365.0]) == pytest.approx(
        365.0 / 365.25
    )


def test_calendar_row_accepts_a_tensor_with_a_trend_scalar():
    row = torch.tensor([21600.0, 9.0, 1.5])   # 06:00 on Jan 10, plus a scalar
    assert year_fraction_from_calendar(row) == pytest.approx(
        (9.0 + 0.25) / 365.25
    )


def test_design_row_is_the_documented_basis():
    row = harmonic_design_row(0.0, 2)[0]
    assert row.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    quarter = harmonic_design_row(0.25, 1)[0]
    assert quarter == pytest.approx([1.0, 0.0, 1.0], abs=1e-12)


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


def test_from_config_is_none_when_unused(artifact):
    cfg = {"sst_anomaly_channel": "none", "sst_climatology_path": str(artifact)}
    assert SSTForcing.from_config(cfg) is None
    # …but the scalar alone still loads the artifact.
    got = SSTForcing.from_config(cfg, requires_scalar=True)
    assert got is not None and got.global_mean_scalar


def test_unknown_mode_is_refused(artifact):
    with pytest.raises(ValueError, match="sst_anomaly_channel must be one of"):
        _forcing(artifact, anomaly_mode="anomaly")


def test_missing_artifact_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_sst_climatology"):
        SSTForcing(str(tmp_path / "nope.npz"), anomaly_mode="append")


def test_config_without_a_path_is_refused():
    with pytest.raises(ValueError, match="sst_climatology_path"):
        SSTForcing.from_config({"sst_anomaly_channel": "append"})


def test_coefficient_count_must_match_n_harmonics(tmp_path):
    p = _write_artifact(tmp_path / "bad.npz", n_harmonics=2)
    z = dict(np.load(p))
    z["n_harmonics"] = np.int32(3)          # claims 7 coefficients, holds 5
    np.savez(p, **z)
    with pytest.raises(ValueError, match="harmonic coefficients"):
        SSTForcing(str(p), anomaly_mode="append")


def test_nonpositive_fitted_scale_is_refused(tmp_path):
    p = _write_artifact(tmp_path / "zero.npz", anom_std=0.0)
    with pytest.raises(ValueError, match="non-positive scale"):
        SSTForcing(str(p), anomaly_mode="append")


# ---------------------------------------------------------------------------
# Scale resolution — the conditioning knob
# ---------------------------------------------------------------------------


def test_fitted_scales_are_the_default(artifact):
    f = _forcing(artifact, anomaly_mode="append", global_mean_scalar=True)
    assert f.anomaly_scale == pytest.approx(0.6)     # anom_std
    assert f.scalar_scale == pytest.approx(0.12)     # gm_std


def test_a_physical_divisor_can_be_given_in_kelvin(artifact):
    # The +2 K / +4 K experiment case: at the fitted scales a uniform +4 K
    # displaces the channel by ~7 sigma, far outside the training envelope.
    f = _forcing(artifact, anomaly_mode="append", anomaly_scale=2.0,
                 global_mean_scalar=True, scalar_scale=1.0)
    assert f.anomaly_scale == 2.0 and f.scalar_scale == 1.0


@pytest.mark.parametrize("bad", ["sigma", -1.0, 0.0])
def test_invalid_scales_are_refused(artifact, bad):
    with pytest.raises(ValueError, match="sst_anomaly_scale"):
        _forcing(artifact, anomaly_mode="append", anomaly_scale=bad)


# ---------------------------------------------------------------------------
# The anomaly itself
# ---------------------------------------------------------------------------


def _normalized_stream(sst_kelvin, mean=290.0, std=12.3):
    """A z-scored varying stream whose SST channel decodes to ``sst_kelvin``."""
    stream = torch.zeros(len(_VARYING), _H, _W)
    stream[_SST_IDX_IN_STREAM] = (sst_kelvin - mean) / std
    return stream, mean, std


def test_anomaly_is_zero_when_the_field_is_its_climatology(artifact):
    f = _forcing(artifact, anomaly_mode="replace")
    row = [0.0, 0.0]                                    # Jan 1
    clim = f.climatology_at(row)
    stream, mean, std = _normalized_stream(clim)
    out, gm = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
    assert gm is None
    assert out[_SST_IDX_IN_STREAM].abs().max() < 1e-3
    assert out.shape[0] == len(_VARYING)                # replace: same width


def test_a_uniform_warming_lands_at_the_expected_sigma(artifact):
    """The reason the feature exists, as arithmetic.

    +0.4 K over a 0.6 K residual std is 0.67 sigma; the same 0.4 K through the
    absolute channel (std 12.3 K) would be 0.03 sigma.
    """
    f = _forcing(artifact, anomaly_mode="append")
    row = [0.0, 100.0]
    clim = f.climatology_at(row)
    stream, mean, std = _normalized_stream(clim + 0.4)
    out, _ = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
    anomaly_channel = out[_SST_IDX_IN_STREAM + 1]
    assert anomaly_channel.mean().item() == pytest.approx(0.4 / 0.6, rel=1e-3)
    absolute_sigma = 0.4 / std
    assert absolute_sigma < 0.04 < anomaly_channel.mean().item()


def test_the_seasonal_cycle_is_removed_not_just_the_mean(artifact):
    """A field that *is* the climatology at two very different phases.

    A plain mean subtraction would leave the +-5 K annual cycle in the channel;
    the harmonic fit leaves ~0 at both.
    """
    f = _forcing(artifact, anomaly_mode="replace")
    for row in ([0.0, 0.0], [0.0, 182.0]):
        clim = f.climatology_at(row)
        stream, mean, std = _normalized_stream(clim)
        out, _ = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
        assert out[_SST_IDX_IN_STREAM].abs().max() < 1e-3
    # …and the two climatologies really do differ by the seasonal amplitude.
    a = f.climatology_at([0.0, 0.0]).mean().item()
    b = f.climatology_at([0.0, 182.0]).mean().item()
    assert abs(a - b) > 5.0


def test_a_one_day_phase_error_is_visible_here(artifact):
    """Guards the 0-vs-1-indexed trap end to end.

    Evaluating Jan 1's field against Jan 2's climatology leaves a residual — so
    if a future refactor drops :func:`year_fraction_from_calendar`, this fails
    rather than quietly biasing the channel.
    """
    f = _forcing(artifact, anomaly_mode="replace")
    clim_correct = f.climatology_at([0.0, 0.0])
    clim_shifted = f.climatology(0, 0)          # upstream units, fed a 0-indexed doy
    # An absolute comparison: allclose's rtol against a ~300 K field would
    # swallow the difference, which is exactly why this class of bug survives.
    assert (clim_correct - clim_shifted).abs().max().item() > 1e-4


# ---------------------------------------------------------------------------
# Coastline behavior
# ---------------------------------------------------------------------------


def test_the_anomaly_does_not_stamp_the_land_sea_mask(artifact):
    """Over land the field and the climatology are the same surface, so ~0.

    The fit runs on the *filled* field, which is why this holds: a climatology
    fitted on ocean-only data would leave the fill value (270 K) as a huge
    constant anomaly over land and hand the model a land-sea mask dressed up as
    ocean variability.
    """
    f = _forcing(artifact, anomaly_mode="replace")
    row = [0.0, 50.0]
    clim = f.climatology_at(row)
    field = clim.clone()
    field[:, 4:] += 0.5                       # a real ocean anomaly, land untouched
    stream, mean, std = _normalized_stream(field)
    out, _ = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
    anom = out[_SST_IDX_IN_STREAM]
    assert anom[:, :4].abs().max() < 1e-3                     # land ~ 0
    assert anom[:, 4:].mean().item() == pytest.approx(0.5 / 0.6, rel=1e-3)
    # No coastline step beyond the physical one: the jump is the 0.5 K signal,
    # not a 30 K fill artifact.
    assert abs(anom[:, 4].mean() - anom[:, 3].mean()).item() < 1.0


# ---------------------------------------------------------------------------
# Channel bookkeeping
# ---------------------------------------------------------------------------


def test_append_adds_one_channel_right_after_sst(artifact):
    f = _forcing(artifact, anomaly_mode="append")
    assert f.adds_channel
    names = f.grid_forcing_names(_VARYING)
    assert names == _VARYING[:3] + [SST_ANOMALY_CHANNEL_NAME] + _VARYING[3:]
    stream, mean, std = _normalized_stream(torch.full((_H, _W), 300.0))
    out, _ = f.apply(stream, _SST_IDX_IN_STREAM, [0.0, 0.0], sst_mean=mean, sst_std=std)
    assert out.shape[0] == len(_VARYING) + 1
    # The absolute channel is untouched and still in place.
    assert torch.equal(out[_SST_IDX_IN_STREAM], stream[_SST_IDX_IN_STREAM])


def test_replace_keeps_the_width_and_takes_sst_place(artifact):
    f = _forcing(artifact, anomaly_mode="replace")
    assert not f.adds_channel
    names = f.grid_forcing_names(_VARYING)
    assert names[_SST_IDX_IN_STREAM] == SST_ANOMALY_CHANNEL_NAME
    assert len(names) == len(_VARYING)


def test_mode_none_leaves_the_names_alone(artifact):
    f = _forcing(artifact, global_mean_scalar=True)
    assert f.grid_forcing_names(_VARYING) == _VARYING


def test_a_stream_without_sst_is_refused(artifact):
    with pytest.raises(ValueError, match="need an SST channel"):
        SSTForcing.sst_index(["global_mean_co2", "DSWRFtoa_24h_lead"])


# ---------------------------------------------------------------------------
# The global-mean trend scalar
# ---------------------------------------------------------------------------


def test_the_scalar_is_the_ocean_area_weighted_anomaly(artifact):
    f = _forcing(artifact, global_mean_scalar=True, scalar_scale=1.0)
    row = [0.0, 30.0]
    clim = f.climatology_at(row)
    field = clim.clone()
    field[:, 4:] += 0.25                      # ocean-only warming
    stream, mean, std = _normalized_stream(field)
    _, gm = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
    # Weights sum to 1 over ocean, so an ocean-uniform +0.25 K reads 0.25.
    assert gm == pytest.approx(0.25, rel=1e-3)


def test_land_warming_does_not_move_the_scalar(artifact):
    f = _forcing(artifact, global_mean_scalar=True, scalar_scale=1.0)
    row = [0.0, 30.0]
    clim = f.climatology_at(row)
    field = clim.clone()
    field[:, :4] += 10.0                      # land only — outside the weights
    stream, mean, std = _normalized_stream(field)
    _, gm = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
    assert abs(gm) < 1e-3


def test_the_scalar_is_z_scored_by_the_fitted_statistics(artifact):
    f = _forcing(artifact, global_mean_scalar=True)   # scalar_scale = gm_std = 0.12
    row = [0.0, 30.0]
    clim = f.climatology_at(row)
    field = clim.clone()
    field[:, 4:] += 0.12
    stream, mean, std = _normalized_stream(field)
    _, gm = f.apply(stream, _SST_IDX_IN_STREAM, row, sst_mean=mean, sst_std=std)
    assert gm == pytest.approx(1.0, rel=1e-2)


def test_no_scalar_unless_asked(artifact):
    f = _forcing(artifact, anomaly_mode="append")
    stream, mean, std = _normalized_stream(torch.full((_H, _W), 300.0))
    _, gm = f.apply(stream, _SST_IDX_IN_STREAM, [0.0, 0.0], sst_mean=mean, sst_std=std)
    assert gm is None


def test_a_grid_mismatch_is_caught(tmp_path):
    """An artifact fitted at another resolution must not broadcast silently."""
    f = SSTForcing(
        str(_write_artifact(tmp_path / "c.npz")),
        global_mean_scalar=True,
    )
    stream = torch.zeros(len(_VARYING), _H * 2, _W * 2)
    with pytest.raises(ValueError, match="fitted on a different grid"):
        f.apply(stream, _SST_IDX_IN_STREAM, [0.0, 0.0], sst_mean=290.0, sst_std=12.3)


def test_describe_reports_the_fit_window_and_both_scales(artifact):
    f = _forcing(artifact, anomaly_mode="append", global_mean_scalar=True)
    text = f.describe()
    assert "1979-2015" in text and "append" in text
    assert "0.600 K" in text and "0.120 K" in text
