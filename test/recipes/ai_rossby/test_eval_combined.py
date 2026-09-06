# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Upstream-parity Combined-cascade eval machinery: the drive's
frame_transform / unpack_wrapper / init_downsample_factor / explicit-IC
hooks, and the headline bias reduction. Synthetic stubs only — CPU,
milliseconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_diffusion import (  # noqa: E402
    DiffusionRolloutValidator,
    _downsample_state_window,
)
from climate_eval_suite import (  # noqa: E402
    DEFAULT_HEADLINE_VARIABLES,
    VariableCatalog,
    _is_combined_model_cfg,
    compute_headline_bias,
)
from test_validate_diffusion import _StubDataset, _StubWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Cascade-shaped stubs
# ---------------------------------------------------------------------------
class _ConstFieldDataset(_StubDataset):
    """Surface fields are SPATIALLY CONSTANT (value = frame index).

    Bilinear down- then up-sampling is exact on constants, so a correct
    cascade — downsample the oracle window, roll, strip the tail, upsample,
    score at full res — must reproduce the truth bit-for-bit. Any indexing,
    stripping or resampling mistake breaks the zero-RMSE assertion.
    """

    def __init__(self, n_time=20, C=2, H=8, W=8):
        super().__init__(n_time=n_time, C=C, H=H, W=W)
        for t in range(n_time):
            self._surface[t] = float(t)


class _EchoWithOceanTailScheduler(nn.Module):
    """Oracle-echo rolling scheduler that appends a 1-channel 'ocean' tail.

    The emitted frames therefore have C+1 channels, like a real ocean-carrying
    forecaster; the drive's frame_transform must see (and strip) that tail.
    """

    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.nocean = 1
        self.num_steps = 2
        self.transform_input_channels: list[int] = []

    def strip_ocean(self, x):
        return x[:, : x.shape[1] - self.nocean]

    def sample_rollout(self, model, init_window, c_grid_traj, c_scalar_traj,
                       horizon, num_steps=None):
        n_anchor = init_window.shape[1] - self.window_size
        frames = init_window[:, n_anchor:n_anchor + horizon]
        tail = torch.zeros(frames.shape[0], horizon, self.nocean,
                           *frames.shape[-2:])
        return torch.cat([frames, tail], dim=2)


class _UpsamplingDownscalerWrapper(nn.Module):
    """The 'downscaler' side of the cascade: unpack at the high-res grid."""

    def unpack_state(self, x):
        return {"surface_in": x}


def _upsample_transform(sched, factor, record):
    def transform(x):
        record.append(int(x.shape[1]))
        bare = sched.strip_ocean(x)
        return F.interpolate(bare, scale_factor=factor, mode="bilinear",
                             align_corners=False)
    return transform


# ---------------------------------------------------------------------------
# End-to-end cascade through the drive
# ---------------------------------------------------------------------------
def test_cascade_scores_zero_rmse_at_the_highres_grid():
    ds = _ConstFieldDataset(H=8, W=8)
    sched = _EchoWithOceanTailScheduler(window_size=3)
    seen_channels: list[int] = []
    v = DiffusionRolloutValidator(
        ds,
        wrapper=_StubWrapper(),
        inference_scheduler=sched,
        log_steps=[1, 2, 3],
        device=torch.device("cpu"),
        horizon=3,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=2,
        frame_transform=_upsample_transform(sched, 2, seen_channels),
        unpack_wrapper=_UpsamplingDownscalerWrapper(),
        init_downsample_factor=2,
    )
    results = v.run(nn.Identity(), epoch=0)
    # scored at full res against full-res truth: exact on constant fields
    for step in (1, 2, 3):
        assert results[f"rmse_step{step}_surface"] == pytest.approx(0.0, abs=1e-6)
    # the transform received the PRE-strip frame (state + ocean tail)
    assert seen_channels and all(c == 2 + 1 for c in seen_channels)


def test_transform_hooks_require_window_mode():
    class _SingleStep:
        num_steps = 2
        def sample(self, model, x, c_grid, c_scalar, num_steps=None):
            return x
    with pytest.raises(ValueError, match="window-mode"):
        DiffusionRolloutValidator(
            _StubDataset(),
            wrapper=_StubWrapper(),
            inference_scheduler=_SingleStep(),
            log_steps=[1],
            device=torch.device("cpu"),
            horizon=1,
            frame_transform=lambda x: x,
        )


# ---------------------------------------------------------------------------
# Init-window downsampling
# ---------------------------------------------------------------------------
def test_downsample_state_window_matches_interpolate_and_skips_forcings():
    torch.manual_seed(3)
    window = {
        "surface_in": torch.randn(2, 3, 4, 8, 8),
        "upper_air_in": torch.randn(2, 3, 5, 2, 8, 8),     # 6-D
        "diagnostic": torch.randn(2, 3, 6, 8, 8),
        "varying_boundary": torch.randn(2, 3, 1, 8, 8),
        "calendar": torch.randn(2, 3, 2),
    }
    out = _downsample_state_window(window, 2)
    for key in ("surface_in", "upper_air_in", "diagnostic"):
        v = window[key]
        ref = F.interpolate(
            v.reshape(-1, 1, 8, 8), scale_factor=0.5, mode="bilinear",
            align_corners=False,
        ).reshape(*v.shape[:-2], 4, 4)
        torch.testing.assert_close(out[key], ref)
    # forcing/calendar side untouched — byte identity, not just shape
    assert out["varying_boundary"] is window["varying_boundary"]
    assert out["calendar"] is window["calendar"]


# ---------------------------------------------------------------------------
# Explicit initial conditions
# ---------------------------------------------------------------------------
def test_explicit_ic_indices_used_verbatim():
    ds = _StubDataset(n_time=20)
    from test_validate_diffusion import _OracleEchoScheduler
    sched = _OracleEchoScheduler(window_size=3)
    v = DiffusionRolloutValidator(
        ds, wrapper=_StubWrapper(), inference_scheduler=sched,
        log_steps=[1], device=torch.device("cpu"), horizon=3,
        max_initial_conditions=4, ic_stride=1, sampler_num_steps=2,
        ic_indices=[7],
    )
    assert v._select_ic_indices(rank=0, world_size=1) == [7]


def test_explicit_ic_out_of_bounds_raises_with_the_bound():
    ds = _StubDataset(n_time=20)
    from test_validate_diffusion import _OracleEchoScheduler
    sched = _OracleEchoScheduler(window_size=3)
    v = DiffusionRolloutValidator(
        ds, wrapper=_StubWrapper(), inference_scheduler=sched,
        log_steps=[1], device=torch.device("cpu"), horizon=3,
        sampler_num_steps=2, ic_indices=[19],
    )
    with pytest.raises(ValueError, match=r"admissible range \[0, \d+\]"):
        v._select_ic_indices(rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Headline bias reduction
# ---------------------------------------------------------------------------
def _toy_catalog():
    return VariableCatalog(
        surface=["2m_temperature", "10m_u_component_of_wind"],
        upper_air=["temperature", "geopotential"],
        diagnostic=["PRATEsfc_24h"],
        levels=[850.0, 500.0],       # deliberately NOT ascending
    )


def _lat_weights(H):
    import math
    phi = torch.linspace(math.pi / 2, -math.pi / 2, H, dtype=torch.float64)
    w = torch.cos(phi)
    return w / w.mean()


def test_headline_bias_matches_hand_computed_reduction():
    torch.manual_seed(5)
    H, W = 6, 12
    maps = {
        "surface_bias": torch.randn(2, H, W),
        "upper_air_bias": torch.randn(2, 2, H, W),
        "diagnostic_bias": torch.randn(1, H, W),
    }
    spec = [
        ["z500", "geopotential", 500],       # by VALUE: level index 1
        ["t2m", "2m_temperature", None],
        ["prate", "PRATEsfc_24h", None],     # diagnostic fallback
    ]
    out = compute_headline_bias(maps, _toy_catalog(), spec)
    w = _lat_weights(H)[:, None]
    for label, bias in (
        ("z500", maps["upper_air_bias"][1, 1]),
        ("t2m", maps["surface_bias"][0]),
        ("prate", maps["diagnostic_bias"][0]),
    ):
        b = bias.double()
        assert out[label]["mean_bias"] == pytest.approx(
            float((b * w).mean()), rel=1e-6)
        assert out[label]["rmse_bias"] == pytest.approx(
            float((b.pow(2) * w).mean().sqrt()), rel=1e-6)


def test_headline_bias_unknown_name_and_level_raise():
    maps = {"surface_bias": torch.randn(2, 4, 8),
            "upper_air_bias": torch.randn(2, 2, 4, 8)}
    with pytest.raises(ValueError, match="not in surface"):
        compute_headline_bias(maps, _toy_catalog(), [["x", "no_such_var", None]])
    with pytest.raises(ValueError, match="not in the model's levels"):
        compute_headline_bias(maps, _toy_catalog(), [["z250", "geopotential", 250]])


def test_default_headline_matches_upstream_field_set():
    labels = [e[0] for e in DEFAULT_HEADLINE_VARIABLES]
    assert labels == ["z500", "u250", "t850", "q850",
                      "t2m", "prate", "q2m", "u10m", "v10m"]


# ---------------------------------------------------------------------------
# Combined-config detection
# ---------------------------------------------------------------------------
def test_is_combined_model_cfg():
    from omegaconf import OmegaConf
    combined = OmegaConf.create(
        {"forecaster": {"model": "a"}, "downscaler": {"model": "b"}}
    )
    plain = OmegaConf.create({"name": "RollingDiTWrapper"})
    assert _is_combined_model_cfg(combined)
    assert not _is_combined_model_cfg(plain)
    assert not _is_combined_model_cfg(None)


# ---------------------------------------------------------------------------
# The scored grid is NOT the store's grid (Campaign A's topology)
# ---------------------------------------------------------------------------
# The cascade test above passes on the pre-fix code only because
# init_downsample_factor=2 shrinks 8->4 and the transform upsamples 4->8, so
# the scored grid coincidentally EQUALS the store probe. That is the Stampede3
# topology (full-res store, coarse forecaster, downscale back to full res).
#
# Driving the same cascade from an already-COARSE store is different: nothing
# downsamples the init window and the transform upscales past the store, so the
# scored grid (16x16) exceeds the probe (8x8). Everything sized from the probe
# -- the lat weights and every per-pixel climatology accumulator -- was then
# wrong by a factor of the downscale ratio and raised at the first frame. This
# was invisible until a coarse-store cascade existed.


class _DeclaredResDownscalerWrapper(_UpsamplingDownscalerWrapper):
    """A downscaler that declares its output grid, as XDDCWrapper does."""

    def __init__(self, nlat, nlon):
        super().__init__()
        self.horizontal_resolution = [nlat, nlon]


def _coarse_store_cascade_drive(*, store=8, scored=16):
    sched = _EchoWithOceanTailScheduler(window_size=3)
    return DiffusionRolloutValidator(
        _ConstFieldDataset(H=store, W=store),
        wrapper=_StubWrapper(),
        inference_scheduler=sched,
        log_steps=[1],
        device=torch.device("cpu"),
        horizon=1,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=1,
        frame_transform=_upsample_transform(sched, scored // store, []),
        unpack_wrapper=_DeclaredResDownscalerWrapper(scored, scored),
        init_downsample_factor=None,
    )


def test_scored_grid_follows_the_unpack_wrapper_not_the_store():
    v = _coarse_store_cascade_drive(store=8, scored=16)
    assert v._probe_grid == (8, 8), "store probe should still read 8x8"
    assert v.scored_grid == (16, 16), "scored grid must be the downscaler's"
    # The lat weights are what actually broke: probe-sized (8) against 16-row
    # fields is a broadcast error at the first frame.
    assert v.n_lat == 16
    assert v.register_lat.shape[0] == 16


def test_scored_shapes_keep_probe_channels_and_scored_grid():
    v = _coarse_store_cascade_drive(store=8, scored=16)
    shapes = v.scored_shapes
    # channels from the probe (the cascade changes resolution, not channels)
    assert shapes["surface"][0] == v.n_surface
    # grid from the scored side
    assert shapes["surface"][-2:] == (16, 16)


def test_climatology_scorer_allocates_at_the_scored_grid():
    """The accumulators are the other probe-sized thing that raised."""
    from climate_eval_suite import ClimatologyScorer

    v = _coarse_store_cascade_drive(store=8, scored=16)
    sc = ClimatologyScorer(n_bins=2, steps_per_bin=1, track_bins=True)
    sc.bind(v)
    mean = sc._pred["surface"]["mean"]
    assert tuple(mean.sum.shape)[-2:] == (16, 16), tuple(mean.sum.shape)
    binned = sc._pred["surface"]["binned"]
    assert tuple(binned.sum.shape)[-2:] == (16, 16), tuple(binned.sum.shape)


def test_non_cascade_path_still_uses_the_store_grid():
    """The mid-training path must be unchanged: no unpack_wrapper, so the
    fallback returns the probe and nothing moves."""
    from test_validate_diffusion import _OracleEchoScheduler

    ds = _ConstFieldDataset(H=8, W=8)
    v = DiffusionRolloutValidator(
        ds,
        wrapper=_StubWrapper(),
        inference_scheduler=_OracleEchoScheduler(window_size=3),
        log_steps=[1],
        device=torch.device("cpu"),
        horizon=1,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
    )
    assert v.scored_grid == (8, 8) == v._probe_grid
    assert v.n_lat == 8


def test_init_downsample_factor_of_one_is_off():
    """climate_eval_suite defaults this to the downscaler's factor; on a
    coarse-store drive that must not shrink the init window."""
    v = _coarse_store_cascade_drive(store=8, scored=16)
    assert v.init_downsample_factor is None
    sched = _EchoWithOceanTailScheduler(window_size=3)
    one = DiffusionRolloutValidator(
        _ConstFieldDataset(H=8, W=8),
        wrapper=_StubWrapper(),
        inference_scheduler=sched,
        log_steps=[1], device=torch.device("cpu"), horizon=1,
        max_initial_conditions=1, batch_size=1, ic_stride=1,
        frame_transform=_upsample_transform(sched, 2, []),
        unpack_wrapper=_DeclaredResDownscalerWrapper(16, 16),
        init_downsample_factor=1,
    )
    assert one.init_downsample_factor is None


# ---------------------------------------------------------------------------
# Streaming rollout (Phase 2)
# ---------------------------------------------------------------------------
# The materialized path holds (B*E, horizon, C, H, W): 36 GB at E=8 over 5
# years, 32 GB at E=1 over 35 years -- both past a 40 GB A100. Streaming makes
# memory O(1) in horizon. Since the two paths must be interchangeable, the
# scoring block is shared (_score_emitted_frame) and equality is pinned here.


def _generator_echo(window_size=3, base=None, record_device=None,
                    record_provider=None):
    """The plain oracle echo (no ocean tail) plus the streaming API, yielding
    the same frames in the same order as rsi/erdm's generator does.

    Built by a factory because _OracleEchoScheduler is imported locally in
    this module, and because the recording variants need closures.
    """
    from test_validate_diffusion import _OracleEchoScheduler

    _Base = base or _OracleEchoScheduler

    class _G(_Base):
        def sample_rollout_generator(self, model, init_window, c_grid_traj,
                                     c_scalar_traj, horizon, num_steps=None,
                                     forcing_provider=None):
            if record_device is not None:
                record_device.append(c_grid_traj.device.type == "cpu")
            if record_provider is not None:
                record_provider.append(forcing_provider)
            traj = self.sample_rollout(model, init_window, c_grid_traj,
                                       c_scalar_traj, horizon,
                                       num_steps=num_steps)
            for k in range(horizon):
                yield k, traj[:, k]

    return _G(window_size=window_size)


def _drive(sched, *, stream, constant_truth=None, H=8):
    return DiffusionRolloutValidator(
        _ConstFieldDataset(H=H, W=H),
        wrapper=_StubWrapper(),
        inference_scheduler=sched,
        log_steps=[1, 2, 3],
        device=torch.device("cpu"),
        horizon=3,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        sampler_num_steps=2,
        stream=stream,
        constant_truth=constant_truth,
    )


def test_streaming_matches_materialized_bitwise():
    a = _drive(_generator_echo(), stream=False)
    b = _drive(_generator_echo(), stream=True)
    assert a.stream is False and b.stream is True
    ra = a.run(nn.Identity(), epoch=0)
    rb = b.run(nn.Identity(), epoch=0)
    assert set(ra) == set(rb)
    for k in ra:
        assert ra[k] == pytest.approx(rb[k], rel=0, abs=0), k


def test_stream_falls_back_when_the_scheduler_has_no_generator():
    """rfm has sample_rollout but no generator; asking to stream must not
    silently do nothing, it must use the materialized path."""
    from test_validate_diffusion import _OracleEchoScheduler

    d = _drive(_OracleEchoScheduler(window_size=3), stream=True)
    assert d.stream is False, "no generator available -> must not claim to stream"
    d.run(nn.Identity(), epoch=0)      # still works


def test_streaming_keeps_the_forcing_trajectory_off_the_device():
    """The forcings are 2.6 GB (5 yr) / 17.8 GB (35 yr); under streaming the
    scheduler moves one window at a time, so they must stay on the CPU."""
    seen: list[bool] = []
    _drive(_generator_echo(record_device=seen), stream=True).run(
        nn.Identity(), epoch=0)
    assert seen and all(seen), "forcing trajectory was moved to the device"


def test_streaming_does_not_pass_a_forcing_provider():
    """Leaving forcing_provider None keeps the scheduler's own one-step ocean
    shift in charge. Handing that to a closure is how it becomes a silent
    identity-copy bug -- the shapes match either way."""
    got: list[object] = []
    _drive(_generator_echo(record_provider=got), stream=True).run(
        nn.Identity(), epoch=0)
    assert got and all(g is None for g in got)


# ---------------------------------------------------------------------------
# Constant truth (the obs-climatology route)
# ---------------------------------------------------------------------------
def test_constant_truth_is_used_for_every_frame():
    """mean_t(pred) - obs == mean_t(pred - obs), so a frame-invariant truth
    gives the ClimatologyScorer exactly the bias map we want."""
    ds = _ConstFieldDataset(H=8, W=8)
    probe = ds[0]
    ct = {"surface_in": torch.zeros_like(probe["surface_in"]).unsqueeze(0)}
    d = _drive(_generator_echo(), stream=True, constant_truth=ct)
    ic = d._select_ic_indices(0, 1)[0]
    res = d.run(nn.Identity(), epoch=0)
    # _ConstFieldDataset's field value IS the frame index, and the oracle echo
    # reproduces it, so against an all-zero constant truth the RMSE at emitted
    # frame k is exactly the field value there: ic + k. That it tracks k (rather
    # than staying at the store-truth value of 0) is the proof the CONSTANT
    # truth was used for every frame instead of the store's own t+k frame.
    for step in (1, 2, 3):
        assert res[f"rmse_step{step}_surface"] == pytest.approx(
            float(ic + step), rel=1e-5
        ), step


def test_constant_truth_defaults_off_so_the_store_is_used():
    d = _drive(_generator_echo(), stream=True)
    assert d.constant_truth is None
    res = d.run(nn.Identity(), epoch=0)
    # oracle echo against the store's own frames: exact
    for step in (1, 2, 3):
        assert res[f"rmse_step{step}_surface"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Per-member spread (the headline's error bar)
# ---------------------------------------------------------------------------
def _member_drive(E, *, track_members=True, H=8):
    from climate_eval_suite import ClimatologyScorer
    from validate import ReplicateOnly

    sc = ClimatologyScorer(n_bins=2, steps_per_bin=1, track_bins=False,
                           track_members=track_members)
    d = DiffusionRolloutValidator(
        _ConstFieldDataset(H=H, W=H),
        wrapper=_StubWrapper(),
        inference_scheduler=_generator_echo(),
        log_steps=[1, 2, 3],
        device=torch.device("cpu"),
        horizon=3,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        ensemble_size=E,
        perturber=ReplicateOnly(),
        scorers=[sc],
    )
    return d, sc


def test_member_spread_emits_one_entry_per_member():
    d, sc = _member_drive(4)
    d.run(nn.Identity(), epoch=0)
    block = sc.finalize()["member_spread"]["surface"]
    assert block["n_members"] == 4
    assert block["mean_bias_members"].shape[0] == 4
    assert block["rmse_bias_members"].shape[0] == 4


def test_member_spread_is_absent_unless_requested():
    d, sc = _member_drive(4, track_members=False)
    d.run(nn.Identity(), epoch=0)
    assert "member_spread" not in sc.finalize()


def test_member_mean_reproduces_the_ensemble_mean_bias():
    """Per-member means are accumulated in NORMALIZED space and denormalized
    at finalize; that is only valid because denormalization is affine, so the
    member average must reproduce the ensemble-mean bias the scorer already
    reports."""
    d, sc = _member_drive(4)
    d.run(nn.Identity(), epoch=0)
    out = sc.finalize()
    ens = out["global_bias"]["surface"]
    per_member_avg = out["member_spread"]["surface"]["mean_bias_mean"]
    assert torch.allclose(ens, per_member_avg, atol=1e-5), (ens, per_member_avg)


def test_replicate_only_members_are_identical_so_spread_is_zero():
    """A sanity floor: with an echo scheduler every member is the same draw,
    so the std must be 0. A non-zero std here would mean the member axis is
    being sliced wrongly."""
    d, sc = _member_drive(4)
    d.run(nn.Identity(), epoch=0)
    std = sc.finalize()["member_spread"]["surface"]["mean_bias_std"]
    assert torch.allclose(std, torch.zeros_like(std), atol=1e-6), std


# ---------------------------------------------------------------------------
# Store-derived lat weights
# ---------------------------------------------------------------------------
def test_store_lat_weights_match_a_cell_centred_grid(tmp_path):
    """The point of reading the coord: the endpoint-inclusive default gives the
    extreme rows weight exactly 0, a cell-centred grid does not."""
    import numpy as np
    import xarray as xr
    from climate_eval_suite import store_lat_weights

    n = 45
    lat = np.linspace(-88.0, 88.0, n)          # the coarsened AMIP grid
    xr.Dataset(coords={"lat": lat, "lon": np.arange(90.0)}).to_zarr(
        tmp_path / "s.zarr", consolidated=True
    )
    w = store_lat_weights(tmp_path / "s.zarr", n, device=torch.device("cpu"))
    assert w.shape == (n,)
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-6)   # mean-1 convention
    assert float(w[0]) > 0.0, "cell-centred poles must carry nonzero weight"

    from validate import cos_lat_weights
    endpoint = cos_lat_weights(n, torch.device("cpu"), torch.float32)
    assert float(endpoint[0]) == pytest.approx(0.0, abs=1e-6), \
        "the default really does zero the extreme rows"
    assert not torch.allclose(w, endpoint, atol=1e-3)


def test_store_lat_weights_refuse_a_wrong_length_coord(tmp_path):
    """A cascade scores at the DOWNSCALER's grid; handing it the coarse store's
    coord must raise rather than broadcast."""
    import numpy as np
    import xarray as xr
    from climate_eval_suite import store_lat_weights

    xr.Dataset(coords={"lat": np.linspace(-88.0, 88.0, 45)}).to_zarr(
        tmp_path / "c.zarr", consolidated=True
    )
    with pytest.raises(ValueError, match="SCORED grid"):
        store_lat_weights(tmp_path / "c.zarr", 180, device=torch.device("cpu"))


def test_headline_bias_accepts_explicit_weights():
    """compute_headline_bias must honour the passed weights, else the headline
    and the scorer would disagree about the convention."""
    from climate_eval_suite import DEFAULT_HEADLINE_VARIABLES, compute_headline_bias

    H, W = 8, 16
    catalog = type("C", (), {"surface": ["2m_temperature"], "upper_air": [],
                             "diagnostic": [], "levels": []})()
    maps = {"surface_bias": torch.ones(1, H, W)}
    spec = [("t2m", "2m_temperature", None)]
    flat = compute_headline_bias(maps, catalog, spec)
    weighted = compute_headline_bias(
        maps, catalog, spec, lat_weights=torch.ones(H)
    )
    # a uniform bias reduces to exactly 1.0 under ANY normalized weighting
    assert flat["t2m"]["mean_bias"] == pytest.approx(1.0, rel=1e-6)
    assert weighted["t2m"]["mean_bias"] == pytest.approx(1.0, rel=1e-6)
