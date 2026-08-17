# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""``rollout.py`` — the two-stage cascade driver (2026-08-17).

Phase 12h.27 built `CombinedModule.windowed_init` / `windowed_step` and left the
driver unwritten. These cover the parts that are specific to a *cascade* and that
`inference.py` cannot express, rather than re-testing the streaming maths (that is
`test/models/amip_si/test_combined_windowed.py`):

* **coords come from the DOWNSCALER's grid**, read from a store rather than
  synthesized. Labelling a 180x360 field with the forecaster's 45 latitudes is
  the failure mode here, and inventing a latitude vector would additionally risk
  the row order — AMIP is S->N where ERA5 is N->S.
* **month buffering** groups frames into one file per calendar month.
* **resume** reloads `(x_bar, eps_prev, step)` and continues rather than restarting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

import rollout as R  # noqa: E402

_LOW = (4, 8)
_FACTOR = 2
_HIGH = (_LOW[0] * _FACTOR, _LOW[1] * _FACTOR)
_SURFACE = ["skin_temperature", "surface_pressure"]
_UPPER = ["temperature"]
_DIAG = ["PRATEsfc_24h"]
_LEVELS = [500.0, 850.0]
_STATE = len(_SURFACE) + len(_DIAG) + len(_UPPER) * len(_LEVELS)


class _Downscaler:
    """Just enough of an XDDCWrapper for the coord/layout checks."""

    horizontal_resolution = list(_HIGH)


def _write_highres(path: Path, *, ascending: bool) -> None:
    lat = np.linspace(-89.0, 89.0, _HIGH[0]).astype("float32")
    if not ascending:
        lat = lat[::-1].copy()
    xr.Dataset(
        {"sea_ice_cover_monthly_interp": (("lat", "lon"), np.zeros(_HIGH, "float32"))},
        coords={
            "lat": ("lat", lat),
            "lon": ("lon", np.linspace(0.0, 350.0, _HIGH[1]).astype("float32")),
        },
    ).to_zarr(path, mode="w", consolidated=True)


# ---------------------------------------------------------------------------
# The resolution crossing
# ---------------------------------------------------------------------------


def test_coords_come_from_a_store_at_the_downscalers_resolution(tmp_path):
    from omegaconf import OmegaConf

    store = tmp_path / "hi.zarr"
    _write_highres(store, ascending=True)
    cfg = OmegaConf.create(
        {"rollout": {"highres_zarr": str(store)}, "dataset": {"boundary_zarr_path": None}}
    )
    lat, lon = R._highres_coords(cfg, _Downscaler())
    assert (len(lat), len(lon)) == _HIGH
    # Row order preserved as stored, not normalized to some assumed convention.
    assert lat[0] < lat[-1]


def test_a_store_on_the_wrong_grid_is_refused(tmp_path):
    """The whole point: the forecaster's grid must not label the output."""
    from omegaconf import OmegaConf

    store = tmp_path / "low.zarr"
    xr.Dataset(
        {"x": (("lat", "lon"), np.zeros(_LOW, "float32"))},
        coords={
            "lat": ("lat", np.linspace(-89.0, 89.0, _LOW[0]).astype("float32")),
            "lon": ("lon", np.linspace(0.0, 350.0, _LOW[1]).astype("float32")),
        },
    ).to_zarr(store, mode="w", consolidated=True)
    cfg = OmegaConf.create(
        {"rollout": {"highres_zarr": str(store)}, "dataset": {"boundary_zarr_path": None}}
    )
    with pytest.raises(ValueError, match="downscaler emits"):
        R._highres_coords(cfg, _Downscaler())


def test_descending_latitudes_are_carried_through_unchanged(tmp_path):
    """An N->S store must stay N->S: silently flipping is the bug to avoid."""
    from omegaconf import OmegaConf

    store = tmp_path / "ns.zarr"
    _write_highres(store, ascending=False)
    cfg = OmegaConf.create(
        {"rollout": {"highres_zarr": str(store)}, "dataset": {"boundary_zarr_path": None}}
    )
    lat, _ = R._highres_coords(cfg, _Downscaler())
    assert lat[0] > lat[-1]


def test_no_coord_source_is_an_error_not_a_guess(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"rollout": {}, "dataset": {"boundary_zarr_path": None}})
    with pytest.raises(ValueError, match="high-resolution lat/lon"):
        R._highres_coords(cfg, _Downscaler())


# ---------------------------------------------------------------------------
# Month buffering
# ---------------------------------------------------------------------------


class _CollectingWriter:
    def __init__(self):
        self.written: list[tuple[str, xr.Dataset]] = []

    def submit(self, path, dataset, **_):
        self.written.append((path, dataset))
        return None


def _layout():
    return {
        "surface_variables": _SURFACE,
        "diagnostic_variables": _DIAG,
        "upper_air_variables": _UPPER,
        "levels": _LEVELS,
        "attrs": {"note": "test"},
    }


def _frame():
    return {
        "surface": torch.randn(len(_SURFACE), *_HIGH),
        "diagnostic": torch.randn(len(_DIAG), *_HIGH),
        "upper_air": torch.randn(len(_UPPER), len(_LEVELS), *_HIGH),
    }


def test_frames_flush_one_file_per_calendar_month(tmp_path):
    writer = _CollectingWriter()
    buf = R._MonthBuffer(
        writer=writer, out_dir=tmp_path, run_name="run",
        lat=np.linspace(-89, 89, _HIGH[0]).astype("float32"),
        lon=np.linspace(0, 350, _HIGH[1]).astype("float32"),
        layout=_layout(),
    )
    # 3 frames in January, 2 in February — one file each, not five.
    for day in (1, 2, 3):
        buf.add(cftime.DatetimeGregorian(1981, 1, day), _frame())
    for day in (1, 2):
        buf.add(cftime.DatetimeGregorian(1981, 2, day), _frame())
    buf.flush()
    assert len(writer.written) == 2, [p for p, _ in writer.written]
    names = [Path(p).name for p, _ in writer.written]
    assert names == ["run__198101.nc", "run__198102.nc"], names
    jan = writer.written[0][1]
    assert jan.sizes["time"] == 3
    # Fields land on the DOWNSCALER's grid with the coords handed in.
    assert jan["surface"].shape == (3, len(_SURFACE), *_HIGH)
    assert jan["upper_air"].shape == (3, len(_UPPER), len(_LEVELS), *_HIGH)
    assert jan.sizes["lat"] == _HIGH[0] and jan.sizes["lon"] == _HIGH[1]


def test_an_empty_buffer_flushes_nothing(tmp_path):
    writer = _CollectingWriter()
    buf = R._MonthBuffer(
        writer=writer, out_dir=tmp_path, run_name="run",
        lat=np.zeros(_HIGH[0], "float32"), lon=np.zeros(_HIGH[1], "float32"),
        layout=_layout(),
    )
    buf.flush()
    buf.flush()
    assert writer.written == []


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_state_round_trips_and_resumes_mid_rollout(tmp_path):
    x_bar = torch.randn(1, 3, _STATE, *_LOW)
    eps = torch.randn(1, 1, _STATE, *_LOW)
    assert R._load_state(tmp_path, torch.device("cpu")) is None  # fresh run
    R._save_state(tmp_path, step=17, x_bar=x_bar, eps_prev=eps)
    step, x2, e2 = R._load_state(tmp_path, torch.device("cpu"))
    assert step == 17
    torch.testing.assert_close(x2, x_bar)
    torch.testing.assert_close(e2, eps)


def test_the_state_write_is_atomic(tmp_path):
    """A killed job must not leave a half-written state that loads as garbage."""
    x_bar = torch.randn(1, 2, _STATE, *_LOW)
    eps = torch.randn(1, 1, _STATE, *_LOW)
    R._save_state(tmp_path, step=1, x_bar=x_bar, eps_prev=eps)
    R._save_state(tmp_path, step=2, x_bar=x_bar * 2, eps_prev=eps)
    # The temp file is renamed, never left behind alongside the real one.
    assert not list(tmp_path.glob("*.tmp"))
    step, x2, _ = R._load_state(tmp_path, torch.device("cpu"))
    assert step == 2
    torch.testing.assert_close(x2, x_bar * 2)


# ---------------------------------------------------------------------------
# The streaming loop, driven end to end with tiny real modules
# ---------------------------------------------------------------------------


def _combined(nocean=()):
    """A real CombinedModule at toy size — same construction as
    test/models/amip_si/test_combined_windowed.py, which pins the streaming maths.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        from physicsnemo.experimental.diffusion import ERDMScheduler
        from physicsnemo.experimental.models.amip_si import (
            CombinedModule,
            RollingDiTWrapper,
            XDDCWrapper,
        )

    W = 3
    fc = RollingDiTWrapper(
        surface_variables=_SURFACE, upper_air_variables=_UPPER,
        diagnostic_variables=_DIAG, constant_boundary_variables=["land_sea_mask"],
        varying_boundary_variables=[
            "sea_surface_temperature_monthly_interp", "sea_ice_cover_monthly_interp",
        ],
        levels=_LEVELS, horizontal_resolution=list(_LOW), channel_layout="v2",
        ocean_state_variables=list(nocean),
        rolling_dit_kwargs=dict(
            dim=32, num_heads=2, num_blocks=1, temporal_num_heads=2, window_size=W,
            input_embed={"mode": "budget", "d_boundary": 8, "d_calendar": 8},
            output_head={"mode": "mix", "num_experts": 2},
        ),
    ).eval()
    sched = ERDMScheduler(
        window_size=W, num_steps=2, noise="gaussian", sigma_data=1.0,
        nocean=len(nocean), ocean_grid_indices=[0, 1][: len(nocean)], S_churn=0.0,
    )
    if nocean:
        sched.ocean_grid_indices = list(fc.ocean_grid_indices)
    ds = XDDCWrapper(
        surface_variables=_SURFACE, upper_air_variables=_UPPER,
        diagnostic_variables=_DIAG, levels=_LEVELS,
        horizontal_resolution=list(_HIGH), downsample_factor=_FACTOR,
        channel_layout="v2", decoder_type="dit",
        dit_kwargs=dict(dim=32, num_heads=2, num_blocks=1, patch_size=2),
    ).eval()

    class _PassThrough:
        """The downscaler's own sampler is covered elsewhere; noise draws here
        would only make the resume comparison depend on RNG ordering."""

        def sample(self, model, cond, num_steps=None):
            return cond

    return CombinedModule(
        forecaster=fc, forecaster_scheduler=sched,
        downscaler=ds, downscaler_scheduler=_PassThrough(),
    ).eval(), fc, sched, W


def _drive(tmp_path, *, horizon, start_step=0, state_every=0, nocean=(), seed=0):
    torch.manual_seed(seed)
    combined, fc, sched, W = _combined(nocean)
    traj_len = W + horizon - 1 + (1 if nocean else 0)
    c_grid = torch.randn(1, traj_len, fc.c_grid_dim, *_LOW)
    c_scalar = torch.randn(1, traj_len, fc.scalar_dim)
    init = torch.randn(1, W, fc.num_state_channels, *_LOW)
    x_bar, eps = combined.windowed_init(init)
    writer = _CollectingWriter()
    buf = R._MonthBuffer(
        writer=writer, out_dir=tmp_path, run_name="cascade",
        lat=np.linspace(-89, 89, _HIGH[0]).astype("float32"),
        lon=np.linspace(0, 350, _HIGH[1]).astype("float32"),
        layout=_layout(),
    )
    times = [cftime.DatetimeGregorian(1981, 1 + (k // 3), 1 + (k % 3))
             for k in range(horizon)]
    R.run_rollout(
        combined=combined, buf=buf, out_dir=tmp_path, times=times,
        x_bar=x_bar, eps_prev=eps, c_grid_traj=c_grid, c_scalar_traj=c_scalar,
        start_step=start_step, horizon=horizon,
        forecaster_num_steps=2, downscaler_num_steps=None,
        ocean_lookahead=bool(nocean), state_every=state_every,
    )
    return writer, buf


def test_the_loop_emits_one_high_res_frame_per_step(tmp_path):
    writer, buf = _drive(tmp_path, horizon=4)
    frames = sum(int(ds.sizes["time"]) for _, ds in writer.written)
    assert frames == 4, [(p, int(d.sizes["time"])) for p, d in writer.written]
    for _, ds in writer.written:
        # The cascade's whole purpose: output on the DOWNSCALER's grid.
        assert (ds.sizes["lat"], ds.sizes["lon"]) == _HIGH
        assert ds["surface"].shape[1] == len(_SURFACE)
        assert np.isfinite(ds["surface"].values).all()


def test_the_loop_runs_with_predicted_ocean_channels(tmp_path):
    """`ocean_win` is the window one step FORWARD, so the trajectory needs a
    lookahead frame — an off-by-one here would index past the end."""
    ocean = [
        "sea_surface_temperature_monthly_interp",
        "sea_ice_cover_monthly_interp",
    ]
    writer, _ = _drive(tmp_path, horizon=3, nocean=ocean)
    assert sum(int(d.sizes["time"]) for _, d in writer.written) == 3


def test_resuming_at_a_later_step_emits_only_the_remainder(tmp_path):
    writer, _ = _drive(tmp_path, horizon=5, start_step=3)
    assert sum(int(d.sizes["time"]) for _, d in writer.written) == 2


def test_state_is_checkpointed_on_the_requested_cadence(tmp_path):
    _drive(tmp_path, horizon=4, state_every=2)
    step, _, _ = R._load_state(tmp_path, torch.device("cpu"))
    assert step == 4                      # final save wins
    assert R._state_path(tmp_path).exists()
