# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the async forecast writer + filename helpers."""

from __future__ import annotations

import datetime as _dt
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from async_writer import (  # noqa: E402
    AsyncForecastWriter,
    format_time_for_filename,
    make_forecast_filename,
    subset_forecast_dataset,
)


def _make_ds(payload_int: int = 0, n: int = 4) -> xr.Dataset:
    """Tiny xarray dataset with a single int channel for shape testing."""
    return xr.Dataset(
        {"pred": (("time", "x"), np.full((n, n), payload_int, dtype=np.float32))},
        coords={"time": np.arange(n), "x": np.arange(n)},
    )


# ---------------------------------------------------------------------------
# AsyncForecastWriter — submit + wait
# ---------------------------------------------------------------------------


def test_writer_submit_and_wait_creates_zarr(tmp_path):
    with AsyncForecastWriter(max_in_flight=2, num_workers=2) as writer:
        for i in range(3):
            writer.submit(str(tmp_path / f"ds_{i}.zarr"), _make_ds(i))
    # On exit, wait_all has been called. Verify all three zarrs exist.
    for i in range(3):
        p = tmp_path / f"ds_{i}.zarr"
        assert p.exists()
        ds = xr.open_zarr(p)
        assert int(ds["pred"].values[0, 0]) == i


def test_writer_submit_netcdf_format(tmp_path):
    with AsyncForecastWriter(max_in_flight=2, num_workers=2) as writer:
        writer.submit(str(tmp_path / "ds.nc"), _make_ds(42))
    assert (tmp_path / "ds.nc").exists()
    ds = xr.open_dataset(tmp_path / "ds.nc")
    assert int(ds["pred"].values[0, 0]) == 42


def test_writer_backpressure_blocks_submitter(tmp_path):
    """A submit when the queue is full should block until a slot frees."""
    # Use a slow xr.Dataset write by interposing a sleep via to_zarr wrapper:
    # the simplest way is to patch _write_one in the module, but a smaller
    # behavioral test is sufficient — just observe that with max_in_flight=1
    # and num_workers=1, two submits in succession take strictly longer than
    # the per-write duration.
    from async_writer import _write_one  # noqa
    import async_writer as _aw

    orig = _aw._write_one

    def slow_write(path, dataset, mode="w"):
        time.sleep(0.15)
        orig(path, dataset, mode=mode)

    _aw._write_one = slow_write
    try:
        writer = AsyncForecastWriter(max_in_flight=1, num_workers=1)
        t0 = time.perf_counter()
        writer.submit(str(tmp_path / "a.zarr"), _make_ds(1))
        writer.submit(str(tmp_path / "b.zarr"), _make_ds(2))
        t1 = time.perf_counter()
        # With max_in_flight=1, the second submit MUST wait for the first to
        # finish — total wall ≥ one slow_write delay.
        assert t1 - t0 >= 0.10, f"expected ≥0.10s, got {t1 - t0:.3f}s"
        writer.wait_all()
        writer._executor.shutdown(wait=True)
    finally:
        _aw._write_one = orig
    assert (tmp_path / "a.zarr").exists()
    assert (tmp_path / "b.zarr").exists()


def test_writer_propagates_exception(tmp_path):
    """A failed write should surface from wait_all()."""
    import async_writer as _aw
    orig = _aw._write_one

    def boom(path, dataset, mode="w"):
        raise IOError("disk on fire")

    _aw._write_one = boom
    try:
        writer = AsyncForecastWriter(max_in_flight=2, num_workers=1)
        writer.submit(str(tmp_path / "x.zarr"), _make_ds())
        with pytest.raises(IOError, match=r"disk on fire"):
            writer.wait_all()
        writer._executor.shutdown(wait=True)
    finally:
        _aw._write_one = orig


def test_writer_refuses_submit_after_shutdown(tmp_path):
    writer = AsyncForecastWriter(max_in_flight=1, num_workers=1)
    writer.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match=r"shut down"):
        writer.submit(str(tmp_path / "post.zarr"), _make_ds())


def test_writer_in_flight_drains_to_zero(tmp_path):
    with AsyncForecastWriter(max_in_flight=4, num_workers=2) as writer:
        for i in range(3):
            writer.submit(str(tmp_path / f"d_{i}.zarr"), _make_ds(i))
        writer.wait_all()
        assert writer.in_flight == 0


def test_writer_ctor_validates_args():
    with pytest.raises(ValueError, match=r"max_in_flight"):
        AsyncForecastWriter(max_in_flight=0)
    with pytest.raises(ValueError, match=r"num_workers"):
        AsyncForecastWriter(max_in_flight=1, num_workers=0)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def test_make_forecast_filename_basic():
    name = make_forecast_filename(
        model_name="SfnoPlasim",
        run_name="bench",
        start_time="19810101T0000",
        end_time="19810115T1800",
    )
    assert name == "SfnoPlasim__bench__19810101T0000_19810115T1800.zarr"


def test_make_forecast_filename_with_extra():
    name = make_forecast_filename(
        model_name="SfnoPlasim",
        run_name="climatology",
        start_time="19810101T0000",
        end_time="19810111T0000",
        extra="chunk000",
        extension="nc",
    )
    assert name == "SfnoPlasim__climatology__19810101T0000_19810111T0000_chunk000.nc"


def test_make_forecast_filename_rejects_leading_dot():
    with pytest.raises(ValueError):
        make_forecast_filename(
            model_name="x", run_name="y", start_time="a", end_time="b", extension=".zarr",
        )


def test_format_time_datetime_yields_iso_compact():
    t = _dt.datetime(1981, 1, 15, 18, 0, 0)
    assert format_time_for_filename(t) == "19810115T1800"


def test_format_time_string_passthrough_strips_separators():
    assert format_time_for_filename("1981-01-15T18:00:00Z") == "19810115T1800"
    assert format_time_for_filename("1981-01-15 18:00") == "19810115T1800"


def test_format_time_handles_cftime_proleptic_gregorian():
    cftime = pytest.importorskip("cftime")
    t = cftime.DatetimeProlepticGregorian(1981, 1, 15, 18, 30)
    assert format_time_for_filename(t) == "19810115T1830"


def test_format_time_handles_cftime_360_day():
    cftime = pytest.importorskip("cftime")
    t = cftime.Datetime360Day(12, 6, 30, 0, 0)
    assert format_time_for_filename(t) == "00120630T0000"


def test_format_time_fallback_int_index():
    assert format_time_for_filename(123) == "idx123"


# ---------------------------------------------------------------------------
# subset_forecast_dataset
# ---------------------------------------------------------------------------


def _make_forecast_ds(*, has_diagnostic: bool = True) -> xr.Dataset:
    """A miniature per-IC forecast dataset matching inference.py's schema."""
    E, F, H, W = 1, 3, 2, 4
    surf_names = ["t2m", "u10", "v10", "msl"]
    upper_names = ["t", "u", "v", "q", "z"]
    diag_names = ["tp", "olr"]
    levels = [50.0, 500.0, 850.0]
    data_vars = {
        "pred_surface": (
            ("ensemble", "frame", "surface_var", "lat", "lon"),
            np.arange(E * F * len(surf_names) * H * W, dtype=np.float32).reshape(
                E, F, len(surf_names), H, W
            ),
        ),
        "pred_upper_air": (
            ("ensemble", "frame", "upper_air_var", "level", "lat", "lon"),
            np.arange(
                E * F * len(upper_names) * len(levels) * H * W, dtype=np.float32
            ).reshape(E, F, len(upper_names), len(levels), H, W),
        ),
    }
    coords = {
        "ensemble": ("ensemble", np.arange(E)),
        "frame": ("frame", np.arange(F)),
        "surface_var": ("surface_var", np.asarray(surf_names)),
        "upper_air_var": ("upper_air_var", np.asarray(upper_names)),
        "level": ("level", np.asarray(levels, dtype=np.float32)),
        "lat": ("lat", np.linspace(-45.0, 45.0, H, dtype=np.float32)),
        "lon": ("lon", np.linspace(0.0, 360.0, W, endpoint=False, dtype=np.float32)),
    }
    if has_diagnostic:
        data_vars["pred_diagnostic"] = (
            ("ensemble", "frame", "diag_var", "lat", "lon"),
            np.arange(E * F * len(diag_names) * H * W, dtype=np.float32).reshape(
                E, F, len(diag_names), H, W
            ),
        )
        coords["diag_var"] = ("diag_var", np.asarray(diag_names))
    return xr.Dataset(data_vars=data_vars, coords=coords)


def test_subset_noop_when_all_keys_none():
    ds = _make_forecast_ds()
    out = subset_forecast_dataset(ds)
    assert out is ds  # explicit identity — no copies on the noop path


def test_subset_surface_keeps_only_requested_names():
    ds = _make_forecast_ds()
    out = subset_forecast_dataset(ds, surface=["t2m", "msl"])
    assert list(out["surface_var"].values) == ["t2m", "msl"]
    # Other groups untouched.
    assert list(out["upper_air_var"].values) == ["t", "u", "v", "q", "z"]
    assert list(out["diag_var"].values) == ["tp", "olr"]
    # Data values preserved by selection (not shuffled).
    np.testing.assert_array_equal(
        out["pred_surface"].sel(surface_var="t2m").values,
        ds["pred_surface"].sel(surface_var="t2m").values,
    )


def test_subset_upper_air_with_level_filter():
    ds = _make_forecast_ds()
    out = subset_forecast_dataset(
        ds, upper_air=["z", "t"], upper_air_levels=[500.0, 850.0]
    )
    assert list(out["upper_air_var"].values) == ["z", "t"]
    assert list(out["level"].values) == [500.0, 850.0]
    # Surface group still has all 4 surface vars.
    assert list(out["surface_var"].values) == ["t2m", "u10", "v10", "msl"]


def test_subset_diagnostic_keeps_requested():
    ds = _make_forecast_ds()
    out = subset_forecast_dataset(ds, diagnostic=["olr"])
    assert list(out["diag_var"].values) == ["olr"]


def test_subset_empty_list_drops_group():
    ds = _make_forecast_ds()
    out = subset_forecast_dataset(ds, diagnostic=[])
    assert "pred_diagnostic" not in out
    # The other groups remain.
    assert "pred_surface" in out and "pred_upper_air" in out


def test_subset_dropping_surface_also_works():
    ds = _make_forecast_ds()
    out = subset_forecast_dataset(ds, surface=[])
    assert "pred_surface" not in out
    assert "pred_upper_air" in out


def test_subset_unknown_surface_var_raises_key_error():
    ds = _make_forecast_ds()
    with pytest.raises(KeyError, match=r"save_variables\.surface.*not in dataset"):
        subset_forecast_dataset(ds, surface=["nope"])


def test_subset_unknown_level_raises_key_error():
    ds = _make_forecast_ds()
    with pytest.raises(KeyError, match=r"upper_air_levels.*not found"):
        subset_forecast_dataset(ds, upper_air_levels=[999.0])


def test_subset_empty_levels_with_present_upper_air_raises():
    ds = _make_forecast_ds()
    with pytest.raises(ValueError, match=r"would empty pred_upper_air"):
        subset_forecast_dataset(ds, upper_air_levels=[])


def test_subset_ignores_diagnostic_when_dataset_has_none():
    ds = _make_forecast_ds(has_diagnostic=False)
    # No raise — the function silently ignores diagnostic= when the
    # dataset doesn't carry it. (Matches the inference path where some
    # models have_diagnostic=False.)
    out = subset_forecast_dataset(ds, diagnostic=["tp"])
    assert "pred_diagnostic" not in out


def test_subset_filtered_dataset_round_trips_through_zarr(tmp_path):
    ds = _make_forecast_ds()
    filtered = subset_forecast_dataset(ds, surface=["t2m"], upper_air=["z"])
    path = tmp_path / "f.zarr"
    with AsyncForecastWriter(max_in_flight=1, num_workers=1) as writer:
        writer.submit(str(path), filtered)
    reloaded = xr.open_zarr(path)
    assert list(reloaded["surface_var"].values) == ["t2m"]
    assert list(reloaded["upper_air_var"].values) == ["z"]
    # Diagnostic untouched on disk since we didn't pass diagnostic=.
    assert "pred_diagnostic" in reloaded
