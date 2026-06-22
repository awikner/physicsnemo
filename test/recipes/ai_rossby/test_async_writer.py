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
