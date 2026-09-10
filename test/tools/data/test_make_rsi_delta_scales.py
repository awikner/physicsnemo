# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""``make_rsi_delta_scales.py``: the lag-L increment scale on the v2 packs.

Every channel is a sinusoid with its own period P (in store rows) and phase,
so the normalized k-row increment std is known in closed form up to the
finite-sample truncation (x = A sin(2 pi t / P): the increment is
2A sin(pi k/P) cos(...), the field std A/sqrt2, i.e. 2 |sin(pi k/P)| over
whole periods). The increment series has T-k rows, not a whole number of
periods, so the tests compare against the same statistic evaluated on the
GENERATING series in float64 (tight) and against the closed form (loose).
Distinct periods at a few pack positions make a permuted channel order a
wrong NUMBER, and the 6-hourly cadence attr makes the model-step stride
(4 rows) observable.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "data" / "amip" / "make_rsi_delta_scales.py"

sys.path.insert(0, str(_TOOL.parent))
import make_rsi_delta_scales as tool  # noqa: E402

_T, _H, _W = 192, 2, 3          # 192 is a multiple of every period below
_P_DEFAULT = 48
#: period per variable (all levels of an upper-air variable share it); the
#: rest use _P_DEFAULT
_P = {
    "surface_pressure": 96,                 # pack index 1
    "temperature": 64,                      # T@1000 is pack index 21
    "u_component_of_wind": 48,              # u@1000 is pack index 22
    tool.SST: 96,
    tool.ICE: 48,
}
_STEP_ROWS = 4                  # 24 h model step on a 6-hourly store


def _phase(v):
    names = tool.SURFACE + tool.DIAG
    if v in names:
        return 0.1 * names.index(v)
    if v in tool.UPPER:
        return 0.2 * tool.UPPER.index(v)
    return 0.0


def _series(v, n=_T):
    t = np.arange(n, dtype="float64")
    return np.sin(2 * math.pi * t / _P.get(v, _P_DEFAULT) + _phase(v))


def _expected(v, k):
    """The tool's statistic on the float64 generating series."""
    x = _series(v)
    return float(np.std(x[k:] - x[:-k]) / np.std(x))


def _closed_form(v, k):
    return 2.0 * abs(math.sin(math.pi * k / _P.get(v, _P_DEFAULT)))


def _write_state(path: Path):
    g = zarr.open_group(str(path), mode="w")
    g.attrs["data_timedelta_hours"] = 6
    for v in tool.SURFACE + tool.DIAG:
        x = _series(v)[:, None, None]
        arr = g.create_array(v, shape=(_T, _H, _W), dtype="float32")
        arr[:] = (x * np.ones((_T, _H, _W))).astype("float32")
    for v in tool.UPPER:
        x = _series(v)[:, None, None, None]
        arr = g.create_array(v, shape=(_T, len(tool.LEVELS), _H, _W), dtype="float32")
        arr[:] = (x * np.ones((_T, len(tool.LEVELS), _H, _W))).astype("float32")


def _write_boundary(path: Path):
    g = zarr.open_group(str(path), mode="w")
    g.attrs["data_timedelta_hours"] = 6
    for v in (tool.SST, tool.ICE):
        x = (_series(v)[:, None, None] * np.ones((_T, _H, _W))).astype("float32")
        if v == tool.SST:
            x[:, 0, 0] = np.nan            # a land point: the ratio is NaN-tolerant
        arr = g.create_array(v, shape=(_T, _H, _W), dtype="float32")
        arr[:] = x


def _run(tmp_path: Path, *extra, out_name="sigma.pt"):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "bnd").mkdir(parents=True, exist_ok=True)
    if not (root / "state" / "1985.zarr").exists():
        _write_state(root / "state" / "1985.zarr")
        _write_boundary(root / "bnd" / "1985.zarr")
    out = tmp_path / out_name
    # noqa justified: the only inputs are sys.executable and paths this test
    # just created, and running the real CLI is part of what is under test.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_TOOL), "--data-root", str(root),
         "--state-store", "state", "--boundary-store", "bnd",
         "--years", "1985", "--floor", "0.0", "--out", str(out), *extra],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return (torch.load(out).reshape(-1).numpy(),
            json.loads(out.with_suffix(".json").read_text()), proc.stdout)


def test_lag_six_on_the_sst_pred_contract_is_the_24_row_increment(tmp_path):
    vals, meta, _ = _run(tmp_path, "--contract", "sst_pred", "--lag", "6")
    assert vals.shape == (153,)
    assert meta["step_rows"] == _STEP_ROWS          # derived from the 6-h attr
    assert meta["increment_rows"] == 24 and meta["lag_steps"] == 6
    k = 24
    assert vals[1] == pytest.approx(_expected("surface_pressure", k), rel=1e-3)
    assert vals[21] == pytest.approx(_expected("temperature", k), rel=1e-3)     # T@1000
    assert vals[22] == pytest.approx(_expected("u_component_of_wind", k), rel=1e-3)
    assert vals[0] == pytest.approx(_expected("skin_temperature", k), rel=1e-3)
    assert meta["channels"][21] == "temperature@1000"
    assert meta["channels"][-2:] == [tool.SST, tool.ICE]
    assert vals[-2] == pytest.approx(_expected(tool.SST, k), rel=1e-3)   # NaN-tolerant
    assert vals[-1] == pytest.approx(_expected(tool.ICE, k), rel=1e-3)
    # the closed form holds up to the finite-sample truncation (T-k rows:
    # 1.75 periods at P = 96, 3.5 at P = 48)
    assert vals[1] == pytest.approx(_closed_form("surface_pressure", k), rel=0.1)
    assert vals[22] == pytest.approx(_closed_form("u_component_of_wind", k), rel=0.02)
    # positions 1, 21 and 22 carry three different periods: a permuted pack
    # order would put the wrong number at each
    assert len({round(float(vals[i]), 2) for i in (1, 21, 22)}) == 3


def test_lag_one_is_the_one_step_increment_not_the_one_row_one(tmp_path):
    """A model step is 4 rows on the 6-hourly store: --lag 1 must use k = 4."""
    vals, meta, _ = _run(tmp_path, "--contract", "sst_pred", "--lag", "1")
    assert meta["increment_rows"] == 4
    assert vals[1] == pytest.approx(_expected("surface_pressure", 4), rel=1e-3)
    assert vals[0] == pytest.approx(_expected("skin_temperature", 4), rel=1e-3)
    assert vals[1] != pytest.approx(_expected("surface_pressure", 1), rel=1e-2)


def test_step_rows_one_reproduces_the_legacy_row_increment(tmp_path):
    """--step-rows 1 --lag 1 is what built the 2026-09-02 artifacts."""
    vals, meta, _ = _run(tmp_path, "--contract", "sst_pred", "--lag", "1",
                         "--step-rows", "1")
    assert meta["increment_rows"] == 1
    assert vals[1] == pytest.approx(_expected("surface_pressure", 1), rel=1e-3)
    assert vals[22] == pytest.approx(_expected("u_component_of_wind", 1), rel=1e-3)


def test_fancy_contract_has_the_derived_anomaly_channel(tmp_path):
    vals, meta, _ = _run(tmp_path, "--contract", "fancy", "--lag", "6")
    assert vals.shape == (154,)
    assert meta["channels"][-3:] == [
        tool.SST, "sea_surface_temperature_anomaly", tool.ICE]
    assert vals[-3] == vals[-2]            # the anomaly takes sst's ratio
    assert vals[-1] == pytest.approx(_expected(tool.ICE, 24), rel=1e-3)


def test_lag_scales_grow_with_the_lag_for_slow_channels(tmp_path):
    """Lag 6 vs lag 1 at period 96 rows: ~ sin(pi 24/96) / sin(pi 4/96) ~ 5."""
    l1, _, _ = _run(tmp_path, "--contract", "sst_pred", "--lag", "1", out_name="l1.pt")
    l6, _, _ = _run(tmp_path, "--contract", "sst_pred", "--lag", "6", out_name="l6.pt")
    assert l6[1] / l1[1] == pytest.approx(
        _expected("surface_pressure", 24) / _expected("surface_pressure", 4), rel=1e-3)
    assert l6[1] / l1[1] > 4.5


def test_compare_reports_the_channel_ratio(tmp_path):
    ref = tmp_path / "ref.pt"
    torch.save(torch.full((153, 1, 1), 0.5), ref)
    _, _, out = _run(tmp_path, "--contract", "sst_pred", "--lag", "6",
                     "--compare", str(ref))
    assert "ratio new/ref" in out


def test_floor_is_applied_and_recorded(tmp_path):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "bnd").mkdir(parents=True)
    _write_state(root / "state" / "1985.zarr")
    _write_boundary(root / "bnd" / "1985.zarr")
    out = tmp_path / "floored.pt"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_TOOL), "--data-root", str(root),
         "--state-store", "state", "--boundary-store", "bnd", "--years", "1985",
         "--contract", "sst_pred", "--lag", "1", "--step-rows", "1",
         "--floor", "0.5", "--out", str(out)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    vals = torch.load(out).reshape(-1).numpy()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert vals.min() >= 0.5
    assert min(meta["values_unfloored"]) < 0.5


def test_ratio_helper_is_analytic_and_nan_tolerant():
    """The lag-P increment of a P-periodic series is identically zero (exact
    regardless of truncation); other lags match the float64 definition; NaNs
    (land points) are ignored; too-short series raise."""
    x = _series("u_component_of_wind")[:, None] * np.ones((_T, 4))
    assert tool._ratio(x, 48) == pytest.approx(0.0, abs=1e-9)          # sin(pi) = 0
    assert tool._ratio(x, 24) == pytest.approx(_expected("u_component_of_wind", 24), rel=1e-9)
    x[:, 0] = np.nan
    assert tool._ratio(x, 24) == pytest.approx(_expected("u_component_of_wind", 24), rel=1e-9)
    with pytest.raises(ValueError):
        tool._ratio(x[:5], 12)
