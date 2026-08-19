# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The RSI spectral-envelope builder (``tools/data/amip/make_rsi_spectrum.py``).

Drives the real CLI on a synthetic store. The properties that matter are that
the artifact is the right LENGTH for the filter that will consume it (they must
agree on ``lmax = nlat // 2``, or loading raises), that it is unit band-mean so
it redistributes rather than rescales, and that it actually tracks the spectral
content of the increments it was fit to.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("torch_harmonics")
import torch  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "data" / "amip" / "make_rsi_spectrum.py"

_NLAT, _NLON = 16, 32
_NT = 24
_LEVELS = [500.0, 850.0]
_SURF = ["s0", "s1"]
_UPPER = ["u0"]
_DIAG = ["d0"]


def _write_store(path: Path, seed: int, smooth: bool) -> None:
    """A store whose increments are either large-scale or small-scale.

    ``smooth=True`` writes a field varying only with longitude at wavenumber 1;
    the increment spectrum is then concentrated at low degree. ``False`` adds
    grid-scale noise, pushing power to high degree.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-80, 80, _NLAT)
    lon = np.linspace(0, 360, _NLON, endpoint=False)
    LON = np.deg2rad(lon)[None, :]

    def field(nt, extra=()):
        base = np.empty((nt, *extra, _NLAT, _NLON), "float32")
        for t in range(nt):
            large = np.broadcast_to(np.sin(LON + 0.3 * t), (_NLAT, _NLON))
            f = large if smooth else large + 0.5 * rng.standard_normal((_NLAT, _NLON))
            base[t] = f if not extra else np.broadcast_to(f, (*extra, _NLAT, _NLON))
        return base

    data = {n: (("time", "lat", "lon"), field(_NT)) for n in _SURF + _DIAG}
    data[_UPPER[0]] = (("time", "pressure_level", "lat", "lon"),
                       field(_NT, extra=(len(_LEVELS),)))
    xr.Dataset(
        data,
        coords={
            "time": ("time", np.arange(_NT)),
            "pressure_level": ("pressure_level", np.array(_LEVELS, "float32")),
            "lat": ("lat", lat.astype("float32")),
            "lon": ("lon", lon.astype("float32")),
        },
        attrs={"data_timedelta_hours": 6, "climate_zarr_schema_version": 1},
    ).to_zarr(path, mode="w", consolidated=True)


def _write_std(path: Path) -> None:
    data = {n: ((), np.float32(1.0)) for n in _SURF + _DIAG}
    data[_UPPER[0]] = (("pressure_level",), np.ones(len(_LEVELS), "float32"))
    xr.Dataset(data, coords={
        "pressure_level": ("pressure_level", np.array(_LEVELS, "float32"))
    }).to_netcdf(path)


def _write_model_cfg(path: Path) -> None:
    path.write_text(yaml.safe_dump({
        "surface_variables": _SURF,
        "upper_air_variables": _UPPER,
        "diagnostic_variables": _DIAG,
        "levels": _LEVELS,
        "horizontal_resolution": [_NLAT, _NLON],
        "timedelta_hours": 24,
    }))


def _run(tmp_path, smooth=True, extra=()):
    root = tmp_path / ("smooth" if smooth else "rough")
    store = root / "store"
    store.mkdir(parents=True)
    for i, year in enumerate((1981, 1982)):
        _write_store(store / f"{year}.zarr", seed=i, smooth=smooth)
    std = root / "std.nc"
    cfg = root / "model.yaml"
    out = root / "g.pt"
    _write_std(std)
    _write_model_cfg(cfg)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_TOOL), "--zarr", str(store),
         "--model-config", str(cfg), "--std", str(std),
         "--year-start", "1981", "--year-end", "1983",
         "--sample-stride", "1", "--out", str(out), *extra],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    return out


def test_envelope_length_matches_the_filters_default_bandwidth(tmp_path):
    """The tool and the filter must agree on lmax, or loading raises."""
    from physicsnemo.experimental.diffusion._spectral import SphericalSpectralFilter

    blob = torch.load(_run(tmp_path))
    env = blob["envelope"]
    assert env.shape == (_NLAT // 2,)
    # The real consumption path: it must load without a length complaint.
    f = SphericalSpectralFilter(_NLAT, _NLON, gamma_0=0.5, gamma_1=0.02,
                                envelope=env)
    assert f.g.shape[0] == _NLAT // 2


def test_envelope_is_unit_band_mean_and_finite(tmp_path):
    env = torch.load(_run(tmp_path))["envelope"]
    assert torch.isfinite(env).all() and (env > 0).all()
    assert float(env.mean()) == pytest.approx(1.0, rel=1e-5)


def test_envelope_tracks_the_increment_spectrum(tmp_path):
    """A large-scale-only field must fit a red envelope; grid noise a bluer one."""
    smooth = torch.load(_run(tmp_path, smooth=True))["envelope"]
    rough = torch.load(_run(tmp_path, smooth=False))["envelope"]
    # Ratio of high-degree to low-degree amplitude.
    def tilt(e):
        h = e[len(e) // 2:].mean()
        lo = e[: len(e) // 2].mean()
        return float(h / lo)
    assert tilt(rough) > tilt(smooth)


def test_sidecar_records_the_provenance(tmp_path):
    out = _run(tmp_path)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["grid"] == [_NLAT, _NLON]
    assert meta["lmax"] == _NLAT // 2
    assert meta["model_step_hours"] == 24 and meta["store_cadence_hours"] == 6
    assert meta["pairs"] > 0
    # The docstring's caveat must travel with the artifact.
    assert "not the conditional spread" in meta["quantity"].lower()


def test_grid_mismatch_is_refused(tmp_path):
    """A config whose resolution disagrees with the store must not fit anything."""
    root = tmp_path / "mismatch"
    store = root / "store"
    store.mkdir(parents=True)
    _write_store(store / "1981.zarr", seed=0, smooth=True)
    std, cfg, out = root / "std.nc", root / "model.yaml", root / "g.pt"
    _write_std(std)
    cfg.write_text(yaml.safe_dump({
        "surface_variables": _SURF, "upper_air_variables": _UPPER,
        "diagnostic_variables": _DIAG, "levels": _LEVELS,
        "horizontal_resolution": [_NLAT * 2, _NLON * 2], "timedelta_hours": 24,
    }))
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_TOOL), "--zarr", str(store), "--model-config", str(cfg),
         "--std", str(std), "--year-start", "1981", "--year-end", "1982",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "horizontal_resolution" in (proc.stdout + proc.stderr)
