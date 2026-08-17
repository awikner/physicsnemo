# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The per-channel ``noise_scales`` builder.

Two properties carry the whole artifact, and both are silent when wrong:

* **channel ORDER** — the tensor is indexed by packed channel, so a v1-packed
  model handed a fork-packed tensor scales every channel by another channel's
  number. The builder derives the order from the wrapper's own ``pack_state``
  rather than restating it, and that is what these tests pin.
* **units** — the scale must be the increment std in NORMALIZED units. Leave the
  division out and the tensor carries physical units, so geopotential
  (~1e4 m^2/s^2) swamps everything else.

A synthetic store with ANALYTICALLY KNOWN increments makes both checkable: each
variable steps by a fixed amount per model step, so its increment std is 0 plus a
controlled ripple, and the expected scale is arithmetic rather than a fixture.
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

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "data" / "amip" / "make_noise_scales.py"

_H, _W = 4, 8
_LEVELS = [850.0, 500.0]
_SURFACE = ["skin_temperature", "surface_pressure"]
_UPPER = ["temperature", "u_component_of_wind"]
_DIAG = ["PRATEsfc_24h"]
#: Per-step increment amplitude planted per variable, in physical units. Distinct
#: values per variable so a permuted pack order cannot pass by coincidence.
_AMPL = {
    "skin_temperature": 2.0,
    "surface_pressure": 50.0,
    "PRATEsfc_24h": 0.5,
    "temperature": 1.0,
    "u_component_of_wind": 4.0,
}
#: Normalization std per variable, also distinct, so the division is observable.
_NORM = {
    "skin_temperature": 4.0,
    "surface_pressure": 500.0,
    "PRATEsfc_24h": 0.25,
    "temperature": 8.0,
    "u_component_of_wind": 2.0,
}
_CADENCE = 6          # store rows are 6-hourly
_STEP_HOURS = 24      # model step -> stride 4


def _write_store(path: Path, n_time: int = 36) -> None:
    """Each variable is a square wave in MODEL-STEP parity, amplitude A.

    Value is 0 through one model step and A through the next, so every 4-row
    increment is exactly +A or -A and the increment std is exactly A — an
    analytic target rather than a recorded fixture.

    n_time = 36 is chosen so the sampled increments split evenly between +A and
    -A (t0 spans blocks 0..7). An unbalanced split leaves a non-zero mean and
    biases the std by ~0.6%, which is enough to fail the tolerance below. A
    cumulative ramp — the obvious first try — is worse: its increment depends on
    where in the block you start, giving 0.61*A.
    """
    stride = _STEP_HOURS // _CADENCE
    t = np.arange(n_time)
    ramp = ((t // stride) % 2).astype("float64")

    data = {}
    for name in _SURFACE + _DIAG:
        field = (_AMPL[name] * ramp)[:, None, None] * np.ones((n_time, _H, _W))
        data[name] = (("time", "lat", "lon"), field.astype("float32"))
    for name in _UPPER:
        field = (_AMPL[name] * ramp)[:, None, None, None] * np.ones(
            (n_time, len(_LEVELS), _H, _W)
        )
        data[name] = (("time", "pressure_level", "lat", "lon"), field.astype("float32"))

    ds = xr.Dataset(
        data,
        coords={
            "time": ("time", np.arange(n_time)),
            "pressure_level": ("pressure_level", np.array(_LEVELS, "float32")),
            "lat": ("lat", np.linspace(-89.5, 89.5, _H).astype("float32")),
            "lon": ("lon", np.linspace(0.5, 359.5, _W).astype("float32")),
        },
        attrs={"data_timedelta_hours": _CADENCE},
    )
    ds.to_zarr(path, mode="w", consolidated=True)


def _write_norm(path: Path) -> None:
    """A std .nc in the shape the builder reads: scalars + per-level vectors."""
    data = {n: ((), np.float32(_NORM[n])) for n in _SURFACE + _DIAG}
    for n in _UPPER:
        data[n] = (
            ("pressure_level",),
            np.full(len(_LEVELS), _NORM[n], dtype="float32"),
        )
    xr.Dataset(
        data,
        coords={"pressure_level": ("pressure_level", np.array(_LEVELS, "float32"))},
    ).to_netcdf(path)


def _write_config(path: Path, layout: str) -> None:
    path.write_text(yaml.safe_dump({
        "name": "AmipDiTWrapper",
        "module": "physicsnemo.experimental.models.amip_si",
        "timedelta_hours": _STEP_HOURS,
        "surface_variables": _SURFACE,
        "upper_air_variables": _UPPER,
        "diagnostic_variables": _DIAG,
        "constant_boundary_variables": ["land_sea_mask"],
        "varying_boundary_variables": ["sea_ice_cover_monthly_interp"],
        "levels": _LEVELS,
        "horizontal_resolution": [_H, _W],
        "scalar_dim": 2,
        "channel_layout": layout,
        "dit_kwargs": {"dim": 32, "num_heads": 2, "num_blocks": 1, "patch_size": 2},
    }))


def _run(tmp_path: Path, layout: str = "v1"):
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True)
    _write_store(store_root / "1981.zarr")
    norm = tmp_path / "std.nc"
    _write_norm(norm)
    cfg = tmp_path / "model.yaml"
    _write_config(cfg, layout)
    out = tmp_path / "sigma.pt"
    # noqa justified: the only inputs are sys.executable and paths this test
    # just created, and running the real CLI is part of what is under test.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_TOOL),
         "--zarr", str(store_root), "--model-config", str(cfg),
         "--mean", str(norm), "--std", str(norm),
         "--year-start", "1981", "--year-end", "1982", "--out", str(out)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import torch

    return torch.load(out), json.loads(out.with_suffix(".json").read_text())


def test_scales_are_the_normalized_increment_std(tmp_path):
    """Analytic target: increment std A / normalization std -> A/N per channel."""
    tensor, meta = _run(tmp_path)
    assert tuple(tensor.shape) == (2 + 1 + 2 * 2, 1, 1), tensor.shape
    flat = tensor.reshape(-1).tolist()
    # v1 pack order: [surface | diagnostic | upper_air], upper-air variable-major.
    expected = (
        [_AMPL[n] / _NORM[n] for n in _SURFACE]
        + [_AMPL[n] / _NORM[n] for n in _DIAG]
        + [_AMPL[n] / _NORM[n] for n in _UPPER for _ in _LEVELS]
    )
    # ddof=1 over a finite sample leaves a ~5e-4 inflation; 2e-3 covers it.
    assert flat == pytest.approx(expected, rel=2e-3), (flat, expected)
    assert meta["channel_layout"] == "v1"


def test_the_pack_order_follows_the_layout(tmp_path):
    """A fork-packed model must get a differently ORDERED tensor.

    Same numbers, different positions — which is exactly the failure a hand-built
    tensor would hide, since both are the right length and finite.
    """
    v1, _ = _run(tmp_path / "a", layout="v1")
    fork, _ = _run(tmp_path / "b", layout="fork")
    a, b = v1.reshape(-1).tolist(), fork.reshape(-1).tolist()
    assert sorted(a) == pytest.approx(sorted(b), rel=2e-3)   # same multiset
    assert a != pytest.approx(b, rel=2e-3), "layouts should order differently"
    # v1 puts diagnostic before upper_air; fork puts it last.
    n_s, n_d = len(_SURFACE), len(_DIAG)
    assert a[n_s:n_s + n_d] == pytest.approx(
        [_AMPL[n] / _NORM[n] for n in _DIAG], rel=2e-3
    )
    assert b[-n_d:] == pytest.approx(
        [_AMPL[n] / _NORM[n] for n in _DIAG], rel=2e-3
    )


def test_the_artifact_is_a_bare_tensor(tmp_path):
    """``noise_scale_path`` torch.loads it directly and multiplies — no unwrap."""
    import torch

    tensor, _ = _run(tmp_path)
    assert isinstance(tensor, torch.Tensor)
    noise = torch.randn(2, tensor.shape[0], _H, _W)
    assert (noise * tensor).shape == noise.shape


def test_provenance_records_what_it_was_built_from(tmp_path):
    _, meta = _run(tmp_path)
    for key in ("source_zarr", "years", "model_config", "channel_layout",
                "model_step_hours", "store_cadence_hours", "normalization_std"):
        assert key in meta, meta
    assert meta["model_step_hours"] == _STEP_HOURS
    assert meta["store_cadence_hours"] == _CADENCE


def test_the_artifact_drives_both_si_schedulers(tmp_path):
    """The payoff: `noise_scale_path` loads it and a training step runs.

    Covers the wiring the artifact exists for — ``DriftScheduler`` and
    ``DynamicInterpolant`` both register it as a buffer and multiply it into
    their sampled noise, so a shape or dtype the schedulers reject would make
    the whole tool useless while every test above still passed.
    """
    import torch
    from omegaconf import OmegaConf

    from physicsnemo.experimental.diffusion import DriftScheduler, DynamicInterpolant
    from physicsnemo.experimental.models import amip_si as models

    _run(tmp_path)                      # writes tmp_path/sigma.pt + model.yaml
    artifact = tmp_path / "sigma.pt"
    cfg = OmegaConf.load(tmp_path / "model.yaml")
    non = {"name", "module", "target", "model_type", "timedelta_hours", "defaults"}
    model = getattr(models, str(cfg.name))(
        **{k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
           if k not in non}
    )
    nlat, nlon = model.horizontal_resolution
    x = torch.randn(2, model.in_channels, nlat, nlon)
    y = torch.randn_like(x)
    c_grid = torch.randn(2, model.c_grid_dim, nlat, nlon)
    c_scalar = torch.randn(2, model.scalar_dim)

    for cls in (DriftScheduler, DynamicInterpolant):
        sch = cls(num_steps=2, noise="gaussian", noise_scale_path=str(artifact))
        assert sch.noise_scales is not None, f"{cls.__name__} did not load it"
        assert tuple(sch.noise_scales.shape) == (model.in_channels, 1, 1)
        loss = sch.compute_loss(model, x, c_grid, c_scalar, y)
        assert torch.isfinite(loss), cls.__name__
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        model.zero_grad()
