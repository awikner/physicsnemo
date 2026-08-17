# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end AMIP SI training smoke over a multi-year archive (2026-08-17).

The unit tests around this each hold one piece: ``_build_dataset`` routes a
directory, ``_resolve_per_year_boundaries`` pairs by year, the noise-scale
builder emits a ``(C, 1, 1)`` tensor. None of them runs ``train_diffusion.py``,
and the pieces only pay off together — the reason the recipe could not train
this model type was never one broken function.

So: build a two-year archive with the FULL ``amip_si`` contract (state 8x16 plus
a 4x c_grid, mirroring the real 45x90-with-1-degree-forcings pairing), build the
noise scales through the real CLI, and drive the real entry point on a shrunken
backbone. Three things are then true only if the whole path works:

* the log says ``multi-year archive ...: 2 sub-store(s)`` — a directory routed,
  and both years' boundary stores were opened;
* the log says ``model step: 4 store row(s) (24 h)`` — a 24 h model over 6 h
  rows. A ``1`` there trains 6-hourly pairs against a daily model, which trains
  happily and is wrong (the 2026-08-13 stride audit);
* the run trains and checkpoints with a noise-scale tensor the builder derived
  from THIS archive's channel order, and the checkpoint round-trips through the
  contract guard on resume. The scheduler indexes the tensor by packed channel,
  so a builder that disagreed with the packer by even one channel would either
  raise on the shape or silently scale the wrong rows.

What this deliberately does NOT assert is that the loss FALLS. It was written
that way first and the assertion was wrong, not merely flaky: the per-batch SI
loss here sits at 19.5k +- 3k and stayed flat across 480 iterations (measured),
and whether a 20-iteration mean happened to dip came down to which device seeded
the RNG. That is not a test artifact either — ``CONVERGENCE.md`` records a
healthy run of this family moving -4.6% over 7300 batches on real data, with
per-batch variance far larger than that. A trend needs the 10-epoch convergence
sbatch; what a smoke can prove is that every loss is finite and nothing
diverges.

Variable names come from ``conf/model/amip_si.yaml`` itself. Hand-typing the 151
channels here would let the fixture drift from the contract under test, which is
the class of bug this file exists to catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

_REPO = Path(__file__).resolve().parents[3]
_RECIPE = _REPO / "examples" / "weather" / "ai_rossby"
_MODEL_CFG = _RECIPE / "conf" / "model" / "amip_si.yaml"
_NOISE_TOOL = _REPO / "tools" / "data" / "amip" / "make_noise_scales.py"

_NT = 40                      # rows per year, 6-hourly -> 10 days
_HS, _WS = 8, 16              # state grid
_HB, _WB = 4 * _HS, 4 * _WS   # c_grid: the config's c_grid_downsample=4
_ITERS = 60

_CFG = yaml.safe_load(_MODEL_CFG.read_text())
_SURF = _CFG["surface_variables"]
_UPPER = _CFG["upper_air_variables"]
_DIAG = _CFG["diagnostic_variables"]
_CONST = _CFG["constant_boundary_variables"]
_VARY = _CFG["varying_boundary_variables"]
_LEVELS = [float(v) for v in _CFG["levels"]]


def _field(nt, h, w, seed, t0):
    """Smooth in time and space, and DIFFERENT per channel.

    Per-channel variation is the point: the noise scale is a per-channel
    increment std, so a fixture of identical channels would make every scale
    equal and hide an indexing error completely.
    """
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 3)
    amp = 0.5 + rng.uniform(0, 1.0)
    t = (np.arange(nt) + t0)[:, None, None]
    y = np.linspace(-1, 1, h)[None, :, None]
    x = np.linspace(0, 2 * np.pi, w, endpoint=False)[None, None, :]
    f = (
        amp * np.sin(2 * np.pi * t / 24.0 + ph[0])
        + 0.7 * amp * np.cos(x + ph[1]) * np.cos(np.pi * y / 2)
        + 0.3 * amp * np.sin(4 * np.pi * t / 40.0 + ph[2]) * y
    )
    return (f + 0.02 * rng.standard_normal((nt, h, w))).astype("float32")


def _times(year, nt):
    # Real datetimes: the recipe opens with emit_calendar=True, and boundary
    # reads are indexed by day-of-year.
    return [cftime.DatetimeGregorian(year, 1, 1 + i // 4, 6 * (i % 4)) for i in range(nt)]


def _write_state(path: Path, year: int, t0: int) -> None:
    data, seed = {}, 0
    for n in _SURF + _DIAG + _VARY:
        data[n] = (("time", "lat", "lon"), _field(_NT, _HS, _WS, seed, t0))
        seed += 1
    for n in _UPPER:
        stack = np.stack(
            [_field(_NT, _HS, _WS, seed + k, t0) for k in range(len(_LEVELS))], axis=1
        )
        data[n] = (("time", "pressure_level", "lat", "lon"), stack.astype("float32"))
        seed += len(_LEVELS)
    for n in _CONST:
        data[n] = (("lat", "lon"), _field(1, _HS, _WS, seed, 0)[0])
        seed += 1
    xr.Dataset(
        data,
        coords={
            "time": ("time", _times(year, _NT)),
            "pressure_level": ("pressure_level", np.array(_LEVELS, "float32")),
            "lat": ("lat", np.linspace(-87.5, 87.5, _HS).astype("float32")),
            "lon": ("lon", np.linspace(0, 360, _WS, endpoint=False).astype("float32")),
        },
        attrs={
            "data_timedelta_hours": 6,
            "surface_variables": _SURF,
            "diagnostic_variables": _DIAG,
            "pressure_upper_air_variables": _UPPER,
            "constant_boundary_variables": _CONST,
            "varying_boundary_variables": _VARY,
            "lat_row_order": "south_to_north",   # AMIP is S->N
            "climate_zarr_schema_version": 1,
        },
    ).to_zarr(path, mode="w", consolidated=True)


def _write_boundary(path: Path, year: int, t0: int) -> None:
    """Boundaries at 4x the state grid — the upstream pairing (Phase 12d (b))."""
    data, seed = {}, 500
    for n in _VARY:
        data[n] = (("time", "lat", "lon"), _field(_NT, _HB, _WB, seed, t0))
        seed += 1
    for n in _CONST:
        data[n] = (("lat", "lon"), _field(1, _HB, _WB, seed, 0)[0])
        seed += 1
    xr.Dataset(
        data,
        coords={
            "time": ("time", _times(year, _NT)),
            "lat": ("lat", np.linspace(-89.5, 89.5, _HB).astype("float32")),
            "lon": ("lon", np.linspace(0, 360, _WB, endpoint=False).astype("float32")),
        },
        attrs={
            "data_timedelta_hours": 6,
            "surface_variables": [],
            "constant_boundary_variables": _CONST,
            "varying_boundary_variables": _VARY,
            "lat_row_order": "south_to_north",
            "climate_zarr_schema_version": 1,
        },
    ).to_zarr(path, mode="w", consolidated=True)


def _write_norm(path: Path, fill: float) -> None:
    """mean 0 / std 1: the fields above are already O(1)."""
    data = {n: ((), np.float32(fill)) for n in _SURF + _DIAG + _VARY + _CONST}
    for n in _UPPER:
        data[n] = (("pressure_level",), np.full(len(_LEVELS), fill, "float32"))
    xr.Dataset(
        data,
        coords={"pressure_level": ("pressure_level", np.array(_LEVELS, "float32"))},
    ).to_netcdf(path)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Build the archive + noise scales, then run the recipe once."""
    root = tmp_path_factory.mktemp("si_multiyear")
    state, bnd = root / "state", root / "bnd"
    state.mkdir()
    bnd.mkdir()
    for i, year in enumerate((1981, 1982)):
        _write_state(state / f"{year}.zarr", year, i * _NT)
        _write_boundary(bnd / f"{year}.zarr", year, i * _NT)
    mean_nc = state / "normalize_mean_dailyavg.nc"
    std_nc = state / "normalize_std_dailyavg.nc"
    _write_norm(mean_nc, 0.0)
    _write_norm(std_nc, 1.0)

    # noqa justified throughout: every argument is sys.executable or a path this
    # fixture just created, and running the real CLIs is the point.
    sigma = root / "sigma_c.pt"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_NOISE_TOOL),
         "--zarr", str(state), "--model-config", str(_MODEL_CFG),
         "--mean", str(mean_nc), "--std", str(std_nc),
         "--year-start", "1981", "--year-end", "1983", "--out", str(sigma)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    tsv = root / "per_batch.tsv"
    ckpt_dir = root / "checkpoints"

    def _train(*extra, out_dir):
        argv = [
            sys.executable, str(_RECIPE / "train_diffusion.py"),
            "model=amip_si", "loss=si", "dataset=amip_dailyavg_coarse_multiyear",
            "training=amip_diffusion", "validation=off",
            f"++dataset.zarr_path={state}",
            f"++dataset.boundary_zarr_path={bnd}",
            f"++dataset.mean_path={mean_nc}",
            f"++dataset.std_path={std_nc}",
            "++dataset.num_workers=0", "++dataset.persistent_workers=False",
            f"++model.horizontal_resolution=[{_HS},{_WS}]",
            # Shrink the backbone only. Channel counts, levels,
            # c_grid_downsample and channel_layout stay exactly as shipped —
            # those are the contract under test.
            "++model.dit_kwargs.dim=64", "++model.dit_kwargs.num_heads=4",
            "++model.dit_kwargs.num_blocks=2", "++model.dit_kwargs.num_ca_blocks=1",
            "++model.dit_kwargs.num_output_blocks=1",
            "++model.dit_kwargs.c_grid_embed_dim=16",
            "++model.dit_kwargs.c_scalar_embed_dim=8",
            # AdamW because Muon is an optional extra, not a repo dependency;
            # the optimizer is not the subject here.
            "++training.optimizer.type=AdamW",
            "++training.ema.warmup_epochs=0",
            f"++loss.noise_scale_path={sigma}",
            f"+checkpoint_dir={ckpt_dir}",
            f"++hydra.run.dir={out_dir}",
            "run_name=si_multiyear_smoke", "wandb.enabled=False",
            *extra,
        ]
        proc = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, cwd=str(_RECIPE)
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-6000:]
        return out

    log = _train(
        "++training.max_epochs=1", "++training.stages.0.num_epochs=1",
        f"++training.stages.0.max_iterations={_ITERS}",
        f"++bench.per_batch_tsv={tsv}",
        out_dir=root / "out",
    )
    # Second entry into the same checkpoint_dir: the resume path.
    resume_log = _train(
        "++training.max_epochs=2", "++training.stages.0.num_epochs=2",
        "++training.stages.0.max_iterations=10",
        out_dir=root / "out_resume",
    )
    rows = [
        line.split("\t")
        for line in tsv.read_text().splitlines()[1:]
        if line.strip()
    ]
    return {
        "log": log,
        "resume_log": resume_log,
        "loss": [float(r[3]) for r in rows],
        "sidecar": json.loads(sigma.with_suffix(".json").read_text()),
    }


@pytest.mark.slow
def test_a_directory_of_year_stores_trains_as_one_timeline(smoke_run):
    assert "multi-year archive" in smoke_run["log"], "a directory did not route"
    assert "2 sub-store(s), 80 rows" in smoke_run["log"], (
        "both years must be visible as one timeline; a single-year open would "
        "report 40 rows"
    )


@pytest.mark.slow
def test_the_model_step_is_four_store_rows(smoke_run):
    """24 h model over 6 h rows. A 1 here is silent and wrong."""
    assert "model step: 4 store row(s) (24 h)" in smoke_run["log"]


@pytest.mark.slow
def test_it_trains_on_builder_derived_noise_scales(smoke_run):
    """The tensor the builder wrote is the one the packer wants.

    151 channels in the sidecar and a run that completes: the scheduler holds
    the scale as a buffer indexed by packed channel, so a builder off by one
    channel raises on the broadcast instead of training.
    """
    assert smoke_run["sidecar"]["channels"] == (
        len(_SURF) + len(_UPPER) * len(_LEVELS) + len(_DIAG)
    )
    assert smoke_run["sidecar"]["channel_layout"] == _CFG["channel_layout"]
    loss = smoke_run["loss"]
    assert len(loss) == _ITERS, f"max_iterations did not cap the stage: {len(loss)}"
    assert all(np.isfinite(loss)), "non-finite loss"
    # Divergence, not a trend — see the module docstring for why.
    n = _ITERS // 3
    first, last = sum(loss[:n]) / n, sum(loss[-n:]) / n
    assert last < 1.5 * first, f"loss diverged: {first:.1f} -> {last:.1f}"


@pytest.mark.slow
def test_the_checkpoint_round_trips_through_the_contract_guard(smoke_run):
    """Trainable has to mean resumable: no 37-year run fits one allocation.

    Re-entering the same ``checkpoint_dir`` puts the freshly written ``.mdlus``
    through ``assert_checkpoint_dir_contract``, which reads ``args.json`` out of
    the archive and compares the stored channel contract to the live model —
    every contract bug in the Phase 12 rebaseline was shape-preserving, so a
    checkpoint that merely loads is not evidence that it matches.

    Epoch numbering is the tell that the resume took: stage one finished epoch 1,
    so a resumed run starts at epoch 2. Starting at 1 means it silently ignored
    the checkpoint. (The log contains no string "resume" — checked, not assumed.)
    """
    log = smoke_run["resume_log"]
    assert "channel contract verified against" in log, log[-3000:]
    assert "Loaded optimizer state dictionary" in log, log[-3000:]
    assert "epoch 2 batch" in log, "restarted at epoch 1 instead of resuming"
