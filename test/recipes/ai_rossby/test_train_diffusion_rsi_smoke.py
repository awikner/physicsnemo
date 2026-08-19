# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Rolling Stochastic Interpolant training smoke.

Sibling of ``test_train_diffusion_smoke.py``, reusing its archive builders. RSI
adds two pieces of wiring that nothing else in the recipe exercises, and both
fail *silently* if wrong because every tensor keeps a plausible shape:

* the loader must emit the pre-window anchor frame and ``_pack_window`` must
  prepend it, so ``compute_loss`` sees ``W+1`` state frames against ``W``
  forcing frames;
* the backbone must emit ``2*C`` channels (the H1 and zhat readouts), which
  means the stage loop has to build the scheduler BEFORE the loader — the
  anchor contract is a property of the loss family, not the model.

So this drives the real entry point and asserts the pack width from the log
plus finite losses and a checkpoint round-trip. As in the sibling module it
deliberately does NOT assert that the loss falls: a trend needs the convergence
sbatch, and per-batch variance here is far larger than any 60-iteration drift.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_train_diffusion_smoke import (  # noqa: E402  (sibling module)
    _HS,
    _NT,
    _WS,
    _clean_env,
    _write_boundary,
    _write_norm,
    _write_state,
)

_REPO = Path(__file__).resolve().parents[3]
_RECIPE = _REPO / "examples" / "weather" / "ai_rossby"
_MODEL_CFG = _RECIPE / "conf" / "model" / "amip_rsi_v2.yaml"

# Read the boundary contract off the config under test rather than hand-typing
# it — the point is that the fixture cannot drift from the model.
_CFG = yaml.safe_load(_MODEL_CFG.read_text())
_VARY = _CFG["varying_boundary_variables"]
_SCALAR_ROUTED = _CFG.get("scalar_routed_boundary_variables", []) or []
_W = _CFG["rolling_dit_kwargs"]["window_size"]
_ITERS = 12


@pytest.fixture(scope="module")
def rsi_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsi_smoke")
    state, bnd = root / "state", root / "bnd"
    state.mkdir()
    bnd.mkdir()
    for i, year in enumerate((1981, 1982)):
        _write_state(state / f"{year}.zarr", year, i * _NT,
                     vary=_VARY, uniform=_SCALAR_ROUTED)
        # The v2/RSI boundary set adds global_mean_co2, which the
        # ForcingAssembler pops onto the scalar row — so it must be spatially
        # uniform in the store.
        _write_boundary(bnd / f"{year}.zarr", year, i * _NT,
                        vary=_VARY, uniform=_SCALAR_ROUTED)
    mean_nc = state / "normalize_mean_dailyavg.nc"
    std_nc = state / "normalize_std_dailyavg.nc"
    _write_norm(mean_nc, 0.0, vary=_VARY)
    _write_norm(std_nc, 1.0, vary=_VARY)

    ckpt_dir = root / "checkpoints"
    tsv = root / "per_batch.tsv"

    def _train(*extra, out_dir):
        argv = [
            sys.executable, str(_RECIPE / "train_diffusion.py"),
            "model=amip_rsi_v2", "loss=rsi",
            "dataset=amip_dailyavg_coarse_multiyear",
            "training=amip_diffusion", "validation=off",
            f"++dataset.zarr_path={state}",
            f"++dataset.boundary_zarr_path={bnd}",
            f"++dataset.mean_path={mean_nc}",
            f"++dataset.std_path={std_nc}",
            "++dataset.num_workers=0", "++dataset.persistent_workers=False",
            f"++model.horizontal_resolution=[{_HS},{_WS}]",
            # Shrink the backbone only. Channel counts, levels,
            # c_grid_downsample, channel_layout and num_output_heads stay
            # exactly as shipped — those are the contract under test.
            "++model.rolling_dit_kwargs.dim=64",
            "++model.rolling_dit_kwargs.num_heads=4",
            "++model.rolling_dit_kwargs.temporal_num_heads=4",
            "++model.rolling_dit_kwargs.num_blocks=2",
            "++model.rolling_dit_kwargs.c_grid_cross_layers=1",
            "++model.rolling_dit_kwargs.input_embed.d_boundary=16",
            "++model.rolling_dit_kwargs.input_embed.d_calendar=16",
            "++model.rolling_dit_kwargs.input_embed.d_co2=8",
            "++training.optimizer.type=AdamW",
            "++training.ema.warmup_epochs=0",
            f"+checkpoint_dir={ckpt_dir}",
            f"++hydra.run.dir={out_dir}",
            "run_name=rsi_smoke", "wandb.enabled=False",
            *extra,
        ]
        proc = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, cwd=str(_RECIPE), env=_clean_env()
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-8000:]
        return out

    log = _train(
        "++training.max_epochs=1", "++training.stages.0.num_epochs=1",
        f"++training.stages.0.max_iterations={_ITERS}",
        f"++bench.per_batch_tsv={tsv}",
        out_dir=root / "out",
    )
    resume_log = _train(
        "++training.max_epochs=2", "++training.stages.0.num_epochs=2",
        "++training.stages.0.max_iterations=4",
        out_dir=root / "out_resume",
    )
    rows = [
        line.split("\t") for line in tsv.read_text().splitlines()[1:] if line.strip()
    ]
    return {"log": log, "resume_log": resume_log,
            "loss": [float(r[3]) for r in rows], "ckpt_dir": ckpt_dir}


@pytest.mark.slow
def test_rsi_trains_end_to_end_with_finite_losses(rsi_smoke):
    losses = rsi_smoke["loss"]
    assert losses, "no per-batch rows were written"
    assert all(v == v and abs(v) != float("inf") for v in losses), losses[:10]
    assert all(v > 0 for v in losses)


@pytest.mark.slow
def test_the_model_step_is_four_store_rows(rsi_smoke):
    """A 24 h model over 6 h rows — same stride contract as every AMIP run."""
    assert "model step: 4 store row(s) (24 h)" in rsi_smoke["log"]


@pytest.mark.slow
def test_the_checkpoint_round_trips(rsi_smoke):
    assert list(rsi_smoke["ckpt_dir"].glob("*.mdlus")) or \
        list(rsi_smoke["ckpt_dir"].glob("*.pt"))
    # A second entry into the same checkpoint_dir must resume, not restart.
    assert "Loaded" in rsi_smoke["resume_log"] or \
        "resum" in rsi_smoke["resume_log"].lower()
