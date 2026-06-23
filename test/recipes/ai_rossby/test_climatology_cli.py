# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the climatology CLI's chunked forecast dumping.

Exercises the new ``run_climatology(forecast_chunk_steps=...)`` path on
a synthetic stub dataset + persistence model. Verifies:

* Files materialize at expected paths via the async writer.
* No two chunk files share a step / time stamp (the user's
  "no repeated dates" requirement).
* The first chunk's first frame is the IC (when included).
* Filenames carry the model + run + chunk index + time range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

# Re-use the inference stubs (located next to this file).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference import _StubDataset, _StubModel  # noqa: E402

from async_writer import AsyncForecastWriter  # noqa: E402
from climatology_cli import _ForecastChunkBuffer, run_climatology  # noqa: E402


# ---------------------------------------------------------------------------
# _ForecastChunkBuffer behavior
# ---------------------------------------------------------------------------


def test_chunk_buffer_fills_and_flushes(tmp_path):
    layout = {
        "surface_variables": ["pl", "tas"],
        "upper_air_variables": ["ta"],
        "diagnostic_variables": [],
        "n_levels": 2,
        "levels_coord": [0.1, 0.5],
        "lat": np.linspace(-90, 90, 4, dtype=np.float32),
        "lon": np.linspace(0, 360, 8, endpoint=False, dtype=np.float32),
        "time": None,
    }
    buf = _ForecastChunkBuffer(
        chunk_steps=3, ensemble_size=1, layout=layout, has_diagnostic=False
    )
    assert buf.is_empty
    for k in (1, 2, 3):
        buf.append(
            step=k,
            time_value=None,
            pred_surface_np=np.full((1, 2, 4, 8), float(k), dtype=np.float32),
            pred_upper_np=np.full((1, 1, 2, 4, 8), float(k), dtype=np.float32),
            pred_diag_np=None,
        )
    assert buf.is_full

    with AsyncForecastWriter(max_in_flight=2, num_workers=1) as writer:
        path = buf.flush(
            writer=writer,
            output_dir=str(tmp_path),
            ic_index=0,
            model_name="SfnoPlasim",
            run_name="ut",
            extension="zarr",
        )
    assert path is not None
    out = xr.open_zarr(path)
    # 3 frames, with step values [1, 2, 3].
    assert out.sizes["frame"] == 3
    np.testing.assert_array_equal(out["step"].values, np.array([1, 2, 3]))
    assert "SfnoPlasim__ut__ic0__" in Path(path).name
    assert int(out.attrs["chunk_index"]) == 0
    # After flush, buffer is reset and chunk index advanced.
    assert buf.is_empty
    assert buf._chunk_idx == 1


# ---------------------------------------------------------------------------
# run_climatology end-to-end (CPU, persistence model)
# ---------------------------------------------------------------------------


def _run_with_chunks(tmp_path, *, chunk_steps, max_step, include_ic=True, ensemble_size=1):
    ds = _StubDataset(n_time=max_step + 4)
    model = _StubModel(n_surface=2, n_upper=5, n_levels=4)
    with AsyncForecastWriter(max_in_flight=2, num_workers=2) as writer:
        agg = run_climatology(
            model,
            ds,
            normalizer=None,
            device=torch.device("cpu"),
            ic_indices=[0],
            max_step=max_step,
            n_bins=2,
            steps_per_bin=1,
            ensemble_size=ensemble_size,
            perturber=None,
            has_diagnostic=False,
            seed=0,
            track_bins=True,
            track_binned_variance=False,
            forecast_chunk_steps=chunk_steps,
            forecast_writer=writer,
            forecast_output_dir=str(tmp_path),
            model_name="SfnoPlasim",
            run_name="ctest",
            forecast_extension="zarr",
            include_ic_in_forecast=include_ic,
        )
    return agg


def test_run_climatology_writes_chunked_files_no_overlap(tmp_path):
    agg = _run_with_chunks(tmp_path, chunk_steps=3, max_step=7, include_ic=True)
    paths = agg["forecast_paths"]
    # max_step=7 with chunk_steps=3 + IC in chunk 0 →
    #   chunk 0: step 0 (IC), 1, 2, 3 → 4 frames (fills then flushes after
    #            step 3 made the buffer reach 4 frames; depending on the
    #            ordering it might flush at 3 or 4 — verify with attrs)
    #   subsequent chunks: 3 frames each (steps 4-6) and the leftover.
    # We verify the strict "no repeated step" invariant directly:
    seen_steps: set[int] = set()
    for p in paths:
        ds = xr.open_zarr(p)
        steps = ds["step"].values.tolist()
        for s in steps:
            assert s not in seen_steps, f"step {s} duplicated across chunk files"
            seen_steps.add(s)
    # Union of seen steps should cover 0..7 (IC + steps 1..7).
    assert seen_steps == set(range(0, 8))


def test_run_climatology_first_chunk_first_frame_is_ic(tmp_path):
    agg = _run_with_chunks(tmp_path, chunk_steps=3, max_step=4, include_ic=True)
    paths = sorted(agg["forecast_paths"])
    chunk0 = xr.open_zarr(paths[0])
    # Step 0 in chunk 0 holds the IC.
    assert int(chunk0["step"].values[0]) == 0
    # With persistence model on a non-shuffled stub, the IC frame equals
    # ds._surface[0].
    ds_stub = _StubDataset(n_time=8)
    expected_ic = ds_stub._surface[0].numpy()
    np.testing.assert_allclose(
        chunk0["pred_surface"].values[0, 0], expected_ic, atol=1e-6
    )


def test_run_climatology_no_chunks_when_disabled(tmp_path):
    agg = _run_with_chunks(tmp_path, chunk_steps=None, max_step=4)
    assert agg["forecast_paths"] == []


def test_run_climatology_filename_includes_chunk_idx_and_ic(tmp_path):
    agg = _run_with_chunks(tmp_path, chunk_steps=2, max_step=4, include_ic=False)
    for p in agg["forecast_paths"]:
        bn = Path(p).name
        assert "SfnoPlasim__" in bn
        assert "ctest__ic0__" in bn
        assert "chunk" in bn
        assert bn.endswith(".zarr")


class _AffineNormalizer:
    """Identical to the one in test_validate.py — local copy keeps the
    test file self-contained (test_validate is intentionally not imported)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        out = dict(sample)
        for k in ("surface_in", "upper_air_in", "target_surface", "target_upper_air"):
            if k in out:
                out[k] = (out[k] - self.mean) / self.std
        return out

    def denormalize_state(self, *, surface=None, upper_air=None, diagnostic=None):
        out = {}
        if surface is not None:
            out["surface"] = surface * self.std + self.mean
        if upper_air is not None:
            out["upper_air"] = upper_air * self.std + self.mean
        if diagnostic is not None:
            out["diagnostic"] = diagnostic
        return out


def test_run_climatology_aggregator_stats_in_physical_units(tmp_path):
    """With a normalizer in play, the finalized climatology aggregator's
    pred mean must match the raw (un-normalized) IC's mean — proving the
    aggregator accumulates in PHYSICAL units, not normalized space."""
    ds = _StubDataset(n_time=8)
    model = _StubModel(n_surface=2, n_upper=5, n_levels=4)
    norm = _AffineNormalizer(mean=10.0, std=2.0)
    with AsyncForecastWriter(max_in_flight=2, num_workers=2) as writer:
        agg = run_climatology(
            model, ds,
            normalizer=norm,
            device=torch.device("cpu"),
            ic_indices=[0],
            max_step=2,
            n_bins=2,
            steps_per_bin=1,
            ensemble_size=1,
            perturber=None,
            has_diagnostic=False,
            seed=0,
            track_bins=True,
            track_binned_variance=False,
            forecast_chunk_steps=None,  # aggregator only
            forecast_writer=writer,
            forecast_output_dir=str(tmp_path),
            model_name="SfnoPlasim",
            run_name="phys",
            forecast_extension="zarr",
            include_ic_in_forecast=False,
        )
    # Persistence model returns the input each step → pred_surface at step k
    # equals the (normalized) IC. After denormalize, the aggregator should
    # have seen IC value=ds._surface[0] in raw units, repeated twice.
    expected_raw = ds._surface[0]  # (2, 8, 8) raw units
    pred_mean = agg["pred"]["surface"]["mean"].cpu()  # finalized = mean of inputs
    # The persistence rollout produces the same IC at each of the two scored
    # steps, so the mean equals the IC itself.
    assert torch.allclose(pred_mean, expected_raw, atol=1e-4), (
        f"pred mean {pred_mean.mean():.4f} != raw IC mean "
        f"{expected_raw.mean():.4f}; aggregator likely accumulating in normalized space"
    )


def test_run_climatology_ensemble_chunk_shape(tmp_path):
    agg = _run_with_chunks(
        tmp_path, chunk_steps=2, max_step=4, include_ic=False, ensemble_size=3
    )
    # Persistence model + Replicate-only perturber produces 3 identical
    # ensemble members.
    chunk0 = xr.open_zarr(agg["forecast_paths"][0])
    # (ensemble=3, frame=2, surface_var=2, lat=8, lon=8)
    assert chunk0["pred_surface"].shape == (3, 2, 2, 8, 8)


def test_run_climatology_aggregator_path_unchanged_by_chunking(tmp_path):
    """The climatology aggregator state should match whether or not we
    happen to also be dumping chunk files alongside."""
    agg_with = _run_with_chunks(tmp_path, chunk_steps=2, max_step=6)
    agg_without = _run_with_chunks(tmp_path, chunk_steps=None, max_step=6)
    # Both runs feed the same persistence model + same fixed seed; the
    # aggregator's per-pixel mean must be identical.
    np.testing.assert_allclose(
        agg_with["pred"]["surface"]["mean"].numpy(),
        agg_without["pred"]["surface"]["mean"].numpy(),
        atol=1e-7,
    )
