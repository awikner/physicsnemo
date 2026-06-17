# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the faithful ``PanguPlasim`` and ``PanguPlasimLegacy`` ports.

Coverage:

* MOD-008a constructor sweep (3 variants × 2 model classes)
* MOD-008b non-regression against committed reference tensors under
  ``test/models/pangu_plasim/data/<ClassName>_v1.0.pth``
* MOD-008c checkpoint roundtrip via ``Module.from_checkpoint``
* GPU smoke test (``@pytest.mark.smoke @pytest.mark.cuda``) submitted on Delta
  ``gpuA40x4-interactive`` via the ``delta-smoke-test`` skill
"""

import warnings
from pathlib import Path

import pytest
import torch

# Importing from physicsnemo.experimental emits ExperimentalFeatureWarning once
# per session; we know these models are experimental and don't need the noise.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning, module=r"physicsnemo\.experimental.*")
    import physicsnemo
    from physicsnemo.experimental.models.pangu_plasim import (
        PanguPlasim,
        PanguPlasimLegacy,
    )

from test import common

_REFERENCE_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Tiny config used by the smoke tests. Designed to fit comfortably on one A40
# and finish in well under a minute including init.
# ---------------------------------------------------------------------------
_SMOKE_KWARGS = dict(
    surface_variables=["t2m", "u10", "v10"],
    upper_air_variables=["t", "u", "v", "q", "z"],
    constant_boundary_variables=["lsm"],
    # Must contain a recognized solar-radiation name; "rsdt" is one of two
    # accepted by both models (see pangu_plasim.py / pangu_plasim_legacy.py).
    varying_boundary_variables=["rsdt"],
    levels=[200, 300, 500, 700, 850, 925, 1000, 1015],
    horizontal_resolution=[32, 64],
    patch_size=[2, 4, 4],
    depths=[1, 1, 1, 1],
    num_heads=[2, 4, 4, 2],
    embed_dim=64,
    window_size=[2, 4, 8],
)


def _make_inputs(device, batch_size=1):
    """Synthetic inputs matching the smoke-config channel/grid layout."""
    n_lat, n_lon = _SMOKE_KWARGS["horizontal_resolution"]
    n_levels = len(_SMOKE_KWARGS["levels"])
    n_surface = len(_SMOKE_KWARGS["surface_variables"])
    n_upper = len(_SMOKE_KWARGS["upper_air_variables"])
    n_const = len(_SMOKE_KWARGS["constant_boundary_variables"])
    n_vary = len(_SMOKE_KWARGS["varying_boundary_variables"])

    surface_in = torch.randn(batch_size, n_surface, n_lat, n_lon, device=device)
    constant_boundary = torch.randn(n_const, n_lat, n_lon, device=device)
    varying_boundary = torch.randn(batch_size, n_vary, n_lat, n_lon, device=device)
    upper_air_in = torch.randn(
        batch_size, n_upper, n_levels, n_lat, n_lon, device=device
    )
    return surface_in, constant_boundary, varying_boundary, upper_air_in


# ---------------------------------------------------------------------------
# Per-model constructor sweep — runs on whichever device(s) the fixture
# provides (CPU on login nodes, CPU + cuda:0 on GPU nodes).
# ---------------------------------------------------------------------------
_CONSTRUCTOR_VARIANTS = [
    # baseline (no diagnostics, no land/ocean)
    dict(_SMOKE_KWARGS),
    # with upper_air_boundary routing solar into the 3D stream
    dict(_SMOKE_KWARGS, upper_air_boundary=True),
    # with diagnostic variables
    dict(_SMOKE_KWARGS, diagnostic_variables=["clt"]),
]


@pytest.mark.parametrize(
    "model_cls", [PanguPlasim, PanguPlasimLegacy], ids=["vae", "legacy"]
)
@pytest.mark.parametrize(
    "kwargs",
    _CONSTRUCTOR_VARIANTS,
    ids=["baseline", "upper_air_boundary", "diagnostic"],
)
def test_pangu_plasim_constructor(device, model_cls, kwargs):
    """Smoke-style constructor coverage on the parameter axes Phase-1 cares
    about (boundary routing, diagnostic-variable path). Verifies the model
    instantiates and produces output of the expected shape on the fixture
    device.
    """
    torch.manual_seed(0)
    model = model_cls(**kwargs).to(device).eval()

    inputs = _make_inputs(device, batch_size=1)
    with torch.no_grad():
        out = model(*inputs)

    has_diag = bool(kwargs.get("diagnostic_variables"))
    expected_len = 7 if has_diag else 6
    assert len(out) == expected_len, (
        f"expected {expected_len}-tuple for diagnostic={has_diag}, got {len(out)}"
    )

    n_lat, n_lon = kwargs["horizontal_resolution"]
    n_levels = len(kwargs["levels"])
    assert out[0].shape == (1, len(kwargs["surface_variables"]), n_lat, n_lon)
    assert out[1].shape == (
        1,
        len(kwargs["upper_air_variables"]),
        n_levels,
        n_lat,
        n_lon,
    )
    if has_diag:
        assert out[2].shape == (
            1,
            len(kwargs["diagnostic_variables"]),
            n_lat,
            n_lon,
        )
    for t in out:
        assert torch.isfinite(t).all(), "non-finite output"

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "model_cls", [PanguPlasim, PanguPlasimLegacy], ids=["vae", "legacy"]
)
def test_pangu_plasim_checkpoint(device, model_cls, tmp_path):
    """`.mdlus` checkpoint roundtrip: save → ``Module.from_checkpoint`` → forward
    matches the pre-save output on the fixture device. Seeds both forwards
    identically so PanguPlasim's stochastic ``reparameterize`` step yields
    matching latent draws (PanguPlasimLegacy is deterministic in eval mode but
    uses the same protocol for symmetry).
    """
    torch.manual_seed(0)
    model = model_cls(**_SMOKE_KWARGS).to(device).eval()

    inputs = _make_inputs(device, batch_size=1)

    torch.manual_seed(42)
    with torch.no_grad():
        out_pre_save = model(*inputs)

    ckpt_path = tmp_path / f"{model_cls.__name__}_roundtrip.mdlus"
    model.save(str(ckpt_path))

    loaded = physicsnemo.Module.from_checkpoint(str(ckpt_path)).to(device).eval()
    torch.manual_seed(42)
    with torch.no_grad():
        out_loaded = loaded(*inputs)

    assert common.compare_output(out_pre_save, out_loaded, rtol=1e-5, atol=1e-5)

    del model, loaded
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MOD-008b non-regression: load committed reference tensors and compare.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_cls", [PanguPlasim, PanguPlasimLegacy], ids=["vae", "legacy"]
)
def test_pangu_plasim_non_regression(device, model_cls):
    """Forward pass with seeded weights matches committed reference output.

    Per MOD-008b. Reference ``.pth`` files at
    ``test/models/pangu_plasim/data/<ClassName>_v1.0.pth`` were generated under
    ``init_seed=0`` (seeds the model's default initializers — trunc-normal,
    Kaiming), ``input_seed=42``, and ``forward_seed=123`` (seeds the VAE
    ``reparameterize`` draw for PanguPlasim; harmless for PanguPlasimLegacy).
    The MOD-008b example overrides parameters with raw ``randn`` — that
    saturates this transformer to ``NaN``, so we seed the constructor instead.
    """
    ref_path = _REFERENCE_DATA_DIR / f"{model_cls.__name__}_v1.0.pth"
    if not ref_path.exists():
        pytest.skip(f"reference data missing: {ref_path}")

    # Fixtures are CPU-generated. PanguPlasim is stochastic in eval mode
    # (reparameterize calls torch.randn_like), and CPU vs CUDA RNG draws differ
    # even under the same torch.manual_seed — so the bitwise reference only
    # holds on CPU. PanguPlasimLegacy is deterministic and matches on both.
    if device.startswith("cuda") and model_cls is PanguPlasim:
        pytest.skip(
            "PanguPlasim is stochastic in eval mode (reparameterize) and the "
            "reference fixture is CPU-generated; CUDA non-regression would "
            "compare against CPU-seeded latent draws. Checkpoint roundtrip + "
            "smoke test cover CUDA fidelity instead."
        )

    data = torch.load(ref_path, weights_only=False)

    torch.manual_seed(data["init_seed"])
    model = model_cls(**data["kwargs"]).to(device).eval()

    inputs = tuple(
        data["inputs"][k].to(device)
        for k in ("surface_in", "constant_boundary", "varying_boundary", "upper_air_in")
    )
    out_ref = tuple(t.to(device) for t in data["output"])

    torch.manual_seed(data["forward_seed"])
    with torch.no_grad():
        out = model(*inputs)

    # Reference fixtures are generated on CPU; CUDA vs. CPU floating-point drift
    # through a transformer-depth network easily exceeds 1e-5. Use 5e-3 atol
    # (matches the precedent set by the shipped `Pangu` model's
    # validate_forward_accuracy tolerance in test/models/pangu/test_pangu.py).
    assert common.compare_output(out, out_ref, rtol=1e-3, atol=5e-3)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GPU smoke test — required on Delta gpuA40x4-interactive (see hpc/delta.md).
# ---------------------------------------------------------------------------
@pytest.mark.smoke
@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_cls", [PanguPlasim, PanguPlasimLegacy], ids=["vae", "legacy"]
)
def test_pangu_plasim_smoke(tmp_path, model_cls):
    """End-to-end CUDA smoke test for both ``PanguPlasim`` flavors.

    Exercises the full wiring on a real A40:

      1. Instantiate on CUDA.
      2. Forward pass (eval mode; PanguPlasim's VAE encoder-2 branch is off).
      3. Backward on a trivial sum-loss; AdamW step.
      4. ``.mdlus`` checkpoint roundtrip via ``Module.from_checkpoint``;
         post-load forward matches the pre-save output in eval mode.

    PanguPlasim's ``reparameterize`` calls ``torch.randn_like`` unconditionally
    (it is stochastic by design even in eval mode); the checkpoint comparison
    seeds ``torch.manual_seed`` identically before each forward so the latent
    draw matches. PanguPlasimLegacy is deterministic in eval mode but uses the
    same protocol for code symmetry.

    Intentionally **not** marked with the standard ``device`` fixture — smoke
    tests are explicitly CUDA-only per ``hpc/delta.md``.
    """
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = "cuda:0"
    torch.manual_seed(0)

    # 1. Instantiate
    model = model_cls(**_SMOKE_KWARGS).to(device)

    inputs = _make_inputs(device, batch_size=1)

    # 2. Forward (eval; for PanguPlasim, no VAE encoder-2 path)
    model.eval()
    with torch.no_grad():
        out_eval = model(*inputs)

    # Both flavors return a 6-tuple in eval mode with diagnostic_variables empty:
    #   PanguPlasim      -> (surface, upper_air, mu, sigma, 0, 0)
    #   PanguPlasimLegacy-> (surface, upper_air, 0, 0, 0, 0)
    assert len(out_eval) == 6
    out_surface, out_upper_air = out_eval[0], out_eval[1]

    n_lat, n_lon = _SMOKE_KWARGS["horizontal_resolution"]
    n_levels = len(_SMOKE_KWARGS["levels"])
    assert out_surface.shape == (
        1,
        len(_SMOKE_KWARGS["surface_variables"]),
        n_lat,
        n_lon,
    )
    assert out_upper_air.shape == (
        1,
        len(_SMOKE_KWARGS["upper_air_variables"]),
        n_levels,
        n_lat,
        n_lon,
    )
    for t in out_eval:
        assert torch.isfinite(t).all(), "non-finite output"

    # 3. Backward + AdamW step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    out_train = model(*inputs)
    loss = sum(t.sum() for t in out_train if t.requires_grad)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).all()

    # 4. Checkpoint roundtrip
    model.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        out_pre_save = model(*inputs)

    ckpt_path = tmp_path / f"{model_cls.__name__}_smoke.mdlus"
    model.save(str(ckpt_path))

    loaded = physicsnemo.Module.from_checkpoint(str(ckpt_path)).to(device).eval()
    torch.manual_seed(42)
    with torch.no_grad():
        out_loaded = loaded(*inputs)

    assert common.compare_output(out_pre_save, out_loaded, rtol=1e-5, atol=1e-5)

    del model, loaded
    torch.cuda.empty_cache()
