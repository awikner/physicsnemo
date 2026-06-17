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

"""Tests for the faithful ``PanguPlasim`` port.

The Phase-1 unit tests will go here too (constructor parameter coverage, forward
shape against a committed reference tensor, checkpoint roundtrip, etc.). This
file currently carries the smoke test that proves the model wires together on a
real A40 — submitted via the ``delta-smoke-test`` skill on Delta's
``gpuA40x4-interactive`` partition (see ``hpc/delta.md``).
"""

import pytest
import torch

import physicsnemo
from physicsnemo.models.pangu_plasim import PanguPlasim
from test import common


# ---------------------------------------------------------------------------
# Tiny config used by the smoke test. Designed to fit comfortably on one A40
# and finish in < 1 minute including compile time.
# ---------------------------------------------------------------------------
_SMOKE_KWARGS = dict(
    surface_variables=["t2m", "u10", "v10"],
    upper_air_variables=["t", "u", "v", "q", "z"],
    constant_boundary_variables=["lsm"],
    # Must contain a recognized solar-radiation name; "rsdt" is one of two
    # accepted by the model (see pangu_plasim.py).
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


@pytest.mark.smoke
@pytest.mark.cuda
def test_pangu_plasim_smoke(tmp_path):
    """End-to-end CUDA smoke test for ``PanguPlasim``.

    Exercises the full wiring on a real A40:

      1. Instantiate on CUDA.
      2. Forward pass (eval mode; the VAE encoder-2 branch is off).
      3. Backward on a trivial sum-loss; AdamW step.
      4. ``.mdlus`` checkpoint roundtrip via ``Module.from_checkpoint``;
         post-load forward matches the pre-save output in eval mode.

    Intentionally **not** marked with the standard ``device`` fixture — smoke
    tests are explicitly CUDA-only per ``hpc/delta.md``.
    """
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    device = "cuda:0"
    torch.manual_seed(0)

    # 1. Instantiate
    model = PanguPlasim(**_SMOKE_KWARGS).to(device)

    inputs = _make_inputs(device, batch_size=1)

    # 2. Forward (eval, no VAE encoder-2)
    model.eval()
    with torch.no_grad():
        out_eval = model(*inputs)

    # PanguPlasim returns a 6-tuple when diagnostic_variables is empty (this
    # smoke config): (surface, upper_air, mu, sigma, mu2_zero, sigma2_zero).
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

    # 4. Checkpoint roundtrip. The VAE ``reparameterize`` step calls
    # ``torch.randn_like`` unconditionally — the model is stochastic in eval
    # mode by design. Seed identically before each forward so the latent draw
    # matches and the comparison reflects only checkpoint fidelity.
    model.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        out_pre_save = model(*inputs)

    ckpt_path = tmp_path / "pangu_plasim_smoke.mdlus"
    model.save(str(ckpt_path))

    loaded = physicsnemo.Module.from_checkpoint(str(ckpt_path)).to(device).eval()
    torch.manual_seed(42)
    with torch.no_grad():
        out_loaded = loaded(*inputs)

    assert common.compare_output(out_pre_save, out_loaded, rtol=1e-5, atol=1e-5)

    del model, loaded
    torch.cuda.empty_cache()
