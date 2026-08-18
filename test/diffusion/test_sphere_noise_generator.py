# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""``SphereNoiseGenerator`` against the torch_harmonics mmax convention (2026-08-18).

The generator built its coefficient tensor with width ``l_max + 1``. That is not a
constant of the maths, it is a convention of the library: for ``lmax=180``,
torch_harmonics 0.8.0 reports ``mmax=181`` and 0.9.1 reports ``mmax=180``, and
``sht.py`` asserts ``x.shape[-1] == self.mmax``. So the hardcoded width worked on
0.8.0 and died on 0.9.1 with a bare ``AssertionError`` naming neither number.

That mattered because ``conf/sampler/x_ddc.yaml`` ships ``noise: spherical``: every
x_DDC rollout on an environment with the newer library hit it. Found while
benchmarking the v2 families — it passed on a laptop (0.8.0) and failed on Delta
(0.9.1), the same shape of environment-dependent failure as the OMP thread-count
one, and worth a test that asks the library rather than restating its answer.
"""

from __future__ import annotations

import warnings

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from physicsnemo.experimental.diffusion._utils import SphereNoiseGenerator

pytest.importorskip("torch_harmonics")


@pytest.mark.parametrize("l_max", [16, 32])
def test_coefficient_width_follows_the_transform_not_a_constant(l_max):
    """The whole bug in one assertion: ask the transform, do not assume."""
    g = SphereNoiseGenerator(l_max=l_max)
    assert g.m_max == int(g.isht.mmax), (
        "coefficient width must track the installed torch_harmonics' mmax; "
        "hardcoding l_max + 1 breaks on >= 0.9"
    )


def test_forward_produces_a_normalized_field_on_the_expected_grid():
    l_max = 32
    g = SphereNoiseGenerator(l_max=l_max)
    out = g(2, 3, device="cpu")
    # The transform is built nlat=l_max, nlon=2*l_max.
    assert tuple(out.shape) == (2, 3, l_max, 2 * l_max)
    assert torch.isfinite(out).all()
    # forward() standardizes per field, which is what makes it usable as noise.
    per_field = out.reshape(2, 3, -1)
    assert per_field.mean(dim=-1).abs().max() < 1e-4
    assert (per_field.std(dim=-1) - 1.0).abs().max() < 1e-3


def test_a_lower_band_limit_zeroes_the_high_degrees_rather_than_reshaping():
    """Passing l_max keeps the coefficient tensor full-width and masks it.

    Pinned because the masking branch indexes ``coeffs[:, l_max:, :]`` — if the
    width ever went back to being derived from the *argument* instead of the
    transform, this path would build a tensor the transform rejects.
    """
    g = SphereNoiseGenerator(l_max=32)
    out = g(1, 2, device="cpu", l_max=8)
    assert tuple(out.shape) == (1, 2, 32, 64)
    assert torch.isfinite(out).all()


def test_a_band_limit_above_the_generators_own_is_refused():
    g = SphereNoiseGenerator(l_max=16)
    with pytest.raises(ValueError, match="exceeds this generator"):
        g(1, 1, device="cpu", l_max=64)
