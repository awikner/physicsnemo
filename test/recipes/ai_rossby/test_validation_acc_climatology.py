# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""ACC only exists if a climatology is supplied — pin that, and its space.

``DiffusionRolloutValidator`` builds no ACC accumulators unless handed a
climatology, so ``metrics.acc: True`` in the validation config was silently
inert from the day it was written until 2026-09-04: the trainer never passed
one, the validator returned RMSE only, and nothing anywhere said so. These
tests make the difference observable.

They also pin the two traps that make an ACC number wrong rather than absent:
ACC is scored in NORMALIZED space (unlike RMSE), and the climatology has to be
a per-PIXEL field — a per-channel scalar leaves the mean state inside the
"anomalies" and inflates ACC.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

_TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TEST_DIR))
_AI_ROSSBY_DIR = _TEST_DIR.parents[2] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from test_validate_diffusion import (  # noqa: E402
        _OracleEchoScheduler,
        _StubDataset,
        _StubWrapper,
    )
    from validate_diffusion import DiffusionRolloutValidator  # noqa: E402

C, H, W = 2, 8, 8


def _validator(climatology_surface=None, *, ensemble_size=1, perturber=None):
    kw = {}
    if perturber is not None:
        kw["perturber"] = perturber
    return DiffusionRolloutValidator(
        _StubDataset(C=C, H=H, W=W),
        wrapper=_StubWrapper(),
        inference_scheduler=_OracleEchoScheduler(window_size=3),
        log_steps=[1, 3],
        device=torch.device("cpu"),
        horizon=3,
        max_initial_conditions=1,
        batch_size=1,
        ic_stride=1,
        ensemble_size=ensemble_size,
        climatology_surface=climatology_surface,
        **kw,
    )


def test_no_climatology_means_no_acc_keys():
    """The pre-2026-09-04 behaviour, stated so a regression is visible."""
    v = _validator(None)
    assert v.acc_surface is None
    keys = v._finalize(local_only=True).keys()
    assert not any(k.startswith("acc_") for k in keys), sorted(keys)
    assert any(k.startswith("rmse_") for k in keys)


def test_climatology_creates_acc_accumulators_and_keys():
    v = _validator(torch.zeros(C, H, W))
    assert v.acc_surface is not None
    keys = v._finalize(local_only=True).keys()
    assert {"acc_step1_surface", "acc_step3_surface"} <= set(keys), sorted(keys)


def test_per_pixel_climatology_is_accepted_and_used():
    """A (C, H, W) field must broadcast against (B, C, H, W) predictions."""
    clim = torch.randn(C, H, W)
    v = _validator(clim)
    assert v.acc_surface.climatology.shape == (C, H, W)
    # A different climatology must change the ACC value — i.e. it is really
    # subtracted, not merely stored.
    pred = torch.randn(1, C, H, W)
    truth = torch.randn(1, C, H, W)
    lw = torch.ones(H)
    a = _validator(torch.zeros(C, H, W))
    b = _validator(clim)
    a.acc_surface.update(0, pred, truth, lw)
    b.acc_surface.update(0, pred, truth, lw)
    va = a.acc_surface.finalize(local_only=True)[0]
    vb = b.acc_surface.finalize(local_only=True)[0]
    assert not torch.allclose(va, vb), "climatology had no effect on ACC"


def test_acc_is_invariant_to_a_shared_affine_rescale():
    """Why the climatology must live in the SAME space as pred/truth.

    ACC is a correlation of anomalies, so scaling pred, truth AND climatology
    by one per-channel affine leaves it unchanged. That is what makes a
    normalized climatology correct here — and what makes mixing a PHYSICAL
    climatology with normalized tensors wrong.
    """
    clim = torch.randn(C, H, W)
    pred = torch.randn(1, C, H, W)
    truth = torch.randn(1, C, H, W)
    lw = torch.rand(H) + 0.5

    plain = _validator(clim)
    plain.acc_surface.update(0, pred, truth, lw)
    got = plain.acc_surface.finalize(local_only=True)[0]

    # (C,1,1) broadcasts against the (C,H,W) climatology; the same stats
    # unsqueezed to (1,C,1,1) broadcast against the (B,C,H,W) tensors.
    mu = torch.randn(C, 1, 1)
    sigma = torch.rand(C, 1, 1) + 1.0
    mu_b, sigma_b = mu.unsqueeze(0), sigma.unsqueeze(0)
    scaled = _validator((clim - mu) / sigma)
    scaled.acc_surface.update(
        0, (pred - mu_b) / sigma_b, (truth - mu_b) / sigma_b, lw
    )
    got_scaled = scaled.acc_surface.finalize(local_only=True)[0]
    assert torch.allclose(got, got_scaled, atol=1e-5), (got, got_scaled)


def test_deterministic_perturber_rejects_the_new_ensemble_size():
    """Guards the config trap: ensemble_size 10 + perturber deterministic.

    Raising is correct, but it raises at VALIDATION time — i.e. epoch 5 of a
    multi-day run — so the pairing is pinned here instead.
    """
    from validate import Deterministic, ReplicateOnly

    with pytest.raises(ValueError, match="ensemble_size=1"):
        Deterministic()({"surface_in": torch.zeros(1, C, H, W)}, 10)
    out = ReplicateOnly()({"surface_in": torch.zeros(1, C, H, W)}, 10)
    assert out["surface_in"].shape[0] == 10
