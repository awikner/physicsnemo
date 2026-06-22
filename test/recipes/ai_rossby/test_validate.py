# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 4a rollout validator.

Covers the streaming metrics (analytic verification), the perturber
ensemble API, and the rollout driver wired to a tiny stub model on a
synthetic ``PlasimClimateDataset``-shaped object. CPU-only; the real
multi-GPU + real-data scenario is covered by the Delta smoke test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Make the ai_rossby example importable without installing it.
_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from validate import (  # noqa: E402
    Deterministic,
    GaussianIC,
    Perturber,
    ReplicateOnly,
    RolloutValidator,
    StreamingLatWeightedACC,
    StreamingLatWeightedRMSE,
    cos_lat_weights,
)


# ---------------------------------------------------------------------------
# Perturbers
# ---------------------------------------------------------------------------


def _make_sample(B=2, Cs=2, Cu=3, L=4, H=8, W=16):
    return {
        "surface_in": torch.randn(B, Cs, H, W),
        "upper_air_in": torch.randn(B, Cu, L, H, W),
        "constant_boundary": torch.randn(B, 1, H, W),
        "varying_boundary": torch.randn(B, 2, H, W),
        "target_surface": torch.randn(B, Cs, H, W),
        "target_upper_air": torch.randn(B, Cu, L, H, W),
    }


def test_deterministic_rejects_ensemble_size_gt_one():
    s = _make_sample()
    with pytest.raises(ValueError):
        Deterministic()(s, ensemble_size=4)


def test_deterministic_passes_through_unchanged():
    s = _make_sample()
    out = Deterministic()(s, ensemble_size=1)
    for k in s:
        assert torch.equal(out[k], s[k])


def test_replicate_only_grows_batch_dim_and_preserves_values():
    s = _make_sample(B=3)
    E = 5
    out = ReplicateOnly()(s, ensemble_size=E)
    for k in s:
        assert out[k].shape[0] == s[k].shape[0] * E
    # repeat_interleave is ordered: [b0]*E ++ [b1]*E ++ ... -> contiguous blocks.
    sb = out["surface_in"]
    for b in range(3):
        block = sb[b * E : (b + 1) * E]
        # All E rows in a block equal the original b-th row.
        assert torch.equal(block[0], s["surface_in"][b])
        assert torch.equal(block[-1], s["surface_in"][b])


def test_gaussian_ic_adds_noise_to_listed_keys_only():
    torch.manual_seed(0)
    s = _make_sample(B=2)
    E = 4
    p = GaussianIC(scales={"surface_in": 0.1, "upper_air_in": 0.05})
    out = p(s, ensemble_size=E, generator=torch.Generator().manual_seed(123))
    # Replicate-then-add, so first replica is *not* equal to the base (noise added).
    base = s["surface_in"][0]
    rep0 = out["surface_in"][0]
    assert not torch.equal(base, rep0)
    # `constant_boundary` and `varying_boundary` were NOT in scales → identical to replicated.
    rep_const0 = out["constant_boundary"][0]
    assert torch.equal(rep_const0, s["constant_boundary"][0])
    # Noise magnitude roughly matches std (sanity, not exact).
    diff = (rep0 - base).flatten()
    # ~0.1 std should give an L2-norm-scaled like sqrt(N)*0.1; just check non-trivial.
    assert diff.abs().mean() > 1e-3


def test_gaussian_ic_is_deterministic_with_explicit_generator():
    s = _make_sample(B=2)
    p = GaussianIC(scales={"surface_in": 0.1})
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = p(s, ensemble_size=3, generator=g1)
    b = p(s, ensemble_size=3, generator=g2)
    assert torch.allclose(a["surface_in"], b["surface_in"])


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------


def test_cos_lat_weights_normalized():
    w = cos_lat_weights(64, torch.device("cpu"), torch.float32)
    assert w.shape == (64,)
    assert math.isclose(w.mean().item(), 1.0, abs_tol=1e-6)
    # poles down-weighted, equator up-weighted
    assert w[0] < w[32] and w[-1] < w[32]


def test_streaming_rmse_matches_closed_form_no_lat_weighting():
    """With uniform weights (= 1), the streaming RMSE collapses to ordinary RMSE."""
    n_steps, n_channels, H, W = 2, 3, 4, 5
    metric = StreamingLatWeightedRMSE(
        n_steps=n_steps, n_channels=n_channels, device=torch.device("cpu")
    )
    uniform = torch.ones(H)  # cos_lat=1 → unweighted
    torch.manual_seed(0)
    pred = torch.randn(7, n_channels, H, W)
    target = torch.randn(7, n_channels, H, W)

    metric.update(0, pred, target, uniform)
    rmse = metric.finalize()
    # Closed form: per-channel RMSE over (B, H, W).
    diff_sq = (pred - target).pow(2)
    expected = torch.sqrt(diff_sq.mean(dim=(0, 2, 3)))
    assert torch.allclose(rmse[0], expected, atol=1e-6)
    # The unused step-1 row stays at NaN/0 — division by zero in finalize is clamped.
    # Verify it doesn't blow up; value is sqrt(0/eps) -> 0.
    assert torch.isfinite(rmse[1]).all()


def test_streaming_rmse_with_lat_weighting_matches_manual():
    """Lat-weighted RMSE against a hand-computed reference."""
    H, W, C = 6, 4, 2
    weights = cos_lat_weights(H, torch.device("cpu"), torch.float32)
    metric = StreamingLatWeightedRMSE(
        n_steps=1, n_channels=C, device=torch.device("cpu")
    )
    torch.manual_seed(1)
    pred = torch.randn(3, C, H, W)
    target = torch.randn(3, C, H, W)
    metric.update(0, pred, target, weights)
    rmse = metric.finalize()

    # Manual: per channel, sum_w (pred-target)^2 / sum_w
    diff_sq = (pred - target).pow(2)
    w_b = weights.view(1, 1, H, 1)
    num = (diff_sq * w_b).sum(dim=(0, 2, 3))
    den = w_b.expand_as(diff_sq).sum(dim=(0, 2, 3))
    expected = torch.sqrt(num / den)
    assert torch.allclose(rmse[0], expected, atol=1e-6)


def test_streaming_rmse_accumulates_across_multiple_updates():
    """Two half-batches should give the same RMSE as one big batch."""
    H, W, C = 4, 4, 2
    weights = cos_lat_weights(H, torch.device("cpu"), torch.float32)
    torch.manual_seed(2)
    pred = torch.randn(8, C, H, W)
    target = torch.randn(8, C, H, W)

    m_split = StreamingLatWeightedRMSE(n_steps=1, n_channels=C, device=torch.device("cpu"))
    m_split.update(0, pred[:5], target[:5], weights)
    m_split.update(0, pred[5:], target[5:], weights)
    r_split = m_split.finalize()

    m_full = StreamingLatWeightedRMSE(n_steps=1, n_channels=C, device=torch.device("cpu"))
    m_full.update(0, pred, target, weights)
    r_full = m_full.finalize()

    assert torch.allclose(r_split, r_full, atol=1e-6)


def test_streaming_acc_perfect_prediction_is_one():
    """ACC of a field against itself (with anomalies) should be exactly 1."""
    H, W, C = 4, 4, 2
    weights = cos_lat_weights(H, torch.device("cpu"), torch.float32)
    clim = torch.zeros(C, H, W)
    torch.manual_seed(3)
    target = torch.randn(2, C, H, W)
    metric = StreamingLatWeightedACC(
        n_steps=1, n_channels=C, climatology=clim, device=torch.device("cpu")
    )
    metric.update(0, target.clone(), target, weights)
    acc = metric.finalize()
    assert torch.allclose(acc[0], torch.ones(C), atol=1e-5)


def test_streaming_acc_zero_for_independent_random():
    """Statistical sanity: independent random fields → ACC near 0."""
    H, W, C = 16, 32, 2
    weights = cos_lat_weights(H, torch.device("cpu"), torch.float32)
    clim = torch.zeros(C, H, W)
    torch.manual_seed(4)
    metric = StreamingLatWeightedACC(
        n_steps=1, n_channels=C, climatology=clim, device=torch.device("cpu")
    )
    # Many independent batches → law of large numbers.
    for _ in range(40):
        p = torch.randn(8, C, H, W)
        t = torch.randn(8, C, H, W)
        metric.update(0, p, t, weights)
    acc = metric.finalize()
    assert (acc[0].abs() < 0.05).all(), f"ACC not near 0 for independent fields: {acc[0]}"


# ---------------------------------------------------------------------------
# RolloutValidator with a stub model + stub dataset
# ---------------------------------------------------------------------------


class _StubDataset:
    """Minimal ``PlasimClimateDataset``-shaped dataset for unit testing.

    Emits a coherent time series of synthetic samples. ``__getitem__((t, lead))``
    behaves like the real dataset: returns a sample dict with ``surface_in``,
    ``upper_air_in``, ``constant_boundary``, ``varying_boundary``,
    ``target_surface``, ``target_upper_air``.
    """

    def __init__(self, n_time=20, Cs=2, Cu=3, L=4, H=8, W=16, has_diagnostic=False):
        self.n_time = n_time
        self.Cs, self.Cu, self.L, self.H, self.W = Cs, Cu, L, H, W
        torch.manual_seed(0)
        # Pre-bake a deterministic time series so consecutive calls are consistent.
        self._surface = torch.randn(n_time, Cs, H, W)
        self._upper = torch.randn(n_time, Cu, L, H, W)
        self._const = torch.randn(1, H, W)  # broadcastable
        self._varying = torch.randn(n_time, 2, H, W)
        self.has_diagnostic = has_diagnostic
        if has_diagnostic:
            self._diag = torch.randn(n_time, 1, H, W)
        self.transform = None

    def __len__(self):
        return self.n_time

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            t, lead = idx
        else:
            t, lead = int(idx), 1
        target_t = t + lead
        out = {
            "surface_in": self._surface[t],
            "upper_air_in": self._upper[t],
            "constant_boundary": self._const,
            "varying_boundary": self._varying[t],
            "target_surface": self._surface[target_t],
            "target_upper_air": self._upper[target_t],
        }
        if self.has_diagnostic:
            out["diagnostic"] = self._diag[target_t]
        return out


class _StubModel(torch.nn.Module):
    """Identity-ish model: returns the input surface/upper_air unchanged.

    Useful for verifying the rollout driver's plumbing: predicted next state
    equals the input state, so we score persistence-like predictions and the
    metric values are non-trivial but finite.
    """

    def __init__(self, has_diagnostic=False):
        super().__init__()
        self.has_diagnostic = has_diagnostic

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in, **_):
        # Persistence: output is the same shape as input on the surface/upper_air axes.
        if self.has_diagnostic:
            # Fake diag head: zeros sized like the surface, one channel.
            diag = torch.zeros_like(surface_in[:, :1])
            return surface_in, upper_air_in, diag, 0, 0, 0, 0
        return surface_in, upper_air_in, 0, 0, 0, 0


def test_rollout_validator_runs_and_emits_expected_keys():
    ds = _StubDataset(n_time=10)
    model = _StubModel()
    rv = RolloutValidator(
        dataset=ds,
        log_steps=[1, 2, 3],
        device=torch.device("cpu"),
        ensemble_size=1,
        perturber=Deterministic(),
        has_diagnostic=False,
        batch_size=2,
        max_initial_conditions=4,
        ic_stride=1,
    )
    out = rv.run(model, epoch=0)
    # Two metric groups (surface, upper_air) × three steps = 6 entries.
    keys = sorted(out.keys())
    assert keys == sorted(
        [
            "rmse_step1_surface",
            "rmse_step2_surface",
            "rmse_step3_surface",
            "rmse_step1_upper_air",
            "rmse_step2_upper_air",
            "rmse_step3_upper_air",
        ]
    )
    # All values must be finite, non-negative scalars.
    for k, v in out.items():
        assert math.isfinite(v) and v >= 0, f"bad metric {k}={v}"


def test_rollout_validator_ensemble_runs_and_reduces():
    """Ensemble path should not crash and should still emit single-value metrics per step."""
    ds = _StubDataset(n_time=10)
    model = _StubModel()
    rv = RolloutValidator(
        dataset=ds,
        log_steps=[1, 2],
        device=torch.device("cpu"),
        ensemble_size=3,
        perturber=GaussianIC(scales={"surface_in": 0.01}),
        has_diagnostic=False,
        batch_size=2,
        max_initial_conditions=2,
        ic_stride=1,
    )
    out = rv.run(model, epoch=0)
    assert set(out.keys()) == {
        "rmse_step1_surface",
        "rmse_step2_surface",
        "rmse_step1_upper_air",
        "rmse_step2_upper_air",
    }
    for v in out.values():
        assert math.isfinite(v) and v >= 0


def test_rollout_validator_perfect_model_zero_rmse():
    """A model that returns the target exactly should produce ~0 RMSE."""
    ds = _StubDataset(n_time=8)

    class PerfectModel(torch.nn.Module):
        has_diagnostic = False

        def __init__(self, ds):
            super().__init__()
            self.ds = ds

        def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in, **_):
            # Walk one step forward by looking up the dataset's next sample.
            # We don't have a t index here, so cheat: assume input *is* a
            # past frame of the dataset; identify which frame by exact match.
            for t in range(self.ds.n_time):
                if surface_in.shape[0] >= 1 and torch.equal(
                    surface_in[0], self.ds._surface[t]
                ):
                    next_t = t + 1
                    if next_t < self.ds.n_time:
                        s = self.ds._surface[next_t].unsqueeze(0).expand_as(surface_in)
                        u = self.ds._upper[next_t].unsqueeze(0).expand_as(upper_air_in)
                        return s, u, 0, 0, 0, 0, 0
            return surface_in, upper_air_in, 0, 0, 0, 0, 0

    rv = RolloutValidator(
        dataset=ds,
        log_steps=[1],
        device=torch.device("cpu"),
        ensemble_size=1,
        perturber=Deterministic(),
        has_diagnostic=False,
        batch_size=1,
        max_initial_conditions=1,
        ic_stride=1,
    )
    out = rv.run(PerfectModel(ds), epoch=0)
    # surface RMSE should be ~zero for step 1 (model returned the exact next frame).
    assert out["rmse_step1_surface"] < 1e-5
    assert out["rmse_step1_upper_air"] < 1e-5


def test_streaming_metric_state_shapes_match_expected_for_known_layout():
    """Internal buffer sanity for a typical SFNO_PLASIM_5412 layout."""
    metric = StreamingLatWeightedRMSE(
        n_steps=5, n_channels=2, device=torch.device("cpu")
    )
    assert metric.sum_sq_w.shape == (5, 2)
    assert metric.weight_total.shape == (5, 2)
