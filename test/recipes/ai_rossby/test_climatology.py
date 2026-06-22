# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 4c streaming time aggregators.

Each aggregator's running result is checked against the closed-form
numpy reference on the full materialized time series — so a passing
test certifies the streaming implementation is bit-equivalent to the
batch reference (up to float-precision noise) without needing to hold
the series in memory in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from climatology import (  # noqa: E402
    StreamingBinnedMean,
    StreamingTimeMean,
    StreamingTimeVariance,
)


# ---------------------------------------------------------------------------
# StreamingTimeMean
# ---------------------------------------------------------------------------


def test_streaming_time_mean_matches_numpy():
    torch.manual_seed(0)
    shape = (3, 4, 5)
    full = torch.randn(50, *shape).double()
    agg = StreamingTimeMean(shape, torch.device("cpu"))
    # Push in mismatched batch sizes — running stats shouldn't care.
    for chunk in torch.split(full, [5, 12, 1, 17, 8, 7], dim=0):
        agg.update(chunk)
    got = agg.finalize(out_dtype=torch.float64)
    expected = full.mean(dim=0)
    assert torch.allclose(got, expected, atol=1e-12)


def test_streaming_time_mean_empty_returns_zero():
    agg = StreamingTimeMean((2, 3), torch.device("cpu"))
    out = agg.finalize()
    assert torch.equal(out, torch.zeros(2, 3))


def test_streaming_time_mean_shape_check():
    agg = StreamingTimeMean((2, 3), torch.device("cpu"))
    with pytest.raises(ValueError, match=r"expected"):
        agg.update(torch.zeros(2, 4))  # wrong inner shape


# ---------------------------------------------------------------------------
# StreamingTimeVariance
# ---------------------------------------------------------------------------


def test_streaming_time_variance_matches_numpy():
    """Chan's parallel update should reproduce numpy's unbiased variance."""
    torch.manual_seed(1)
    shape = (3, 4)
    # ddof=1 (unbiased Bessel-corrected variance) matches the aggregator.
    full = torch.randn(80, *shape).double()
    agg = StreamingTimeVariance(shape, torch.device("cpu"))
    for chunk in torch.split(full, [3, 1, 20, 50, 6], dim=0):
        agg.update(chunk)
    mean, var = agg.finalize(out_dtype=torch.float64)
    expected_mean = full.mean(dim=0)
    expected_var = full.var(dim=0, unbiased=True)
    assert torch.allclose(mean, expected_mean, atol=1e-12)
    assert torch.allclose(var, expected_var, atol=1e-10)


def test_streaming_time_variance_large_mean_no_cancellation():
    """A common climate-data regime: mean ~273, var ~10. Naive E[X²]-E[X]² cancels;
    Welford should not."""
    torch.manual_seed(2)
    base = 273.15
    full = base + 3.0 * torch.randn(200, 4, 4).double()
    agg = StreamingTimeVariance((4, 4), torch.device("cpu"))
    for chunk in torch.split(full, 7, dim=0):
        agg.update(chunk)
    _, var = agg.finalize(out_dtype=torch.float64)
    expected_var = full.var(dim=0, unbiased=True)
    # Within reasonable f64 epsilon — verifies stability under large
    # offset; the naive formula would have ~1e-6 relative error here
    # in f32 and several ULP in f64.
    rel = (var - expected_var).abs() / expected_var.clamp(min=1e-12)
    assert rel.max() < 1e-9, f"max rel err = {rel.max()}"


def test_streaming_time_variance_n_one_returns_zero():
    """With a single sample, variance is undefined; we set it to zero
    rather than NaN."""
    agg = StreamingTimeVariance((2, 3), torch.device("cpu"))
    agg.update(torch.ones(1, 2, 3))
    mean, var = agg.finalize()
    assert torch.allclose(mean, torch.ones(2, 3), atol=1e-7)
    assert torch.equal(var, torch.zeros(2, 3))


def test_streaming_time_variance_chunked_matches_full():
    """Same input fed as one batch vs many small batches should agree."""
    torch.manual_seed(3)
    full = torch.randn(60, 5, 3).double()
    agg_a = StreamingTimeVariance((5, 3), torch.device("cpu"))
    agg_b = StreamingTimeVariance((5, 3), torch.device("cpu"))
    agg_a.update(full)
    for c in torch.split(full, 1, dim=0):
        agg_b.update(c)
    m_a, v_a = agg_a.finalize(out_dtype=torch.float64)
    m_b, v_b = agg_b.finalize(out_dtype=torch.float64)
    assert torch.allclose(m_a, m_b, atol=1e-12)
    assert torch.allclose(v_a, v_b, atol=1e-10)


# ---------------------------------------------------------------------------
# StreamingBinnedMean
# ---------------------------------------------------------------------------


def test_binned_mean_matches_numpy_groupby():
    torch.manual_seed(4)
    n_bins = 7
    shape = (3, 4)
    n_samples = 50
    full = torch.randn(n_samples, *shape).double()
    bins = torch.randint(0, n_bins, (n_samples,))
    agg = StreamingBinnedMean(n_bins, shape, torch.device("cpu"))
    for i in range(0, n_samples, 8):
        agg.update(full[i : i + 8], bins[i : i + 8])
    got = agg.finalize(out_dtype=torch.float64)
    expected = torch.zeros(n_bins, *shape).double()
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            expected[b] = full[mask].mean(dim=0)
    assert torch.allclose(got, expected, atol=1e-12)


def test_binned_mean_empty_bins_are_zero():
    agg = StreamingBinnedMean(5, (2, 3), torch.device("cpu"))
    agg.update(torch.ones(3, 2, 3), torch.tensor([0, 0, 1]))
    out = agg.finalize()
    # Bins 0 and 1 sampled, 2/3/4 empty → zero.
    assert torch.allclose(out[0], torch.ones(2, 3), atol=1e-7)
    assert torch.allclose(out[1], torch.ones(2, 3), atol=1e-7)
    for empty_bin in (2, 3, 4):
        assert torch.equal(out[empty_bin], torch.zeros(2, 3))


def test_binned_mean_rejects_out_of_range_bin():
    agg = StreamingBinnedMean(3, (2,), torch.device("cpu"))
    with pytest.raises(ValueError, match=r"out of range"):
        agg.update(torch.zeros(2, 2), torch.tensor([0, 3]))


def test_binned_mean_counts_per_bin():
    agg = StreamingBinnedMean(4, (2,), torch.device("cpu"))
    agg.update(torch.zeros(5, 2), torch.tensor([0, 0, 1, 1, 3]))
    counts = agg.counts_per_bin.tolist()
    assert counts == [2, 2, 0, 1]


# ---------------------------------------------------------------------------
# Climatology-bias pattern (the actual Phase 4c usage)
# ---------------------------------------------------------------------------


def test_climatology_bias_pattern_end_to_end():
    """Predicted and truth series share the same length and aggregator;
    the climatological bias is the difference of finalized means."""
    torch.manual_seed(5)
    shape = (3, 4)
    truth = torch.randn(100, *shape).double() + 273.0
    # Pred is a noisy biased estimate of truth.
    pred = truth + 0.5 + 0.1 * torch.randn(100, *shape).double()

    pred_agg = StreamingTimeMean(shape, torch.device("cpu"))
    truth_agg = StreamingTimeMean(shape, torch.device("cpu"))
    for k in range(0, 100, 10):
        pred_agg.update(pred[k : k + 10])
        truth_agg.update(truth[k : k + 10])

    bias = pred_agg.finalize(out_dtype=torch.float64) - truth_agg.finalize(out_dtype=torch.float64)
    # Mean of the noise is ~0, so bias ≈ 0.5 ± O(0.1 / √100) = ± 0.01.
    assert torch.allclose(bias, torch.full(shape, 0.5, dtype=torch.float64), atol=0.03)
