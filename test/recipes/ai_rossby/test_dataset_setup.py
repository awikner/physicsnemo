# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12d.13 tests for the shared forcing pipeline (dataset_setup).

Pins the two divergences that motivated centralizing the chain:

* **Order** — NaN-fill values are in PHYSICAL units, so the fill must run
  before the normalizer. ``train_diffusion`` previously composed
  ``nan_fill(normalizer(sample))``, which substituted 270 K into z-scored
  space (~+20σ over every masked gridpoint).
* **Contract** — the model must be sized for the ``c_grid`` / ``c_scalar``
  the pipeline actually emits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from dataset_setup import (  # noqa: E402
    build_forcing_assembler,
    build_forcing_pipeline,
    build_nan_fill,
)

_H, _W = 4, 8
_SST_MEAN, _SST_STD = 285.0, 10.0


def _cfg(**dataset_overrides):
    data = {
        "nan_fill_values": {"sst": 270.0},
        "nan_fill_default": 0.0,
    }
    data.update(dataset_overrides)
    return OmegaConf.create(
        {
            "model": {
                "surface_variables": ["t2m"],
                "diagnostic_variables": [],
                "constant_boundary_variables": ["lsm"],
                "varying_boundary_variables": ["global_mean_co2", "sst"],
                "scalar_routed_boundary_variables": ["global_mean_co2"],
            },
            "dataset": data,
        }
    )


class _StubNormalizer:
    """z-scores ``varying_boundary`` channel-wise, like ClimateNormalizer."""

    def __init__(self):
        self.mean = torch.tensor([0.0, _SST_MEAN]).view(-1, 1, 1)
        self.std = torch.tensor([1.0, _SST_STD]).view(-1, 1, 1)

    def __call__(self, sample):
        out = dict(sample)
        out["varying_boundary"] = (out["varying_boundary"] - self.mean) / self.std
        return out

    def to(self, device):
        return self

    def denormalize_state(self, **kw):  # pragma: no cover - proxy delegation
        return kw


def _sample(co2: float = 380.0):
    vb = torch.empty(2, _H, _W)
    vb[0] = co2
    vb[1] = _SST_MEAN
    vb[1, :, :4] = float("nan")  # "land" half of SST
    return {
        "varying_boundary": vb,
        "constant_boundary": torch.zeros(1, _H, _W),
        "surface_in": torch.zeros(1, _H, _W),
        "calendar": torch.tensor([0.0, 12.0]),
    }


# ---------------------------------------------------------------------------
# Ordering: fill in physical units, THEN normalize.
# ---------------------------------------------------------------------------


def test_fill_runs_before_normalizer():
    pipeline = build_forcing_pipeline(_cfg(), normalizer=_StubNormalizer())
    out = pipeline.dataset_transform(_sample())
    sst = out["varying_boundary"][0]  # co2 popped -> sst is channel 0 now
    # 270 K filled BEFORE z-scoring becomes (270-285)/10 = -1.5.
    assert float(sst[0, 0]) == pytest.approx(-1.5)
    # Ocean half is the mean -> 0.
    assert float(sst[0, -1]) == pytest.approx(0.0)


def test_inverted_order_would_leave_the_raw_fill_value():
    # Guard the regression itself: the old train_diffusion compose
    # (normalizer first) leaves 270.0 sitting in normalized space.
    nan_fill = build_nan_fill(_cfg())
    normalizer = _StubNormalizer()
    bad = nan_fill(normalizer(_sample()))
    assert float(bad["varying_boundary"][1][0, 0]) == pytest.approx(270.0)


def test_scalar_routing_happens_after_normalization():
    pipeline = build_forcing_pipeline(_cfg(), normalizer=_StubNormalizer())
    out = pipeline.dataset_transform(_sample(co2=380.0))
    # CO2 channel has mean 0 / std 1 in the stub, so the routed scalar is the
    # normalized value — same frame as the rest of the boundary stream.
    assert out["calendar"].tolist() == pytest.approx([0.0, 12.0, 380.0])
    assert out["varying_boundary"].shape == (1, _H, _W)


# ---------------------------------------------------------------------------
# Placement variants
# ---------------------------------------------------------------------------


def test_normalize_at_use_placement_matches_dataset_placement():
    in_dataset = build_forcing_pipeline(_cfg(), normalizer=_StubNormalizer())
    at_use = build_forcing_pipeline(
        _cfg(), normalizer=_StubNormalizer(), normalize_in_dataset=False
    )
    a = in_dataset.dataset_transform(_sample())
    b = at_use.finalize(at_use.dataset_transform(_sample()))
    assert torch.allclose(a["varying_boundary"], b["varying_boundary"])
    assert torch.allclose(a["calendar"], b["calendar"])


def test_as_normalizer_proxy_routes_and_delegates():
    at_use = build_forcing_pipeline(
        _cfg(), normalizer=_StubNormalizer(), normalize_in_dataset=False
    )
    proxy = at_use.as_normalizer()
    out = proxy(at_use.dataset_transform(_sample()))
    assert out["calendar"].shape == (3,)
    # Non-call attributes still reach the wrapped normalizer.
    assert proxy.denormalize_state(x=1) == {"x": 1}
    assert proxy.to(torch.device("cpu")) is proxy


def test_as_normalizer_is_the_bare_normalizer_when_nothing_is_routed():
    cfg = _cfg()
    cfg.model.scalar_routed_boundary_variables = []
    norm = _StubNormalizer()
    p = build_forcing_pipeline(cfg, normalizer=norm, normalize_in_dataset=False)
    # Provably unchanged behavior for recipes that gain nothing.
    assert p.as_normalizer() is norm


def test_extra_transforms_run_first():
    calls = []

    def extra(sample):
        calls.append(bool(torch.isnan(sample["varying_boundary"]).any()))
        return sample

    p = build_forcing_pipeline(
        _cfg(), normalizer=_StubNormalizer(), extra_transforms=[extra]
    )
    p.dataset_transform(_sample())
    assert calls == [True]  # saw the raw NaN, i.e. ran before the fill


# ---------------------------------------------------------------------------
# Anti-fork guard
# ---------------------------------------------------------------------------


class _StubWrapper:
    def __init__(self, c_grid_dim, scalar_dim):
        self.c_grid_dim = c_grid_dim
        self.scalar_dim = scalar_dim


def test_assert_matches_accepts_the_matching_model():
    p = build_forcing_pipeline(_cfg())
    p.assert_matches(_StubWrapper(c_grid_dim=2, scalar_dim=3))


@pytest.mark.parametrize(
    "c_grid_dim,scalar_dim,expect",
    [(3, 3, "c_grid_dim"), (2, 2, "scalar_dim")],
)
def test_assert_matches_rejects_mismatch(c_grid_dim, scalar_dim, expect):
    p = build_forcing_pipeline(_cfg())
    with pytest.raises(ValueError, match=expect):
        p.assert_matches(_StubWrapper(c_grid_dim, scalar_dim))


def test_assert_matches_skips_models_without_the_attributes():
    # Deterministic Pangu / SFNO wrappers have no c_grid_dim / scalar_dim.
    build_forcing_pipeline(_cfg()).assert_matches(object())


def test_assembler_dims_match_upstream_erdm_co2_contract():
    cfg = _cfg()
    cfg.model.varying_boundary_variables = [
        "global_mean_co2",
        "DSWRFtoa_24h_lead",
        "sea_surface_temperature_monthly_interp",
        "sea_ice_cover_monthly_interp",
    ]
    cfg.model.constant_boundary_variables = [
        "geopotential_at_surface",
        "land_sea_mask",
    ]
    a = build_forcing_assembler(cfg)
    assert (a.c_grid_dim, a.scalar_dim) == (5, 3)


def test_smoothing_knobs_reach_the_nan_fill():
    fill = build_nan_fill(
        _cfg(
            smooth_nan_boundaries=True,
            smooth_sigma=2.0,
            smooth_kernel_size=7,
            smooth_n_iters=3,
        )
    )
    assert fill._smooth is True
    assert (fill._smooth_sigma, fill._smooth_kernel_size, fill._smooth_n_iters) == (
        2.0,
        7,
        3,
    )
