# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The obs climatology must fail loudly, never plausibly.

A misaligned climatology produces a bias field that looks entirely normal and
is wrong by up to the pole-to-pole temperature range. The physical anchors are
checked on every load because metadata can be mislabelled but the Antarctic
plateau cannot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_AI_ROSSBY_DIR = Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from obs_climatology import (  # noqa: E402
    ObsClimatologyError,
    as_constant_truth,
    assert_physical_anchors,
    load_obs_climatology,
    normalize_like,
)

H, W, L = 180, 360, 26


def _synthetic(*, flip_lat=False, reverse_levels=False):
    """A physically plausible stand-in: cold poles, warm equator, geopotential
    decreasing with ascending pressure."""
    lat = torch.linspace(-89.5, 89.5, H)                     # S->N
    # cold at both poles, warmest at the equator, north pole warmer than south
    t2m = 300.0 - 70.0 * (lat.abs() / 90.0) ** 1.5 + 16.0 * (lat / 90.0)
    surface = torch.zeros(6, H, W)
    surface[2] = t2m[:, None].expand(H, W)
    surface[0] = surface[2]
    surface[1] = 101325.0
    ua = torch.zeros(5, L, H, W)
    z = torch.linspace(3.4e5, 700.0, L)                      # ascending pressure
    ua[3] = z[:, None, None].expand(L, H, W)
    tprof = torch.linspace(240.0, 296.0, L)
    ua[0] = tprof[:, None, None].expand(L, H, W)
    if flip_lat:
        surface = surface.flip(-2)
        ua = ua.flip(-2)
    if reverse_levels:
        ua = ua.flip(1)
    return {"surface": surface, "upper_air": ua}


def _write(tmp_path, obs, *, levels=None, names=True):
    torch.save(obs["surface"], tmp_path / "climatology_surface_obs.pt")
    torch.save(obs["upper_air"], tmp_path / "climatology_multilevel_obs.pt")
    meta = {"start_date": "1996-01-01", "end_date": "2001-01-01",
            "n_frames": 1827, "complete": True,
            "horizontal_resolution": [H, W],
            "levels": list(levels) if levels else list(range(1, L + 1))}
    if names:
        meta["surface_variables"] = ["skin_temperature", "surface_pressure",
                                     "2m_temperature", "2m_specific_humidity",
                                     "10m_u_component_of_wind",
                                     "10m_v_component_of_wind"]
        meta["upper_air_variables"] = ["temperature", "u_component_of_wind",
                                       "v_component_of_wind", "geopotential",
                                       "specific_humidity"]
    (tmp_path / "climatology_obs_meta.json").write_text(json.dumps(meta))
    return tmp_path


# ---------------------------------------------------------------------------
# Physical anchors — the checks that matter
# ---------------------------------------------------------------------------
def test_anchors_pass_on_a_correctly_oriented_file():
    a = assert_physical_anchors(_synthetic())
    assert a["t2m_row0"] < a["t2m_rowN"], "row 0 must be the south pole"
    assert a["z_level0"] > a["z_levelN"]


def test_flipped_latitude_is_refused():
    """The failure this module exists for: shapes match, bias is doubled."""
    with pytest.raises(ObsClimatologyError, match="row order looks N->S"):
        assert_physical_anchors(_synthetic(flip_lat=True))


def test_reversed_level_axis_is_refused():
    with pytest.raises(ObsClimatologyError, match="level axis looks reversed"):
        assert_physical_anchors(_synthetic(reverse_levels=True))


def test_nonphysical_units_are_refused():
    obs = _synthetic()
    obs["surface"][2] -= 273.15          # Kelvin -> Celsius
    with pytest.raises(ObsClimatologyError, match="physical range"):
        assert_physical_anchors(obs)


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------
class _Catalog:
    surface = ["skin_temperature", "surface_pressure", "2m_temperature",
               "2m_specific_humidity", "10m_u_component_of_wind",
               "10m_v_component_of_wind"]
    upper_air = ["temperature", "u_component_of_wind", "v_component_of_wind",
                 "geopotential", "specific_humidity"]
    diagnostic: list[str] = []


def test_loads_and_validates(tmp_path):
    _write(tmp_path, _synthetic(), levels=list(range(1, L + 1)))
    obs, meta, anchors = load_obs_climatology(
        tmp_path, catalog=_Catalog(), grid=(H, W),
        levels=list(range(1, L + 1)),
    )
    assert tuple(obs["surface"].shape) == (6, H, W)
    assert tuple(obs["upper_air"].shape) == (5, L, H, W)
    assert meta["n_frames"] == 1827
    assert anchors["t2m_row0"] < anchors["t2m_rowN"]


def test_grid_mismatch_raises(tmp_path):
    _write(tmp_path, _synthetic())
    with pytest.raises(ObsClimatologyError, match="grid"):
        load_obs_climatology(tmp_path, grid=(45, 90))


def test_level_value_mismatch_raises(tmp_path):
    _write(tmp_path, _synthetic(), levels=list(range(1, L + 1)))
    with pytest.raises(ObsClimatologyError, match="level VALUES differ"):
        load_obs_climatology(tmp_path, levels=[x + 1 for x in range(1, L + 1)])


def test_channel_order_mismatch_raises(tmp_path):
    _write(tmp_path, _synthetic())

    class _Shuffled(_Catalog):
        surface = list(reversed(_Catalog.surface))

    with pytest.raises(ObsClimatologyError, match="variable ORDER differs"):
        load_obs_climatology(tmp_path, catalog=_Shuffled(), grid=(H, W))


def test_missing_required_file_raises(tmp_path):
    obs = _synthetic()
    torch.save(obs["surface"], tmp_path / "climatology_surface_obs.pt")
    with pytest.raises(ObsClimatologyError, match="missing"):
        load_obs_climatology(tmp_path)


# ---------------------------------------------------------------------------
# Normalization round trip — ACC/bias space correctness
# ---------------------------------------------------------------------------
class _Norm:
    def __init__(self):
        self.surface_mean = torch.full((6,), 280.0)
        self.surface_std = torch.full((6,), 15.0)
        self.upper_air_mean = torch.full((5, L), 250.0)
        self.upper_air_std = torch.full((5, L), 30.0)
        self.diagnostic_mean = None
        self.diagnostic_std = None


def test_normalize_round_trips_to_fp32_rounding():
    obs = _synthetic()
    n = _Norm()
    z = normalize_like(n, "surface", obs["surface"])
    back = z * n.surface_std[:, None, None] + n.surface_mean[:, None, None]
    assert torch.allclose(back, obs["surface"], rtol=1e-5, atol=1e-3)


def test_diagnostic_passthrough_when_never_z_scored():
    """If diagnostics were not normalized, the constant truth must not be
    either -- otherwise truth and prediction live in different spaces."""
    x = torch.randn(3, H, W)
    assert torch.equal(normalize_like(_Norm(), "diagnostic", x), x)


def test_as_constant_truth_expands_without_copying():
    obs = _synthetic()
    ct = as_constant_truth(obs, normalizer=_Norm(), batch_size=4)
    assert ct["surface_in"].shape == (4, 6, H, W)
    assert ct["upper_air_in"].shape == (4, 5, L, H, W)
    # expand, not repeat: a zero stride on the batch dim
    assert ct["surface_in"].stride()[0] == 0


# ---------------------------------------------------------------------------
# The real staged artifact, when present
# ---------------------------------------------------------------------------
_REAL = Path("/tmp/claude-1000/obs_clim")


@pytest.mark.skipif(not (_REAL / "climatology_surface_obs.pt").exists(),
                    reason="staged obs climatology not present")
def test_real_amip_v2_file_passes_every_anchor():
    obs, meta, anchors = load_obs_climatology(
        _REAL, grid=(180, 360),
        levels=[5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300,
                400, 500, 600, 700, 800, 850, 875, 900, 925, 950, 975, 1000],
    )
    assert meta["n_frames"] == 1827 and meta["complete"] is True
    # measured values, pinned so a re-staged/regenerated file is noticed
    assert anchors["t2m_row0"] == pytest.approx(226.8, abs=0.5)
    assert anchors["t2m_rowN"] == pytest.approx(258.9, abs=0.5)
    assert tuple(obs["upper_air"].shape) == (5, 26, 180, 360)
