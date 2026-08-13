# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12g — the SST suite through the recipe's forcing pipeline.

The library tests cover the anomaly math; these cover the wiring the recipes
depend on: which trend scalar owns the calendar row's third slot (CO2 and
``global_mean_sst`` both want it, so they are mutually exclusive), how the
appended anomaly channel changes ``c_grid_dim``, and that the whole suite is
inert for every config that does not ask for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    build_sst_rescaler,
    resolve_scalar_forcing,
)

from physicsnemo.experimental.datapipes.climate import (  # noqa: E402
    SST_ANOMALY_CHANNEL_NAME,
)

_H, _W = 8, 16
_SST = "sea_surface_temperature_monthly_interp"
_STORED = ["global_mean_co2", "DSWRFtoa_24h_lead", _SST, "sea_ice_cover_monthly_interp"]
_CONST = ["geopotential_at_surface", "land_sea_mask"]


def _artifact(tmp_path, *, anom_std=0.6, gm_std=0.12):
    n_coef = 5
    coeffs = np.zeros((n_coef, _H, _W), dtype=np.float32)
    coeffs[0] = 300.0
    ocean = np.zeros((_H, _W), dtype=bool)
    ocean[:, 4:] = True
    weight = ocean.astype(np.float32)
    weight /= weight.sum()
    path = tmp_path / "sst_climatology.npz"
    np.savez(
        path,
        harmonic_coeffs=coeffs,
        n_harmonics=np.int32(2),
        anom_std=np.float32(anom_std),
        gm_mean=np.float32(0.0),
        gm_std=np.float32(gm_std),
        ocean_weight=weight,
        fit_year_start=np.int32(1979),
        fit_year_end=np.int32(2015),
    )
    return path


class _Normalizer:
    """Just the pieces ``build_sst_rescaler`` reads."""

    def __init__(self, names):
        self.varying_boundary_variables = list(names)
        n = len(names)
        self.varying_mean = torch.full((n, 1, 1), 290.0)
        self.varying_std = torch.full((n, 1, 1), 12.3)

    def __call__(self, batch):
        return batch


def _cfg(tmp_path, *, model_varying=None, routed=("global_mean_co2",),
         anomaly="none", scalar="auto", path=None, **data_extra):
    data = {
        "sst_anomaly_channel": anomaly,
        "scalar_forcing": scalar,
        "sst_anomaly_scale": "anom_std",
        "sst_scalar_scale": "gm_std",
        "sst_climatology_path": str(path) if path else None,
        "nan_fill_default": 0.0,
        "nan_fill_values": {},
    }
    data.update(data_extra)
    return OmegaConf.create(
        {
            "model": {
                "varying_boundary_variables": list(
                    model_varying if model_varying is not None else _STORED
                ),
                "constant_boundary_variables": list(_CONST),
                "scalar_routed_boundary_variables": list(routed),
                "surface_variables": [],
                "diagnostic_variables": [],
            },
            "dataset": data,
        }
    )


# ---------------------------------------------------------------------------
# Which scalar owns the third calendar slot
# ---------------------------------------------------------------------------


def test_auto_picks_co2_when_it_is_routed(tmp_path):
    assert resolve_scalar_forcing(_cfg(tmp_path)) == "co2"


def test_auto_picks_nothing_without_co2(tmp_path):
    assert resolve_scalar_forcing(_cfg(tmp_path, routed=())) is None


def test_auto_matches_historical_behavior(tmp_path):
    """``auto`` must not change any shipped config's scalar_dim.

    Every AMIP config routes CO2 today, so ``auto`` has to keep resolving to
    ``co2`` — the 12g default cannot silently re-slot the calendar row.
    """
    cfg = _cfg(tmp_path)
    assert resolve_scalar_forcing(cfg) == "co2"
    assert build_forcing_assembler(cfg).scalar_dim == 3


def test_none_drops_the_scalar_even_with_co2_routed(tmp_path):
    assert resolve_scalar_forcing(_cfg(tmp_path, scalar="none")) is None


def test_co2_without_a_routed_channel_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needs 'global_mean_co2'"):
        resolve_scalar_forcing(_cfg(tmp_path, scalar="co2", routed=()))


def test_global_mean_sst_and_co2_are_mutually_exclusive(tmp_path):
    # Both occupy the calendar row's third slot.
    with pytest.raises(ValueError, match="conflicts with the routed"):
        resolve_scalar_forcing(_cfg(tmp_path, scalar="global_mean_sst"))


def test_global_mean_sst_needs_an_sst_channel(tmp_path):
    cfg = _cfg(
        tmp_path,
        scalar="global_mean_sst",
        routed=(),
        model_varying=["DSWRFtoa_24h_lead"],
    )
    with pytest.raises(ValueError, match="needs an SST channel"):
        resolve_scalar_forcing(cfg)


def test_an_unknown_scalar_forcing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="scalar_forcing must be one of"):
        resolve_scalar_forcing(_cfg(tmp_path, scalar="global_mean_pressure"))


# ---------------------------------------------------------------------------
# The rescaler as the assembler's hook
# ---------------------------------------------------------------------------


def test_no_rescaler_unless_the_config_asks(tmp_path):
    cfg = _cfg(tmp_path, path=_artifact(tmp_path))
    assert build_sst_rescaler(cfg, normalizer=_Normalizer(_STORED)) is None


def test_a_missing_path_is_refused_when_the_feature_is_on(tmp_path):
    cfg = _cfg(tmp_path, anomaly="append", path=None)
    with pytest.raises(ValueError, match="sst_climatology_path"):
        build_sst_rescaler(cfg, normalizer=_Normalizer(_STORED))


def test_the_rescaler_needs_the_normalizer(tmp_path):
    cfg = _cfg(tmp_path, anomaly="append", path=_artifact(tmp_path))
    with pytest.raises(ValueError, match="normalizer"):
        build_sst_rescaler(cfg, normalizer=None)


def test_stats_are_read_by_name_not_position(tmp_path):
    """A normalizer whose channel order differs from the model config's.

    The rescaler inverts *this channel's* z-score, so it has to find the stats by
    name; taking them positionally would decode SST with sea-ice statistics.
    """
    reordered = [_SST] + [n for n in _STORED if n != _SST]
    norm = _Normalizer(reordered)
    norm.varying_mean = torch.tensor([290.0, 1.0, 2.0, 3.0]).reshape(-1, 1, 1)
    norm.varying_std = torch.tensor([12.3, 1.0, 1.0, 1.0]).reshape(-1, 1, 1)
    cfg = _cfg(tmp_path, anomaly="append", path=_artifact(tmp_path))
    r = build_sst_rescaler(cfg, normalizer=norm)
    assert r.sst_index == 0
    assert r.sst_mean == pytest.approx(290.0) and r.sst_std == pytest.approx(12.3)


# ---------------------------------------------------------------------------
# Channel / scalar widths through the assembler
# ---------------------------------------------------------------------------


def _assembler_with(tmp_path, **kw):
    cfg = _cfg(tmp_path, path=_artifact(tmp_path), **kw)
    rescaler = build_sst_rescaler(cfg, normalizer=_Normalizer(_STORED))
    return cfg, build_forcing_assembler(cfg, sst_rescaler=rescaler)


def test_append_widens_c_grid_by_one(tmp_path):
    _, plain = _assembler_with(tmp_path)
    _, appended = _assembler_with(tmp_path, anomaly="append")
    assert appended.c_grid_dim == plain.c_grid_dim + 1


def test_replace_keeps_c_grid_width(tmp_path):
    _, plain = _assembler_with(tmp_path)
    _, replaced = _assembler_with(tmp_path, anomaly="replace")
    assert replaced.c_grid_dim == plain.c_grid_dim


def test_the_sst_scalar_takes_co2s_slot_not_a_fourth(tmp_path):
    _, co2 = _assembler_with(tmp_path)                       # routed CO2
    cfg, sst = _assembler_with(
        tmp_path, anomaly="append", scalar="global_mean_sst", routed=()
    )
    assert co2.scalar_dim == 3
    assert sst.scalar_dim == 3          # 2 calendar + the SST scalar, not 4
    assert resolve_scalar_forcing(cfg) == "global_mean_sst"


# ---------------------------------------------------------------------------
# End to end through the assembler's __call__
# ---------------------------------------------------------------------------


def _sample(sst_kelvin, *, mean=290.0, std=12.3, calendar=(0.0, 0.0)):
    vb = torch.zeros(len(_STORED), _H, _W)
    vb[0] = (400.0 - mean) / std                     # a uniform CO2 map
    vb[_STORED.index(_SST)] = (sst_kelvin - mean) / std
    return {
        "varying_boundary": vb,
        "constant_boundary": torch.zeros(len(_CONST), _H, _W),
        "calendar": torch.tensor(calendar, dtype=torch.float32),
    }


def test_append_then_co2_pop_leaves_the_documented_stream(tmp_path):
    """Ordering: the anomaly is inserted while SST is still gridded, then CO2 pops.

    Upstream's step 2 before step 3 — so the anomaly lands next to the absolute
    channel and the CO2 pop still finds CO2 at index 0.
    """
    cfg, asm = _assembler_with(tmp_path, anomaly="append")
    out = asm(_sample(torch.full((_H, _W), 300.5)))
    # 4 stored - 1 popped CO2 + 1 anomaly = 4 varying channels.
    assert out["varying_boundary"].shape[0] == 4
    assert out["calendar"].shape[-1] == 3            # CO2 in the third slot
    names = asm.sst_rescaler.grid_forcing_names
    assert names[names.index(_SST) + 1] == SST_ANOMALY_CHANNEL_NAME


def test_appending_does_not_drop_the_channel_after_sst(tmp_path):
    """The failure this ordering caused, pinned by name.

    ``append`` inserts the anomaly right after SST, so every stored channel
    beyond SST shifts by one. With the pop indices taken against the *stored*
    order, ``_keep_idx`` dropped the last channel — sea ice in the shipped AMIP
    list — and nothing complained, because 4 stored - 1 popped + 1 derived is
    still 4 channels wide.
    """
    cfg, asm = _assembler_with(tmp_path, anomaly="append")
    out = asm(_sample(torch.full((_H, _W), 300.5)))
    assert asm.varying_boundary_variables_out == [
        "DSWRFtoa_24h_lead",
        _SST,
        SST_ANOMALY_CHANNEL_NAME,
        "sea_ice_cover_monthly_interp",
    ]
    assert out["varying_boundary"].shape[0] == 4
    assert asm.c_grid_dim == len(_CONST) + 4


def test_the_sst_scalar_reaches_the_calendar_row(tmp_path):
    cfg, asm = _assembler_with(
        tmp_path, anomaly="none", scalar="global_mean_sst", routed=()
    )
    clim = asm.sst_rescaler.forcing.climatology_at([0.0, 0.0])
    out = asm(_sample(clim + 0.12))                   # +gm_std over the ocean
    assert out["calendar"].shape[-1] == 3
    assert float(out["calendar"][-1]) == pytest.approx(1.0, rel=1e-2)
    # No channel was added or removed: nothing is routed and the mode is none.
    assert out["varying_boundary"].shape[0] == len(_STORED)


def test_the_suite_is_inert_by_default(tmp_path):
    """Every shipped config ships `sst_anomaly_channel: none`, so nothing moves."""
    cfg, asm = _assembler_with(tmp_path)
    sample = _sample(torch.full((_H, _W), 300.0))
    before = sample["varying_boundary"].clone()
    out = asm(sample)
    assert asm.sst_rescaler is None
    # Only the CO2 pop happened.
    assert out["varying_boundary"].shape[0] == len(_STORED) - 1
    assert torch.equal(out["varying_boundary"], before[1:])


def test_build_forcing_pipeline_attaches_the_hook(tmp_path):
    cfg = _cfg(tmp_path, anomaly="append", path=_artifact(tmp_path))
    pipe = build_forcing_pipeline(cfg, normalizer=_Normalizer(_STORED))
    assert pipe.assembler.sst_rescaler is not None
    assert pipe.assembler.c_grid_dim == len(_CONST) + len(_STORED)  # -1 CO2 +1 anom


@pytest.mark.parametrize(
    "name", ["amip_1981", "amip_dailyavg", "amip_dailyavg_coarse"]
)
def test_shipped_amip_configs_carry_inert_defaults(name):
    cfg = OmegaConf.load(_AI_ROSSBY_DIR / "conf" / "dataset" / f"{name}.yaml")
    assert cfg.sst_anomaly_channel == "none"
    assert cfg.scalar_forcing == "auto"
    assert cfg.sst_climatology_path is None
