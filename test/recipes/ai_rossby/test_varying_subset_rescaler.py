# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The varying-boundary subset must be resolved in the STORE's stage.

``cfg.model.varying_boundary_variables`` is written in POST-rescaler order —
``amip_erdm_fancy`` and ``amip_rsi_fancy`` both list
``sea_surface_temperature_anomaly``, which no store holds; ``SSTRescaler``
derives it inside the assembler. But the subset slice and the normalizer both
run BEFORE the assembler, on raw store channels.

Comparing those two stages directly meant the subset test failed on the derived
name, no slice was applied, and ``global_mean_co2`` survived into the grid —
``c_grid_dim=6 but the data pipeline produces 7``. Verified on DeltaAI
2026-08-19: ``amip_erdm_fancy`` could not start a training run at all, and
neither could its RSI twin. Nothing caught it because the config health gates
never build a pipeline and the checkpoint translator reconstructs the channel
order arithmetically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from dataset_setup import (  # noqa: E402
    model_varying_pre_rescaler,
    resolve_varying_subset,
)

from physicsnemo.experimental.datapipes.climate.sst_forcing import (  # noqa: E402
    grid_forcing_names,
)

_STORE = [
    "global_mean_co2",
    "DSWRFtoa_24h_lead",
    "sea_surface_temperature_monthly_interp",
    "sea_ice_cover_monthly_interp",
]
_FANCY = [
    "DSWRFtoa_24h_lead",
    "sea_surface_temperature_monthly_interp",
    "sea_surface_temperature_anomaly",
    "sea_ice_cover_monthly_interp",
]


def _cfg(model_varying, mode="none"):
    return OmegaConf.create({
        "model": {"varying_boundary_variables": list(model_varying)},
        "dataset": {"sst_anomaly_channel": mode},
    })


def test_append_mode_drops_the_derived_channel():
    pre = model_varying_pre_rescaler(_cfg(_FANCY, "append"))
    assert "sea_surface_temperature_anomaly" not in pre
    assert pre == [n for n in _FANCY if n != "sea_surface_temperature_anomaly"]


def test_replace_mode_restores_the_absolute_sst_channel():
    from physicsnemo.experimental.datapipes.climate.sst_forcing import (
        SST_VARIABLE_NAMES,
    )

    replaced = [n if n != "sea_surface_temperature_monthly_interp"
                else "sea_surface_temperature_anomaly"
                for n in _STORE]
    pre = model_varying_pre_rescaler(_cfg(replaced, "replace"))
    assert "sea_surface_temperature_anomaly" not in pre
    assert any(n in SST_VARIABLE_NAMES for n in pre)


def test_fancy_contract_slices_co2_out():
    """The whole point: the SST-trend contract has no CO2 channel."""
    cfg = _cfg(_FANCY, "append")
    idx = resolve_varying_subset(cfg, _STORE)
    assert idx is not None, "no slice resolved — CO2 would survive into c_grid"
    sliced = [_STORE[i] for i in idx]
    assert "global_mean_co2" not in sliced


def test_slice_then_rescale_reproduces_the_model_list_exactly():
    """Round-trip: store -> slice -> rescaler must equal the model's own list.

    This is the invariant the widths depend on; c_grid_dim is derived from the
    model list, so any divergence here is a contract mismatch at startup.
    """
    cfg = _cfg(_FANCY, "append")
    idx = resolve_varying_subset(cfg, _STORE)
    sliced = [_STORE[i] for i in idx]
    assert grid_forcing_names(sliced, "append") == _FANCY


@pytest.mark.parametrize("mode", ["none", "append"])
def test_a_model_list_without_the_anomaly_is_untouched(mode):
    """Non-SST contracts (amip_rsi_v2, amip_erdm_v2) must not change behaviour."""
    v2 = list(_STORE)          # model list == store list
    assert model_varying_pre_rescaler(_cfg(v2, mode)) == v2
    assert resolve_varying_subset(_cfg(v2, mode), _STORE) is None


# ---------------------------------------------------------------------------
# The assembler carries the rescaler, so it must run even with nothing routed
# ---------------------------------------------------------------------------


def test_assembler_is_active_for_a_rescaler_with_nothing_routed():
    """``active`` gates whether the assembler joins the transform chain.

    It used to be ``bool(scalar_routed_variables)`` alone — but the assembler
    also carries the Phase-12g ``sst_rescaler`` hook, so a contract that routes
    nothing dropped the assembler AND the rescaler. That is exactly the fancy
    pair (``scalar_routed_boundary_variables: []``, the SST trend scalar taking
    CO2's slot): the anomaly channel was never derived and the trend scalar
    never appended, while ``c_grid_dim`` *does* count the rescaler's channel —
    so the run died at ``assert_matches`` with 6-vs-7.
    """
    from physicsnemo.experimental.datapipes.climate.forcing import ForcingAssembler

    names = ["DSWRFtoa_24h_lead", "sea_surface_temperature_monthly_interp"]
    bare = ForcingAssembler(varying_boundary_variables=names)
    assert not bare.active, "a do-nothing assembler should stay a passthrough"

    with_rescaler = ForcingAssembler(
        varying_boundary_variables=names, sst_rescaler=lambda s: s
    )
    assert with_rescaler.active, "a rescaler-only assembler must join the chain"


def test_rescaler_only_assembler_does_not_touch_the_calendar():
    """With nothing routed it must pass the sample through, not stack an empty list.

    The early return is gated on the routed list, not on ``active`` — which is
    now True for a rescaler-only assembler, and falling through reached
    ``torch.stack([], dim=-1)``.
    """
    import torch

    from physicsnemo.experimental.datapipes.climate.forcing import ForcingAssembler

    seen = {}

    def _rescaler(sample):
        seen["ran"] = True
        return sample

    asm = ForcingAssembler(
        varying_boundary_variables=["a", "b"], sst_rescaler=_rescaler
    )
    sample = {
        "varying_boundary": torch.zeros(2, 4, 8),
        "calendar": torch.tensor([0.0, 12.0]),
    }
    out = asm(sample)
    assert seen.get("ran"), "the rescaler hook never fired"
    assert out["calendar"].shape == (2,), "calendar must be untouched"
    assert out["varying_boundary"].shape == (2, 4, 8)
