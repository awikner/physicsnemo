# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12h — deriving wrapper kwargs from amip_v2 checkpoint hparams.

Built against the two real v2-trained checkpoints (``project: amip_v2``):

    ERDM_fancy_42_2026-08-10T13-21-13   in/out 154, c_grid_dim 6, scalar_dim 3,
                                        nocean 3 (SST + anomaly + ice)
    x_DDC_42_2026-08-07T09-34-49        decoder_type dit, in 302 / out 151

Their ``config.yml`` blocks are inlined below rather than read from disk, so the
suite runs anywhere. Each states its channel widths by hand while this fork
*derives* them, which is what makes these assertions worth anything.

The bug this suite would have caught: the backbone kwargs live under the
**backbone-named** key (``model.ERDM.DiT``), not ``model.ERDM.model``. Reading
the latter yielded ``{}`` — so the wrapper was built with class-default geometry,
``scalar_dim`` fell back to 2, and the ``c_grid_dim`` reconciliation was skipped
for want of a target. It survived because the live sweeps pass
``--model-config``, which bypasses this code path entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[3] / "tools" / "checkpoint_translation"
sys.path.insert(0, str(_TOOLS))

from amip_si import wrapper_kwargs_from_hparams  # noqa: E402

_SURFACE = [
    "skin_temperature", "surface_pressure", "2m_temperature",
    "2m_specific_humidity", "10m_u_component_of_wind", "10m_v_component_of_wind",
]
_UPPER = [
    "temperature", "u_component_of_wind", "v_component_of_wind",
    "geopotential", "specific_humidity",
]
_DIAG = [
    "USWRFtoa_24h", "ULWRFtoa_24h", "USWRFsfc_24h", "ULWRFsfc_24h",
    "DSWRFsfc_24h", "DLWRFsfc_24h", "LHTFLsfc_24h", "SHTFLsfc_24h",
    "PRATEsfc_24h", "hcc_24h", "lcc_24h", "mcc_24h", "mn2t_24h", "mx2t_24h",
    "mxtpr_24h",
]
_LEVELS = [
    5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 400, 500,
    600, 700, 800, 850, 875, 900, 925, 950, 975, 1000,
]
_SST = "sea_surface_temperature_monthly_interp"
_ANOM = "sea_surface_temperature_anomaly"


def _fancy_blob():
    """``ERDM_fancy``'s hparams, as the Lightning ckpt carries them."""
    return {
        "hyper_parameters": {
            "config": {
                "model": {
                    "model_name": "ERDM",
                    "backbone": "DiT",
                    "ERDM": {
                        "DiT": {
                            "in_channels": 154,
                            "out_channels": 154,
                            "c_grid_dim": 6,
                            "scalar_dim": 3,
                            "dim": 1024,
                            "num_heads": 16,
                            "temporal_num_heads": 8,
                            "num_blocks": 20,
                            "window_size": 6,
                            "c_grid_downsample": 4,
                            "c_grid_cross_layers": 4,
                            "c_grid_cross_heads": 8,
                            "global_cond": True,
                            "input_embed": {
                                "mode": "budget", "d_boundary": 256,
                                "d_calendar": 128, "d_co2": 48,
                                "state_encoder": "column", "d_level": 16,
                                "boundary_encoder": "conv2",
                                "boundary_pool_stats": True,
                                "boundary_static_bias": True,
                                "co2_linear": True, "source_norm": True,
                            },
                            "output_head": {
                                "mode": "mix", "num_experts": 2,
                                "decoder": "flat", "d_level": 16,
                            },
                        },
                        "scheduler": {"window_size": 6, "sigma_data": 1.0},
                    },
                },
                "data": {
                    "surface_variables": _SURFACE,
                    "upper_air_variables": _UPPER,
                    "diagnostic_variables": _DIAG,
                    "diagnostic_input": True,
                    "constant_boundary_variables": [
                        "geopotential_at_surface", "land_sea_mask",
                    ],
                    # STORED order: no anomaly (derived), no CO2 at all.
                    "varying_boundary_variables": [
                        "DSWRFtoa_24h_lead", _SST, "sea_ice_cover_monthly_interp",
                    ],
                    "ocean_state_variables": [
                        _SST, _ANOM, "sea_ice_cover_monthly_interp",
                    ],
                    "sst_anomaly_channel": "append",
                    "scalar_forcing": "global_mean_sst",
                    "levels": _LEVELS,
                    "horizontal_resolution": [180, 360],
                    # The state is pre-coarsened by this; the forecaster runs on
                    # 180/4 x 360/4 while c_grid still arrives full-res.
                    "downsample_factor": 4,
                    "multistep_rollout": 6,
                },
            }
        }
    }


def _xddc_blob():
    """``x_DDC``'s hparams (decoder_type: dit)."""
    return {
        "hyper_parameters": {
            "config": {
                "model": {
                    "model_name": "x_DDC",
                    "x_DDC": {
                        "decoder_type": "dit",
                        "dit": {
                            "in_channels": 302, "out_channels": 151,
                            "dim": 1024, "num_blocks": 20, "num_heads": 16,
                            "patch_size": 4, "nlat": 180, "nlon": 360,
                            "unpatch": "vanilla", "dropout": 0.0,
                        },
                        "encoder": {"downsample_factor": 4},
                        "scheduler": {"num_steps": 5},
                    },
                },
                "data": {
                    "surface_variables": _SURFACE,
                    "upper_air_variables": _UPPER,
                    "diagnostic_variables": _DIAG,
                    "levels": _LEVELS,
                    "horizontal_resolution": [180, 360],
                    "varying_boundary_variables": [],
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# The forecaster
# ---------------------------------------------------------------------------


def test_the_backbone_block_is_found_under_its_backbone_name():
    """``model.ERDM.DiT``, not ``model.ERDM.model``.

    An empty backbone dict is the silent failure: the wrapper would build at
    class-default geometry and only a shape mismatch at load time — if any —
    would reveal it.
    """
    kw = wrapper_kwargs_from_hparams(
        _fancy_blob(), "RollingDiTWrapper", source_contract="v2"
    )
    rdk = kw["rolling_dit_kwargs"]
    assert rdk, "backbone kwargs came back empty"
    assert rdk["dim"] == 1024
    assert rdk["num_blocks"] == 20
    assert rdk["window_size"] == 6
    assert rdk["global_cond"] is True
    assert rdk["c_grid_cross_layers"] == 4
    # The whole 12e feature set has to survive, nested dicts included.
    assert rdk["input_embed"]["mode"] == "budget"
    assert rdk["input_embed"]["state_encoder"] == "column"
    assert rdk["output_head"]["mode"] == "mix"
    # ...and the sibling scheduler block must NOT leak in as backbone kwargs.
    assert "scheduler" not in rdk
    assert "sigma_data" not in rdk


def test_scalar_dim_comes_from_the_checkpoint_not_a_default():
    kw = wrapper_kwargs_from_hparams(
        _fancy_blob(), "RollingDiTWrapper", source_contract="v2"
    )
    assert kw["scalar_dim"] == 3       # (sod, doy, global-mean SST anomaly)


def test_the_derived_sst_anomaly_channel_is_spliced_in():
    """``sst_anomaly_channel: append`` makes c_grid_dim 6 add up.

    The store lists three varying channels; the model sees four because the
    rescaler derives one. Without the splice the reconciliation reports
    "requires 4 entries but only 3 are listed".
    """
    kw = wrapper_kwargs_from_hparams(
        _fancy_blob(), "RollingDiTWrapper", source_contract="v2"
    )
    assert kw["varying_boundary_variables"] == [
        "DSWRFtoa_24h_lead", _SST, _ANOM, "sea_ice_cover_monthly_interp",
    ]


def test_ocean_state_variables_reach_the_wrapper():
    kw = wrapper_kwargs_from_hparams(
        _fancy_blob(), "RollingDiTWrapper", source_contract="v2"
    )
    assert kw["ocean_state_variables"] == [
        _SST, _ANOM, "sea_ice_cover_monthly_interp",
    ]


def test_ocean_channels_on_a_family_that_cannot_carry_them_is_refused():
    with pytest.raises(NotImplementedError, match="no ocean-channel support"):
        wrapper_kwargs_from_hparams(
            _fancy_blob(), "ERDMWrapper", source_contract="v1"
        )


def test_the_derived_kwargs_rebuild_the_checkpoints_stated_contract():
    """End to end: hparams -> wrapper, four widths vs their config.yml."""
    from physicsnemo.experimental.models.amip_si import RollingDiTWrapper

    kw = wrapper_kwargs_from_hparams(
        _fancy_blob(), "RollingDiTWrapper", source_contract="v2"
    )
    rdk = kw["rolling_dit_kwargs"]
    rdk.update(
        dim=64, num_heads=2, num_blocks=1, temporal_num_heads=2,
        c_grid_cross_layers=1, c_grid_cross_heads=2,
    )
    rdk["input_embed"].update(d_boundary=16, d_calendar=16, d_co2=8)
    w = RollingDiTWrapper(**kw)

    assert w.in_channels == 154
    assert w.c_grid_dim == 6
    assert w.scalar_dim == 3
    assert w.num_ocean == 3
    assert w.ocean_grid_indices == [1, 2, 3]


def test_the_backbone_grid_is_the_coarse_state_grid():
    """``horizontal_resolution`` must be the grid the FORECASTER runs on.

    Found by translating the real checkpoint: it failed with

        input_embed.boundary_embed.static_bias:
          ckpt [256, 45, 90] vs model [256, 180, 360]

    Upstream never writes nlat/nlon in an ERDM config — ``RollingDiT`` defaults
    them to 45x90 — while the *data* resolution is 180x360 and the state is
    pre-coarsened by ``downsample_factor``. Handing over the data resolution
    built a learned geographic bias 16x too large. Only ``static_bias`` is
    grid-shaped, so with ``boundary_static_bias: false`` this would have loaded
    cleanly and simply been wrong.
    """
    kw = wrapper_kwargs_from_hparams(
        _fancy_blob(), "RollingDiTWrapper", source_contract="v2"
    )
    assert kw["horizontal_resolution"] == [45, 90]


def test_an_explicit_backbone_grid_wins_over_the_division():
    blob = _fancy_blob()
    dit = blob["hyper_parameters"]["config"]["model"]["ERDM"]["DiT"]
    dit["nlat"], dit["nlon"] = 60, 120
    kw = wrapper_kwargs_from_hparams(
        blob, "RollingDiTWrapper", source_contract="v2"
    )
    assert kw["horizontal_resolution"] == [60, 120]


def test_erdm_on_the_dit_backbone_resolves_to_the_rolling_wrapper():
    """``model_name: ERDM`` is ambiguous across the rebaseline.

    v1 could mean the ADM-style UNet; amip_v2 deleted that and runs ERDM on
    RollingDiT. ``model.backbone`` says which, so the family cross-check must
    read it instead of warning about a mismatch the user cannot fix.
    """
    from amip_si import resolve_target_for_source

    assert resolve_target_for_source("ERDM", "DiT") == "RollingDiTWrapper"
    assert resolve_target_for_source("ERDM", "UNet") == "ERDMWrapper"
    assert resolve_target_for_source("ERDM", None) == "ERDMWrapper"
    assert resolve_target_for_source("RFM", "DiT") == "RollingDiTWrapper"


def test_trimming_never_takes_the_derived_anomaly_channel():
    """The trim heuristic must prefer any stored channel over the derived one.

    It exists for stored channels that were scalar-routed (``global_mean_co2``).
    The anomaly cannot be one of those — the translator derived it from this
    config's own ``sst_anomaly_channel`` — so dropping it would contradict the
    source. Here a too-small c_grid_dim forces one drop; sea ice goes, the
    anomaly stays.
    """
    blob = _fancy_blob()
    blob["hyper_parameters"]["config"]["model"]["ERDM"]["DiT"]["c_grid_dim"] = 5
    kw = wrapper_kwargs_from_hparams(
        blob, "RollingDiTWrapper", source_contract="v2"
    )
    assert _ANOM in kw["varying_boundary_variables"]
    assert "sea_ice_cover_monthly_interp" not in kw["varying_boundary_variables"]


def test_a_config_that_only_reconciles_by_dropping_the_anomaly_is_refused():
    """When no stored channel is left to drop, refuse rather than guess."""
    blob = _fancy_blob()
    # 2 constant + 4 varying (3 stored + 1 derived); c_grid_dim 2 leaves room for
    # zero varying channels, so reconciling would have to discard the derived one.
    blob["hyper_parameters"]["config"]["model"]["ERDM"]["DiT"]["c_grid_dim"] = 2
    with pytest.raises(ValueError, match="derived"):
        wrapper_kwargs_from_hparams(
            blob, "RollingDiTWrapper", source_contract="v2"
        )


# ---------------------------------------------------------------------------
# The downscaler
# ---------------------------------------------------------------------------


def test_the_dit_decoder_is_selected_and_its_kwargs_read():
    kw = wrapper_kwargs_from_hparams(
        _xddc_blob(), "XDDCWrapper", source_contract="v2"
    )
    assert kw["decoder_type"] == "dit"
    assert "unet_kwargs" not in kw
    dk = kw["dit_kwargs"]
    assert dk["dim"] == 1024 and dk["num_blocks"] == 20
    assert dk["patch_size"] == 4 and dk["unpatch"] == "vanilla"
    # Auto-derived by the wrapper, so they must not be carried over — a stale
    # copy here could disagree with horizontal_resolution.
    for k in ("in_channels", "out_channels", "nlat", "nlon"):
        assert k not in dk
    assert kw["downsample_factor"] == 4


def test_the_derived_xddc_kwargs_rebuild_their_stated_widths():
    from physicsnemo.experimental.models.amip_si import DiTAE, XDDCWrapper

    kw = wrapper_kwargs_from_hparams(
        _xddc_blob(), "XDDCWrapper", source_contract="v2"
    )
    kw["dit_kwargs"].update(dim=32, num_heads=2, num_blocks=1)
    w = XDDCWrapper(**kw)
    assert isinstance(w.backbone, DiTAE)
    assert w.backbone.in_channels == 302
    assert w.backbone.out_channels == 151
    assert (w.backbone.nlat, w.backbone.nlon) == (180, 360)


def test_build_target_wrapper_picks_the_dit_kwargs_key():
    """The builder's backbone-kwargs key follows ``decoder_type``, not the class.

    x_DDC is the only wrapper with two possible backbones, and the key table is
    per-wrapper — so the first real dit checkpoint died with
    ``KeyError('unet_kwargs')`` inside the builder, after the mapper had done
    everything right.
    """
    from amip_si import build_target_wrapper
    from physicsnemo.experimental.models.amip_si import DiTAE

    blob = _xddc_blob()
    blob["hyper_parameters"]["config"]["model"]["x_DDC"]["dit"].update(
        dim=32, num_heads=2, num_blocks=1
    )
    w = build_target_wrapper(
        blob=blob, target_class="XDDCWrapper", source_contract="v2"
    )
    assert isinstance(w.backbone, DiTAE)
    assert w.backbone.in_channels == 302 and w.backbone.out_channels == 151


def test_the_v1_unet_decoder_still_maps():
    blob = _xddc_blob()
    xddc = blob["hyper_parameters"]["config"]["model"]["x_DDC"]
    xddc["decoder_type"] = "unet"
    xddc.pop("dit")
    xddc["decoder"] = {"model_channels": 384, "num_res_blocks": 3}
    kw = wrapper_kwargs_from_hparams(blob, "XDDCWrapper", source_contract="v1")
    assert kw["decoder_type"] == "unet"
    assert kw["unet_kwargs"]["model_channels"] == 384
    assert "dit_kwargs" not in kw


def test_an_unknown_decoder_type_is_refused():
    blob = _xddc_blob()
    blob["hyper_parameters"]["config"]["model"]["x_DDC"]["decoder_type"] = "vqgan"
    with pytest.raises(NotImplementedError, match="decoder_type"):
        wrapper_kwargs_from_hparams(blob, "XDDCWrapper", source_contract="v2")


# ---------------------------------------------------------------------------
# Routing vs dropping (2026-08-14). Before this, a scalar-routed channel was
# DELETED from varying_boundary_variables — which loads but cannot run: with
# nothing routed, the calendar row comes out 2 wide against a model wanting 3.
# ---------------------------------------------------------------------------


def _wco2_blob(scalar_dim=3, c_grid_dim=5):
    """An SI_X wCO2-shaped blob: 4 varying (CO2 first), CO2 not gridded."""
    blob = _fancy_blob()
    cfg = blob["hyper_parameters"]["config"]
    cfg["model"]["model_name"] = "SI_X"
    data = cfg["data"]
    n_state = (
        len(data["surface_variables"])
        + len(data.get("diagnostic_variables", []) or [])
        + len(data["upper_air_variables"]) * len(data["levels"])
    )
    # An AmipDiT block, not a copy of the rolling one: `temporal_num_heads` and
    # friends are RollingDiT-only, and wrapper_kwargs_from_hparams passes the
    # block through verbatim (the translator's unknown-kwarg filter runs later,
    # in build_target_wrapper).
    cfg["model"]["SI_X"] = {
        "model": {
            "in_channels": 2 * n_state,
            "out_channels": n_state,
            "c_grid_dim": c_grid_dim,
            "scalar_dim": scalar_dim,
            "dim": 32,
            "num_heads": 2,
            "num_blocks": 1,
            "patch_size": 2,
            "nlat": 16,
            "nlon": 32,
            "c_grid_downsample": 4,
        }
    }
    cfg["model"].pop("ERDM", None)
    cfg["model"].pop("backbone", None)
    cfg["data"]["varying_boundary_variables"] = [
        "global_mean_co2",
        "DSWRFtoa_24h_lead",
        "sea_surface_temperature_monthly_interp",
        "sea_ice_cover_monthly_interp",
    ]
    cfg["data"]["sst_anomaly_channel"] = "none"
    cfg["data"].pop("ocean_state_variables", None)
    return blob


def test_a_scalar_routed_channel_is_routed_not_deleted():
    kw = wrapper_kwargs_from_hparams(
        _wco2_blob(), "AmipDiTWrapper", source_contract="v1"
    )
    # Still listed (the NaN-fill needs the stored order) AND named as routed.
    assert "global_mean_co2" in kw["varying_boundary_variables"]
    assert kw["scalar_routed_boundary_variables"] == ["global_mean_co2"]
    assert kw["scalar_dim"] == 3


def test_the_routed_artifact_is_self_consistent():
    """The wrapper's own validation is the check that matters here."""
    from physicsnemo.experimental.models.amip_si import AmipDiTWrapper

    kw = wrapper_kwargs_from_hparams(
        _wco2_blob(), "AmipDiTWrapper", source_contract="v1"
    )
    model = AmipDiTWrapper(**kw)
    # 2 constant + 3 gridded: CO2 rides the calendar row instead.
    assert model.c_grid_dim == 5
    assert model.scalar_dim == 3


def test_routing_is_refused_when_scalar_dim_does_not_pay_for_it():
    """scalar_dim 2 cannot carry a routed channel, so fall back to dropping."""
    kw = wrapper_kwargs_from_hparams(
        _wco2_blob(scalar_dim=2), "AmipDiTWrapper", source_contract="v1"
    )
    assert kw.get("scalar_routed_boundary_variables", []) == []
    assert "global_mean_co2" not in kw["varying_boundary_variables"]


def test_a_trailing_drop_is_never_called_routing():
    """Sea ice is a real gridded forcing; routing it would be nonsense.

    Guards the trap the first version of this fell into: the fancy configs carry
    scalar_dim 3 for the global_mean_sst TREND scalar, which coincidentally
    equals ``2 + 1`` and so looked like permission to route whatever the trim
    dropped.
    """
    blob = _fancy_blob()
    blob["hyper_parameters"]["config"]["model"]["ERDM"]["DiT"]["c_grid_dim"] = 5
    kw = wrapper_kwargs_from_hparams(
        blob, "RollingDiTWrapper", source_contract="v2"
    )
    assert kw.get("scalar_routed_boundary_variables", []) == []
    assert "sea_ice_cover_monthly_interp" not in kw["varying_boundary_variables"]
