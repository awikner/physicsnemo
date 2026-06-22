# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU smoke test for the Phase 5 Pangu_Plasim translator.

End-to-end on Delta `gpuA40x4-interactive`: build tiny
PanguPlasimLegacy on GPU → save its state_dict three ways (raw,
DDP-prefixed, DDP+compile stacked) → translate each via the
translator → load into a fresh GPU model → forward output bit-matches
the source. Catches the "module." prefix foot-gun the user called out,
on real CUDA tensors.

Skipped when ``$AI_ROSSBY_TEST_DATA`` isn't staged (the test doesn't
read the fixture but stays gated by the same env var so it co-runs
with the rest of the smoke pack on Delta).
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import yaml

_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "checkpoint_translation"
sys.path.insert(0, str(_TOOLS_DIR))

from pangu_plasim import (  # noqa: E402
    build_target_model_from_yaml,
    load_panguweather_state_dict,
    translate_state_dict,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.pangu_plasim import PanguPlasimLegacy


_HAS_FIXTURE = bool(os.environ.get("AI_ROSSBY_TEST_DATA"))
_skip_no_fixture = pytest.mark.skipif(
    not _HAS_FIXTURE, reason="$AI_ROSSBY_TEST_DATA unset (smoke-test gate)"
)


_TINY_CFG = dict(
    surface_variables=["pl", "tas"],
    upper_air_variables=["ta", "ua", "va", "hus", "zg"],
    constant_boundary_variables=["lsm", "sg", "z0"],
    varying_boundary_variables=["rsdt", "sst", "sic"],
    diagnostic_variables=["pr_6h"],
    land_variables=[],
    ocean_variables=[],
    levels=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    horizontal_resolution=[16, 32],
    patch_size=[2, 4, 4],
    window_size=[1, 2, 2],
    depths=[1, 1, 1, 1],
    num_heads=[2, 4, 4, 2],
    embed_dim=32,
    updown_scale_factor=2,
    predict_delta=False,
    mask_output=False,
    upper_air_boundary=False,
    vertical_windowing=False,
    subpixel_deconv=True,
    polar_pad=False,
    grid_has_poles=False,
    recovery_head=True,
    diagnostic_head=False,
    has_diagnostic=True,
    drop_rate=0.0,
    checkpointing=0,
    use_reentrant=False,
)


def _write_yaml(path: Path) -> None:
    blob = dict(_TINY_CFG, name="PanguPlasimLegacy")
    with open(path, "w") as fh:
        yaml.safe_dump(blob, fh)


def _make_inputs_gpu(device):
    H, W = _TINY_CFG["horizontal_resolution"]
    n_surface = len(_TINY_CFG["surface_variables"])
    n_const = len(_TINY_CFG["constant_boundary_variables"])
    n_varying = len(_TINY_CFG["varying_boundary_variables"])
    n_upper = len(_TINY_CFG["upper_air_variables"])
    n_levels = len(_TINY_CFG["levels"])
    return {
        "surface": torch.randn(1, n_surface, H, W, device=device),
        "constant_boundary": torch.randn(1, n_const, H, W, device=device),
        "varying_boundary": torch.randn(1, n_varying, H, W, device=device),
        "upper_air": torch.randn(1, n_upper, n_levels, H, W, device=device),
    }


@pytest.mark.smoke
@pytest.mark.cuda
@_skip_no_fixture
@pytest.mark.parametrize(
    "wrap_kind",
    ["raw", "ddp_only", "ddp_then_compile", "compile_then_ddp"],
)
def test_pangu_translator_round_trip_on_gpu(tmp_path, wrap_kind):
    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")
    device = torch.device("cuda:0")

    torch.manual_seed(0)
    src = PanguPlasimLegacy(**_TINY_CFG).to(device).eval()

    # Synthesize a PanguWeather-style checkpoint with the requested
    # wrapper prefixes prepended to every key. The translator must
    # collapse all four variants to the same target state.
    raw = src.state_dict()
    if wrap_kind == "raw":
        wrapped = OrderedDict(raw)
    elif wrap_kind == "ddp_only":
        wrapped = OrderedDict((f"module.{k}", v) for k, v in raw.items())
    elif wrap_kind == "ddp_then_compile":
        wrapped = OrderedDict((f"module._orig_mod.{k}", v) for k, v in raw.items())
    elif wrap_kind == "compile_then_ddp":
        wrapped = OrderedDict((f"_orig_mod.module.{k}", v) for k, v in raw.items())
    else:
        pytest.fail(f"unknown wrap_kind {wrap_kind!r}")

    blob = {"model_state": wrapped, "ema_state": wrapped}
    ckpt_path = tmp_path / f"src_{wrap_kind}.pt"
    torch.save(blob, ckpt_path)

    yaml_path = tmp_path / "model.yaml"
    _write_yaml(yaml_path)

    translated = translate_state_dict(load_panguweather_state_dict(ckpt_path))
    target = build_target_model_from_yaml(yaml_path).to(device).eval()
    incoming = target.load_state_dict(translated, strict=False)
    assert incoming.missing_keys == [], incoming.missing_keys[:5]
    assert incoming.unexpected_keys == [], incoming.unexpected_keys[:5]

    inp = _make_inputs_gpu(device)
    with torch.no_grad():
        out_src = src(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )
        out_tgt = target(
            inp["surface"], inp["constant_boundary"], inp["varying_boundary"], inp["upper_air"]
        )

    # Surface + upper_air outputs bit-match (PanguPlasimLegacy has no
    # stochasticity at inference).
    assert torch.allclose(out_src[0], out_tgt[0], atol=1e-5), (
        f"surface mismatch under {wrap_kind}: "
        f"max_diff={(out_src[0] - out_tgt[0]).abs().max().item():.3e}"
    )
    assert torch.allclose(out_src[1], out_tgt[1], atol=1e-5)
    # And the predictions are non-degenerate (real values).
    assert torch.isfinite(out_tgt[0]).all()
    assert (out_tgt[0].abs().max() > 0).item()
