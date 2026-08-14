# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12h.29 — health gates over every shipped AMIP model config.

The useful subset of upstream's ``tools/check_repo.py``, as tests:

* **instantiate-everything** — every ``conf/model/amip_*.yaml`` builds.
* **shape-signature hash** — a digest of ``{param name: shape}`` per config,
  pinned. This is the gate that would have caught Phase 12h's worst bug directly:
  the translator built ``input_embed.boundary_embed.static_bias`` at
  ``[256, 180, 360]`` instead of ``[256, 45, 90]`` because it passed the data
  resolution instead of the token grid. Only one parameter in the tree is
  grid-shaped, so with ``boundary_static_bias: false`` that error would have
  loaded cleanly and been silently wrong. A shape digest notices regardless.
* **synthetic forward** — each config runs on tensors of its own declared widths.

Configs are shrunk before building (a 561M-param model per config would make this
a benchmark, not a test), so the digests pin *relative* structure — how the
channel contract and the module tree derive from the variable lists — rather than
production parameter counts. That is the part that silently breaks.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models import amip_si as amip_si_models

_CONF = (
    Path(__file__).resolve().parents[2].parent
    / "examples" / "weather" / "ai_rossby" / "conf" / "model"
)

#: Config-only keys that are not wrapper constructor args (mirrors
#: ``train._MODEL_CONFIG_ONLY_KEYS``; duplicated rather than imported so this
#: suite does not depend on the recipe's sys.path juggling).
_NON_KWARGS = {"name", "module", "target", "model_type", "timedelta_hours", "defaults"}

#: Shrink every backbone so the suite runs in seconds. Keyed by the wrapper's
#: backbone-kwargs name.
_SHRINK = {
    "rolling_dit_kwargs": dict(
        dim=64, num_heads=2, num_blocks=1, temporal_num_heads=2,
        c_grid_cross_layers=1, c_grid_cross_heads=2,
    ),
    "dit_kwargs": dict(dim=64, num_heads=2, num_blocks=1),
    "erdm_kwargs": dict(model_channels=32, channel_mult=[1, 2], num_res_blocks=1),
    "unet_kwargs": dict(model_channels=32, channel_mult=[1, 2], num_res_blocks=1),
}
_EMBED_SHRINK = dict(d_boundary=16, d_calendar=16, d_co2=8)

#: Every AMIP model config that builds a single wrapper. ``amip_combined`` is
#: excluded: it composes two *checkpoints* rather than declaring a backbone.
_CONFIGS = sorted(
    p.stem
    for p in _CONF.glob("amip_*.yaml")
    if p.stem not in {"amip_combined"}
)


def _load_composed(stem: str):
    """Load a config, merging any single-level Hydra ``defaults`` parent.

    ``amip_si_x.yaml`` is ``defaults: [amip_si]`` plus overrides, so reading it
    alone yields a config missing every inherited key — which looks exactly like
    a broken config to the gates below.
    """
    cfg = OmegaConf.load(_CONF / f"{stem}.yaml")
    parents = [p for p in (cfg.get("defaults", []) or []) if isinstance(p, str)]
    if not parents:
        return cfg
    merged = OmegaConf.create({})
    for parent in parents:
        merged = OmegaConf.merge(merged, _load_composed(parent))
    return OmegaConf.merge(merged, cfg)


def _build(stem: str):
    cfg = _load_composed(stem)
    args = {
        k: v
        for k, v in OmegaConf.to_container(cfg, resolve=True).items()
        if k not in _NON_KWARGS
    }
    for key, shrink in _SHRINK.items():
        if key in args and isinstance(args[key], dict):
            args[key].update(shrink)
            if isinstance(args[key].get("input_embed"), dict):
                args[key]["input_embed"].update(_EMBED_SHRINK)
    cls = getattr(amip_si_models, str(cfg.name))
    return cls(**args), cfg


def _shape_signature(model) -> str:
    """Stable digest of ``{param name: shape}`` — structure, not values."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
    return h.hexdigest()[:16]


# Regenerate deliberately when a config or a backbone changes shape:
#   pytest test/models/amip_si/test_config_health_gates.py -k signatures -s
_EXPECTED_SIGNATURES = {
    "amip_erdm": "889569e55c7edeb8",          # frozen v1 family (ERDM UNet)
    "amip_erdm_v2": "045200eff82a8f2b",       # 12e feature set, no ocean
    "amip_erdm_v2_ocean": "f993dea98b28b39e",  # + 12f nocean=2
    "amip_erdm_fancy": "8565f2e7c781696d",    # + 12g anomaly, nocean=3
    "amip_rfm": "07de01c248b4c471",           # frozen v1 family
    "amip_si": "04782906ab186194",
    # Identical to amip_si BY DESIGN: amip_si_x inherits it and changes only the
    # scheduler pairing, not a single shape. A future divergence here means one
    # of them moved.
    "amip_si_x": "04782906ab186194",
    "amip_x_ddc": "ba2d45a0801366be",         # v1 convolutional denoiser
    "amip_x_ddc_dit": "fc21c0c4de4d300c",     # 12h DiTAE denoiser
}


#: The channel contract each config is pinned to. Separate from the shape
#: digest ON PURPOSE: ``channel_layout`` changes how channels are PACKED, not
#: how many, so a flip is invisible to a digest over parameter shapes
#: (``amip_x_ddc`` hashes to the same value under ``v1`` and ``v2``). Without
#: this table, changing a layout — the single most consequential edit anyone can
#: make to these configs — would pass the entire suite silently.
_EXPECTED_LAYOUTS = {
    # Frozen v1 families. `fork` is this fork's own Phase-8 packing: fine for
    # training from scratch here, and never right for real upstream weights
    # (translated v1 artifacts need ++model.channel_layout=v1 at run time).
    "amip_si": "fork",
    "amip_si_x": "fork",
    "amip_erdm": "fork",
    "amip_rfm": "fork",
    # The v1 CONVOLUTIONAL downscaler. `v1` since 2026-08-14: XDDCUNet exists
    # only in amip v1, so every checkpoint this config can load is a v1
    # artifact, and nothing in-repo trains it.
    "amip_x_ddc": "v1",
    # amip_v2 proper.
    "amip_erdm_v2": "v2",
    "amip_erdm_v2_ocean": "v2",
    "amip_erdm_fancy": "v2",
    "amip_x_ddc_dit": "v2",
}


@pytest.mark.parametrize("stem", _CONFIGS)
def test_channel_layout_is_pinned(stem):
    model, _ = _build(stem)
    expected = _EXPECTED_LAYOUTS.get(stem)
    assert expected is not None, (
        f"{stem} has no pinned channel_layout; add one to _EXPECTED_LAYOUTS. A "
        f"config whose packing order nobody asserts can be flipped silently."
    )
    assert getattr(model, "channel_layout", None) == expected, (
        f"{stem} packs channels as "
        f"{getattr(model, 'channel_layout', None)!r}, expected {expected!r}. This "
        f"changes channel ORDER without changing any shape, so no other gate in "
        f"this file can see it — update the pin deliberately, or fix the config."
    )


def test_at_least_the_known_configs_are_present():
    """A renamed or deleted config should fail loudly, not shrink coverage."""
    assert set(_CONFIGS) >= {
        "amip_erdm_v2", "amip_erdm_v2_ocean", "amip_erdm_fancy",
        "amip_si", "amip_si_x", "amip_rfm", "amip_x_ddc", "amip_x_ddc_dit",
    }, _CONFIGS


@pytest.mark.parametrize("stem", _CONFIGS)
def test_every_config_instantiates(stem):
    model, _ = _build(stem)
    n = sum(p.numel() for p in model.parameters())
    assert n > 0


@pytest.mark.parametrize("stem", _CONFIGS)
def test_shape_signatures(stem, capsys):
    """Pin the per-config parameter-shape digest.

    A ``None`` expectation prints the measured digest instead of failing, so the
    table can be filled in from one run; once pinned, any unintended change to a
    channel contract or a module tree fails here.
    """
    model, _ = _build(stem)
    sig = _shape_signature(model)
    expected = _EXPECTED_SIGNATURES.get(stem)
    if expected is None:
        with capsys.disabled():
            print(f'\n    "{stem}": "{sig}",')
        pytest.skip(f"{stem}: signature not pinned yet ({sig})")
    assert sig == expected, (
        f"{stem}'s parameter shapes changed (got {sig}, expected {expected}). If "
        f"deliberate, update _EXPECTED_SIGNATURES; if not, a channel contract or "
        f"a backbone geometry moved."
    )


@pytest.mark.parametrize(
    "stem",
    ["amip_erdm_v2", "amip_erdm_v2_ocean", "amip_erdm_fancy", "amip_rfm"],
)
def test_rolling_configs_run_on_their_own_declared_widths(stem):
    """Synthetic forward at the widths the config itself derives.

    Uses the wrapper's own ``in_channels`` / ``c_grid_dim`` / ``scalar_dim``, so
    a config whose lists and backbone disagree fails here rather than at the
    first real batch.
    """
    model, cfg = _build(stem)
    model.eval()
    b = 1
    W = int(cfg.rolling_dit_kwargs.get("window_size", 6))
    nlat, nlon = model.horizontal_resolution
    down = int(cfg.rolling_dit_kwargs.get("c_grid_downsample", 1) or 1)
    z = torch.randn(b, W, model.in_channels, nlat, nlon)
    c_grid = (
        torch.randn(b, W, model.c_grid_dim, nlat * down, nlon * down)
        if model.c_grid_dim
        else None
    )
    c_scalar = torch.randn(b, W, model.scalar_dim) if model.scalar_dim else None
    with torch.no_grad():
        out = model(z, torch.zeros(b, W), c_grid=c_grid, c_scalar=c_scalar)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("stem", ["amip_x_ddc", "amip_x_ddc_dit"])
def test_xddc_configs_run_on_their_own_declared_widths(stem):
    model, _ = _build(stem)
    model.eval()
    nlat, nlon = model.horizontal_resolution
    x = torch.randn(1, model.in_channels, nlat, nlon)
    # (b, 1): XDDCUNet's timestep embedder needs the trailing axis, DiTAE takes
    # either.
    with torch.no_grad():
        out = model(x, x.clone(), torch.zeros(1, 1))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("stem", _CONFIGS)
def test_channel_contract_is_self_consistent(stem):
    """``in_channels`` must equal what the variable lists imply.

    The arithmetic every phase of this rebaseline has depended on: surface +
    diagnostic + levels x upper_air, plus the Phase-12f ocean tail.
    """
    model, cfg = _build(stem)
    implied = (
        len(cfg.get("surface_variables", []) or [])
        + len(cfg.get("diagnostic_variables", []) or [])
        + len(cfg.get("levels", []) or [])
        * len(cfg.get("upper_air_variables", []) or [])
    )
    ocean = len(cfg.get("ocean_state_variables", []) or [])
    assert model.in_channels == implied + ocean, (
        f"{stem}: in_channels {model.in_channels} != {implied} state + {ocean} ocean"
    )
