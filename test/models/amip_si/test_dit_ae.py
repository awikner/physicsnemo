# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12h — ``DiTAE``, the amip_v2 x_DDC denoiser.

Upstream deleted the convolutional x_DDC (``ae_module.py`` raises
``NotImplementedError`` for any ``decoder_type`` but ``"dit"``), so a v2-trained
downscaler checkpoint can only be loaded through this backbone. Our
:class:`XDDCUNet` stays for the frozen v1 family, which is why the wrapper now
picks between them and refuses ambiguous kwargs.

The tests that matter here are the ones a translated checkpoint depends on: the
**submodule names** (they must match upstream key-for-key, or translation has to
rename and can silently mismatch) and the channel convention
(``in_channels = 2 x state`` because the conditioning is concatenated).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.amip_si import DiTAE, XDDCWrapper
    from physicsnemo.experimental.models.amip_si.x_ddc import XDDCUNet

_SURF = [f"s{i}" for i in range(6)]
_UA = [f"u{i}" for i in range(5)]
_DIAG = [f"d{i}" for i in range(15)]
_LEVELS = list(range(26))          # 6 + 15 + 5*26 = 151 state channels


def _tiny(**kw):
    base = dict(
        in_channels=8, out_channels=4, dim=32, num_heads=2, num_blocks=1,
        patch_size=4, nlat=16, nlon=32,
    )
    base.update(kw)
    return DiTAE(**base)


# ---------------------------------------------------------------------------
# Key layout — what the translator depends on
# ---------------------------------------------------------------------------


def test_submodule_names_match_upstreams():
    """Translation must be key-for-key, so these names are load-bearing.

    Upstream ``DiTAE`` exposes exactly these four families. A rename here would
    force the translator to map names, and a mapping is a place a v2 checkpoint
    can be silently mismatched.
    """
    families = {k.split(".")[0] for k in _tiny().state_dict()}
    assert families == {
        "patch_embed_main",
        "t_embedder",
        "sa_blocks",
        "unpatchify_layer",
    }


def test_block_count_follows_num_blocks():
    m = _tiny(num_blocks=3)
    idx = {k.split(".")[1] for k in m.state_dict() if k.startswith("sa_blocks.")}
    assert idx == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------


def test_forward_returns_the_full_resolution_state():
    torch.manual_seed(0)
    m = _tiny().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 4, 16, 32), torch.randn(2, 4, 16, 32), torch.zeros(2))
    assert out.shape == (2, 4, 16, 32)
    assert torch.isfinite(out).all()


def test_conditioning_is_concatenated_not_ignored():
    """``in_channels`` is 2x the state width because cond rides on the channel axis.

    If the concat were dropped the model would still run — the widths would just
    be wrong by a factor of two — so this checks the *conditioning changes the
    output*, which is the observable that distinguishes the two.
    """
    torch.manual_seed(0)
    m = _tiny().eval()
    x = torch.randn(1, 4, 16, 32)
    t = torch.zeros(1)
    with torch.no_grad():
        a = m(x, torch.zeros(1, 4, 16, 32), t)
        b = m(x, torch.ones(1, 4, 16, 32), t)
    assert not torch.allclose(a, b)


def test_diffusion_time_changes_the_output():
    torch.manual_seed(0)
    m = _tiny().eval()
    x, cond = torch.randn(1, 4, 16, 32), torch.randn(1, 4, 16, 32)
    with torch.no_grad():
        a = m(x, cond, torch.zeros(1))
        b = m(x, cond, torch.full((1,), 0.9))
    assert not torch.allclose(a, b)


def test_scalar_and_column_time_shapes_agree():
    torch.manual_seed(0)
    m = _tiny().eval()
    x, cond = torch.randn(1, 4, 16, 32), torch.randn(1, 4, 16, 32)
    with torch.no_grad():
        a = m(x, cond, torch.full((1,), 0.3))
        b = m(x, cond, torch.full((1, 1), 0.3))
    assert torch.allclose(a, b)


def test_a_non_divisible_grid_is_padded_and_cropped_back():
    # 18x34 at patch 4 needs padding; the output must still be 18x34.
    torch.manual_seed(0)
    m = _tiny(nlat=18, nlon=34).eval()
    assert (m.pad_lat, m.pad_lon) == (2, 2)
    with torch.no_grad():
        out = m(torch.randn(1, 4, 18, 34), torch.randn(1, 4, 18, 34), torch.zeros(1))
    assert out.shape == (1, 4, 18, 34)


def test_only_vanilla_unpatch_exists():
    with pytest.raises(ValueError, match="unpatch"):
        _tiny(unpatch="pixel_shuffle")


# ---------------------------------------------------------------------------
# Wrapper selection
# ---------------------------------------------------------------------------


def _wrapper(**kw):
    return XDDCWrapper(
        surface_variables=_SURF,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        levels=_LEVELS,
        horizontal_resolution=(180, 360),
        downsample_factor=4,
        **kw,
    )


def test_the_default_backbone_is_still_the_v1_unet():
    # The frozen v1 family must not change under 12h.
    assert isinstance(_wrapper().backbone, XDDCUNet)


def test_dit_selects_ditae_and_derives_the_channel_widths():
    w = _wrapper(
        decoder_type="dit",
        dit_kwargs=dict(dim=32, num_heads=2, num_blocks=1, patch_size=4),
    )
    assert isinstance(w.backbone, DiTAE)
    # Their checkpoint's config states 302 / 151; we derive both.
    assert w.backbone.in_channels == 2 * w.in_channels == 302
    assert w.backbone.out_channels == w.in_channels == 151
    # DiTAE emits the full-resolution grid, so it must have been told it.
    assert (w.backbone.nlat, w.backbone.nlon) == (180, 360)


@pytest.mark.parametrize(
    "kw",
    [
        dict(decoder_type="dit", unet_kwargs=dict(model_channels=64)),
        dict(decoder_type="unet", dit_kwargs=dict(dim=64)),
    ],
)
def test_mismatched_backbone_kwargs_are_refused(kw):
    # Ambiguity here means an ambiguous checkpoint key layout.
    with pytest.raises(ValueError, match="pick one"):
        _wrapper(**kw)


def test_an_unknown_decoder_type_is_refused():
    with pytest.raises(ValueError, match="decoder_type"):
        _wrapper(decoder_type="unet3d")


# ---------------------------------------------------------------------------
# The shipped v2 config
# ---------------------------------------------------------------------------


def test_the_v2_config_matches_the_real_checkpoints_geometry():
    """``amip_x_ddc_dit.yaml`` vs ``x_DDC_42_2026-08-07T09-34-49``'s config.yml.

    Their file states in_channels 302 / out_channels 151 / dim 1024 /
    num_blocks 20 / num_heads 16 / patch_size 4 by hand; we derive the channel
    counts from the variable lists.
    """
    from omegaconf import OmegaConf

    conf = (
        Path(__file__).resolve().parents[2].parent
        / "examples" / "weather" / "ai_rossby" / "conf" / "model"
        / "amip_x_ddc_dit.yaml"
    )
    cfg = OmegaConf.load(conf)
    assert cfg.decoder_type == "dit"
    assert int(cfg.dit_kwargs.dim) == 1024
    assert int(cfg.dit_kwargs.num_blocks) == 20
    assert int(cfg.dit_kwargs.num_heads) == 16
    assert int(cfg.dit_kwargs.patch_size) == 4
    assert str(cfg.dit_kwargs.unpatch) == "vanilla"
    # Shrink the transformer; the contract under test is the channel derivation.
    args = {
        k: v
        for k, v in OmegaConf.to_container(cfg, resolve=True).items()
        if k not in {"name", "module", "timedelta_hours"}
    }
    args["dit_kwargs"].update(dim=32, num_heads=2, num_blocks=1)
    w = XDDCWrapper(**args)
    assert w.backbone.in_channels == 302
    assert w.backbone.out_channels == 151
