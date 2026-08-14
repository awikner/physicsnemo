# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint/config channel-contract guard (2026-08-14).

Every driver builds the model from ``cfg.model`` and *then* loads weights into
it, so run-time packing follows the YAML rather than the artifact. The failure
mode this guard exists for is **shape-preserving**: flipping ``channel_layout``
permutes the upper-air block without moving a single parameter shape, so

* ``load_state_dict`` succeeds — nothing to complain about;
* the Phase-12h shape digest is *identical* (asserted below), so the config
  health gates structurally cannot catch it;
* the run produces plausible output that is wrong everywhere.

That combination is why it gets a hard failure rather than a doc paragraph.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

_AI_ROSSBY_DIR = (
    Path(__file__).resolve().parents[2].parent / "examples" / "weather" / "ai_rossby"
)
sys.path.insert(0, str(_AI_ROSSBY_DIR))

from train_loop import (  # noqa: E402
    _WARM_START_WARN_KEYS,
    assert_checkpoint_contract,
    assert_checkpoint_dir_contract,
    find_mdlus_for_model,
    load_partial_weights,
    mdlus_stored_args,
)

from physicsnemo.experimental.models.amip_si import (  # noqa: E402
    RollingDiTWrapper,
    XDDCWrapper,
)

_SURF = ["skin_temperature", "surface_pressure"]
_UA = ["temperature"]
_DIAG = ["PRATEsfc_24h"]
_LEVELS = [500.0, 850.0]
_OCEAN = [
    "sea_surface_temperature_monthly_interp",
    "sea_ice_cover_monthly_interp",
]


def _xddc(**over):
    kwargs = dict(
        surface_variables=_SURF,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        levels=_LEVELS,
        horizontal_resolution=[8, 16],
        downsample_factor=2,
        channel_layout="v2",
        decoder_type="dit",
        dit_kwargs=dict(dim=32, num_heads=2, num_blocks=1, patch_size=4),
    )
    kwargs.update(over)
    return XDDCWrapper(**kwargs)


def _rolling(**over):
    kwargs = dict(
        surface_variables=_SURF,
        upper_air_variables=_UA,
        diagnostic_variables=_DIAG,
        constant_boundary_variables=["land_sea_mask"],
        varying_boundary_variables=list(_OCEAN),
        levels=_LEVELS,
        horizontal_resolution=[8, 16],
        channel_layout="v2",
        rolling_dit_kwargs=dict(
            dim=32,
            num_heads=2,
            num_blocks=1,
            temporal_num_heads=2,
            window_size=3,
            input_embed={"mode": "budget", "d_boundary": 8, "d_calendar": 8},
            output_head={"mode": "mix", "num_experts": 2},
        ),
    )
    kwargs.update(over)
    return RollingDiTWrapper(**kwargs)


def _save(model, path):
    model.save(str(path))
    return Path(path)


# ---------------------------------------------------------------------------
# Why a shape check cannot stand in for this
# ---------------------------------------------------------------------------


def test_a_layout_flip_moves_no_parameter_shape():
    """The premise of the guard, asserted rather than assumed."""
    v1, v2 = _xddc(channel_layout="v1"), _xddc(channel_layout="v2")
    shapes_v1 = {k: tuple(t.shape) for k, t in v1.state_dict().items()}
    shapes_v2 = {k: tuple(t.shape) for k, t in v2.state_dict().items()}
    assert shapes_v1 == shapes_v2
    # ...and therefore the weights load happily across the mismatch.
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=True)
    assert not missing and not unexpected


# ---------------------------------------------------------------------------
# mdlus_stored_args
# ---------------------------------------------------------------------------


def test_stored_args_include_defaults_the_caller_never_passed(tmp_path):
    """``Module.save`` records RESOLVED kwargs, so coverage is total.

    If it only recorded explicitly-passed kwargs, an artifact from a config
    that omits ``channel_layout`` (``amip_si`` and friends) could not be
    checked at all.
    """
    path = _save(_xddc(), tmp_path / "m.mdlus")  # channel_layout passed here...
    args = mdlus_stored_args(path)
    assert args["channel_layout"] == "v2"
    # ...but downsample_factor's sibling defaults are present regardless.
    assert "levels" in args and "surface_variables" in args


def test_stored_args_is_none_for_a_non_mdlus_file(tmp_path):
    plain = tmp_path / "weights.pt"
    torch.save({"a": torch.zeros(1)}, plain)
    assert mdlus_stored_args(plain) is None


def test_stored_args_is_none_for_a_corrupt_archive(tmp_path):
    bad = tmp_path / "truncated.mdlus"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    assert mdlus_stored_args(bad) is None


# ---------------------------------------------------------------------------
# assert_checkpoint_contract
# ---------------------------------------------------------------------------


def test_matching_contract_passes(tmp_path):
    path = _save(_xddc(), tmp_path / "m.mdlus")
    assert assert_checkpoint_contract(_xddc(), path) == {}


def test_layout_mismatch_raises_and_names_the_key(tmp_path):
    path = _save(_xddc(channel_layout="v1"), tmp_path / "v1.mdlus")
    with pytest.raises(ValueError, match="channel_layout") as excinfo:
        assert_checkpoint_contract(_xddc(channel_layout="v2"), path)
    msg = str(excinfo.value)
    # The message has to be actionable, not just correct.
    assert "checkpoint" in msg and "cfg.model" in msg
    assert "channel_layout" in msg


def test_a_permuted_variable_list_is_caught(tmp_path):
    """Also shape-preserving, also silently wrong."""
    path = _save(_xddc(), tmp_path / "m.mdlus")
    swapped = _xddc(surface_variables=list(reversed(_SURF)))
    with pytest.raises(ValueError, match="surface_variables"):
        assert_checkpoint_contract(swapped, path)


def test_a_level_reorder_is_caught(tmp_path):
    path = _save(_xddc(), tmp_path / "m.mdlus")
    with pytest.raises(ValueError, match="levels"):
        assert_checkpoint_contract(_xddc(levels=list(reversed(_LEVELS))), path)


def test_int_versus_float_levels_do_not_false_positive(tmp_path):
    """A YAML may say ``500`` where the artifact round-tripped ``500.0``."""
    path = _save(_xddc(levels=[500.0, 850.0]), tmp_path / "m.mdlus")
    assert assert_checkpoint_contract(_xddc(levels=[500, 850]), path) == {}


def test_an_unreadable_checkpoint_warns_rather_than_passing_silently(tmp_path, caplog):
    plain = tmp_path / "weights.pt"
    torch.save({"a": torch.zeros(1)}, plain)
    with caplog.at_level("WARNING"):
        assert assert_checkpoint_contract(_xddc(), plain) == {}
    assert "skipping the contract check" in caplog.text


def test_warn_keys_downgrade_selected_mismatches(tmp_path, caplog):
    path = _save(_xddc(), tmp_path / "m.mdlus")
    swapped = _xddc(surface_variables=list(reversed(_SURF)))
    with caplog.at_level("WARNING"):
        diff = assert_checkpoint_contract(
            swapped, path, warn_keys=("surface_variables",)
        )
    assert "surface_variables" in diff
    assert "tolerates" in caplog.text


def test_warn_keys_never_excuse_a_layout_flip(tmp_path):
    """``channel_layout`` is fatal at every call site, warm start included."""
    path = _save(_xddc(channel_layout="v1"), tmp_path / "v1.mdlus")
    assert "channel_layout" not in _WARM_START_WARN_KEYS
    with pytest.raises(ValueError, match="channel_layout"):
        assert_checkpoint_contract(
            _xddc(channel_layout="v2"), path, warn_keys=_WARM_START_WARN_KEYS
        )


def test_a_key_absent_from_the_artifact_is_skipped(tmp_path):
    """An artifact predating a kwarg cannot be judged against it."""
    path = _save(_xddc(), tmp_path / "m.mdlus")
    # Rewrite args.json without channel_layout, as an older artifact would be.
    with zipfile.ZipFile(path) as z:
        members = {n: z.read(n) for n in z.namelist()}
    blob = json.loads(members["args.json"])
    blob["__args__"].pop("channel_layout")
    members["args.json"] = json.dumps(blob).encode()
    older = tmp_path / "older.mdlus"
    with zipfile.ZipFile(older, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    assert assert_checkpoint_contract(_xddc(channel_layout="v1"), older) == {}


# ---------------------------------------------------------------------------
# Directory resolution — must agree with load_checkpoint's own choice
# ---------------------------------------------------------------------------


def test_find_mdlus_picks_the_highest_index(tmp_path):
    model = _xddc()
    for idx in (1, 40, 7):
        _save(model, tmp_path / f"XDDCWrapper.0.{idx}.mdlus")
    found = find_mdlus_for_model(model, tmp_path)
    assert found is not None and found.name == "XDDCWrapper.0.40.mdlus"


def test_find_mdlus_ignores_other_classes(tmp_path):
    _save(_rolling(), tmp_path / "RollingDiTWrapper.0.3.mdlus")
    assert find_mdlus_for_model(_xddc(), tmp_path) is None


def test_dir_guard_is_a_noop_when_nothing_matches(tmp_path):
    # load_checkpoint logs its own miss; the guard must not add an error.
    assert_checkpoint_dir_contract(_xddc(), tmp_path)
    assert_checkpoint_dir_contract(_xddc(), tmp_path / "does-not-exist")


def test_dir_guard_raises_on_the_file_that_would_be_loaded(tmp_path):
    _save(_xddc(channel_layout="v1"), tmp_path / "XDDCWrapper.0.0.mdlus")
    with pytest.raises(ValueError, match="channel_layout"):
        assert_checkpoint_dir_contract(_xddc(channel_layout="v2"), tmp_path)


# ---------------------------------------------------------------------------
# Warm start (``training.partial_checkpoint``)
# ---------------------------------------------------------------------------


def test_warm_start_refuses_differently_packed_weights(tmp_path):
    path = _save(_rolling(channel_layout="v1"), tmp_path / "v1.mdlus")
    with pytest.raises(ValueError, match="channel_layout"):
        load_partial_weights(_rolling(channel_layout="v2"), path)


def test_the_ocean_warm_start_still_works(tmp_path):
    """The documented Phase-12f path: no-ocean -> ocean, ZERO skipped keys.

    ``ocean_state_variables`` differing is the *point* of this warm start, so
    the guard must tolerate it while still checking the packing order.
    """
    base = _rolling()
    path = _save(base, tmp_path / "base.mdlus")
    ocean = _rolling(ocean_state_variables=list(_OCEAN))
    report = load_partial_weights(ocean, path)
    assert report["skipped"] == [], report["skipped"]
    assert report["loaded"], "warm start loaded nothing"
    # The added ocean parameters are the ones left at init.
    assert any("ocean" in k for k in report["fresh"]), report["fresh"][:5]


def test_warm_start_from_a_plain_pt_is_unaffected(tmp_path):
    """A ``.pt`` state dict carries no args, so the guard cannot apply."""
    model = _rolling()
    plain = tmp_path / "weights.pt"
    torch.save(model.state_dict(), plain)
    report = load_partial_weights(_rolling(), plain)
    assert report["skipped"] == []


def test_a_bool_kwarg_is_not_confused_with_a_number(tmp_path):
    """``True == 1.0`` in Python; the normalizer must not let that through.

    No contract key is a bool today, but the normalizer is generic and the
    ``bool``-is-an-``int`` trap is exactly the sort of thing a later key would
    step on silently.
    """
    from train_loop import _normalize_contract_value as norm

    assert norm([True, False]) != norm([1, 0])
    assert norm(1) == norm(1.0)
    assert norm(["a", 2]) == norm(["a", 2.0])


# ---------------------------------------------------------------------------
# Resume (``train_diffusion.py``) — the fourth call site.
# ---------------------------------------------------------------------------


def test_the_resume_path_is_guarded_too(tmp_path):
    """Relaunching a run_name against an edited layout must not resume.

    This is the site where the config looks self-evidently right — it is "the
    same run" — so an edited ``channel_layout`` resumes cleanly and trains on
    differently-packed weights for however long the job lasts.
    """
    _save(_rolling(channel_layout="v1"), tmp_path / "RollingDiTWrapper.0.7.mdlus")
    with pytest.raises(ValueError, match="channel_layout"):
        assert_checkpoint_dir_contract(_rolling(channel_layout="v2"), tmp_path)


def test_a_fresh_run_directory_is_not_an_error(tmp_path):
    """Epoch 1 of a new run has nothing to check against."""
    assert_checkpoint_dir_contract(_rolling(), tmp_path / "checkpoints")


def test_train_diffusion_calls_the_guard_before_loading():
    """Order matters: after the load, the weights are already in the model."""
    import inspect

    import train_diffusion

    src = inspect.getsource(train_diffusion.main)
    guard = src.index("assert_checkpoint_dir_contract(inner_model")
    resume = src.index("resumed_epoch = load_checkpoint(")
    assert guard < resume, "the contract guard must run before the resume load"
