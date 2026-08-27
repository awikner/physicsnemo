#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Translate ThorCast SFNO_PLASIM checkpoints into the PanguWeather v2.0 layout.

ThorCast (``/eagle/MDClimSim/troyarcomano/ThorCast``, ANL Polaris) trains the
*same* vendored Modulus SFNO this repo ships as
:mod:`physicsnemo.experimental.models.modulus_sfno` — same module tree, same
``state_dict`` key names, no PyTorch Lightning anywhere despite the ``.pth``
suffix. What differs is **how the flat input vector is assembled**, and that
difference is invisible to a key-name check:

.. code-block:: text

    ThorCast  (trainer_PLASIM_v4.py::flatten_input_data)
        [ surface | upper_air x levels | constant_boundary | varying_boundary ]

    PanguWeather v2.0 / ai-rossby  (SphericalFourierNeuralOperatorNet_v2.forward,
                                    SfnoPlasim.forward)
        [ surface | constant_boundary | varying_boundary | upper_air x levels ]

Loading ThorCast weights into either target without re-ordering runs cleanly
and produces plausible-looking nonsense, so this translator's whole job is the
permutation. It rewrites exactly two tensors:

``encoder.0.weight``
    Shape ``(embed_dim, in_chans, 1, 1)``; gather along dim 1.

``decoder.0.weight``
    Shape ``(hidden, embed_dim + in_chans, 1, 1)`` when ``big_skip=True``. Both
    implementations do ``torch.cat((x, residual), dim=1)``, so the **trailing**
    ``in_chans`` columns are the raw input and need the *same* gather. Permuting
    only the encoder is the classic silent-corruption bug here.

Biases are per-output-channel and untouched. The output ordering
(``surface | upper_air | diagnostic``) is identical in ThorCast and in both
targets, so ``decoder.2.weight`` normally needs no change — but the output
permutation is derived and applied on the same footing, so a future family with
a different output order is handled rather than silently mistranslated.

The emitted payload is the PanguWeather ``torch.save(dict, "*.tar")`` blob (the
``.tar`` is a naming convention, not a tarball). It is **weights-only**:
ThorCast's optimizer/scheduler state is dropped, and ``ema_state`` is ``None``
because ThorCast's trainer has no EMA. That ``None`` is deliberate and safe —
:mod:`sfno_plasim`'s loader prefers ``ema_state`` only when it is not ``None``
and otherwise falls through to ``model_state``.

Pipeline
--------
Stage A (this tool) does the semantic work; stage B is a pure key re-prefix::

    # A: ThorCast .pth -> PanguWeather-layout .tar
    python tools/checkpoint_translation/thorcast_sfno.py \
        --source  .../..._no_soil_ff_best_val.pth \
        --model-config examples/weather/ai_rossby/conf/model/sfno_plasim_5412.yaml \
        --output  .../sfno_plasim_thorcast_no_soil_ff.tar

    # B: .tar -> .mdlus  (existing translator, unmodified)
    python tools/checkpoint_translation/sfno_plasim.py \
        --source  .../sfno_plasim_thorcast_no_soil_ff.tar \
        --model-config examples/weather/ai_rossby/conf/model/sfno_plasim_5412.yaml \
        --output  .../sfno_plasim_thorcast_no_soil_ff.mdlus --strict
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel-layout bookkeeping
# ---------------------------------------------------------------------------

# Group concatenation order on each side. These are properties of *code*, not of
# config, so they are constants here rather than CLI knobs:
#   source -> ThorCast trainer_PLASIM_v4.py::flatten_input_data
#   target -> modulus_sfno/sfnonet.py::SphericalFourierNeuralOperatorNet_v2.forward
#             and models/sfno_plasim/sfno_plasim.py::SfnoPlasim.forward
THORCAST_INPUT_GROUP_ORDER = (
    "surface",
    "upper_air",
    "constant_boundary",
    "varying_boundary",
)
PANGU_INPUT_GROUP_ORDER = (
    "surface",
    "constant_boundary",
    "varying_boundary",
    "upper_air",
)

# Output ordering: identical on both sides (flatten_target_data vs the forward's
# positional split), but kept explicit so a mismatch is derived, not assumed.
THORCAST_OUTPUT_GROUP_ORDER = ("surface", "upper_air", "diagnostic")
PANGU_OUTPUT_GROUP_ORDER = ("surface", "upper_air", "diagnostic")

# ThorCast PLASIM v4 defaults, from config/PANGU_NEW_0171_no_soil.yaml (the
# config `inference_PLASIM_v4.py` pairs with the _no_soil_ff checkpoint).
# Note `constant_boundary` is lsm,z0,sg here but lsm,sg,z0 in the ai-rossby
# configs and in the PLASIM zarr's own attrs — that swap folds into the same
# permutation vector.
THORCAST_DEFAULTS = {
    "surface": ["pl", "tas"],
    "upper_air": ["ta", "ua", "va", "hus", "zg"],
    "constant_boundary": ["lsm", "z0", "sg"],
    "varying_boundary": ["sst", "rsdt", "sic"],
    "diagnostic": ["pr_6h"],
}

# Mirrors train.py::_MODEL_CONFIG_ONLY_KEYS — identity fields plus the
# recipe-metadata `timedelta_hours`, none of which are constructor args.
_IDENTITY_KEYS = ("name", "module", "target", "model_type", "timedelta_hours")
_WRAP_PREFIXES = ("module.", "_orig_mod.")


def strip_wrap_prefixes(key: str) -> str:
    """Strip leading DDP / ``torch.compile`` wrapper prefixes from a key.

    Iterative, so stacked prefixes such as ``module._orig_mod.encoder.0.weight``
    collapse to ``encoder.0.weight``. Idempotent.
    """
    changed = True
    while changed:
        changed = False
        for pref in _WRAP_PREFIXES:
            if key.startswith(pref):
                key = key[len(pref) :]
                changed = True
    return key


def expand_channels(
    groups: dict, group_order: Sequence[str], n_levels: int
) -> list[str]:
    """Expand variable groups into the flat channel-name vector, in order.

    Upper-air variables are var-major: every level of ``ta`` precedes the first
    level of ``ua``, matching both ThorCast's nested
    ``for var: for level:`` loop and the target's
    ``view(b, n_upper * n_levels, ...)`` flattening of a ``(var, level)`` tensor.

    Parameters
    ----------
    groups : dict
        Maps group name -> list of variable names.
    group_order : sequence of str
        Group concatenation order.
    n_levels : int
        Number of vertical levels per upper-air variable.

    Returns
    -------
    list of str
        Channel names; upper-air entries are spelled ``"<var>@<level_index>"``.
    """
    names: list[str] = []
    for group in group_order:
        for var in groups.get(group, ()):
            if group == "upper_air":
                names.extend(f"{var}@{k}" for k in range(n_levels))
            else:
                names.append(var)
    return names


def derive_permutation(source_names: Sequence[str], target_names: Sequence[str]) -> list[int]:
    """Build the gather index mapping the source layout onto the target layout.

    Returns ``perm`` such that ``target[i] == source[perm[i]]`` — i.e. suitable
    for ``tensor[:, perm]``.

    Raises
    ------
    ValueError
        If the two layouts do not describe the same *set* of channels. This is
        the load-bearing safety net: a wrong variable list (or level count)
        surfaces here rather than as silently wrong weights.
    """
    if len(source_names) != len(target_names):
        raise ValueError(
            f"channel count differs: source has {len(source_names)}, "
            f"target has {len(target_names)}"
        )
    if sorted(source_names) != sorted(target_names):
        only_src = sorted(set(source_names) - set(target_names))
        only_tgt = sorted(set(target_names) - set(source_names))
        raise ValueError(
            "source and target describe different channel sets; "
            f"only in source = {only_src[:8]}, only in target = {only_tgt[:8]}"
        )
    index = {name: i for i, name in enumerate(source_names)}
    return [index[name] for name in target_names]


# ---------------------------------------------------------------------------
# Weight surgery
# ---------------------------------------------------------------------------


def permute_state_dict(
    state_dict: OrderedDict,
    *,
    input_perm: Sequence[int],
    output_perm: Optional[Sequence[int]] = None,
) -> OrderedDict:
    """Re-order the input (and optionally output) channel axes of an SFNO.

    Parameters
    ----------
    state_dict : OrderedDict
        Base-SFNO state dict, wrapper prefixes already stripped.
    input_perm : sequence of int
        Gather index over input channels, as returned by
        :func:`derive_permutation`.
    output_perm : sequence of int, optional
        Gather index over output channels. ``None`` (or an identity
        permutation) leaves ``decoder.2.weight`` untouched.

    Returns
    -------
    OrderedDict
        New dict; tensors that needed no change are passed through by reference.

    Raises
    ------
    KeyError
        If the expected encoder/decoder keys are absent.
    ValueError
        If ``decoder.0.weight``'s width matches neither the ``big_skip`` nor the
        no-skip geometry (i.e. the architecture is not what we think it is).
    """
    out = OrderedDict(state_dict)

    for required in ("encoder.0.weight", "encoder.2.weight", "decoder.0.weight"):
        if required not in out:
            raise KeyError(
                f"{required!r} missing — is this a Modulus-SFNO state dict? "
                f"got keys like {list(out)[:5]}"
            )

    enc = out["encoder.0.weight"]
    in_chans = enc.shape[1]
    if len(input_perm) != in_chans:
        raise ValueError(
            f"input_perm has {len(input_perm)} entries but encoder.0.weight "
            f"expects {in_chans} input channels"
        )
    out["encoder.0.weight"] = enc[:, list(input_perm), :, :].clone()

    # `big_skip` concatenates the *raw input* after the latent:
    # torch.cat((x, residual), dim=1) -> [embed_dim | in_chans].
    embed_dim = out["encoder.2.weight"].shape[0]
    dec = out["decoder.0.weight"]
    width = dec.shape[1]
    if width == embed_dim + in_chans:
        head, skip = dec[:, :embed_dim], dec[:, embed_dim:]
        out["decoder.0.weight"] = torch.cat(
            (head, skip[:, list(input_perm), :, :]), dim=1
        ).clone()
        logger.info(
            "permuted encoder.0.weight and the big_skip block of "
            "decoder.0.weight (%d latent + %d raw-input columns)",
            embed_dim,
            in_chans,
        )
    elif width == embed_dim:
        logger.info("big_skip absent — only encoder.0.weight permuted")
    else:
        raise ValueError(
            f"decoder.0.weight width {width} matches neither big_skip "
            f"({embed_dim + in_chans}) nor no-skip ({embed_dim}) geometry"
        )

    if output_perm is not None and list(output_perm) != list(range(len(output_perm))):
        dec_out = out["decoder.2.weight"]
        if len(output_perm) != dec_out.shape[0]:
            raise ValueError(
                f"output_perm has {len(output_perm)} entries but "
                f"decoder.2.weight emits {dec_out.shape[0]} channels"
            )
        out["decoder.2.weight"] = dec_out[list(output_perm), :, :, :].clone()
        logger.info("permuted decoder.2.weight output channels")

    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_thorcast_checkpoint(source: Path) -> tuple[OrderedDict, dict]:
    """Load a ThorCast ``.pth`` and return ``(state_dict, provenance)``.

    ThorCast's payload (``trainer_PLASIM_v4.py::save_model``) is
    ``{"epoch", "step", "model_state_dict", "optimizer_state_dict",
    "scheduler_state_dict"}``. Keys carry a ``module.`` prefix (DDP) which is
    stripped here.

    Note
    ----
    ThorCast's ``*_latest_epoch.pth`` files often carry a stale CRC on the
    archive's ``data.pkl`` record (the tensor storages are intact). ``torch.load``
    tolerates it but strict zip readers do not, so prefer ``*_best_val.pth``
    where the two are equivalent.
    """
    blob = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(
            f"{source}: expected a dict checkpoint, got {type(blob).__name__}"
        )
    if "model_state_dict" in blob:
        raw = blob["model_state_dict"]
    elif "model_state" in blob:
        raw = blob["model_state"]
    else:
        raw = blob
        logger.info("no model_state_dict key — treating %s as a raw state dict", source)

    sd = OrderedDict(
        (strip_wrap_prefixes(k), v) for k, v in raw.items() if k != "_metadata"
    )
    provenance = {
        "epoch": blob.get("epoch") if isinstance(blob, dict) else None,
        "step": blob.get("step") if isinstance(blob, dict) else None,
    }
    logger.info(
        "loaded %d tensors from %s (epoch=%s step=%s)",
        len(sd),
        source,
        provenance["epoch"],
        provenance["step"],
    )
    return sd, provenance


def load_target_groups(model_yaml: Path) -> tuple[dict, int]:
    """Read variable groups + level count from an ai-rossby model YAML.

    Returns
    -------
    tuple
        ``(groups, n_levels)`` where ``groups`` is keyed by the short group
        names used in :data:`PANGU_INPUT_GROUP_ORDER`.
    """
    with open(model_yaml) as fh:
        cfg = yaml.safe_load(fh)
    for k in _IDENTITY_KEYS:
        cfg.pop(k, None)
    groups = {
        "surface": list(cfg.get("surface_variables", [])),
        "upper_air": list(cfg.get("upper_air_variables", [])),
        "constant_boundary": list(cfg.get("constant_boundary_variables", [])),
        "varying_boundary": list(cfg.get("varying_boundary_variables", [])),
        "diagnostic": list(cfg.get("diagnostic_variables", []) or []),
    }
    return groups, len(cfg.get("levels", []))


def _parse_list(value: Optional[str], default: list[str]) -> list[str]:
    """Parse a comma-separated CLI list, falling back to ``default``."""
    if value is None:
        return list(default)
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", type=Path, required=True, help="ThorCast .pth checkpoint."
    )
    p.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Target model YAML defining the destination channel layout "
        "(e.g. examples/weather/ai_rossby/conf/model/sfno_plasim_5412.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PanguWeather-layout .tar (a torch.save blob, not a tarball).",
    )
    for group, default in THORCAST_DEFAULTS.items():
        p.add_argument(
            f"--source-{group.replace('_', '-')}",
            dest=f"source_{group}",
            default=None,
            help=f"Comma-separated ThorCast {group} variables "
            f"(default: {','.join(default)}).",
        )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Derive and report the permutation without writing the output.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    target_groups, n_levels = load_target_groups(args.model_config)
    if n_levels == 0:
        logger.error("%s declares no `levels` — cannot size upper air", args.model_config)
        return 1
    source_groups = {
        g: _parse_list(getattr(args, f"source_{g}"), default)
        for g, default in THORCAST_DEFAULTS.items()
    }

    src_in = expand_channels(source_groups, THORCAST_INPUT_GROUP_ORDER, n_levels)
    tgt_in = expand_channels(target_groups, PANGU_INPUT_GROUP_ORDER, n_levels)
    src_out = expand_channels(source_groups, THORCAST_OUTPUT_GROUP_ORDER, n_levels)
    tgt_out = expand_channels(target_groups, PANGU_OUTPUT_GROUP_ORDER, n_levels)

    try:
        input_perm = derive_permutation(src_in, tgt_in)
        output_perm = derive_permutation(src_out, tgt_out)
    except ValueError as exc:
        logger.error("layout mismatch: %s", exc)
        return 1

    logger.info(
        "input layout: %d channels, %s",
        len(input_perm),
        "identity" if input_perm == list(range(len(input_perm))) else "permuted",
    )
    logger.info(
        "output layout: %d channels, %s",
        len(output_perm),
        "identity" if output_perm == list(range(len(output_perm))) else "permuted",
    )
    logger.debug("input_perm = %s", input_perm)

    if args.dry_run:
        logger.info("--dry-run: not reading weights, not writing output")
        return 0

    sd, provenance = load_thorcast_checkpoint(args.source)
    permuted = permute_state_dict(
        sd, input_perm=input_perm, output_perm=output_perm
    )

    payload = {
        "model_state": permuted,
        # ThorCast's trainer computes no EMA. `None` (not a copy of
        # model_state) keeps that honest; sfno_plasim.py falls through to
        # model_state on its own.
        "ema_state": None,
        "epoch": provenance["epoch"],
        "iters": provenance["step"],
        "translated_from": str(args.source),
        "channel_layout": "panguweather_v2",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info("wrote %s (%d tensors, weights-only)", args.output, len(permuted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
