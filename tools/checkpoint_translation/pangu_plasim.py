#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Translate PanguWeather v2.0 PANGU_PLASIM ``.pt`` checkpoints to ``.mdlus``.

PanguWeather's checkpoint format (per ``train.py:3573``):

.. code-block:: python

    {
        "iters": int,
        "epoch": int,
        "model_state": OrderedDict,        # raw model.state_dict() or DDP-wrapped
        "ema_state": OrderedDict or None,  # EMA shadow weights (preferred at inference)
        "optimizer_state_dict": ...,
        "scheduler_state_dict": ...,
    }

PanguWeather's PanguModel_Plasim and ai-rossby's
:class:`PanguPlasim` / :class:`PanguPlasimLegacy` use the same
submodule attribute names (``layer1`` … ``layer4``, ``downsample``,
``upsample``, ``patchembed2d``/``patchembed3d``, ``patchrecovery2d``/
``patchrecovery3d``, ``land_mask``). The state-dict keys align after
stripping the common DDP / torch.compile wrapping prefixes — there's
no re-prefix step like the SFNO translator's ``sfno.`` (the Pangu
model is the top-level module on both sides).

Strategy:

1. Load the source ``.pt``.
2. Prefer ``ema_state`` (PanguWeather's documented inference-time
   preference); fall back to ``model_state`` / ``state_dict`` / a bare
   state-dict blob.
3. Robustly strip leading wrapping prefixes from every key —
   ``module.`` (added by ``DistributedDataParallel``) and
   ``_orig_mod.`` (added by ``torch.compile``). Strip iteratively so a
   stacked ``module.module._orig_mod.foo`` key collapses to ``foo``.
4. Instantiate a fresh :class:`PanguPlasim` or :class:`PanguPlasimLegacy`
   from CLI-supplied YAML; load the translated state dict via
   :meth:`torch.nn.Module.load_state_dict` and report missing /
   unexpected keys; save via :meth:`Module.save`.

The ``--target-class`` flag is required when both PanguPlasim and
PanguPlasimLegacy are valid for a YAML; it's also auto-detected from
the YAML's ``name:`` field when present.

Usage
-----

::

    python tools/checkpoint_translation/pangu_plasim.py \\
        --source /path/to/panguweather_pangu_checkpoint.pt \\
        --model-config examples/weather/ai_rossby/conf/model/pangu_plasim_legacy.yaml \\
        --output /path/to/pangu_plasim.mdlus
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)


# Prefixes added by common wrapping patterns. Stripped iteratively
# from the start of every key so stacked combinations decompose
# correctly (e.g. ``module._orig_mod.layer1.weight`` → ``layer1.weight``).
_WRAP_PREFIXES = ("module.", "_orig_mod.")


def _strip_wrap_prefixes(key: str) -> str:
    """Strip leading DDP / torch.compile wrapper prefixes from a key.

    Idempotent: ``_strip_wrap_prefixes(_strip_wrap_prefixes(k)) == _strip_wrap_prefixes(k)``.
    """
    changed = True
    while changed:
        changed = False
        for pref in _WRAP_PREFIXES:
            if key.startswith(pref):
                key = key[len(pref) :]
                changed = True
    return key


def load_panguweather_state_dict(
    source: Path, prefer_ema: bool = True
) -> OrderedDict:
    """Load PanguWeather's .pt and return the preferred raw state dict.

    Parameters
    ----------
    source : Path
        PanguWeather checkpoint file (or any ``torch.save``-d state-dict
        / wrapped dict). When the file contains a dict, we pick
        ``ema_state`` over ``model_state`` over ``state_dict``; when
        it's a bare ordered dict (rare), we use it directly.
    prefer_ema : bool, optional, default=True
        Whether to prefer ``ema_state`` when both are present.
    """
    blob = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        if prefer_ema and blob.get("ema_state") is not None:
            sd = blob["ema_state"]
            logger.info("using ema_state from %s", source)
        elif "model_state" in blob:
            sd = blob["model_state"]
            logger.info("using model_state from %s", source)
        elif "state_dict" in blob:
            sd = blob["state_dict"]
            logger.info("using state_dict from %s (older format)", source)
        else:
            # Old-style: the dict IS the state_dict (no wrapping keys).
            sd = blob
            logger.info("treating %s as a raw state dict", source)
    else:
        # Some scripts saved an OrderedDict directly (no wrapper dict).
        sd = blob
        logger.info("treating %s as a raw state dict", source)
    return OrderedDict(sd)


def translate_state_dict(panguweather_sd: OrderedDict) -> OrderedDict:
    """Strip wrapping prefixes from every key (no re-prefix needed for Pangu).

    The Pangu submodule names align between PanguWeather and ai-rossby,
    so the only transformation we need is to peel off any DDP /
    torch.compile prefixes that the source checkpoint accumulated at
    save time. Tolerates checkpoints saved both DDP-wrapped and
    unwrapped without any change in the call site.
    """
    out: OrderedDict = OrderedDict()
    for k, v in panguweather_sd.items():
        out[_strip_wrap_prefixes(k)] = v
    return out


def _resolve_target_class(yaml_cfg: dict, cli_class: Optional[str]) -> str:
    """Pick the target class from the CLI or the YAML's ``name`` field."""
    if cli_class:
        return cli_class
    name = str(yaml_cfg.get("name", "")).strip()
    if name in ("PanguPlasim", "PanguPlasimLegacy"):
        return name
    raise ValueError(
        f"could not determine target class from YAML (name={name!r}); "
        f"pass --target-class PanguPlasim or --target-class PanguPlasimLegacy"
    )


def build_target_model_from_yaml(
    model_yaml: Path, target_class: Optional[str] = None
):
    """Instantiate a fresh PanguPlasim / PanguPlasimLegacy from a Hydra-style YAML.

    The class is determined from ``target_class`` (CLI override) or the
    YAML's ``name:`` field. Hydra metadata keys (``name``, ``module``,
    ``target``, ``model_type``) are stripped before calling the
    constructor.
    """
    import warnings

    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.pangu_plasim import (
        PanguPlasim,
        PanguPlasimLegacy,
    )

    with open(model_yaml) as fh:
        cfg = yaml.safe_load(fh)
    cls_name = _resolve_target_class(cfg, target_class)
    cls = {"PanguPlasim": PanguPlasim, "PanguPlasimLegacy": PanguPlasimLegacy}[
        cls_name
    ]
    for k in ("name", "module", "target", "model_type"):
        cfg.pop(k, None)
    return cls(**cfg)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", type=Path, required=True, help="PanguWeather .pt checkpoint."
    )
    p.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help=(
            "Hydra YAML describing the PanguPlasim / PanguPlasimLegacy "
            "constructor kwargs (e.g. "
            "examples/weather/ai_rossby/conf/model/pangu_plasim_legacy.yaml)."
        ),
    )
    p.add_argument("--output", type=Path, required=True, help="Output .mdlus path.")
    p.add_argument(
        "--target-class",
        type=str,
        default=None,
        choices=("PanguPlasim", "PanguPlasimLegacy"),
        help=(
            "Target class. When omitted, read from the YAML's ``name`` field. "
            "Set explicitly to disambiguate."
        ),
    )
    p.add_argument(
        "--prefer-model-state",
        action="store_true",
        help="Use model_state instead of ema_state even when both exist.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Refuse to write the output if the translated state dict has any "
            "missing or unexpected keys against the target model."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    src_sd = load_panguweather_state_dict(
        args.source, prefer_ema=not args.prefer_model_state
    )
    tgt_sd = translate_state_dict(src_sd)
    logger.info(
        "translated %d tensors from PanguWeather → ai-rossby layout", len(tgt_sd)
    )

    model = build_target_model_from_yaml(args.model_config, args.target_class)
    incoming = model.load_state_dict(tgt_sd, strict=False)
    missing = list(incoming.missing_keys)
    unexpected = list(incoming.unexpected_keys)
    if missing:
        logger.warning("%d missing keys (first 5): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning(
            "%d unexpected keys (first 5): %s", len(unexpected), unexpected[:5]
        )
    if args.strict and (missing or unexpected):
        logger.error("strict mode: refusing to write checkpoint with key mismatches")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
