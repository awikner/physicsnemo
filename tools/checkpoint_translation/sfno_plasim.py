#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate PanguWeather v2.0 SFNO_PLASIM ``.pt`` checkpoints to ``.mdlus``.

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

The SFNO model is the ``SphericalFourierNeuralOperatorNet_v2`` wrapper around
the base ``SphericalFourierNeuralOperatorNet``; PanguWeather's
``model.state_dict()`` keys are direct (``encoder.0.weight``, etc.). The
ai-rossby :class:`SfnoPlasim` wrapper holds the base SFNO under
``self.sfno``, so all keys are prefixed with ``sfno.`` after translation.

Strategy:

1. Load the source ``.pt``.
2. Prefer ``ema_state`` (PanguWeather's documented inference-time preference);
   fall back to ``model_state``.
3. Strip the ``module.`` prefix (DDP-wrapped checkpoints) and re-prefix with
   ``sfno.``.
4. Instantiate a fresh :class:`SfnoPlasim` from CLI-supplied model config
   (or a YAML); load the translated state dict; verify no missing/unexpected
   keys; save via :meth:`Module.save`.

Usage
-----

::

    python tools/checkpoint_translation/sfno_plasim.py \\
        --source /path/to/panguweather_sfno_checkpoint.pt \\
        --model-config examples/weather/ai_rossby/conf/model/sfno_plasim.yaml \\
        --output /path/to/sfno_plasim.mdlus
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


def load_panguweather_state_dict(
    source: Path, prefer_ema: bool = True
) -> OrderedDict:
    """Load PanguWeather's .pt and return the (preferred) raw state dict.

    Parameters
    ----------
    source : Path
        PanguWeather checkpoint file.
    prefer_ema : bool, optional, default=True
        Whether to return ``ema_state`` over ``model_state`` when both exist.
    """
    blob = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(
            f"{source}: expected dict checkpoint, got {type(blob).__name__}"
        )
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
        # Some checkpoints save the state dict directly without a wrapping dict.
        sd = blob
        logger.info("treating %s as a raw state dict", source)
    return OrderedDict(sd)


def translate_state_dict(panguweather_sd: OrderedDict) -> OrderedDict:
    """Strip ``module.`` and re-prefix with ``sfno.`` for the ai-rossby wrapper.

    PanguWeather's SFNO_v2 inherits directly from the base SFNO, so its
    ``state_dict`` keys are the base's keys. Our :class:`SfnoPlasim` holds the
    base under ``self.sfno``, so we just prefix every key with ``sfno.``.
    """
    out: OrderedDict = OrderedDict()
    for k, v in panguweather_sd.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[f"sfno.{k}"] = v
    return out


def build_target_model_from_yaml(model_yaml: Path):
    """Instantiate a fresh :class:`SfnoPlasim` from a Hydra-style YAML."""
    # Suppress the experimental-namespace warning during the heavy import.
    import warnings

    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim

    with open(model_yaml) as fh:
        cfg = yaml.safe_load(fh)

    # Strip Hydra-only metadata keys.
    for k in ("name", "target", "model_type"):
        cfg.pop(k, None)
    return SfnoPlasim(**cfg)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, required=True, help="PanguWeather .pt checkpoint.")
    p.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Hydra YAML describing the SfnoPlasim constructor kwargs (e.g. "
        "examples/weather/ai_rossby/conf/model/sfno_plasim.yaml).",
    )
    p.add_argument("--output", type=Path, required=True, help="Output .mdlus path.")
    p.add_argument(
        "--prefer-model-state",
        action="store_true",
        help="Use model_state instead of ema_state even when both exist.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Refuse to write the output if the translated state dict has any "
        "missing or unexpected keys against the target model.",
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
    logger.info("translated %d tensors from PanguWeather → ai-rossby layout", len(tgt_sd))

    model = build_target_model_from_yaml(args.model_config)
    incoming = model.load_state_dict(tgt_sd, strict=False)
    missing = list(incoming.missing_keys)
    unexpected = list(incoming.unexpected_keys)
    if missing:
        logger.warning(
            "%d missing keys (first 5): %s", len(missing), missing[:5]
        )
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
