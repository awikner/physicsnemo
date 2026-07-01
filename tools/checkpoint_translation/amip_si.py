#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Translate upstream amip Lightning ``.ckpt`` blobs to ``.mdlus``.

Upstream amip (``/work/nvme/bdiu/awikner/amip``, commit ``497827e``)
trains the SI / SI_X / ERDM / RFM / EDM diffusion families with
PyTorch Lightning. Their checkpoint payload follows the standard
Lightning layout *plus* a ``WeightAveraging`` callback that splits the
weights into two parallel state dicts:

.. code-block:: text

    {
        "state_dict":            EMA-averaged weights (inference-preferred)
        "current_model_state":   live (training) weights (resume only)
        "averaging_state":       {"n_averaged": N}
        "hyper_parameters":      {"config": {"model": {...}, "data": {...}, "training": {...}}}
        "epoch", "global_step", "optimizer_states", "lr_schedulers", ...
    }

This translator does the round-trip from that layout to a single
PhysicsNeMo wrapper checkpoint (``.mdlus``):

1. Load the Lightning ``.ckpt`` (``weights_only=False`` since the
   pickle still references upstream amip's pickled ``GetDataset``
   normalizer — the translator only needs a sys.path entry to
   ``/work/nvme/bdiu/awikner/amip``).
2. Read ``hyper_parameters.config.model.model_name`` and cross-check
   it against the supplied wrapper YAML's class name.
3. Pick the source state dict — ``state_dict`` (EMA-averaged, default)
   or ``current_model_state`` (live). Drop ``scheduler.*`` keys with
   a warning; the scheduler is rebuilt from ``cfg.loss`` /
   ``cfg.sampler`` at train / inference time.
4. Re-prefix the remaining keys — upstream uses ``model.X``, our
   wrappers expose the backbone at ``self.backbone``, so the keys
   transform as ``model.X`` → ``backbone.X``.
5. Instantiate a fresh :class:`AmipDiTWrapper` / :class:`RollingDiTWrapper`
   / :class:`ERDMWrapper` from the wrapper YAML, load the translated
   state dict via :meth:`load_state_dict`, report missing /
   unexpected keys, and save via :meth:`Module.save`.

Supported source ``model_name`` values:

* ``SI`` → :class:`AmipDiTWrapper` (paired training scheduler:
  :class:`DriftScheduler`)
* ``SI_X`` → :class:`AmipDiTWrapper`
  (:class:`DynamicInterpolant`)
* ``ERDM`` → :class:`ERDMWrapper` (:class:`ERDMScheduler`)
* ``RFM`` → :class:`RollingDiTWrapper` (:class:`RFMScheduler`)
* ``EDM`` → :class:`AmipDiTWrapper` (no Phase 8c training wiring;
  inference-only; the translator still produces a valid ``.mdlus``
  that can be sampled with ``sampler=edm``)
* ``x_DDC`` → :class:`XDDCWrapper` (:class:`DataDependentInterpolant`;
  Phase 8f — only the ``decoder_type: unet`` denoiser is vendored,
  ``decoder_type: dit`` raises ``NotImplementedError``)

Not supported:

* ``Combined`` (two-stage forecaster + downscaler) — has no
  standalone checkpoint to translate; compose it at runtime from an
  independently-translated forecaster + x_DDC pair via
  :class:`~physicsnemo.experimental.models.amip_si.wrappers.CombinedModule`.

Usage
-----

::

    python tools/checkpoint_translation/amip_si.py \\
        --source /work/nvme/bdiu/awikner/amip-checkpoints/AMIP_logs/SI_X_AIMIP_interp_gaussian_42_2026-05-28T09-27-49/last.ckpt \\
        --model-config examples/weather/ai_rossby/conf/model/amip_si_x.yaml \\
        --output /work/nvme/bdiu/awikner/checkpoints/amip/si_x_aimip_interp_gaussian.mdlus
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


# Source LightningModule attribute layout: ``self.model = AmipDiT(...)``
# at the top of upstream amip's ``modules/diffusion_module.py`` etc., so
# every backbone tensor's key begins with ``model.``.
_SOURCE_BACKBONE_PREFIX = "model."

# Target wrapper layout: ``self.backbone = AmipDiT(...)`` /
# ``RollingDiT(...)`` / ``ERDM(...)`` — see wrappers.py L198 etc.
_TARGET_BACKBONE_PREFIX = "backbone."

# Other key prefixes the upstream Lightning module saves alongside the
# backbone weights. They don't map onto our wrapper (the wrapper holds
# only the backbone — schedulers are rebuilt from yaml) so the
# translator drops them with a count, never silently.
_DROPPED_SOURCE_PREFIXES = ("scheduler.",)

# Common torch.compile / DDP wrapping prefixes; stripped iteratively so
# stacked combinations like ``module._orig_mod.model.X`` collapse to
# ``model.X`` before the backbone re-prefix runs.
_WRAP_PREFIXES = ("module.", "_orig_mod.")

# ``varying_boundary_variables`` names upstream amip's ``wCO2`` variants
# route through the c_scalar path rather than c_grid (see the trim
# heuristic in wrapper_kwargs_from_hparams). Matched case-insensitively,
# by exact name or prefix (covers e.g. "co2", "co2_anomaly").
_SCALAR_ROUTED_VARYING_BOUNDARY_NAMES = ("global_mean_co2", "co2")


def _is_scalar_routed_varying_boundary_name(name: str) -> bool:
    name_l = name.lower()
    return any(
        name_l == pat or name_l.startswith(pat)
        for pat in _SCALAR_ROUTED_VARYING_BOUNDARY_NAMES
    )


# Mapping from upstream ``hyper_parameters.config.model.model_name`` to
# our wrapper class name (resolved against the wrapper YAML below).
_MODEL_NAME_TO_WRAPPER = {
    "SI": "AmipDiTWrapper",
    "SI_X": "AmipDiTWrapper",
    "EDM": "AmipDiTWrapper",
    "ERDM": "ERDMWrapper",
    "RFM": "RollingDiTWrapper",
    "x_DDC": "XDDCWrapper",
}

# Source ``model_name`` values explicitly out of Phase 8 scope. The
# translator surfaces a clear error rather than producing junk.
#
# ``Combined`` has no standalone Lightning checkpoint to translate —
# upstream composes it at runtime from an independently-trained
# forecaster checkpoint + an independently-trained x_DDC checkpoint
# (see ``configs/combined_midway.yaml``). Translate each of those two
# checkpoints separately (forecaster via the normal SI/SI_X/ERDM/RFM
# path, downscaler via the ``x_DDC`` path above), then compose them at
# runtime with :class:`~physicsnemo.experimental.models.amip_si.wrappers.CombinedModule`
# — there is nothing for this translator to do for "Combined".
_UNSUPPORTED_MODEL_NAMES = ("Combined",)


def _strip_wrap_prefixes(key: str) -> str:
    """Strip iterated DDP / torch.compile wrapper prefixes from a key.

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


def load_lightning_ckpt(
    source: Path, *, amip_repo: Optional[Path] = None
) -> dict:
    """Load a Lightning ``.ckpt`` and return the full top-level dict.

    Upstream amip pickles its data-side normalizer (``GetDataset`` from
    ``data.amip_new``) inside the ckpt's ``hyper_parameters`` block, so
    Python needs ``amip_repo`` on ``sys.path`` before
    :func:`torch.load` unpickles. Pass the upstream repo root via
    ``amip_repo`` (defaults to ``/work/nvme/bdiu/awikner/amip``, which
    is where it lives on Delta — override only if the upstream tree
    moved).
    """
    repo = Path(amip_repo) if amip_repo else Path("/work/nvme/bdiu/awikner/amip")
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    blob = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(
            f"expected a Lightning checkpoint dict at {source}, got "
            f"{type(blob).__name__}"
        )
    return blob


def pick_source_state_dict(
    blob: dict, *, prefer_live: bool = False
) -> OrderedDict:
    """Pick ``state_dict`` (EMA-averaged) or ``current_model_state`` (live).

    Upstream amip's ``WeightAveraging`` Lightning callback writes two
    parallel state dicts to the checkpoint:

    * ``state_dict`` — the EMA-averaged weights. This is what inference
      should use; it's the translator's default.
    * ``current_model_state`` — the live (non-averaged) weights for
      training resume. Pick this when comparing pre-EMA training
      trajectories or when EMA wasn't run long enough to be useful
      (``averaging_state.n_averaged`` low).
    """
    if prefer_live:
        if "current_model_state" not in blob:
            raise KeyError(
                "checkpoint has no 'current_model_state' key — pass "
                "without --prefer-live to use 'state_dict' (EMA-averaged) instead"
            )
        sd = blob["current_model_state"]
        logger.info("using current_model_state (live weights, pre-EMA)")
    else:
        if "state_dict" not in blob:
            raise KeyError(
                "checkpoint has no 'state_dict' key; try --prefer-live to "
                "fall back to 'current_model_state'"
            )
        sd = blob["state_dict"]
        n_avg = blob.get("averaging_state", {}).get("n_averaged", None)
        logger.info(
            "using state_dict (EMA-averaged weights; n_averaged=%s)", n_avg
        )
    return OrderedDict(sd)


def translate_state_dict(
    src_sd: OrderedDict,
) -> tuple[OrderedDict, dict[str, int]]:
    """Re-prefix backbone keys and drop scheduler buffers.

    Returns ``(translated_sd, stats)`` where ``stats`` is a dict of
    ``{"kept": N, "dropped_scheduler": N, "dropped_unknown": N}`` for
    the caller to log.
    """
    out: OrderedDict = OrderedDict()
    stats = {"kept": 0, "dropped_scheduler": 0, "dropped_unknown": 0}
    for k, v in src_sd.items():
        peeled = _strip_wrap_prefixes(k)
        if peeled.startswith(_DROPPED_SOURCE_PREFIXES):
            stats["dropped_scheduler"] += 1
            continue
        if not peeled.startswith(_SOURCE_BACKBONE_PREFIX):
            # Some Lightning runs add extra registered buffers we
            # don't recognize. Surface the count without crashing so
            # the user can sanity-check what got dropped.
            stats["dropped_unknown"] += 1
            logger.debug("dropping unknown source key: %s", k)
            continue
        new_key = _TARGET_BACKBONE_PREFIX + peeled[len(_SOURCE_BACKBONE_PREFIX) :]
        out[new_key] = v
        stats["kept"] += 1
    return out, stats


def detect_model_name(blob: dict) -> str:
    """Read the source ``model_name`` from the Lightning hparams blob.

    Errors with a helpful message when the field is missing or set to
    one of the explicitly-unsupported model families.
    """
    hp = blob.get("hyper_parameters", None)
    if hp is None:
        raise KeyError(
            "checkpoint has no 'hyper_parameters' block — cannot detect "
            "source model_name automatically; pass --model-name to "
            "override"
        )
    cfg_model = hp.get("config", {}).get("model", {})
    name = cfg_model.get("model_name", None)
    if name is None:
        raise KeyError(
            "hyper_parameters.config.model.model_name missing — pass "
            "--model-name to override"
        )
    if name in _UNSUPPORTED_MODEL_NAMES:
        raise NotImplementedError(
            f"source model_name={name!r} is not supported by this "
            f"translator — see phase8e_midway3_checkpoint_inventory.md "
            f"for why (Combined has no standalone checkpoint; compose "
            f"it at runtime from a translated forecaster + x_DDC pair)."
        )
    if name not in _MODEL_NAME_TO_WRAPPER:
        raise ValueError(
            f"unrecognized source model_name={name!r}; expected one of "
            f"{sorted(_MODEL_NAME_TO_WRAPPER)}"
        )
    return name


def _resolve_target_wrapper(yaml_cfg: dict, cli_class: Optional[str]) -> str:
    """Pick the target wrapper class name from CLI or YAML's ``name`` field."""
    if cli_class:
        return cli_class
    name = str(yaml_cfg.get("name", "")).strip()
    if name in _MODEL_NAME_TO_WRAPPER.values():
        return name
    raise ValueError(
        f"could not determine target wrapper class from YAML (name={name!r}); "
        f"pass --target-class explicitly. Valid options: "
        f"{sorted(set(_MODEL_NAME_TO_WRAPPER.values()))}"
    )


# Backbone kwargs the wrapper derives itself from the channel-group
# kwargs. Stripping them up front prevents subtle silent mismatches
# from user-edited hparams blobs.
#
# Note ``nlat`` / ``nlon`` are *not* stripped — upstream amip's legacy
# ckpts use a backbone working resolution (e.g. 45×90) that's smaller
# than the data resolution (180×360); the c_grid stream comes in at
# data resolution and is downsampled by ``c_grid_downsample`` to match
# the backbone. Preserving the source ``nlat`` / ``nlon`` keeps that
# two-resolution pipeline intact.
_BACKBONE_AUTO_KEYS = ("in_channels", "out_channels", "c_grid_dim")


# Per-wrapper kwarg name that holds the backbone constructor kwargs.
_WRAPPER_BACKBONE_KWARGS_KEY = {
    "AmipDiTWrapper": "dit_kwargs",
    "RollingDiTWrapper": "rolling_dit_kwargs",
    "ERDMWrapper": "erdm_kwargs",
    "XDDCWrapper": "unet_kwargs",
}


def _xddc_wrapper_kwargs_from_hparams(blob: dict) -> dict:
    """Extract :class:`XDDCWrapper` constructor kwargs from a Lightning hparams blob.

    x_DDC's hparams layout differs structurally from the SI/SI_X/ERDM/RFM
    family: backbone kwargs live under ``model.x_DDC.{encoder,decoder,dit}``
    (dispatched on ``model.x_DDC.decoder_type``), not
    ``model.x_DDC.model`` — and there's no c_grid/c_scalar conditioning,
    so no constant/varying boundary reconciliation or scalar_dim
    extraction applies here. See upstream ``modules/ae_module.py``
    (``AutoencoderModule.__init__``) for the source layout this mirrors.
    """
    hp = blob.get("hyper_parameters", None)
    if hp is None:
        raise KeyError(
            "checkpoint has no 'hyper_parameters' block — cannot auto-derive "
            "wrapper kwargs; pass --model-config explicitly"
        )
    cfg = hp.get("config", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    xddc_cfg = model.get("x_DDC", None)
    if xddc_cfg is None:
        raise KeyError(
            "hyper_parameters.config.model is missing the 'x_DDC' "
            "sub-block; cannot auto-derive backbone kwargs"
        )

    decoder_type = xddc_cfg.get("decoder_type", "unet")
    if decoder_type != "unet":
        raise NotImplementedError(
            f"x_DDC decoder_type={decoder_type!r} is not supported by "
            "this translator — only the convolutional UNet denoiser is "
            "vendored (XDDCUNet). The DiT autoencoder denoiser "
            "(decoder_type='dit', modules/models/DiTAE.py) is deferred; "
            "see phase8e_midway3_checkpoint_inventory.md."
        )
    unet_cfg = dict(xddc_cfg.get("decoder", {}))
    for k in _BACKBONE_AUTO_KEYS:
        unet_cfg.pop(k, None)

    encoder_cfg = dict(xddc_cfg.get("encoder", {}))
    downsample_factor = int(encoder_cfg.get("downsample_factor", 4))

    return {
        "surface_variables": list(data.get("surface_variables", [])),
        "upper_air_variables": list(data.get("upper_air_variables", [])),
        "diagnostic_variables": list(data.get("diagnostic_variables", [])),
        "levels": list(data.get("levels", [])),
        "horizontal_resolution": list(data.get("horizontal_resolution", [])),
        "downsample_factor": downsample_factor,
        "unet_kwargs": unet_cfg,
    }


def wrapper_kwargs_from_hparams(
    blob: dict, target_class_name: str, *, source_model_name: Optional[str] = None
) -> dict:
    """Extract wrapper constructor kwargs from a Lightning hparams blob.

    The upstream amip ckpt's ``hyper_parameters`` block carries both
    the data layout (variable lists, levels, resolution) and the
    backbone kwargs (DiT dim, num_blocks, etc.) used at training time.
    This helper bridges them into the wrapper's signature so the
    translated state dict loads cleanly against a *matching-shape*
    wrapper — no need to maintain hand-written legacy YAMLs per
    checkpoint variant.

    ``source_model_name`` is read from the blob when omitted; pass it
    explicitly when the hparams blob doesn't carry one.

    Mirrors the constructor signature of
    :class:`AmipDiTWrapper` / :class:`RollingDiTWrapper` /
    :class:`ERDMWrapper` / :class:`XDDCWrapper`.

    x_DDC's hparams layout differs structurally from the rest (no
    ``model.x_DDC.model`` sub-block, no c_grid/c_scalar) — dispatched
    to :func:`_xddc_wrapper_kwargs_from_hparams` instead of the generic
    logic below.
    """
    if target_class_name == "XDDCWrapper":
        return _xddc_wrapper_kwargs_from_hparams(blob)

    hp = blob.get("hyper_parameters", None)
    if hp is None:
        raise KeyError(
            "checkpoint has no 'hyper_parameters' block — cannot auto-derive "
            "wrapper kwargs; pass --model-config explicitly"
        )
    cfg = hp.get("config", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})

    if source_model_name is None:
        source_model_name = model.get("model_name", None)
    if source_model_name is None or source_model_name not in model:
        raise KeyError(
            f"hyper_parameters.config.model is missing the {source_model_name!r} "
            "sub-block; cannot auto-derive backbone kwargs"
        )
    source_backbone_cfg = dict(model[source_model_name].get("model", {}))
    backbone_cfg = dict(source_backbone_cfg)
    for k in _BACKBONE_AUTO_KEYS:
        backbone_cfg.pop(k, None)

    # Diagnostic channels are only consumed by SI-family models when
    # ``diagnostic_input: True`` (the SI-family "predicts diagnostics
    # too" flag). The flag's absence defaults to True per upstream's
    # config convention.
    diagnostic_input = bool(data.get("diagnostic_input", True))
    diagnostic_variables = (
        list(data.get("diagnostic_variables", []))
        if diagnostic_input
        else []
    )

    # Reconcile the data-side variable lists with the model-side
    # channel counts. Upstream amip's ``wCO2`` variants list
    # ``global_mean_co2`` under ``varying_boundary_variables`` but
    # route it through a separate scalar path, so the model's actual
    # ``c_grid_dim`` is one less than ``len(const + varying)``. We
    # trust the model-side count (it's what determines the on-disk
    # weight shape) and truncate the varying-boundary list to match,
    # logging which entries we dropped so the user can sanity-check.
    const_boundary = list(data.get("constant_boundary_variables", []))
    varying_boundary = list(data.get("varying_boundary_variables", []))
    target_c_grid_dim = source_backbone_cfg.get("c_grid_dim", None)
    if target_c_grid_dim is not None:
        target_c_grid_dim = int(target_c_grid_dim)
        n_const = len(const_boundary)
        wanted_varying = target_c_grid_dim - n_const
        if wanted_varying < 0:
            raise ValueError(
                f"hparams c_grid_dim={target_c_grid_dim} is smaller than "
                f"the listed constant_boundary count {n_const}; cannot "
                f"reconcile"
            )
        if wanted_varying < len(varying_boundary):
            n_drop = len(varying_boundary) - wanted_varying
            # Prefer dropping name-matched scalar-routed channels (e.g.
            # ``global_mean_co2``, ``co2*``) over trailing entries — the
            # log message then stays truthful about *which* channel got
            # dropped instead of just assuming it's the last one(s).
            # Falls back to trailing entries when there aren't enough
            # name matches (preserves prior behavior for non-wCO2 ckpts).
            scalar_routed = [
                v for v in varying_boundary
                if _is_scalar_routed_varying_boundary_name(v)
            ]
            dropped = scalar_routed[:n_drop]
            if len(dropped) < n_drop:
                remaining = [v for v in varying_boundary if v not in dropped]
                dropped = dropped + remaining[-(n_drop - len(dropped)):]
            logger.warning(
                "trimming varying_boundary_variables to match "
                "c_grid_dim=%d (dropped %d entries: %s); upstream amip "
                "likely routed these via the c_scalar path",
                target_c_grid_dim,
                len(dropped),
                dropped,
            )
            varying_boundary = [v for v in varying_boundary if v not in dropped]
        elif wanted_varying > len(varying_boundary):
            raise ValueError(
                f"hparams c_grid_dim={target_c_grid_dim} requires "
                f"{wanted_varying} varying_boundary entries but only "
                f"{len(varying_boundary)} are listed in data config"
            )

    # Wrapper.horizontal_resolution is *metadata* in the wrapper — it's
    # only used to setdefault dit_kwargs.nlat / nlon. Since we preserve
    # the source's nlat / nlon in backbone_cfg explicitly, the
    # horizontal_resolution we report here is the source's data-side
    # resolution, which matches what callers should be feeding for
    # c_grid (the boundary stream) in the legacy two-resolution layout.
    backbone_kwargs_name = _WRAPPER_BACKBONE_KWARGS_KEY[target_class_name]
    return {
        "surface_variables": list(data.get("surface_variables", [])),
        "upper_air_variables": list(data.get("upper_air_variables", [])),
        "diagnostic_variables": diagnostic_variables,
        "constant_boundary_variables": const_boundary,
        "varying_boundary_variables": varying_boundary,
        "levels": list(data.get("levels", [])),
        "horizontal_resolution": list(data.get("horizontal_resolution", [])),
        "scalar_dim": int(backbone_cfg.pop("scalar_dim", 2)),
        backbone_kwargs_name: backbone_cfg,
    }


def _filter_unknown_backbone_kwargs(
    backbone_kwargs: dict, backbone_cls
) -> dict:
    """Drop backbone kwargs the vendored class doesn't accept.

    Upstream amip's training configs accumulate extra knobs over time
    (e.g. ``unpatch: vanilla`` in late-May checkpoints) that haven't
    landed in our vendored backbone. Silently dropping them is safer
    than letting :class:`AmipDiT` raise ``TypeError`` — the dropped
    keys are listed in a warning so the user can audit.
    """
    import inspect

    sig = inspect.signature(backbone_cls.__init__)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return backbone_kwargs
    accepted = set(sig.parameters)
    kept = {k: v for k, v in backbone_kwargs.items() if k in accepted}
    dropped = {k: v for k, v in backbone_kwargs.items() if k not in accepted}
    if dropped:
        logger.warning(
            "dropping %d backbone kwarg(s) the vendored %s doesn't accept: %s",
            len(dropped),
            backbone_cls.__name__,
            sorted(dropped),
        )
    return kept


def build_target_wrapper(
    *,
    model_yaml: Optional[Path] = None,
    blob: Optional[dict] = None,
    target_class: Optional[str] = None,
    source_model_name: Optional[str] = None,
):
    """Instantiate a fresh wrapper.

    Two paths:

    * ``model_yaml`` is given → build from the Hydra YAML (legacy /
      explicit path, mirrors the deterministic translator).
    * ``model_yaml`` is ``None`` and ``blob`` is given → auto-derive
      the wrapper kwargs from the ckpt's ``hyper_parameters`` block.
      This is the recommended path for translating upstream amip
      ckpts whose channel layout differs from the in-repo wrapper
      defaults.
    """
    import warnings

    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.models.amip_si import (
        AmipDiTWrapper,
        ERDMWrapper,
        RollingDiTWrapper,
        XDDCWrapper,
    )

    classes = {
        "AmipDiTWrapper": AmipDiTWrapper,
        "RollingDiTWrapper": RollingDiTWrapper,
        "ERDMWrapper": ERDMWrapper,
        "XDDCWrapper": XDDCWrapper,
    }

    if model_yaml is not None:
        with open(model_yaml) as fh:
            cfg = yaml.safe_load(fh)
        cls_name = _resolve_target_wrapper(cfg, target_class)
        for k in ("name", "module", "target", "model_type"):
            cfg.pop(k, None)
        return classes[cls_name](**cfg)

    if blob is None:
        raise ValueError(
            "build_target_wrapper requires either model_yaml or blob; got neither"
        )
    if target_class is None:
        if source_model_name is None:
            source_model_name = detect_model_name(blob)
        target_class = _MODEL_NAME_TO_WRAPPER[source_model_name]
    if target_class not in classes:
        raise ValueError(
            f"unknown target wrapper class {target_class!r}; expected one of "
            f"{sorted(classes)}"
        )
    kwargs = wrapper_kwargs_from_hparams(
        blob, target_class, source_model_name=source_model_name
    )
    # Filter any backbone kwargs the vendored class doesn't accept (e.g.
    # late-May ``unpatch: vanilla`` knobs in SI_V_new). We need to know
    # the backbone class for each wrapper to introspect it; the mapping
    # mirrors the wrapper init signatures.
    from physicsnemo.experimental.models.amip_si.dit import AmipDiT
    from physicsnemo.experimental.models.amip_si.erdm_unet import ERDM
    from physicsnemo.experimental.models.amip_si.rolling_dit import RollingDiT
    from physicsnemo.experimental.models.amip_si.x_ddc import XDDCUNet

    backbone_cls = {
        "AmipDiTWrapper": AmipDiT,
        "RollingDiTWrapper": RollingDiT,
        "ERDMWrapper": ERDM,
        "XDDCWrapper": XDDCUNet,
    }[target_class]
    backbone_kwargs_name = _WRAPPER_BACKBONE_KWARGS_KEY[target_class]
    kwargs[backbone_kwargs_name] = _filter_unknown_backbone_kwargs(
        kwargs[backbone_kwargs_name], backbone_cls
    )
    return classes[target_class](**kwargs)


def cross_check_compatibility(
    source_model_name: str, target_class_name: str
) -> None:
    """Verify the source ``model_name`` is compatible with the target wrapper.

    The mapping is many-to-one (e.g. both ``SI`` and ``SI_X`` use
    :class:`AmipDiTWrapper`), so this is *informational* — it warns
    when the ckpt's training family doesn't match the wrapper but
    doesn't refuse to translate. Catches obvious YAML mix-ups
    (translating an ERDM ckpt against an SI wrapper, for example).
    """
    expected = _MODEL_NAME_TO_WRAPPER.get(source_model_name)
    if expected is None:
        # Already handled in detect_model_name; defensive.
        return
    if expected != target_class_name:
        logger.warning(
            "source model_name=%r expects target wrapper %r but YAML "
            "resolves to %r — translation may produce missing/unexpected "
            "keys; pass --no-cross-check to silence this",
            source_model_name,
            expected,
            target_class_name,
        )


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Upstream amip Lightning .ckpt file.",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=(
            "Hydra YAML describing the wrapper constructor kwargs "
            "(e.g. examples/weather/ai_rossby/conf/model/amip_si_x.yaml). "
            "Optional — when omitted, the translator auto-derives the "
            "wrapper kwargs from the ckpt's ``hyper_parameters`` block, "
            "which is the recommended path for upstream amip ckpts whose "
            "channel layout doesn't match the in-repo wrapper defaults."
        ),
    )
    p.add_argument("--output", type=Path, required=True, help="Output .mdlus path.")
    p.add_argument(
        "--target-class",
        type=str,
        default=None,
        choices=sorted(set(_MODEL_NAME_TO_WRAPPER.values())),
        help=(
            "Target wrapper class. When omitted, read from the YAML's "
            "``name`` field."
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Override the source ``model_name`` (skips ckpt hparams "
            "detection). Use when the ckpt's hyper_parameters block is "
            "missing or wrong."
        ),
    )
    p.add_argument(
        "--prefer-live",
        action="store_true",
        help=(
            "Use ``current_model_state`` (live training weights) instead "
            "of the default ``state_dict`` (EMA-averaged)."
        ),
    )
    p.add_argument(
        "--amip-repo",
        type=Path,
        default=None,
        help=(
            "Path to the upstream amip repo (used to unpickle the data "
            "normalizer). Defaults to /work/nvme/bdiu/awikner/amip."
        ),
    )
    p.add_argument(
        "--no-cross-check",
        action="store_true",
        help="Skip the source/target wrapper-family compatibility warning.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Refuse to write the output if the translated state dict has "
            "any missing or unexpected keys against the target wrapper."
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

    blob = load_lightning_ckpt(args.source, amip_repo=args.amip_repo)
    epoch = blob.get("epoch", "?")
    global_step = blob.get("global_step", "?")
    logger.info(
        "loaded Lightning ckpt %s (epoch=%s, global_step=%s)",
        args.source,
        epoch,
        global_step,
    )

    src_model_name = args.model_name or detect_model_name(blob)
    logger.info("source model_name=%s", src_model_name)

    if args.model_config is not None:
        model = build_target_wrapper(
            model_yaml=args.model_config, target_class=args.target_class
        )
        config_source = f"yaml ({args.model_config})"
    else:
        model = build_target_wrapper(
            blob=blob,
            target_class=args.target_class,
            source_model_name=src_model_name,
        )
        config_source = "ckpt hyper_parameters (auto-derived)"
    tgt_class_name = type(model).__name__
    logger.info("target wrapper class=%s (from %s)", tgt_class_name, config_source)
    if not args.no_cross_check:
        cross_check_compatibility(src_model_name, tgt_class_name)

    src_sd = pick_source_state_dict(blob, prefer_live=args.prefer_live)
    tgt_sd, stats = translate_state_dict(src_sd)
    logger.info(
        "translated %d tensors → wrapper layout (kept=%d, dropped_scheduler=%d, "
        "dropped_unknown=%d)",
        stats["kept"],
        stats["kept"],
        stats["dropped_scheduler"],
        stats["dropped_unknown"],
    )

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
