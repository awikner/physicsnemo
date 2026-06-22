#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compare an ai-rossby translated model's predictions to a stored reference.

This is the **Phase 5 fidelity gate's compute step**: after the sbatch
has produced (a) a translated ``.mdlus`` from a PanguWeather ``.pt``
and (b) a reference NetCDF of the source model's predictions, this
script loads the translated model, runs forward on the same initial
condition(s), and asserts the per-channel-group output matches the
reference within configurable tolerance.

The script is **generic across the three target classes** —
``PanguPlasim``, ``PanguPlasimLegacy``, ``SfnoPlasim`` — selected via
``--target-class``. The same harness handles all three, fed by the
matching translator + model YAML.

Usage
-----

::

    python tools/checkpoint_translation/fidelity_compare.py \\
        --mdlus /path/to/translated.mdlus \\
        --model-config examples/weather/ai_rossby/conf/model/pangu_plasim_legacy.yaml \\
        --target-class PanguPlasimLegacy \\
        --reference /path/to/reference_predictions.nc \\
        --tolerance 1e-4 \\
        [--device cuda:0]

The reference NetCDF must carry per-channel-group fields named
``pred_surface`` and ``pred_upper_air`` (and optionally
``pred_diagnostic``) of shape ``(frame, C, [L,] H, W)``, plus a saved
input pack (the IC and boundary tensors) under group ``input/``. Run
PanguWeather's own ``ensemble_inference.py`` once on the source
checkpoint to produce this reference NetCDF — see
``hpc/scripts/fidelity_pangu_plasim.sbatch`` for the full pipeline.

Exit codes
----------
* ``0`` — all groups within tolerance.
* ``1`` — at least one group exceeded tolerance (max-abs-diff printed
  per group).
* ``2`` — setup error (missing reference, shape mismatch, can't load
  checkpoint).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr
import yaml


logger = logging.getLogger(__name__)


def _build_model(model_yaml: Path, target_class: str):
    """Build a fresh model of the requested class, no weights loaded."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
        )
        from physicsnemo.experimental.models.pangu_plasim import (
            PanguPlasim,
            PanguPlasimLegacy,
        )
        from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim

    classes = {
        "PanguPlasim": PanguPlasim,
        "PanguPlasimLegacy": PanguPlasimLegacy,
        "SfnoPlasim": SfnoPlasim,
    }
    if target_class not in classes:
        raise ValueError(
            f"unknown --target-class={target_class!r}; expected one of "
            f"{sorted(classes)}"
        )
    with open(model_yaml) as fh:
        cfg = yaml.safe_load(fh)
    for k in ("name", "module", "target", "model_type"):
        cfg.pop(k, None)
    return classes[target_class](**cfg)


def _load_mdlus_into(model: torch.nn.Module, mdlus_path: Path, device) -> None:
    """Load the .mdlus checkpoint produced by the translator into ``model``."""
    state = torch.load(str(mdlus_path), map_location=device, weights_only=False)
    # Module.save emits a dict with "model_state_dict" at the top level
    # (per physicsnemo.core.module.Module.save). Be tolerant of older
    # plain-state-dict saves.
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state
    model.load_state_dict(sd, strict=True)


def _load_reference(reference_path: Path) -> xr.Dataset:
    """Open the reference predictions file (NetCDF or Zarr)."""
    if str(reference_path).endswith(".zarr"):
        return xr.open_zarr(reference_path)
    return xr.open_dataset(reference_path)


def _build_inputs_from_reference(
    ref: xr.Dataset, *, frame: int, device
) -> dict:
    r"""Extract the IC + boundary tensors from the reference dataset.

    Expects the reference to carry an ``input/`` group (or top-level
    variables) with at least:

    * ``surface_in``: ``(frame, C_s, H, W)``
    * ``upper_air_in``: ``(frame, C_u, L, H, W)``
    * ``constant_boundary``: ``(C_c, H, W)`` (time-invariant)
    * ``varying_boundary``: ``(frame, C_v, H, W)``
    """
    # Pull the per-frame state at ``frame`` — the reference's IC for the
    # comparison. Some producers write this under a sub-group called
    # ``input``; allow either layout.
    src = ref
    if "input" in ref.groups if hasattr(ref, "groups") else False:
        src = ref["input"]

    def _t(name):
        if name not in src.variables:
            raise KeyError(
                f"reference is missing variable {name!r}; available: "
                f"{list(src.data_vars)}"
            )
        arr = src[name].values
        if name in ("surface_in", "upper_air_in", "varying_boundary"):
            arr = arr[frame]
        return torch.from_numpy(np.asarray(arr)).to(device, dtype=torch.float32).unsqueeze(0)

    out = {
        "surface_in": _t("surface_in"),
        "upper_air_in": _t("upper_air_in"),
        "varying_boundary": _t("varying_boundary"),
        # constant_boundary is time-invariant → no frame index.
        "constant_boundary": torch.from_numpy(
            np.asarray(src["constant_boundary"].values)
        ).to(device, dtype=torch.float32),
    }
    return out


def _diff_summary(name: str, a: torch.Tensor, b: np.ndarray) -> tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff) between two same-shape arrays."""
    if a.shape != tuple(b.shape):
        raise ValueError(
            f"{name}: shape mismatch — model output {tuple(a.shape)} vs "
            f"reference {tuple(b.shape)}"
        )
    a_np = a.cpu().numpy()
    diff = np.abs(a_np - b)
    return float(diff.max()), float(diff.mean())


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mdlus", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument(
        "--target-class",
        type=str,
        required=True,
        choices=("PanguPlasim", "PanguPlasimLegacy", "SfnoPlasim"),
    )
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index in the reference to use as the IC + truth.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max-abs-diff tolerance per channel group.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.mdlus.exists():
        logger.error("--mdlus does not exist: %s", args.mdlus)
        return 2
    if not args.reference.exists():
        logger.error("--reference does not exist: %s", args.reference)
        return 2

    device = torch.device(args.device) if torch.cuda.is_available() or args.device == "cpu" else torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")

    # 1) Build the target model + load the translated weights.
    model = _build_model(args.model_config, args.target_class).to(device).eval()
    _load_mdlus_into(model, args.mdlus, device)
    has_diagnostic = bool(getattr(model, "has_diagnostic", False))
    logger.info(
        "loaded %s from %s on %s (has_diagnostic=%s)",
        args.target_class, args.mdlus, device, has_diagnostic,
    )

    # 2) Read the reference + extract the IC at the requested frame.
    ref = _load_reference(args.reference)
    inputs = _build_inputs_from_reference(ref, frame=args.frame, device=device)
    logger.info(
        "loaded reference %s; comparing frame=%d", args.reference, args.frame
    )

    # 3) Forward pass.
    with torch.no_grad():
        out = model(
            inputs["surface_in"],
            inputs["constant_boundary"],
            inputs["varying_boundary"],
            inputs["upper_air_in"],
        )
        if has_diagnostic:
            out_surface, out_upper, out_diag = out[0], out[1], out[2]
        else:
            out_surface, out_upper = out[0], out[1]
            out_diag = None

    # 4) Compare against the reference predictions for the next frame
    #    (frame+1, since the model is producing the prediction *from*
    #    the IC at ``frame``).
    target_frame = args.frame + 1
    if target_frame >= ref.sizes.get("frame", 0):
        logger.error(
            "reference doesn't have frame %d (max %d)",
            target_frame, ref.sizes.get("frame", 0) - 1,
        )
        return 2

    ref_surface = ref["pred_surface"].values[target_frame]
    ref_upper = ref["pred_upper_air"].values[target_frame]
    ref_diag = (
        ref["pred_diagnostic"].values[target_frame] if "pred_diagnostic" in ref else None
    )

    # Strip the batch axis from the model output to match reference shape.
    out_s = out_surface[0]
    out_u = out_upper[0]
    out_d = out_diag[0] if out_diag is not None else None

    rc = 0
    summary: list[str] = []
    for name, mod, ref_arr in (
        ("surface", out_s, ref_surface),
        ("upper_air", out_u, ref_upper),
    ):
        max_d, mean_d = _diff_summary(name, mod, ref_arr)
        summary.append(f"  {name:>10s}: max_abs_diff={max_d:.4e}  mean_abs_diff={mean_d:.4e}")
        if max_d > args.tolerance:
            rc = 1
    if out_d is not None and ref_diag is not None:
        max_d, mean_d = _diff_summary("diagnostic", out_d, ref_diag)
        summary.append(f"  diagnostic: max_abs_diff={max_d:.4e}  mean_abs_diff={mean_d:.4e}")
        if max_d > args.tolerance:
            rc = 1

    print(f"\nfidelity_compare {args.target_class} (tolerance={args.tolerance:.1e})")
    print("\n".join(summary))
    if rc == 0:
        print("\nPASS — translated model matches reference within tolerance.")
    else:
        print(
            f"\nFAIL — one or more groups exceed tolerance ({args.tolerance:.1e}).\n"
            f"Sources of expected residual: float32 vs bf16 source-side AMP, "
            f"non-deterministic CUDA kernels, EMA-vs-model-state mismatch, "
            f"normalization-stats version skew. Investigate per group."
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
