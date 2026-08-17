#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Ours vs upstream on a REAL batch out of our own data pipeline.

``verify_v2_numerical.py`` compares the two backbones on ``torch.randn``. That
isolates the translation, which is what it is for — but it says nothing about
whether *our config and loader* hand the model the tensors upstream's did. Those
are separate failure modes, and for the v1 SI checkpoints the second one is the
live risk: they are 45x90-state models with 180x360 forcings
(``c_grid_downsample: 4``), a pairing no shipped SI config described until
``amip_si.yaml``.

So this script closes the loop end to end:

1. compose the recipe's own Hydra config (``model=``, ``dataset=``) and build the
   dataset through ``train_diffusion._build_dataset`` — the real NaN-fill,
   normalization and forcing-routing path;
2. build the wrapper from that MODEL CONFIG (not from the checkpoint's stored
   args) and assert its channel contract against the translated artifact, so a
   config that merely *looks* right cannot pass;
3. load the translated ``.mdlus`` into it with ``strict=True``;
4. build upstream's own ``DiT`` from the source checkpoint's ``config.yml`` and
   give it the same weights;
5. pack a real sample with the wrapper's own ``pack_state`` / ``pack_c_grid``,
   forward both, compare per channel block.

A note on what a pass does and does not mean. These checkpoints were trained on
the 6-hourly AMIP archive, not on daily averages, so running them against
``amip_dailyavg_coarse`` is an **implementation A/B** — identical inputs in,
identical outputs out — and not a skill test. The normalization statistics
differ; the forecast is not expected to be meteorologically good, and this
script deliberately reports no error metric against truth.

Usage::

    python tools/checkpoint_translation/verify_si_realdata.py \
        --model amip_si --dataset amip_dailyavg_coarse \
        --translated $CKPT/translated/si_v_42_2026-06-02T20-10-55.mdlus \
        --source $CKPT/SI_v_42_2026-06-02T20-10-55/last.ckpt \
        --amip-repo ~/amip-497827e
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_RECIPE = _REPO / "examples" / "weather" / "ai_rossby"
for p in (str(_REPO), str(_RECIPE), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger("verify_si_realdata")


def _compose(model: str, dataset: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_RECIPE / "conf"), version_base="1.2"):
        return compose(
            config_name="config",
            overrides=[f"model={model}", f"dataset={dataset}", *overrides],
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="conf/model stem")
    p.add_argument("--dataset", required=True, help="conf/dataset stem")
    p.add_argument("--translated", required=True, help="our .mdlus")
    p.add_argument("--source", required=True, help="upstream .ckpt (for its config)")
    p.add_argument("--amip-repo", required=True, help="upstream v1 checkout")
    p.add_argument("--index", type=int, default=0, help="dataset row to pull")
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--override", action="append", default=[],
                   help="extra Hydra override, repeatable")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    repo = Path(args.amip_repo).expanduser().resolve()

    from amip_si import load_lightning_ckpt, pick_source_state_dict  # noqa: E402
    from verify_v2_numerical import (  # noqa: E402
        _state_blocks,
        _strip_model_prefix,
        build_upstream_si,
    )

    cfg = _compose(args.model, args.dataset, args.override)

    # ---- our side: the MODEL CONFIG builds it, then the artifact fills it ----
    # Deliberately not Module.from_checkpoint: that would rebuild from the
    # artifact's own stored args and prove nothing about the config under test.
    from train import build_model  # noqa: E402
    from train_loop import assert_checkpoint_contract  # noqa: E402

    ours = build_model(cfg.model)
    logger.info(
        "config %s -> %s: state %d ch, c_grid %d, scalar %d, grid %s, layout %s",
        args.model, type(ours).__name__, ours.in_channels, ours.c_grid_dim,
        ours.scalar_dim, tuple(ours.horizontal_resolution), ours.channel_layout,
    )
    # The config's contract vs the artifact's: channel_layout, variable lists,
    # levels. A config that is merely plausible fails here.
    assert_checkpoint_contract(ours, args.translated, log=logger)

    sd = _mdlus_sd(args.translated)
    missing, unexpected = ours.load_state_dict(sd, strict=True)
    logger.info("loaded translated artifact strictly (0 missing, 0 unexpected)")
    ours.eval()

    # ---- upstream's side, same weights -----------------------------------
    blob = load_lightning_ckpt(args.source, amip_repo=repo)
    src_state = pick_source_state_dict(blob, prefer_live=False)
    src_cfg = blob["hyper_parameters"]["config"]
    theirs, backbone_cfg = build_upstream_si(src_cfg, repo)
    theirs.load_state_dict(_strip_model_prefix(src_state), strict=True)
    theirs.eval()
    n_t = sum(q.numel() for q in theirs.parameters())
    n_o = sum(q.numel() for q in ours.backbone.parameters())
    logger.info("upstream %.2fM params | ours %.2fM", n_t / 1e6, n_o / 1e6)
    if n_t != n_o:
        logger.error("PARAMETER COUNT DIFFERS")
        return 2

    # ---- a real batch, through our own pipeline --------------------------
    from train_diffusion import _build_dataset  # noqa: E402
    from train_loop import model_step_rows  # noqa: E402

    ds = _build_dataset(cfg)
    stride = model_step_rows(cfg, ds)
    logger.info("dataset %s: %d rows, model step %d row(s)",
                args.dataset, getattr(ds, "n_time", -1), stride)
    sample = ds[(int(args.index), stride)]
    sample = {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
    }
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            logger.debug("  sample[%s] %s", k, tuple(v.shape))

    x = ours.pack_state(sample)
    c_grid = ours.pack_c_grid(sample)
    c_scalar = sample["calendar"]
    logger.info(
        "packed from real data: x %s  c_grid %s  c_scalar %s",
        tuple(x.shape), tuple(c_grid.shape), tuple(c_scalar.shape),
    )
    # The geometry claim this whole config exists to get right: state on the
    # backbone grid, forcings `down x` bigger.
    down = int(backbone_cfg.get("c_grid_downsample", 1) or 1)
    exp_c = (x.shape[-2] * down, x.shape[-1] * down)
    if tuple(c_grid.shape[-2:]) != exp_c:
        logger.error(
            "c_grid is %s but c_grid_downsample=%d implies %s — the dataset "
            "pairing does not match the model's forcing geometry",
            tuple(c_grid.shape[-2:]), down, exp_c,
        )
        return 2
    logger.info("c_grid resolution agrees with c_grid_downsample=%d", down)

    # A single-step wrapper takes (x_noised, cond, t). Feeding the same field as
    # both is fine here: this compares implementations, not a sampler.
    t = torch.zeros(x.shape[0], 1)
    with torch.no_grad():
        out_o = ours(x, x, t, c_grid=c_grid, c_scalar=c_scalar)
        out_t = theirs(x, x, t, c_grid=c_grid, c_scalar=c_scalar)
    if isinstance(out_o, tuple):
        out_o = out_o[0]
    if isinstance(out_t, tuple):
        out_t = out_t[0]

    if out_t.shape != out_o.shape:
        logger.error("OUTPUT SHAPE DIFFERS: %s vs %s",
                     tuple(out_t.shape), tuple(out_o.shape))
        return 2
    diff = (out_t - out_o).abs()
    logger.info("output %s | upstream |max| %.4e", tuple(out_t.shape),
                out_t.abs().max().item())
    logger.info("max |diff| %.4e   mean |diff| %.4e",
                diff.max().item(), diff.mean().item())
    for name, lo, hi in _state_blocks(
        ours.num_surface, ours.num_diagnostic, ours.num_levels,
        ours.num_upper_air_vars, 0,
    ):
        logger.info("  %-11s channels %3d:%-3d  max |diff| %.4e",
                    name, lo, hi, diff.narrow(1, lo, hi - lo).max().item())

    ok = diff.max().item() <= args.tol
    print(f"\n{'PASS' if ok else 'FAIL'}: real-data max |diff| "
          f"{diff.max().item():.4e} (tol {args.tol:.1e})")
    return 0 if ok else 1


def _mdlus_sd(path):
    """State dict out of a ``.mdlus``, wrapper-prefixed keys intact."""
    from train_loop import _mdlus_state_dict

    return _mdlus_state_dict(Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
