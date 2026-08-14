#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Numerical equivalence: upstream amip_v2's forward vs our translated wrapper.

Phase 12h. Key parity — every source key landing with a matching shape — is
necessary but **not sufficient**: it cannot catch a channel permutation, and every
contract bug in this phase was shape-preserving (the Phase-8e channel scramble,
the forcing-lag shift, the sea-ice drop). The only check that catches those is
running both implementations on identical inputs and comparing outputs.

This is cheap to do properly because upstream's backbones import almost nothing:
``modules/models/RollingDiT.py`` and ``modules/models/DiTAE.py`` need only torch,
einops and their own ``modules/layers/*`` — no Lightning, no config machinery, no
``norm_stats`` artifacts. So both models are built in ONE process, handed the SAME
weights out of the same checkpoint, and called on the same seeded tensors.

What is compared is the **network forward**, deliberately, not a sampled rollout:
the sampler draws noise, so a rollout could only be compared by also reproducing
their RNG stream. The forward is where the translation lives — weights, channel
order, module wiring — and it is deterministic.

Usage::

    python tools/checkpoint_translation/verify_v2_numerical.py \
        --source $CKPT/ERDM_fancy_.../last.ckpt \
        --amip-v2-repo ~/amip_v2 --family erdm

    # harness self-test with random weights, no 5 GB blob needed
    python tools/checkpoint_translation/verify_v2_numerical.py \
        --amip-v2-repo ~/amip_v2 --family erdm --synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amip_si import (  # noqa: E402
    load_lightning_ckpt,
    pick_source_state_dict,
    translate_state_dict,
    wrapper_kwargs_from_hparams,
)

logger = logging.getLogger("verify_v2")

#: Channel-group boundaries for the 151-channel AMIP state, in v2 pack order
#: ([surface | diagnostic | upper_air]) plus the Phase-12f ocean tail. Used to
#: localise a mismatch: a permutation inside one block reads very differently
#: from a whole-block offset.
def _state_blocks(nsurface, ndiagnostic, nlevels, n_upper_air, nocean):
    i = 0
    blocks = []
    for name, width in (
        ("surface", nsurface),
        ("diagnostic", ndiagnostic),
        ("upper_air", nlevels * n_upper_air),
        ("ocean", nocean),
    ):
        if width:
            blocks.append((name, i, i + width))
            i += width
    return blocks


def _strip_model_prefix(sd):
    """``model.X`` -> ``X`` for feeding upstream's bare backbone."""
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
    return out


def _hparams(blob):
    return blob["hyper_parameters"]["config"]


def build_upstream_erdm(cfg, repo: Path):
    """Instantiate upstream's ``RollingDiT`` from its own config block."""
    sys.path.insert(0, str(repo))
    from modules.models.RollingDiT import RollingDiT  # noqa: E402

    model_cfg = cfg["model"]
    family = model_cfg[model_cfg["model_name"]]
    backbone = dict(family[model_cfg.get("backbone", "DiT")])
    data = cfg["data"]

    # Upstream's TrainModule injects state_layout (common/utils.state_layout);
    # rebuild it here from the same data config so the ocean tail and the column
    # encoder see the same block sizes.
    backbone.setdefault(
        "state_layout",
        {
            "nsurface": len(data["surface_variables"]),
            "ndiagnostic": len(data.get("diagnostic_variables", []) or []),
            "nlevels": len(data["levels"]),
            "n_upper_air": len(data["upper_air_variables"]),
            "nocean": len(data.get("ocean_state_variables", []) or []),
        },
    )
    return RollingDiT(**backbone), backbone


def build_upstream_ditae(cfg, repo: Path):
    sys.path.insert(0, str(repo))
    from modules.models.DiTAE import DiTAE  # noqa: E402

    xddc = cfg["model"]["x_DDC"]
    backbone = dict(xddc["dit"])
    return DiTAE(**backbone), backbone


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=None, help="upstream Lightning .ckpt")
    p.add_argument("--amip-v2-repo", required=True)
    p.add_argument("--family", choices=("erdm", "x_ddc"), required=True)
    p.add_argument("--tol", type=float, default=1e-5,
                   help="max |diff| accepted (fp32 forward over 20 blocks)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--synthetic", action="store_true",
                   help="random weights + a shrunk model: tests the harness, "
                        "not a checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    repo = Path(args.amip_v2_repo).expanduser().resolve()
    if not (repo / "modules").is_dir():
        raise SystemExit(f"{repo} does not look like an amip_v2 checkout")

    from physicsnemo.experimental.models.amip_si import (  # noqa: E402
        RollingDiTWrapper,
        XDDCWrapper,
    )

    if args.synthetic:
        blob = _synthetic_blob(args.family)
        state = "synthetic"      # filled in below, once ``theirs`` exists
    else:
        if not args.source:
            raise SystemExit("--source is required unless --synthetic")
        blob = load_lightning_ckpt(args.source, amip_repo=None)
        state = pick_source_state_dict(blob, prefer_live=False)

    cfg = _hparams(blob)
    data = cfg["data"]

    # ---- build both sides -------------------------------------------------
    if args.family == "erdm":
        theirs, backbone_cfg = build_upstream_erdm(cfg, repo)
        kwargs = wrapper_kwargs_from_hparams(
            blob, "RollingDiTWrapper", source_contract="v2"
        )
        ours = RollingDiTWrapper(**kwargs)
    else:
        theirs, backbone_cfg = build_upstream_ditae(cfg, repo)
        kwargs = wrapper_kwargs_from_hparams(
            blob, "XDDCWrapper", source_contract="v2"
        )
        ours = XDDCWrapper(**kwargs)

    n_theirs = sum(p.numel() for p in theirs.parameters())
    n_ours = sum(p.numel() for p in ours.backbone.parameters())
    logger.info("upstream backbone %.2fM params | ours %.2fM", n_theirs / 1e6,
                n_ours / 1e6)
    if n_theirs != n_ours:
        logger.error("PARAMETER COUNT DIFFERS — %d vs %d", n_theirs, n_ours)
        return 2

    # ---- give both the same weights --------------------------------------
    # Synthetic mode fabricates an upstream-shaped state dict from ``theirs`` and
    # then goes through the SAME path as a real checkpoint, so the translation
    # itself is covered locally. (It was not, once: the harness unpacked
    # ``translate_state_dict`` wrongly and only the Polaris run found out.)
    if state == "synthetic":
        state = {
            f"model.{k}": v.clone() for k, v in theirs.state_dict().items()
        }
        state["scheduler.noise_scales"] = torch.zeros(3)   # must be dropped
    if True:
        theirs.load_state_dict(_strip_model_prefix(state), strict=True)
        # ``translate_state_dict`` returns (sd, stats) — the stats are how many
        # keys were kept vs dropped as scheduler/unknown, worth logging here since
        # a surprising drop count is itself a translation problem.
        translated, stats = translate_state_dict(state)
        logger.info("translated keys: %s", stats)
        missing, unexpected = ours.load_state_dict(translated, strict=False)
        if missing or unexpected:
            logger.error("key mismatch: %d missing, %d unexpected",
                         len(missing), len(unexpected))
            for k in list(missing)[:10]:
                logger.error("  missing: %s", k)
            for k in list(unexpected)[:10]:
                logger.error("  unexpected: %s", k)
            return 2
    theirs.eval()
    ours.eval()

    # ---- identical inputs -------------------------------------------------
    torch.manual_seed(args.seed)
    b = args.batch
    if args.family == "erdm":
        # Read the geometry off the INSTANTIATED model, not the config block:
        # upstream's ERDM configs omit nlat/nlon entirely and rely on
        # RollingDiT's 45x90 defaults (the same omission that made the
        # translator build a 180x360 static_bias). Asking the object is the only
        # way to be sure both sides agree on the grid.
        W = int(getattr(theirs, "window_size", backbone_cfg.get("window_size", 6)))
        C = int(theirs.in_channels)
        nlat, nlon = int(theirs.nlat), int(theirs.nlon)
        cg = int(getattr(theirs, "c_grid_dim", 0) or 0)
        sd = int(getattr(theirs, "scalar_dim", 0) or 0)
        down = int(backbone_cfg.get("c_grid_downsample", 1) or 1)
        logger.info(
            "erdm inputs: W=%d C=%d grid=%dx%d c_grid=%d@%dx%d scalar=%d",
            W, C, nlat, nlon, cg, nlat * down, nlon * down, sd,
        )
        # c_grid arrives at the FULL forcing resolution; the backbone's strided
        # conv reduces it to the token grid.
        z = torch.randn(b, W, C, nlat, nlon)
        t = torch.rand(b, W)
        c_grid = torch.randn(b, W, cg, nlat * down, nlon * down) if cg else None
        c_scalar = torch.randn(b, W, sd) if sd else None
        with torch.no_grad():
            out_t = theirs(z, t, c_grid=c_grid, c_scalar=c_scalar)
            out_o = ours(z, t, c_grid=c_grid, c_scalar=c_scalar)
    else:
        C = int(theirs.out_channels)
        nlat, nlon = int(theirs.nlat), int(theirs.nlon)
        x = torch.randn(b, C, nlat, nlon)
        cond = torch.randn(b, C, nlat, nlon)
        t = torch.rand(b)
        with torch.no_grad():
            out_t = theirs(x, cond, t)
            out_o = ours(x, cond, t)

    # ---- compare ----------------------------------------------------------
    if out_t.shape != out_o.shape:
        logger.error("OUTPUT SHAPE DIFFERS: %s vs %s", tuple(out_t.shape),
                     tuple(out_o.shape))
        return 2
    diff = (out_t - out_o).abs()
    scale = out_t.abs().max().item() or 1.0
    logger.info("output %s | upstream |max| %.4e", tuple(out_t.shape), scale)
    logger.info("max |diff| %.4e   mean |diff| %.4e   relative %.4e",
                diff.max().item(), diff.mean().item(), diff.max().item() / scale)

    # Localise any mismatch: a permutation inside one block looks very different
    # from a whole-block offset, and this is the observable that distinguishes
    # them.
    layout = kwargs.get("ocean_state_variables", [])
    blocks = _state_blocks(
        len(data["surface_variables"]),
        len(data.get("diagnostic_variables", []) or []),
        len(data["levels"]),
        len(data["upper_air_variables"]),
        len(layout),
    )
    ch_axis = 2 if args.family == "erdm" else 1
    for name, lo, hi in blocks:
        if hi <= out_t.shape[ch_axis]:
            sl = diff.narrow(ch_axis, lo, hi - lo)
            logger.info("  %-11s channels %3d:%-3d  max |diff| %.4e",
                        name, lo, hi, sl.max().item())

    ok = diff.max().item() <= args.tol
    print(f"\n{'PASS' if ok else 'FAIL'}: max |diff| {diff.max().item():.4e} "
          f"(tol {args.tol:.1e})")
    return 0 if ok else 1


def _synthetic_blob(family: str) -> dict:
    """A shrunk hparams blob, for testing this harness without a checkpoint."""
    surface = [f"s{i}" for i in range(6)]
    upper = [f"u{i}" for i in range(5)]
    diag = [f"d{i}" for i in range(3)]
    levels = [1000.0, 850.0, 500.0]
    common_data = {
        "surface_variables": surface,
        "upper_air_variables": upper,
        "diagnostic_variables": diag,
        "diagnostic_input": True,
        "levels": levels,
        "horizontal_resolution": [16, 32],
    }
    n_state = len(surface) + len(diag) + len(upper) * len(levels)
    if family == "erdm":
        nocean = 2
        return {
            "hyper_parameters": {
                "config": {
                    "model": {
                        "model_name": "ERDM",
                        "backbone": "DiT",
                        "ERDM": {
                            "DiT": {
                                "in_channels": n_state + nocean,
                                "out_channels": n_state + nocean,
                                # 2 constant + 2 stored varying + 1 derived
                                # anomaly = 5. Getting this wrong is what the
                                # reconciliation is there to catch.
                                "c_grid_dim": 5,
                                "scalar_dim": 3,
                                "dim": 64,
                                "num_heads": 2,
                                "temporal_num_heads": 2,
                                "num_blocks": 1,
                                "window_size": 3,
                                # nlat/nlon deliberately ABSENT, as in every real
                                # upstream ERDM config: the harness must read the
                                # grid off the model, not the config.
                                "nlat": 4,
                                "nlon": 8,
                                "c_grid_downsample": 4,
                                "c_grid_cross_layers": 1,
                                "c_grid_cross_heads": 2,
                                "global_cond": True,
                                "input_embed": {
                                    "mode": "budget", "d_boundary": 16,
                                    "d_calendar": 16, "d_co2": 8,
                                    "state_encoder": "column", "d_level": 8,
                                },
                                "output_head": {"mode": "mix", "num_experts": 2},
                            },
                            "scheduler": {"window_size": 3},
                        },
                    },
                    "data": {
                        **common_data,
                        "constant_boundary_variables": ["z", "lsm"],
                        "varying_boundary_variables": [
                            "DSWRFtoa_24h_lead",
                            "sea_surface_temperature_monthly_interp",
                        ],
                        "ocean_state_variables": [
                            "sea_surface_temperature_monthly_interp",
                            "sea_surface_temperature_anomaly",
                        ],
                        "sst_anomaly_channel": "append",
                        "scalar_forcing": "global_mean_sst",
                    },
                }
            }
        }
    return {
        "hyper_parameters": {
            "config": {
                "model": {
                    "model_name": "x_DDC",
                    "x_DDC": {
                        "decoder_type": "dit",
                        "dit": {
                            "in_channels": 2 * n_state,
                            "out_channels": n_state,
                            "dim": 32, "num_heads": 2, "num_blocks": 1,
                            "patch_size": 4, "nlat": 16, "nlon": 32,
                            "unpatch": "vanilla",
                        },
                        "encoder": {"downsample_factor": 4},
                    },
                },
                "data": {**common_data, "varying_boundary_variables": []},
            }
        }
    }


if __name__ == "__main__":
    raise SystemExit(main())
