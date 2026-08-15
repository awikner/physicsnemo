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

**Which upstream checkout to compare against depends on the family.** ERDM and
x_DDC are amip_v2's; SI / SI_X / EDM exist *only* in v1 (``497827e``), because
amip_v2 deleted the single-step families — so ``--family si`` wants the v1
checkout. ``--amip-repo`` names it either way (``--amip-v2-repo`` still works).

Usage::

    python tools/checkpoint_translation/verify_v2_numerical.py \
        --source $CKPT/ERDM_fancy_.../last.ckpt \
        --amip-repo ~/amip_v2 --family erdm

    # frozen v1 family — note the v1 checkout
    python tools/checkpoint_translation/verify_v2_numerical.py \
        --source $CKPT/SI_X_AIMIP_wCO2_interp_gaussian_.../last.ckpt \
        --amip-repo $AI_ROSSBY_AMIP_REPO --family si

    # harness self-test with random weights, no multi-GB blob needed
    python tools/checkpoint_translation/verify_v2_numerical.py \
        --amip-repo ~/amip_v2 --family erdm --synthetic

The sampler's noise type (gaussian vs spherical) does not enter into any of this:
what is compared is the deterministic network forward, so the scheduler — and
therefore the noise basis — is never constructed.
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


def _geom(model, backbone_cfg: dict, name: str, default):
    """A geometry number, preferring the INSTANTIATED model over the config.

    The model is authoritative when it exposes the attribute — upstream's ERDM
    configs omit ``nlat``/``nlon`` and lean on class defaults, and trusting the
    config there is what built a 180x360 ``static_bias`` against a 45x90 one
    (Phase 12h bug 2). But not every backbone stores every kwarg: ``AmipDiT``
    keeps ``c_grid_dim`` and not ``scalar_dim``, so a missing attribute must
    fall back to the config block rather than to a default that silently
    disables a whole conditioning stream.
    """
    value = getattr(model, name, None)
    if value is None:
        value = backbone_cfg.get(name, default)
    return default if value is None else value


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


def _backbone_block(cfg) -> tuple[dict, str]:
    """The raw backbone kwargs out of a v1/v2 hparams block.

    Mirrors ``amip_si.wrapper_kwargs_from_hparams``'s precedence exactly —
    ``model.<family>.<model.backbone>``, else a legacy ``model.<family>.model``
    — because both sides of this comparison have to read the *same* block. This
    is the lookup whose earlier ``.get("model", {})`` version silently returned
    ``{}`` and built the class-default geometry (Phase 12h bug 1).
    """
    model = cfg["model"]
    family_name = model["model_name"]
    family = model[family_name]
    backbone_key = model.get("backbone", None)
    if backbone_key and backbone_key in family:
        return dict(family[backbone_key]), f"model.{family_name}.{backbone_key}"
    if "model" in family:
        return dict(family["model"]), f"model.{family_name}.model"
    raise KeyError(
        f"cannot find backbone kwargs under model.{family_name} "
        f"(has {sorted(family)}); model.backbone={backbone_key!r}"
    )


def build_upstream_si(cfg, repo: Path):
    """Instantiate upstream **v1**'s ``DiT`` — the SI / SI_X / EDM backbone.

    Vendored here as :class:`AmipDiT` (``dit.py`` carries the provenance:
    ``amip`` @ ``497827e``, ``modules/models/DiT.py``). amip_v2 deleted this
    family, so the comparison repo for ``--family si`` is the **v1** checkout,
    not the v2 one.
    """
    sys.path.insert(0, str(repo))
    from modules.models.DiT import DiT  # noqa: E402

    backbone, where = _backbone_block(cfg)
    logger.info("upstream backbone kwargs from %s (%d keys)", where, len(backbone))
    return DiT(**backbone), backbone


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=None, help="upstream Lightning .ckpt")
    # Which checkout to compare against depends on the family: ERDM/x_DDC live
    # in amip_v2, while SI/SI_X/EDM exist ONLY in v1 (v2 deleted them). Both
    # spellings are accepted so the Phase-12h command lines and
    # translate_v2_checkpoints_polaris.pbs keep working verbatim.
    p.add_argument("--amip-repo", "--amip-v2-repo", dest="amip_repo", required=True,
                   help="upstream checkout to compare against: amip_v2 for "
                        "--family erdm/x_ddc, amip v1 (497827e) for --family si")
    p.add_argument("--family", choices=("erdm", "x_ddc", "si"), required=True)
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
    repo = Path(args.amip_repo).expanduser().resolve()
    if not (repo / "modules").is_dir():
        raise SystemExit(f"{repo} does not look like an amip checkout")

    from physicsnemo.experimental.models.amip_si import (  # noqa: E402
        AmipDiTWrapper,
        RollingDiTWrapper,
        XDDCWrapper,
    )

    if args.synthetic:
        blob = _synthetic_blob(args.family)
        state = "synthetic"      # filled in below, once ``theirs`` exists
    else:
        if not args.source:
            raise SystemExit("--source is required unless --synthetic")
        # Hand the repo over: a v1 Lightning blob pickles a reference to
        # upstream's dataset/normalizer, so unpickling needs it on sys.path.
        # (The v2 pair happened not to, which is why ``None`` sufficed in 12h.)
        blob = load_lightning_ckpt(args.source, amip_repo=repo)
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
    elif args.family == "si":
        # SI / SI_X / EDM are the FROZEN v1 families, so the source contract is
        # v1: group order [surface | diagnostic | upper_air] with the upper-air
        # block variable-major. Translating this one as "fork" is the Phase-8e
        # channel scramble, and it is shape-preserving — which is the whole
        # reason this comparison exists.
        theirs, backbone_cfg = build_upstream_si(cfg, repo)
        kwargs = wrapper_kwargs_from_hparams(
            blob, "AmipDiTWrapper", source_contract="v1"
        )
        ours = AmipDiTWrapper(**kwargs)
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
    elif args.family == "si":
        # Single-step signature: forward(x_noised, cond, t, c_grid, c_scalar),
        # ``t`` shaped (b, 1) and c_grid at the MODEL grid (v1's DiT has no
        # strided forcing conv, unlike the v2 RollingDiT).
        # ``in_channels`` is the PatchEmbed width — x_noised and cond
        # CONCATENATED (upstream writes ``in_channels: 302  # 151*2``), so each
        # stream is half of it.
        C = int(theirs.in_channels) // 2
        nlat, nlon = int(theirs.nlat), int(theirs.nlon)
        cg = int(_geom(theirs, backbone_cfg, "c_grid_dim", 0))
        # NOT getattr alone: AmipDiT keeps c_grid_dim as an attribute but NOT
        # scalar_dim, so a bare getattr default silently passed c_scalar=None
        # and the forward assembled 4 channels short of its own PatchEmbed.
        sd = int(_geom(theirs, backbone_cfg, "scalar_dim", 0))
        # Same asymmetry as the v2 RollingDiT: with a strided forcing conv the
        # c_grid stream arrives at ``down x`` the state grid and is reduced onto
        # it. Upstream's SI recipe pre-downsamples x_noised+cond to that state
        # grid; ours instead runs c_grid_downsample=1 with both streams native
        # (see AmipDiTWrapper). The checkpoint's own value decides which
        # geometry these weights were trained in, so read it, don't assume.
        down = int(backbone_cfg.get("c_grid_downsample", 1) or 1)
        logger.info(
            "si inputs: C=%d (x2 concat) grid=%dx%d c_grid=%d@%dx%d scalar=%d",
            C, nlat, nlon, cg, nlat * down, nlon * down, sd,
        )
        x_noised = torch.randn(b, C, nlat, nlon)
        cond = torch.randn(b, C, nlat, nlon)
        t = torch.rand(b, 1)
        c_grid = torch.randn(b, cg, nlat * down, nlon * down) if cg else None
        c_scalar = torch.randn(b, sd) if sd else None
        with torch.no_grad():
            out_t = theirs(x_noised, cond, t, c_grid=c_grid, c_scalar=c_scalar)
            out_o = ours(x_noised, cond, t, c_grid=c_grid, c_scalar=c_scalar)
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
    if family == "si":
        # SI_X with CO2 routed out of the grid stream, which is the shape of the
        # real wCO2 checkpoints: the data lists 5 varying boundaries while the
        # backbone says c_grid_dim=6 (2 constant + 4 varying), because upstream
        # sends the 5th through the scalar path. The translator's reconciliation
        # is what has to notice that, so the synthetic blob exercises it.
        return {
            "hyper_parameters": {
                "config": {
                    "model": {
                        "model_name": "SI_X",
                        "backbone": "DiT",
                        "SI_X": {
                            "DiT": {
                                # DOUBLED, as every real SI config is
                                # (upstream writes ``in_channels: 302  # 151*2``):
                                # AmipDiT.forward concatenates [x_noised, cond],
                                # so in_channels is the PatchEmbed width, not the
                                # state width. out_channels is the state width.
                                "in_channels": 2 * n_state,
                                "out_channels": n_state,
                                "c_grid_dim": 6,
                                "scalar_dim": 3,
                                "dim": 64,
                                "num_heads": 2,
                                "num_blocks": 1,
                                "patch_size": 2,
                                "nlat": 16,
                                "nlon": 32,
                                # Stated explicitly, as a real config does.
                                # It has to be: upstream's AmipDiT defaults it
                                # to 4 while our WRAPPER defaults it to 1 (our
                                # recipe keeps c_grid at native resolution
                                # instead of upstream's pre-downsample of
                                # x_noised+cond). The checkpoint's own value
                                # flows through and settles it — but a config
                                # that OMITS it would build a 4x4 forcing conv
                                # upstream and a 1x1 here.
                                "c_grid_downsample": 4,
                            },
                            "scheduler": {"num_steps": 10},
                        },
                    },
                    "data": {
                        **common_data,
                        "constant_boundary_variables": ["z", "lsm"],
                        "varying_boundary_variables": [
                            "sea_surface_temperature",
                            "sea_ice_cover",
                            "DSWRFtoa",
                            "DSWRFtoa_24h_lead",
                            "global_mean_co2",
                        ],
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
