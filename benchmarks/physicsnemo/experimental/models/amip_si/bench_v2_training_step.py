# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Per-batch training-step speed: our v2 port vs upstream amip_v2 (2026-08-18).

Times one full optimizer step — forward, loss, backward, ``optimizer.step()`` —
for the two v2 families, on **identical synthetic tensors**, and reports ours
against upstream's own implementation.

Synthetic on purpose. The question here is how fast the *model and scheduler*
are, and a real loader adds Lustre variance that swamps the differences we are
looking for. End-to-end iteration time (loader included) is a separate
measurement, taken from ``cfg.bench.per_batch_tsv`` on a real run.

Why the geometry is trustworthy: each side is built from **its own** config —
ours from ``conf/model/*.yaml``, upstream's from its repo's ``configs/*.yaml`` —
and the harness then asserts the parameter counts agree. A mismatch means the two
configs describe different models and the timing comparison is meaningless, so it
is a hard error rather than a footnote.

The comparable pairs:

===========================  ===========================  ==========================
ours                         upstream                     geometry
===========================  ===========================  ==========================
``amip_erdm_v2``             ``RollingDiT`` / ERDM_co2    dim 1024, 20 blocks, W=6
``amip_x_ddc_dit``           ``DiTAE`` / DDC              dim 1024, 20 blocks, patch 4
===========================  ===========================  ==========================

``amip_x_ddc`` (the UNet denoiser) is deliberately absent: upstream's
``ae_module`` raises ``NotImplementedError`` for any ``decoder_type`` other than
``dit``, so that config has no upstream counterpart to race. Use
``--ours-config amip_x_ddc --side ours`` to profile it on its own.

Fairness rules, all of them deliberate:

* Both sides get the SAME AdamW (upstream's configured lr), because the subject
  is the model and scheduler, not the optimizer.
* Both sides see tensors drawn from the same seed, materialised once.
* Timing brackets each step with ``torch.cuda.synchronize()``. Without that,
  CUDA's async launch queue makes a fast step look ~free and the numbers are
  fiction.
* Warmup iterations are discarded (allocator growth, cuDNN autotuning, and the
  first kernel JIT all land there).
* The median is the headline, not the mean: a single scheduler hiccup or an
  allocator flush skews a mean over 20 iterations.
* For x_DDC the bilinear encode that produces ``x_lowres`` runs ONCE outside the
  timed region and the same tensor feeds both sides, so this measures the
  denoiser and scheduler rather than two implementations of a resize. ``--report
  encode`` times that resize separately if you want to confirm it is negligible.

Run::

    python benchmarks/physicsnemo/experimental/models/amip_si/bench_v2_training_step.py \\
        --family erdm --amip-repo ~/amip_v2 --iters 20

    # our side only, no upstream checkout needed
    python .../bench_v2_training_step.py --family x_ddc --side ours

    # opt-in speed knobs, measured one at a time
    python .../bench_v2_training_step.py --family erdm --side ours --tf32
    python .../bench_v2_training_step.py --family erdm --side ours --amp bf16
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[5]
_RECIPE = _REPO / "examples" / "weather" / "ai_rossby"
for _p in (str(_REPO), str(_RECIPE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("bench_v2")

# Our configs name the backbone block per family: RollingDiT uses
# `rolling_dit_kwargs`, the x_DDC DiT `dit_kwargs`, the x_DDC UNet `unet_kwargs`.
# Resolved rather than assumed, because assuming one of them is a KeyError the
# moment a family is added.
_BACKBONE_KEYS = ("rolling_dit_kwargs", "dit_kwargs", "unet_kwargs")


def backbone_block(cfg: dict) -> dict:
    for key in _BACKBONE_KEYS:
        if key in cfg:
            return cfg[key]
    raise KeyError(
        f"no backbone block in config (looked for {_BACKBONE_KEYS}); "
        f"has {sorted(k for k in cfg if k.endswith('_kwargs'))}"
    )

# family -> (our model config, our scheduler config, upstream yaml, upstream key path)
FAMILIES = {
    "erdm": {
        "ours_model": "amip_erdm_v2",
        "ours_sched": ("loss", "erdm_v2"),
        "upstream_yaml": "configs/ERDM_co2.yaml",
    },
    "x_ddc": {
        "ours_model": "amip_x_ddc_dit",
        "ours_sched": ("sampler", "x_ddc"),
        "upstream_yaml": "configs/DDC.yaml",
    },
}


# --------------------------------------------------------------------------- #
# Building the two sides
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _shrink(cfg: dict, factor: int) -> dict:
    """Cut the backbone down so the harness itself can be tested on a small GPU.

    Only for validating the harness; headline numbers must come from the shipped
    geometry. Touches width, depth and the width-proportional embed budgets only
    — never channel counts, window size,
    patch size or downsample factor, since those change the *shape* of the work
    and would make a shrunken run unrepresentative in kind rather than just size.
    """
    for key in _BACKBONE_KEYS:
        blk = cfg.get(key)
        if not blk:
            continue
        if "dim" in blk:
            blk["dim"] = max(64, blk["dim"] // factor)
        if "model_channels" in blk:
            blk["model_channels"] = max(32, blk["model_channels"] // factor)
        for depth in ("num_blocks", "num_res_blocks"):
            if depth in blk:
                blk[depth] = max(1, blk[depth] // factor)
        # Heads must divide the width; keep them small enough to stay legal.
        if "num_heads" in blk:
            blk["num_heads"] = max(1, min(blk["num_heads"], blk.get("dim", 64) // 32))
        if "temporal_num_heads" in blk:
            blk["temporal_num_heads"] = max(1, blk["num_heads"] // 2)
        # The input-embed BUDGET dims are fractions of the width by construction
        # (d_boundary + d_calendar must fit inside dim, and the layer enforces
        # it), so they scale with dim or a shrunken run cannot even be built.
        for sub in ("input_embed", "output_head"):
            nested = blk.get(sub)
            if isinstance(nested, dict):
                for k, v in list(nested.items()):
                    if k.startswith("d_") and isinstance(v, int):
                        nested[k] = max(8, v // factor)
        for k in ("c_grid_embed_dim", "c_scalar_embed_dim"):
            if k in blk and isinstance(blk[k], int):
                blk[k] = max(8, blk[k] // factor)
        # Cross-attention layers are interleaved among the blocks, so the count
        # cannot exceed a shrunken depth (RollingDiT enforces 0 < layers <= depth).
        depth = blk.get("num_blocks")
        for k in ("c_grid_cross_layers", "num_ca_blocks", "num_output_blocks"):
            if k in blk and depth:
                blk[k] = max(1, min(blk[k], depth))
    return cfg


def upstream_scheduler_kwargs(family: str, repo: Path) -> dict:
    """Upstream's own scheduler block, so both sides can be given identical ones."""
    ucfg = _load_yaml(repo / FAMILIES[family]["upstream_yaml"])
    if family == "erdm":
        model_cfg = ucfg["model"]
        kw = dict(model_cfg[model_cfg["model_name"]]["scheduler"])
    else:
        kw = dict(ucfg["model"]["x_DDC"]["scheduler"])
    kw.pop("noise_scale_path", None)
    return kw


def build_ours(family: str, config_name: str | None, shrink: int, device,
               sched_kwargs: dict | None = None):
    from omegaconf import OmegaConf
    from train import build_model  # the recipe's builder, so this is what trains

    spec = FAMILIES[family]
    name = config_name or spec["ours_model"]
    cfg = _load_yaml(_RECIPE / "conf" / "model" / f"{name}.yaml")
    if shrink > 1:
        cfg = _shrink(cfg, shrink)
    model = build_model(OmegaConf.create(cfg)).to(device)

    group, sched_name = spec["ours_sched"]
    sched_cfg = _load_yaml(_RECIPE / "conf" / group / f"{sched_name}.yaml")
    target = sched_cfg.pop("_target_")
    module_path, cls_name = target.rsplit(".", 1)
    import importlib

    sched_cls = getattr(importlib.import_module(module_path), cls_name)
    sched_cfg.pop("noise_scale_path", None)   # artifact path, not a speed knob
    if sched_kwargs is not None:
        # Upstream's kwargs, so the A/B measures implementation rather than a
        # settings difference. This matters concretely for x_DDC: our shipped
        # sampler config asks for `noise: spherical`, which runs an
        # InverseRealSHT over every channel per step, and upstream's x_DDC
        # interpolant REFUSES anything but gaussian -- so leaving it on would
        # time a feature upstream cannot run and call the gap a port regression.
        # (Our x_DDC sampler config is also inference-only: no x_DDC training
        # config is shipped, so there is no "our training settings" to defend.)
        sched_cfg = dict(sched_kwargs)
    scheduler = sched_cls(**sched_cfg)
    if hasattr(scheduler, "to"):
        scheduler = scheduler.to(device)
    return model, scheduler, cfg, name


def build_upstream(family: str, repo: Path, shrink: int, device):
    """Instantiate upstream's own module and scheduler from its own config."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = FAMILIES[family]
    ucfg = _load_yaml(repo / spec["upstream_yaml"])

    if family == "erdm":
        from modules.diffusion.erdm import ERDMScheduler
        from modules.models.RollingDiT import RollingDiT

        model_cfg = ucfg["model"]
        fam = model_cfg[model_cfg["model_name"]]
        backbone = dict(fam[model_cfg.get("backbone", "DiT")])
        data = ucfg["data"]
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
        if shrink > 1:
            backbone = _shrink({"dit_kwargs": backbone}, shrink)["dit_kwargs"]
        model = RollingDiT(**backbone).to(device)
        sched_kwargs = dict(fam["scheduler"])
        sched_kwargs.pop("noise_scale_path", None)
        scheduler = ERDMScheduler(**sched_kwargs)
    else:
        from modules.diffusion.x_DDC import DataDependentInterpolant
        from modules.models.DiTAE import DiTAE

        xddc = ucfg["model"]["x_DDC"]
        backbone = dict(xddc["dit"])
        if shrink > 1:
            backbone = _shrink({"dit_kwargs": backbone}, shrink)["dit_kwargs"]
        model = DiTAE(**backbone).to(device)
        sched_kwargs = dict(xddc["scheduler"])
        sched_kwargs.pop("noise_scale_path", None)
        scheduler = DataDependentInterpolant(**sched_kwargs)
    if hasattr(scheduler, "to"):
        scheduler = scheduler.to(device)
    return model, scheduler, ucfg


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def make_inputs(family: str, our_cfg: dict, model, batch: int, device, seed: int = 0):
    """Synthetic tensors matching what the recipe's packer produces.

    Shapes come from the config and from the BUILT wrapper, not from constants, so
    a config change cannot quietly make this benchmark measure a different problem
    size. ``c_grid_dim`` and ``scalar_dim`` in particular are *derived* by the
    wrapper from its variable lists and deliberately absent from the YAML, so they
    are read off the model — re-deriving them here would be a second, driftable
    copy of that rule.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    nsurf = len(our_cfg["surface_variables"])
    ndiag = len(our_cfg.get("diagnostic_variables", []) or [])
    nlev = len(our_cfg["levels"])
    nupper = len(our_cfg["upper_air_variables"])
    channels = nsurf + ndiag + nlev * nupper
    h, w = our_cfg["horizontal_resolution"]

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(device)

    if family == "erdm":
        bb = backbone_block(our_cfg)
        window = bb["window_size"]
        c_grid_dim = int(model.c_grid_dim)
        scalar_dim = int(model.scalar_dim)
        stride = bb.get("c_grid_downsample", 1)
        # Forcings live on the FULL-res grid and the model reduces them with its
        # stride-N conv; that is the upstream pairing and it dominates the
        # cross-attention cost, so getting it wrong here would flatter us.
        return {
            "y": randn(batch, window, channels, h, w),
            "c_grid": randn(batch, window, c_grid_dim, h * stride, w * stride),
            "c_scalar": randn(batch, window, scalar_dim),
        }
    # x_DDC: build the unpacked sample and let the WRAPPER pack it and produce the
    # blurred conditioning field, exactly as upstream's ae_module does
    # (`y = assemble_input(...)`, `z = encode(...)`). Hand-rolling the pack here
    # would be a second copy of the channel order — the class of bug the whole
    # Phase 12 rebaseline was about.
    sample = {
        "surface_in": randn(batch, nsurf, h, w),
        "upper_air_in": randn(batch, nupper, nlev, h, w),
    }
    if ndiag:
        sample["diagnostic"] = randn(batch, ndiag, h, w)
    with torch.no_grad():
        x_highres = model.pack_state(sample)
        x_lowres = model.downsample_then_upsample(sample)
    if x_highres.shape[-3] != channels:
        raise ValueError(
            f"packed {x_highres.shape[-3]} channels, config implies {channels} — "
            f"the benchmark and the wrapper disagree about the state"
        )
    return {"x_highres": x_highres, "x_lowres": x_lowres}


def step_fn(family: str, model, scheduler, inputs):
    """One loss evaluation, identical in structure to each side's training_step."""
    if family == "erdm":
        return lambda: scheduler.compute_loss(
            model, inputs["c_grid"], inputs["c_scalar"], inputs["y"]
        )
    return lambda: scheduler.compute_loss(
        model, inputs["x_lowres"], inputs["x_highres"]
    )


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _muon_groups_like_upstream(model, lr: float) -> list[dict]:
    """Upstream's split, applied identically to both sides.

    ``TrainModule.get_rolling_dit_muon_param_groups`` sends the transformer
    blocks' weight matrices to Muon at 10x lr and everything else — gains, biases,
    embedders, patchify/unpatchify, output head — to the aux Adam at 1x. Neither
    "all 2-D tensors" nor our wrapper's own helper is the same rule, and since
    Muon and Adam carry different amounts of state per parameter, using different
    rules on the two sides would turn a memory comparison into a grouping
    comparison. Mirrored here once, for both.
    """
    # Our wrapper holds the backbone one level down; upstream's model IS the
    # backbone. Both name their containers the same way — RollingDiT's
    # spatial_blocks / temporal_blocks / forcing_blocks and DiTAE's sa_blocks —
    # which is itself a sign the port is structural rather than a rewrite.
    root = getattr(model, "backbone", model)
    containers = [
        getattr(root, n)
        for n in ("spatial_blocks", "temporal_blocks", "forcing_blocks",
                  "sa_blocks", "blocks")
        if getattr(root, n, None) is not None
    ]
    if not containers:
        raise ValueError(
            f"{type(model).__name__} exposes no transformer block containers "
            f"(looked for spatial_blocks/temporal_blocks/forcing_blocks/blocks)"
        )
    hidden = [q for c in containers for q in c.parameters() if q.ndim >= 2]
    hidden_ids = {id(q) for q in hidden}
    rest = [q for q in model.parameters() if id(q) not in hidden_ids]
    return [
        dict(params=hidden, use_muon=True, lr=lr * 10, weight_decay=0.01),
        dict(params=rest, use_muon=False, lr=lr, betas=(0.9, 0.95),
             weight_decay=0.01),
    ]


def build_optimizer(kind: str, model, lr: float):
    """AdamW or Muon.

    Not cosmetic: Muon keeps ONE momentum buffer for the matrices it owns where
    AdamW keeps two moments for every parameter, so the choice moves peak memory
    by gigabytes at this size. It matters here because upstream's ERDM config says
    ``optimizer: muon`` and trains on 40 GB A100s — benchmarking ours with AdamW
    and calling the memory difference a port property would be wrong.
    """
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    from muon import MuonWithAuxAdam

    # MuonWithAuxAdam.step() pads its parameter list to a multiple of the world
    # size, so it cannot run outside a process group at all — the same trap
    # train_loop._make_muon_optimizer guards, hit on Polaris job 7438576. Rather
    # than demand torchrun for a single-GPU benchmark, stand up a one-rank group
    # the way upstream's own common.utils.ensure_process_group does: a HashStore
    # is an in-process rendezvous, so it needs no free port and no shared file.
    if not torch.distributed.is_initialized():
        if not torch.distributed.is_available():
            raise RuntimeError("--optimizer muon needs torch.distributed")
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            store=torch.distributed.HashStore(), rank=0, world_size=1,
        )
        atexit.register(
            lambda: torch.distributed.is_initialized()
            and torch.distributed.destroy_process_group()
        )
    return MuonWithAuxAdam(_muon_groups_like_upstream(model, lr))


def time_steps(loss_fn, model, *, iters: int, warmup: int, lr: float,
               amp_dtype, device, optimizer: str = "adamw") -> dict:
    opt = build_optimizer(optimizer, model, lr)
    samples: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i in range(warmup + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp_dtype is not None,
            dtype=amp_dtype or torch.float32,
        ):
            loss = loss_fn()
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            samples.append(dt)

    del opt
    out = {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iters": len(samples),
        "loss": float(loss.detach().float().item()),
    }
    if len(samples) > 1:
        out["stdev_ms"] = statistics.stdev(samples)
    if device.type == "cuda":
        out["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("--side", choices=("ours", "upstream", "both"), default="both")
    p.add_argument("--amip-repo", type=Path, default=None,
                   help="upstream amip_v2 checkout (required unless --side ours)")
    p.add_argument("--ours-config", default=None,
                   help="override our model config name (e.g. amip_x_ddc)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch", type=int, default=1, help="upstream trains at 1")
    p.add_argument("--lr", type=float, default=5.0e-5, help="upstream's configured lr")
    p.add_argument("--shrink", type=int, default=1,
                   help="divide backbone width/depth by N — for testing the "
                        "harness on a small GPU, NOT for headline numbers")
    p.add_argument("--tf32", action="store_true",
                   help="enable TF32 matmuls (upstream's train.py sets "
                        "set_float32_matmul_precision('high'); our recipe does not)")
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none",
                   help="upstream trains precision: 32-true, so 'none' is parity")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--scheduler", choices=("upstream", "ours"), default=None,
                   help="whose scheduler kwargs BOTH sides use. Default: upstream "
                        "when --amip-repo is given (isolates implementation), else "
                        "ours. Use 'ours' to price our own options, e.g. x_DDC's "
                        "spherical noise.")
    p.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                   help="upstream's ERDM/DDC configs say muon; AdamW is the "
                        "default here only because muon is an optional extra")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.side != "ours" and args.amip_repo is None:
        p.error("--amip-repo is required unless --side ours")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "family": args.family,
        "batch": args.batch,
        "tf32": bool(args.tf32),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "amp": args.amp,
        "compile": bool(args.compile),
        "optimizer": args.optimizer,
        "shrink": args.shrink,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    logger.info("environment: %s", json.dumps(env))
    if args.shrink > 1:
        logger.warning("--shrink=%d: harness self-test geometry, NOT a headline "
                       "number", args.shrink)

    sched_source = args.scheduler or ("upstream" if args.amip_repo else "ours")
    if sched_source == "upstream" and args.amip_repo is None:
        p.error("--scheduler upstream needs --amip-repo")
    sched_kwargs = (
        upstream_scheduler_kwargs(args.family, args.amip_repo)
        if sched_source == "upstream" else None
    )
    env["scheduler_kwargs_from"] = sched_source
    logger.info("scheduler kwargs from: %s", sched_source)
    our_model, our_sched, our_cfg, our_name = build_ours(
        args.family, args.ours_config, args.shrink, device, sched_kwargs
    )
    inputs = make_inputs(args.family, our_cfg, our_model, args.batch, device, args.seed)
    results: dict[str, dict] = {}
    n_params = {}

    sides = ("ours", "upstream") if args.side == "both" else (args.side,)
    for side in sides:
        if side == "ours":
            model, sched = our_model, our_sched
            label = f"ours/{our_name}"
        else:
            model, sched, _ = build_upstream(
                args.family, args.amip_repo, args.shrink, device
            )
            label = f"upstream/{args.family}"
        n_params[side] = sum(q.numel() for q in model.parameters())
        if args.compile:
            model = torch.compile(model)
        model.train()
        results[side] = time_steps(
            step_fn(args.family, model, sched, inputs),
            model,
            iters=args.iters, warmup=args.warmup, lr=args.lr,
            amp_dtype=amp_dtype, device=device, optimizer=args.optimizer,
        )
        results[side]["label"] = label
        results[side]["params_m"] = n_params[side] / 1e6
        # Free this side COMPLETELY before building the next one. Without this the
        # second side's peak-memory reading includes the first side's resident
        # weights, grads and optimizer state — which silently made upstream look
        # 2.9 GB hungrier than ours in the first A100 run. Peak memory is only
        # attributable if one model is alive at a time.
        if side == "ours":
            our_model = None
        del model, sched
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # A geometry mismatch invalidates the whole comparison, so it is fatal.
    if len(n_params) == 2 and n_params["ours"] != n_params["upstream"]:
        raise SystemExit(
            f"parameter counts differ — ours {n_params['ours']:,} vs upstream "
            f"{n_params['upstream']:,}. The two configs describe different models, "
            f"so the timings are not comparable. Reconcile the geometry first."
        )

    print()
    print(f"{'side':28s} {'params':>10s} {'median':>10s} {'min':>9s} "
          f"{'stdev':>8s} {'peak mem':>9s}")
    print("-" * 78)
    for side in sides:
        r = results[side]
        print(f"{r['label']:28s} {r['params_m']:9.1f}M {r['median_ms']:9.1f}ms "
              f"{r['min_ms']:8.1f}ms {r.get('stdev_ms', 0):7.1f}ms "
              f"{r.get('peak_mem_gib', 0):8.2f}G")
    if len(sides) == 2:
        ratio = results["ours"]["median_ms"] / results["upstream"]["median_ms"]
        verdict = "SLOWER" if ratio > 1.02 else ("faster" if ratio < 0.98 else "parity")
        print(f"\nours / upstream = {ratio:.3f}x  ({verdict})")

    if args.json:
        args.json.write_text(json.dumps({"env": env, "results": results}, indent=2))
        logger.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
