# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Train the PlaSim 6-hourly SFNO emulator.

Two-stage curriculum, following the plan:
  stage 1  single-step  (v11-like optimizer settings)
  stage 2  short autoregressive rollout fine-tune (unroll_steps 2 -> 4)

v11 trained single-step only (``multistep_count: 1``), which is a plausible
cause of drift over the long trajectories AI-RES rolls out; stage 2 addresses it.

Launch (single node):
    python train.py --config config/base.yaml
Multi-GPU under SLURM:
    srun python train.py --config config/base.yaml
"""

import argparse
import faulthandler
import json
import os
import sys
import time

# A native crash (SIGSEGV in NCCL, HDF5 or a CUDA kernel) otherwise gives only
# "exitcode: -11" with no Python frame, which is close to undebuggable on a
# batch system. This prints a Python traceback on fatal signals.
faulthandler.enable()

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channels import DIAGNOSTIC_CHANNELS, STATE_CHANNELS  # noqa: E402
from dataset import PlasimSequenceDataset  # noqa: E402
from loss import (  # noqa: E402
    DailyAggregateCRPS,
    DeterministicDiagnosticLoss,
    DiagnosticLoss,
    EnsembleStateLoss,
    StateLoss,
    default_precip_weight,
)
from model import PlasimEmulator, rollout  # noqa: E402


def _prepare_dist_env():
    """Make PhysicsNeMo's ENV initialization path viable in every launch mode.

    `DistributedManager.initialize()` tries `initialize_env()` first and only
    falls back to the SLURM path on TypeError. That fallback reads
    SLURM_LAUNCH_NODE_IPADDR, which is unset under plain `sbatch` (it is only
    populated by `srun`), so MASTER_ADDR becomes None and os.environ raises
    "str expected, not NoneType". Filling the standard variables here keeps a
    single code path working for sbatch, srun and torchrun.
    """
    if "MASTER_ADDR" not in os.environ:
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")
        if not addr:
            nodelist = os.environ.get("SLURM_NODELIST", "")
            if nodelist:
                import subprocess

                try:
                    addr = subprocess.check_output(
                        ["scontrol", "show", "hostnames", nodelist],
                        text=True,
                    ).split()[0]
                except Exception:
                    addr = None
        os.environ["MASTER_ADDR"] = addr or "localhost"
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))


class _SingleProcess:
    """Stand-in for DistributedManager for a one-process run.

    Torch's `init_process_group` requires an accelerator with an index, so the
    real DistributedManager cannot initialize on a CPU-only node. Single-process
    runs do not need a process group at all, and skipping it keeps the smoke test
    runnable anywhere - which is the whole point of a smoke test.
    """

    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


def get_dist():
    _prepare_dist_env()
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return _SingleProcess()

    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    return DistributedManager()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg, dist, n_steps, batch_size=None):
    ds_train = PlasimSequenceDataset(
        cfg["data"]["root"], "train", n_steps=n_steps, dt=cfg["data"].get("dt", 1)
    )
    ds_valid = PlasimSequenceDataset(
        cfg["data"]["root"], "valid", n_steps=n_steps, dt=cfg["data"].get("dt", 1)
    )
    kw = dict(
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["data"].get("num_workers", 4) > 0,
    )
    if dist.distributed:
        from torch.utils.data.distributed import DistributedSampler

        s_tr = DistributedSampler(ds_train, num_replicas=dist.world_size,
                                  rank=dist.rank, shuffle=True)
        s_va = DistributedSampler(ds_valid, num_replicas=dist.world_size,
                                  rank=dist.rank, shuffle=False)
    else:
        s_tr = s_va = None
    bs = batch_size or cfg["training"]["batch_size"]
    return (
        DataLoader(ds_train, batch_size=bs, sampler=s_tr,
                   shuffle=(s_tr is None), **kw),
        DataLoader(ds_valid, batch_size=bs, sampler=s_va, shuffle=False, **kw),
        ds_train,
    )


def build_losses(cfg, lat, device, probabilistic):
    state_loss = StateLoss(lat).to(device)
    if probabilistic:
        with open(os.path.join(cfg["data"]["root"], "stats",
                               "diagnostic_stats.json")) as f:
            scales = json.load(f)
        diag_loss = DiagnosticLoss(
            lat, scales, weights=default_precip_weight(cfg["loss"]["precip_weight"])
        ).to(device)
    else:
        diag_loss = DeterministicDiagnosticLoss(
            lat, weights=default_precip_weight(cfg["loss"]["precip_weight"])
        ).to(device)
    return state_loss, diag_loss


def run_stage(stage, cfg, model, dist, device, lat, log):
    n_steps = stage["unroll_steps"]
    # A longer rollout holds n_steps times the activations for backprop, so the
    # batch that fits at unroll=1 will OOM at unroll=4 (it did: 95 GB exhausted).
    # Each stage may therefore shrink its batch and recover the effective batch
    # size through gradient accumulation.
    bs = stage.get("batch_size", cfg["training"]["batch_size"])
    accum = max(1, int(stage.get("grad_accum", 1)))
    train_loader, valid_loader, _ = build_loaders(cfg, dist, n_steps, batch_size=bs)
    state_loss, diag_loss = build_losses(cfg, lat, device,
                                         cfg["model"]["probabilistic"])
    # An L2 state loss is minimised by the conditional mean, so the model is
    # rewarded for ignoring its noise channels and the ensemble collapses. A
    # stage with ensemble_members > 1 switches to the energy score, a proper
    # scoring rule whose spread term makes the noise worth using.
    ens_m = int(stage.get("ensemble_members", 1))
    if ens_m > 1:
        state_loss = EnsembleStateLoss(lat).to(device)

    # Per-stage loss overrides. Round 1 applied one global `loss:` block to every
    # stage, and the state L2 (~0.03) was ~100x smaller than the diagnostic NLL
    # (~-3.0), so the optimiser effectively trained precipitation alone and let
    # the prognostic state drift (t2m day-1 RMSE went 1.57 K -> 4.61 K across
    # stages 2-3). Stages can now rebalance.
    lcfg = dict(cfg["loss"])
    lcfg.update(stage.get("loss", {}))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=stage["lr"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )
    # T_max counts OPTIMIZER steps, not dataloader iterations: sched.step() fires
    # once per accumulation group, and a stage may cap iterations per epoch via
    # max_iters. Getting this wrong leaves the cosine unfinished (lr never
    # reaches min_lr) or annealed far too early.
    iters_per_epoch = min(len(train_loader), stage.get("max_iters") or 10**9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, stage["epochs"] * iters_per_epoch // accum),
        eta_min=stage["min_lr"],
    )
    amp_dtype = torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float32
    lam = lcfg["diagnostic_weight"]
    noise_sigma = cfg["training"].get("state_noise_sigma", 0.02)

    # Daily-aggregate CRPS: only meaningful once the rollout spans a whole day.
    daily_w = float(lcfg.get("daily_crps_weight", 0.0))
    daily_loss = None
    if daily_w > 0 and n_steps >= 4 and cfg["model"]["probabilistic"]:
        with open(os.path.join(cfg["data"]["root"], "stats",
                               "diagnostic_stats.json")) as f:
            _sc = json.load(f)
        daily_loss = DailyAggregateCRPS(
            lat, _sc, weights=default_precip_weight(lcfg["precip_weight"]),
            n_samples=int(lcfg.get("daily_crps_samples", 8)),
        ).to(device)
    elif daily_w > 0:
        log(f"  daily CRPS requested but skipped (unroll={n_steps} < 4)")

    log(f"  batch={bs} accum={accum} ens_members={ens_m} "
        f"lam={lam} daily_w={daily_w if daily_loss else 0.0} "
        f"-> effective batch {bs * accum * max(1, dist.world_size)}")
    for epoch in range(stage["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, nb, run = time.time(), 0, 0.0
        for batch in train_loader:
            s0 = batch["state_in"].to(device, non_blocking=True)
            fo = batch["forcing"].to(device, non_blocking=True)
            st = batch["state_tgt"].to(device, non_blocking=True)
            dg = batch["diag_tgt"].to(device, non_blocking=True)

            # v11's white input noise, applied to the state channels only.
            if noise_sigma > 0:
                s0 = s0 + noise_sigma * torch.randn_like(s0)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                if ens_m > 1:
                    members = [rollout(model, s0, fo) for _ in range(ens_m)]
                    ens_s = torch.stack([m[0] for m in members], 0)
                    l_state = state_loss(ens_s, st)
                    # Each member's diagnostic head should fit the truth.
                    dls = [diag_loss(m[1], dg) for m in members]
                    l_diag = sum(d[0] for d in dls) / ens_m
                    parts = dls[0][1]
                else:
                    pred_s, pred_d = rollout(model, s0, fo)
                    l_state = state_loss(pred_s, st)
                    l_diag, parts = diag_loss(pred_d, dg)
                loss = l_state + lam * l_diag
                if daily_loss is not None:
                    l_day, _ = daily_loss(pred_d if ens_m == 1 else members[0][1], dg)
                    loss = loss + daily_w * l_day

            (loss / accum).backward()
            if (nb + 1) % accum == 0:
                if cfg["training"].get("max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["training"]["max_grad_norm"]
                    )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run += float(loss.detach())
            nb += 1
            if nb % cfg["training"].get("log_every", 50) == 0:
                log(f"  [{stage['name']}] ep{epoch} it{nb} "
                    f"loss={run / nb:.4f} state={float(l_state.detach()):.4f} "
                    f"diag={float(l_diag.detach()):.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}")
            if stage.get("max_iters") and nb >= stage["max_iters"]:
                break

        va = validate(model, valid_loader, state_loss, diag_loss, device,
                      amp_dtype, lam, cfg)
        log(f"[{stage['name']}] epoch {epoch} train={run / max(nb,1):.4f} "
            f"valid={va:.4f} ({time.time() - t0:.0f}s)")
        save_ckpt(cfg, model, stage, epoch, va, dist)


@torch.no_grad()
def validate(model, loader, state_loss, diag_loss, device, amp_dtype, lam, cfg):
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= cfg["training"].get("valid_iters", 50):
            break
        s0 = batch["state_in"].to(device)
        fo = batch["forcing"].to(device)
        st = batch["state_tgt"].to(device)
        dg = batch["diag_tgt"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            if getattr(state_loss, "_is_ensemble", False):
                mem = [rollout(model, s0, fo) for _ in range(2)]
                ens_s = torch.stack([m[0] for m in mem], 0)
                l = state_loss(ens_s, st) + lam * diag_loss(mem[0][1], dg)[0]
            else:
                pred_s, pred_d = rollout(model, s0, fo)
                l = state_loss(pred_s, st) + lam * diag_loss(pred_d, dg)[0]
        tot += float(l)
        n += 1
    return tot / max(n, 1)


def save_ckpt(cfg, model, stage, epoch, va, dist):
    if dist.rank != 0:
        return
    d = cfg["training"]["checkpoint_dir"]
    # A smoke/preflight run trains a deliberately tiny model. Writing it to the
    # real checkpoint directory would leave a 0.2M-parameter file with the same
    # name as the trained model - harmless once the full run overwrites it, but
    # actively misleading if the full run dies first.
    if cfg.get("_smoke"):
        d = d.rstrip("/") + "_smoke"
    os.makedirs(d, exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    torch.save(
        {"model": m.state_dict(), "stage": stage["name"], "epoch": epoch,
         "valid": va, "cfg": cfg,
         "channels": {"state": STATE_CHANNELS, "diagnostic": DIAGNOSTIC_CHANNELS}},
        os.path.join(d, f"ckpt_{stage['name']}.pt"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, few iters - pipeline check only")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to load model weights from before training")
    ap.add_argument("--only-stages", default=None,
                    help="comma-separated stage names to run (default: all)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cfg["_smoke"] = bool(args.smoke)

    dist = get_dist()
    device = dist.device

    def log(msg):
        if dist.rank == 0:
            print(msg, flush=True)

    lat = np.load(os.path.join(cfg["data"]["root"], "stats", "lat.npy"))

    mcfg = dict(cfg["model"].get("sfno", {}))
    if args.smoke:
        mcfg.update(embed_dim=32, num_layers=2)
    model = PlasimEmulator(
        n_noise=cfg["model"]["n_noise"],
        probabilistic=cfg["model"]["probabilistic"],
        **mcfg,
    ).to(device)
    log(f"model: in={model.inp_chans} out={model.out_chans} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if args.resume and args.smoke:
        # --smoke builds a deliberately tiny model (embed_dim 32), so a full-size
        # checkpoint cannot load into it. The preflight exists to exercise the
        # code path, not the weights, so skip the load rather than fail.
        log(f"smoke run: NOT loading {args.resume} (tiny model, shapes differ)")
    elif args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        log(f"resumed weights from {args.resume} "
            f"(stage={ck.get('stage')} epoch={ck.get('epoch')} "
            f"valid={ck.get('valid')})")

    if dist.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank], output_device=dist.local_rank
        )

    stages = cfg["training"]["stages"]
    if args.only_stages:
        want = [x.strip() for x in args.only_stages.split(",")]
        stages = [st for st in stages if st["name"] in want]
        if not stages:
            raise SystemExit(f"no stages matched {want}")
    if args.smoke:
        for s in stages:
            s["epochs"] = 1
            s["max_iters"] = 5
        stages = stages[:1]
    for stage in stages:
        log(f"=== stage {stage['name']} unroll={stage['unroll_steps']} ===")
        run_stage(stage, cfg, model, dist, device, lat, log)


if __name__ == "__main__":
    main()
