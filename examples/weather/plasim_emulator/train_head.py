# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Model B: a separate SFNO diagnostic head over a frozen prognostic backbone.

Model A predicts state and diagnostics jointly. Model B splits them: the
backbone's 52 prognostic channels are frozen, and a smaller SFNO maps
(state, forcing) -> diagnostic distribution parameters. The head is small and
trains single-step, which is what makes running both variants affordable inside
a one-node budget.

The PhysicsNeMo `examples/weather/diagnostic` recipe uses an AFNO head; this uses
an SFNO so the head shares the backbone's spherical geometry rather than treating
the globe as a flat image.

    python train_head.py --config config/base.yaml --backbone <ckpt.pt>

Pass --backbone none to train the head directly on ground-truth state, which
isolates head capacity from backbone error.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PlasimSequenceDataset  # noqa: E402
from loss import DiagnosticLoss, default_precip_weight  # noqa: E402
from model import DiagnosticHead, PlasimEmulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="none",
                    help="checkpoint for the frozen prognostic model, or 'none'")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = cfg["data"]["root"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat = np.load(os.path.join(root, "stats", "lat.npy"))

    backbone = None
    if args.backbone != "none":
        ck = torch.load(args.backbone, map_location="cpu")
        backbone = PlasimEmulator(
            n_noise=cfg["model"]["n_noise"],
            probabilistic=cfg["model"]["probabilistic"],
            **cfg["model"].get("sfno", {}),
        )
        backbone.load_state_dict(ck["model"])
        backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

    hcfg = dict(cfg["model"].get("head", {}))
    if args.smoke:
        hcfg.update(embed_dim=32, num_layers=2)
    head = DiagnosticHead(
        probabilistic=cfg["model"]["probabilistic"], **hcfg
    ).to(device)
    print(f"head params: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M",
          flush=True)

    ds = PlasimSequenceDataset(root, "train", n_steps=1)
    dv = PlasimSequenceDataset(root, "valid", n_steps=1)
    nw = cfg["data"].get("num_workers", 8)
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                    num_workers=nw, pin_memory=True, drop_last=True)
    dlv = DataLoader(dv, batch_size=cfg["training"]["batch_size"], shuffle=False,
                     num_workers=nw, pin_memory=True)

    scales = json.load(open(os.path.join(root, "stats", "diagnostic_stats.json")))
    crit = DiagnosticLoss(
        lat, scales, weights=default_precip_weight(cfg["loss"]["precip_weight"])
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl), eta_min=1e-6
    )
    amp = torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float32

    for ep in range(args.epochs):
        head.train()
        t0, run, nb = time.time(), 0.0, 0
        for batch in dl:
            s0 = batch["state_in"].to(device, non_blocking=True)
            fo = batch["forcing"][:, 0].to(device, non_blocking=True)
            st = batch["state_tgt"][:, 0].to(device, non_blocking=True)
            dg = batch["diag_tgt"][:, 0].to(device, non_blocking=True)

            # The head consumes the state it will see at inference: the
            # backbone's prediction if there is one, else ground truth.
            with torch.no_grad():
                if backbone is not None:
                    with torch.autocast("cuda", dtype=amp,
                                        enabled=device.type == "cuda"):
                        state_for_head, _ = backbone(s0, fo)
                else:
                    state_for_head = st

            with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
                raw = head(state_for_head, fo)
                loss, parts = crit(raw, dg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
            nb += 1
            if args.smoke and nb >= 5:
                break
            if nb % 50 == 0:
                print(f"  ep{ep} it{nb} loss={run / nb:.4f} "
                      + " ".join(f"{k}={float(v):.4f}" for k, v in parts.items()),
                      flush=True)
        va, nv = 0.0, 0
        head.eval()
        with torch.no_grad():
            for i, batch in enumerate(dlv):
                if i >= cfg["training"].get("valid_iters", 50) or args.smoke:
                    break
                fo = batch["forcing"][:, 0].to(device)
                st = batch["state_tgt"][:, 0].to(device)
                dg = batch["diag_tgt"][:, 0].to(device)
                s0 = batch["state_in"].to(device)
                with torch.autocast("cuda", dtype=amp,
                                    enabled=device.type == "cuda"):
                    sfh = backbone(s0, fo)[0] if backbone is not None else st
                    va += float(crit(head(sfh, fo), dg)[0])
                nv += 1
        print(f"epoch {ep} train={run / max(nb,1):.4f} valid={va / max(nv,1):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        d = cfg["training"]["checkpoint_dir"] + "_head"
        os.makedirs(d, exist_ok=True)
        torch.save({"head": head.state_dict(), "epoch": ep, "cfg": cfg},
                   os.path.join(d, "ckpt_head.pt"))
        if args.smoke:
            break


if __name__ == "__main__":
    main()
