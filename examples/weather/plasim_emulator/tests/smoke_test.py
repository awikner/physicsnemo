# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone wiring checks - no packed data required.

Covers the failure modes that are silent rather than loud:
  * the diagnostic channels leaking into the autoregressive feedback path
  * the noise channels being ignored, collapsing the "ensemble" to identical
    members while every loss curve still looks healthy
  * hurdle-Gamma parameters going invalid or NaN at y = 0

Run on a compute node (importing torch on a login node gets OOM-killed):
    python tests/smoke_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import (  # noqa: E402
    GRID_SHAPE,
    N_DIAG_PARAMS,
    N_DIAGNOSTIC,
    N_FORCING,
    N_STATE,
)
from distributions import HurdleGamma  # noqa: E402
from loss import DiagnosticLoss, StateLoss  # noqa: E402
from model import PlasimEmulator, rollout  # noqa: E402

OK, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((OK if cond else FAIL, name, detail))
    print(f"[{OK if cond else FAIL}] {name} {detail}", flush=True)


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type == "cuda":
        print(f"gpus visible: {torch.cuda.device_count()} "
              f"({torch.cuda.get_device_name(0)})")

    B = 2
    H, W = GRID_SHAPE

    # --- tiny model so this runs anywhere -----------------------------------
    model = PlasimEmulator(
        n_noise=2, probabilistic=True, embed_dim=32, num_layers=2
    ).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    check("SFNO builds from makani", True, f"({n_par/1e6:.2f}M params)")
    check("input channel count", model.inp_chans == N_STATE + N_FORCING + 2,
          f"= {model.inp_chans}")
    check("output channel count", model.out_chans == N_STATE + N_DIAG_PARAMS,
          f"= {model.out_chans}")

    state = torch.randn(B, N_STATE, H, W, device=dev)
    forcing = torch.randn(B, N_FORCING, H, W, device=dev)

    s1, d1 = model(state, forcing)
    check("state output shape", tuple(s1.shape) == (B, N_STATE, H, W),
          str(tuple(s1.shape)))
    check("diagnostic output shape",
          tuple(d1.shape) == (B, N_DIAG_PARAMS, H, W), str(tuple(d1.shape)))

    # --- THE SPLIT: diagnostics must never re-enter the loop ----------------
    fseq = torch.randn(B, 3, N_FORCING, H, W, device=dev)
    ss, dd = rollout(model, state, fseq)
    check("rollout feeds back exactly 52 channels", ss.shape[2] == N_STATE,
          f"state {tuple(ss.shape)}")
    check("rollout keeps diagnostics separate", dd.shape[2] == N_DIAG_PARAMS,
          f"diag {tuple(dd.shape)}")

    # --- ensemble must actually diverge under different noise draws ---------
    g1 = torch.Generator(device=dev).manual_seed(1)
    g2 = torch.Generator(device=dev).manual_seed(2)
    with torch.no_grad():
        a, _ = rollout(model, state, fseq, generator=g1)
        b, _ = rollout(model, state, fseq, generator=g2)
    spread = float((a - b).abs().mean())
    check("noise produces distinct trajectories", spread > 1e-6,
          f"mean|a-b| = {spread:.3e}")

    # A model with n_noise=0 must be exactly reproducible - the control case.
    det = PlasimEmulator(
        n_noise=0, probabilistic=True, embed_dim=32, num_layers=2
    ).to(dev)
    with torch.no_grad():
        c1, _ = rollout(det, state, fseq)
        c2, _ = rollout(det, state, fseq)
    check("n_noise=0 is deterministic", torch.allclose(c1, c2),
          f"max|c1-c2| = {float((c1-c2).abs().max()):.3e}")

    # --- hurdle-Gamma validity, including the y == 0 branch -----------------
    raw = torch.randn(B, 3, H, W, device=dev, requires_grad=True)
    dist = HurdleGamma(raw, scale=1e-4)
    p, k, th = dist.p, dist.k, dist.theta
    check("0 < p < 1", bool(((p > 0) & (p < 1)).all()))
    check("k > 0 and theta > 0", bool((k > 0).all() and (th > 0).all()))

    y = torch.rand(B, H, W, device=dev) * 1e-4
    y[y < 3e-5] = 0.0  # a realistic point mass at zero
    nll = dist.nll(y)
    check("NLL finite with exact zeros present", bool(torch.isfinite(nll).all()),
          f"zero fraction {(y == 0).float().mean():.2f}")

    nll.mean().backward()
    check("NLL gradient finite", bool(torch.isfinite(raw.grad).all())
          if raw.grad is not None else False)

    check("mean() finite", bool(torch.isfinite(dist.mean()).all()))
    ex = dist.exceedance(1e-4)
    check("exceedance in [0,1]", bool(((ex >= 0) & (ex <= 1)).all()))
    smp = dist.sample()
    check("samples non-negative and finite",
          bool((smp >= 0).all() and torch.isfinite(smp).all()))

    # --- losses run end to end ---------------------------------------------
    lat = torch.linspace(87.86, -87.86, H).numpy()
    sl = StateLoss(lat).to(dev)
    check("state loss finite", bool(torch.isfinite(sl(s1, torch.randn_like(s1)))))

    scales = {"pr_6h": {"scale": 1e-4}, "evap": {"scale": 1e-7}}
    dl = DiagnosticLoss(lat, scales, weights={"pr_6h": 5.0, "evap": 1.0}).to(dev)
    dtgt = torch.rand(B, N_DIAGNOSTIC, H, W, device=dev) * 1e-4
    dtgt[dtgt < 3e-5] = 0.0
    tot, parts = dl(d1, dtgt)
    check("diagnostic loss finite", bool(torch.isfinite(tot)),
          " ".join(f"{k}={float(v):.3f}" for k, v in parts.items()))

    # --- overfit a single batch: catches wiring that trains but learns nothing
    small = PlasimEmulator(
        n_noise=0, probabilistic=True, embed_dim=32, num_layers=2
    ).to(dev)
    opt = torch.optim.Adam(small.parameters(), lr=1e-3)
    tgt = torch.randn(B, N_STATE, H, W, device=dev)
    first = last = None
    for i in range(30):
        ps, _ = small(state, forcing)
        loss = ((ps - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    check("overfits a single batch", last < first * 0.7,
          f"{first:.4f} -> {last:.4f}")

    nfail = sum(1 for r, _, _ in results if r == FAIL)
    print(f"\n{len(results) - nfail}/{len(results)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
