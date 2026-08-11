# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Fit each RollingDiT output-head variant on the real EDM target.

Port of upstream amip_v2 ``tools/bench_output_head.py`` (Phase 12e.17). The
question a head has to answer is whether it can represent

    F_target = a(sigma) * y + b(sigma) * eps

across a window whose frames carry *different* sigma in one forward pass. This
fits each variant directly on that target, under the real f(sigma) loss
weighting, from an idealised trunk (a frozen random embedding of y and eps) so
the number reflects head expressiveness alone rather than trunk quality.

Run::

    python benchmarks/physicsnemo/experimental/models/amip_si/bench_output_head.py
"""

from __future__ import annotations

import argparse

import torch

from physicsnemo.experimental.models.amip_si.layers.output_head import (
    RollingDiTOutputHead,
)
from physicsnemo.experimental.models.amip_si.layers.unpatchify import Unpatchify

LAYOUT = dict(nsurface=6, ndiagnostic=15, nlevels=26, n_upper_air=5)
C_OUT = LAYOUT["nsurface"] + LAYOUT["ndiagnostic"] + LAYOUT["nlevels"] * LAYOUT["n_upper_air"]


def _edm_terms(sigma, sigma_data=1.0):
    """EDM preconditioning coefficients and the loss weight for one sigma."""
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    a = (1 - c_skip) / c_out
    b = -c_skip * sigma / c_out
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    return a, b, weight


def _make_batch(n, nlat, nlon, dim, sigmas, generator):
    """Idealised trunk output plus the matching F_target."""
    y = torch.randn(n, nlat * nlon, C_OUT, generator=generator)
    eps = torch.randn(n, nlat * nlon, C_OUT, generator=generator)
    # Frozen random embedding of (y, eps) stands in for a perfect trunk.
    proj = torch.randn(2 * C_OUT, dim, generator=generator) / (2 * C_OUT) ** 0.5
    h = torch.cat([y, eps], dim=-1) @ proj
    a, b, w = _edm_terms(sigmas.view(-1, 1, 1))
    target = a * y + b * eps
    cond = torch.log(sigmas).view(-1, 1) / 4.0
    return h, target, w, cond


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--nlat", type=int, default=8)
    p.add_argument("--nlon", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=6)
    args = p.parse_args()

    # The window's sigma span at global t=0 for the reference ERDM schedule.
    sigmas = torch.tensor([0.007, 0.030, 0.162, 1.241, 15.99, 500.0])[: args.batch]

    variants = {
        "legacy (fixed Linear)": lambda: Unpatchify(
            grid_size=(args.nlat, args.nlon), patch_size=(1, 1),
            in_dim=args.dim, out_dim=C_OUT, cond_dim=args.dim),
        "mix K=1": lambda: RollingDiTOutputHead(
            args.dim, C_OUT, args.nlat, args.nlon, args.dim, num_experts=1),
        "mix K=2": lambda: RollingDiTOutputHead(
            args.dim, C_OUT, args.nlat, args.nlon, args.dim, num_experts=2),
        "mix K=2 column": lambda: RollingDiTOutputHead(
            args.dim, C_OUT, args.nlat, args.nlon, args.dim, num_experts=2,
            decoder="column", **LAYOUT),
    }

    print(f"{'head':<24} {'params':>12} {'final weighted MSE':>20}")
    for name, build in variants.items():
        g = torch.Generator().manual_seed(0)
        h, target, w, cond = _make_batch(len(sigmas), args.nlat, args.nlon, args.dim, sigmas, g)
        torch.manual_seed(0)
        head = build()
        # The conditioning vector the real model feeds its head is the flow-time
        # embedding; here c_noise is broadcast to that width.
        cond_full = cond.expand(-1, args.dim).contiguous()
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        loss = float("nan")
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            out = head(h, cond_full).view(len(sigmas), args.nlat * args.nlon, C_OUT)
            loss_t = (w * (out - target) ** 2).mean()
            loss_t.backward()
            opt.step()
            loss = float(loss_t)
        print(f"{name:<24} {sum(q.numel() for q in head.parameters()):>12,} {loss:>20.6f}")


if __name__ == "__main__":
    main()
