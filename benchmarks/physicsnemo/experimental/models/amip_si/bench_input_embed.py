# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Compare RollingDiT input-projection variants before committing to a run.

Port of upstream amip_v2 ``tools/bench_input_embed.py`` (Phase 12e.17). For
each variant it reports the projection's parameter count, the realised channel
budget, and — the actually interesting number — how far the *token*
representation moves under a +1σ shift of the boundary forcings, of SST alone,
and of the trend scalar alone, each relative to a +1σ shift of the whole
state. That ratio is the concrete form of "do the forcings have a voice": the
legacy concat gives CO₂ ~2 effective channels of 1024, and the point of budget
mode is to make that a number you choose.

Run::

    python benchmarks/physicsnemo/experimental/models/amip_si/bench_input_embed.py
"""

from __future__ import annotations

import argparse

import torch

from physicsnemo.experimental.models.amip_si.layers.input_embed import (
    RollingDiTInputEmbed,
)

# The upstream ERDM_co2 state contract: 6 surface + 15 diagnostic + 5x26 upper air.
LAYOUT = dict(nsurface=6, ndiagnostic=15, nlevels=26, n_upper_air=5)
IN_CHANNELS = LAYOUT["nsurface"] + LAYOUT["ndiagnostic"] + LAYOUT["nlevels"] * LAYOUT["n_upper_air"]

VARIANTS = {
    "budget-flat": dict(state_encoder="flat"),
    "budget-column": dict(state_encoder="column"),
    "budget-column-conv1": dict(state_encoder="column", boundary_encoder="conv1"),
    "budget-column-nostats": dict(state_encoder="column", boundary_pool_stats=False),
    "budget-column-nobias": dict(state_encoder="column", boundary_static_bias=False),
    # Fraction of dim so the sweep scales with --dim (384/1024 upstream).
    "budget-column-d_boundary=3/8": dict(state_encoder="column", d_boundary_frac=0.375),
    "budget-column-nosourcenorm": dict(state_encoder="column", source_norm=False),
}


def _sensitivity(embed, x, cg, cs, *, sst_index=2):
    """Relative token movement under +1σ shifts of each source."""
    with torch.no_grad():
        base, _ = embed(x, cg, cs)

        def delta(**kw):
            tok, _ = embed(kw.get("x", x), kw.get("cg", cg), kw.get("cs", cs))
            return (tok - base).norm().item()

        d_state = delta(x=x + 1.0)
        if d_state == 0:
            return {}
        cg_sst = cg.clone()
        cg_sst[:, sst_index] += 1.0
        cs_trend = cs.clone()
        if cs.shape[1] >= 3:
            cs_trend[:, 2] += 1.0
        return {
            "boundary/state": delta(cg=cg + 1.0) / d_state,
            "sst/state": delta(cg=cg_sst) / d_state,
            "trend/state": delta(cs=cs_trend) / d_state,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--nlat", type=int, default=45)
    p.add_argument("--nlon", type=int, default=90)
    p.add_argument("--c-grid-dim", type=int, default=5)
    p.add_argument("--scalar-dim", type=int, default=3)
    p.add_argument("--downsample", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(0)
    x = torch.randn(2, IN_CHANNELS, args.nlat, args.nlon)
    cg = torch.randn(2, args.c_grid_dim, args.nlat * args.downsample, args.nlon * args.downsample)
    cs = torch.randn(2, args.scalar_dim)

    print(f"{'variant':<34} {'params':>12} {'d_state':>8} {'d_bnd':>6} {'d_cal':>6} "
          f"{'bnd/state':>10} {'sst/state':>10} {'trend/state':>12}")
    for name, kw in VARIANTS.items():
        kw = dict(kw)
        frac = kw.pop("d_boundary_frac", None)
        if frac is not None:
            kw["d_boundary"] = int(args.dim * frac)
        try:
            embed = RollingDiTInputEmbed(
                dim=args.dim, in_channels=IN_CHANNELS, nlat=args.nlat, nlon=args.nlon,
                c_grid_dim=args.c_grid_dim, scalar_dim=args.scalar_dim,
                c_grid_downsample=args.downsample, **LAYOUT, **kw,
            ).eval()
        except ValueError as exc:
            # A budget that cannot fit this --dim (the layers check, loudly);
            # report and keep sweeping rather than killing the run.
            print(f"{name:<34} {'skipped':>12}  {exc}")
            continue
        d = embed.describe()
        s = _sensitivity(embed, x, cg, cs)
        print(f"{name:<34} {sum(q.numel() for q in embed.parameters()):>12,} "
              f"{d['d_state']:>8} {d['d_boundary']:>6} {d['d_calendar']:>6} "
              f"{s.get('boundary/state', float('nan')):>10.3f} "
              f"{s.get('sst/state', float('nan')):>10.3f} "
              f"{s.get('trend/state', float('nan')):>12.3f}")


if __name__ == "__main__":
    main()
