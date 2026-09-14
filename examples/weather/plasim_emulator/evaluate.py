# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the PlaSim emulator: deterministic skill, AR stability, and calibration.

Deterministic skill alone is not enough here. An ensemble can look skilful while
being badly overconfident, which would silently corrupt a rare-event pipeline, so
spread-skill ratio and rank histograms are reported at every lead time alongside
RMSE/ACC.

Precipitation is additionally scored at the aggregation AI-RES actually uses:
a daily mean, then a 7-day rolling window, over a 3x3 cell Bay-of-Bengal box, in
mm/day - not 6-hourly gridpoint RMSE.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channels import DIAGNOSTIC_CHANNELS, PRECIP_CHANNEL, STATE_CHANNELS  # noqa: E402
from dataset import PlasimSequenceDataset  # noqa: E402
from distributions import HurdleGamma  # noqa: E402
from model import PlasimEmulator  # noqa: E402

# AI-RES scoring region (plasim/regions.json) and unit conversion.
BOB_LON = [84.375, 87.1875, 90.0]
BOB_LAT = [9.767145559195571, 12.557756115230681, 15.348364759491487]
MM_PER_M = 1000.0
STEPS_PER_DAY = 4.0
PRECIP_THRESHOLDS_MMDAY = [1.0, 10.0, 50.0]


def lat_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def nearest_idx(axis, values):
    return [int(np.argmin(np.abs(axis - v))) for v in values]


@torch.no_grad()
def ensemble_rollout(model, state0, forcing_seq, n_members, device, seed=0):
    """[M, T, ...] ensemble by re-running the rollout with different noise draws."""
    outs_s, outs_d = [], []
    for m in range(n_members):
        g = torch.Generator(device=device)
        g.manual_seed(seed * 100003 + m)
        s = state0.clone()
        ss, dd = [], []
        for t in range(forcing_seq.shape[1]):
            noise = model.draw_noise(s.shape[0], device, s.dtype, g)
            s, d = model(s, forcing_seq[:, t], noise)
            ss.append(s)
            dd.append(d)
        outs_s.append(torch.stack(ss, 1))
        outs_d.append(torch.stack(dd, 1))
    return torch.stack(outs_s, 0), torch.stack(outs_d, 0)


def kcrps(ens, obs):
    """Kernel CRPS. ens [M, ...], obs [...]. Mirrors
    physicsnemo.metrics.general.crps.kcrps."""
    M = ens.shape[0]
    t1 = (ens - obs.unsqueeze(0)).abs().mean(0)
    t2 = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum((0, 1)) / (2 * M * M)
    return t1 - t2


def rank_of_obs(ens, obs):
    """Rank of the observation within the ensemble; flat histogram == calibrated."""
    return (ens < obs.unsqueeze(0)).sum(0)


def spread_skill(ens, obs, w):
    """(ensemble spread, rmse of ensemble mean). Ratio should be ~1."""
    mean = ens.mean(0)
    var = ens.var(0, unbiased=True)
    M = ens.shape[0]
    # Inflate spread by sqrt((M+1)/M) so it is comparable to the error of a
    # finite-size ensemble mean.
    spread = torch.sqrt((var * (M + 1) / M * w).mean())
    rmse = torch.sqrt((((mean - obs) ** 2) * w).mean())
    return float(spread), float(rmse)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--lead-steps", type=int, default=40, help="40 x 6h = 10 days")
    ap.add_argument("--members", type=int, default=16,
                    help="AI-RES operational ensemble size")
    ap.add_argument("--samples", type=int, default=32, help="ICs to evaluate")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.config))
    root = cfg["data"]["root"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lat = np.load(os.path.join(root, "stats", "lat.npy"))
    lon = np.load(os.path.join(root, "stats", "lon.npy"))
    w = lat_weights(lat).to(device)[None, :, None]

    ck = torch.load(args.checkpoint, map_location="cpu")
    model = PlasimEmulator(
        n_noise=cfg["model"]["n_noise"],
        probabilistic=cfg["model"]["probabilistic"],
        **cfg["model"].get("sfno", {}),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = PlasimSequenceDataset(root, args.split, n_steps=args.lead_steps)
    stats = json.load(open(os.path.join(root, "stats", "diagnostic_stats.json")))
    scales = [stats[n]["scale"] for n in DIAGNOSTIC_CHANNELS]
    pr_i = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)

    state_mean = torch.tensor(ds.state_mean).to(device)
    state_std = torch.tensor(ds.state_std).to(device)

    bob = (nearest_idx(lat, BOB_LAT), nearest_idx(lon, BOB_LON))

    rmse = np.zeros(args.lead_steps)
    spread = np.zeros(args.lead_steps)
    crps_pr = np.zeros(args.lead_steps)
    ranks = []
    brier = {t: np.zeros(args.lead_steps) for t in PRECIP_THRESHOLDS_MMDAY}
    n_used = 0
    blew_up = 0

    stride = max(1, len(ds) // args.samples)
    for si in range(0, min(len(ds), args.samples * stride), stride):
        b = ds[si]
        s0 = b["state_in"][None].to(device)
        fo = b["forcing"][None].to(device)
        st = b["state_tgt"][None].to(device)
        dg = b["diag_tgt"][None].to(device)

        ens_s, ens_d = ensemble_rollout(
            model, s0, fo, args.members, device, seed=si
        )  # [M,1,T,C,H,W]
        ens_s = ens_s[:, 0]
        ens_d = ens_d[:, 0]

        if not torch.isfinite(ens_s).all():
            blew_up += 1
            continue

        for t in range(args.lead_steps):
            # --- state: tas (channel 1) as the headline deterministic field
            e = ens_s[:, t, 1]
            o = st[0, t, 1]
            sp, rm = spread_skill(e, o, w)
            spread[t] += sp
            rmse[t] += rm
            ranks.append(int(rank_of_obs(e, o).float().mean()))

            # --- precipitation: sample the predictive distribution, score in mm/day
            raw = ens_d[:, t, 3 * pr_i : 3 * (pr_i + 1)]
            dist = HurdleGamma(raw, scale=scales[pr_i])
            draws = dist.sample() * MM_PER_M * STEPS_PER_DAY
            obs_pr = dg[0, t, pr_i] * MM_PER_M * STEPS_PER_DAY
            crps_pr[t] += float((kcrps(draws, obs_pr) * w).mean())

            for thr in PRECIP_THRESHOLDS_MMDAY:
                p = dist.exceedance(thr / (MM_PER_M * STEPS_PER_DAY))
                ph = p.mean(0)
                oh = (obs_pr > thr).float()
                brier[thr][t] += float((((ph - oh) ** 2) * w).mean())
        n_used += 1

    if n_used == 0:
        raise SystemExit("every rollout produced non-finite state - model diverged")

    res = {
        "n_samples": n_used,
        "n_diverged": blew_up,
        "members": args.members,
        "lead_hours": [6 * (t + 1) for t in range(args.lead_steps)],
        "tas_rmse": (rmse / n_used).tolist(),
        "tas_spread": (spread / n_used).tolist(),
        "tas_spread_skill_ratio": (spread / np.maximum(rmse, 1e-12)).tolist(),
        "precip_crps_mmday": (crps_pr / n_used).tolist(),
        "precip_brier": {str(k): (v / n_used).tolist() for k, v in brier.items()},
        "rank_histogram": np.bincount(
            np.array(ranks), minlength=args.members + 1
        ).tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"samples={n_used} diverged={blew_up}")
    for t in (0, min(3, args.lead_steps - 1), args.lead_steps - 1):
        print(f"  +{6*(t+1):3d}h  tas_rmse={res['tas_rmse'][t]:.4f} "
              f"spread/skill={res['tas_spread_skill_ratio'][t]:.3f} "
              f"precip_crps={res['precip_crps_mmday'][t]:.4f} mm/day")
    print(f"rank histogram (flat == calibrated): {res['rank_histogram']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
