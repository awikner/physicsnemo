# SPDX-License-Identifier: Apache-2.0
"""Forecast skill vs lead time for precipitation and 2 m temperature.

Scores RMSE and CRPS at every 6 h step to 15 days, globally (cos-lat weighted)
and over a 3x3 AI-RES region stencil, for the v11 baseline and our stages.

t2m (`tas`) is the clean comparison: it is a prognostic state channel and v11's
state channels share our units and climatology (its tas mean 277.87 K vs our
277.6), so no rescaling is involved. Precipitation needs the z-score matching
described in tools/compare_precip.py, because v11's precip channel is a
different distribution, not merely different units.

CRPS is computed from a batched ensemble of state rollouts driven by the model's
own noise channels. Stages 1-3 collapsed that ensemble (spread ~0.04% of signal),
so for them CRPS is effectively MAE; only stage 4, trained with the energy score,
has real spread. That difference is the point of the plot, not an artifact.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import DIAGNOSTIC_CHANNELS, FORCING_CHANNELS, PRECIP_CHANNEL  # noqa: E402
from dataset import PlasimSequenceDataset                                    # noqa: E402
from distributions import HurdleGamma                                        # noqa: E402
from model import PlasimEmulator                                             # noqa: E402
import v11_adapter as V11                                                    # noqa: E402

REGIONS = json.load(open("/work2/11079/aasch/stampede3/AI-RES-clean/plasim/regions.json"))
MM = 1000.0 * 4.0          # pr_6h m/6h -> mm/day
TAS_IDX = 1
M_ENS = 8                  # state-ensemble members
N_PR = 16                  # precip distribution draws


def near(ax, vals):
    return [int(np.argmin(np.abs(ax - v))) for v in vals]


def reg_mean(x, w, box=None):
    if box is None:
        return (x * w).mean(dim=(-2, -1))
    la, lo = box
    return x[..., la, :][..., lo].mean(dim=(-2, -1))


def kcrps(ens, obs):
    M = ens.shape[0]
    if M == 1:
        return (ens[0] - obs).abs()          # deterministic -> MAE
    t1 = (ens - obs.unsqueeze(0)).abs().mean(0)
    t2 = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum((0, 1)) / (2 * M * M)
    return t1 - t2


@torch.no_grad()
def run_ours(ck, cfg, ds, idxs, T, dev, scales):
    m = PlasimEmulator(n_noise=cfg["model"]["n_noise"],
                       probabilistic=cfg["model"]["probabilistic"],
                       **cfg["model"].get("sfno", {})).to(dev).eval()
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
    pi = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)
    sm = torch.tensor(ds.state_mean).to(dev); ss = torch.tensor(ds.state_std).to(dev)
    out = []
    for i in idxs:
        b = ds[i]
        s = b["state_in"][None].to(dev).repeat(M_ENS, 1, 1, 1)   # batched ensemble
        fo = b["forcing"][None].to(dev)
        dg = b["diag_tgt"][None].to(dev)
        st = b["state_tgt"][None].to(dev)
        tas_e, pr_m, pr_s = [], [], []
        for t in range(T):
            f = fo[:, t].repeat(M_ENS, 1, 1, 1)
            s, d = m(s, f)
            tas_e.append(s[:, TAS_IDX] * ss[0, TAS_IDX] + sm[0, TAS_IDX])
            d0 = d[:1, 3 * pi:3 * (pi + 1)]
            dist = HurdleGamma(d0, scale=scales[pi])
            pr_m.append(dist.mean()[0] * MM)
            pr_s.append(torch.stack([dist.sample()[0] for _ in range(N_PR)], 0) * MM)
        out.append({
            "tas_ens": torch.stack(tas_e, 1),                       # [M,T,H,W]
            "tas_true": (st[0, :, TAS_IDX] * ss[0, TAS_IDX] + sm[0, TAS_IDX]),
            "pr_mean": torch.stack(pr_m), "pr_ens": torch.stack(pr_s, 1),
            "pr_true": dg[0, :, pi] * MM,
        })
    return out


@torch.no_grad()
def run_v11(ds, idxs, T, dev, pr_mean, pr_std):
    net, vcfg, missing, _ = V11.build_v11(dev)
    if missing:
        print(f"  WARNING v11 missing {len(missing)} tensors e.g. {missing[:3]}")
    m, s, fm, fs = V11.v11_stats(dev)
    pi = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)
    v11m, v11s = float(m.reshape(-1)[52]), float(s.reshape(-1)[52])
    sm = torch.tensor(ds.state_mean).to(dev); ss = torch.tensor(ds.state_std).to(dev)
    fmn = torch.tensor(ds.forcing_mean).to(dev); fsd = torch.tensor(ds.forcing_std).to(dev)
    out, gate = [], []
    for i in idxs:
        year, t0 = ds.index[i]
        sic, sst = V11.v11_cyclic(year)
        b = ds[i]
        sp = b["state_in"][None].to(dev) * ss + sm
        fo = b["forcing"][None].to(dev) * fsd + fmn
        dg = b["diag_tgt"][None].to(dev); st = b["state_tgt"][None].to(dev)
        z = (sp - m[:, :52]) / s[:, :52]
        tas, pr = [], []
        for t in range(T):
            ti = (t0 + t + 1) % sic.shape[0]
            f = torch.empty(1, 6, 64, 128, device=dev)
            for j, n in enumerate(V11.V11_FORCING):
                if n == "sic":   f[:, j] = torch.tensor(sic[ti], device=dev)
                elif n == "sst": f[:, j] = torch.tensor(sst[ti], device=dev)
                else:            f[:, j] = fo[:, t, FORCING_CHANNELS.index(n)]
            y = net(torch.cat([z, (f - fm) / fs], 1))
            z = y[:, :52]
            phys = z * s[:, :52] + m[:, :52]
            tas.append(phys[0, TAS_IDX])
            praw = y[:, 52] * v11s + v11m
            pr.append(((praw[0] - v11m) / v11s * pr_std + pr_mean) * MM)
            if t == 0:
                truth = st[0, 0, TAS_IDX] * ss[0, TAS_IDX] + sm[0, TAS_IDX]
                gate.append(float((phys[0, TAS_IDX] - truth).pow(2).mean().sqrt()))
        out.append({
            "tas_ens": torch.stack(tas)[None],                      # [1,T,H,W]
            "tas_true": st[0, :, TAS_IDX] * ss[0, TAS_IDX] + sm[0, TAS_IDX],
            "pr_mean": torch.stack(pr), "pr_ens": None,
            "pr_true": dg[0, :, pi] * MM,
        })
    return out, gate


def score(series, w, boxes, T):
    keys = []
    for var in ("tas", "pr"):
        for met in ("rmse", "crps"):
            for rn in ["global"] + list(boxes):
                keys.append(f"{var}_{met}_{rn}")
    res = {k: np.zeros(T) for k in keys}
    for s in series:
        for var in ("tas", "pr"):
            ens = s[f"{var}_ens"]
            truth = s[f"{var}_true"]
            mean = ens.mean(0) if ens is not None else s["pr_mean"]
            for t in range(T):
                e2 = (mean[t] - truth[t]) ** 2
                c = kcrps(ens[:, t], truth[t]) if ens is not None \
                    else (mean[t] - truth[t]).abs()
                for rn, box in [("global", None)] + list(boxes.items()):
                    res[f"{var}_rmse_{rn}"][t] += float(torch.sqrt(reg_mean(e2, w, box)))
                    res[f"{var}_crps_{rn}"][t] += float(reg_mean(c, w, box))
    n = len(series)
    return {k: (v / n).tolist() for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--regions", default="Chicago,Bay_of_Bengal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(a.config)); root = cfg["data"]["root"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = a.days * 4
    lat = np.load(os.path.join(root, "stats", "lat.npy"))
    lon = np.load(os.path.join(root, "stats", "lon.npy"))
    wl = np.cos(np.deg2rad(lat)); wl /= wl.mean()
    w = torch.tensor(wl, dtype=torch.float32, device=dev)[:, None]
    boxes = {r: (near(lat, REGIONS[r]["lat"]), near(lon, REGIONS[r]["lon"]))
             for r in a.regions.split(",")}
    print("regions:", {k: (v[0], v[1]) for k, v in boxes.items()})

    ds = PlasimSequenceDataset(root, a.split, n_steps=T)
    stride = max(1, len(ds) // a.samples)
    idxs = list(range(0, min(len(ds), a.samples * stride), stride))
    dst = json.load(open(os.path.join(root, "stats", "diagnostic_stats.json")))
    scales = [dst[n]["scale"] for n in DIAGNOSTIC_CHANNELS]

    res = {"lead_hours": [6 * (t + 1) for t in range(T)],
           "regions": list(boxes), "models": {}}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        print(f"=== {name} ===", flush=True)
        if path == "builtin":
            ser, gate = run_v11(ds, idxs, T, dev, dst[PRECIP_CHANNEL]["mean"], 1.22302e-3)
            g = float(np.mean(gate)); print(f"  gate 1-step tas RMSE = {g:.3f} K")
            if g > 5.0:
                print("  GATE FAILED - skipping v11"); continue
            res["models"][name] = score(ser, w, boxes, T)
            res["models"][name]["gate_1step_tas_rmse_K"] = g
        else:
            ser = run_ours(path, cfg, ds, idxs, T, dev, scales)
            res["models"][name] = score(ser, w, boxes, T)
        r = res["models"][name]
        for d in (1, 5, 15):
            t = d * 4 - 1
            print(f"  +{d:2d}d tas_rmse_glob={r['tas_rmse_global'][t]:.3f}K "
                  f"tas_rmse_Chicago={r['tas_rmse_Chicago'][t]:.3f}K "
                  f"tas_crps_glob={r['tas_crps_global'][t]:.3f}K", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
