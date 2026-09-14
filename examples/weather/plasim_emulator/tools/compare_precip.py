# SPDX-License-Identifier: Apache-2.0
"""Precipitation skill vs lead time: our stages against the v11 baseline.

Reports RMSE and CRPS in mm/day at every 6 h step out to 15 days, globally
(latitude weighted) and over the AI-RES Bay-of-Bengal box.

v11 caveat: its precipitation channel is NOT our field in different units - the
climatological mean ratio (3499) and std ratio (4354) disagree, so the fields
differ in shape, most likely because v11's postprocessing interpolates. Its
precip is therefore mapped onto our distribution by z-score matching before
scoring, which turns the comparison into one of PATTERN skill rather than
absolute calibration. Absolute v11 precip values are not comparable.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import DIAGNOSTIC_CHANNELS, PRECIP_CHANNEL          # noqa: E402
from dataset import PlasimSequenceDataset                          # noqa: E402
from distributions import HurdleGamma                              # noqa: E402
from model import PlasimEmulator                                   # noqa: E402
import v11_adapter as V11                                          # noqa: E402

BOB_LON = [84.375, 87.1875, 90.0]
BOB_LAT = [9.767145559195571, 12.557756115230681, 15.348364759491487]
MM = 1000.0 * 4.0            # m/6h -> mm/day
N_ENS = 16


def near(ax, vals):
    return [int(np.argmin(np.abs(ax - v))) for v in vals]


def wmean(x, w, box=None):
    if box is not None:
        la, lo = box
        return x[..., la, :][..., lo].mean(dim=(-2, -1))
    return (x * w).mean(dim=(-2, -1))


def kcrps(ens, obs):
    M = ens.shape[0]
    t1 = (ens - obs.unsqueeze(0)).abs().mean(0)
    t2 = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum((0, 1)) / (2 * M * M)
    return t1 - t2


@torch.no_grad()
def run_ours(ck_path, cfg, ds, idxs, T, device, scales):
    m = PlasimEmulator(n_noise=cfg["model"]["n_noise"],
                       probabilistic=cfg["model"]["probabilistic"],
                       **cfg["model"].get("sfno", {})).to(device).eval()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    pi = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)
    sm = torch.tensor(ds.state_mean).to(device); ss = torch.tensor(ds.state_std).to(device)
    out = []
    for i in idxs:
        b = ds[i]
        s = b["state_in"][None].to(device)
        fo = b["forcing"][None].to(device)
        dg = b["diag_tgt"][None].to(device)
        prm, prs, tgt = [], [], []
        for t in range(T):
            s, d = m(s, fo[:, t])
            dist = HurdleGamma(d[:, 3 * pi:3 * (pi + 1)], scale=scales[pi])
            prm.append(dist.mean()[0] * MM)
            prs.append(torch.stack([dist.sample()[0] for _ in range(N_ENS)], 0) * MM)
            tgt.append(dg[0, t, pi] * MM)
        out.append((torch.stack(prm), torch.stack(prs, 1), torch.stack(tgt)))
    return out


@torch.no_grad()
def run_v11(ds, idxs, T, device, our_pr_mean, our_pr_std):
    net, vcfg, missing, unexpected = V11.build_v11(device)
    if missing:
        print(f"  WARNING v11 missing {len(missing)} tensors e.g. {missing[:3]}")
    m, s, fm, fs = V11.v11_stats(device)
    pi = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)
    v11m, v11s = float(m.reshape(-1)[52]), float(s.reshape(-1)[52])

    fnames = ds_forcing_names()
    out, gate = [], []
    for i in idxs:
        year, t0 = ds.index[i]
        sic, sst = V11.v11_cyclic(year)
        b = ds[i]
        # our packed state is z-scored with OUR stats -> back to physical
        sp = b["state_in"][None].to(device) * torch.tensor(ds.state_std).to(device) \
             + torch.tensor(ds.state_mean).to(device)
        fo_phys = b["forcing"][None].to(device) * torch.tensor(ds.forcing_std).to(device) \
                  + torch.tensor(ds.forcing_mean).to(device)
        dg = b["diag_tgt"][None].to(device)

        z = (sp - m[:, :52]) / s[:, :52]
        prm, tgt = [], []
        for t in range(T):
            ti = (t0 + t + 1) % sic.shape[0]
            f = torch.empty(1, 6, 64, 128, device=device)
            for j, n in enumerate(V11.V11_FORCING):
                if n == "sic":
                    f[:, j] = torch.tensor(sic[ti], device=device)
                elif n == "sst":
                    f[:, j] = torch.tensor(sst[ti], device=device)
                else:
                    f[:, j] = fo_phys[:, t, fnames.index(n)]
            fz = (f - fm) / fs
            y = net(torch.cat([z, fz], 1))
            z = y[:, :52]
            pr = y[:, 52] * v11s + v11m
            # z-score match onto our precip distribution (see module docstring)
            prm.append(((pr[0] - v11m) / v11s * our_pr_std + our_pr_mean) * MM)
            tgt.append(dg[0, t, pi] * MM)
            if t == 0:
                phys = z * s[:, :52] + m[:, :52]
                gate.append(float((phys[0, 1] - (b["state_tgt"][0, 1].to(device)
                            * torch.tensor(ds.state_std).to(device)[0, 1]
                            + torch.tensor(ds.state_mean).to(device)[0, 1])).pow(2).mean().sqrt()))
        out.append((torch.stack(prm), None, torch.stack(tgt)))
    return out, gate


def ds_forcing_names():
    from channels import FORCING_CHANNELS
    return list(FORCING_CHANNELS)


def score(series, w, box, T):
    """series: list of (mean[T,H,W], samples[N,T,H,W] or None, truth[T,H,W])."""
    res = {k: np.zeros(T) for k in ["rmse_global", "rmse_bob", "crps_global", "crps_bob"]}
    n = len(series)
    for mean, samp, tgt in series:
        for t in range(T):
            e2 = (mean[t] - tgt[t]) ** 2
            res["rmse_global"][t] += float(torch.sqrt(wmean(e2, w)))
            res["rmse_bob"][t] += float(torch.sqrt(wmean(e2, w, box)))
            if samp is not None:
                c = kcrps(samp[:, t], tgt[t])
            else:                       # deterministic: CRPS reduces to MAE
                c = (mean[t] - tgt[t]).abs()
            res["crps_global"][t] += float(wmean(c, w))
            res["crps_bob"][t] += float(wmean(c, w, box))
    return {k: (v / n).tolist() for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True,
                    help="name=/path/ckpt.pt,... ; use v11=builtin for the baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(a.config))
    root = cfg["data"]["root"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = a.days * 4

    lat = np.load(os.path.join(root, "stats", "lat.npy"))
    lon = np.load(os.path.join(root, "stats", "lon.npy"))
    wl = np.cos(np.deg2rad(lat)); wl = wl / wl.mean()
    w = torch.tensor(wl, dtype=torch.float32, device=dev)[:, None]
    box = (near(lat, BOB_LAT), near(lon, BOB_LON))

    ds = PlasimSequenceDataset(root, a.split, n_steps=T)
    stride = max(1, len(ds) // a.samples)
    idxs = list(range(0, min(len(ds), a.samples * stride), stride))
    dstats = json.load(open(os.path.join(root, "stats", "diagnostic_stats.json")))
    scales = [dstats[n]["scale"] for n in DIAGNOSTIC_CHANNELS]
    our_pr_mean = dstats[PRECIP_CHANNEL]["mean"]
    our_pr_std = float(np.sqrt(max(dstats[PRECIP_CHANNEL].get("var", 0.0),
                                   (1.22302e-3) ** 2)))

    results = {"lead_hours": [6 * (t + 1) for t in range(T)], "models": {}}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        print(f"=== {name} ===", flush=True)
        if path == "builtin":
            series, gate = run_v11(ds, idxs, T, dev, our_pr_mean, our_pr_std)
            g = float(np.mean(gate))
            print(f"  v11 gate: 1-step tas RMSE = {g:.3f} K")
            if g > 5.0:
                print("  GATE FAILED - v11 input reconstruction is wrong; skipping")
                continue
            results["models"][name] = score(series, w, box, T)
            results["models"][name]["gate_1step_tas_rmse_K"] = g
            results["models"][name]["note"] = "precip z-score matched onto our distribution"
        else:
            series = run_ours(path, cfg, ds, idxs, T, dev, scales)
            results["models"][name] = score(series, w, box, T)
        r = results["models"][name]
        for d in (1, 5, 15):
            t = d * 4 - 1
            if t < T:
                print(f"  +{d:2d}d rmse_glob={r['rmse_global'][t]:.3f} "
                      f"rmse_bob={r['rmse_bob'][t]:.3f} "
                      f"crps_glob={r['crps_global'][t]:.3f} mm/day", flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
