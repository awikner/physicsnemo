# SPDX-License-Identifier: Apache-2.0
"""Does the round-2 smoothing cost us the extremes?

Round 2 cut daily-mean precip CRPS by ~34% at days 3-10, but its high-wavenumber
power ratio fell from ~0.93 to ~0.55. Better CRPS obtained by damping variance is
a bad trade for a rare-event pipeline, so this scores the quantity that actually
matters: daily-mean exceedance probability at MODERATE thresholds, plus the
predicted vs observed upper quantiles.

Thresholds stay <= 20 mm/day (~p99 of the daily field) and every one is reported
with its event count - a Brier score built on a handful of events is not evidence.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import DIAGNOSTIC_CHANNELS, PRECIP_CHANNEL     # noqa: E402
from dataset import PlasimSequenceDataset                     # noqa: E402
from distributions import HurdleGamma                         # noqa: E402
from metrics_daily import (DEFAULT_THRESHOLDS_MMDAY, aggregate_daily,  # noqa: E402
                           brier, exceedance_prob, kcrps)
from model import PlasimEmulator                              # noqa: E402

MM = 1000.0 * 4.0
M_ENS = 24


@torch.no_grad()
def run(ck_path, cfg, ds, idxs, T, dev, scale, pi):
    m = PlasimEmulator(n_noise=cfg["model"]["n_noise"],
                       probabilistic=cfg["model"]["probabilistic"],
                       **cfg["model"].get("sfno", {})).to(dev).eval()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    out = []
    for i in idxs:
        b = ds[i]
        s = b["state_in"][None].to(dev)
        fo = b["forcing"][None].to(dev)
        dg = b["diag_tgt"][None].to(dev)
        ens = []
        for t in range(T):
            s, d = m(s, fo[:, t])
            dist = HurdleGamma(d[:, 3 * pi:3 * (pi + 1)], scale=scale)
            ens.append(torch.stack([dist.sample()[0] for _ in range(M_ENS)], 0) * MM)
        out.append((torch.stack(ens, 1), dg[0, :, pi] * MM))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True, help="name=path,...")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(a.config)); root = cfg["data"]["root"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = a.days * 4
    lat = np.load(os.path.join(root, "stats", "lat.npy"))
    w = np.cos(np.deg2rad(lat)); w /= w.mean()
    W = torch.tensor(w, dtype=torch.float32, device=dev)[:, None]
    ds = PlasimSequenceDataset(root, "test", n_steps=T)
    stt = json.load(open(os.path.join(root, "stats", "diagnostic_stats.json")))
    pi = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)
    scale = stt[PRECIP_CHANNEL]["scale"]
    stride = max(1, len(ds) // a.samples)
    idxs = list(range(0, min(len(ds), a.samples * stride), stride))

    res = {}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        series = run(path, cfg, ds, idxs, T, dev, scale, pi)
        nday = T // 4
        br = {str(t): np.zeros(nday) for t in DEFAULT_THRESHOLDS_MMDAY}
        ev = {str(t): 0 for t in DEFAULT_THRESHOLDS_MMDAY}
        q99p, q99o, crps = np.zeros(nday), np.zeros(nday), np.zeros(nday)
        for ens6, tru6 in series:
            e = aggregate_daily(ens6, dim=1); o = aggregate_daily(tru6, dim=0)
            for d in range(nday):
                crps[d] += float((kcrps(e[:, d], o[d]) * W).mean())
                q99p[d] += float(torch.quantile(e[:, d].reshape(-1).float(), 0.99))
                q99o[d] += float(torch.quantile(o[d].reshape(-1).float(), 0.99))
                for t in DEFAULT_THRESHOLDS_MMDAY:
                    p = exceedance_prob(e[:, d], t)
                    b_ = brier(p, o[d], t, weights=W)
                    br[str(t)][d] += b_["brier"]; ev[str(t)] += b_["n_events"]
        n = len(series)
        res[name] = {"crps": (crps / n).tolist(),
                     "q99_pred": (q99p / n).tolist(), "q99_obs": (q99o / n).tolist(),
                     "brier": {k: (v / n).tolist() for k, v in br.items()},
                     "events": ev}
        print(f"\n=== {name} ===")
        print("%-5s %8s %9s %9s %9s %9s %9s" % ("day","CRPS","q99 pred","q99 obs","BS@1","BS@10","BS@20"))
        for d in [0, 2, 4, 6, 9]:
            print("%-5d %8.3f %9.2f %9.2f %9.4f %9.4f %9.4f"
                  % (d+1, res[name]["crps"][d], res[name]["q99_pred"][d],
                     res[name]["q99_obs"][d], res[name]["brier"]["1.0"][d],
                     res[name]["brier"]["10.0"][d], res[name]["brier"]["20.0"][d]))
    print("\nevent counts (all days pooled):",
          {k: v for k, v in list(res.values())[0]["events"].items()})
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
