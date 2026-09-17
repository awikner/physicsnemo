# SPDX-License-Identifier: Apache-2.0
"""Corrected spread/structure diagnostics.

`diagnose_precip.py` compared the spectrum of the ENSEMBLE MEAN against truth.
That is apples-to-oranges: the ensemble mean of a calibrated forecast is the
conditional mean, which is legitimately smoother than any single realisation and
gets smoother as predictability drops. A falling power ratio is then EXPECTED,
not a defect - so the earlier "round 2 over-smooths" reading is confounded.

Measured here instead:
  member_spec  spectrum of a SINGLE ensemble member vs truth   <- the right test
  mean_spec    spectrum of the ensemble mean vs truth          <- old metric, kept
  q99_member   p99 of one member (realism of a single field)
  q99_pooled   p99 of all members pooled (the predictive marginal)
  spread_skill ensemble spread vs error of the ensemble mean; ~1 = calibrated
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import DIAGNOSTIC_CHANNELS, PRECIP_CHANNEL   # noqa: E402
from dataset import PlasimSequenceDataset                   # noqa: E402
from distributions import HurdleGamma                       # noqa: E402
from metrics_daily import aggregate_daily                   # noqa: E402
from model import PlasimEmulator                            # noqa: E402

MM = 1000.0 * 4.0
M = 16


def zspec(x):
    return (torch.fft.rfft(x, dim=-1).abs() ** 2).mean(dim=-2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--samples", type=int, default=6)
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
    sm = torch.tensor(ds.state_mean).to(dev); ss = torch.tensor(ds.state_std).to(dev)
    stride = max(1, len(ds) // a.samples)
    idxs = list(range(0, min(len(ds), a.samples * stride), stride))
    nday = T // 4
    res = {}

    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        m = PlasimEmulator(n_noise=cfg["model"]["n_noise"],
                           probabilistic=cfg["model"]["probabilistic"],
                           **cfg["model"].get("sfno", {})).to(dev).eval()
        m.load_state_dict(torch.load(path, map_location="cpu",
                                     weights_only=False)["model"])
        acc = {k: np.zeros(nday) for k in
               ["member_spec", "mean_spec", "q99_member", "q99_pooled", "q99_obs",
                "spread_tas", "rmse_tas"]}
        for si in idxs:
            b = ds[si]
            s = b["state_in"][None].to(dev).repeat(M, 1, 1, 1)
            fo = b["forcing"][None].to(dev)
            dg = b["diag_tgt"][None].to(dev); stg = b["state_tgt"][None].to(dev)
            pr, tas = [], []
            for t in range(T):
                s, d = m(s, fo[:, t].repeat(M, 1, 1, 1))
                dist = HurdleGamma(d[:, 3 * pi:3 * (pi + 1)], scale=scale)
                pr.append(dist.sample() * MM)                 # [M,H,W] - one draw each
                tas.append(s[:, 1] * ss[0, 1] + sm[0, 1])
            PR = torch.stack(pr, 1); TAS = torch.stack(tas, 1)   # [M,T,H,W]
            OB = dg[0, :, pi] * MM
            OT = stg[0, :, 1] * ss[0, 1] + sm[0, 1]
            prd = aggregate_daily(PR, dim=1); obd = aggregate_daily(OB, dim=0)
            for d_ in range(nday):
                hi = slice(prd.shape[-1] // 4, None)
                sm_ = zspec(prd[0, d_]); mn_ = zspec(prd[:, d_].mean(0))
                ot_ = zspec(obd[d_])
                acc["member_spec"][d_] += float(sm_[hi].sum() / ot_[hi].sum())
                acc["mean_spec"][d_] += float(mn_[hi].sum() / ot_[hi].sum())
                acc["q99_member"][d_] += float(torch.quantile(prd[0, d_].reshape(-1).float(), .99))
                acc["q99_pooled"][d_] += float(torch.quantile(prd[:, d_].reshape(-1).float(), .99))
                acc["q99_obs"][d_] += float(torch.quantile(obd[d_].reshape(-1).float(), .99))
                sl = slice(d_ * 4, (d_ + 1) * 4)
                e = TAS[:, sl].mean(1); o = OT[sl].mean(0)
                acc["spread_tas"][d_] += float(torch.sqrt((e.var(0, unbiased=True)
                                                           * (M + 1) / M * W).mean()))
                acc["rmse_tas"][d_] += float(torch.sqrt(((e.mean(0) - o) ** 2 * W).mean()))
        n = len(idxs)
        r = {k: (v / n).tolist() for k, v in acc.items()}
        r["spread_skill"] = [s_ / max(e_, 1e-9) for s_, e_ in
                             zip(r["spread_tas"], r["rmse_tas"])]
        res[name] = r
        print(f"\n=== {name} ===")
        print("%-5s %12s %10s %10s %10s %10s %12s" % ("day", "member spec", "mean spec",
              "q99 memb", "q99 pool", "q99 obs", "spread/skill"))
        for d_ in [0, 2, 4, 6, 9]:
            print("%-5d %12.3f %10.3f %10.2f %10.2f %10.2f %12.3f"
                  % (d_ + 1, r["member_spec"][d_], r["mean_spec"][d_],
                     r["q99_member"][d_], r["q99_pooled"][d_], r["q99_obs"][d_],
                     r["spread_skill"][d_]))
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
