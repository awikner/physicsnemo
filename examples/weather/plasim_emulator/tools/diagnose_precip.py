# SPDX-License-Identifier: Apache-2.0
"""Track A diagnostics: WHY does precipitation skill collapse by day 5-7?

A1 attribution - precip diagnosed from the TRUE state vs from the PREDICTED
   state, at each lead. The model emits diagnostics conditioned on the current
   state, so feeding ground truth isolates head error from state error. If
   head-on-truth is also poor, the head is the bottleneck and extra prognostic
   channels would buy little.
A2 hurdle split - the Bernoulli gate and the Gamma amount are only ever scored
   jointly. POD/FAR/CSI on the wet/dry decision, and CRPS conditional on wet for
   the amount, say which half is failing.
A3 spectra - zonal power spectrum of predicted vs true precip against lead, to
   test whether the field is being smoothed away.

Everything is scored on DAILY MEANS via metrics_daily.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import DIAGNOSTIC_CHANNELS, PRECIP_CHANNEL     # noqa: E402
from dataset import PlasimSequenceDataset                     # noqa: E402
from distributions import HurdleGamma                         # noqa: E402
from metrics_daily import aggregate_daily, kcrps              # noqa: E402
from model import PlasimEmulator                              # noqa: E402

MM = 1000.0 * 4.0
M_ENS = 16


def zonal_spectrum(x):
    """Mean power spectrum along longitude, averaged over latitude."""
    f = torch.fft.rfft(x, dim=-1)
    return (f.abs() ** 2).mean(dim=-2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
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

    m = PlasimEmulator(n_noise=cfg["model"]["n_noise"],
                       probabilistic=cfg["model"]["probabilistic"],
                       **cfg["model"].get("sfno", {})).to(dev).eval()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    print(f"loaded {os.path.basename(a.checkpoint)} stage={ck.get('stage')}")

    ds = PlasimSequenceDataset(root, a.split, n_steps=T)
    st = json.load(open(os.path.join(root, "stats", "diagnostic_stats.json")))
    pi = DIAGNOSTIC_CHANNELS.index(PRECIP_CHANNEL)
    scale = st[PRECIP_CHANNEL]["scale"]
    stride = max(1, len(ds) // a.samples)
    idxs = list(range(0, min(len(ds), a.samples * stride), stride))

    nday = T // 4
    acc = {k: np.zeros(nday) for k in
           ["crps_pred", "crps_true", "rmse_pred", "rmse_true",
            "pod", "far", "csi", "crps_wet", "base_wet"]}
    spec_p, spec_t, nsp = None, None, 0

    for si in idxs:
        b = ds[si]
        s0 = b["state_in"][None].to(dev)
        fo = b["forcing"][None].to(dev)
        stt = b["state_tgt"][None].to(dev)
        dg = b["diag_tgt"][None].to(dev)

        ens_pred, ens_true, p_pred = [], [], []
        s = s0
        for t in range(T):
            # predicted-state path (normal autoregressive rollout)
            s, d = m(s, fo[:, t])
            dp = HurdleGamma(d[:, 3 * pi:3 * (pi + 1)], scale=scale)
            ens_pred.append(torch.stack([dp.sample()[0] for _ in range(M_ENS)], 0) * MM)
            p_pred.append(dp.p[0])
            # true-state path: same head, ground-truth state in
            s_true = s0 if t == 0 else stt[:, t - 1]
            _, dt_ = m(s_true, fo[:, t])
            dtd = HurdleGamma(dt_[:, 3 * pi:3 * (pi + 1)], scale=scale)
            ens_true.append(torch.stack([dtd.sample()[0] for _ in range(M_ENS)], 0) * MM)

        EP = torch.stack(ens_pred, 1)            # [M,T,H,W]
        ET = torch.stack(ens_true, 1)
        OB = dg[0, :, pi] * MM                   # [T,H,W]
        PP = torch.stack(p_pred, 0)              # [T,H,W]

        ep = aggregate_daily(EP, dim=1); et = aggregate_daily(ET, dim=1)
        ob = aggregate_daily(OB, dim=0)
        for d in range(nday):
            acc["crps_pred"][d] += float((kcrps(ep[:, d], ob[d]) * W).mean())
            acc["crps_true"][d] += float((kcrps(et[:, d], ob[d]) * W).mean())
            acc["rmse_pred"][d] += float(torch.sqrt((((ep[:, d].mean(0) - ob[d]) ** 2) * W).mean()))
            acc["rmse_true"][d] += float(torch.sqrt((((et[:, d].mean(0) - ob[d]) ** 2) * W).mean()))

        # A2 on the native 6-hourly field: wet/dry decision from p, amount given wet
        for d in range(nday):
            sl = slice(d * 4, (d + 1) * 4)
            pw = PP[sl].reshape(-1)
            ow = (OB[sl].reshape(-1) > 0.1)          # >0.1 mm/day counts as wet
            fw = pw > 0.5
            hit = float((fw & ow).sum()); miss = float((~fw & ow).sum())
            fa = float((fw & ~ow).sum())
            acc["pod"][d] += hit / max(hit + miss, 1)
            acc["far"][d] += fa / max(hit + fa, 1)
            acc["csi"][d] += hit / max(hit + miss + fa, 1)
            acc["base_wet"][d] += float(ow.float().mean())
            wm = ob[d] > 0.1
            if wm.any():
                acc["crps_wet"][d] += float(kcrps(ep[:, d], ob[d])[wm].mean())

        s_ = zonal_spectrum(ep.mean(0)); t_ = zonal_spectrum(ob)
        spec_p = s_ if spec_p is None else spec_p + s_
        spec_t = t_ if spec_t is None else spec_t + t_
        nsp += 1

    n = len(idxs)
    res = {k: (v / n).tolist() for k, v in acc.items()}
    res["days"] = list(range(1, nday + 1))
    res["spectrum_pred"] = (spec_p / nsp).cpu().numpy().tolist()
    res["spectrum_true"] = (spec_t / nsp).cpu().numpy().tolist()
    res["n_samples"] = n
    json.dump(res, open(a.out, "w"), indent=1)

    print("\nA1 attribution (daily-mean precip CRPS, mm/day):")
    print("%-5s %11s %11s %9s" % ("day", "pred-state", "TRUE-state", "gap"))
    for d in range(nday):
        print("%-5d %11.3f %11.3f %9.3f"
              % (d + 1, res["crps_pred"][d], res["crps_true"][d],
                 res["crps_pred"][d] - res["crps_true"][d]))
    print("\nA2 hurdle split (wet = >0.1 mm/day):")
    print("%-5s %7s %7s %7s %9s %9s" % ("day", "POD", "FAR", "CSI", "base", "CRPS|wet"))
    for d in range(nday):
        print("%-5d %7.3f %7.3f %7.3f %9.3f %9.3f"
              % (d + 1, res["pod"][d], res["far"][d], res["csi"][d],
                 res["base_wet"][d], res["crps_wet"][d]))
    sp = np.array(res["spectrum_pred"]); stt_ = np.array(res["spectrum_true"])
    hi = slice(len(sp[0]) // 2, None)
    print("\nA3 high-wavenumber power ratio pred/true (<1 = over-smoothed):")
    for d in [0, nday // 2, nday - 1]:
        print("  day %2d  %.3f" % (d + 1, float(sp[d][hi].sum() / stt_[d][hi].sum())))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
