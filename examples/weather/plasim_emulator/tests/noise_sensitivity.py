"""Does the TRAINED model actually use its noise channels?

The smoke test checks this on random weights, where noise propagates trivially.
The question that matters is whether it survives training: an L2 state loss is
minimised by the conditional mean, which gives the network every incentive to
learn to ignore the noise and collapse the ensemble.
"""
import os, sys, torch, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import PlasimEmulator, rollout
from dataset import PlasimSequenceDataset

cfg = yaml.safe_load(open(sys.argv[1]))
ck_path = sys.argv[2]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = PlasimEmulator(n_noise=cfg["model"]["n_noise"],
                   probabilistic=cfg["model"]["probabilistic"],
                   **cfg["model"].get("sfno", {})).to(dev).eval()
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
m.load_state_dict(ck["model"])
print(f"loaded {os.path.basename(ck_path)} stage={ck.get('stage')}")

ds = PlasimSequenceDataset(cfg["data"]["root"], "test", n_steps=8)
b = ds[0]
s0 = b["state_in"][None].to(dev)
fo = b["forcing"][None].to(dev)

with torch.no_grad():
    # 1. weight magnitude on the noise input channels vs the state channels
    w = m.net.encoder.fwd[0].weight if hasattr(m.net, "encoder") else None
    if w is not None and w.dim() >= 2:
        ns, nf = m.n_state, m.n_forcing
        wf = w.flatten(2).abs().mean(dim=(0, 2)) if w.dim() > 2 else w.abs().mean(0)
        print(f"encoder |w| state={wf[:ns].mean():.5f} "
              f"forcing={wf[ns:ns+nf].mean():.5f} noise={wf[ns+nf:].mean():.5f}")

    # 2. trajectory divergence from identical IC, different noise draws
    outs = []
    for seed in range(4):
        g = torch.Generator(device=dev); g.manual_seed(seed)
        st, _ = rollout(m, s0, fo, generator=g)
        outs.append(st)
    E = torch.stack(outs, 0)                      # [4,1,T,52,H,W]
    spread = E.std(0).mean(dim=(0, 2, 3, 4))      # per lead time
    sig = s0.std()
    print(f"state std of data ~ {float(sig):.4f}")
    for t in [0, 3, 7]:
        print(f"  lead +{6*(t+1):3d}h  ensemble std = {float(spread[t]):.3e} "
              f"({100*float(spread[t])/float(sig):.4f}% of signal)")

    # 3. control: does a large noise perturbation move the output at all?
    z = torch.zeros(1, m.n_noise, 64, 128, device=dev)
    a, _ = m(s0, fo[:, 0], z)
    bb, _ = m(s0, fo[:, 0], z + 10.0)
    print(f"noise 0 vs +10 sigma: max|delta| = {float((a-bb).abs().max()):.3e}")
