# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compute normalization statistics over the packed PlaSim training split.

Statistics are computed over the TRAINING years only - never the validation or
test years - so that no information from held-out years leaks into the inputs.

Writes into ``<out>/stats/``:
    global_means.npy          [1, 52, 1, 1]   state
    global_stds.npy           [1, 52, 1, 1]
    forcing_global_means.npy  [1,  8, 1, 1]
    forcing_global_stds.npy   [1,  8, 1, 1]
    diagnostic_stats.json     per-diagnostic Gamma scale, zero fraction, quantiles

Diagnostics get no mean/std: the hurdle-Gamma head models them in physical units.
What it needs instead is the scale of the strictly positive part, plus the zero
fraction as a sanity check on the Bernoulli branch.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from channels import DIAGNOSTIC_CHANNELS, FORCING_CHANNELS, STATE_CHANNELS  # noqa: E402


def accumulate(paths, nch, stride):
    """Streaming per-channel mean/std over [T, C, H, W] files."""
    count = 0
    s1 = np.zeros(nch, dtype=np.float64)
    s2 = np.zeros(nch, dtype=np.float64)
    for p in paths:
        with h5py.File(p, "r") as f:
            d = f["fields"]
            for t0 in range(0, d.shape[0], stride * 64):
                t1 = min(t0 + stride * 64, d.shape[0])
                blk = np.asarray(d[t0:t1:stride], dtype=np.float64)
                if blk.size == 0:
                    continue
                s1 += blk.sum(axis=(0, 2, 3))
                s2 += (blk**2).sum(axis=(0, 2, 3))
                count += blk.shape[0] * blk.shape[2] * blk.shape[3]
        print(f"  {os.path.basename(p)} done", flush=True)
    mean = s1 / count
    var = np.maximum(s2 / count - mean**2, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), count


def diagnostic_stats(paths, stride):
    """Positive-part scale, zero fraction and quantiles per diagnostic."""
    vals = [[] for _ in DIAGNOSTIC_CHANNELS]
    for p in paths:
        with h5py.File(p, "r") as f:
            d = f["fields"]
            blk = np.asarray(d[::stride * 8], dtype=np.float32)
            for c in range(len(DIAGNOSTIC_CHANNELS)):
                vals[c].append(blk[:, c].reshape(-1))
    out = {}
    for c, name in enumerate(DIAGNOSTIC_CHANNELS):
        v = np.concatenate(vals[c])
        pos = v[v > 0]
        out[name] = {
            "scale": float(pos.mean()) if pos.size else 1.0,
            "zero_fraction": float((v == 0).mean()),
            "mean": float(v.mean()),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
            "n_sampled": int(v.size),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="packed dataset root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=4,
                    help="timestep stride; 4 = one sample/day, plenty for stats")
    ap.add_argument("--expect-years", type=int, default=100,
                    help="fail if fewer years are packed than this (0 disables)")
    args = ap.parse_args()

    stats_dir = os.path.join(args.out, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    def paths(kind):
        p = sorted(glob.glob(os.path.join(args.out, kind, args.split, "*.h5")))
        if not p:
            raise SystemExit(f"no packed files under {kind}/{args.split}")
        # Computing statistics from a partially-packed tree is a silent error:
        # the numbers look plausible but are drawn from the wrong sample.
        if args.expect_years and len(p) < args.expect_years:
            raise SystemExit(
                f"{kind}/{args.split}: found {len(p)} years, expected "
                f"{args.expect_years}. Packing is probably still running. "
                f"Re-run once it finishes, or pass --expect-years {len(p)} "
                "to accept this deliberately."
            )
        return p

    print("state stats...")
    m, s, n = accumulate(paths("state"), len(STATE_CHANNELS), args.stride)
    if not np.all(s > 0):
        bad = [STATE_CHANNELS[i] for i in np.where(s <= 0)[0]]
        raise SystemExit(f"zero-variance state channels: {bad}")
    np.save(os.path.join(stats_dir, "global_means.npy"), m[None, :, None, None])
    np.save(os.path.join(stats_dir, "global_stds.npy"), s[None, :, None, None])
    print(f"  {n} samples/channel")

    print("forcing stats...")
    m, s, n = accumulate(paths("forcing"), len(FORCING_CHANNELS), args.stride)
    if not np.all(s > 0):
        bad = [FORCING_CHANNELS[i] for i in np.where(s <= 0)[0]]
        raise SystemExit(
            f"zero-variance forcing channels: {bad} - drop them from "
            "channels.FORCING_CHANNELS rather than dividing by zero"
        )
    np.save(os.path.join(stats_dir, "forcing_global_means.npy"),
            m[None, :, None, None])
    np.save(os.path.join(stats_dir, "forcing_global_stds.npy"),
            s[None, :, None, None])

    print("diagnostic stats...")
    ds = diagnostic_stats(paths("diagnostic"), args.stride)
    with open(os.path.join(stats_dir, "diagnostic_stats.json"), "w") as f:
        json.dump(ds, f, indent=1)
    for k, v in ds.items():
        print(f"  {k}: scale={v['scale']:.4g} zero_frac={v['zero_fraction']:.3f} "
              f"p99={v['p99']:.4g} max={v['max']:.4g}")

    print(f"wrote stats to {stats_dir}")


if __name__ == "__main__":
    main()
