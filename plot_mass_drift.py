#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot dry/water mass drift (%/day) by lead time, mean +/- std across inits per model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_STYLES = {
    "pangu_s2s": {"color": "#D55E00", "marker": "s", "label": "Pangu-S2S"},
    "sfno_s2s": {"color": "#0072B2", "marker": "o", "label": "SFNO-S2S"},
}

METRICS = {
    "dry_mass_drift_pct_per_day": "Dry Air Mass Drift [%/day]",
    "water_mass_drift_pct_per_day": "Water Mass Drift [%/day]",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to mass_drift_week2.csv")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for col, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for model, style in MODEL_STYLES.items():
            mdf = df[df["model"] == model]
            if mdf.empty:
                continue
            agg = mdf.groupby("lead_days")[col].agg(["mean", "std"]).reset_index()
            x = agg["lead_days"].values
            y = agg["mean"].values
            y_std = agg["std"].fillna(0.0).values

            ax.plot(x, y, label=style["label"], color=style["color"],
                     marker=style["marker"], markersize=6)
            ax.fill_between(x, y - y_std, y + y_std, color=style["color"],
                             alpha=0.18, linewidth=0)

        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(title)
        ax.legend()
        fig.tight_layout()
        outpath = outdir / f"{col}.png"
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
