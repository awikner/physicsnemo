#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot absolute mean dry/water mass vs lead day, to see the actual trajectory shape."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_STYLES = {
    "era5": {"color": "#1baf7a", "marker": "^", "label": "ERA5", "linestyle": "-"},
    "pangu_s2s": {"color": "#D55E00", "marker": "s", "label": "Pangu-S2S", "linestyle": "-"},
    "pangu_s2s_msl_derived": {"color": "#D55E00", "marker": "s", "label": "Pangu-S2S (MSL-derived PS)", "linestyle": "--"},
    "sfno_s2s": {"color": "#0072B2", "marker": "o", "label": "SFNO-S2S", "linestyle": "-"},
}

METRICS = {
    "dry_mass_Eg": ("Dry Air Mass (Eg)", "dry_mass_Eg_mean", "dry_mass_Eg_std"),
    "water_mass_kg": ("Water Mass (kg)", "water_mass_kg_mean", "water_mass_kg_std"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to mass_trajectory.csv")
    parser.add_argument("--outdir", type=str, default="plots")
    parser.add_argument(
        "--models", type=str, default="pangu_s2s,sfno_s2s,era5",
        help="Comma-separated models to plot, from: "
             f"{','.join(MODEL_STYLES.keys())} (default: pangu_s2s,sfno_s2s,era5)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in models if m not in MODEL_STYLES]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {list(MODEL_STYLES.keys())}")

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for key, (title, mean_col, std_col) in METRICS.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for model in models:
            style = MODEL_STYLES[model]
            mdf = df[df["model"] == model].sort_values("lead_day")
            if mdf.empty:
                continue
            x = mdf["lead_day"].values
            y = mdf[mean_col].values
            y_std = mdf[std_col].values

            ax.plot(x, y, label=style["label"], color=style["color"],
                     marker=style["marker"], markersize=6,
                     linestyle=style.get("linestyle", "-"))
            ax.fill_between(x, y - y_std, y + y_std, color=style["color"],
                             alpha=0.18, linewidth=0)

        ax.set_title(title)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(title)
        ax.legend()
        fig.tight_layout()
        outpath = outdir / f"trajectory_{key}.png"
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
