#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot + sanity-check the AMIP SI diffusion convergence smoke's per-batch loss TSV.

Reads the TSV emitted by ``train_diffusion.py``'s ``bench.per_batch_tsv``
wiring (columns: ``epoch\\tbatch_idx\\twall_s\\tloss``), plots loss vs.
global step, and prints a first-epoch-vs-last-epoch mean-loss
comparison so a Delta job log records the sanity check without
requiring the PNG to be viewed.

Usage::

    python plot_convergence.py --tsv <path>.tsv --output <path>.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [
            {
                "epoch": int(row["epoch"]),
                "batch_idx": int(row["batch_idx"]),
                "wall_s": float(row["wall_s"]),
                "loss": float(row["loss"]),
            }
            for row in reader
        ]


def _sanity_check(rows: list[dict]) -> str:
    epochs = sorted({r["epoch"] for r in rows})
    if len(epochs) < 2:
        return "only one epoch present — can't compare first vs. last."
    first_losses = [r["loss"] for r in rows if r["epoch"] == epochs[0]]
    last_losses = [r["loss"] for r in rows if r["epoch"] == epochs[-1]]
    first_mean = sum(first_losses) / len(first_losses)
    last_mean = sum(last_losses) / len(last_losses)
    pct_change = 100.0 * (last_mean - first_mean) / abs(first_mean) if first_mean else float("nan")
    verdict = "DECREASING (converging)" if last_mean < first_mean else "NOT decreasing"
    return (
        f"epoch {epochs[0]} mean loss = {first_mean:.4e}, "
        f"epoch {epochs[-1]} mean loss = {last_mean:.4e} "
        f"({pct_change:+.1f}%) — {verdict}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = _read_tsv(args.tsv)
    if not rows:
        raise ValueError(f"{args.tsv} has no data rows")

    print(_sanity_check(rows))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(len(rows)))
    losses = [r["loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("global batch step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"AMIP SI diffusion convergence — {args.tsv.name}")
    ax.grid(True, which="both", alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
