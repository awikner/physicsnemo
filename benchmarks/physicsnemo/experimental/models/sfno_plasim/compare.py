#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare ai-rossby vs. PanguWeather v2.0 SFNO_S2S benchmark outputs.

The two sbatch scripts emit:

* **ai-rossby**: a TSV at ``ai_rossby_<jobid>.tsv`` with columns
  ``epoch, batch_idx, wall_s, loss, surface, upper_air, diagnostic, vae_kl, lr``.
* **PanguWeather**: tqdm-formatted stdout in
  ``bench-sfno-panguweather-<jobid>.out`` containing
  ``Loss: 1.2345`` lines per minibatch (no wall-clock).

This script parses both, aligns by batch index, and writes a markdown
results section to stdout for embedding in RESULTS.md.

Usage::

    python benchmarks/physicsnemo/experimental/models/sfno_plasim/compare.py \\
        --ai-rossby-tsv /work/hdd/bdiu/awikner/sfno_bench/ai_rossby_<jobid>.tsv \\
        --panguweather-log hpc/scripts/logs/bench-sfno-panguweather-<jobid>.out
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


_PW_LOSS_RE = re.compile(r"Loss:\s*([0-9.]+(?:[eE][+-]?\d+)?)")
_PW_WALL_RE = re.compile(r"epoch[_\s]+(\d+).*?(\d+(?:\.\d+)?)\s*s")


def parse_ai_rossby_tsv(path: Path):
    """Yield ``(batch_idx, wall_s, loss)`` triples from the ai-rossby TSV."""
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            yield int(row["batch_idx"]), float(row["wall_s"]), float(row["loss"])


def parse_panguweather_log(path: Path):
    """Yield ``(batch_idx, loss)`` from the PanguWeather stdout log.

    PanguWeather logs per-batch loss via tqdm's set_description ``Loss: X.YY``.
    We can't get exact wall_s from the log (tqdm doesn't print it per-batch),
    so the comparison's wall-clock cell is at the epoch-level only.
    """
    batch_idx = 0
    for line in open(path):
        m = _PW_LOSS_RE.search(line)
        if m:
            yield batch_idx, float(m.group(1))
            batch_idx += 1


def summarize(name: str, losses: list[float], wall_s: list[float] | None) -> dict:
    if not losses:
        return {"name": name, "n_batches": 0, "final_loss": float("nan"),
                "median_loss": float("nan"), "wall_s": float("nan"),
                "samples_per_s": float("nan")}
    out = {
        "name": name,
        "n_batches": len(losses),
        "final_loss": losses[-1],
        "median_loss": statistics.median(losses),
    }
    if wall_s:
        out["wall_s"] = wall_s[-1]
        # global batch_size = 32 → samples = 32 × n_batches
        out["samples_per_s"] = (32 * len(losses)) / max(wall_s[-1], 1e-9)
    else:
        out["wall_s"] = float("nan")
        out["samples_per_s"] = float("nan")
    return out


def render_markdown(ai_rossby: dict, pw: dict) -> str:
    lines = [
        "## Headline numbers",
        "",
        "| Metric | ai-rossby (`SfnoPlasim`) | PanguWeather v2.0 (`SFNO_v2`) |",
        "|---|---|---|",
        f"| batches/epoch | {ai_rossby['n_batches']} | {pw['n_batches']} |",
        f"| final-batch loss | {ai_rossby['final_loss']:.4f} | {pw['final_loss']:.4f} |",
        f"| median-batch loss | {ai_rossby['median_loss']:.4f} | {pw['median_loss']:.4f} |",
        f"| wall (epoch, s) | {ai_rossby['wall_s']:.1f} | n/a |",
        f"| samples/s | {ai_rossby['samples_per_s']:.1f} | n/a |",
        "",
        "PanguWeather's stdout doesn't carry per-batch wall-clock; the",
        "samples/s comparison is left blank in that column. To match, the",
        "epoch-total wall is read from the SLURM `seff` accounting at the end",
        "of each run.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ai-rossby-tsv", type=Path, required=True)
    p.add_argument("--panguweather-log", type=Path, required=True)
    args = p.parse_args(argv)

    ai_rossby_rows = list(parse_ai_rossby_tsv(args.ai_rossby_tsv))
    pw_rows = list(parse_panguweather_log(args.panguweather_log))

    ai_summary = summarize(
        "ai-rossby",
        losses=[r[2] for r in ai_rossby_rows],
        wall_s=[r[1] for r in ai_rossby_rows],
    )
    pw_summary = summarize(
        "panguweather",
        losses=[r[1] for r in pw_rows],
        wall_s=None,
    )
    print(render_markdown(ai_summary, pw_summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
