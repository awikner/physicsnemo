#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare ai-rossby vs. PanguWeather v2.0 SFNO_PLASIM_5412 benchmark outputs.

The two sbatch scripts emit:

* **ai-rossby**: a TSV at ``ai_rossby_<jobid>.tsv`` with columns
  ``epoch, batch_idx, wall_s, loss, surface, upper_air, diagnostic, vae_kl, lr``.
  The ``loss`` column is the SUM of surface + upper_air + diagnostic MSEs
  (weights default to 1.0).
* **PanguWeather**: tqdm-formatted stdout in
  ``bench-sfno-panguweather-<jobid>.out`` with ``Loss: 0.XXXX:   N/45``
  per minibatch. PanguWeather's raw_l2 reduction is a channel-count
  weighted average of the same per-group MSEs:
  ``(loss_pl * C_u * L + loss_sfc * C_s + loss_diag * C_d) / (C_u * L + C_s + C_d)``
  For the 5412 config that's ``(loss_pl*50 + loss_sfc*2 + loss_diag*1)/53``.

To make the comparison apples-to-apples, this script computes the
PanguWeather-style channel-weighted average from ai-rossby's per-group
losses, so both sides report the SAME aggregation.

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


# ``Loss: 0.9876:   3/45`` — captures the loss value AND the batch index.
# We need the batch index because tqdm rewrites the same line many times per
# batch, so the regex hits multiple times — keying on batch_idx lets us
# deduplicate.
_PW_LOSS_RE = re.compile(
    r"Loss:\s*([0-9.]+(?:[eE][+-]?\d+)?):\s*\d+%[^|]*\|[^|]*\|\s*(\d+)/(\d+)"
)
# ``Time taken for epoch 1 is 1531.6020696163177 sec``
_PW_WALL_RE = re.compile(r"Time taken for epoch \d+ is\s+([0-9.]+)\s*sec")


def parse_ai_rossby_tsv(path: Path):
    """Yield ``(batch_idx, wall_s, loss, surface, upper_air, diagnostic)``."""
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            yield (
                int(row["batch_idx"]),
                float(row["wall_s"]),
                float(row["loss"]),
                float(row["surface"]),
                float(row["upper_air"]),
                float(row["diagnostic"]),
            )


def parse_panguweather_log(path: Path) -> tuple[list[tuple[int, float]], float | None]:
    """Return (``[(batch_idx, loss), ...]``, ``epoch_wall_s``).

    PanguWeather's tqdm rewrites the description many times per batch, so
    multiple ``Loss: X: N/45`` substrings end up on a single physical line
    after carriage-return flushing. We dedupe by ``batch_idx``, keeping the
    LAST loss we see for each batch (= the loss recorded after that batch's
    backward pass).
    """
    per_batch: dict[int, float] = {}
    wall_s: float | None = None
    with open(path) as fh:
        for line in fh:
            for m in _PW_LOSS_RE.finditer(line):
                loss, batch_idx, _total = float(m.group(1)), int(m.group(2)), int(m.group(3))
                # tqdm's first description set fires BEFORE batch_idx increments —
                # i.e. at batch_idx=0, the loss reported is from the prior
                # iteration's setup (typically the first batch).  Storing the
                # last value per batch_idx gives the post-step loss.
                per_batch[batch_idx] = loss
            wm = _PW_WALL_RE.search(line)
            if wm:
                wall_s = float(wm.group(1))
    return sorted(per_batch.items()), wall_s


def channel_weighted_total(
    surface: float, upper_air: float, diagnostic: float,
    *, n_surface: int, n_upper: int, n_levels: int, n_diag: int,
) -> float:
    """PanguWeather raw_l2 total reduction across the three groups."""
    num = (upper_air * n_upper * n_levels
           + surface * n_surface
           + diagnostic * n_diag)
    den = n_upper * n_levels + n_surface + n_diag
    return num / den


def summarize(name: str, losses: list[float], wall_s: float | None,
              global_batch: int) -> dict:
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
    if wall_s is not None:
        out["wall_s"] = wall_s
        out["samples_per_s"] = (global_batch * len(losses)) / max(wall_s, 1e-9)
    else:
        out["wall_s"] = float("nan")
        out["samples_per_s"] = float("nan")
    return out


def render_markdown(ai_rossby: dict, pw: dict, divergence: dict) -> str:
    lines = [
        "## Headline numbers",
        "",
        "Both columns report the **channel-count weighted average** of the",
        "surface + upper-air + diagnostic MSEs (PanguWeather's raw_l2",
        "aggregation: `(loss_pl × 50 + loss_sfc × 2 + loss_diag × 1) / 53`).",
        "ai-rossby's per-group MSEs are taken from the TSV `surface`,",
        "`upper_air`, `diagnostic` columns and re-aggregated to match.",
        "",
        "| Metric | ai-rossby (`SfnoPlasim`) | PanguWeather v2.0 (`SFNO_v2`) |",
        "|---|---|---|",
        f"| batches/epoch | {ai_rossby['n_batches']} | {pw['n_batches']} |",
        f"| final-batch loss | {ai_rossby['final_loss']:.4f} | {pw['final_loss']:.4f} |",
        f"| median-batch loss | {ai_rossby['median_loss']:.4f} | {pw['median_loss']:.4f} |",
        f"| wall (epoch, s) | {ai_rossby['wall_s']:.1f} | {pw['wall_s']:.1f} |",
        f"| samples/s | {ai_rossby['samples_per_s']:.1f} | {pw['samples_per_s']:.1f} |",
        "",
        "## Divergence vs. decision rule",
        "",
        f"* Max relative |Δ loss| at any batch (over the overlapping prefix): "
        f"**{divergence['max_rel_delta']:.1%}**",
        f"* Loss-curve correlation (Pearson): **{divergence['correlation']:.4f}**",
        f"* Relative |Δ wall|: **{divergence['rel_wall_delta']:.1%}**",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ai-rossby-tsv", type=Path, required=True)
    p.add_argument("--panguweather-log", type=Path, required=True)
    # SFNO_PLASIM_5412 channel layout (see model config):
    p.add_argument("--n-surface", type=int, default=2)
    p.add_argument("--n-upper", type=int, default=5)
    p.add_argument("--n-levels", type=int, default=10)
    p.add_argument("--n-diag", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=32)
    args = p.parse_args(argv)

    ai_rossby_rows = list(parse_ai_rossby_tsv(args.ai_rossby_tsv))
    pw_per_batch, pw_wall = parse_panguweather_log(args.panguweather_log)

    # Channel-weighted aggregation of ai-rossby's per-group MSEs to match
    # PanguWeather's reported single Loss.
    ai_rossby_aggregated = [
        channel_weighted_total(
            row[3], row[4], row[5],
            n_surface=args.n_surface, n_upper=args.n_upper,
            n_levels=args.n_levels, n_diag=args.n_diag,
        ) for row in ai_rossby_rows
    ]
    ai_summary = summarize(
        "ai-rossby",
        losses=ai_rossby_aggregated,
        wall_s=ai_rossby_rows[-1][1] if ai_rossby_rows else None,
        global_batch=args.global_batch,
    )
    pw_summary = summarize(
        "panguweather",
        losses=[v for _, v in pw_per_batch],
        wall_s=pw_wall,
        global_batch=args.global_batch,
    )

    # Per-batch divergence over the overlapping prefix.
    n = min(len(ai_rossby_aggregated), len(pw_per_batch))
    deltas = []
    if n > 0:
        for i in range(n):
            a, b = ai_rossby_aggregated[i], pw_per_batch[i][1]
            denom = max(abs(b), 1e-9)
            deltas.append(abs(a - b) / denom)
    max_rel_delta = max(deltas) if deltas else float("nan")

    # Pearson correlation
    correlation = float("nan")
    if n >= 2:
        a_pref = ai_rossby_aggregated[:n]
        b_pref = [v for _, v in pw_per_batch[:n]]
        am, bm = statistics.mean(a_pref), statistics.mean(b_pref)
        num = sum((a - am) * (b - bm) for a, b in zip(a_pref, b_pref))
        den_a = sum((a - am) ** 2 for a in a_pref) ** 0.5
        den_b = sum((b - bm) ** 2 for b in b_pref) ** 0.5
        denom = den_a * den_b
        if denom > 0:
            correlation = num / denom

    rel_wall_delta = float("nan")
    if not (ai_summary["wall_s"] != ai_summary["wall_s"]) and pw_wall:
        rel_wall_delta = abs(ai_summary["wall_s"] - pw_wall) / pw_wall

    divergence = {
        "max_rel_delta": max_rel_delta,
        "correlation": correlation,
        "rel_wall_delta": rel_wall_delta,
    }
    print(render_markdown(ai_summary, pw_summary, divergence))
    return 0


if __name__ == "__main__":
    sys.exit(main())
