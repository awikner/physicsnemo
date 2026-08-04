#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Week-2 ACC of WEEKLY-ACCUMULATED precip, by month, for every source.

This scores the project's *primary* target rather than the daily one the
training metrics use: for each initialization the daily fields at leads 8-14 are
summed into one week-2 total (mm/week) and compared against the same-week IMERG
total.

Anomalies reference a WEEKLY climatology built by summing the day-of-year daily
climatology over the same seven valid days, so the reference matches the
predictand exactly. Using a daily or monthly reference here would leave the
seasonal cycle in both anomalies and inflate the correlation.

Each init is attributed to the month of its week MIDPOINT (the valid day at
tau=11), so a week straddling a month boundary lands in the month it mostly
covers. ACC per month pools every init in that month across all validation
years, over the IMD-supervised region with cos-lat weights -- i.e. exactly the
region the gate is trained on.

An expert contributes to an init only if it is live at all seven leads;
otherwise its weekly total would be a partial sum. Counts are reported.

``+matched=true`` (recommended for a like-for-like comparison) restricts every
source to the weeks where ALL of them are complete. Without it each source is
scored on the weeks it happens to cover -- the index takes the UNION of
initializations, so pangu/sfno cover ~280 of 465 weeks and aifs ~265, while the
gate and equal-weight are defined for every week. Comparing across different
samples would flatter whichever model happens to be present on the easier ones.

    python tools/plot_week2_acc.py \\
        dataset=hindcast_midway2 \\
        +checkpoint=/scratch/midway3/awikner/mowe_runs/outputs/mowe_v6_var1/checkpoints_best \\
        +out=/scratch/midway3/awikner/mowe_runs/week2_acc.png
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import cftime
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

_RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

from datapipes.factory import build_dataset  # noqa: E402
from datapipes.sampler import MixturePairSampler  # noqa: E402
from losses import denormalize_precip, imd_valid_mask, region_weights  # noqa: E402
from mowe_precip import MoWEPrecipGate, mix  # noqa: E402
from seeps import SeepsClimatology, doy_from_hours_since_1900  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

logger = logging.getLogger("week2_acc")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Okabe-Ito, designed to stay distinguishable under deuteranopia/protanopia.
# The gate is blue and hatched so it reads even in greyscale, and the risky
# vermillion/green pairing is avoided entirely.
COLOURS = {
    "pangu_s2s": "#E69F00",       # orange
    "sfno_era5": "#56B4E9",       # sky blue
    "graphcast": "#CC79A7",       # reddish purple
    "aifs_single_v2": "#009E73",  # bluish green
    "equal_weight": "#999999",    # grey
    "gate": "#0072B2",            # blue
    "debiased_mean": "#999999",   # grey
}
LABELS = {
    "pangu_s2s": "Pangu-S2S", "sfno_era5": "SFNO-S2S",
    "graphcast": "GraphCast", "aifs_single_v2": "AIFS",
    "equal_weight": "Equal weight", "gate": "MoWE gate",
    "debiased_mean": "Mean of debiased",
}


def _plot(months, acc, sd, sources, n_init, *, out, show_sd, debias,
          matched, lo, hi, years):
    """Render the grouped bar chart. Pure presentation: takes plain numbers so
    it can be driven from a computed pass or replayed from a saved CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    xpos = np.arange(len(months))
    width = 0.8 / len(sources)
    for k, src in enumerate(sources):
        ax.bar(xpos + (k - (len(sources) - 1) / 2) * width, acc[src], width,
               yerr=sd[src] if show_sd else None,
               error_kw=dict(ecolor="#333333", elinewidth=0.9,
                             capsize=2, capthick=0.9),
               label=LABELS.get(src, src), color=COLOURS.get(src, "#999999"),
               edgecolor="black", linewidth=0.5,
               hatch="//" if src == "gate" else None, zorder=3)
    ax.set_xticks(xpos, [MONTHS[m - 1] for m in months])
    ax.set_ylabel("Anomaly correlation (ACC)"
                  + ("\nmean of per-week ACC, error bars 1 s.d."
                     if show_sd else "\nmean of per-week ACC"))
    ax.set_xlabel("Month (of the week-2 midpoint)")
    ax.set_title(
        "Week-2 accumulated precipitation ACC, IMD region\n"
        f"leads {lo}-{hi} summed to weekly totals; "
        f"{years[0]}-{years[1]} pooled per month"
        + ("; weeks where every source is available"
           if matched else "; each source on the weeks it covers")
        + ("\nexperts DEBIASED by the gate (single-expert input)" if debias else "")
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    if show_sd:
        top = max(a + d for src in sources for a, d in zip(acc[src], sd[src]))
        bot = min(0.0, min(a - d for src in sources
                           for a, d in zip(acc[src], sd[src])))
        ax.set_ylim(bot * 1.08, top * 1.05)
    else:
        ax.set_ylim(0.0, max(max(v) for v in acc.values()) * 1.06)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(ncol=6, frameon=False, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.11))
    ax.set_axisbelow(True)
    ax.annotate("weeks per month: " + "  ".join(
        f"{MONTHS[m-1]} {n_init[m]}" for m in months),
        xy=(0.5, -0.22), xycoords="axes fraction", ha="center", va="top",
        fontsize=8, color="#444444")
    fig.savefig(out, dpi=180, bbox_inches="tight")


def replot_from_csv(csv_path: str, out: str, *, show_sd: bool) -> None:
    """Restyle a previously written CSV -- no model, no data pass, no GPU."""
    inv = {v: k for k, v in LABELS.items()}

    def _key(name: str) -> str:
        # The CSV keeps a "(debiased)" suffix for readability; the legend does
        # not, so strip it before mapping a row back to its source key.
        return inv[name.replace(" (debiased)", "")]

    rows = [ln.rstrip("\n").split(",") for ln in open(csv_path) if ln.strip()]
    months = [MONTHS.index(h) + 1 for h in rows[0][1:]]
    acc, sd, sources, n_init = {}, {}, [], {}
    for r in rows[1:]:
        name, vals = r[0], [float(v) for v in r[1:]]
        if name == "n_weeks":
            n_init = {m: int(v) for m, v in zip(months, vals)}
        elif name.endswith(" sd"):
            sd[_key(name[:-3])] = vals
        elif name.endswith(" pooled"):
            continue
        else:
            key = _key(name)
            sources.append(key)
            acc[key] = vals
    debias = "debiased_mean" in sources
    _plot(months, acc, sd, sources, n_init, out=out, show_sd=show_sd,
          debias=debias, matched=True, lo=8, hi=14, years=(2020, 2024))


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for k, v in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
                 ("MASTER_ADDR", "localhost"), ("MASTER_PORT", str(29500 + os.getpid() % 1000))):
        os.environ.setdefault(k, v)
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dev = DistributedManager().device

    ds = build_dataset(cfg.dataset, "val")
    lo, hi = (int(v) for v in cfg.dataset.val.lead_days)
    n_lead = hi - lo + 1
    experts = list(ds.expert_names)
    # debias_experts: feed the model ONE expert at a time. The masked softmax
    # then puts weight 1.0 on it, so the output is exactly P_i + b_i -- the
    # DEBIASED expert. Comparing those against their own mean and the full gate
    # separates the two mechanisms the gate bundles: the learned per-expert bias
    # correction, and the state-dependent weighting on top of it.
    # Note b_i is not the same as in the full-ensemble pass: the mask planes are
    # part of the input, so the model sees a genuine single-expert request (which
    # it is trained for, via expert_dropout).
    debias = bool(cfg.get("debias_experts", False))
    show_sd = bool(cfg.get("error_bars", True))
    sources = ([*experts, "debiased_mean", "gate"] if debias
               else [*experts, "equal_weight", "gate"])
    mix_space = str(cfg.model.get("mix_space", "physical"))

    box = list(cfg.region.lat) + list(cfg.region.lon)
    imd_cfg = cfg.dataset.get("imd", None)
    imd = imd_valid_mask(
        str(imd_cfg.store), ds.lat, ds.lon,
        min_finite_frac=float(imd_cfg.get("min_finite_frac", 0.99)),
    )
    w2d = region_weights(ds.lat, ds.lon, tuple(box), extra_mask=imd)
    sel = (w2d > 0).numpy()
    wts = w2d.numpy()[sel]                     # (n_cells,) cos-lat weights
    logger.info("region: %d gridpoints, leads %d-%d, experts %s",
                sel.sum(), lo, hi, experts)

    clim = SeepsClimatology(to_absolute_path(str(cfg.validation.seeps_climatology)))
    if clim.clim_mean_daily is None:
        raise ValueError("need clim_mean_daily; regenerate the climatology store")
    clim_daily = clim.clim_mean_daily.numpy()[:, sel]      # (366, n_cells)

    model = MoWEPrecipGate(
        input_size=(ds.lat.size, ds.lon.size),
        in_channels=ds.layout.num_channels,
        n_experts=len(experts),
        **OmegaConf.to_container(cfg.model.params, resolve=True),
    ).to(dev)
    logger.info("loaded %s (epoch %s)", cfg.checkpoint,
                load_checkpoint(str(cfg.checkpoint), models=model, device=dev))
    model.eval()

    # Noise-conditioned gate: score the ACC of the ENSEMBLE MEAN (the correct
    # deterministic summary of a probabilistic forecast). Seeded per batch so
    # the figure is reproducible.
    noise_dim = getattr(model, "noise_dim", None)
    acc_ens = int(cfg.get("acc_ens_size", 8)) if noise_dim else 0
    acc_seed = int(cfg.get("acc_noise_seed", 0))
    batch_counter = 0

    def _noise(n_rows: int) -> torch.Tensor:
        nonlocal batch_counter
        g = torch.Generator()
        g.manual_seed(acc_seed * 1_000_003 + batch_counter)
        batch_counter += 1
        return torch.randn(n_rows, acc_ens, int(noise_dim), generator=g).to(dev)

    if acc_ens:
        logger.info("ensemble gate: ACC of the %d-member mean", acc_ens)

    # Per init: running weekly sums, per-source lead counts, and the clim sum.
    sums = defaultdict(lambda: defaultdict(lambda: np.zeros(sel.sum())))
    counts = defaultdict(lambda: defaultdict(int))
    obs = defaultdict(lambda: np.zeros(sel.sum()))
    obs_n = defaultdict(int)
    climsum = defaultdict(lambda: np.zeros(sel.sum()))
    mid_month = {}

    loader = DataLoader(
        ds, batch_size=int(cfg.dataset.loader.get("batch_size", 4)),
        sampler=MixturePairSampler(len(ds), shuffle=False),
        num_workers=int(cfg.dataset.loader.get("num_workers", 4)),
    )
    with torch.no_grad():
        for batch in loader:
            x = batch["expert_inputs"].to(dev)
            mask = batch["expert_mask"].to(dev)
            taus = batch["lead_days"].to(dev)
            if acc_ens:
                weights, biases = model(x, mask, taus, _noise(x.shape[0]))
            else:
                weights, biases = model(x, mask, taus)
            e_mm = denormalize_precip(
                x[:, :, 0], mean=ds.precip_mean, std=ds.precip_std,
                transform=ds.precip_transform,
            )
            if mix_space == "physical":
                g_mm = mix(weights, biases, e_mm, mask=mask).clamp(min=0.0)
            else:
                g_mm = denormalize_precip(
                    mix(weights, biases, x[:, :, 0], mask=mask),
                    mean=ds.precip_mean, std=ds.precip_std,
                    transform=ds.precip_transform)
            if g_mm.ndim == 4:  # (B, ens, H, W) -> ensemble mean
                g_mm = g_mm.mean(dim=1)
            live_t = mask > 0
            live = live_t.cpu().numpy()
            if debias:
                # One pass per expert, on just the samples where it is live.
                per = torch.zeros_like(e_mm)          # (B, E, H, W) debiased
                for ei in range(len(experts)):
                    sub = live_t[:, ei]
                    if not sub.any():
                        continue
                    xi = torch.zeros_like(x[sub])
                    xi[:, ei] = x[sub][:, ei]
                    mi = torch.zeros_like(mask[sub])
                    mi[:, ei] = 1.0
                    if acc_ens:
                        wi, bi = model(xi, mi, taus[sub], _noise(xi.shape[0]))
                    else:
                        wi, bi = model(xi, mi, taus[sub])
                    ei_mm = denormalize_precip(
                        xi[:, :, 0], mean=ds.precip_mean, std=ds.precip_std,
                        transform=ds.precip_transform)
                    if mix_space == "physical":
                        out = mix(wi, bi, ei_mm, mask=mi).clamp(min=0.0)
                    else:
                        out = denormalize_precip(
                            mix(wi, bi, xi[:, :, 0], mask=mi),
                            mean=ds.precip_mean, std=ds.precip_std,
                            transform=ds.precip_transform)
                    if out.ndim == 4:  # (B, ens, H, W) -> ensemble mean
                        out = out.mean(dim=1)
                    per[sub, ei] = out
                src_fields = per
                agg = (per * live_t.float()[..., None, None]).sum(1) / (
                    live_t.sum(1).clamp(min=1)[:, None, None])
            else:
                src_fields = e_mm
                agg = (e_mm * live_t.float()[..., None, None]).sum(1) / (
                    live_t.sum(1).clamp(min=1)[:, None, None])
            e_np = src_fields.cpu().numpy()[..., sel]
            g_np = g_mm.cpu().numpy()[..., sel]
            eq_np = agg.cpu().numpy()[..., sel]
            t_np = batch["target_mm"].squeeze(1).numpy()[..., sel]
            doys = doy_from_hours_since_1900(batch["valid_time"]).numpy()

            for b in range(g_np.shape[0]):
                row = ds.index.pairs[int(batch["pair_idx"][b])]
                i = int(row["init_row"])
                tau = int(batch["lead_days"][b])
                for ei, name in enumerate(experts):
                    if live[b, ei]:
                        sums[i][name] += e_np[b, ei]
                        counts[i][name] += 1
                agg_key = "debiased_mean" if debias else "equal_weight"
                sums[i][agg_key] += eq_np[b]
                counts[i][agg_key] += 1
                sums[i]["gate"] += g_np[b]
                counts[i]["gate"] += 1
                good = np.isfinite(t_np[b])
                obs[i] += np.where(good, t_np[b], 0.0)
                obs_n[i] += 1
                climsum[i] += clim_daily[doys[b] - 1]
                if tau == (lo + hi) // 2:
                    key = cftime.DatetimeGregorian(*ds.index.init_keys[i])
                    mid_month[i] = (
                        key + datetime.timedelta(days=tau - 1)
                    ).month

    # Pool complete weeks by month.
    num = defaultdict(lambda: defaultdict(float))
    den_p = defaultdict(lambda: defaultdict(float))
    den_t = defaultdict(lambda: defaultdict(float))
    # Per-WEEK ACC, so the bars can carry a spread across forecasts. The pooled
    # value (ratio of summed cross-products) weights weeks by anomaly
    # magnitude; the per-week mean does not, so the two differ slightly and
    # both are logged.
    per_week = defaultdict(lambda: defaultdict(list))
    n_init = defaultdict(int)
    dropped = defaultdict(int)
    matched = bool(cfg.get("matched", False))
    if matched:
        keep = {
            i for i in sums
            if obs_n[i] == n_lead and i in mid_month
            and all(counts[i].get(s, 0) == n_lead for s in sources)
        }
        logger.info(
            "matched mode: %d of %d weeks have every source complete",
            len(keep), len(sums),
        )
    else:
        keep = None

    for i, per_src in sums.items():
        if obs_n[i] != n_lead or i not in mid_month:
            dropped["incomplete_obs_or_month"] += 1
            continue
        if keep is not None and i not in keep:
            dropped["not_matched"] += 1
            continue
        m = mid_month[i]
        a_t = obs[i] - climsum[i]
        n_init[m] += 1
        for s in sources:
            if counts[i].get(s, 0) != n_lead:
                dropped[s] += 1
                continue
            a_p = per_src[s] - climsum[i]
            n_i = float((wts * a_p * a_t).sum())
            dp_i = float((wts * a_p * a_p).sum())
            dt_i = float((wts * a_t * a_t).sum())
            num[s][m] += n_i
            den_p[s][m] += dp_i
            den_t[s][m] += dt_i
            # A week with a near-zero anomaly in either field has an unstable
            # single-week ACC; exclude it from the spread rather than let it
            # dominate the standard deviation.
            if dp_i > 1e-9 and dt_i > 1e-9:
                per_week[s][m].append(n_i / np.sqrt(dp_i * dt_i))
            else:
                dropped[f"{s}_degenerate_week"] += 1

    months = sorted(n_init)
    pooled = {s: [num[s][m] / np.sqrt(max(den_p[s][m] * den_t[s][m], 1e-12))
                  for m in months] for s in sources}
    # Bars show the mean of the per-week ACCs so the error bars (their standard
    # deviation) describe the same distribution.
    acc = {s: [float(np.mean(per_week[s][m])) if per_week[s][m] else np.nan
               for m in months] for s in sources}
    sd = {s: [float(np.std(per_week[s][m], ddof=1))
              if len(per_week[s][m]) > 1 else 0.0
              for m in months] for s in sources}
    logger.info("weeks per month: %s", {MONTHS[m - 1]: n_init[m] for m in months})
    logger.info("weeks dropped for an incomplete 7-lead week: %s", dict(dropped))
    for s in sources:
        logger.info("  %-16s %s", LABELS[s], " ".join(
            f"{MONTHS[m-1]}={a:.3f}+-{d:.3f}"
            for m, a, d in zip(months, acc[s], sd[s])))
    for s in sources:
        logger.info("  %-16s pooled %s", LABELS[s], " ".join(
            f"{MONTHS[m-1]}={a:.3f}" for m, a in zip(months, pooled[s])))

    _plot(months, acc, sd, sources, n_init, out=str(cfg.out), show_sd=show_sd,
          debias=debias, matched=matched, lo=lo, hi=hi,
          years=(cfg.dataset.val.years[0], cfg.dataset.val.years[1]))
    out = str(cfg.out)
    logger.info("wrote %s", out)

    csv = Path(out).with_suffix(".csv")
    with open(csv, "w") as fh:
        fh.write("source," + ",".join(MONTHS[m - 1] for m in months) + "\n")
        for s in sources:
            nm = LABELS[s] + (" (debiased)" if debias and s in experts else "")
            fh.write(nm + "," + ",".join(f"{a:.4f}" for a in acc[s]) + "\n")
            fh.write(nm + " sd," + ",".join(f"{d:.4f}" for d in sd[s]) + "\n")
            fh.write(nm + " pooled," + ",".join(
                f"{a:.4f}" for a in pooled[s]) + "\n")
        fh.write("n_weeks," + ",".join(str(n_init[m]) for m in months) + "\n")
    logger.info("wrote %s", csv)


if __name__ == "__main__":
    main()
