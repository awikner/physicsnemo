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
}
LABELS = {
    "pangu_s2s": "Pangu-S2S", "sfno_era5": "SFNO-S2S",
    "graphcast": "GraphCast", "aifs_single_v2": "AIFS",
    "equal_weight": "Equal weight", "gate": "MoWE gate",
}


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
    sources = [*experts, "equal_weight", "gate"]
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
            live = (mask > 0).cpu().numpy()
            eq = (e_mm * (mask > 0).float()[..., None, None]).sum(1) / (
                mask > 0).sum(1).clamp(min=1)[:, None, None]
            e_np = e_mm.cpu().numpy()[..., sel]
            g_np = g_mm.cpu().numpy()[..., sel]
            eq_np = eq.cpu().numpy()[..., sel]
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
                sums[i]["equal_weight"] += eq_np[b]
                counts[i]["equal_weight"] += 1
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
            num[s][m] += float((wts * a_p * a_t).sum())
            den_p[s][m] += float((wts * a_p * a_p).sum())
            den_t[s][m] += float((wts * a_t * a_t).sum())

    months = sorted(n_init)
    acc = {s: [num[s][m] / np.sqrt(max(den_p[s][m] * den_t[s][m], 1e-12))
               for m in months] for s in sources}
    logger.info("weeks per month: %s", {MONTHS[m - 1]: n_init[m] for m in months})
    logger.info("weeks dropped for an incomplete 7-lead week: %s", dict(dropped))
    for s in sources:
        logger.info("  %-16s %s", LABELS[s],
                    " ".join(f"{MONTHS[m-1]}={a:.3f}" for m, a in zip(months, acc[s])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.2))
    xpos = np.arange(len(months))
    width = 0.8 / len(sources)
    for k, s in enumerate(sources):
        ax.bar(xpos + (k - (len(sources) - 1) / 2) * width, acc[s], width,
               label=LABELS[s], color=COLOURS[s],
               edgecolor="black", linewidth=0.5,
               hatch="//" if s == "gate" else None, zorder=3)
    ax.set_xticks(xpos, [MONTHS[m - 1] for m in months])
    ax.set_ylabel("Anomaly correlation (ACC)")
    ax.set_xlabel("Month (of the week-2 midpoint)")
    ax.set_title(
        "Week-2 accumulated precipitation ACC, IMD region\n"
        f"leads {lo}-{hi} summed to weekly totals; "
        f"{cfg.dataset.val.years[0]}-{cfg.dataset.val.years[1]} pooled per month"
        + ("; weeks where every source is available"
           if matched else "; each source on the weeks it covers")
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    # Headroom so the legend never sits on top of a bar.
    ax.set_ylim(0.0, max(max(v) for v in acc.values()) * 1.22)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(ncol=6, frameon=False, loc="upper center", fontsize=9)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    # Below the axes, not over the bars.
    fig.text(0.5, 0.012, "weeks per month: " + "  ".join(
        f"{MONTHS[m-1]} {n_init[m]}" for m in months),
        ha="center", va="bottom", fontsize=8, color="#444444")
    out = str(cfg.out)
    fig.savefig(out, dpi=180)
    logger.info("wrote %s", out)

    csv = Path(out).with_suffix(".csv")
    with open(csv, "w") as fh:
        fh.write("source," + ",".join(MONTHS[m - 1] for m in months) + "\n")
        for s in sources:
            fh.write(LABELS[s] + "," + ",".join(f"{a:.4f}" for a in acc[s]) + "\n")
        fh.write("n_weeks," + ",".join(str(n_init[m]) for m in months) + "\n")
    logger.info("wrote %s", csv)


if __name__ == "__main__":
    main()
