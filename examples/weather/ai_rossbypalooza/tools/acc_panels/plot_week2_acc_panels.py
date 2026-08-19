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

"""OFFLINE: per-gridpoint week-2 ACC maps for every source, plus gate deltas.

The driver: parse arguments, call the steps in ``acc_panels_utils.py`` in
order, hand the results to ``acc_panels_plots.py``. Two figures come out --
a 2x3 grid of the six sources, and a 2x1 gate-minus-reference difference.
No checkpoint and no GPU.

**Read ``README.md`` in this folder first**: it carries the diagram, the
workflow and the reasoning behind every step below.

Usage (Derecho)::

    python examples/weather/ai_rossbypalooza/tools/acc_panels/plot_week2_acc_panels.py \\
        --forecast /glade/derecho/scratch/syback/mowe_forecasts/cv5_physvar.zarr \\
        --dataset-config examples/weather/ai_rossbypalooza/conf/dataset/hindcast_derecho.yaml \\
        --climatology .../imerg_seeps_climatology_daily.zarr \\
        --cartopy-data /glade/derecho/scratch/$USER/cartopy_data \\
        --out /glade/derecho/scratch/$USER/mowe_runs/week2_acc_panels.png

or, with every Derecho path already filled in::

    qsub examples/weather/ai_rossbypalooza/tools/acc_panels/run_week2_acc_panels_derecho.sh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# The recipe root for `datapipes` / `losses`, then this folder for the two
# sibling modules. Hence the E402 markers on the imports below.
_RECIPE_DIR = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (str(_RECIPE_DIR), str(_PACKAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from acc_panels_plots import plot_gate_difference, plot_source_grid  # noqa: E402
from acc_panels_utils import (  # noqa: E402
    accumulate_weekly_totals,
    find_scored_weeks,
    load_daily_climatology,
    load_gate_weekly_totals,
    panel_source_order,
    pick_reference_expert,
    save_score_maps,
    score_every_source,
    select_scored_region,
)
from datapipes.factory import build_dataset  # noqa: E402

logger = logging.getLogger("week2_acc_panels")

SCRIPT_REL_PATH = (
    "examples/weather/ai_rossbypalooza/tools/acc_panels/plot_week2_acc_panels.py"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments for the panel figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])

    # --- inputs -----------------------------------------------------------
    parser.add_argument(
        "--forecast",
        type=Path,
        required=True,
        help="gate inference zarr written by tools/infer_mowe.py",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="conf/dataset/*.yaml; supplies the experts, leads and IMD store",
    )
    parser.add_argument(
        "--climatology",
        type=Path,
        required=True,
        help="SEEPS climatology store carrying clim_mean_daily (366, lat, lon)",
    )
    parser.add_argument(
        "--imd-store",
        default=None,
        help="IMD analysis zarr; defaults to the one named in the dataset config",
    )

    # --- what gets scored -------------------------------------------------
    parser.add_argument(
        "--var",
        default="total_precipitation_24hr",
        help="precip variable name in the inference zarr",
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=[5.0, 35.0, 60.0, 100.0],
        metavar=("LAT0", "LAT1", "LON0", "LON1"),
        help="monsoon box, intersected with the IMD gauge mask",
    )
    parser.add_argument(
        "--min-finite-frac",
        type=float,
        default=0.99,
        help="a gridpoint needs this fraction of finite IMD days to be scored",
    )
    parser.add_argument(
        "--matched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score every source on the weeks all of them cover (recommended)",
    )
    parser.add_argument(
        "--metric",
        choices=("acc", "rmse"),
        default="acc",
        help="acc: anomaly correlation (higher better); "
        "rmse: mm/week error of the weekly total (lower better)",
    )

    # --- how the datapipe pass runs ---------------------------------------
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)

    # --- outputs ----------------------------------------------------------
    parser.add_argument(
        "--vmin", type=float, default=None, help="colour scale floor (default 0)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="colour scale ceiling; default 1.0 for acc, data-driven for rmse",
    )
    parser.add_argument(
        "--cartopy-data",
        default=None,
        help="dir holding shapefiles/natural_earth/, for offline basemaps",
    )
    parser.add_argument("--out", type=Path, required=True, help="six-panel figure")
    parser.add_argument(
        "--out-diff",
        type=Path,
        default=None,
        help="difference figure; defaults to <out>_diff.png",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compute the six skill maps and draw the two figures."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --- 0. the datapipe: it defines the experts, the leads and the grid --
    dataset_config = OmegaConf.load(args.dataset_config)
    dataset = build_dataset(dataset_config, "val")
    first_lead_day, last_lead_day = (int(v) for v in dataset_config.val.lead_days)
    n_leads_per_week = last_lead_day - first_lead_day + 1
    imd_store = args.imd_store or str(dataset_config.imd.store)
    logger.info(
        "validation split: leads %d-%d (%d per week), experts %s",
        first_lead_day,
        last_lead_day,
        n_leads_per_week,
        ", ".join(dataset.expert_names),
    )

    # --- 1. which gridpoints are scored -----------------------------------
    region = select_scored_region(
        np.asarray(dataset.lat),
        np.asarray(dataset.lon),
        tuple(args.region),
        imd_store,
        args.min_finite_frac,
    )

    # --- 2. the day-of-year reference the anomalies are taken against -----
    climatology_daily = load_daily_climatology(args.climatology, region)

    # --- 3. the gate's weekly totals, straight off the inference zarr -----
    gate_weekly_totals, lead_days = load_gate_weekly_totals(
        args.forecast, args.var, region
    )

    # --- 4. experts, equal-weight, truth and weekly climatology -----------
    totals = accumulate_weekly_totals(
        dataset,
        region,
        climatology_daily,
        n_leads_per_week,
        args.batch_size,
        args.num_workers,
    )

    # --- 5. which initializations every source covers ---------------------
    scored_weeks = find_scored_weeks(totals, gate_weekly_totals, args.matched)

    # --- 6. one skill map per source --------------------------------------
    scores = score_every_source(
        totals, gate_weekly_totals, region, scored_weeks, args.metric
    )
    source_names = panel_source_order(totals.expert_names)

    # --- 7. the reference the difference figure subtracts -----------------
    best_expert = pick_reference_expert(scores, totals.expert_names, args.metric)

    # --- 8. the two figures, and the numbers behind them ------------------
    plot_source_grid(
        scores=scores,
        source_names=source_names,
        region=region,
        metric=args.metric,
        matched=args.matched,
        n_weeks=scored_weeks.n_weeks,
        lead_days=lead_days,
        out_path=args.out,
        vmin=args.vmin,
        vmax=args.vmax,
        cartopy_data=args.cartopy_data,
    )
    plot_gate_difference(
        scores=scores,
        best_expert=best_expert,
        region=region,
        metric=args.metric,
        out_path=args.out_diff or args.out.with_name(args.out.stem + "_diff.png"),
        cartopy_data=args.cartopy_data,
    )
    save_score_maps(
        out_path=args.out.with_suffix(".nc"),
        scores=scores,
        source_names=source_names,
        region=region,
        metric=args.metric,
        matched=args.matched,
        n_weeks=scored_weeks.n_weeks,
        best_expert=best_expert,
        generator=SCRIPT_REL_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
