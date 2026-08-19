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

"""The two figures for the week-2 skill panels. No science here.

Draws maps that ``acc_panels_utils.py`` has already computed. One convention
holds throughout: blue means better and better points UP, which forces both
the colormap and the colourbar direction to flip between ACC and RMSE.

Matplotlib and cartopy are imported inside the functions, so ``--help`` does
not pay for them.

See ``README.md`` §3 for the panel layouts and the colour-scale traps.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from acc_panels_utils import (  # noqa: E402
    EQUAL_WEIGHT,
    GATE,
    SOURCE_LABELS,
    ScoredRegion,
)

logger = logging.getLogger("week2_acc_panels")

# Padding in degrees around the data box, so the region does not touch the
# frame and the coastline stays readable at its edge.
MAP_PADDING_DEG = 1.5


def _import_plotting_backends(cartopy_data: str | None):
    """Import matplotlib/cartopy headlessly and point cartopy at local data.

    ``pre_existing_data_dir`` wins over the download cache, so the basemap
    resolves offline on a compute node with no network.
    """
    import matplotlib

    matplotlib.use("Agg")
    import cartopy

    if cartopy_data:
        cartopy.config["pre_existing_data_dir"] = cartopy_data
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    return ccrs, cfeature, plt


def _cell_edges(region: ScoredRegion) -> tuple[np.ndarray, np.ndarray]:
    """Cell EDGES for ``pcolormesh``, from the gridpoint CENTRES.

    Handing pcolormesh the centres would shift the field half a gridbox.
    """
    lon_spacing = float(np.diff(region.lon).mean())
    lat_spacing = float(np.diff(region.lat).mean())
    lon_edges = np.append(
        region.lon - lon_spacing / 2, region.lon[-1] + lon_spacing / 2
    )
    lat_edges = np.append(
        region.lat - lat_spacing / 2, region.lat[-1] + lat_spacing / 2
    )
    return lon_edges, lat_edges


def _map_extent(lon_edges: np.ndarray, lat_edges: np.ndarray) -> list[float]:
    """The padded ``[lon0, lon1, lat0, lat1]`` extent shared by every panel."""
    return [
        lon_edges.min() - MAP_PADDING_DEG,
        lon_edges.max() + MAP_PADDING_DEG,
        lat_edges.min() - MAP_PADDING_DEG,
        lat_edges.max() + MAP_PADDING_DEG,
    ]


def _add_geo_axes(figure, n_rows, n_cols, panel_index, extent, projection, cfeature):
    """One PlateCarree panel with coastlines, borders and gridline labels.

    Geography sits over the data but under the labels, thin and grey, so it
    frames the field without competing with it.
    """
    axes = figure.add_subplot(n_rows, n_cols, panel_index, projection=projection)
    axes.set_extent(extent, crs=projection)
    axes.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.5,
        edgecolor="#52514e",
        zorder=2,
    )
    axes.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.4,
        edgecolor="#7a7975",
        zorder=2,
    )
    gridlines = axes.gridlines(
        draw_labels=True, linewidth=0.3, color="#e1e0d9", zorder=3
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 7}
    gridlines.ylabel_style = {"size": 7}
    return axes


def plot_source_grid(
    scores: dict[str, np.ndarray],
    source_names: list[str],
    region: ScoredRegion,
    metric: str,
    matched: bool,
    n_weeks: int,
    lead_days: np.ndarray,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cartopy_data: str | None = None,
) -> Path:
    """Six source panels on one shared colour scale, 2 rows x 3 columns.

    ACC is pinned to 0-1 so panels stay comparable across runs; RMSE has no
    natural bound and is fitted to the data unless ``vmax`` is given.
    Colormap direction is the trap here -- README §3.
    """
    ccrs, cfeature, plt = _import_plotting_backends(cartopy_data)
    projection = ccrs.PlateCarree()
    lon_edges, lat_edges = _cell_edges(region)
    extent = _map_extent(lon_edges, lat_edges)

    is_acc = metric == "acc"
    colormap = "RdYlBu" if is_acc else "RdYlBu_r"
    scale_min = vmin if vmin is not None else 0.0
    if vmax is not None:
        scale_max = vmax
    elif is_acc:
        scale_max = 1.0
    else:
        # 98th percentile, so one extreme gridpoint cannot set the scale.
        scale_max = float(
            np.ceil(max(np.nanpercentile(scores[s], 98) for s in source_names) / 5.0)
            * 5.0
        )
    colorbar_label = (
        "Anomaly correlation (ACC) — better ↑"
        if is_acc
        else "RMSE of weekly total, mm/week — better ↑"
    )
    median_format = "{:.3f}" if is_acc else "{:.1f}"

    figure = plt.figure(figsize=(15.5, 8.8))
    panel_tags = "abcdefgh"
    for panel_index, source in enumerate(source_names):
        axes = _add_geo_axes(
            figure, 2, 3, panel_index + 1, extent, projection, cfeature
        )
        mesh = axes.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_invalid(scores[source]),
            cmap=colormap,
            vmin=scale_min,
            vmax=scale_max,
            shading="flat",
            transform=projection,
            zorder=1,
        )
        axes.set_title(
            f"({panel_tags[panel_index]}) {SOURCE_LABELS[source]}   median "
            + median_format.format(np.nanmedian(scores[source])),
            fontsize=10,
        )

    figure.subplots_adjust(
        left=0.04, right=0.89, top=0.89, bottom=0.05, hspace=0.18, wspace=0.12
    )
    colorbar_axes = figure.add_axes((0.905, 0.15, 0.014, 0.68))
    colorbar = figure.colorbar(
        mesh, cax=colorbar_axes, label=colorbar_label, extend="max"
    )
    colorbar.outline.set_visible(False)
    # "Better" points UP: ACC already does, RMSE must be inverted. This
    # flips the legend only -- the data-to-colour mapping is untouched.
    if not is_acc:
        colorbar.ax.invert_yaxis()

    figure.suptitle(
        "Week-2 accumulated precipitation "
        f"{'ACC' if is_acc else 'RMSE'} by source\n"
        f"leads {lead_days.min()}-{lead_days.max()} summed to weekly totals; "
        f"per-gridpoint over {n_weeks} "
        f"{'matched' if matched else 'available'} weeks, IMD region",
        fontsize=12,
    )
    figure.savefig(out_path, dpi=170, bbox_inches="tight")
    logger.info("wrote %s", out_path)
    plt.close(figure)
    return out_path


def plot_gate_difference(
    scores: dict[str, np.ndarray],
    best_expert: str,
    region: ScoredRegion,
    metric: str,
    out_path: Path,
    cartopy_data: str | None = None,
) -> Path:
    """Gate minus best expert and minus equal-weight, diverging about zero.

    Blue keeps meaning "gate better", so the colormap flips with the metric:
    ACC improves when the difference is POSITIVE, RMSE when it is NEGATIVE.
    The "gate better at X%" tally flips with it -- README §3.
    """
    ccrs, cfeature, plt = _import_plotting_backends(cartopy_data)
    projection = ccrs.PlateCarree()
    lon_edges, lat_edges = _cell_edges(region)
    extent = _map_extent(lon_edges, lat_edges)

    panels = [
        (
            scores[GATE] - scores[best_expert],
            f"MoWE gate − {SOURCE_LABELS[best_expert]}",
        ),
        (
            scores[GATE] - scores[EQUAL_WEIGHT],
            f"MoWE gate − {SOURCE_LABELS[EQUAL_WEIGHT]}",
        ),
    ]
    is_acc = metric == "acc"
    # Symmetric 99th-percentile limits, so one outlier cannot wash the
    # figure out to pale colours.
    scale_max = max(
        float(np.nanpercentile(np.abs(difference), 99)) for difference, _ in panels
    )
    scale_max = max(0.02, np.ceil(scale_max * 100) / 100)
    colormap = "RdBu" if is_acc else "RdBu_r"
    median_format = "{:+.3f}" if is_acc else "{:+.2f}"

    figure = plt.figure(figsize=(6.8, 10.6))
    for panel_index, (difference, title) in enumerate(panels):
        axes = _add_geo_axes(
            figure, 2, 1, panel_index + 1, extent, projection, cfeature
        )
        mesh = axes.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_invalid(difference),
            cmap=colormap,
            vmin=-scale_max,
            vmax=scale_max,
            shading="flat",
            transform=projection,
            zorder=1,
        )

        # Restrict to finite cells EXPLICITLY: np.nanmean(difference > 0)
        # silently averages over the whole bounding box -- README §3.
        is_finite = np.isfinite(difference)
        gate_is_better = (
            difference[is_finite] > 0 if is_acc else difference[is_finite] < 0
        )
        better_fraction = float(np.mean(gate_is_better))
        axes.set_title(
            f"({'ab'[panel_index]}) {title}\nmedian "
            + median_format.format(np.nanmedian(difference))
            + f", gate better at {100 * better_fraction:.0f}% of points",
            fontsize=10,
        )

    figure.subplots_adjust(left=0.08, right=0.80, top=0.93, bottom=0.04, hspace=0.22)
    colorbar_axes = figure.add_axes((0.835, 0.20, 0.030, 0.60))
    colorbar = figure.colorbar(
        mesh,
        cax=colorbar_axes,
        label=(
            "Δ ACC  (blue = gate better ↑)"
            if is_acc
            else "Δ RMSE, mm/week  (blue = gate better ↑)"
        ),
    )
    colorbar.outline.set_visible(False)
    # Same convention as the grid: gate-better points up, so RMSE inverts.
    if not is_acc:
        colorbar.ax.invert_yaxis()

    figure.savefig(out_path, dpi=170, bbox_inches="tight")
    logger.info("wrote %s", out_path)
    plt.close(figure)
    return out_path
