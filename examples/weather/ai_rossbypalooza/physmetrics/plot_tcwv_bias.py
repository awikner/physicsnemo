#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot mean TCWV (model, ERA5, and bias) maps per lead day from tcwv_bias_maps_<model>.npz."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Sequential blue ramp (light -> dark), for magnitude (TCWV) panels
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# Diverging blue <-> red pair with neutral gray midpoint, for bias panels
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8a1f1e"],
)

DEFAULT_REGION = (5, 35, 60, 100)  # lat_min, lat_max, lon_min, lon_max (monsoon domain)


def _crop(lat, lon, field, region):
    lat_min, lat_max, lon_min, lon_max = region
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    return lat[lat_mask], lon[lon_mask], field[np.ix_(lat_mask, lon_mask)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=str, help="Path to tcwv_bias_maps_<model>.npz")
    parser.add_argument("--outdir", type=str, default="plots")
    parser.add_argument(
        "--region", type=float, nargs=4, default=DEFAULT_REGION,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Crop region (default: monsoon domain 5-35N, 60-100E)",
    )
    parser.add_argument(
        "--flip-lat", action="store_true",
        help="Reverse the latitude axis before plotting (temporary workaround "
             "while the upstream lat-orientation bug is being fixed)",
    )
    args = parser.parse_args()

    data = np.load(args.npz)
    lat, lon = data["lat"], data["lon"]
    model_name = Path(args.npz).stem.replace("tcwv_bias_maps_", "")

    lead_days = sorted(
        int(k.replace("model_mean_lead", ""))
        for k in data.files if k.startswith("model_mean_lead")
    )

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for ld in lead_days:
        model_tcwv = data[f"model_mean_lead{ld}"]
        era5_tcwv = data[f"era5_mean_lead{ld}"]
        bias = model_tcwv - era5_tcwv

        lat_c, lon_c, model_c = _crop(lat, lon, model_tcwv, args.region)
        _, _, era5_c = _crop(lat, lon, era5_tcwv, args.region)
        _, _, bias_c = _crop(lat, lon, bias, args.region)

        if args.flip_lat:
            # Reverse only the coordinate array, not the data rows: this mislabels
            # each row's latitude on purpose, producing the upside-down render.
            lat_c = lat_c[::-1]

        vmax_mag = max(np.nanmax(model_c), np.nanmax(era5_c))
        vmin_mag = min(np.nanmin(model_c), np.nanmin(era5_c))
        bias_abs = np.nanmax(np.abs(bias_c))

        fig, axes = plt.subplots(
            1, 3, figsize=(16, 5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        for ax, field, title, cmap, vmin, vmax in [
            (axes[0], model_c, f"{model_name} TCWV (kg/m²)", SEQ_BLUE, vmin_mag, vmax_mag),
            (axes[1], era5_c, "ERA5 TCWV (kg/m²)", SEQ_BLUE, vmin_mag, vmax_mag),
            (axes[2], bias_c, f"Bias ({model_name} − ERA5)", DIV_BLUE_RED, -bias_abs, bias_abs),
        ]:
            im = ax.pcolormesh(
                lon_c, lat_c, field, cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto", transform=ccrs.PlateCarree(),
            )
            ax.coastlines(linewidth=0.8, color="#52514e")
            ax.set_title(title, fontsize=12)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#e1e0d9")
            gl.top_labels = False
            gl.right_labels = False
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.9)

        fig.suptitle(f"Lead day {ld}", fontsize=14)
        fig.tight_layout()
        outpath = outdir / f"tcwv_bias_{model_name}_lead{ld}.png"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
