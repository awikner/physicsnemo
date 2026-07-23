<!--
SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PhysMetrics.Weather — physics-consistency diagnostics for the hackathon hindcasts

[PhysMetrics.Weather](https://github.com/Emmakast/PhysMetrics.Weather) computes
physics-based diagnostics (dry-air-mass / water-mass / energy conservation,
spectral resolution, hydrostatic/geostrophic balance, lapse rate) for AI
weather models. We use it to check whether `pangu_s2s` / `sfno_s2s` conserve
mass and how their total-column water vapor (TCWV) compares to ERA5, for the
`ai-rossbypalooza` hindcast archives.

This note covers: (1) installing the package (its `pyproject.toml` ships
broken — needs a one-time fix), and (2) three custom scripts that call its
functions directly rather than its CLI, plus their matching plotting
scripts.

## 1. Installing PhysMetrics.Weather

```bash
git clone https://github.com/Emmakast/PhysMetrics.Weather.git ~/PhysMetrics.Weather
cd ~/PhysMetrics.Weather
```

The repo's `pyproject.toml` has the package/module names wrong everywhere
(`physeval_weather`/`physeval-*` instead of the actual source directory
`src/physmetrics_weather/` and the README's documented `physmetrics-run`/
`physmetrics-plot` commands), and is missing `requires-python`. Fix both:

```bash
sed -i.bak 's/physeval/physmetrics/g' pyproject.toml
sed -i '/^readme = "README.md"$/a requires-python = ">=3.10"' pyproject.toml
uv sync --reinstall
```

Verify:
```bash
uv run physmetrics-run --help
```

## 2. Available codes

Instead of running every diagnostic from the library, we chose to cherry-pick.
Only water-related metrics are computed: total water mass, total dry air mass
and TCWV maps.

Running these on a compute node will use multi-CPU efficiently.

### `mass_drift_only.py`

Per-`(model, init, lead_day)` dry/water mass drift (%/day), matching the
package's own `compute_drift_percentages` regression formula (linear slope
over the window `[12h, lead_day×24h]`, as % of the starting value; water mass
is scored against ERA5's own trend over the same window, since real water mass
legitimately fluctuates). Has a persistent per-worker ERA5 cache (keyed by
valid time) and interleaves `(model, init)` tasks so nearby inits/shared valid
times land in the same worker chunk and hit the cache.

```bash
uv run --project ~/PhysMetrics.Weather python mass_drift_only.py --workers 8 --lead-days 8,9,10,11,12,13,14
```

`--lead-days` accepts any comma-separated list (minimum usable value is 2 —
day 1 only has one point left in the window once the 12h spin-up cutoff
excludes 0h, so no slope can be fit). Output: `results/mass_drift_week2.csv`.
Source: `mass_drift_only.py` in the repo.

### `mass_trajectory.py`

Raw (non-drift) mean dry/water mass per lead day for `pangu_s2s`, `sfno_s2s`,
`era5` (looked up at each init's matching valid time), and
`pangu_s2s_msl_derived` (Pangu forced through the same MSL-hypsometric surface-
pressure derivation `sfno_s2s` needs, isolating derivation-method effects from
genuine model differences). Use this to see the actual trajectory shape — e.g.
Pangu's sharp day-0→1 dip (an autoregressive initialization/spin-up shock)
followed by a multi-day climb that a regression-slope summary alone can hide.

```bash
uv run --project ~/PhysMetrics.Weather python mass_trajectory.py --workers 8
```

Output: `results/mass_trajectory.csv` (columns: `model`, `lead_day`,
`dry_mass_Eg_mean`, `dry_mass_Eg_std`, `water_mass_kg_mean`,
`water_mass_kg_std`, `n`). Source: `mass_trajectory.py` in the repo.

### `tcwv_bias_maps.py`

Mean TCWV `(lat, lon)` bias maps (`model − ERA5`), averaged over all inits, per
lead day, for one model at a time.

```bash
uv run --project ~/PhysMetrics.Weather python tcwv_bias_maps.py --workers 8 --model pangu_s2s --lead-days 8,9,10,11,12,13,14
uv run --project ~/PhysMetrics.Weather python tcwv_bias_maps.py --workers 8 --model sfno_s2s --lead-days 8,9,10,11,12,13,14
```

Output: `results/tcwv_bias_maps_<model>.npz` (keys: `lat`, `lon`,
`model_mean_lead{N}`, `era5_mean_lead{N}`, `n_lead{N}` per requested lead day).
Source: `tcwv_bias_maps.py` in the repo.

## 3. Plotting the results

Each of the three scripts above has a matching plotting script — run it right
after, on the same machine, pointed at the CSV/`.npz` each one writes into
`results/`. No display needed (`matplotlib` writes straight to PNG files);
needs `pandas`, `matplotlib`, and — for the TCWV maps only — `cartopy`.

### `plot_mass_drift.py` — dry/water mass drift (%/day) vs. lead time

```bash
python3 plot_mass_drift.py results/mass_drift_week2.csv --outdir plots
```

Source: `plot_mass_drift.py` in the repo.

### `plot_mass_trajectory.py` — absolute dry/water mass vs. lead time

```bash
# Default: pangu_s2s, sfno_s2s, era5
python3 plot_mass_trajectory.py results/mass_trajectory.csv --outdir plots

# Or pick any subset, e.g. include the MSL-derived-PS comparison:
python3 plot_mass_trajectory.py results/mass_trajectory.csv --outdir plots \
    --models pangu_s2s,pangu_s2s_msl_derived,sfno_s2s,era5
```

Source: `plot_mass_trajectory.py` in the repo.

### `plot_tcwv_bias.py` — TCWV maps (model, ERA5, bias) per lead day

```bash
python3 plot_tcwv_bias.py results/tcwv_bias_maps_pangu_s2s.npz --outdir plots
python3 plot_tcwv_bias.py results/tcwv_bias_maps_sfno_s2s.npz --outdir plots

# Wider/global region instead of the default monsoon domain (5-35N, 60-100E):
# NOTE lon is 0-360 in this data, not -180/180 -- use 0 360 for "global", not -180 180.
python3 plot_tcwv_bias.py results/tcwv_bias_maps_pangu_s2s.npz --outdir plots --region -90 90 0 360
```

Produces a 3-panel figure per lead day (model TCWV, ERA5 TCWV on a shared
sequential-blue scale, and the bias on a diverging blue↔red scale centered on
zero), cropped to the monsoon domain by default. Also has `--flip-lat` (only
useful as a temporary manual workaround if a similar lat-orientation bug shows
up again before it's fixed at the source — reverses just the coordinate
labels, not the data, to intentionally mis-render for comparison). Source:
`plot_tcwv_bias.py` in the repo.

## 4. Interpreting results — gotchas worth knowing up front

- **Mass drift is a *rate* (%/day, regression slope), not a cumulative amount**
  — it can be non-zero even at short lead times, and doesn't have to trend
  toward 0 as lead time shrinks.
- **The regression window expands, not slides** — `lead_day=14`'s drift
  averages the *entire* `[12h, 336h]` trajectory, including everything that
  went into `lead_day=8`'s estimate. If the true trajectory is front-loaded
  (fast early change, flat later), the expanding-window slope will *decrease*
  with lead day even if nothing "slowed down" in a smooth sense — check
  `mass_trajectory.py`'s raw values if a drift curve's shape looks surprising.
- **Mass-drift RMSE-like reasoning doesn't transfer.** Distance-from-truth
  (RMSE vs. ERA5) grows with lead time almost universally (chaotic error
  growth); mass-conservation-violation *rate* is a different, often
  front-loaded/transient failure mode (e.g. an autoregressive
  initialization/spin-up shock in the first 1-2 days) — the two can move in
  opposite directions without contradiction.
- **`sfno_s2s` lacks a direct `surface_pressure` variable** (so does `era5`,
  which does have it directly available as of the current store — only
  `sfno_s2s` needs derivation). `_get_ps` falls back to the hypsometric
  US Standard Atmosphere formula from MSL + era5's static
  `geopotential_at_surface`. This is a real, tested fallback in the package,
  not something we added — but it means `sfno_s2s`'s mass/TCWV numbers carry
  more structural uncertainty than `pangu_s2s`'s.
