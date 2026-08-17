#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Build the per-channel ``noise_scales`` tensor the SI schedulers multiply into.

``DriftScheduler`` (SI) and ``DynamicInterpolant`` (SI_X) both do

    noise = self.get_noise(x)
    if self.noise_scales is not None:
        noise = noise * self.noise_scales

so ``noise_scales`` weights the injected noise per channel. Upstream's SI runs
load one (``noise_scale_path: .../sigma_c_lowres_26.pt``); this fork shipped
``noise_scale_path: null`` and had no way to make one, which is the artifact that
was missing to train these models. Nothing here reads upstream's file — it is
built from our own store.

**What the scale is.** The per-channel standard deviation of the model's own
24-hour increment, in NORMALIZED units:

    sigma_c = std_t[ (x(t + step) - x(t)) / std_c ]

i.e. how much each channel actually moves over one model step, measured in the
same z-scored units the model sees. A channel that barely changes step to step
(deep-soil temperature, stratospheric geopotential) gets a small scale; a fast
one (precipitation, near-surface wind) gets a large one. Injecting isotropic
noise instead over-perturbs the slow channels, which is what this exists to fix.

**Channel ORDER is derived, never restated.** The tensor is indexed by packed
channel, and the pack order is ``channel_layout``-dependent — v1 is
``[surface | diagnostic | upper_air]`` with the upper-air block variable-major,
the fork order differs, and getting it wrong silently mis-scales every channel.
So this builds a per-variable dict and hands it to the wrapper's OWN
``pack_state``, which is the same code the training step uses. Pass the model
config that will consume the artifact.

Output is a ``.pt`` holding a ``(C, 1, 1)`` float32 tensor, ready to broadcast
against ``(B, C, H, W)``, plus the provenance the scheduler never sees:

    {"noise_scales": (C,1,1) tensor, "meta": {...}}

``torch.load`` of that dict is what ``noise_scale_path`` expects to be a bare
tensor, so the tensor is saved at the top level and the metadata alongside it in
a sidecar ``.json`` — see ``--out``.

Usage::

    python tools/data/amip/make_noise_scales.py \
        --zarr $AI_ROSSBY_DATA/amip_dailyavg_coarse \
        --model-config examples/weather/ai_rossby/conf/model/amip_si.yaml \
        --mean $AI_ROSSBY_DATA/amip_dailyavg_coarse/normalize_mean_dailyavg.nc \
        --std $AI_ROSSBY_DATA/amip_dailyavg_coarse/normalize_std_dailyavg.nc \
        --year-start 1979 --year-end 2015 \
        --out $AI_ROSSBY_DATA/norm_stats/sigma_c_amip_dailyavg_coarse.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from xarray.coding.times import CFDatetimeCoder

logger = logging.getLogger("make_noise_scales")

_cftime = CFDatetimeCoder(use_cftime=True)


def _open(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True, decode_times=_cftime)


def _acc_init(shape):
    """Streaming (count, sum, sum-of-squares) — one entry per output scale."""
    return {"n": np.zeros(shape, "int64"), "s": np.zeros(shape, "float64"),
            "ss": np.zeros(shape, "float64")}


def _acc_update(acc, values: np.ndarray, pool_axes: tuple[int, ...]) -> None:
    """Fold ``values`` in, reducing over ``pool_axes``.

    Vectorized on purpose: a per-sample Welford loop would iterate tens of
    millions of grid points for 47 years of 6-hourly increments. float64 sums
    over increments this size have ample headroom, so the shortcut costs nothing.
    """
    acc["n"] += np.prod([values.shape[a] for a in pool_axes], dtype="int64")
    acc["s"] += values.sum(axis=pool_axes)
    acc["ss"] += np.square(values).sum(axis=pool_axes)


def _acc_std(acc) -> np.ndarray:
    n = acc["n"]
    if np.any(n < 2):
        raise ValueError(f"need >= 2 samples to take a std, got {n}")
    var = (acc["ss"] - np.square(acc["s"]) / n) / (n - 1)
    # Cancellation can leave a tiny negative for a near-constant channel.
    return np.sqrt(np.clip(var, 0.0, None))


def _per_variable_norm(nc_path: Path, name: str, levels, is_upper: bool):
    """Normalization std for one variable, as a scalar or a per-level vector."""
    ds = xr.open_dataset(nc_path, decode_times=_cftime)
    if name not in ds:
        raise KeyError(f"{name!r} not in {nc_path}")
    arr = ds[name]
    if not is_upper:
        return float(np.asarray(arr).reshape(-1)[0])
    # Upper air: select the model's levels, in the model's order.
    dim = next((d for d in arr.dims if "level" in d), None)
    if dim is None:
        raise KeyError(f"{name!r} in {nc_path} has no level dim; dims={arr.dims}")
    coord = np.asarray(ds[dim].values, dtype=float)
    idx = [int(np.argmin(np.abs(coord - float(lv)))) for lv in levels]
    picked = np.asarray(arr.isel({dim: idx}).values, dtype=float).reshape(len(levels), -1)
    return picked[:, 0]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--zarr", required=True,
                   help="directory of per-year {year}.zarr stores")
    p.add_argument("--model-config", required=True,
                   help="conf/model/*.yaml whose pack order the tensor must match")
    p.add_argument("--mean", required=True, help="normalization mean .nc")
    p.add_argument("--std", required=True, help="normalization std .nc")
    p.add_argument("--year-start", type=int, required=True)
    p.add_argument("--year-end", type=int, required=True, help="exclusive")
    p.add_argument("--out", required=True, help="output .pt")
    p.add_argument("--sample-stride", type=int, default=1,
                   help="take every Nth increment (a std converges long before "
                        "47 years of 6-hourly rows are exhausted)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # torch is imported late so --help works without it.
    import torch

    cfg = yaml.safe_load(Path(args.model_config).read_text())
    surface = list(cfg.get("surface_variables") or [])
    upper = list(cfg.get("upper_air_variables") or [])
    diagnostic = list(cfg.get("diagnostic_variables") or [])
    levels = [float(v) for v in (cfg.get("levels") or [])]
    step_rows = int(cfg.get("timedelta_hours", 24))  # divided by the store's cadence

    root = Path(args.zarr)
    years = list(range(int(args.year_start), int(args.year_end)))
    stores = [root / f"{y}.zarr" for y in years]
    missing = [s.name for s in stores if not s.exists()]
    if missing:
        raise SystemExit(f"missing stores under {root}: {missing}")

    # Row stride: the MODEL step in store rows, from the store's own cadence.
    with _open(stores[0]) as probe:
        cadence = int(probe.attrs.get("data_timedelta_hours", 0) or 0)
    if cadence <= 0:
        raise SystemExit(f"{stores[0]} declares no data_timedelta_hours")
    if step_rows % cadence:
        raise SystemExit(
            f"model timedelta_hours={step_rows} is not a multiple of the store's "
            f"{cadence} h rows"
        )
    stride = step_rows // cadence
    logger.info(
        "model step %d h over %d h rows -> %d row(s); %d year(s), sample stride %d",
        step_rows, cadence, stride, len(years), args.sample_stride,
    )

    # Accumulate the increment std per variable, pooling all grid points and
    # times. Surface/diagnostic give one number; upper air one per level.
    acc = {name: _acc_init(()) for name in surface + diagnostic}
    acc.update({name: _acc_init((len(levels),)) for name in upper})

    for store in stores:
        with _open(store) as ds:
            n_time = ds.sizes["time"]
            t0 = np.arange(0, n_time - stride, args.sample_stride)
            if not len(t0):
                logger.warning("%s: too short for a %d-row step, skipped",
                               store.name, stride)
                continue
            for name in surface + diagnostic:
                if name not in ds:
                    raise KeyError(f"{name!r} not in {store}")
                a = ds[name].isel(time=t0).values
                b = ds[name].isel(time=t0 + stride).values
                d = (b - a).astype("float64")          # (n, H, W)
                # Pool time AND space: one scale per surface/diagnostic variable.
                _acc_update(acc[name], d, pool_axes=(0, 1, 2))
            for name in upper:
                if name not in ds:
                    raise KeyError(f"{name!r} not in {store}")
                dim = next(d for d in ds[name].dims if "level" in d)
                coord = np.asarray(ds[dim].values, dtype=float)
                idx = [int(np.argmin(np.abs(coord - lv))) for lv in levels]
                a = ds[name].isel(time=t0, **{dim: idx}).values
                b = ds[name].isel(time=t0 + stride, **{dim: idx}).values
                d = (b - a).astype("float64")          # (n, L, H, W)
                # Pool time and space, KEEP levels: one scale per (var, level).
                _acc_update(acc[name], d, pool_axes=(0, 2, 3))
        logger.info("  %s done (n=%d samples for %s)", store.name,
                    int(np.asarray(acc[surface[0]]["n"]).reshape(-1)[0]), surface[0])

    # Physical increment std -> NORMALIZED, by dividing through the same
    # per-channel std the loader z-scores with. Without this the scales carry
    # units and the biggest-magnitude channel dominates the noise.
    std_path = Path(args.std)
    scales: dict[str, np.ndarray] = {}
    for name in surface + diagnostic:
        phys = _acc_std(acc[name])
        norm = _per_variable_norm(std_path, name, levels, is_upper=False)
        scales[name] = np.asarray(float(phys) / float(norm), dtype="float32")
    for name in upper:
        phys = _acc_std(acc[name])
        norm = _per_variable_norm(std_path, name, levels, is_upper=True)
        scales[name] = (phys / norm).astype("float32")

    # PACK with the wrapper's own code, so the channel order cannot drift from
    # what the training step uses. This is the only step that needs physicsnemo.
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        from physicsnemo.experimental.models import amip_si as models

    non_kwargs = {"name", "module", "target", "model_type", "timedelta_hours", "defaults"}
    wrapper = getattr(models, str(cfg["name"]))(
        **{k: v for k, v in cfg.items() if k not in non_kwargs}
    )
    one = lambda v: torch.as_tensor(np.broadcast_to(v, (1, 1)).copy())  # noqa: E731
    sample = {
        "surface_in": torch.stack([one(scales[n]) for n in surface]),
        "diagnostic": (
            torch.stack([one(scales[n]) for n in diagnostic]) if diagnostic else None
        ),
        "upper_air_in": (
            torch.stack([
                torch.stack([one(scales[n][li]) for li in range(len(levels))])
                for n in upper
            ]) if upper else None
        ),
    }
    packed = wrapper.pack_state(sample)          # (C, 1, 1) in pack order
    if packed.shape[0] != wrapper.in_channels:
        raise SystemExit(
            f"packed {packed.shape[0]} channels but the wrapper expects "
            f"{wrapper.in_channels}"
        )
    tensor = packed.to(torch.float32).contiguous()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Saved as a BARE tensor: that is what `noise_scale_path` torch.loads.
    torch.save(tensor, out)
    meta = {
        "source_zarr": str(root),
        "years": f"{args.year_start}-{args.year_end} (end exclusive)",
        "sample_stride": args.sample_stride,
        "model_config": str(args.model_config),
        "channel_layout": getattr(wrapper, "channel_layout", None),
        "channels": int(tensor.shape[0]),
        "model_step_hours": step_rows,
        "store_cadence_hours": cadence,
        "samples_per_variable": int(np.asarray(acc[surface[0]]["n"]).reshape(-1)[0]),
        "normalization_std": str(std_path),
        "scale_min": float(tensor.min()),
        "scale_max": float(tensor.max()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s  %s  (min %.4g, max %.4g)", out, tuple(tensor.shape),
                meta["scale_min"], meta["scale_max"])
    logger.info("provenance -> %s", out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
