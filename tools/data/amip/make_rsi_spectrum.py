#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Fit the spectral envelope ``g(l)`` for Rolling Stochastic Interpolants.

RSI's latent amplitude can be shaped in spherical-harmonic space,

    Gamma(tau, l) = gamma_0 * g(l) * h(tau, l),

turning ERDM's 1-D progressive schedule into a 2-D one over
(lead time, wavenumber). ``h`` is analytic (see ``_spectral.py``); ``g`` is what
this tool fits.

**What g is.** The per-degree amplitude spectrum of the model's own one-step
increment, in NORMALIZED units, band-averaged and normalized to unit mean:

    g(l)^2  ~  < |SHT[ (x(t+step) - x(t)) / std_c ]_{l,m}|^2 >_{m, t, c}

so the injected latent is shaped like the transition variability actually is,
rather than white. At tau ~ 0 a slot then looks like "predecessor plus a
physically plausible perturbation" instead of "predecessor plus white noise".

**What g is NOT.** This is the INCREMENT spectrum, which at large scales is
dominated by deterministic advection rather than by uncertainty — it is not
literally the conditional spread, and the proposal's phrasing conflates the two.
It is still the right default: it is the scale on which the entering slot's
anchor is wrong, and it keeps the velocity regression well-conditioned band by
band. A narrower "true spread" envelope (e.g. fit to ensemble spread rather than
to increments) is a separate ablation, and this tool's ``--out`` is where it
would go.

**Normalization.** The envelope is renormalized to unit band-mean on load by
``SphericalSpectralFilter``, so it only ever redistributes amplitude across
scales — the overall level stays ``gamma_0``'s job and an envelope swap cannot
silently move the noise magnitude.

**Bandwidth.** ``lmax`` defaults to ``nlat // 2``, the exactness limit of
equiangular quadrature. At ``lmax = nlat`` the analysis/synthesis pair is
aliased badly enough that it is not even a projection, which would break the
score identity silently — the filter defaults the same way, and the two must
agree or the envelope will be the wrong length (which raises).

Output is a ``.pt`` holding ``{"envelope": (L,) float32, "meta": {...}}`` plus a
sidecar ``.json``. Point ``loss.spectrum_path`` at it.

Usage::

    python tools/data/amip/make_rsi_spectrum.py \
        --zarr $AI_ROSSBY_DATA/amip_dailyavg_coarse \
        --model-config examples/weather/ai_rossby/conf/model/amip_rsi_v2.yaml \
        --std $AI_ROSSBY_DATA/amip_dailyavg_coarse/normalize_std_dailyavg.nc \
        --year-start 1979 --year-end 2015 \
        --out $AI_ROSSBY_DATA/norm_stats/g_l_amip_dailyavg_coarse.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from xarray.coding.times import CFDatetimeCoder

logger = logging.getLogger("make_rsi_spectrum")

_cftime = CFDatetimeCoder(use_cftime=True)


def _open(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True, decode_times=_cftime)


def _per_variable_norm(nc_path: Path, name: str, levels, is_upper: bool):
    """Normalization std for one variable — a scalar, or one per level."""
    ds = xr.open_dataset(nc_path, decode_times=_cftime)
    if name not in ds:
        raise KeyError(f"{name!r} not in {nc_path}")
    arr = ds[name]
    if not is_upper:
        return float(np.asarray(arr).reshape(-1)[0])
    dim = next((d for d in arr.dims if "level" in d), None)
    if dim is None:
        raise KeyError(f"{name!r} in {nc_path} has no level dim; dims={arr.dims}")
    coord = np.asarray(ds[dim].values, dtype=float)
    idx = [int(np.argmin(np.abs(coord - float(lv)))) for lv in levels]
    picked = np.asarray(arr.isel({dim: idx}).values, dtype=float)
    return picked.reshape(len(levels), -1)[:, 0]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--zarr", required=True,
                   help="directory of per-year {year}.zarr stores")
    p.add_argument("--model-config", required=True,
                   help="conf/model/*.yaml giving the variables, levels, grid "
                        "and model step")
    p.add_argument("--std", required=True, help="normalization std .nc")
    p.add_argument("--year-start", type=int, required=True)
    p.add_argument("--year-end", type=int, required=True, help="exclusive")
    p.add_argument("--out", required=True, help="output .pt")
    p.add_argument("--lmax", type=int, default=None,
                   help="bandwidth; default nlat // 2 (must match the filter's)")
    p.add_argument("--sample-stride", type=int, default=8,
                   help="take every Nth increment — a band-averaged spectrum "
                        "converges in a few thousand pairs")
    p.add_argument("--max-samples", type=int, default=4000,
                   help="stop after this many increment pairs")
    p.add_argument("--floor", type=float, default=1e-3,
                   help="minimum envelope value as a fraction of the mean. A "
                        "band with (near-)zero increment power would give Gamma "
                        "a null direction, which the score Gamma^{-1} zhat then "
                        "blows up; the floor keeps the operator invertible.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    import torch                     # late, so --help works without it
    from torch_harmonics import RealSHT

    cfg = yaml.safe_load(Path(args.model_config).read_text())
    surface = list(cfg.get("surface_variables") or [])
    upper = list(cfg.get("upper_air_variables") or [])
    diagnostic = list(cfg.get("diagnostic_variables") or [])
    levels = [float(v) for v in (cfg.get("levels") or [])]
    nlat, nlon = (int(v) for v in cfg["horizontal_resolution"])
    step_hours = int(cfg.get("timedelta_hours", 24))
    lmax = int(args.lmax) if args.lmax is not None else max(2, nlat // 2)

    root = Path(args.zarr)
    years = list(range(int(args.year_start), int(args.year_end)))
    stores = [root / f"{y}.zarr" for y in years]
    missing = [s.name for s in stores if not s.exists()]
    if missing:
        raise SystemExit(f"missing stores under {root}: {missing}")

    with _open(stores[0]) as probe:
        cadence = int(probe.attrs.get("data_timedelta_hours", 0) or 0)
    if cadence <= 0:
        raise SystemExit(f"{stores[0]} declares no data_timedelta_hours")
    if step_hours % cadence:
        raise SystemExit(
            f"model timedelta_hours={step_hours} is not a multiple of the "
            f"store's {cadence} h rows"
        )
    stride = step_hours // cadence

    sht = RealSHT(nlat=nlat, nlon=nlon, lmax=lmax, grid="equiangular")
    L = int(getattr(sht, "lmax", lmax))
    logger.info(
        "grid %dx%d, lmax %d; model step %d h over %d h rows -> %d row(s); "
        "%d year(s)", nlat, nlon, L, step_hours, cadence, stride, len(years),
    )

    power = np.zeros(L, dtype="float64")     # sum over (m, t, channel)
    count = 0
    n_pairs = 0

    def _accumulate(field: np.ndarray, norm) -> None:
        """field: (n, [L_lev,] H, W) physical increments for one variable."""
        nonlocal power, count
        arr = np.asarray(field, dtype="float32")
        if arr.ndim == 4:                                  # (n, lev, H, W)
            arr = arr / np.asarray(norm, "float32")[None, :, None, None]
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        else:                                              # (n, H, W)
            arr = arr / float(norm)
        if arr.shape[-2:] != (nlat, nlon):
            raise SystemExit(
                f"store grid {arr.shape[-2:]} != model horizontal_resolution "
                f"({nlat}, {nlon}); point --zarr at the matching store"
            )
        coeff = sht(torch.from_numpy(arr))                 # (n, L, M) complex
        # Band power: sum over m, mean over the batch of 2-D fields.
        power += coeff.abs().pow(2).sum(dim=-1).mean(dim=0).double().numpy()
        count += 1

    for store in stores:
        if n_pairs >= args.max_samples:
            break
        with _open(store) as ds:
            n_time = ds.sizes["time"]
            t0 = np.arange(0, n_time - stride, args.sample_stride)
            room = args.max_samples - n_pairs
            t0 = t0[:room]
            if not len(t0):
                logger.warning("%s: too short for a %d-row step, skipped",
                               store.name, stride)
                continue
            for name in surface + diagnostic:
                if name not in ds:
                    raise KeyError(f"{name!r} not in {store}")
                a = ds[name].isel(time=t0).values
                b = ds[name].isel(time=t0 + stride).values
                _accumulate(b - a,
                            _per_variable_norm(Path(args.std), name, levels, False))
            for name in upper:
                if name not in ds:
                    raise KeyError(f"{name!r} not in {store}")
                dim = next(d for d in ds[name].dims if "level" in d)
                coord = np.asarray(ds[dim].values, dtype=float)
                idx = [int(np.argmin(np.abs(coord - lv))) for lv in levels]
                a = ds[name].isel(time=t0, **{dim: idx}).values
                b = ds[name].isel(time=t0 + stride, **{dim: idx}).values
                _accumulate(b - a,
                            _per_variable_norm(Path(args.std), name, levels, True))
            n_pairs += len(t0)
        logger.info("  %s done (%d/%d pairs)", store.name, n_pairs,
                    args.max_samples)

    if not count:
        raise SystemExit("no increments accumulated — check the year range")

    # Amplitude, not power; unit band-mean (the filter renormalizes too, but an
    # artifact that is already normalized is easier to eyeball).
    envelope = np.sqrt(power / count)
    if not np.all(np.isfinite(envelope)) or envelope.max() <= 0:
        raise SystemExit(f"degenerate envelope: {envelope[:8]}")
    envelope = envelope / envelope.mean()
    # Floor before the final renormalization: an empty band makes Gamma
    # singular in that direction, and the score would divide by it.
    n_floored = int((envelope < args.floor).sum())
    if n_floored:
        logger.warning(
            "%d/%d degree(s) below the floor %.1e x mean — clamped. Bands with "
            "no increment power would make Gamma non-invertible there.",
            n_floored, len(envelope), args.floor,
        )
    envelope = np.maximum(envelope, args.floor)
    envelope = (envelope / envelope.mean()).astype("float32")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "envelope_len": int(L),
        "lmax": int(L),
        "grid": [nlat, nlon],
        "model_config": str(args.model_config),
        "years": [int(args.year_start), int(args.year_end)],
        "pairs": int(n_pairs),
        "variable_groups_accumulated": int(count),
        "sample_stride": int(args.sample_stride),
        "floor": float(args.floor),
        "degrees_floored": n_floored,
        "model_step_hours": step_hours,
        "store_cadence_hours": cadence,
        "quantity": "unit-band-mean amplitude spectrum of the normalized "
                    "one-step increment (NOT the conditional spread — see the "
                    "module docstring)",
    }
    torch.save({"envelope": torch.from_numpy(envelope), "meta": meta}, out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s (L=%d, min %.3f, max %.3f)", out, L,
                float(envelope.min()), float(envelope.max()))
    logger.info("sidecar %s", out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
