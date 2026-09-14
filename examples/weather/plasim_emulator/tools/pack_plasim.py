# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pack raw PlaSim sim52 output into per-year HDF5 for PhysicsNeMo training.

Writes the three-dataset layout (state / forcing / diagnostic), each as a
``fields`` dataset of shape ``[time, channel, lat, lon]`` - the layout
``physicsnemo/datapipes/climate/era5_hdf5.py`` expects.

Source facts established by inspection of the sim52 tree (see README):
  * ``sigma_data/<year>_gaussian.nc`` holds ta/ua/va/hus on 10 sigma levels, zg on
    13 pressure levels, plus pl/tas/pr_6h/evap/lsm/sg/z0 - one file covers exactly
    calendar year <year>, 6-hourly, Jan 1 00:00 -> Dec 31 18:00.
  * That NetCDF exists for years 7-107 only.  Years 108-111 fall back to
    ``h5/sigma_data/<year>_<idx>.h5``, one file per timestep.
  * sst/rsdt are prescribed REPEATING annual cycles (verified bit-identical across
    years), read from ``boundary_data/``.  Leap and non-leap cycles are separately
    stretched - you cannot derive one from the other by dropping Feb 29.
  * There is no non-leap ``rsdt`` file, so the non-leap cycle is extracted once
    from the h5 tree and cached (``--build-rsdt-cache``).

Usage
-----
    python pack_plasim.py --build-rsdt-cache --out DIR
    python pack_plasim.py --years 12-111 --split train --out DIR
    python pack_plasim.py --years 50 --split train --out DIR --verify
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from channels import (  # noqa: E402
    DIAGNOSTIC_CHANNELS,
    DIAGNOSTIC_SIGN,
    FORCING_CHANNELS,
    GRID_SHAPE,
    N_SIGMA,
    SIGMA_VARS,
    STATE_CHANNELS,
    ZG_NAMES,
    ZG_PLEV_PA,
)

SIM52 = "/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52"
SIGMA_NC = os.path.join(SIM52, "sigma_data", "{year}_gaussian.nc")
PLEV_NC = os.path.join(SIM52, "plev_data", "{year}_gaussian.nc")
H5_DIR = os.path.join(SIM52, "h5", "sigma_data")
BOUNDARY = os.path.join(SIM52, "boundary_data")

# Years for which the fast per-year NetCDF exists.
NC_YEARS = set(range(7, 108))

SOLAR_CONST = 1367.0  # W m-2, only used to sanity-check cos_zenith against rsdt
OBLIQUITY = np.deg2rad(23.441)

# Calibrated against PlaSim's own rsdt field (see check_solar).  sim52's subsolar
# longitude at t=0 sits at 222.19 deg, not the 180 deg a naive UTC convention gives,
# so the hour angle carries a fixed -42.1875 deg offset; the declination phase is
# likewise shifted 4 days.  With these two constants,
# corr(S0*max(cosz,0), rsdt) = 0.990 over daylight.  Do not "fix" them to round
# numbers without re-running --check-solar.
LON_OFFSET_DEG = -42.1875
DECL_PHASE_DAYS = -4.0
VERNAL_DAY = 79.0
CHUNK = 120  # timesteps per write chunk, keeps peak memory ~1 GB


def is_leap(year: int) -> bool:
    """Proleptic Gregorian leap rule - year 100 is NOT a leap year here, and the
    sim52 files agree (year 100 has 1460 timesteps, year 108 has 1464)."""
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def n_steps(year: int) -> int:
    return 1464 if is_leap(year) else 1460


# ---------------------------------------------------------------- solar geometry


def solar_cos_zenith(year: int, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """cos(solar zenith angle) for every 6-hourly step of `year`.

    Circular-orbit approximation (PlaSim's default setup has negligible
    eccentricity).  Returned UNCLIPPED so night carries "how far below the
    horizon" - that is the signal rsdt throws away by flooring at zero.

    Returns array [T, nlat, nlon].
    """
    T = n_steps(year)
    ndays = 366.0 if is_leap(year) else 365.0
    # Fractional day-of-year, 0-based, at each 6-hourly step.
    doy = np.arange(T, dtype=np.float64) / 4.0

    # Ecliptic longitude measured from the vernal equinox, with sim52's fitted phase.
    lam = 2.0 * np.pi * (doy - VERNAL_DAY + DECL_PHASE_DAYS) / ndays
    decl = np.arcsin(np.sin(OBLIQUITY) * np.sin(lam))  # [T]

    # Hour angle: 0 at local solar noon, with sim52's fitted longitude offset.
    frac_day = (doy % 1.0)  # 0, .25, .5, .75
    lon = np.deg2rad(lon_deg + LON_OFFSET_DEG)[None, :]  # [1, nlon]
    hour_angle = 2.0 * np.pi * frac_day[:, None] + lon - np.pi  # [T, nlon]

    phi = np.deg2rad(lat_deg)[None, :, None]  # [1, nlat, 1]
    d = decl[:, None, None]  # [T, 1, 1]
    cosz = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(
        hour_angle[:, None, :]
    )
    return cosz.astype(np.float32)  # [T, nlat, nlon]


# ---------------------------------------------------------------- source readers


def _fill_masked(a: np.ndarray) -> np.ndarray:
    """Replace masked/NaN entries with the field's own valid mean.

    Only `sst` is masked (over land, ~32% of points). Filling with the valid mean
    keeps the field smooth and finite; `lsm` tells the model where land is, so no
    information is lost by not encoding land with a sentinel.
    """
    if np.ma.isMaskedArray(a):
        m = np.ma.getmaskarray(a)
        a = np.array(a.filled(np.nan), dtype=np.float32)
    else:
        a = np.asarray(a, dtype=np.float32)
        m = ~np.isfinite(a)
    if m.any():
        valid = a[~m]
        a[m] = float(valid.mean()) if valid.size else 0.0
    return a


def read_state_diag_nc(year, t0, t1):
    """Read state[52] and diagnostic[2] from the per-year NetCDF, steps [t0,t1)."""
    import netCDF4 as nc

    g = nc.Dataset(SIGMA_NC.format(year=year))
    nt = t1 - t0
    state = np.empty((nt, len(STATE_CHANNELS)) + GRID_SHAPE, dtype=np.float32)

    state[:, 0] = np.asarray(g.variables["pl"][t0:t1], dtype=np.float32)
    state[:, 1] = np.asarray(g.variables["tas"][t0:t1], dtype=np.float32)

    c = 2
    for v in SIGMA_VARS:  # ta, ua, va, hus -> [t, lev, lat, lon]
        block = np.asarray(g.variables[v][t0:t1], dtype=np.float32)
        state[:, c : c + N_SIGMA] = block
        c += N_SIGMA

    plev = np.asarray(g.variables["plev"][:], dtype=np.float64)
    zg_idx = [int(np.argmin(np.abs(plev - p))) for p in ZG_PLEV_PA]
    zg = np.asarray(g.variables["zg"][t0:t1], dtype=np.float32)  # [t, 13, lat, lon]
    state[:, c : c + len(ZG_NAMES)] = zg[:, zg_idx]
    c += len(ZG_NAMES)
    assert c == len(STATE_CHANNELS), c

    diag = np.empty((nt, len(DIAGNOSTIC_CHANNELS)) + GRID_SHAPE, dtype=np.float32)
    for i, name in enumerate(DIAGNOSTIC_CHANNELS):
        raw = np.asarray(g.variables[name][t0:t1], dtype=np.float32)
        diag[:, i] = DIAGNOSTIC_SIGN[name] * raw  # store as non-negative
    g.close()
    return state, diag


def _h5_key(var, level=None):
    return f"input/{var}" if level is None else f"input/{var}_{level}"


def read_state_diag_h5(year, t0, t1, sigma_levels, plev_keys=None):
    """Reader for years without a per-year sigma NetCDF (108-111, 112-131).

    Sigma-level and surface fields come from the per-timestep h5 tree; `zg` comes
    from `plev_data/<year>_gaussian.nc` instead. That is both faster (one strided
    NetCDF read rather than 10 dataset reads per timestep) and necessary: some
    years' h5 files carry no `zg_*` datasets at all (year 131 has 75 keys where
    year 108 has 86), which would otherwise drop the year silently.
    """
    import netCDF4 as nc

    nt = t1 - t0
    state = np.empty((nt, len(STATE_CHANNELS)) + GRID_SHAPE, dtype=np.float32)
    diag = np.empty((nt, len(DIAGNOSTIC_CHANNELS)) + GRID_SHAPE, dtype=np.float32)

    for j, t in enumerate(range(t0, t1)):
        path = os.path.join(H5_DIR, f"{year}_{t:04d}.h5")
        with h5py.File(path, "r") as f:
            state[j, 0] = f[_h5_key("pl")][:]
            state[j, 1] = f[_h5_key("tas")][:]
            c = 2
            for v in SIGMA_VARS:
                for lv in sigma_levels:
                    state[j, c] = f[_h5_key(v, lv)][:]
                    c += 1
            for i, name in enumerate(DIAGNOSTIC_CHANNELS):
                diag[j, i] = DIAGNOSTIC_SIGN[name] * f[_h5_key(name)][:]

    # zg block, from the pressure-level NetCDF (complete for years 7-132).
    zg_start = 2 + len(SIGMA_VARS) * N_SIGMA
    g = nc.Dataset(PLEV_NC.format(year=year))
    plev = np.asarray(g.variables["plev"][:], dtype=np.float64)
    zg_idx = [int(np.argmin(np.abs(plev - p))) for p in ZG_PLEV_PA]
    zg = np.asarray(g.variables["zg"][t0:t1], dtype=np.float32)
    state[:, zg_start : zg_start + len(ZG_NAMES)] = zg[:, zg_idx]
    g.close()
    return state, diag


def h5_level_keys(year):
    """Discover the exact level suffixes used in the h5 filenames."""
    path = os.path.join(H5_DIR, f"{year}_0000.h5")
    with h5py.File(path, "r") as f:
        keys = list(f["input"].keys())
    sig, pl = [], []
    for k in keys:
        if k.startswith("ta_"):
            sig.append(k[len("ta_") :])
        elif k.startswith("zg_"):
            pl.append(k[len("zg_") :])
    sig.sort(key=float)  # ascending sigma == TOA -> surface, matches NetCDF `lev`
    assert len(sig) == N_SIGMA, (len(sig), sig)
    # zg is deliberately NOT taken from the h5 tree - some years omit it entirely
    # (year 131 has no zg_* datasets). read_state_diag_h5 reads it from plev_data.
    pl = [k for k in sorted(pl, key=float) if float(k) in set(ZG_PLEV_PA)]
    return sig, pl


# ---------------------------------------------------------------- forcing


def load_cyclic_forcing(out_dir, leap):
    """sst and rsdt for one full annual cycle, [T, lat, lon] each."""
    import netCDF4 as nc

    suffix = "_leap" if leap else ""
    sst = nc.Dataset(os.path.join(BOUNDARY, f"sst_masked_6h{suffix}.nc")).variables["sst"]
    sst = _fill_masked(sst[:])

    if leap:
        rsdt = nc.Dataset(
            os.path.join(BOUNDARY, "rsdt_masked_6h_leap.nc")
        ).variables["rsdt"][:]
        rsdt = _fill_masked(rsdt)
    else:
        cache = os.path.join(out_dir, "stats", "rsdt_nonleap.npy")
        if not os.path.exists(cache):
            raise SystemExit(
                f"missing {cache}\nRun with --build-rsdt-cache first (there is no "
                "non-leap rsdt NetCDF; it is extracted from the h5 tree)."
            )
        rsdt = np.load(cache)
    return sst.astype(np.float32), rsdt.astype(np.float32)


def build_rsdt_cache(out_dir):
    """Extract the non-leap rsdt annual cycle from the h5 tree (one-time).

    rsdt is a repeating annual cycle, so any non-leap year gives the same field;
    year 109 is used.  Verified bit-identical to the boundary NetCDF for leap
    year 108, which is why this substitution is sound.
    """
    year, T = 109, 1460
    assert not is_leap(year)
    out = np.empty((T,) + GRID_SHAPE, dtype=np.float32)
    for t in range(T):
        with h5py.File(os.path.join(H5_DIR, f"{year}_{t:04d}.h5"), "r") as f:
            out[t] = f["input/rsdt"][:]
        if t % 200 == 0:
            print(f"  rsdt cache {t}/{T}", flush=True)
    os.makedirs(os.path.join(out_dir, "stats"), exist_ok=True)
    path = os.path.join(out_dir, "stats", "rsdt_nonleap.npy")
    np.save(path, out)
    print(f"wrote {path} {out.shape}")


def build_forcing(year, out_dir, lat, lon, static):
    """Assemble the full [T, 8, lat, lon] forcing block for one year."""
    T = n_steps(year)
    leap = is_leap(year)
    sst, rsdt = load_cyclic_forcing(out_dir, leap)
    assert sst.shape[0] == T, (sst.shape, T)
    assert rsdt.shape[0] == T, (rsdt.shape, T)

    cosz = solar_cos_zenith(year, lat, lon)
    lat_g = np.broadcast_to(lat[:, None], GRID_SHAPE).astype(np.float32)
    lon_g = np.broadcast_to(lon[None, :], GRID_SHAPE).astype(np.float32)

    f = np.empty((T, len(FORCING_CHANNELS)) + GRID_SHAPE, dtype=np.float32)
    for i, name in enumerate(FORCING_CHANNELS):
        if name in static:
            f[:, i] = static[name][None]
        elif name == "sst":
            f[:, i] = sst
        elif name == "rsdt":
            f[:, i] = rsdt
        elif name == "cos_zenith":
            f[:, i] = cosz
        elif name == "lat_grid":
            f[:, i] = lat_g[None]
        elif name == "lon_grid":
            f[:, i] = lon_g[None]
        else:
            raise KeyError(name)
    return f


def load_static_and_grid(year):
    """lsm/sg/z0 (time-invariant) plus the lat/lon axes."""
    import netCDF4 as nc

    src = year if year in NC_YEARS else 50
    g = nc.Dataset(SIGMA_NC.format(year=src))
    lat = np.asarray(g.variables["lat"][:], dtype=np.float32)
    lon = np.asarray(g.variables["lon"][:], dtype=np.float32)
    static = {k: _fill_masked(g.variables[k][0]) for k in ("lsm", "sg", "z0")}
    g.close()
    return static, lat, lon


# ---------------------------------------------------------------- writing


def write_year(year, split, out_dir, verify=False):
    T = n_steps(year)
    static, lat, lon = load_static_and_grid(year)

    paths = {}
    for kind, nch in (
        ("state", len(STATE_CHANNELS)),
        ("forcing", len(FORCING_CHANNELS)),
        ("diagnostic", len(DIAGNOSTIC_CHANNELS)),
    ):
        d = os.path.join(out_dir, kind, split)
        os.makedirs(d, exist_ok=True)
        paths[kind] = (os.path.join(d, f"{year}.h5"), nch)

    forcing = build_forcing(year, out_dir, lat, lon, static)

    use_nc = year in NC_YEARS
    if not use_nc:
        sig_keys, plev_keys = h5_level_keys(year)

    fh = {
        k: h5py.File(p, "w")
        for k, (p, _) in paths.items()
    }
    try:
        ds = {
            k: fh[k].create_dataset(
                "fields",
                shape=(T, nch) + GRID_SHAPE,
                dtype=np.float32,
                chunks=(1, nch) + GRID_SHAPE,
            )
            for k, (_, nch) in paths.items()
        }
        ds["forcing"][...] = forcing

        for t0 in range(0, T, CHUNK):
            t1 = min(t0 + CHUNK, T)
            if use_nc:
                st, dg = read_state_diag_nc(year, t0, t1)
            else:
                st, dg = read_state_diag_h5(year, t0, t1, sig_keys, plev_keys)
            if not np.isfinite(st).all():
                raise ValueError(f"year {year}: non-finite in state [{t0},{t1})")
            if not np.isfinite(dg).all():
                raise ValueError(f"year {year}: non-finite in diagnostic [{t0},{t1})")
            if (dg < 0).any():
                raise ValueError(
                    f"year {year}: negative diagnostic after sign convention "
                    f"[{t0},{t1}) - check DIAGNOSTIC_SIGN"
                )
            ds["state"][t0:t1] = st
            ds["diagnostic"][t0:t1] = dg
        for k in fh:
            fh[k].attrs["year"] = year
            fh[k].attrs["dhours"] = 6
            fh[k].attrs["source"] = "netcdf" if use_nc else "h5"
    finally:
        for f in fh.values():
            f.close()

    print(f"year {year} [{split}] T={T} source={'nc' if use_nc else 'h5'} -> ok", flush=True)
    if verify:
        verify_year(year, out_dir, split)


def verify_year(year, out_dir, split):
    """Round-trip a handful of timesteps against the raw source."""
    import netCDF4 as nc

    if year not in NC_YEARS:
        print(f"  verify: year {year} uses the h5 source; skipping NetCDF round-trip")
        return
    g = nc.Dataset(SIGMA_NC.format(year=year))
    with h5py.File(os.path.join(out_dir, "state", split, f"{year}.h5"), "r") as f:
        packed = f["fields"]
        for t in (0, n_steps(year) // 2, n_steps(year) - 1):
            ref_pl = np.asarray(g.variables["pl"][t], dtype=np.float32)
            ref_tas = np.asarray(g.variables["tas"][t], dtype=np.float32)
            assert np.allclose(packed[t, 0], ref_pl), f"pl mismatch t={t}"
            assert np.allclose(packed[t, 1], ref_tas), f"tas mismatch t={t}"
            ref_ta1 = np.asarray(g.variables["ta"][t, 0], dtype=np.float32)
            assert np.allclose(packed[t, 2], ref_ta1), f"ta1 mismatch t={t}"
            plev = np.asarray(g.variables["plev"][:], dtype=np.float64)
            k = int(np.argmin(np.abs(plev - ZG_PLEV_PA[0])))
            ref_zg = np.asarray(g.variables["zg"][t, k], dtype=np.float32)
            assert np.allclose(packed[t, 42], ref_zg), f"zg200 mismatch t={t}"
    print(f"  verify: year {year} round-trip OK (pl, tas, ta1, zg200)")


def check_solar(year, out_dir):
    """cos_zenith should track rsdt closely where the sun is up."""
    static, lat, lon = load_static_and_grid(year)
    cosz = solar_cos_zenith(year, lat, lon)
    _, rsdt = load_cyclic_forcing(out_dir, is_leap(year))
    day = rsdt > 1.0
    est = SOLAR_CONST * np.clip(cosz, 0, None)
    r = np.corrcoef(est[day].ravel(), rsdt[day].ravel())[0, 1]
    print(f"solar check year {year}: corr(S0*max(cosz,0), rsdt) over daylight = {r:.4f}")
    print(f"  night agreement: frac(rsdt==0 where cosz<0) = "
          f"{(rsdt[cosz < 0] == 0).mean():.4f}")
    return r


def parse_years(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, help="e.g. 12-111 or 50,51")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--build-rsdt-cache", action="store_true")
    ap.add_argument("--check-solar", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.build_rsdt_cache:
        build_rsdt_cache(args.out)
        return
    if args.check_solar:
        check_solar(int(args.years or 50), args.out)
        return

    if not args.years:
        raise SystemExit("--years required")

    meta_dir = os.path.join(args.out, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # Grid axes: needed for latitude-weighted losses and metrics.
    stats_dir = os.path.join(args.out, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    if not os.path.exists(os.path.join(stats_dir, "lat.npy")):
        _, lat, lon = load_static_and_grid(50)
        np.save(os.path.join(stats_dir, "lat.npy"), lat)
        np.save(os.path.join(stats_dir, "lon.npy"), lon)
    with open(os.path.join(meta_dir, "data.json"), "w") as f:
        json.dump(
            {
                "coords": {
                    "channel": STATE_CHANNELS,
                    "forcing_channel": FORCING_CHANNELS,
                    "diagnostic_channel": DIAGNOSTIC_CHANNELS,
                },
                "dhours": 6,
                "grid": {"type": "legendre-gauss", "shape": list(GRID_SHAPE)},
            },
            f,
            indent=1,
        )

    for y in parse_years(args.years):
        write_year(y, args.split, args.out, verify=args.verify)


if __name__ == "__main__":
    main()
