# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""amip_v2's observed (ERA5) climatology, as a frame-invariant truth.

Upstream ``amip_v2``'s ``bias.py`` never reads per-frame full-resolution truth:
it accumulates the predicted time-mean and subtracts a **precomputed obs
climatology**. That is what makes a 180x360 bias evaluation possible on Polaris,
which holds only the coarse 45x90 store.

The trick that lets our existing scorers do this unchanged: feed the obs
climatology in as a **constant** (frame-invariant) truth. Since

    mean_t(pred) - obs  ==  mean_t(pred - obs)

``ClimatologyScorer``'s ``{kind}_bias`` map becomes exactly
``pred_time_mean - obs_climatology`` and ``compute_headline_bias`` needs no
change at all. The alternative -- threading ``Optional[truth]`` through
``StepContext`` and every scorer -- would also have destroyed the blow-up
detector, since ``scan_rmse_trace`` would see 1827 non-finite steps.

The cost of the trick is a semantic footnote: under constant truth the
``rmse_step*`` block is RMSE-vs-**climatology**, not skill. It is still a
useful stability signal (it saturates at the climatological variance rather
than at the truth), but it must be labelled, which is why callers echo
``truth_source`` into the results.

**Verified properties of the shipped 1996-2001 file** (measured 2026-09-06, and
asserted below rather than assumed):

* latitude is **S->N ascending**, matching the AMIP convention -- row 0 is the
  South Pole (mean t2m 226.8 K), row 179 the North Pole (258.9 K), equator
  298.8 K. No flip is needed. This is the highest-consequence silent failure in
  the whole comparison, so :func:`assert_physical_anchors` checks it every load.
* the level axis is 26 levels of **ascending pressure** (5..1000 hPa):
  geopotential falls 3.465e5 -> 739 m^2/s^2 with index, and equatorial
  temperature rises 242.9 -> 298.6 K.
* units are physical (K, Pa, kg/kg, m/s), values finite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import torch

# amip_v2's on-disk names. Note "multilevel" upstream == "upper_air" here.
OBS_FILES = {
    "surface": "climatology_surface_obs.pt",
    "upper_air": "climatology_multilevel_obs.pt",
    "diagnostic": "climatology_diagnostic_obs.pt",
}
OBS_META = "climatology_obs_meta.json"

# Channel indices within each group, by upstream's variable order.
_T2M = 2          # surface_variables[2] == 2m_temperature
_GEOPOTENTIAL = 3  # upper_air_variables[3] == geopotential
_TEMPERATURE = 0   # upper_air_variables[0] == temperature


class ObsClimatologyError(ValueError):
    """Raised on any misalignment. Never a warning: a silently misaligned
    climatology produces a plausible-looking but wrong bias."""


def assert_physical_anchors(obs: dict[str, torch.Tensor]) -> dict[str, float]:
    r"""Confirm row order and level order from PHYSICS, not from metadata.

    Metadata can be mislabelled; the Antarctic plateau cannot. Returns the
    measured anchors so callers can log them.

    * ``t2m[0].mean() < t2m[-1].mean()`` -- row 0 must be the SOUTH pole
      (annual-mean t2m ~227 K vs ~259 K at the north). A flipped file passes
      every shape check and doubles the apparent bias.
    * geopotential must DECREASE with level index under ascending pressure.
    * equatorial temperature must INCREASE with level index.
    """
    out: dict[str, float] = {}
    if "surface" in obs:
        t2m = obs["surface"][_T2M]
        south, north = float(t2m[0].mean()), float(t2m[-1].mean())
        out.update(t2m_row0=south, t2m_rowN=north,
                   t2m_min=float(t2m.min()), t2m_max=float(t2m.max()))
        if not south < north:
            raise ObsClimatologyError(
                f"latitude row order looks N->S: row 0 mean t2m {south:.1f} K is "
                f"not colder than the last row {north:.1f} K. The AMIP stores are "
                f"S->N, so this file would need flipping -- refusing rather than "
                f"silently doubling the bias."
            )
        if not (180.0 < out["t2m_min"] and out["t2m_max"] < 340.0):
            raise ObsClimatologyError(
                f"t2m outside a physical range [{out['t2m_min']:.1f}, "
                f"{out['t2m_max']:.1f}] K -- wrong units or wrong channel?"
            )
    if "upper_air" in obs:
        ua = obs["upper_air"]
        z_top, z_bot = float(ua[_GEOPOTENTIAL, 0].mean()), float(ua[_GEOPOTENTIAL, -1].mean())
        out.update(z_level0=z_top, z_levelN=z_bot)
        if not z_top > z_bot:
            raise ObsClimatologyError(
                f"level axis looks reversed: geopotential at index 0 "
                f"({z_top:.4g}) is not greater than at the last index "
                f"({z_bot:.4g}); ascending pressure requires it to decrease."
            )
        eq = slice(ua.shape[-2] // 2 - 2, ua.shape[-2] // 2 + 2)
        t_top = float(ua[_TEMPERATURE, 0, eq].mean())
        t_bot = float(ua[_TEMPERATURE, -1, eq].mean())
        out.update(t_eq_level0=t_top, t_eq_levelN=t_bot)
        if not t_bot > t_top:
            raise ObsClimatologyError(
                f"equatorial temperature does not increase with level index "
                f"({t_top:.1f} -> {t_bot:.1f} K); level axis reversed?"
            )
    return out


def load_obs_climatology(
    root: str | Path,
    *,
    catalog=None,
    grid: Optional[tuple[int, int]] = None,
    levels: Optional[Sequence[float]] = None,
    device="cpu",
    log=None,
) -> dict[str, torch.Tensor]:
    r"""Load and fully validate the obs climatology. Physical units, float32.

    Every check raises :class:`ObsClimatologyError` rather than warning, except
    the averaging SPAN, which warns (upstream's own ``check_obs_clim_meta``
    makes the same split: a differing span shifts the bias slightly, a
    differing channel axis makes it meaningless).
    """
    root = Path(root)
    obs: dict[str, torch.Tensor] = {}
    for group, fname in OBS_FILES.items():
        path = root / fname
        if not path.exists():
            if group == "diagnostic":
                continue          # optional; not every contract has diagnostics
            raise ObsClimatologyError(f"missing {path}")
        t = torch.load(path, map_location="cpu", weights_only=True).float()
        if not torch.isfinite(t).all():
            raise ObsClimatologyError(f"{fname} contains non-finite values")
        obs[group] = t

    meta = {}
    mpath = root / OBS_META
    if mpath.exists():
        meta = json.loads(mpath.read_text())

    # Grid.
    if grid is not None:
        for group, t in obs.items():
            if tuple(t.shape[-2:]) != tuple(grid):
                raise ObsClimatologyError(
                    f"{group} grid {tuple(t.shape[-2:])} != expected {tuple(grid)}"
                )
    # Channels, by NAME where the catalog gives them.
    if catalog is not None:
        for group, names in (("surface", getattr(catalog, "surface", None)),
                             ("upper_air", getattr(catalog, "upper_air", None)),
                             ("diagnostic", getattr(catalog, "diagnostic", None))):
            if group not in obs or not names:
                continue
            if obs[group].shape[0] != len(names):
                raise ObsClimatologyError(
                    f"{group}: obs has {obs[group].shape[0]} channels, the model "
                    f"catalog has {len(names)} -- the channel axes would not line "
                    f"up, so the bias would be silently wrong"
                )
            mnames = meta.get(f"{group}_variables") or meta.get("upper_air_variables"
                                                                if group == "upper_air" else "")
            if isinstance(mnames, list) and list(mnames) != list(names):
                raise ObsClimatologyError(
                    f"{group} variable ORDER differs:\n  obs   {list(mnames)}\n"
                    f"  model {list(names)}"
                )
    # Levels, matched BY VALUE.
    if levels is not None and "upper_air" in obs:
        n_obs = obs["upper_air"].shape[1]
        if n_obs != len(levels):
            raise ObsClimatologyError(
                f"upper_air has {n_obs} levels, model expects {len(levels)}"
            )
        mlev = meta.get("levels")
        if isinstance(mlev, list) and [float(x) for x in mlev] != [float(x) for x in levels]:
            raise ObsClimatologyError(
                f"level VALUES differ:\n  obs   {mlev}\n  model {list(levels)}"
            )

    anchors = assert_physical_anchors(obs)
    if log is not None:
        log.info(
            f"obs climatology {root.name}: "
            + ", ".join(f"{k}={tuple(v.shape)}" for k, v in obs.items())
            + f" | span {meta.get('start_date')}..{meta.get('end_date')} "
            f"n_frames={meta.get('n_frames')} complete={meta.get('complete')}"
        )
        log.info(
            f"obs anchors: t2m row0={anchors.get('t2m_row0', float('nan')):.1f}K "
            f"rowN={anchors.get('t2m_rowN', float('nan')):.1f}K (S->N confirmed), "
            f"z level0={anchors.get('z_level0', float('nan')):.4g} -> "
            f"levelN={anchors.get('z_levelN', float('nan')):.4g}"
        )
    return {k: v.to(device) for k, v in obs.items()}, meta, anchors


def normalize_like(normalizer, group: str, phys: torch.Tensor) -> torch.Tensor:
    r"""Exact inverse of ``ClimateNormalizer.denormalize_state`` for one group.

    ``(phys - mean) / std`` off the normalizer's own registered buffers, so the
    round trip is exact to fp32 rounding (~3e-5 K at 288 K). The diagnostic
    group is a passthrough when ``normalize_diagnostic=False``, mirroring
    ``denormalize_state``'s idempotent branch -- otherwise the constant truth
    would be z-scored while the predictions were not.
    """
    key = {"surface": "surface", "upper_air": "upper_air", "diagnostic": "diagnostic"}[group]
    mean = getattr(normalizer, f"{key}_mean", None)
    std = getattr(normalizer, f"{key}_std", None)
    if mean is None or std is None:
        return phys        # never z-scored (diagnostic passthrough)
    mean = mean.to(phys.device, phys.dtype)
    std = std.to(phys.device, phys.dtype)
    while mean.dim() < phys.dim():
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    return (phys - mean) / std


def as_constant_truth(
    obs: dict[str, torch.Tensor],
    *,
    normalizer,
    batch_size: int = 1,
    device=None,
) -> dict[str, torch.Tensor]:
    r"""-> the normalized, batch-expanded dict the drive scores against.

    Keys are the drive's own state names (``surface_in`` / ``upper_air_in`` /
    ``diagnostic``). Expanded, not repeated: one copy of the field is held
    regardless of batch size.
    """
    name = {"surface": "surface_in", "upper_air": "upper_air_in",
            "diagnostic": "diagnostic"}
    out: dict[str, torch.Tensor] = {}
    for group, t in obs.items():
        z = normalize_like(normalizer, group, t)
        if device is not None:
            z = z.to(device)
        out[name[group]] = z.unsqueeze(0).expand(batch_size, *z.shape)
    return out


def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Bilinear downsample the trailing (H, W), matching the coarse store's own
    build operator (``F.interpolate(bilinear, align_corners=False)``)."""
    import torch.nn.functional as F

    if factor == 1:
        return x
    lead, (h, w) = x.shape[:-2], x.shape[-2:]
    flat = x.reshape(1, -1, h, w)
    out = F.interpolate(
        flat, size=(h // factor, w // factor), mode="bilinear",
        align_corners=False,
    )
    return out.reshape(*lead, h // factor, w // factor)


def verify_against_store(
    obs: dict[str, torch.Tensor],
    *,
    dataset,
    normalizer,
    rows: Sequence[int],
    downsample_factor: int = 1,
    log=None,
    rel_tol: float = 0.05,
) -> dict:
    r"""THE alignment gate. Run this before any expensive rollout.

    Compares the obs climatology against the STORE's own time-mean over the
    same rows, after downsampling the obs to the store's grid.

    Why this is decisive rather than merely suggestive: bilinear downsampling
    and time-averaging are both linear, so they commute exactly, and the obs
    climatology's source archive is the same ``ERA5 h5_dailyavg`` the
    ``amip_dailyavg`` store was converted from. Agreement should therefore sit
    at float rounding. Every misalignment produces a discrepancy orders of
    magnitude larger, and each has its own signature:

    * **lat flip** -- max|diff| ~50 K in t2m, AND the flipped comparison agrees
      instead. Reported explicitly as ``flip_is_better``.
    * **level-axis reversal** -- z500 lands at the wrong level, off by ~1e5
      m^2/s^2. (:func:`assert_physical_anchors` already catches this at load.)
    * **channel permutation** -- one channel wildly off, its neighbours fine.
    * **unit mismatch** -- an exact 273.15 or power-of-ten offset.
    * **wrong averaging span** -- a small ~0.1-0.3 K residual, which is how you
      tell it apart from all of the above.

    ``rows`` should be a strided sample (every 10th row is plenty); the mean of
    a strided sample is an unbiased estimate of the full mean.

    Raises when the flipped comparison beats the aligned one -- that is a
    definite flip, and continuing would silently double every bias.
    """
    groups = {"surface": "surface_in", "upper_air": "upper_air_in",
              "diagnostic": "diagnostic"}
    sums: dict[str, torch.Tensor] = {}
    n = 0
    for t in rows:
        try:
            sample = dataset[(int(t), 1)]
        except (TypeError, KeyError):
            sample = dataset[int(t)]
        kw = {g: sample[key].unsqueeze(0) for g, key in groups.items()
              if key in sample}
        phys = normalizer.denormalize_state(**kw)
        for g, v in phys.items():
            v = v[0].double()
            sums[g] = v if g not in sums else sums[g] + v
        n += 1
    if n == 0:
        raise ObsClimatologyError("no rows sampled")
    store_mean = {g: v / n for g, v in sums.items()}

    report: dict = {"n_rows": n, "downsample_factor": int(downsample_factor),
                    "groups": {}}
    flips: list[str] = []
    for g, ref in store_mean.items():
        if g not in obs:
            continue
        o = _downsample(obs[g].double(), int(downsample_factor))
        if o.shape != ref.shape:
            raise ObsClimatologyError(
                f"{g}: obs downsampled to {tuple(o.shape)} but the store's "
                f"time-mean is {tuple(ref.shape)}"
            )
        d = (o - ref).abs()
        d_flip = (o.flip(-2) - ref).abs()
        # PER-CHANNEL relative residual, then the worst. A group-wide scale
        # would let a large-magnitude channel hide a broken small one: surface
        # holds both surface_pressure (~1e5 Pa) and t2m (~288 K), so a full
        # 273.15 K unit error in t2m came out as a 1.6% group-relative residual
        # and slid under a 5% tolerance.
        per_ch = d.reshape(d.shape[0], -1).mean(dim=1)
        # Floor the per-channel scale at a fraction of the group's own scale.
        # A relative residual is undefined for a channel whose reference is
        # ~0, and without a floor such a channel divides by ~0 and dominates
        # the verdict on numerical noise alone.
        ch_scale = ref.reshape(ref.shape[0], -1).abs().mean(dim=1)
        ch_scale = ch_scale.clamp_min(
            1e-3 * ref.abs().mean().clamp_min(1e-12)
        )
        per_ch_rel = per_ch / ch_scale
        entry = {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "rel_mean_abs": float(per_ch_rel.max()),
            "rel_per_channel": [float(v) for v in per_ch_rel],
            "mean_abs_if_flipped": float(d_flip.mean()),
            "flip_is_better": bool(d_flip.mean() < d.mean()),
        }
        # Which channel is worst -- a permutation shows up as exactly one.
        entry["worst_channel"] = int(per_ch_rel.argmax())
        entry["worst_channel_mean_abs"] = float(per_ch[per_ch_rel.argmax()])
        report["groups"][g] = entry
        if entry["flip_is_better"]:
            flips.append(g)
        if log is not None:
            log.info(
                f"align {g}: mean|diff|={entry['mean_abs']:.4g} "
                f"(rel {entry['rel_mean_abs']:.2e}), max={entry['max_abs']:.4g}, "
                f"flipped would be {entry['mean_abs_if_flipped']:.4g}, "
                f"worst ch {entry['worst_channel']}"
            )
    if flips:
        raise ObsClimatologyError(
            f"LATITUDE FLIP detected for {flips}: the row-reversed comparison "
            f"agrees better than the aligned one. Continuing would silently "
            f"double every bias. Report: {report['groups']}"
        )
    bad = {g: e["rel_mean_abs"] for g, e in report["groups"].items()
           if e["rel_mean_abs"] > rel_tol}
    report["within_tolerance"] = not bad
    report["rel_tol"] = rel_tol
    if bad and log is not None:
        log.warning(
            f"alignment residual above {rel_tol:.0%} for {bad} — not a flip "
            f"(that is checked separately), so suspect a differing averaging "
            f"span, a channel permutation or a unit mismatch"
        )
    return report
