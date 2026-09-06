# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Pre-run gates for the amip_v2-style bias + trend campaigns.

Every check here is cheap and every one of them guards a failure that would
otherwise produce a plausible-looking but wrong number after hours of GPU
time. Run this before submitting either campaign; it needs no GPU.

Gates:

1. **Obs-climatology alignment** -- the highest-consequence one. Compares the
   obs climatology against the store's own time-mean over the same rows and
   refuses a latitude flip, localizes a channel permutation, and separates a
   unit error from a differing averaging span. See
   ``obs_climatology.verify_against_store``.
2. **Physical anchors** -- row order and level order from physics, not
   metadata (checked on every load by ``load_obs_climatology``).
3. **Normalize round trip** -- ``normalize_like`` then ``denormalize_state``
   must return the obs climatology to fp32 rounding, since the constant truth
   is fed in normalized while the headline is quoted in physical units.
4. **Row budget** -- the store must actually hold the rows the horizon needs.
   Campaign A reaches ``step_size * (W + horizon - 2 + nocean)`` past the IC;
   at a 6-hourly store with a 24 h model step ``step_size`` is 4, NOT 1, so a
   1827-frame run needs ~7.3k rows and a 12,783-frame one ~51k.
5. **IC resolution** -- the requested calendar date must resolve to exactly
   one row, logged with its decoded timestamp.

Usage (same overrides as the campaign, plus the two below)::

    python preflight_campaigns.py --config-dir=conf --config-name=config \
        model=amip_rsi_sst_pred loss=rsi loss.window_size=6 \
        dataset=amip_dailyavg_coarse_multiyear ... \
        ++preflight.obs_dir=$AI_ROSSBY_DATA/norm_stats/obs_climatology_1996_2001 \
        ++preflight.ic_date=1996-01-01 ++preflight.horizon=1827
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("preflight")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from inference import resolve_init_schedule, _full_time_coord
    from obs_climatology import (
        load_obs_climatology,
        normalize_like,
        verify_against_store,
    )
    from train import _resolve_path
    from train_diffusion import _build_dataset
    from train_loop import model_step_rows
    from physicsnemo.experimental.datapipes.climate import ClimateNormalizer

    pf = cfg.get("preflight", {})
    obs_dir = pf.get("obs_dir", None)
    ic_date = str(pf.get("ic_date", "1996-01-01"))
    horizon = int(pf.get("horizon", 1827))
    stride = int(pf.get("stride", 10))
    window = int(cfg.loss.get("window_size", 6) or 6)
    nocean = int(pf.get("nocean", 1))

    failures: list[str] = []

    log.info("=" * 72)
    log.info("building the eval dataset")
    ds = _build_dataset(cfg)
    step_rows = model_step_rows(cfg, ds)
    log.info(f"store rows={ds.n_time}  step_size={step_rows} rows/model step")
    if step_rows != 1:
        log.info(
            f"  (a {24 // step_rows if step_rows else 0}-hourly store under a "
            f"24 h model step -- step_size is NOT 1)"
        )

    normalizer = ClimateNormalizer.from_dataset(
        ds,
        mean_path=_resolve_path(cfg.dataset.mean_path),
        std_path=_resolve_path(cfg.dataset.std_path),
        normalize_constant_boundary=bool(
            cfg.dataset.get("normalize_constant_boundary", False)
        ),
        constant_stats=str(cfg.dataset.get("constant_boundary_stats", "file")),
        normalize_diagnostic=bool(cfg.dataset.get("normalize_diagnostic", False)),
    )

    # ---- Gate 5: IC resolution -------------------------------------------
    log.info("-" * 72)
    log.info(f"GATE ic: resolving {ic_date}")
    try:
        times = _full_time_coord(ds)
        y, m, d = (int(x) for x in ic_date.split("-"))
        # hours=[0] is REQUIRED on a 6-hourly store: without it a calendar
        # date legitimately resolves to 4 rows (00/06/12/18Z) and the "exactly
        # one" assertion below fires on healthy data. 00Z is also amip_v2's own
        # convention (its obs meta records start_date 1996-01-01 00:00:00).
        ic_hour = int(pf.get("ic_hour", 0))
        ics = resolve_init_schedule(
            times, months=[m], days=[d], years=[y], hours=[ic_hour]
        )
        if len(ics) != 1:
            failures.append(
                f"ic_date {ic_date} {ic_hour:02d}Z resolved to {len(ics)} rows "
                f"(expected exactly 1)"
            )
        else:
            log.info(f"  row {ics[0]} = {times[ics[0]]}")
    except Exception as exc:                       # noqa: BLE001
        failures.append(f"ic resolution failed: {exc}")
        ics = [0]

    # ---- Gate 4: row budget ----------------------------------------------
    log.info("-" * 72)
    reach = step_rows * (window + horizon - 2 + nocean)
    need = ics[0] + reach
    log.info(
        f"GATE rows: horizon {horizon} reaches {reach} rows past the IC "
        f"(row {need} of {ds.n_time})"
    )
    if need >= ds.n_time:
        failures.append(
            f"row budget: need row {need} but the store has {ds.n_time}"
        )
    else:
        log.info("  OK")

    if not obs_dir:
        log.info("-" * 72)
        log.info("no preflight.obs_dir given -- skipping the obs gates")
    else:
        # ---- Gates 1-3 ---------------------------------------------------
        log.info("-" * 72)
        log.info(f"GATE obs: loading {obs_dir}")
        levels = list(cfg.model.get("levels", []) or []) or None
        try:
            obs, meta, anchors = load_obs_climatology(
                _resolve_path(str(obs_dir)), levels=levels, log=log
            )
        except Exception as exc:                   # noqa: BLE001
            failures.append(f"obs load/anchors failed: {exc}")
            obs = None

        if obs is not None:
            log.info("-" * 72)
            log.info("GATE round-trip: normalize_like -> denormalize_state")
            for g, t in obs.items():
                z = normalize_like(normalizer, g, t)
                back = normalizer.denormalize_state(**{g: z.unsqueeze(0)})[g][0]
                rel = float((back - t).abs().max() / t.abs().max().clamp_min(1e-12))
                log.info(f"  {g}: max rel error {rel:.2e}")
                if rel > 1e-4:
                    failures.append(f"round trip {g}: rel {rel:.2e} > 1e-4")

            log.info("-" * 72)
            probe = ds[0] if not isinstance(ds[0], tuple) else ds[(0, 1)]
            store_h = probe["surface_in"].shape[-2]
            factor = obs["surface"].shape[-2] // store_h
            rows = list(range(ics[0], min(ics[0] + horizon * step_rows,
                                          ds.n_time), stride * step_rows))
            log.info(
                f"GATE align: obs {obs['surface'].shape[-2:]} vs store "
                f"{tuple(probe['surface_in'].shape[-2:])} (factor {factor}), "
                f"{len(rows)} sampled rows"
            )
            try:
                rep = verify_against_store(
                    obs, dataset=ds, normalizer=normalizer, rows=rows,
                    downsample_factor=factor, log=log,
                )
                if not rep["within_tolerance"]:
                    failures.append(
                        f"alignment residual above tolerance: "
                        f"{ {g: e['rel_mean_abs'] for g, e in rep['groups'].items()} }"
                    )
            except Exception as exc:               # noqa: BLE001
                failures.append(f"alignment gate failed: {exc}")

    # ---- Gate 6: weights provenance --------------------------------------
    ckpt = pf.get("checkpoint", None)
    if ckpt:
        log.info("-" * 72)
        log.info(f"GATE weights: {ckpt}")
        cpath = Path(_resolve_path(str(ckpt)))
        if not cpath.exists():
            failures.append(f"checkpoint missing: {cpath}")
        else:
            log.info(f"  mdlus present, {cpath.stat().st_size / 2**30:.2f} GiB")
            # save_checkpoint writes the LIVE weights here and the EMA shadow
            # into metadata["ema"] of the sibling .pt, while the trainer
            # validates EMA-applied. Report which exist so use_ema is a
            # decision rather than a default.
            stem = cpath.name.split(".")
            idx = stem[-2] if len(stem) >= 3 else None
            sib = cpath.parent / f"checkpoint.0.{idx}.pt"
            if not sib.exists():
                log.warning(
                    f"  no sibling {sib.name}: EMA weights are NOT available, "
                    f"so use_ema would silently fall back to raw"
                )
            else:
                blob = torch.load(sib, map_location="cpu", weights_only=False)
                meta = (blob.get("metadata") or {}) if isinstance(blob, dict) else {}
                ema = meta.get("ema")
                if ema is None:
                    log.warning(f"  {sib.name} has no metadata['ema']")
                else:
                    avg = ema.get("avg_model_state", {}) if isinstance(ema, dict) else {}
                    n_avg = avg.get("n_averaged")
                    log.info(
                        f"  EMA present: decay={ema.get('decay')} "
                        f"warmup_epochs={ema.get('warmup_epochs')} "
                        f"n_averaged={int(n_avg) if n_avg is not None else '?'}"
                    )
                    log.info(
                        "  -> use_ema=true matches our training-time validation "
                        "numbers; use_ema=false matches upstream amip_v2, which "
                        "does no EMA swap at inference. Record which you quote."
                    )

    log.info("=" * 72)
    if failures:
        log.error(f"PREFLIGHT FAILED ({len(failures)}):")
        for f in failures:
            log.error(f"  - {f}")
        raise SystemExit(1)
    log.info("PREFLIGHT PASSED -- safe to submit")


if __name__ == "__main__":
    main()
