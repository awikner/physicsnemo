# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""Per-pixel time-mean climatology for the rollout validator's ACC.

``DiffusionRolloutValidator`` builds no ACC accumulators unless it is handed a
climatology, so ``metrics.acc: True`` was silently inert until 2026-09-04.
This produces the tensors it wants.

**The space matters, and it is not the obvious one.** The validator scores
RMSE on DENORMALIZED tensors but ACC on the normalized ones
(``acc.update(ctx.pred_mean, ctx.truth, ...)`` vs
``rmse.update(ctx.pred_phys, ctx.truth_phys, ...)``), so the climatology must
be **normalized**, matching the dataset transform's output.

It must also be a **per-pixel** field, not a per-channel scalar. Normalization
subtracts one scalar per channel, so a normalized field still carries the full
spatial structure of its time-mean; a scalar climatology would leave the mean
state inside the "anomalies" and inflate ACC — the classic ACC mistake.

Rather than convert upstream amip_v2's obs climatology
(``obs_climatology_1996_2001/climatology_*_obs.pt`` on Midway3 + Stampede3),
this recomputes from the same store the run trains on. That file is physical
units on a 180x360 grid with 26 levels, while our coarse runs are 45x90 on a
level subset, so using it would mean coarsening + level subsetting + channel
reordering + normalizing — four chances to misalign silently, and latitude
row order is NOT uniform across these archives (see CLAUDE.md). Going through
``_build_dataset`` makes the output correct by construction: identical store,
identical normalizer, identical transform order.

Usage (same overrides as the training run, plus the two below)::

    python make_val_climatology.py \
        --config-dir=conf --config-name=config \
        model=amip_rsi_sst_pred loss=rsi loss.window_size=6 \
        dataset=amip_dailyavg_coarse_multiyear \
        ++dataset.zarr_path=... ++dataset.mean_path=... (etc.) \
        ++clim.out_dir=$AI_ROSSBY_DATA/norm_stats/val_clim_coarse \
        ++clim.stride=5

``stride`` subsamples in time (5 = every 5th day). The mean of a strided
sample is an unbiased estimate of the full mean, and at 45x90 the sampling
error is far below the ACC signal; stride 1 is available if wanted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_diffusion import _build_dataset  # noqa: E402

GROUPS = ("surface_in", "upper_air_in", "diagnostic")
# The validator's ctor names; surface_in -> surface, upper_air_in -> upper_air.
OUT_NAME = {"surface_in": "surface", "upper_air_in": "upper_air",
            "diagnostic": "diagnostic"}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    clim_cfg = cfg.get("clim", {})
    out_dir = Path(str(clim_cfg.get("out_dir", "./val_clim")))
    stride = int(clim_cfg.get("stride", 5))
    if stride < 1:
        raise ValueError(f"clim.stride must be >= 1, got {stride}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _build_dataset(cfg)
    n_time = int(ds.n_time)
    times = list(range(0, n_time, stride))
    print(f"store rows: {n_time}, stride {stride} -> {len(times)} frames", flush=True)

    # float64 accumulation: 12k frames of float32 partial sums lose low-order
    # bits, and this runs once so the cost is irrelevant.
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for i, t in enumerate(times):
        try:
            sample = ds[(int(t), 1)]
        except (TypeError, KeyError):
            sample = ds[int(t)]
        for g in GROUPS:
            if g not in sample:
                continue
            v = sample[g].detach().to(torch.float64)
            if not torch.isfinite(v).all():
                raise ValueError(
                    f"non-finite values in {g} at row {t} — the climatology "
                    f"would be poisoned silently; fix the store or the fill"
                )
            sums[g] = v if g not in sums else sums[g] + v
        count += 1
        if i % 500 == 0:
            print(f"  {i}/{len(times)} (row {t})", flush=True)

    if count == 0:
        raise ValueError("no frames accumulated")

    meta = {"n_frames": count, "stride": stride, "n_time": n_time,
            "zarr_path": str(cfg.dataset.zarr_path),
            "mean_path": str(cfg.dataset.mean_path),
            "space": "normalized (dataset transform output) — matches what "
                     "the validator's ACC compares",
            "groups": {}}
    for g, tot in sums.items():
        clim = (tot / count).to(torch.float32)
        # Frames come out with no batch axis; ACC broadcasts (C, [L,] H, W)
        # against (B, C, [L,] H, W), so store exactly what _fetch returned.
        path = out_dir / f"climatology_{OUT_NAME[g]}.pt"
        torch.save(clim, path)
        meta["groups"][OUT_NAME[g]] = {"shape": list(clim.shape),
                                       "mean": float(clim.mean()),
                                       "std": float(clim.std())}
        print(f"wrote {path}  shape={tuple(clim.shape)} "
              f"mean={clim.mean():.4f} std={clim.std():.4f}", flush=True)
    (out_dir / "climatology_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / 'climatology_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
