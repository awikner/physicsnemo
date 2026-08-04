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

"""Run a trained MoWE gate and save its forecasts (and optionally its weights).

Loads a checkpoint, replays a split deterministically, and writes a dense
``(init_time, lead_time, lat, lon)`` zarr of the mixture's daily precip in
mm/day -- the artefact downstream analysis needs, which the training-time
validation does not produce (it only reduces to scalar metrics).

Single process on purpose: this is I/O bound and inference is cheap, so there is
no DDP to keep in lockstep and no risk of ranks disagreeing about the output
region.

Usage (dataset/region/split all come from the training config, so the forecasts
are guaranteed to line up with the metrics)::

    python tools/infer_mowe.py \\
        dataset=hindcast_derecho \\
        +checkpoint=/glade/derecho/scratch/awikner/mowe_runs/outputs/mowe_cv5_physvar/checkpoints_best \\
        +out=/glade/derecho/scratch/awikner/mowe_forecasts/cv5_physvar.zarr \\
        +split=val +save_weights=true

``+split`` is ``val`` (default) or ``train``. ``+save_gate=true`` additionally
writes the gate's own fields, each ``(init_time, lead_time, expert, lat, lon)``
and so E times the size of the forecast -- hence off by default:

* ``gate_weights`` -- masked-softmax weight per expert, summing to 1 over the
  live experts (exactly 0 for a masked one). These are what to inspect for the
  monsoon-structure question: which expert the gate trusts where and when.
* ``gate_biases`` -- the learned per-expert additive correction, in the MIXING
  space. With ``model.mix_space=physical`` (the default) that is mm/day, so the
  values are directly interpretable; with ``log`` they are offsets in
  standardized log space and are NOT mm/day.

The mixture is ``sum_i weights_i * (expert_precip_i + biases_i)``, so the two
arrays plus the harmonised expert stores reproduce the forecast exactly.

Missing (init, lead) pairs stay NaN, which is how a pair absent from the index
(an IMERG gap, or too few live experts) is represented; do not confuse that with
a zero forecast.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cftime
import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

# Recipe modules are imported by bare name, which only works when the recipe
# root is on sys.path -- true for train.py at the root, not for a tool in
# tools/. Same shim as verify_precip_alignment.py.
_RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from physicsnemo.distributed import DistributedManager  # noqa: E402

from datapipes.factory import build_dataset  # noqa: E402
from datapipes.sampler import MixturePairSampler  # noqa: E402
from losses import denormalize_precip  # noqa: E402
from mowe_precip import MoWEPrecipGate, mix  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

logger = logging.getLogger("mowe_infer")
EPOCH = cftime.DatetimeGregorian(1900, 1, 1)


def _skeleton(
    ds,
    leads: np.ndarray,
    *,
    save_gate: bool,
    mix_space: str,
    attrs: dict,
    ens_size: int = 0,
    save_members: bool = False,
) -> xr.Dataset:
    """All-NaN store laid out on the split's own (init, lead) axes."""
    inits = [cftime.DatetimeGregorian(*k) for k in ds.index.init_keys]
    shape = (len(inits), leads.size, ds.lat.size, ds.lon.size)
    data = {
        "total_precipitation_24hr": (
            ("init_time", "lead_time", "lat", "lon"),
            np.full(shape, np.nan, "float32"),
            {
                "units": "mm/day",
                "description": (
                    "MoWE gate mixture"
                    + (f" (mean of {ens_size} noise-draw members)" if ens_size else "")
                ),
            },
        )
    }
    if ens_size:
        for tag, q in (("q10", 10), ("q50", 50), ("q90", 90)):
            data[f"total_precipitation_24hr_{tag}"] = (
                ("init_time", "lead_time", "lat", "lon"),
                np.full(shape, np.nan, "float32"),
                {
                    "units": "mm/day",
                    "description": f"member {q}th percentile of the "
                    f"{ens_size}-member gate ensemble",
                },
            )
        if save_members:
            mshape = (shape[0], shape[1], ens_size, *shape[2:])
            data["total_precipitation_24hr_members"] = (
                ("init_time", "lead_time", "member", "lat", "lon"),
                np.full(mshape, np.nan, "float32"),
                {
                    "units": "mm/day",
                    "description": "individual gate ensemble members "
                    "(seeded per (init, lead) pair — reproducible)",
                },
            )
    if save_gate:
        gshape = (shape[0], shape[1], len(ds.experts), *shape[2:])
        data["gate_weights"] = (
            ("init_time", "lead_time", "expert", "lat", "lon"),
            np.full(gshape, np.nan, "float32"),
            {"description": "masked-softmax weight per expert; sums to 1 over "
                            "live experts, exactly 0 for a masked one"},
        )
        data["gate_biases"] = (
            ("init_time", "lead_time", "expert", "lat", "lon"),
            np.full(gshape, np.nan, "float32"),
            {"units": "mm/day" if mix_space == "physical" else "standardized log",
             "description": "learned per-expert additive correction, in the "
                            "mixing space (model.mix_space=" + mix_space + "); "
                            "forecast = sum_i w_i * (P_i + b_i)"},
        )
    coords = {
        "init_time": ("init_time", inits),
        "lead_time": ("lead_time", leads.astype("int32")),
        "lat": ("lat", ds.lat),
        "lon": ("lon", ds.lon),
        "expert": ("expert", list(ds.expert_names)),
    }
    if ens_size and save_members:
        coords["member"] = ("member", np.arange(ens_size, dtype="int32"))
    return xr.Dataset(
        {k: (d, v, a) for k, (d, v, a) in data.items()},
        attrs=attrs,
        coords=coords,
    )


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ckpt = str(cfg.checkpoint)
    out = str(cfg.out)
    split = str(cfg.get("split", "val"))
    save_gate = bool(cfg.get("save_gate", cfg.get("save_weights", False)))

    ds = build_dataset(cfg.dataset, split)
    lo, hi = (int(v) for v in cfg.dataset[split].lead_days)
    leads = np.arange(lo, hi + 1)
    mix_space = str(cfg.model.get("mix_space", "physical"))
    logger.info(
        "split=%s pairs=%d inits=%d leads=%d-%d experts=%s mix_space=%s",
        split, len(ds), len(ds.index.init_keys), lo, hi, ds.expert_names, mix_space,
    )

    # load_checkpoint reaches into DistributedManager. Inside an sbatch job it
    # picks the SLURM path and raises (RANK unset, SLURM vars incomplete for a
    # non-srun launch), so declare a single-process world explicitly and let it
    # take the ENV path. This is inference: there is nothing to distribute.
    for k, v in (
        ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "localhost"), ("MASTER_PORT", str(29500 + os.getpid() % 1000)),
    ):
        os.environ.setdefault(k, v)
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dev = DistributedManager().device
    model = MoWEPrecipGate(
        input_size=(ds.lat.size, ds.lon.size),
        in_channels=ds.layout.num_channels,
        n_experts=len(ds.experts),
        **OmegaConf.to_container(cfg.model.params, resolve=True),
    ).to(dev)
    epoch = load_checkpoint(ckpt, models=model, device=dev)
    logger.info("loaded %s (epoch %s)", ckpt, epoch)
    model.eval()

    # Noise-conditioned gates replay a seeded ensemble; the written
    # total_precipitation_24hr is then the ENSEMBLE MEAN (downstream tooling
    # unchanged) plus member quantiles. Noise is seeded per (init, lead) pair,
    # so outputs are reproducible regardless of batch size or order.
    noise_dim = getattr(model, "noise_dim", None)
    ens_size = int(cfg.get("ens_size", 16)) if noise_dim else 0
    noise_seed = int(cfg.get("noise_seed", 0))
    save_members = bool(cfg.get("save_members", False)) and bool(ens_size)
    if noise_dim:
        logger.info(
            "ensemble inference: %d members (noise_dim %d, seed %d)",
            ens_size, noise_dim, noise_seed,
        )

    # The gate emits fields at all 64,800 gridpoints but is supervised only
    # inside the training region, so record that in the store: outside it the
    # weights and especially the BIASES are unconstrained extrapolation
    # (measured mean -15 mm/day, 1st pct -82, versus -0.5 inside).
    box = list(cfg.region.lat) + list(cfg.region.lon)
    attrs = {
        "checkpoint": ckpt,
        "split": split,
        "mix_space": mix_space,
        "supervised_region_box": f"lat {box[0]}..{box[1]}, lon {box[2]}..{box[3]}",
        "supervised_region_note": (
            "intersected with the IMD gauge mask (dataset.imd.store). The gate "
            "is trained ONLY there; weights and biases elsewhere are untrained "
            "extrapolation and must not be interpreted."
        ),
        "generator": "examples/weather/ai_rossbypalooza/tools/infer_mowe.py",
    }
    if ens_size:
        attrs["ens_size"] = int(ens_size)
        attrs["noise_seed"] = int(noise_seed)
        attrs["ensemble_note"] = (
            "total_precipitation_24hr is the ensemble MEAN over the seeded "
            "noise-draw members; gate_weights/gate_biases are member means"
        )
    skel = _skeleton(
        ds,
        leads,
        save_gate=save_gate,
        mix_space=mix_space,
        attrs=attrs,
        ens_size=ens_size,
        save_members=save_members,
    )
    # Written eagerly rather than with compute=False, which needs dask; the
    # all-NaN chunks compress to almost nothing so the skeleton is cheap.
    skel.to_zarr(out, mode="w", zarr_format=3, consolidated=True)
    lead_row = {int(t): i for i, t in enumerate(leads)}

    loader = DataLoader(
        ds,
        batch_size=int(cfg.dataset.loader.get("batch_size", 4)),
        sampler=MixturePairSampler(len(ds), shuffle=False),
        num_workers=int(cfg.dataset.loader.get("num_workers", 4)),
        pin_memory=False,
    )
    written = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["expert_inputs"].to(dev)
            mask = batch["expert_mask"].to(dev)
            taus = batch["lead_days"].to(dev)
            if ens_size:
                noise = torch.stack(
                    [
                        torch.randn(
                            ens_size,
                            int(noise_dim),
                            generator=torch.Generator().manual_seed(
                                noise_seed * 1_000_003 + int(pi)
                            ),
                        )
                        for pi in batch["pair_idx"]
                    ]
                ).to(dev)
                weights, biases = model(x, mask, taus, noise)
            else:
                weights, biases = model(x, mask, taus)
            expert_precip = x[:, :, 0]
            if mix_space == "physical":
                expert_precip = denormalize_precip(
                    expert_precip,
                    mean=ds.precip_mean,
                    std=ds.precip_std,
                    transform=ds.precip_transform,
                )
                pred_full = mix(weights, biases, expert_precip, mask=mask).clamp(min=0.0)
            else:
                pred_full = denormalize_precip(
                    mix(weights, biases, expert_precip, mask=mask),
                    mean=ds.precip_mean,
                    std=ds.precip_std,
                    transform=ds.precip_transform,
                )
            if ens_size:
                ens_np = pred_full.float().cpu().numpy()  # (B, N, H, W)
                pred_mm = ens_np.mean(axis=1)
                quants = np.quantile(ens_np, [0.1, 0.5, 0.9], axis=1)
                # Gate fields: member means (attrs note this).
                weights = weights.mean(dim=1)
                biases = biases.mean(dim=1)
            else:
                ens_np = quants = None
                pred_mm = pred_full.float().cpu().numpy()
            w_np = weights.float().cpu().numpy() if save_gate else None
            b_np = biases.float().cpu().numpy() if save_gate else None

            # One region write per sample: the pairs are scattered across the
            # (init, lead) grid, so contiguous slabs are not available.
            for b in range(pred_mm.shape[0]):
                row = ds.index.pairs[int(batch["pair_idx"][b])]
                i = int(row["init_row"])
                j = lead_row.get(int(batch["lead_days"][b]))
                if j is None:
                    continue
                sl = {"init_time": slice(i, i + 1), "lead_time": slice(j, j + 1)}
                grid = ("init_time", "lead_time", "lat", "lon")
                piece = {
                    "total_precipitation_24hr": (grid, pred_mm[b][None, None])
                }
                if ens_size:
                    for qi, tag in enumerate(("q10", "q50", "q90")):
                        piece[f"total_precipitation_24hr_{tag}"] = (
                            grid,
                            quants[qi, b][None, None],
                        )
                    if save_members:
                        piece["total_precipitation_24hr_members"] = (
                            ("init_time", "lead_time", "member", "lat", "lon"),
                            ens_np[b][None, None],
                        )
                if save_gate:
                    dims = ("init_time", "lead_time", "expert", "lat", "lon")
                    piece["gate_weights"] = (dims, w_np[b][None, None])
                    piece["gate_biases"] = (dims, b_np[b][None, None])
                xr.Dataset(piece).to_zarr(out, region=sl)
                written += 1
            if written and written % 500 < pred_mm.shape[0]:
                logger.info("wrote %d/%d pairs", written, len(ds))

    logger.info("wrote %d pairs to %s", written, out)


if __name__ == "__main__":
    main()
