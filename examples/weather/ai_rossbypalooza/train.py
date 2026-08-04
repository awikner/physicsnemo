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

"""Method 0 (vanilla MoWE) training: a DiT gate over frozen weather experts.

Trains only the gate: per sample the dataset provides each expert's daily
precip (channel 0) + dynamical predictors at a sampled lead ``tau``; the
gate emits per-expert weight + bias fields and the mixture
``P_hat = sum_i w_i (P_i + b_i)`` is scored with a regional loss against
IMERG. Validation reports per-lead regional RMSE / bias / SEEPS for the
gate vs each expert and the equal-weight mean.

Run (single GPU)::

    python train.py

Multi-GPU::

    torchrun --standalone --nproc-per-node=4 train.py [overrides]
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from datapipes.factory import build_dataset
from ema import ModelEMA
from datapipes.sampler import MixturePairSampler
from losses import (
    build_loss,
    denormalize_precip,
    gate_smoothness_penalty,
    imd_valid_mask,
    region_weights,
    resolve_fss_thresholds,
)
from mowe_precip import MoWEPrecipGate, expert_dropout, mix
from seeps import SeepsClimatology
from validation import MixtureValidator

logger = logging.getLogger("mowe_train")


def _maybe_init_wandb(cfg: DictConfig, *, dist) -> bool:
    """wandb on EVERY rank so wandb.run exists everywhere for LaunchLogger;
    only rank 0 is online and drives logging. Non-rank-0 uses mode="disabled"
    (no-op run, no wandb-core service): concurrent offline-mode services on one
    node fail port-file startup on Midway3 (wandb 0.28.1, ServicePollForToken).
    Must run BEFORE LaunchLogger.initialize."""
    wb = cfg.get("wandb", None)
    if wb is None or not bool(wb.get("enabled", False)):
        return False
    try:
        from physicsnemo.utils.logging.wandb import initialize_wandb
    except ImportError:
        if dist.rank == 0:
            PythonLogger("mowe_train").warning(
                "wandb.enabled=True but wandb is not importable; console only."
            )
        return False
    _ent = wb.get("entity", None)
    initialize_wandb(
        project=str(wb.get("project", "ai-rossbypalooza")),
        entity=str(_ent) if _ent else None,
        name=str(wb.get("name", cfg.get("run_name", "mowe"))),
        mode=str(wb.get("mode", "offline")) if dist.rank == 0 else "disabled",
        config=OmegaConf.to_container(cfg, resolve=True),
        init_timeout=int(wb.get("init_timeout", 300)),
    )
    return dist.rank == 0


def _ddp_mean_scalars(values: dict, *, dist) -> dict:
    """Single all-reduce mean over stacked scalars (no per-step host sync)."""
    if not (getattr(dist, "distributed", False) and dist.world_size > 1):
        return values
    import torch.distributed as tdist

    keys = list(values.keys())
    vec = torch.stack(
        [
            (
                values[k].detach()
                if torch.is_tensor(values[k])
                else torch.as_tensor(float(values[k]), device=dist.device)
            )
            .to(device=dist.device, dtype=torch.float32)
            .reshape(())
            for k in keys
        ]
    )
    tdist.all_reduce(vec, op=tdist.ReduceOp.SUM)
    vec = vec / dist.world_size
    return {k: vec[i] for i, k in enumerate(keys)}


def _build_loader(dataset, sampler, loader_cfg) -> DataLoader:
    num_workers = int(loader_cfg.get("num_workers", 4))
    kwargs = dict(
        batch_size=int(loader_cfg.get("batch_size", 4)),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            loader_cfg.get("persistent_workers", True)
        )
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def _is_crps_family(cfg_loss) -> bool:
    """True when the loss scores an ensemble (regional_crps, alone or as the
    anchor of a regional_fss composite) — such losses require the
    noise-conditioned gate, and vice versa."""
    name = str(cfg_loss.get("name", "regional_mse"))
    if name == "regional_crps":
        return True
    if name == "regional_fss":
        anchor = cfg_loss.get("anchor", None)
        return anchor is not None and str(anchor.get("name", "")) == "regional_crps"
    return False


def _latest_mdlus(path: Path) -> Path:
    """Newest .mdlus in a checkpoint dir ({Class}.{rank}.{epoch}.mdlus)."""
    if path.is_file():
        return path
    cands = sorted(
        path.glob("*.mdlus"),
        key=lambda p: (
            int(p.stem.split(".")[-1]) if p.stem.split(".")[-1].isdigit() else -1
        ),
    )
    if not cands:
        raise FileNotFoundError(f"training.init_from: no .mdlus files in {path}")
    return cands[-1]


def _build_scheduler(optimizer, *, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        frac = min(1.0, frac)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * frac)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run(cfg: DictConfig) -> None:
    """Training entry point; separated from the hydra wrapper so tests can
    call it with a programmatic config (and repeatedly, for resume)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dist = DistributedManager()
    torch.manual_seed(int(cfg.seed) + dist.rank)

    wandb_active = _maybe_init_wandb(cfg, dist=dist)
    LaunchLogger.initialize(use_wandb=wandb_active)
    plog = PythonLogger("mowe_train")

    # ---------------- data ----------------
    train_ds = build_dataset(cfg.dataset, "train")
    has_val = cfg.dataset.get("val") is not None
    val_ds = build_dataset(cfg.dataset, "val") if has_val else None
    plog.info(
        f"train pairs: {len(train_ds)} | experts: {train_ds.expert_names} | "
        f"channels: {train_ds.channel_names}"
        + (f" | val pairs: {len(val_ds)}" if has_val else "")
    )

    loader_cfg = cfg.dataset.loader
    train_sampler = MixturePairSampler(
        len(train_ds),
        num_samples=loader_cfg.get("num_samples_per_epoch") or None,
        shuffle=bool(loader_cfg.get("shuffle", True)),
        seed=int(cfg.seed),
        rank=dist.rank,
        world_size=dist.world_size,
    )
    train_loader = _build_loader(train_ds, train_sampler, loader_cfg)
    if has_val:
        val_sampler = MixturePairSampler(
            len(val_ds),
            shuffle=False,
            rank=dist.rank,
            world_size=dist.world_size,
        )
        val_loader = _build_loader(val_ds, val_sampler, loader_cfg)

    # ---------------- model ----------------
    h, w = train_ds.lat.size, train_ds.lon.size
    model_kwargs = OmegaConf.to_container(cfg.model.params, resolve=True)
    model = MoWEPrecipGate(
        input_size=(h, w),
        in_channels=train_ds.layout.num_channels,
        n_experts=len(train_ds.experts),
        **model_kwargs,
    ).to(dist.device)
    inner_model = model
    if dist.distributed and dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )

    # FGN-style probabilistic gate: noise_dim is an architecture knob
    # (model.params, plumbed to DiT's condition path), ens_size a loop knob.
    # Fresh noise is drawn per training forward pass; each member is one
    # noise draw through the SAME gate on the SAME inputs.
    noise_dim = model_kwargs.get("noise_dim") or None
    ens_size = int(cfg.training.get("ens_size", 2)) if noise_dim else 0

    # ---------------- loss / optimizer ----------------
    region = list(cfg.region.lat) + list(cfg.region.lon)
    box = (region[0], region[1], region[2], region[3])
    # Optional IMD-coverage restriction: with dataset.imd.store set, the
    # training loss (and the monthly validation metrics) only see the
    # gridpoints where the IMD gauge analysis has data.
    imd_mask = None
    imd_cfg = cfg.dataset.get("imd", None)
    if imd_cfg is not None and imd_cfg.get("store"):
        imd_mask = imd_valid_mask(
            str(imd_cfg.store),
            train_ds.lat,
            train_ds.lon,
            min_finite_frac=float(imd_cfg.get("min_finite_frac", 0.99)),
        )
        plog.info(f"IMD-coverage mask: {int(imd_mask.sum())} gridpoints")
    # Space the mixture is formed in. "physical": experts' precip is
    # inverted to mm/day first, so P_hat = sum_i w_i (P_i + b_i) is an
    # ARITHMETIC mean in mm/day and the loss log-transforms it. "log": mix
    # the standardized log channels directly (a weighted GEOMETRIC mean in
    # mm/day, which is structurally dry) -- kept for ablation.
    mix_space = str(cfg.model.get("mix_space", "physical"))
    if mix_space not in ("physical", "log"):
        raise ValueError(f"model.mix_space must be physical|log, got {mix_space!r}")
    plog.info(f"mixture space: {mix_space}")
    loss_fn = build_loss(
        cfg.loss,
        lat=train_ds.lat,
        lon=train_ds.lon,
        box=box,
        precip_mean=train_ds.precip_mean,
        precip_std=train_ds.precip_std,
        precip_transform=train_ds.precip_transform,
        extra_mask=imd_mask,
        pred_space="physical" if mix_space == "physical" else "normalized",
    ).to(dist.device)

    # The three ensemble knobs must agree; guessing here would train the
    # wrong objective silently (an MSE over members scores them independently,
    # a 1-member CRPS is just the MAE).
    crps_family = _is_crps_family(cfg.loss)
    if crps_family and not noise_dim:
        raise ValueError(
            "a CRPS-family loss trains an ensemble: set model.params.noise_dim "
            "(e.g. 32, model=mowe_precip_ens) and training.ens_size >= 2"
        )
    if noise_dim and not crps_family:
        raise ValueError(
            "model.params.noise_dim is set but the loss is deterministic; "
            "use loss=regional_crps (or regional_fss with a regional_crps "
            "anchor), or unset noise_dim"
        )
    if noise_dim and ens_size < 2:
        raise ValueError(
            f"training.ens_size must be >= 2 with noise_dim set, got {ens_size} "
            "(fair CRPS needs at least two members per step)"
        )

    cfg_train = cfg.training
    optimizer = torch.optim.AdamW(
        inner_model.parameters(),
        lr=float(cfg_train.optimizer.lr),
        betas=tuple(cfg_train.optimizer.get("betas", (0.9, 0.999))),
        weight_decay=float(cfg_train.optimizer.get("weight_decay", 0.05)),
    )
    steps_per_epoch = len(train_loader)
    max_epochs = int(cfg_train.max_epochs)
    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=int(cfg_train.get("warmup_epochs", 1)) * steps_per_epoch,
        total_steps=max_epochs * steps_per_epoch,
        min_lr_ratio=float(cfg_train.get("min_lr_ratio", 0.02)),
    )
    amp = str(cfg_train.get("amp", "none"))
    amp_enabled = amp == "bf16" and dist.device.type == "cuda"
    grad_clip = float(cfg_train.get("grad_clip_norm", 0.0) or 0.0)
    dropout_p = float(cfg_train.get("expert_dropout", 0.0))

    # ---------------- validation harness ----------------
    validator = None
    if has_val and cfg.validation.get("enabled", True):
        seeps_clim = SeepsClimatology(
            to_absolute_path(str(cfg.validation.seeps_climatology))
        )
        val_lead_days = tuple(cfg.dataset.val.lead_days)
        # Score exactly where the loss trains: box (x IMD coverage). The gate
        # emits weights globally but is only supervised in this region, so
        # metrics anywhere else would measure untrained extrapolation.
        val_weights = region_weights(
            val_ds.lat, val_ds.lon, box, extra_mask=imd_mask
        )
        plog.info(
            f"validation region: {int((val_weights > 0).sum())} gridpoints"
            + (" (box n IMD coverage)" if imd_mask is not None else " (box)")
        )
        # Hard-threshold FSS verification metric (all arms). Uses the same
        # threshold machinery as the FSS loss so metric and loss agree.
        fss_cfg = cfg.validation.get("fss", None)
        val_fss_windows = None
        val_fss_maps = None
        val_fss_labels = None
        if fss_cfg is not None:
            val_fss_windows = [int(k) for k in fss_cfg.get("windows", (3, 5))]
            val_fss_maps, val_fss_labels = resolve_fss_thresholds(
                fss_cfg.thresholds, val_ds.lat, val_ds.lon
            )
        validator = MixtureValidator(
            expert_names=val_ds.expert_names,
            lead_days=(int(val_lead_days[0]), int(val_lead_days[1])),
            region_weights=val_weights,
            seeps_climatology=seeps_clim,
            precip_mean=val_ds.precip_mean,
            precip_std=val_ds.precip_std,
            precip_transform=val_ds.precip_transform,
            device=dist.device,
            monthly=True,
            loss_fn=loss_fn,
            mix_space=mix_space,
            ens_size=int(cfg.validation.get("ens_size", 8)) if noise_dim else 0,
            noise_dim=noise_dim,
            noise_seed=int(cfg.validation.get("noise_seed", 0)),
            fss_windows=val_fss_windows,
            fss_threshold_maps=val_fss_maps,
            fss_threshold_labels=val_fss_labels,
            amp_band_windows=[
                int(k) for k in cfg.validation.get("amp_band_windows", (3, 7))
            ],
        )

    # ---------------- warm start ----------------
    # Optional: initialize from a deterministic checkpoint (e.g. a
    # checkpoints_best dir). Placed BEFORE the EMA wrapper so the average
    # starts from the loaded weights. The condition-embedder Linear that the
    # checkpoint lacks is zero-initialized, so a warm-started ensemble gate
    # initially reproduces the deterministic gate identically for every
    # member and grows spread by gradient. Resume beats init_from.
    init_from = cfg_train.get("init_from", None)
    if init_from:
        if any(Path("./checkpoints").glob("*.mdlus")):
            plog.info("training.init_from ignored: resuming from ./checkpoints")
        else:
            src = _latest_mdlus(Path(to_absolute_path(str(init_from))))
            try:
                inner_model.load(str(src), map_location=dist.device, strict=False)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(
                    f"warm start from {src} failed. Checkpoints are NOT "
                    "transferable across patch_size (the learnable pos_embed "
                    "and detokenizer head change shape) — train that arm from "
                    f"scratch. Original error: {e}"
                ) from e
            plog.info(
                f"warm-started from {src} (strict=False; missing "
                "condition-embedder keys stay zero-initialized)"
            )

    ema_cfg = cfg_train.get("ema", None)
    ema = None
    if ema_cfg is not None and bool(ema_cfg.get("enabled", False)):
        ema = ModelEMA(
            inner_model,
            decay=float(ema_cfg.get("decay", 0.999)),
            warmup_epochs=int(ema_cfg.get("warmup_epochs", 2)),
            steps_per_epoch=steps_per_epoch,
        )
        plog.info(
            f"EMA enabled (decay {ema.decay}); validating with "
            f"{'EMA' if bool(ema_cfg.get('validate_with_ema', True)) else 'raw'} weights"
        )
    validate_with_ema = ema is not None and bool(
        ema_cfg.get("validate_with_ema", True)
    )

    # ---------------- resume ----------------
    ckpt_dir = Path("./checkpoints")
    best_dir = Path("./checkpoints_best")
    loaded_epoch = load_checkpoint(
        str(ckpt_dir),
        models=inner_model,
        optimizer=optimizer,
        scheduler=None,
        device=dist.device,
    )
    start_epoch = max(int(cfg.get("start_epoch", 0)), loaded_epoch + 1 if loaded_epoch else 0)
    for _ in range(start_epoch * steps_per_epoch):
        scheduler.step()

    es_cfg = cfg_train.get("early_stopping", None)
    es_enabled = es_cfg is not None and bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 8)) if es_enabled else 0
    es_min_delta = float(es_cfg.get("min_delta", 0.0)) if es_enabled else 0.0
    best_loss = float("inf")
    epochs_since_best = 0

    gate_tv_weight = float(cfg_train.get("gate_tv_weight", 0.0) or 0.0)

    # ---------------- training loop ----------------
    # A resume at/after max_epochs skips the loop entirely; the final save
    # below must still see a defined epoch.
    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)  # FSS-weight ramp (train mode only)
        model.train()
        # LaunchLogger epochs are 1-indexed (iter starts at (epoch-1)*num_mini_batch).
        with LaunchLogger(
            "train", epoch=epoch + 1, num_mini_batch=steps_per_epoch, epoch_alert_freq=1
        ) as log:
            for batch in train_loader:
                x = batch["expert_inputs"].to(dist.device, non_blocking=True)
                mask = batch["expert_mask"].to(dist.device, non_blocking=True)
                target = batch["target"].to(dist.device, non_blocking=True)
                target_mm = batch["target_mm"].to(dist.device, non_blocking=True)
                taus = batch["lead_days"].to(dist.device, non_blocking=True)

                if dropout_p > 0:
                    x, mask = expert_dropout(x, mask, dropout_p)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist.device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    if noise_dim:
                        # Fresh draws per forward pass (FGN recipe); the
                        # per-rank torch.manual_seed above decorrelates ranks.
                        noise = torch.randn(
                            x.shape[0], ens_size, noise_dim, device=dist.device
                        )
                        weights, biases = model(x, mask, taus, noise)
                    else:
                        weights, biases = model(x, mask, taus)
                    expert_precip = x[:, :, 0]
                    if mix_space == "physical":
                        expert_precip = denormalize_precip(
                            expert_precip,
                            mean=train_ds.precip_mean,
                            std=train_ds.precip_std,
                            transform=train_ds.precip_transform,
                        )
                    pred = mix(weights, biases, expert_precip, mask=mask)
                    loss = loss_fn(pred.float(), target, target_mm)
                    tv_term = None
                    if gate_tv_weight > 0:
                        tv_term = gate_smoothness_penalty(
                            weights.float(), biases.float(), loss_fn.weights
                        )
                        loss = loss + gate_tv_weight * tv_term
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        inner_model.parameters(), grad_clip
                    )
                optimizer.step()
                if ema is not None:
                    ema.update(inner_model, epoch=epoch)
                scheduler.step()
                scalars = {"loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                # Composite losses: log the individual terms so their weights
                # are tunable from the curves rather than by guesswork. For an
                # FSS composite the anchor carries the per-term diagnostics.
                term_src = getattr(loss_fn, "anchor", loss_fn)
                if getattr(term_src, "bias_weight", 0.0) > 0:
                    scalars["mse_term"] = term_src.last_mse
                    scalars["bias_mm"] = term_src.last_bias_mm
                if getattr(term_src, "var_weight", 0.0) > 0:
                    scalars["mse_term"] = term_src.last_mse
                    scalars["amp_spatial"] = term_src.last_amp
                if getattr(term_src, "alpha", None) is not None:  # CRPS family
                    scalars["crps_skill"] = term_src.last_skill
                    scalars["crps_spread"] = term_src.last_spread
                    scalars["ens_std_mm"] = term_src.last_ens_std
                if hasattr(term_src, "last_amp_bands") and term_src.last_amp_bands:
                    for bi, v in enumerate(term_src.last_amp_bands):
                        scalars[f"amp_band{bi}"] = v
                if hasattr(loss_fn, "last_fss_term"):
                    scalars["fss_term"] = loss_fn.last_fss_term
                    scalars["anchor_term"] = loss_fn.last_anchor
                if tv_term is not None:
                    scalars["tv_term"] = tv_term
                log.log_minibatch(_ddp_mean_scalars(scalars, dist=dist))

        # ---------------- validation ----------------
        is_best = False
        if (
            validator is not None
            and (epoch + 1) % int(cfg.validation.get("every_n_epochs", 1)) == 0
        ):
            with LaunchLogger("valid", epoch=epoch + 1) as vlog:
                if validate_with_ema:
                    ema.apply_to(inner_model)
                try:
                    metrics, extras = validator.run(model, val_loader)
                finally:
                    if validate_with_ema:
                        ema.restore(inner_model)
                # `loss` is the training criterion on the val split -- the
                # quantity early stopping and best-checkpoint selection use.
                # It is all-reduced inside the validator, so every rank sees
                # the same value and decides identically.
                monitored = metrics.get("loss", None)
                if monitored is not None:
                    if monitored < best_loss - es_min_delta:
                        best_loss = float(monitored)
                        epochs_since_best = 0
                        is_best = True
                    else:
                        epochs_since_best += 1
                        is_best = False
                    metrics["best_loss"] = best_loss
                    metrics["epochs_since_best"] = float(epochs_since_best)
                else:
                    is_best = False
                vlog.log_epoch(metrics)
                if dist.rank == 0 and extras.get("weight_maps"):
                    import numpy as np

                    np.savez(
                        f"weight_maps_epoch{epoch}.npz", **extras["weight_maps"]
                    )

        # ---------------- checkpoint ----------------
        if dist.distributed and dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            if (epoch + 1) % int(cfg.get("checkpoint_save_interval", 5)) == 0:
                save_checkpoint(
                    str(ckpt_dir),
                    models=inner_model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                )
            # Best-so-far weights kept separately: the LAST epoch is the most
            # overfit one, so `checkpoints/` must not be what gets shipped.
            if is_best:
                if validate_with_ema:
                    ema.apply_to(inner_model)
                try:
                    save_checkpoint(
                        str(best_dir),
                        models=inner_model,
                        optimizer=optimizer,
                        scheduler=None,
                        epoch=epoch,
                    )
                finally:
                    if validate_with_ema:
                        ema.restore(inner_model)
                plog.info(
                    f"new best validation loss {best_loss:.4f} at epoch "
                    f"{epoch} -> {best_dir}"
                )

        if es_enabled and epochs_since_best >= es_patience:
            plog.info(
                f"early stopping at epoch {epoch}: {epochs_since_best} epochs "
                f"without improving on {best_loss:.4f} (patience {es_patience})"
            )
            if dist.distributed and dist.world_size > 1:
                torch.distributed.barrier()
            break

    if dist.rank == 0:
        save_checkpoint(
            str(ckpt_dir),
            models=inner_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch,
        )
    plog.info(
        f"training complete; best validation loss {best_loss:.4f} "
        f"(best weights in {best_dir})"
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
