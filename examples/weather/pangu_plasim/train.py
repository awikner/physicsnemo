# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra entrypoint for Pangu_Plasim training (Phase 3 v1: PanguPlasimLegacy).

Wires together:

* :class:`physicsnemo.experimental.datapipes.plasim.PlasimClimateDatapipe`
  (Zarr-backed PLASIM datapipe with normalizer + NaN fill)
* :class:`physicsnemo.experimental.models.pangu_plasim.PanguPlasimLegacy`
* :func:`.train_loop.train_step` + :func:`.train_loop.make_optimizer/_scheduler`
* :class:`.loss.PanguPlasimLoss`
* :class:`.ema.ModelEMA`
* :class:`physicsnemo.distributed.DistributedManager` for DDP
* :func:`physicsnemo.utils.save_checkpoint` /
  :func:`physicsnemo.utils.load_checkpoint`
* :class:`physicsnemo.utils.logging.LaunchLogger`

Launch single-GPU:

.. code-block:: bash

    python train.py data.zarr_path=/path/to/store.zarr \\
                    data.mean_path=... data.std_path=...

Launch multi-GPU with torchrun (Delta convention):

.. code-block:: bash

    torchrun --standalone --nproc-per-node=2 train.py [overrides]
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=Warning, module=r"physicsnemo\.experimental.*"
    )
    from physicsnemo.experimental.datapipes.plasim import (
        NanFillTransform,
        PlasimClimateDatapipe,
        PlasimClimateDataset,
        PlasimNormalizer,
    )
    from physicsnemo.experimental.models.pangu_plasim import PanguPlasimLegacy

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from ema import ModelEMA
from loss import PanguPlasimLoss
from train_loop import make_optimizer, make_scheduler, train_step


def _resolve_path(p: str | None) -> str | None:
    return to_absolute_path(p) if p else None


def build_model(cfg_model: DictConfig) -> PanguPlasimLegacy:
    """Instantiate PanguPlasimLegacy from the model sub-config."""
    return PanguPlasimLegacy(
        surface_variables=list(cfg_model.surface_variables),
        upper_air_variables=list(cfg_model.upper_air_variables),
        constant_boundary_variables=list(cfg_model.constant_boundary_variables),
        varying_boundary_variables=list(cfg_model.varying_boundary_variables),
        diagnostic_variables=list(cfg_model.diagnostic_variables),
        land_variables=list(cfg_model.get("land_variables", [])),
        ocean_variables=list(cfg_model.get("ocean_variables", [])),
        levels=list(cfg_model.levels),
        horizontal_resolution=list(cfg_model.horizontal_resolution),
        patch_size=list(cfg_model.patch_size),
        window_size=list(cfg_model.window_size),
        depths=list(cfg_model.depths),
        num_heads=list(cfg_model.num_heads),
        embed_dim=int(cfg_model.embed_dim),
        updown_scale_factor=int(cfg_model.updown_scale_factor),
        predict_delta=bool(cfg_model.predict_delta),
        mask_output=bool(cfg_model.mask_output),
        upper_air_boundary=bool(cfg_model.upper_air_boundary),
        vertical_windowing=bool(cfg_model.vertical_windowing),
        subpixel_deconv=bool(cfg_model.subpixel_deconv),
        polar_pad=bool(cfg_model.polar_pad),
        grid_has_poles=bool(cfg_model.grid_has_poles),
        recovery_head=bool(cfg_model.recovery_head),
        diagnostic_head=bool(cfg_model.diagnostic_head),
        has_diagnostic=bool(cfg_model.has_diagnostic),
        drop_rate=float(cfg_model.drop_rate),
        checkpointing=int(cfg_model.checkpointing),
        use_reentrant=bool(cfg_model.use_reentrant),
    )


def build_datapipe(
    cfg: DictConfig,
    *,
    zarr_path: str,
    distributed: bool,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> PlasimClimateDatapipe:
    """Construct a PlasimClimateDatapipe wired with normalizer + NaN fill."""
    data = cfg.data
    model = cfg.model

    raw_dataset = PlasimClimateDataset(
        zarr_path,
        boundary_zarr_path=_resolve_path(data.boundary_zarr_path),
        yearly_repeating_boundary=bool(data.yearly_repeating_boundary),
        leap_boundary_zarr_path=_resolve_path(data.leap_boundary_zarr_path),
        non_leap_boundary_zarr_path=_resolve_path(data.non_leap_boundary_zarr_path),
    )

    normalizer_kwargs: dict = {}
    if data.delta_std_path:
        normalizer_kwargs["predict_delta"] = True
        normalizer_kwargs["delta_std_path"] = _resolve_path(data.delta_std_path)

    normalizer = PlasimNormalizer.from_dataset(
        raw_dataset,
        mean_path=_resolve_path(data.mean_path),
        std_path=_resolve_path(data.std_path),
        **normalizer_kwargs,
    )

    nan_fill = NanFillTransform(
        constant_boundary_variables=list(model.constant_boundary_variables),
        varying_boundary_variables=list(model.varying_boundary_variables),
        fill_values=dict(OmegaConf.to_container(data.nan_fill_values, resolve=True) or {}),
        default=float(data.nan_fill_default),
    )

    pipe = PlasimClimateDatapipe(
        zarr_path,
        forecast_lead_times=list(data.forecast_lead_times),
        normalizer=normalizer,
        nan_fill=None,  # per-variable NaN fill runs CPU-side via dataset.transform below.
        batch_size=int(data.batch_size),
        num_samples_per_epoch=data.num_samples_per_epoch,
        shuffle=shuffle,
        num_workers=int(data.num_workers),
        prefetch_factor=int(data.prefetch_factor),
        persistent_workers=bool(data.persistent_workers),
        pin_memory=bool(data.pin_memory),
        device=device,
        seed=int(seed),
        distributed=distributed,
        boundary_zarr_path=_resolve_path(data.boundary_zarr_path),
        yearly_repeating_boundary=bool(data.yearly_repeating_boundary),
        leap_boundary_zarr_path=_resolve_path(data.leap_boundary_zarr_path),
        non_leap_boundary_zarr_path=_resolve_path(data.non_leap_boundary_zarr_path),
    )

    # Attach the per-variable NaN fill as the dataset's transform so it runs
    # in worker processes (CPU) before pin_memory + device transfer.
    pipe.dataset.transform = nan_fill
    return pipe


def build_loss(cfg: DictConfig) -> PanguPlasimLoss:
    """Build :class:`PanguPlasimLoss` aligned with the model's variable groups.

    ``cfg.model.upper_air_variables`` is assumed to be in sigma-then-pressure
    order so it matches the channel ordering produced by
    :class:`PlasimClimateDataset` (and the model's forward output).
    """
    cfg_loss = cfg.loss
    cfg_model = cfg.model

    def _maybe_dict(v):
        if v is None:
            return None
        return dict(OmegaConf.to_container(v, resolve=True) or {})

    return PanguPlasimLoss(
        surface_variables=list(cfg_model.surface_variables),
        upper_air_variable_names=list(cfg_model.upper_air_variables),
        diagnostic_variables=list(cfg_model.diagnostic_variables),
        num_lat=int(cfg_model.horizontal_resolution[0]),
        loss_type=str(cfg_loss.loss_type),
        surface_weight=float(cfg_loss.surface_weight),
        upper_air_weight=float(cfg_loss.upper_air_weight),
        diagnostic_weight=float(cfg_loss.diagnostic_weight),
        surface_var_weights=_maybe_dict(cfg_loss.surface_var_weights),
        upper_air_var_weights=_maybe_dict(cfg_loss.upper_air_var_weights),
        diagnostic_var_weights=_maybe_dict(cfg_loss.diagnostic_var_weights),
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("pangu_plasim_train")

    torch.manual_seed(int(cfg.seed) + dist.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed) + dist.rank)

    LaunchLogger.initialize()

    # --- Data -------------------------------------------------------------
    datapipe = build_datapipe(
        cfg,
        zarr_path=_resolve_path(cfg.data.zarr_path),
        distributed=(dist.world_size > 1),
        device=dist.device,
        shuffle=bool(cfg.data.shuffle),
        seed=int(cfg.seed),
    )

    has_val = cfg.data.val_zarr_path is not None
    val_datapipe = None
    if has_val:
        val_datapipe = build_datapipe(
            cfg,
            zarr_path=_resolve_path(cfg.data.val_zarr_path),
            distributed=(dist.world_size > 1),
            device=dist.device,
            shuffle=False,
            seed=int(cfg.seed) + 1,
        )

    steps_per_epoch = len(datapipe)
    total_steps = max(1, steps_per_epoch * int(cfg.max_epochs))
    logger.info(
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"world_size={dist.world_size}, device={dist.device}"
    )

    # --- Model + DDP ------------------------------------------------------
    model = build_model(cfg.model).to(dist.device)
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            output_device=dist.device if dist.device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )
    inner_model = model.module if hasattr(model, "module") else model

    # --- Loss + optim ------------------------------------------------------
    loss_fn = build_loss(cfg).to(dist.device)

    # Cosine-warmup wants ``num_warmup_steps`` filled from epoch-count.
    sched_cfg = OmegaConf.to_container(cfg.scheduler, resolve=True)
    if sched_cfg.get("scheduler") == "LinearWarmupCosineAnnealingLR":
        sched_cfg["num_warmup_steps"] = int(
            sched_cfg.get("num_warmup_epochs", 0) * steps_per_epoch
        )
    sched_cfg = OmegaConf.create(sched_cfg)

    optimizer = make_optimizer(inner_model, sched_cfg)
    scheduler = make_scheduler(optimizer, sched_cfg, total_steps=total_steps)

    # --- EMA --------------------------------------------------------------
    ema = None
    if bool(cfg.ema.enabled):
        ema = ModelEMA(
            inner_model,
            decay=float(cfg.ema.decay),
            warmup_epochs=int(cfg.ema.warmup_epochs),
        )

    # --- Checkpoint resume ------------------------------------------------
    ckpt_dir = Path("./checkpoints")
    loaded_epoch = load_checkpoint(
        str(ckpt_dir),
        models=inner_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )
    start_epoch = max(int(cfg.start_epoch), loaded_epoch + 1)

    # --- Training loop ----------------------------------------------------
    has_diagnostic = inner_model.has_diagnostic
    for epoch in range(start_epoch, int(cfg.max_epochs) + 1):
        datapipe.set_epoch(epoch)
        model.train()
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=steps_per_epoch,
            epoch_alert_freq=1,
        ) as log:
            for batch in datapipe:
                losses = train_step(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    batch=batch,
                    has_diagnostic=has_diagnostic,
                )
                if float(cfg.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        inner_model.parameters(), float(cfg.grad_clip_norm)
                    )
                if ema is not None:
                    ema.update(inner_model, epoch=epoch)
                log.log_minibatch(
                    {
                        "loss": losses["loss"].detach(),
                        "surface": losses["surface"],
                        "upper_air": losses["upper_air"],
                        "diagnostic": losses["diagnostic"],
                    }
                )
            log.log_epoch({"lr": optimizer.param_groups[0]["lr"]})

        # --- Validation (optional) ---------------------------------------
        if val_datapipe is not None:
            val_datapipe.set_epoch(epoch)
            if ema is not None:
                ema.apply_to(inner_model)
            model.eval()
            with LaunchLogger("valid", epoch=epoch) as log:
                with torch.no_grad():
                    total = 0
                    accum = torch.zeros((), device=dist.device)
                    for batch in val_datapipe:
                        out = inner_model(
                            batch["surface_in"],
                            batch["constant_boundary"],
                            batch["varying_boundary"],
                            batch["upper_air_in"],
                        )
                        if has_diagnostic:
                            o_s, o_u, o_d = out[0], out[1], out[2]
                        else:
                            o_s, o_u = out[0], out[1]
                            o_d = None
                        l = loss_fn(
                            o_s,
                            o_u,
                            batch["target_surface"],
                            batch["target_upper_air"],
                            out_diagnostic=o_d,
                            target_diagnostic=batch.get("diagnostic")
                            if has_diagnostic
                            else None,
                        )["loss"]
                        accum += l.detach() * batch["surface_in"].shape[0]
                        total += batch["surface_in"].shape[0]
                    val_loss = (accum / max(total, 1)).item()
                log.log_epoch({"val_loss": val_loss})
            if ema is not None:
                ema.restore(inner_model)
            model.train()

        if dist.world_size > 1:
            torch.distributed.barrier()

        # --- Save checkpoint ---------------------------------------------
        if (
            epoch % int(cfg.checkpoint_save_interval) == 0
            and dist.rank == 0
        ):
            save_checkpoint(
                str(ckpt_dir),
                models=inner_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metadata={"ema": ema.state_dict() if ema is not None else None},
            )

    if dist.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
