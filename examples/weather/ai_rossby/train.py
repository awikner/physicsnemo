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
    from physicsnemo.experimental.models.pangu_plasim import (
        PanguPlasim,
        PanguPlasimLegacy,
    )
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from ema import ModelEMA
from loss import PanguPlasimLoss
from train_loop import (
    _resolve_amp_dtype,
    make_optimizer,
    make_scheduler,
    train_step,
)
from validate import (
    Deterministic,
    GaussianIC,
    Perturber,
    ReplicateOnly,
    RolloutValidator,
)


def _resolve_path(p: str | None) -> str | None:
    return to_absolute_path(p) if p else None


def _maybe_init_wandb(cfg: DictConfig, *, dist) -> None:
    """Initialize wandb if ``cfg.wandb.enabled`` and rank == 0.

    No-op otherwise. LaunchLogger detects wandb at construction time and
    routes ``log_minibatch`` / ``log_epoch`` dicts through it under the
    section name (``train`` / ``valid``).
    """
    wb = cfg.get("wandb", None)
    if wb is None or not bool(wb.get("enabled", False)) or dist.rank != 0:
        return
    from physicsnemo.utils.logging.wandb import initialize_wandb

    initialize_wandb(
        project=str(wb.get("project", "ai-rossby")),
        entity=str(wb.get("entity", "")) or None,
        name=str(wb.get("name", cfg.get("run_name", "train"))),
        mode=str(wb.get("mode", "offline")),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _build_perturber(cfg_val: DictConfig) -> Perturber:
    """Translate ``cfg.validation.rollout.perturber`` into a Perturber."""
    kind = str(cfg_val.get("perturber", "deterministic")).lower()
    if kind in ("deterministic", "off", "none"):
        return Deterministic()
    if kind in ("replicate", "replicate_only", "stochastic_model"):
        return ReplicateOnly()
    if kind in ("gaussian_ic", "ic_gaussian", "gaussian"):
        scales = OmegaConf.to_container(
            cfg_val.get("perturber_scales", {}), resolve=True
        )
        if not scales:
            raise ValueError(
                "validation.rollout.perturber=gaussian_ic requires "
                "validation.rollout.perturber_scales={var: std, ...}"
            )
        return GaussianIC(scales=scales)
    raise ValueError(f"unknown validation.rollout.perturber={kind!r}")


def build_model(cfg_model: DictConfig):
    """Instantiate the model selected by ``cfg.model.model_type``.

    ``model_type``:

    * ``"PanguPlasimLegacy"`` (default) — deterministic no-VAE Pangu variant.
    * ``"PanguPlasim"`` — VAE-enabled Pangu variant; pairs with
      ``cfg.loss.vae_kl_weight > 0`` to enable the KL term in ``train_step``.
    * ``"SfnoPlasim"`` — vendored Modulus SFNO with the PLASIM-routing
      wrapper; pairs with the ``raw_l2`` loss and ``cosine_warmup`` scheduler
      defaults per PanguWeather v2.0 SFNO_PLASIM_H5_DERECHO_5412.yaml.

    Each ``model_type`` reads only the subset of ``cfg_model`` keys it needs;
    Pangu-specific (patch_size, depths, num_heads, …) and SFNO-specific
    (filter_type, spectral_layers, …) kwargs coexist in their respective YAML
    groups.
    """
    model_type = str(cfg_model.get("model_type", "PanguPlasimLegacy"))
    if model_type in ("PanguPlasim", "PanguPlasimLegacy"):
        cls = {"PanguPlasim": PanguPlasim, "PanguPlasimLegacy": PanguPlasimLegacy}[model_type]
        return cls(
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
    if model_type == "SfnoPlasim":
        return SfnoPlasim(
            surface_variables=list(cfg_model.surface_variables),
            upper_air_variables=list(cfg_model.upper_air_variables),
            constant_boundary_variables=list(cfg_model.constant_boundary_variables),
            varying_boundary_variables=list(cfg_model.varying_boundary_variables),
            diagnostic_variables=list(cfg_model.diagnostic_variables),
            levels=list(cfg_model.levels),
            horizontal_resolution=list(cfg_model.horizontal_resolution),
            spectral_transform=str(cfg_model.get("spectral_transform", "sht")),
            filter_type=str(cfg_model.get("filter_type", "linear")),
            operator_type=str(cfg_model.get("operator_type", "dhconv")),
            scale_factor=int(cfg_model.get("scale_factor", 1)),
            embed_dim=int(cfg_model.embed_dim),
            num_layers=int(cfg_model.get("num_layers", 12)),
            use_mlp=bool(cfg_model.get("use_mlp", True)),
            mlp_ratio=float(cfg_model.get("mlp_ratio", 2.0)),
            activation_function=str(cfg_model.get("activation_function", "gelu")),
            encoder_layers=int(cfg_model.get("encoder_layers", 1)),
            pos_embed=bool(cfg_model.get("pos_embed", False)),
            drop_rate=float(cfg_model.get("drop_rate", 0.0)),
            drop_path_rate=float(cfg_model.get("drop_path_rate", 0.0)),
            num_blocks=int(cfg_model.get("num_blocks", 8)),
            sparsity_threshold=float(cfg_model.get("sparsity_threshold", 0.0)),
            normalization_layer=str(cfg_model.get("normalization_layer", "instance_norm")),
            hard_thresholding_fraction=float(cfg_model.get("hard_thresholding_fraction", 1.0)),
            use_complex_kernels=bool(cfg_model.get("use_complex_kernels", True)),
            big_skip=bool(cfg_model.get("big_skip", True)),
            rank=float(cfg_model.get("rank", 1.0)),
            factorization=cfg_model.get("factorization", None),
            separable=bool(cfg_model.get("separable", False)),
            complex_network=bool(cfg_model.get("complex_network", True)),
            complex_activation=str(cfg_model.get("complex_activation", "real")),
            spectral_layers=int(cfg_model.get("spectral_layers", 3)),
            checkpointing=int(cfg_model.get("checkpointing", 0)),
            data_grid=str(cfg_model.get("data_grid", "equiangular")),
        )
    raise ValueError(
        f"Unknown cfg.model.model_type={model_type!r}; expected "
        "'PanguPlasim', 'PanguPlasimLegacy', or 'SfnoPlasim'"
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
    # Opt-in normalization for constant boundary + diagnostic fields. Off by
    # default for back-compat (ERA5 stats often skip the const-boundary vars);
    # PLASIM stats include lsm/sg/z0, so PLASIM recipes should flip these on
    # to match PanguWeather's data loader (which always normalizes them).
    normalizer_kwargs["normalize_constant_boundary"] = bool(
        data.get("normalize_constant_boundary", False)
    )
    normalizer_kwargs["normalize_diagnostic"] = bool(
        data.get("normalize_diagnostic", False)
    )

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
        latitude_weighted=bool(cfg_loss.get("latitude_weighted", True)),
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # A100 TF32 + cudnn autotune. PanguWeather's reference SFNO trainer flips
    # these on by default; without them ai-rossby was running true fp32 matmul
    # against PanguWeather's TF32 and losing ~15% throughput per benchmarks.
    # TF32 changes mantissa precision (10 bits vs 23) but on A100 stays within
    # ~3 decimals of fp32 — well below typical training noise.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("pangu_plasim_train")

    torch.manual_seed(int(cfg.seed) + dist.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed) + dist.rank)

    LaunchLogger.initialize()

    # --- Wandb (optional) -------------------------------------------------
    # When cfg.wandb is present and enabled, hook wandb up to LaunchLogger
    # so all log_minibatch / log_epoch dicts (training AND validation) flow
    # to the run automatically. Rank-0 only.
    _maybe_init_wandb(cfg, dist=dist)

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

    # --- Rollout validator (optional) -------------------------------------
    # Multi-step autoregressive RMSE + ACC against the held-out year. Streams
    # metrics on the fly (no per-step history retained) and is ensemble-aware
    # via the Perturber API.
    rollout_validator = None
    cfg_rollout = cfg.get("validation", {}).get("rollout", None) if cfg.get("validation") else None
    if cfg_rollout is not None and bool(cfg_rollout.get("enabled", False)):
        if not has_val:
            logger.warning(
                "validation.rollout.enabled=True but data.val_zarr_path is None; "
                "skipping rollout validation."
            )
        else:
            rollout_validator = RolloutValidator(
                dataset=val_datapipe.dataset,
                log_steps=list(cfg_rollout.get("log_steps", [1])),
                device=dist.device,
                ensemble_size=int(cfg_rollout.get("ensemble_size", 1)),
                perturber=_build_perturber(cfg_rollout),
                has_diagnostic=False,  # populated after model is built
                batch_size=int(cfg_rollout.get("batch_size", 1)),
                max_initial_conditions=int(
                    cfg_rollout.get("max_initial_conditions", 4)
                ),
                ic_stride=int(cfg_rollout.get("ic_stride", 1)),
                normalizer=val_datapipe.normalizer,
                seed=int(cfg.seed) + 17,
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
            gradient_as_bucket_view=True,
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

    # --- Mixed precision --------------------------------------------------
    amp_dtype = _resolve_amp_dtype(cfg.get("amp", None))
    # fp16 needs a GradScaler; bf16 retains enough dynamic range to skip it.
    grad_scaler = None
    if amp_dtype == torch.float16 and dist.device.type == "cuda":
        grad_scaler = torch.amp.GradScaler(device="cuda")
        logger.info("AMP enabled with fp16 + GradScaler")
    elif amp_dtype == torch.bfloat16:
        logger.info("AMP enabled with bf16 (no GradScaler)")
    elif amp_dtype is None:
        logger.info(f"AMP disabled (cfg.amp={cfg.get('amp', None)})")

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

    # --- Per-batch loss TSV (benchmarking) --------------------------------
    # When cfg.bench.per_batch_tsv is set, every minibatch's wall-clock time
    # + loss components are appended to a TSV for downstream comparison
    # against PanguWeather. Only rank 0 writes to avoid file contention.
    bench_tsv_path = None
    bench_tsv_file = None
    if cfg.get("bench") and cfg.bench.get("per_batch_tsv"):
        if dist.rank == 0:
            bench_tsv_path = Path(_resolve_path(cfg.bench.per_batch_tsv))
            bench_tsv_path.parent.mkdir(parents=True, exist_ok=True)
            bench_tsv_file = open(bench_tsv_path, "w", buffering=1)  # line-buffered
            bench_tsv_file.write(
                "epoch\tbatch_idx\twall_s\tloss\tsurface\tupper_air\tdiagnostic\tvae_kl\tlr\n"
            )
            logger.info(f"benchmark per-batch TSV → {bench_tsv_path}")

    import time as _time
    _bench_start_wall = _time.perf_counter() if bench_tsv_file is not None else None

    # --- Training loop ----------------------------------------------------
    has_diagnostic = inner_model.has_diagnostic
    if rollout_validator is not None:
        rollout_validator.has_diagnostic = has_diagnostic
    for epoch in range(start_epoch, int(cfg.max_epochs) + 1):
        datapipe.set_epoch(epoch)
        model.train()
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=steps_per_epoch,
            epoch_alert_freq=1,
        ) as log:
            for batch_idx, batch in enumerate(datapipe):
                losses = train_step(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    batch=batch,
                    has_diagnostic=has_diagnostic,
                    vae_kl_weight=float(cfg.loss.get("vae_kl_weight", 0.0)),
                    amp_dtype=amp_dtype,
                    grad_scaler=grad_scaler,
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
                        "vae_kl": losses["vae_kl"],
                    }
                )
                if bench_tsv_file is not None:
                    bench_tsv_file.write(
                        f"{epoch}\t{batch_idx}\t"
                        f"{_time.perf_counter() - _bench_start_wall:.4f}\t"
                        f"{float(losses['loss'].detach()):.6f}\t"
                        f"{float(losses['surface']):.6f}\t"
                        f"{float(losses['upper_air']):.6f}\t"
                        f"{float(losses['diagnostic']):.6f}\t"
                        f"{float(losses['vae_kl']):.6f}\t"
                        f"{optimizer.param_groups[0]['lr']:.6e}\n"
                    )
            log.log_epoch({"lr": optimizer.param_groups[0]["lr"]})

        # --- Validation (optional) ---------------------------------------
        # Two-pass: (1) single-step val_loss on the held-out datapipe — fast
        # MSE sanity number on every epoch; (2) multi-step rollout RMSE + ACC
        # via the RolloutValidator, gated on validation.rollout.enabled and
        # cadence cfg.validation.every_n_epochs.
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

                # Multi-step rollout (cadenced).
                rollout_metrics: dict[str, float] = {}
                if rollout_validator is not None:
                    every = int(
                        (cfg.get("validation") or {}).get("every_n_epochs", 1)
                    )
                    if every > 0 and epoch % every == 0:
                        rollout_metrics = rollout_validator.run(
                            inner_model, epoch=epoch
                        )

                log.log_epoch({"val_loss": val_loss, **rollout_metrics})
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
