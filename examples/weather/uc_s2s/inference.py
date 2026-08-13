"""
UC S2S inference / validation.

Stepper wraps the full Pangu + VAE + SI pipeline for multi-step ensemble
prediction. Uses physicsnemo DistributedManager + Hydra for configuration.

Optimizations integrated from inference_optimized.py:
  - NVTX range markers for Nsight profiling
  - Pinned CPU buffers (lazy-allocated) for async D2H transfers
  - ThreadPoolExecutor for async NetCDF saves overlapping with GPU work
  - NVME / local-scratch output support via params.nvme_dir
"""

import logging
import os
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import cf_xarray as cfxr
import dask
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import xarray as xr
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger

from networks.pangu import PanguModel_Plasim
from networks.vae import VAE
from networks.diffusion import ConditionalDiffusionModel
from networks.stochastic_interpolant import StochasticInterpolant
from train import Trainer
from datapipe.data_loader_multifiles import get_data_loader
from utils import logging_utils

dask.config.set(scheduler="synchronous")
logging_utils.config_logger()

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Stepper(Trainer):
    """
    Inference engine for the UC S2S ensemble pipeline.

    Loads Pangu backbone + VAE + SI/diffusion model and runs multi-step
    autoregressive ensemble rollout over the validation dataset.
    """

    def __init__(self, params, dist_mgr: DistributedManager,
                 async_save: bool = False, disable_save: bool = False):
        self.params = params
        self.dist = dist_mgr
        self.world_rank = dist_mgr.rank
        self.device = dist_mgr.device
        self.async_save = async_save
        self.disable_save = disable_save

        self._first_batch_loaded = False
        self._first_forward_done = False
        self._pending_saves = []
        self._save_executor = ThreadPoolExecutor(max_workers=2) if async_save else None

        # Pinned CPU buffers — lazy-allocated on first use
        self._surf_cpu = None
        self._upper_cpu = None
        self._diag_cpu = None

        self.iters = 0
        self.startEpoch = 0
        self.epoch = 0
        self.run_uuid = str(uuid.uuid4())

        if async_save:
            logging.info("Asynchronous saving enabled")
        if disable_save:
            logging.info("NetCDF saving disabled (profiling mode)")

        self.has_land = False
        self.has_ocean = False
        self.mask_output = False
        if hasattr(params, "land_variables") and len(params.land_variables) > 0:
            self.has_land = True
        else:
            params["land_variables"] = []
        if hasattr(params, "ocean_variables") and len(params.ocean_variables) > 0:
            self.has_ocean = True
        else:
            params["ocean_variables"] = []
        if hasattr(params, "mask_output"):
            self.mask_output = params.mask_output

        # ── Data loader ───────────────────────────────────────────────────────
        logging.info("rank %d, begin data loader init", self.world_rank)
        nvtx.range_push("dataloader creation")
        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            params, params.data_dir, dist_mgr.distributed,
            year_start=params.val_year_start, year_end=params.val_year_end,
            train=False, num_inferences=params.num_inferences, validate=True,
        )
        nvtx.range_pop()
        logging.info("Valid dataset length: %d", len(self.valid_dataset))

        self.constant_boundary_data = (
            self.valid_dataset.constant_boundary_data
            .unsqueeze(0)
            .expand(params.batch_size, -1, -1, -1)
            .to(self.device)
        )
        logging.info("rank %d, data loader initialized", self.world_rank)

        # ── Models ────────────────────────────────────────────────────────────
        if params.nettype == "pangu_plasim":
            land_mask = None
            if (self.has_land or self.has_ocean) and self.mask_output:
                land_mask = self.valid_dataset.land_mask.detach().to(self.device)
            self.model_vae = VAE(params).to(self.device)
            self.model_det = PanguModel_Plasim(
                params, land_mask=land_mask, mask_fill=params.mask_fill
            ).to(self.device)
        else:
            raise NotImplementedError(f"nettype {params.nettype} not implemented")

        model_type = params.get("diffusion_model_type", "SI")
        if model_type == "SI":
            self.diff_model = StochasticInterpolant(
                VAEEncoder=self.model_vae, DETEncoder=self.model_det, params=params
            ).to(self.device)
        else:
            self.diff_model = ConditionalDiffusionModel(
                T=1000, VAEEncoder=self.model_vae, DETEncoder=self.model_det, params=params
            ).to(self.device)
        self.model_diff = self.diff_model

        # ── Checkpoints ───────────────────────────────────────────────────────
        self.restore_diff_checkpoint(params.checkpoint_path_diff)
        self._reload_vae_checkpoint(
            checkpoint_path_vae=params.checkpoint_path_vae_c1,
            checkpoint_path_det=params.checkpoint_path_det,
        )

        finetune_ckpt = params.get("checkpoint_path_finetune", None)
        if finetune_ckpt and finetune_ckpt != "null" and os.path.isfile(finetune_ckpt):
            self.restore_finetune_checkpoint(finetune_ckpt)
            logging.info("Fine-tuned decoder loaded from %s", finetune_ckpt)

        # ── Output directory (NVME-aware) ─────────────────────────────────────
        nvme_dir = params.get("nvme_dir", "")
        base_out = params.get("output_dir", "./inference_output")
        if nvme_dir:
            self.output_dir = os.path.join(nvme_dir, "inference_output")
            logging.info("NVME enabled — saving predictions to %s", self.output_dir)
        else:
            self.output_dir = base_out
        if self.world_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def restore_diff_checkpoint(self, checkpoint_path_diff):
        nvtx.range_push("diff checkpoint load")
        ckpt = torch.load(checkpoint_path_diff,
                          map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
        raw_state = ckpt["model_state"]
        if any(k.startswith("module.") for k in raw_state):
            raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
        unet_state = {
            k: v for k, v in raw_state.items()
            if not k.startswith("encoder.") and not k.startswith("model_det.")
        }
        self.diff_model.load_state_dict(unet_state, strict=False)
        self.iters = ckpt.get("iters", 0)
        self.startEpoch = ckpt.get("epoch", 0)
        self.epoch = self.startEpoch
        nvtx.range_pop()

    def restore_finetune_checkpoint(self, finetune_ckpt_path):
        nvtx.range_push("finetune checkpoint load")
        ckpt = torch.load(finetune_ckpt_path,
                          map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
        state = ckpt["model_state"]
        if any(k.startswith("module.") for k in state):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        decoder_state = {k: v for k, v in state.items() if k.startswith("model_det.")}
        self.diff_model.load_state_dict(decoder_state, strict=False)
        logging.info("Loaded %d decoder keys from %s", len(decoder_state), finetune_ckpt_path)
        nvtx.range_pop()

    def _reload_vae_checkpoint(self, checkpoint_path_vae=None, checkpoint_path_det=None):
        nvtx.range_push("vae+det checkpoint load")

        def _load(path, module):
            ckpt = torch.load(path,
                              map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
            raw_state = ckpt.get("model_state", ckpt)
            if any(k.startswith("module.") for k in raw_state):
                raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
            module.load_state_dict(raw_state, strict=False)

        _load(checkpoint_path_vae, self.diff_model.encoder)
        _load(checkpoint_path_det, self.diff_model.model_det)
        self.model_vae = self.diff_model.encoder
        self.model_det = self.diff_model.model_det
        self.diff_model.freeze_encoder()
        logging.info("VAE and det weights reloaded; encoder frozen")
        nvtx.range_pop()

    # ── Inference entry point ─────────────────────────────────────────────────

    def predict(self):
        if self.params.log_to_screen:
            logging.info("Starting Model Inference Loop...")
        total_time = self._validate_one_epoch()
        if self.world_rank == 0:
            logging.info("Inference wall time (seconds): %.3f", total_time)

    def _validate_one_epoch(self):
        self.diff_model.eval()
        params = self.params
        device = self.device
        total_start = time.time()

        inference_steps = int(params.get("inference_steps",
                                         max(params.forecast_lead_times)))
        num_ensemble = int(params.get("num_ensemble_members", 1))
        temperature = float(params.get("temperature", 1.5))
        has_diag = params.get("has_diagnostic", False)

        with torch.inference_mode(), \
             torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):

            data_iter = iter(self.valid_data_loader)
            for i, data in enumerate(data_iter):
                nvtx.range_push(f"batch {i} H2D")

                # Unpack according to data loader return signature:
                # (surf, ua, tgt_surf, tgt_ua [, tgt_diag], vbc, start_time_tensor)
                if has_diag:
                    surf_cpu, ua_cpu, _, _, _, vbc_cpu, times = data
                else:
                    surf_cpu, ua_cpu, _, _, vbc_cpu, times = data

                surf = surf_cpu.to(device, dtype=torch.float32, non_blocking=True)
                ua   = ua_cpu.to(device, dtype=torch.float32, non_blocking=True)
                vbc  = vbc_cpu.to(device, dtype=torch.float32, non_blocking=True)
                nvtx.range_pop()
                self._first_batch_loaded = True

                # Decode start times from tensor [year, month, day, hour]
                times_np = times.numpy().astype(int)
                start_times = [
                    self.valid_dataset.datetime_class(
                        times_np[idx, 0], times_np[idx, 1],
                        times_np[idx, 2], hour=times_np[idx, 3])
                    for idx in range(times_np.shape[0])
                ]
                save_mask = [t.strftime("%H") == "00" for t in start_times]
                should_save = (not self.disable_save) and any(save_mask)

                for ens_id in range(num_ensemble):
                    nvtx.range_push(f"inference batch={i} ens={ens_id}")

                    s_cur, ua_cur = surf, ua

                    if should_save:
                        _surf_gpu  = [surf.detach()]
                        _ua_gpu    = [ua.detach()]
                        _diag_gpu  = [torch.zeros(
                            surf.shape[0], self.diff_model.num_diagnostic_vars,
                            surf.shape[2], surf.shape[3],
                            dtype=torch.float32, device=device)]

                    nvtx.range_push(f"rollout batch={i} ens={ens_id}")
                    for step in range(inference_steps):
                        nvtx.range_push(f"forward step={step}")

                        vbc_step = vbc[:, step] if vbc is not None else torch.zeros(1, device=device)

                        if (not self._first_forward_done) and i == 0 and ens_id == 0 and step == 0:
                            nvtx.range_push("first forward pass")
                            pred_sfc, pred_ua, pred_diag = self.diff_model.prediction(
                                surface_in=s_cur,
                                constant_boundary=self.constant_boundary_data,
                                varying_boundary=vbc_step,
                                upper_air_in=ua_cur,
                                num_samples=1,
                                device=device,
                                temperature=temperature,
                            )
                            nvtx.range_pop()
                            self._first_forward_done = True
                        else:
                            pred_sfc, pred_ua, pred_diag = self.diff_model.prediction(
                                surface_in=s_cur,
                                constant_boundary=self.constant_boundary_data,
                                varying_boundary=vbc_step,
                                upper_air_in=ua_cur,
                                num_samples=1,
                                device=device,
                                temperature=temperature,
                            )

                        if should_save:
                            _surf_gpu.append(pred_sfc.detach())
                            _ua_gpu.append(pred_ua.detach())
                            if pred_diag is not None:
                                _diag_gpu.append(pred_diag.detach())

                        s_cur  = pred_sfc.detach()
                        ua_cur = pred_ua.detach()
                        nvtx.range_pop()  # forward step
                    nvtx.range_pop()  # rollout

                    if not should_save:
                        nvtx.range_pop()  # inference batch
                        continue

                    # ── Stack GPU tensors ─────────────────────────────────────
                    nvtx.range_push("output stack")
                    surf_stack = torch.stack(_surf_gpu, dim=1)   # (B, T, C, H, W)
                    ua_stack   = torch.stack(_ua_gpu,   dim=1)
                    diag_stack = torch.stack(_diag_gpu, dim=1) if _diag_gpu else None
                    nvtx.range_pop()

                    B, T = surf_stack.shape[:2]

                    # ── Inverse-normalise on GPU, then async D2H into pinned buffers
                    nvtx.range_push("inv-transform + D2H")
                    surf_t  = self.valid_dataset.surface_inv_transform(
                        surf_stack.view(B * T, *surf_stack.shape[2:]))
                    ua_t    = self.valid_dataset.upper_air_inv_transform(
                        ua_stack.view(B * T, *ua_stack.shape[2:]))
                    diag_t  = (self.valid_dataset.diagnostic_inv_transform(
                        diag_stack.view(B * T, *diag_stack.shape[2:]))
                        if diag_stack is not None else None)

                    # Lazy-allocate pinned CPU buffers to avoid repeated cudaHostAlloc
                    if self._surf_cpu is None or self._surf_cpu.shape != surf_t.shape:
                        self._surf_cpu = torch.empty(surf_t.shape, dtype=torch.float32, pin_memory=True)
                    if self._upper_cpu is None or self._upper_cpu.shape != ua_t.shape:
                        self._upper_cpu = torch.empty(ua_t.shape, dtype=torch.float32, pin_memory=True)

                    self._surf_cpu.copy_(surf_t,  non_blocking=True)
                    self._upper_cpu.copy_(ua_t,   non_blocking=True)

                    diag_np = None
                    if diag_t is not None:
                        if self._diag_cpu is None or self._diag_cpu.shape != diag_t.shape:
                            self._diag_cpu = torch.empty(diag_t.shape, dtype=torch.float32, pin_memory=True)
                        self._diag_cpu.copy_(diag_t, non_blocking=True)

                    torch.cuda.synchronize()

                    surf_np  = self._surf_cpu.numpy().reshape(B, T, *surf_stack.shape[2:]).copy()
                    ua_np    = self._upper_cpu.numpy().reshape(B, T, *ua_stack.shape[2:]).copy()
                    if diag_t is not None:
                        diag_np = self._diag_cpu.numpy().reshape(B, T, *diag_stack.shape[2:]).copy()
                    nvtx.range_pop()  # inv-transform + D2H

                    nvtx.range_pop()  # inference batch

                    # ── Save (async or sync) ──────────────────────────────────
                    nvtx.range_push(f"save batch={i} ens={ens_id}")
                    if self._save_executor is not None:
                        f = self._save_executor.submit(
                            self.save_prediction,
                            surf_np, ua_np, list(start_times), diag_np, ens_id)
                        self._pending_saves.append(f)
                    else:
                        self.save_prediction(surf_np, ua_np, start_times, diag_np, ens_id=ens_id)
                    nvtx.range_pop()

        # Wait for all background saves before returning
        for f in self._pending_saves:
            f.result()
        self._pending_saves.clear()
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None

        return time.time() - total_start

    # ── NetCDF output ─────────────────────────────────────────────────────────

    def save_prediction(self, surface_prediction, upper_air_prediction,
                        start_times, diagnostic_prediction=None, ens_id=None):
        """Write one ensemble member's rollout to NetCDF, one file per 00 UTC IC."""
        params = self.params
        savedir = os.path.join(self.output_dir, "predictions")
        os.makedirs(savedir, exist_ok=True)

        if ens_id == 0:
            logging.info("start_times for ens 0: %s", start_times)

        B = surface_prediction.shape[0]
        for sample in range(B):
            if start_times[sample].strftime("%H") != "00":
                logging.debug("Skipping non-00UTC start: %s", start_times[sample])
                continue

            time_range = xr.cftime_range(
                start=start_times[sample],
                end=start_times[sample] + timedelta(
                    hours=params.timedelta_hours * (surface_prediction.shape[1] - 1 + params.get("inference_steps", 1))),
                freq=f"{params.timedelta_hours}h",
                inclusive="both",
            )

            coordinates = {
                "time":      time_range,
                "level":     params.levels,
                "latitude":  params.lat,
                "longitude": params.lon,
            }

            filename = "%s_%dh_%dstep_%s_ens_%s.nc" % (
                params.nettype,
                params.timedelta_hours,
                params.get("inference_steps", surface_prediction.shape[1]),
                start_times[sample].strftime("%Y%m%d%H"),
                ens_id,
            )

            ds = xr.Dataset(
                data_vars={},
                coords=coordinates,
                attrs={"description": f"Prediction from {params.nettype}"},
            )
            ds["level"].attrs.update({"axis": "Z", "positive": "down"})
            ds["latitude"].attrs["axis"] = "Y"
            ds["longitude"].attrs["axis"] = "X"
            ds = ds.cf.guess_coord_axis()

            for idx, var in enumerate(self.valid_dataset.surface_variables):
                ds[var] = xr.DataArray(
                    data=surface_prediction[sample, :, idx],
                    dims=["time", "latitude", "longitude"],
                    coords={"time": time_range,
                            "latitude": ds.latitude.values,
                            "longitude": ds.longitude.values},
                )

            for idx, var in enumerate(self.valid_dataset.upper_air_variables):
                ds[var] = xr.DataArray(
                    data=upper_air_prediction[sample, :, idx],
                    dims=["time", "level", "latitude", "longitude"],
                    coords=coordinates,
                )

            if params.get("has_diagnostic", False) and diagnostic_prediction is not None:
                for idx, var in enumerate(self.valid_dataset.diagnostic_variables):
                    ds[var] = xr.DataArray(
                        data=diagnostic_prediction[sample, :, idx],
                        dims=["time", "latitude", "longitude"],
                        coords={"time": time_range,
                                "latitude": ds.latitude.values,
                                "longitude": ds.longitude.values},
                    )

            ds["latitude"]  = ds["latitude"].astype("float32").assign_attrs(
                {"long_name": "Latitude",  "unit": "degrees_north"})
            ds["longitude"] = ds["longitude"].astype("float32").assign_attrs(
                {"long_name": "Longitude", "unit": "degrees_east"})
            ds["time"]  = ds["time"].assign_attrs({"long_name": "Forecast Valid Time"})
            ds["level"] = ds["level"].astype("float32").assign_attrs(
                {"long_name": "Level", "unit": "hPa"})

            out_path = os.path.join(savedir, filename)
            if not self.disable_save:
                nvtx.range_push("file write")
                ds.to_netcdf(out_path, mode="w", engine="h5netcdf")
                nvtx.range_pop()
                logging.info("Saved: %s", out_path)
            else:
                nvtx.range_push("save disabled")
                nvtx.range_pop()


@hydra.main(version_base="1.2", config_path="conf", config_name="config_inference")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist_mgr = DistributedManager()
    logger = PythonLogger("inference")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    async_save   = bool(cfg.get("async_save",   False))
    disable_save = bool(cfg.get("disable_save", False))

    s_time = time.time()
    stepper = Stepper(cfg, dist_mgr, async_save=async_save, disable_save=disable_save)
    stepper.predict()
    logger.info("Inference done — rank %d  total=%.1fs", dist_mgr.rank, time.time() - s_time)


if __name__ == "__main__":
    main()
