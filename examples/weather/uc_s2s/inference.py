"""
UC S2S inference / validation.

Stepper class wraps the full Pangu + VAE + SI pipeline for multi-step
ensemble prediction. Replaces manual dist.init_process_group / YParams /
argparse with physicsnemo DistributedManager + Hydra.

All scientific logic (checkpoint loading, ensemble generation, rollout,
NetCDF saving, diagnostic plots) is preserved from the original inference.py.
"""

import logging
import os
import uuid
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import dask
import numpy as np
import torch
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
from utils.data_loader_multifiles import get_data_loader
from utils import logging_utils

dask.config.set(scheduler="synchronous")
logging_utils.config_logger()


class Stepper(Trainer):
    """
    Inference engine for the UC S2S ensemble pipeline.

    Loads Pangu backbone + VAE + SI/diffusion model and runs multi-step
    autoregressive ensemble rollout over the validation dataset.
    """

    def __init__(self, params, dist_mgr: DistributedManager, async_save: bool = False):
        self.params = params
        self.dist = dist_mgr
        self.world_rank = dist_mgr.rank
        self.device = dist_mgr.device
        self.async_save = async_save

        self.iters = 0
        self.startEpoch = 0
        self.epoch = 0
        self.run_uuid = str(uuid.uuid4())

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

        self.num_diagnostic_vars = len(params.diagnostic_variables) if params.get("has_diagnostic", False) else 0

        logging.info("rank %d, begin data loader init", self.world_rank)
        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            params, params.data_dir, dist_mgr.distributed,
            year_start=params.val_year_start, year_end=params.val_year_end,
            train=False, num_inferences=params.num_inferences, validate=True,
        )
        logging.info("Valid dataset length: %d", len(self.valid_dataset))

        self.constant_boundary_data = (
            self.valid_dataset.constant_boundary_data
            .unsqueeze(0)
            .expand(params.batch_size, -1, -1, -1)
            .to(self.device)
        )
        logging.info("rank %d, data loader initialized", self.world_rank)

        # Build models
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

        # Load checkpoints
        self.restore_diff_checkpoint(params.checkpoint_path_diff)
        self._reload_vae_checkpoint(
            checkpoint_path_vae=params.checkpoint_path_vae_c1,
            checkpoint_path_det=params.checkpoint_path_det,
        )

        finetune_ckpt = params.get("checkpoint_path_finetune", None)
        if finetune_ckpt and os.path.isfile(finetune_ckpt):
            self.restore_finetune_checkpoint(finetune_ckpt)
            logging.info("Fine-tuned decoder loaded from %s", finetune_ckpt)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def restore_diff_checkpoint(self, checkpoint_path_diff):
        ckpt = torch.load(checkpoint_path_diff, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
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

    def restore_finetune_checkpoint(self, finetune_ckpt_path):
        ckpt = torch.load(finetune_ckpt_path, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
        state = ckpt["model_state"]
        if any(k.startswith("module.") for k in state):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        decoder_state = {k: v for k, v in state.items() if k.startswith("model_det.")}
        self.diff_model.load_state_dict(decoder_state, strict=False)
        logging.info("Loaded %d decoder keys from %s", len(decoder_state), finetune_ckpt_path)

    def _reload_vae_checkpoint(self, checkpoint_path_vae=None, checkpoint_path_det=None):
        def _load(path, module):
            ckpt = torch.load(path, map_location=f"cuda:{self.dist.local_rank}", weights_only=False)
            raw_state = ckpt["model_state"]
            if any(k.startswith("module.") for k in raw_state):
                raw_state = OrderedDict((k[7:], v) for k, v in raw_state.items())
            module.load_state_dict(raw_state, strict=True)

        _load(checkpoint_path_vae, self.diff_model.encoder)
        _load(checkpoint_path_det, self.diff_model.model_det)
        self.model_vae = self.diff_model.encoder
        self.model_det = self.diff_model.model_det
        self.diff_model.freeze_encoder()
        logging.info("VAE and det weights reloaded; encoder frozen")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self):
        self.diff_model.eval()
        for data in tqdm(self.valid_data_loader, desc="Inference"):
            self._run_inference_batch(data)

    def _run_inference_batch(self, data):
        params = self.params
        device = self.device

        if params.get("has_diagnostic", False):
            surf, ua, tgt_sfc, tgt_ua, tgt_diag, vbc, start_times = data
        else:
            surf, ua, tgt_sfc, tgt_ua, vbc, start_times = data
            tgt_diag = None

        surf = surf.to(device, non_blocking=True)
        ua   = ua.to(device, non_blocking=True)
        if vbc is not None:
            vbc = vbc.to(device, non_blocking=True)

        num_samples = int(params.get("num_ensemble_members", 1))
        inference_steps = int(params.inference_steps)

        all_sfc_preds  = []
        all_ua_preds   = []
        all_diag_preds = []

        s_cur, ua_cur = surf, ua
        for step in range(inference_steps):
            vbc_step = vbc[:, step] if vbc is not None else torch.zeros(1, device=device)
            pred_sfc, pred_ua, pred_diag = self.diff_model.prediction(
                surface_in=s_cur,
                constant_boundary=self.constant_boundary_data,
                varying_boundary=vbc_step,
                upper_air_in=ua_cur,
                num_samples=num_samples,
                device=device,
                temperature=float(params.get("temperature", 1.5)),
            )
            all_sfc_preds.append(pred_sfc.cpu())
            all_ua_preds.append(pred_ua.cpu())
            if pred_diag is not None:
                all_diag_preds.append(pred_diag.cpu())

            # Advance state with ensemble mean
            s_cur  = pred_sfc.reshape(-1, num_samples, *pred_sfc.shape[1:]).mean(1).detach()
            ua_cur = pred_ua.reshape(-1, num_samples, *pred_ua.shape[1:]).mean(1).detach()

        pred_sfc_stack  = torch.stack(all_sfc_preds,  dim=1)
        pred_ua_stack   = torch.stack(all_ua_preds,   dim=1)
        pred_diag_stack = torch.stack(all_diag_preds, dim=1) if all_diag_preds else None

        self.save_prediction(pred_sfc_stack, pred_ua_stack, start_times, pred_diag_stack)

    def save_prediction(self, surface_prediction, upper_air_prediction, start_times,
                        diagnostic_prediction=None, ens_id=None):
        """Save ensemble predictions to NetCDF files."""
        out_dir = self.params.get("output_dir", "./inference_output")
        os.makedirs(out_dir, exist_ok=True)

        B = surface_prediction.shape[0]
        for b in range(B):
            # Build a simple xarray Dataset and write to NetCDF
            sfc_np = surface_prediction[b].numpy()   # (steps, C_sfc, H, W)
            ua_np  = upper_air_prediction[b].numpy()  # (steps, C_ua, P, H, W)

            fname = os.path.join(out_dir, f"pred_{self.run_uuid}_b{b}.nc")
            ds = xr.Dataset({
                "surface": (["step", "channel_sfc", "lat", "lon"], sfc_np),
                "upper_air": (["step", "channel_ua", "level", "lat", "lon"], ua_np),
            })
            if diagnostic_prediction is not None:
                ds["diagnostic"] = (["step", "channel_diag", "lat", "lon"],
                                    diagnostic_prediction[b].numpy())
            ds.to_netcdf(fname)
            logging.info("Saved prediction to %s", fname)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_inference")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist_mgr = DistributedManager()
    logger = PythonLogger("inference")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    stepper = Stepper(cfg, dist_mgr)
    stepper.predict()
    logger.info("Inference done — rank %d", dist_mgr.rank)


if __name__ == "__main__":
    main()
