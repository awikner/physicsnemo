# SPDX-License-Identifier: Apache-2.0
"""Load Zhixing's SFNO "v11" as a baseline, from the local checkpoint copy.

v11 is 58 -> 53: 52 prognostic state + 6 forcing in, 52 state + pr_6h out. Its
forcing set differs from ours, so the inputs are rebuilt here to v11's own
conventions, recovered from its packaged metadata:

  * `sic`  - a BINARY sea-ice indicator taken from the mask of
             boundary_data/sic_masked_6h*.nc (mask -> 0, unmasked -> 1). Verified
             against v11's own stats: mean 0.12498/0.33070 vs 0.12497/0.33068.
  * `sst`  - land filled with 271.35 K (`sst_land_fill_K` in v11's metadata), NOT
             the field mean our packer uses.
  * `rsdt` - v11 recomputed this astronomically (`rsdt_method: astronomical`).
             We substitute PlaSim's stored rsdt, whose climatology is close
             (303.7 vs 300.98 W/m2) but not identical. This is an approximation
             and is the main caveat on the baseline.
"""
import json, os
import numpy as np
import torch

V11_DIR = "/work2/11079/aasch/stampede3/shared/sfno_weights/v11"
V11_STATS = "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128_zgplev_v11/stats"
BOUNDARY = "/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52/boundary_data"
SST_LAND_FILL_K = 271.35
V11_FORCING = ["lsm", "sg", "z0", "sst", "rsdt", "sic"]


def build_v11(device):
    from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet
    cfg = json.load(open(os.path.join(V11_DIR, "config.json")))
    net = SphericalFourierNeuralOperatorNet(
        inp_shape=(cfg["img_shape_x"], cfg["img_shape_y"]),
        out_shape=(cfg["img_shape_x"], cfg["img_shape_y"]),
        inp_chans=cfg["N_in_channels"], out_chans=cfg["N_out_channels"],
        embed_dim=cfg["embed_dim"], num_layers=cfg["num_layers"],
        scale_factor=cfg["scale_factor"], filter_type=cfg["filter_type"],
        operator_type=cfg["operator_type"], spectral_transform=cfg["spectral_transform"],
        model_grid_type=cfg["model_grid_type"], sht_grid_type=cfg["sht_grid_type"],
        pos_embed=cfg["pos_embed"], big_skip=cfg["big_skip"], use_mlp=cfg["use_mlp"],
        mlp_ratio=cfg["mlp_ratio"], normalization_layer=cfg["normalization_layer"],
        activation_function=cfg["activation_function"],
        hard_thresholding_fraction=cfg["hard_thresholding_fraction"],
        complex_activation=cfg["complex_activation"], separable=cfg["separable"],
        encoder_layers=cfg["encoder_layers"],
    ).to(device).eval()

    ck = torch.load(os.path.join(V11_DIR, "training_checkpoints",
                                 "best_ckpt_ema_mp0.tar"), map_location="cpu",
                    weights_only=False)
    sd = ck["model_state"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    return net, cfg, list(missing), list(unexpected)


def v11_stats(device):
    m = np.load(os.path.join(V11_DIR, "global_means.npy"))
    s = np.load(os.path.join(V11_DIR, "global_stds.npy"))
    fm = np.load(os.path.join(V11_STATS, "forcing_global_means.npy"))
    fs = np.load(os.path.join(V11_STATS, "forcing_global_stds.npy"))
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return t(m), t(s), t(fm), t(fs)


def _leap(y):
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def v11_cyclic(year):
    """Binary sic and 271.35-filled sst for one year, [T,64,128] each."""
    import netCDF4 as nc
    suf = "_leap" if _leap(year) else ""
    sicv = nc.Dataset(os.path.join(BOUNDARY, f"sic_masked_6h{suf}.nc")).variables["sic"][:]
    sic = np.where(np.ma.getmaskarray(sicv), 0.0, 1.0).astype(np.float32)
    sstv = nc.Dataset(os.path.join(BOUNDARY, f"sst_masked_6h{suf}.nc")).variables["sst"][:]
    sst = np.array(sstv.filled(np.nan), dtype=np.float32)
    sst[~np.isfinite(sst)] = SST_LAND_FILL_K
    return sic, sst
