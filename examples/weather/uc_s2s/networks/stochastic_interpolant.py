import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from tqdm.auto import tqdm
from torch import nn, einsum, optim
from torch.nn import functional as F
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .diffusion import ConditionalDiffusionModel
from .diffusion import ConUNet_1degV2
from .vae import VAE



# class DriftScheduler(nn.Module):
#     def __init__(self,
#                  num_refinement_steps,  # this corresponds to physical time steps
#                  num_train_steps=None,  # number of training steps
#                  integrator='em', 
#                  sigma_coef=1.0,  
#                  beta_fn = "t",
#                  use_gF = False,
#                  antithetic_sampling=True,
#                  sigma_sample=None,
#                  ndim=2,
#                  ):
#         super(DriftScheduler, self).__init__()
        

class StochasticInterpolant(ConditionalDiffusionModel):
    def __init__(self, path='linear', gamma_type='brownian', **kwargs):
        super(StochasticInterpolant, self).__init__(**kwargs)
        self.path = path
        self.gamma_type = gamma_type
   

    def alpha(self, t):
        return 1-t
    
    def alpha_dot(self, t):
        return -1.0
    
    
    def beta(self, t):
        return t
    
    def beta_dot(self, t):
        return 1.0
    
    def gamma(self, t): 
        if self.gamma_type == 'brownian':
            return torch.sqrt(t*(1-t))
        elif self.gamma_type == 'a-brownian':
            return torch.sqrt(self.aval*t*(1-t))
        elif self.gamma_type == 'zero':
            return torch.zeros_like(t)
        elif self.gamma_type == 'bsquared':
            return t*(1-t)
        elif self.gamma_type == 'sinesquared':
            return torch.sin(math.pi * t)**2
        elif self.gamma_type == 'sigmoid':  
            f = torch.tensor(10.0)
            return torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f*(t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
        elif self.gamma_type == None:
            return torch.zeros(1) ### no gamma
        else:
            raise NotImplementedError("The gamma you specified is not implemented.")
    
    
    def gamma_dot(self, t):
        if self.gamma_type == 'brownian':
            denom = torch.clamp(torch.sqrt(t*(1-t)), min=1e-3)
            return denom
        elif self.gamma_type == 'a-brownian':
            return (1/(2*torch.sqrt(self.aval*t*(1-t)))) * self.aval*(1 -2*t)
        elif self.gamma_type == 'zero':
            return torch.zeros_like(t)
        elif self.gamma_type == 'bsquared':
            return 1 -2*t
        elif self.gamma_type == 'sinesquared':
            return 2*math.pi*torch.sin(math.pi * t)*torch.cos(math.pi*t)
        elif self.gamma_type == 'sigmoid':  
            f = torch.tensor(10.0)
            return (-f)*( 1 - torch.sigmoid(-1 + f*(t - (1/2))) )*torch.sigmoid(-1 + f*(t - (1/2)))  + f*(1 - torch.sigmoid(1 + f*(t - (1/2)))  )*torch.sigmoid(1 + f*(t - (1/2)))
        elif self.gamma_type == None:
            return torch.zeros(1) ### no gamma
        else:
            raise NotImplementedError("The gamma you specified is not implemented.")
        
    # def image_sq_norm(self, x):
    #     return x.pow(2).sum(-1).sum(-1).sum(-1)
    
    def image_sq_norm(self, x):
        return x.pow(2).mean(-1).mean(-1).mean(-1)

    def I(self, x0, x1, t):
        return self.alpha(t) * x0 + self.beta(t) * x1
    
    def dIdt(self, x0, x1, t):
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1       
        
    
    def source_distribution(self, x: torch.Tensor, sigma: float = 1.0):
        """Return N(0, sigma*I) with the same shape as `x`.

        Setting sigma to the empirical std of the residual distribution ensures
        the source and target have matched scale, so the SI flow transports
        meaningful variance rather than mapping large noise -> near-zero residual.
        """
        normal = torch.distributions.Normal(
            loc=torch.zeros_like(x),
            scale=torch.full_like(x, sigma),
        )
        dist = torch.distributions.Independent(normal, reinterpreted_batch_ndims=x.dim() - 1)
        return dist

    def sample_from_source(self, x: torch.Tensor,
                           n_samples: int = 1,
                           reparam: bool = True,
                           sigma: float = 1.0) -> torch.Tensor:
        """Draw samples from N(0, sigma*I) with the same shape as `x`.

        Args:
            x:         reference tensor for shape/device/dtype.
            n_samples: number of independent samples per batch item.
            reparam:   use rsample (reparameterized) if True.
            sigma:     source distribution std; match to residual std for best spread.
        """
        dist = self.source_distribution(x, sigma=sigma)
        if n_samples is None or n_samples <= 1:
            return dist.rsample() if reparam else dist.sample()
        samples = dist.rsample((n_samples,)) if reparam else dist.sample((n_samples,))
        return samples
        
    
    def training_step(self, surface_in, constant_boundary, varying_boundary,
                      upper_air_in, target_surface_in=None, target_upper_air=None,
                      train = True, plot_freq = 0,
                      plot_path = "noise_comparison.png", iter=0, lower_upper: tuple = (0.0001, 0.9999),
                      plot_scatter: bool = False, scatter_path: str = "scatter_pred_gt.png"):

        """Single diffusion training step. Returns RMSE loss.

        The SI is trained on the residual between the ground-truth t+1 latent
        (encoded by the frozen deterministic Pangu encoder) and Pangu's own
        smoothed mean-latent prediction, i.e. what Pangu's encoder misses.
        """
        lower, upper = lower_upper[0], lower_upper[1]

        if target_surface_in is None or target_upper_air is None:
            raise ValueError(
                "training_step requires target_surface_in/target_upper_air for the "
                "x_true diagnostic print."
            )

        surface = self._prepare_surface(surface_in, constant_boundary, varying_boundary)
        target_surface = self._prepare_surface(target_surface_in, constant_boundary, varying_boundary)
        B = surface.size(0)
        device = surface.device
        ts  = lower + (upper - lower)*torch.rand(size=(B,), device=device)

        # Encode stochastic condition (VAE) and deterministic features
        z = self._encode_vae(surface, upper_air_in)
        x, skip = self._encode_det(surface, upper_air_in, train=True)

        # Ground-truth t+1 latent, encoded through the same frozen Pangu path.
        with torch.no_grad():
            x_true, _ = self._encode_det(target_surface, target_upper_air, train=False)

        # Train SI to map N(0, x_sigma) -> x (Pangu deterministic forecast).
        # The SDE integration (brownian gamma) creates stochastic spread at inference.
        # x (Pangu latent) is also concatenated with the path point so the UNet has
        # full state information; this requires dim_in=20 in the UNet.
        x_sigma = x.std().item()

        # if torch.rand(1).item() < 0.01:
        #     print(f"[SI spread diag] x std={x_sigma:.4f}  x_true std={x_true.std().item():.4f}")

        base = self.sample_from_source(x, n_samples=B, reparam=True, sigma=x_sigma)
        assert base.shape == x.shape, f"Shape mismatch: base {base.shape} vs x {x.shape}."

        target = x
        It = self.I(x0=base, x1=target, t=ts)
        dIdt = self.dIdt(x0=base, x1=target, t=ts)
        It_p = It + self.gamma(ts)*torch.randn_like(It).to(device)
        It_m = It - self.gamma(ts)*torch.randn_like(It).to(device)

        assert not torch.isnan(It_p).any(), f"It_p has NaN"
        assert not torch.isnan(z).any(), f"z has NaN"

        # Concatenate Pangu latent x as additional channels so the UNet can
        # condition its drift on the full current atmospheric state.
        noise    = torch.randn_like(base).to(device)
        drift_p  = self.unet(torch.cat([It_p, x], dim=1), z, ts)
        drift_m  = self.unet(torch.cat([It_m, x], dim=1), z, ts)

        target_p = dIdt - noise * self.gamma_dot(ts)
        target_m = dIdt + noise * self.gamma_dot(ts)

        loss_p= self.image_sq_norm(drift_p - target_p).mean()
        loss_m= self.image_sq_norm(drift_m - target_m).mean()

        loss = loss_p + loss_m

        if plot_scatter:
            self._plot_scatter(drift_p, target_p, iter=iter, save_path=scatter_path)

        return loss

    def _plot_scatter(self, pred: torch.Tensor, target: torch.Tensor,
                      iter: int = 0, save_path: str = "scatter_pred_gt.png",
                      max_points: int = 4096) -> None:
        """Scatter plot of predicted vs ground-truth drift for the first batch item."""
        pred_np   = pred[0].detach().cpu().float().flatten().numpy()
        target_np = target[0].detach().cpu().float().flatten().numpy()

        if len(pred_np) > max_points:
            idx = np.random.default_rng(iter).choice(len(pred_np), max_points, replace=False)
            pred_np, target_np = pred_np[idx], target_np[idx]

        corr = float(np.corrcoef(pred_np, target_np)[0, 1]) if len(pred_np) > 1 else float('nan')

        lim = max(float(np.abs(target_np).max()), float(np.abs(pred_np).max())) * 1.05
        lim = lim if lim > 0 else 1.0

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(target_np, pred_np, s=2, alpha=0.25, linewidths=0, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.2, label='y = x')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Ground truth drift")
        ax.set_ylabel("Predicted drift")
        ax.set_title(f"Pred vs GT drift  iter={iter}  r={corr:.3f}")
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    
    def step_forward(self, x, drift, dt, g, eps = torch.tensor(2.0)):
        """
        Perform one step of the forward SDE: x_{t+dt} = x_t + drift*dt + squire(dt)*gamma(t)*eps, where dW ~ N(0, dt).
        """
        dW = torch.sqrt(dt)*torch.randn_like(x, device=x.device)
    
        x = x + dt*drift + torch.sqrt(2*eps) * dW
        return x
    
    
        
    @torch.no_grad()
    def generate(
        self,
        model_diff = None,
        z: torch.Tensor= None,
        num_samples: int = 1,
        sample_shape: tuple | None = None,
        device =  None,
        x = None,
        temperature: float = 1.0,
        mc_dropout: bool = False,
        source_sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate images conditioned on a source image's VAE encoding.

        Args:
            z: conditional information
            num_samples:     how many samples to draw per condition
            x: the latent space from deterministic encoder
            temperature:     additional scale on top of source_sigma; >1.0 increases spread
            mc_dropout:      if True, keep UNet dropout active during sampling for MC dropout
            source_sigma:    std of source N(0, sigma*I); set to empirical residual std
                             so source and target distributions have matched scale
        """
        z = z.repeat_interleave(num_samples, dim=0)
        x = x.repeat_interleave(num_samples, dim=0)
        if sample_shape is None:
            C, H, W = z.shape[1:]
        else:
            if len(sample_shape) == 4:
                C, H, W = sample_shape[1:]
            elif len(sample_shape) == 3:
                C, H, W = sample_shape
            else:
                raise ValueError(f"sample_shape must have 3 or 4 dims, got {sample_shape}")

        shape = (z.size(0), C, H, W)

        output = self.sample(model=model_diff, shape=shape, x=x, cond=z, device=device,
                             temperature=temperature, mc_dropout=mc_dropout,
                             source_sigma=source_sigma)
        return output


    @torch.no_grad()
    def sample(
        self,
        model: ConUNet_1degV2,
        shape: tuple,
        cond: torch.Tensor,
        device: torch.device,
        T: int = 15,
        start_end  = (0, 1),
        x = None,
        show_progress: bool = True,
        temperature: float = 1.0,
        mc_dropout: bool = False,
        source_sigma: float = 1.0,
    ) -> torch.Tensor:
        """Full reverse diffusion loop."""
        x_pangu = x  # keep Pangu latent for state-conditioning at every step

        x_t = self.sample_from_source(x_pangu, n_samples=1, reparam=True,
                                      sigma=source_sigma).to(device) * temperature

        if mc_dropout:
            model.train()

        self.start, self.end = start_end[0], start_end[1]
        self.ts = torch.linspace(self.start, self.end, T)

        for ii, t in enumerate(self.ts[:-1]):
            t_current = self.ts[ii]
            t_next    = self.ts[ii + 1]
            dt = t_next - t_current
            x_t = self.p_sample(model=model, x_t=x_t, t=t, cond=cond, dt=dt,
                                 x_pangu=x_pangu)

        if mc_dropout:
            model.eval()

        return x_t

    @torch.no_grad()
    def p_sample(
        self,
        model: ConUNet_1degV2,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
        x_pangu: torch.Tensor = None,
        eps: float = 2.0,
    ) -> torch.Tensor:

        t = t.to(x_t.device)
        dt = dt.to(x_t.device)
        # Concatenate Pangu latent as state-conditioning channels (matches training).
        model_input = torch.cat([x_t, x_pangu], dim=1) if x_pangu is not None else x_t
        drift = model(model_input, cond, t)
        y  = self.step_forward(x_t, drift, dt, self.gamma(t), eps=torch.tensor(eps))

        return y


    def crps_finetune_step(self, surface_in, constant_boundary, varying_boundary,
                           upper_air_in, target_surface, target_upper_air,
                           latitudes, num_samples: int = 2, target_diagnostic=None):
        """Generate num_samples ensemble members and compute CRPS loss in output space.

        SI/UNet and encoders stay frozen (no_grad). Only the Pangu decoder
        (upsample, layer4, patchrecovery2d, patchrecovery3d) receives gradients.
        """
        from utils.losses import Latitude_weighted_CRPSLoss

        surface = self._prepare_surface(surface_in, constant_boundary, varying_boundary)
        B = surface.size(0)
        device = surface.device

        # Encode — all frozen, no grad needed through SI
        with torch.no_grad():
            z = self._encode_vae(surface, upper_air_in)
            x, skip = self._encode_det(surface, upper_air_in, train=False)
            # x: [B, Pl, seq, embed*scale], skip: [B, seq, embed]
            Pl_det = x.shape[1]
            Lat, Lon = self.downscale_resolution_det[1], self.downscale_resolution_det[2]

            source_sigma = float(getattr(self.params, 'source_sigma', x.std().item())) * 1.5
            x_si = self.generate(
                model_diff=self.unet,
                z=z,
                x=x,
                num_samples=num_samples,
                sample_shape=x.shape,
                device=device,
                temperature=1.5,
                source_sigma=source_sigma,
            )  # [B*num_samples, Pl, seq, embed*scale]

        BS = x_si.shape[0]  # B * num_samples

        # Decoder — gradients flow through these layers only
        Pl_est, Lat_est, Lon_est = self.model_det.EST_input_resolution
        x_dec = x_si.reshape(BS, -1, 240 * self.params.updown_scale_factor)
        x_dec = self.model_det.upsample(x_dec)
        x_dec = self.model_det.layer4(x_dec, train=False)
        skip_expanded = skip.repeat_interleave(num_samples, dim=0)
        output = torch.cat([x_dec, skip_expanded], dim=-1)
        output = output.transpose(1, 2).reshape(BS, -1, Pl_est, Lat_est, Lon_est)

        output_surface = output[:, :, -1, :, :]
        output_upper_air = output[:, :, :-1, :, :]
        output_2D = self.model_det.patchrecovery2d(output_surface)
        pred_surface = output_2D[:, self.surface_prognostic_idxs]
        pred_upper_air = self.model_det.patchrecovery3d(output_upper_air)
        pred_diagnostic = output_2D[:, self.num_surface_vars:self.num_surface_vars + self.num_diagnostic_vars].reshape(
            BS, -1, pred_surface.shape[-2], pred_surface.shape[-1])

        # CRPS — targets repeated to match [B*num_samples, ...]
        target_sfc_rep = target_surface.repeat_interleave(num_samples, dim=0)
        target_ua_rep = target_upper_air.repeat_interleave(num_samples, dim=0)
        crps_loss_fn = Latitude_weighted_CRPSLoss(latitudes.to(device), num_ensemble_members=num_samples)

        crps_sfc = crps_loss_fn(pred_surface, target_sfc_rep)
        crps_ua  = crps_loss_fn(pred_upper_air, target_ua_rep)
        loss = 0.2 * (crps_sfc + crps_ua)

        crps_diag = torch.tensor(0.0, device=device)
        precip_rmse = torch.tensor(0.0, device=device)

        if target_diagnostic is not None and self.num_diagnostic_vars > 0:
            target_diag_rep = target_diagnostic.repeat_interleave(num_samples, dim=0)
            crps_diag = crps_loss_fn(pred_diagnostic, target_diag_rep)
            loss = loss + 0.8 * crps_diag

            # RMSE on precipitation to prevent dry-grid-point dominance in CRPS.
            # Averaged over ensemble members so each member is penalised individually.
            precip_idxs = [i for i, v in enumerate(self.params.diagnostic_variables)
                           if 'precipitation' in v]
            if precip_idxs:
                lat_w = torch.cos(torch.tensor(self.params.lat, dtype=torch.float32,
                                               device=device) * math.pi / 180.0)
                lat_w = (lat_w / lat_w.mean()).view(1, 1, -1, 1)
                for m in range(num_samples):
                    pred_precip = pred_diagnostic[m::num_samples][:, precip_idxs]
                    tgt_precip  = target_diagnostic[:, precip_idxs]
                    precip_rmse = precip_rmse + torch.sqrt((lat_w * (pred_precip - tgt_precip) ** 2).mean())
                    loss = loss + 0.1 * precip_rmse

        import logging
        logging.info(
            f"[CRPS finetune] sfc={crps_sfc.item():.4f}  ua={crps_ua.item():.4f}  "
            f"diag={crps_diag.item():.4f}  precip_rmse={precip_rmse.item():.4f}  "
            f"total={loss.item():.4f}"
        )

        return loss, crps_sfc, crps_ua, crps_diag, precip_rmse

    def prediction(self, surface_in, constant_boundary,
                   varying_boundary, upper_air_in,
                   num_samples = 1, device = None,
                   seed = None,
                   temperature: float = 1.5,
                   mc_dropout: bool = False,
                   use_gaussian_latent: bool = False,
                   gaussian_latent_std: float = 1.0):
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        surface_in = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)

        B = surface_in.size(0)
        device = surface_in.device
        self.scheduler_diff.to(device)
        
        ###############encoder 2 start ########################
        # 1. Encode condition
        #######VAE ENCODER START ########

        z  = self._encode_vae(surface_in, upper_air_in)

        # Enable Pangu dropout for MC dropout inference (parameters stay frozen).
        if mc_dropout:
            self.model_det.train()

        ###############encoder 1 start (deterministic) ########################
        surface_det = self.model_det.patchembed2d(surface_in)
        upper_air_det = self.model_det.patchembed3d(upper_air_in)
        x = torch.concat([upper_air_det, surface_det.unsqueeze(2)], dim=2)
        _, _, Pl_det, Lat, Lon = x.shape

        x, skip = self._encode_det(surface_in, upper_air_in, train=False)
        target_latent_shape = x.shape

        # SI is trained with source N(0, x.std()), target = x, so use x.std() here.
        # Can be overridden via config source_sigma for tuning.
        source_sigma = float(getattr(self.params, 'source_sigma', x.std().item())) * 1.5

        if use_gaussian_latent:
            # Diagnostic bypass: skip the SI entirely and perturb the Pangu
            # deterministic latent with isotropic Gaussian noise. If this produces
            # more spread than the SI, it means the SI is failing to generate
            # diversity (training collapsed toward the deterministic mean).
            x_si = x + gaussian_latent_std * x.std() * torch.randn_like(x)
        else:
            x_si = self.generate(
                model_diff=self.unet,
                z=z,
                x=x,
                num_samples=num_samples,
                sample_shape=target_latent_shape,
                device=device,
                temperature=temperature,
                mc_dropout=mc_dropout,
                source_sigma=source_sigma,
            )

        # SI targets x (Pangu latent); SDE integration creates spread around x.
        x = x_si
        BS = x.shape[0]  # B * num_samples

        x = x.reshape(BS, -1, 240*self.params.updown_scale_factor)

        ######## DETERMINISTIC DECODER START ######
        x = self.model_det.upsample(x)
        x = self.model_det.layer4(x, train=False)
        # Expand skip so each sample gets its own copy of the skip connection.
        skip_expanded = skip.repeat_interleave(num_samples, dim=0)
        output = torch.concat([x, skip_expanded], dim=-1)
        output = output.transpose(1, 2).reshape(BS, -1, Pl_det, Lat, Lon)
        output_surface = output[:, :, -1, :, :]
        output_upper_air = output[:, :, :-1, :, :]
        output_2D = self.model_det.patchrecovery2d(output_surface)
        output_surface = output_2D[:, self.surface_prognostic_idxs]
        # if torch.isnan(output_surface).any():
        #     print(f"[NaN check] output_surface (after patchrecovery2d) has NaN: {torch.isnan(output_surface).sum().item()} NaNs")
        # else:
        #     print("[NaN check] output_surface (after patchrecovery2d): no NaN")

        output_upper_air = self.model_det.patchrecovery3d(output_upper_air)
        output_diagnostic = output_2D[:, self.num_surface_vars:self.num_surface_vars + self.num_diagnostic_vars].reshape(
            output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1])

        if mc_dropout:
            self.model_det.eval()

        return output_surface, output_upper_air, output_diagnostic 
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        