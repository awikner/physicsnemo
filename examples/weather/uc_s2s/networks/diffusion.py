
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from functools import partial
from tqdm.auto import tqdm
from torch import nn, einsum, optim
from torch.nn import functional as F
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint

Tensor = torch.Tensor




def Downsample_1deg(dim_in, dim_out, scale=2):
    class DownsampleModule(nn.Module):
        def __init__(self):
            super(DownsampleModule, self).__init__()
            self.maxpool = nn.MaxPool2d(scale)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        def forward(self, x):
            x = self.maxpool(x) 
            x = self.conv(x)
            return x

    return DownsampleModule()

def Upsample_1deg(dim_in, dim_out, scale=2):
    class UpsampleModule(nn.Module):
        def __init__(self):
            super(UpsampleModule, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.upsample(x)
            x = self.conv(x)
            return x  

    return UpsampleModule()

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Ensure time is on the correct device
        device = time.device
        #device = next(self.parameters()).device
        #time = time.to(device)
        # Ensure time is at least 1-dimensional
        if time.dim() == 0:
            time = time.unsqueeze(0)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * 1000.
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift!=None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x
    
class UpProject(nn.Module):
    def __init__(self, dim=10, time_emb_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.mlp_t = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim * 2))
                        if time_emb_dim is not None else None)

    def forward(self, x, time_emb=None):  # x: (N,10,1035,384)
        x = F.interpolate(x, size=( 4050, 480), mode="bilinear", align_corners=False)
        x = self.conv(x)
        if self.mlp_t is not None and time_emb is not None:
            time_emb = self.mlp_t(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            x = x * (scale + 1) + shift
        return x
    
    
class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8, dropout=0.1):
        super().__init__()
        self.out_channels = dim_out    
        self.mlp_t = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
                        if time_emb_dim is not None else None)
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.dropout = nn.Dropout(dropout)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift_t = None
        if self.mlp_t is not None and time_emb is not None:
            time_emb = self.mlp_t(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift_t = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift_t)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)

# class ResnetBlock(nn.Module):
#     def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
#         super().__init__()
#         self.out_channels = dim_out    
#         self.mlp_t = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
#                         if time_emb_dim is not None else None)
#         self.block1 = Block(dim_in, dim_out, groups=groups)
#         self.block2 = Block(dim_out, dim_out, groups=groups)
#         self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         scale_shift_t = None
#         if self.mlp_t is not None and time_emb is not None:
#             time_emb = self.mlp_t(time_emb)
#             time_emb = rearrange(time_emb, "b c -> b c 1 1")
#             scale_shift_t = time_emb.chunk(2, dim=1)
#         h = self.block1(x, scale_shift=scale_shift_t)
#         h = self.block2(h)
#         return h + self.res_conv(x)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class CrossAttentionConditionFuse(nn.Module):
    """Fuse conditioning feature maps into model activations with pooled cross-attention."""
    def __init__(self, dim_x, dim_cond, heads=4, dim_head=32, pool_size=(8, 8)):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.pool_size = pool_size

        self.to_q = nn.Conv2d(dim_x, inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim_cond, inner_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim_cond, inner_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim_x, kernel_size=1)
        self.norm_x = nn.GroupNorm(1, dim_x)
        self.norm_c = nn.GroupNorm(1, dim_cond)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, cond):
        b, _, h, w = x.shape

        x_pooled = F.adaptive_avg_pool2d(self.norm_x(x), self.pool_size)
        c_pooled = F.adaptive_avg_pool2d(self.norm_c(cond), self.pool_size)

        q = self.to_q(x_pooled)
        k = self.to_k(c_pooled)
        v = self.to_v(c_pooled)

        q_h, q_w = q.shape[-2], q.shape[-1]
        n_tokens = q_h * q_w
        q = q.view(b, self.heads, self.dim_head, n_tokens).permute(0, 1, 3, 2)
        k = k.view(b, self.heads, self.dim_head, n_tokens).permute(0, 1, 3, 2)
        v = v.view(b, self.heads, self.dim_head, n_tokens).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.heads * self.dim_head, q_h, q_w)
        out = self.to_out(out)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return x + self.gate * out
    

class ConUNet_1degV2(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, c, c_mults=(1, 2, 4, 8),
                    resnet_block_groups=4, scale = [2,2,2],
                    is_guide=False, drop_prob=0.1, dropout=0.1,
                    checkpointing=0, use_reentrant=False):
        super().__init__()

        self.checkpointing = checkpointing
        self.use_reentrant = use_reentrant
        self.init_conv = nn.Conv2d(dim_in, c, 1, padding=0)
        dims = [c*x for x in c_mults]
        self.drop_prob = drop_prob
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups, dropout=dropout)
        # block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # time embeddings
        time_dim = c * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(d_in, d_in, time_emb_dim=time_dim),
                        block_klass(d_in, d_in, time_emb_dim=time_dim),
                        Downsample_1deg(d_in, d_out, scale = scale[ind])
                        if not is_last
                        else nn.Conv2d(d_in, d_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                        block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                        Upsample_1deg(d_out, d_in, scale = scale[-ind-1])
                        if not is_last
                        else nn.Conv2d(d_out, d_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(c * 2, c, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(c, dim_out, 1)
        self.cond_to_c = nn.Conv2d(dim_cond, c, kernel_size=1)
        self.cross_attn_in = CrossAttentionConditionFuse(dim_x=c, dim_cond=c, heads=4, dim_head=32, pool_size=(8, 8))
        self.cond_to_mid = nn.Conv2d(c, mid_dim, kernel_size=1)
        self.cross_attn_mid = CrossAttentionConditionFuse(dim_x=mid_dim, dim_cond=mid_dim, heads=4, dim_head=32, pool_size=(8, 8))


    def _maybe_checkpoint(self, block, x, t):
        if self.checkpointing > 0 and self.training:
            return checkpoint(block, x, t, use_reentrant=self.use_reentrant)
        return block(x, t)

    def forward(self, x, cond, time):

        t = self.time_mlp(time)
        # print("x shape in diffusion ", x.shape) # 2, 10, 1035, 384
        cond_c = self.cond_to_c(cond)
        # print("cond_c  shape", cond_c.shape) # 2, 64, 1035, 384
        # print(" cond shape after projection ", cond.shape) #1, 2, 1035, 384
        # print("x shape after projection ", x.shape) # 2, 10, 1035, 384
        x = self.init_conv(x)
        x = self.cross_attn_in(x, cond_c)
        r = x.clone()
        h = []
        for block1, block2, downsample in self.downs:
            x = self._maybe_checkpoint(block1, x, t)
            h.append(x)
            x = self._maybe_checkpoint(block2, x, t)
            h.append(x)
            scale_factor = 2
            H, W = x.shape[2], x.shape[3]
            pad_h = (scale_factor - H % scale_factor) % scale_factor
            pad_w = (scale_factor - W % scale_factor) % scale_factor
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            x = downsample(x)

        x = self._maybe_checkpoint(self.mid_block1, x, t)
        cond_mid = self.cond_to_mid(cond_c)
        x = self.cross_attn_mid(x, cond_mid)
        x = self._maybe_checkpoint(self.mid_block2, x, t)

        for block1, block2, upsample in self.ups:
            skip = h.pop()
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                x = x[:, :, :skip.shape[2], :skip.shape[3]]
            x = torch.cat((x, skip), dim=1)
            x = self._maybe_checkpoint(block1, x, t)
            skip = h.pop()
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                x = x[:, :, :skip.shape[2], :skip.shape[3]]
            x = torch.cat((x, skip), dim=1)
            x = self._maybe_checkpoint(block2, x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[:, :, :r.shape[2], :r.shape[3]]
        return x



# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """
    Linear beta schedule and forward / reverse diffusion utilities.
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_bar", alphas_bar)
        self.register("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())

    def register(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    def to(self, device):
        for attr in ["betas", "alphas", "alphas_bar",
                     "sqrt_alphas_bar", "sqrt_one_minus_alphas_bar"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t given x_0 and t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: ConUNet_1degV2,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: sample x_{t-1} given x_t."""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_recip_a = (1.0 / self.alphas[t].sqrt()).view(-1, 1, 1, 1)
        eps_pred = model(x_t, cond, t)
        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_a * (x_t - betas_t / sqrt_1mab * eps_pred)

        torch.isnan(eps_pred).any() and print(f"NaN detected in eps_pred at t={t.tolist()}")
        torch.isnan(x_t).any() and print(f"NaN detected in x_t at t={t.tolist()}")
        if torch.isnan(mean).any():
            print(f"NaN detected in p_sample mean at t={t.tolist()}")
            print(f"NaN ")
            raise ValueError(f"NaN detected in p_sample mean at t={t.tolist()}")

        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        
        return mean + betas_t.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: ConUNet_1degV2,
        shape: tuple,
        cond: torch.Tensor,
        device: torch.device,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Full reverse diffusion loop using self.T steps."""
        x = torch.randn(shape, device=device)
        disable_bar = not show_progress
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            disable_bar = disable_bar or torch.distributed.get_rank() != 0

        steps = tqdm(
            reversed(range(self.T)),
            total=self.T,
            desc="Diffusion sampling",
            leave=False,
            disable=disable_bar,
        )
        ts = time.time()
        for step in steps:
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            x = self.p_sample(model, x, t, cond)
        print(f"Sampling took {time.time() - ts:.2f} seconds")
        return x

# ---------------------------------------------------------------------------
# DDIM sampler  (Song et al., 2020 — https://arxiv.org/abs/2010.02502)
# ---------------------------------------------------------------------------

class DDIMScheduler:
    """
    DDIM accelerated sampler.

    Reuses the same linear-beta noise schedule as DDPMScheduler but iterates
    over a short subsequence of timesteps, making sampling 10-50x faster.

    Args:
        T: total training timesteps (must match the trained DDPMScheduler).
        beta_start / beta_end: same values used during training.
        num_inference_steps: how many denoising steps to use at sampling time
            (e.g. 50 instead of 1000).  Can also be overridden per-call in
            ``sample()``.
        eta: controls stochasticity.  0 → fully deterministic (original DDIM);
            1 → recovers DDPM variance; values in between interpolate.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.T = T
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_bar", alphas_bar)
        self.register("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())

    def register(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    def to(self, device):
        for attr in ["betas", "alphas", "alphas_bar",
                     "sqrt_alphas_bar", "sqrt_one_minus_alphas_bar"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ------------------------------------------------------------------
    # Forward process (identical to DDPM — used during training)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t given x_0 and t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # DDIM reverse step
    # ------------------------------------------------------------------

    def _make_timestep_sequence(self, num_inference_steps: int) -> list[int]:
        """Return a decreasing list of training-timestep indices of length num_inference_steps."""
        # Evenly spaced in [0, T-1], then reversed so we go from noisy to clean.
        step = self.T // num_inference_steps
        timesteps = list(range(0, self.T, step))[-num_inference_steps:]
        return list(reversed(timesteps))  # e.g. [999, 979, ..., 19, 0] for T=1000, S=50

    @torch.no_grad()
    def ddim_step(
        self,
        model: ConUNet_1degV2,
        x_t: torch.Tensor,
        t_idx: int,       # index into the subsequence (0 = first/noisiest step)
        t_next_idx: int | None,  # index of the next (cleaner) timestep; None at the last step
        timesteps: list[int],
        cond: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """
        One DDIM reverse step: x_{t} -> x_{t_prev}.

        The DDIM update (eq. 12 in the paper):
            predicted_x0  = (x_t - sqrt(1-ᾱ_t) * eps) / sqrt(ᾱ_t)
            sigma_t       = eta * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
            direction     = sqrt(1 - ᾱ_{t-1} - sigma_t²) * eps
            x_{t-1}       = sqrt(ᾱ_{t-1}) * predicted_x0 + direction + sigma_t * noise
        """
        t = timesteps[t_idx]
        t_prev = timesteps[t_next_idx] if t_next_idx is not None else -1

        B = x_t.shape[0]
        device = x_t.device
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        # Predict noise
        eps_pred = model(x_t, cond, t_batch)

        # ᾱ values
        ab_t = self.alphas_bar[t]
        ab_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.ones(1, device=device)

        sqrt_ab_t    = ab_t.sqrt()
        sqrt_1mab_t  = (1.0 - ab_t).sqrt()
        sqrt_ab_prev = ab_prev.sqrt()

        # Predicted clean sample (clamp for stability)
        predicted_x0 = (x_t - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        predicted_x0 = predicted_x0.clamp(-5.0, 5.0)

        # DDIM sigma (eq. 16): eta=0 → deterministic, eta=1 → DDPM
        # sigma_t = eta * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
        # Guard against the edge case where ab_t > ab_prev (shouldn't happen with linear schedule)
        ratio = (ab_t / ab_prev).clamp(max=1.0)
        sigma_t = eta * ((1.0 - ab_prev) / (1.0 - ab_t)).sqrt() * (1.0 - ratio).sqrt()

        # Direction toward x_t
        coeff_dir = (1.0 - ab_prev - sigma_t ** 2).clamp(min=0.0).sqrt()
        direction = coeff_dir * eps_pred

        # Optional stochastic noise term
        noise = sigma_t * torch.randn_like(x_t) if (eta > 0.0 and t_prev >= 0) else 0.0

        x_prev = sqrt_ab_prev * predicted_x0 + direction + noise
        return x_prev

    # ------------------------------------------------------------------
    # Full DDIM reverse loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: ConUNet_1degV2,
        shape: tuple,
        cond: torch.Tensor,
        device: torch.device,
        num_inference_steps: int | None = None,
        eta: float | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        DDIM reverse diffusion loop.

        Args:
            model:               noise-predicting network.
            shape:               output shape (B, C, H, W).
            cond:                conditioning tensor.
            device:              target device.
            num_inference_steps: overrides the instance default if provided.
            eta:                 overrides the instance default if provided.
            show_progress:       whether to show a tqdm bar.
        Returns:
            Denoised sample of `shape`.
        """
        S   = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        eta = eta if eta is not None else self.eta

        timesteps = self._make_timestep_sequence(S)

        x = torch.randn(shape, device=device)

        disable_bar = not show_progress
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            disable_bar = disable_bar or torch.distributed.get_rank() != 0

        steps = tqdm(
            range(len(timesteps)),
            total=len(timesteps),
            desc=f"DDIM sampling (S={S}, eta={eta})",
            leave=False,
            disable=disable_bar,
        )

        ts = time.time()
        for i in steps:
            t_next_idx = i + 1 if i + 1 < len(timesteps) else None
            x = self.ddim_step(model, x, i, t_next_idx, timesteps, cond, eta)
        print(f"DDIM sampling took {time.time() - ts:.2f}s  ({S} steps, eta={eta})")
        return x


# ---------------------------------------------------------------------------
# Combined model + training step
# ---------------------------------------------------------------------------

class ConditionalDiffusionModel(nn.Module):
    """
    Full pipeline:
      encoder (VAE) -> latent z -> diffusion UNet conditioned on z.
    """
    def __init__(
        self,
        T: int = 1000,
        kl_weight: float = 1e-4,
        VAEEncoder = None,
        DETEncoder = None,
        params = None,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.params = params
        # self.downscale_resolution = (VAEEncoder.patchembed3d.output_size[0]+1+1*self.params.upper_air_boundary,
        #                     (VAEEncoder.patchembed2d.output_size[0] - VAEEncoder.patchembed2d.output_size[0] % self.params.updown_scale_factor) \
        #                     // self.params.updown_scale_factor + VAEEncoder.patchembed2d.output_size[0] % self.params.updown_scale_factor,
        #                     (VAEEncoder.patchembed2d.output_size[1] - VAEEncoder.patchembed2d.output_size[1] % self.params.updown_scale_factor) \
        #                     // self.params.updown_scale_factor + VAEEncoder.patchembed2d.output_size[1] % self.params.updown_scale_factor)


        self.downscale_resolution_det = (DETEncoder.patchembed3d.output_size[0]+1+1*self.params.upper_air_boundary,
                            (DETEncoder.patchembed2d.output_size[0] - DETEncoder.patchembed2d.output_size[0] % self.params.updown_scale_factor) \
                            // self.params.updown_scale_factor + DETEncoder.patchembed2d.output_size[0] % self.params.updown_scale_factor,
                            (DETEncoder.patchembed2d.output_size[1] - DETEncoder.patchembed2d.output_size[1] % self.params.updown_scale_factor) \
                            // self.params.updown_scale_factor + DETEncoder.patchembed2d.output_size[1] % self.params.updown_scale_factor)        
        self.num_surface_vars = len(params.surface_variables)
        self.num_diagnostic_vars = len(params.diagnostic_variables)
        self.num_land_vars = len(params.land_variables)
        self.num_ocean_vars = len(params.ocean_variables)
        self.surface_prognostic_idxs = torch.cat((torch.arange(self.num_surface_vars).long(),
                                                  torch.arange(self.num_surface_vars + self.num_diagnostic_vars, self.num_surface_vars + self.num_diagnostic_vars + self.num_land_vars + self.num_ocean_vars).long()))
    
        self.encoder = VAEEncoder
        self.model_det = DETEncoder
        self.freeze_encoder()
        self._encoder_param_checksums = self._snapshot_encoder_params()
        self.unet = ConUNet_1degV2(dim_in =20, dim_cond=2, dim_out=10, c = 64,
                                   c_mults=(1, 2, 2, 4),scale=[2, 2, 2], resnet_block_groups=4,
                                   checkpointing=getattr(params, 'checkpointing', 0),
                                   use_reentrant=getattr(params, 'use_reentrant', False))

        self.scheduler_diff = DDPMScheduler(T=T)
        self.scheduler_ddim = DDIMScheduler(
            T=T,
            num_inference_steps=ddim_steps,
            eta=ddim_eta,
        )

    def freeze_encoder(self):
        self.encoder.eval()
        self.model_det.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.model_det.parameters():
            param.requires_grad = False

    def _snapshot_encoder_params(self) -> dict:
        """Store a lightweight checksum of all encoder parameters for drift detection."""
        checksums = {}
        for name, param in self.encoder.named_parameters():
            checksums[f"encoder.{name}"] = param.data.sum().item()
        for name, param in self.model_det.named_parameters():
            checksums[f"model_det.{name}"] = param.data.sum().item()
        return checksums

    def assert_encoder_frozen(self):
        """Assert encoder/det parameters are frozen (requires_grad=False) and unchanged."""
        for name, param in self.encoder.named_parameters():
            assert not param.requires_grad, \
                f"encoder param '{name}' has requires_grad=True — encoder is not frozen!"
            key = f"encoder.{name}"
            current = param.data.sum().item()
            expected = self._encoder_param_checksums[key]
            assert abs(current - expected) < 1e-6, \
                f"encoder param '{name}' changed during training: {expected:.6f} -> {current:.6f}"
        for name, param in self.model_det.named_parameters():
            assert not param.requires_grad, \
                f"model_det param '{name}' has requires_grad=True — encoder is not frozen!"
            key = f"model_det.{name}"
            current = param.data.sum().item()
            expected = self._encoder_param_checksums[key]
            assert abs(current - expected) < 1e-6, \
                f"model_det param '{name}' changed during training: {expected:.6f} -> {current:.6f}"

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.model_det.eval()
        return self

    def _prepare_surface(
        self,
        surface_in: torch.Tensor,
        constant_boundary: torch.Tensor,
        varying_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate surface fields with boundary conditions."""
        if constant_boundary.dim() == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        return torch.cat([surface_in, constant_boundary, varying_boundary], dim=1)

    def _encode_vae(
        self, surface: torch.Tensor, upper_air: torch.Tensor
    ) -> torch.Tensor:
        """Stochastic VAE encoding -> latent z."""
        x = self.encoder._select_surface_component(surface) 
        z, mean, logvar = self.encoder.encode(x)  #1, 2, 23, 45
       
        return z

    def _encode_det(
        self, surface: torch.Tensor, upper_air: torch.Tensor, train: bool
    ) -> torch.Tensor:
        """Deterministic encoding -> conditioned feature map."""
        surface_emb = self.model_det.patchembed2d(surface)
        upper_air_emb = self.model_det.patchembed3d(upper_air)
        x = torch.cat([upper_air_emb, surface_emb.unsqueeze(2)], dim=2)
        B, C, Pl, _, _ = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = self.model_det.layer1(x, train)
        skip = x
        x = self.model_det.downsample(x)
        x = self.model_det.layer2(x, train)
        x = self.model_det.layer3(x, train)
        x = x.reshape(B, Pl, -1, 240 * self.params.updown_scale_factor)
        return x, skip

    def plot_noise_comparison(
        self,
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        save_path: str = "noise_comparison.png",
        n_channels: int = 4,
    ) -> None:
        """Plot real vs predicted noise for a single batch item.

        Args:
            noise:      real noise, shape (B, C, H, W)
            noise_pred: predicted noise, same shape
            t:          timestep tensor, shape (B,)
            save_path:  where to save the figure
            n_channels: how many channels (columns) to show
        """
        real = noise[0].detach().cpu().float()       # (C, H, W)
        pred = noise_pred[0].detach().cpu().float()  # (C, H, W)
        diff = real - pred

        C = real.shape[0]
        n_channels = min(n_channels, C)
        fig, axes = plt.subplots(3, n_channels, figsize=(4 * n_channels, 9))
        if n_channels == 1:
            axes = axes[:, None]

        row_labels = ["Real noise", "Predicted noise", "Difference"]
        for col, ch in enumerate(range(n_channels)):
            for row, (data, label) in enumerate(zip([real, pred, diff], row_labels)):
                ax = axes[row, col]
                im = ax.imshow(data[ch].numpy(), cmap="RdBu_r", origin="upper")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                title = f"ch {ch}" if row > 0 else f"{label}  ch {ch}"
                if col == 0:
                    title = f"{label}\nch {ch}"
                ax.set_title(title, fontsize=8)
                ax.axis("off")

        t_val = t[0].item()
        mse = ((real - pred) ** 2).mean().item()
        fig.suptitle(f"Noise comparison  t={t_val}  MSE={mse:.4f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_noise_comparison] saved → {save_path}")

    def training_step(
        self,
        surface_in: torch.Tensor,
        constant_boundary: torch.Tensor,
        varying_boundary: torch.Tensor,
        upper_air_in: torch.Tensor,
        train: bool = True,
        plot_freq: int = 0,
        plot_path: str = "noise_comparison.png",
        iter = 0
    ) -> torch.Tensor:
        """Single diffusion training step. Returns RMSE loss."""
        self.assert_encoder_frozen()
        surface = self._prepare_surface(surface_in, constant_boundary, varying_boundary)

        B = surface.size(0)
        device = surface.device
        self.scheduler_diff.to(device)

        # Encode stochastic condition (VAE) and deterministic features
        z = self._encode_vae(surface, upper_air_in)
        x, skip = self._encode_det(surface, upper_air_in, train)

        # Sample timestep and forward-diffuse x
        t = torch.randint(0, self.scheduler_diff.T, (B,), device=device)
        x_t, noise = self.scheduler_diff.q_sample(x, t)

        # Predict and score noise
        noise_pred = self.unet(x_t, z, t)
        loss = F.mse_loss(noise_pred, noise)

        if loss.item() > 2:
            print(f"Warning: High diffusion loss detected: {loss.item():.4f}")
            print("noise_pred[0]:", noise_pred[0])
            print("noise[0]:", noise[0])

        if plot_freq > 0 and iter % plot_freq == 0: 
            self.plot_noise_comparison(noise, noise_pred, t, save_path=plot_path)

        return loss

    @torch.no_grad()
    def generate(
        self,
        model_diff = None,
        z: torch.Tensor= None,
        num_samples: int = 1,
        sample_shape: tuple | None = None,
        device = None,
        sampler: str = "ddpm",
        ddim_steps: int | None = None,
        ddim_eta: float | None = None,
    ) -> torch.Tensor:
        """
        Generate samples conditioned on z.

        Args:
            model_diff:  noise-predicting UNet (self.unet).
            z:           conditioning tensor from the VAE encoder.
            num_samples: ensemble members per condition.
            sample_shape: shape of the latent to sample (B,C,H,W) or (C,H,W).
            device:      target device.
            sampler:     ``"ddpm"`` for full 1000-step DDPM, or ``"ddim"`` for
                         accelerated DDIM sampling.
            ddim_steps:  number of DDIM denoising steps (overrides instance
                         default; only used when sampler="ddim").
            ddim_eta:    DDIM stochasticity (0=deterministic, 1=DDPM-like).
        """
        z = z.repeat_interleave(num_samples, dim=0)

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

        if sampler == "ddim":
            self.scheduler_ddim.to(device)
            return self.scheduler_ddim.sample(
                model_diff, shape, z, device,
                num_inference_steps=ddim_steps,
                eta=ddim_eta,
            )
        else:
            return self.scheduler_diff.sample(model_diff, shape, z, device)


    def prediction(self, surface_in, constant_boundary,
                   varying_boundary, upper_air_in,
                   num_samples=1, device=None,
                   sampler: str = "ddpm",
                   ddim_steps: int | None = None,
                   ddim_eta: float | None = None):

        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        surface_in = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)

        B = surface_in.size(0)
        device = surface_in.device
        self.scheduler_diff.to(device)
        ###############encoder 2 start ########################
        # 1. Encode condition
        #######VAE ENCODER START ########
        surface_vae = self.encoder.patchembed2d(surface_in)
        upper_air_vae = self.encoder.patchembed3d(upper_air_in)
        x = torch.concat([upper_air_vae, surface_vae.unsqueeze(2)], dim=2)

        B_vae, C_vae, Pl_vae, _, _ = x.shape

        x_vae = x.reshape(B_vae, C_vae, -1).transpose(1, 2)
        x_vae = self.encoder.layer1(x_vae)
        # skip = x_vae
        x_vae = self.encoder.downsample(x_vae) #8, 10350, 384
        x_vae = self.encoder.layer2(x_vae)
        x_vae = self.encoder.layer3(x_vae)
        x_vae = x_vae.reshape(B, self.downscale_resolution[0], self.downscale_resolution[1], self.downscale_resolution[2], -1).permute(0, 4, 1, 2, 3)
        mu = self.encoder.layer_mu(x_vae) # 
        sigma = self.encoder.layer_sigma(x_vae) 
        norm = self.encoder.reparameterize(mu, sigma) #1, 192, 10, 23, 45
    
        z = norm.permute(0, 2, 3,4, 1).reshape(B_vae, Pl_vae, -1, 192 * self.params.updown_scale_factor) #8, 10350, 384
        # print("x shape after VAE reparameterize ", z.shape) # 2, 10, 1035, 384
        
        ###############encoder 1 start (deterministic) ########################
        surface_det = self.model_det.patchembed2d(surface_in)
        upper_air_det = self.model_det.patchembed3d(upper_air_in)
        x = torch.concat([upper_air_det, surface_det.unsqueeze(2)], dim=2)
   
        B_det, C_det, Pl_det, Lat, Lon = x.shape
        #print("x_det shape before reshape ", x.shape)  #torch.Size([2, 192, 10, 45, 90])

        x_det = x.reshape(B_det, C_det, -1).transpose(1, 2)
        x_det = self.model_det.layer1(x_det, train=False)
        skip = x_det
        x = self.model_det.downsample(x_det)
        x = self.model_det.layer2(x, train=False)
        x = self.model_det.layer3(x, train=False)
        x = x.reshape(B, Pl_det, -1,240*self.params.updown_scale_factor)
        target_latent_shape = x.shape
        
        if torch.isnan(x).any():
            print(f"[NaN check] x before diffusion has NaN: {torch.isnan(x).sum().item()} NaNs")
  
            
        x = self.generate(
            model_diff=self.unet,
            z=z,
            num_samples=num_samples,
            sample_shape=target_latent_shape,
            device=device,
            sampler=sampler,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        
        if torch.isnan(x).any():
            print(f"[NaN check] x has NaN: {torch.isnan(x).sum().item()} NaNs")
        else:
            print("[NaN check] x : no NaN")
          
        x = x.reshape(B, -1 ,240*self.params.updown_scale_factor)
        
        ######## DETERMINISTIC DECODER START ######
        x = self.model_det.upsample(x)
        x = self.model_det.layer4(x, train=False)
        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl_det, Lat, Lon)
        output_surface = output[:, :, -1, :, :]
        output_upper_air = output[:, :, :-1, :, :]
        output_2D = self.model_det.patchrecovery2d(output_surface)
        output_surface = output_2D[:, self.surface_prognostic_idxs]
        if torch.isnan(output_surface).any():
            print(f"[NaN check] output_surface (after patchrecovery2d) has NaN: {torch.isnan(output_surface).sum().item()} NaNs")
   
        output_upper_air = self.model_det.patchrecovery3d(output_upper_air)
        output_diagnostic = output_2D[:, self.num_surface_vars:self.num_surface_vars + self.num_diagnostic_vars].reshape(
            output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1])
        
        
        return output_surface, output_upper_air, output_diagnostic
        #######start decoder #########
        
        

# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
