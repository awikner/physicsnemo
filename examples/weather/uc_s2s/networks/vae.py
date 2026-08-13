import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np 
from utils.patch_embed import PatchEmbed2D, PatchEmbed3D
from utils.patch_recovery import PatchRecovery2D, PatchRecovery3D
from utils.earth_position_index import get_earth_position_index
from utils.pad import get_pad3d
from utils.shift_window_mask import get_shift_window_mask, window_partition, window_reverse
from utils.crop import crop3d
from timm.models.layers import trunc_normal_, DropPath
from utils.integrate import Integrator, forward_euler
import os
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels, max_groups=32):
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


def _match_spatial_shape(tensor, target_shape):
    target_height, target_width = target_shape
    height, width = tensor.shape[-2:]

    if height > target_height:
        start = (height - target_height) // 2
        tensor = tensor[..., start:start + target_height, :]
    elif height < target_height:
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        tensor = F.pad(tensor, (0, 0, pad_top, pad_bottom))

    if width > target_width:
        start = (width - target_width) // 2
        tensor = tensor[..., :, start:start + target_width]
    elif width < target_width:
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, 0, 0))

    return tensor


# ----------------------------
# Basic building blocks
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            _group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ----------------------------
# Encoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, z_channels=4):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = nn.Sequential(
            ResBlock(base_channels, base_channels),
            Downsample(base_channels)
        )
        self.down2 = nn.Sequential(
            ResBlock(base_channels, base_channels * 2),
            Downsample(base_channels * 2)
        )
        self.down3 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels * 4),
            Downsample(base_channels * 4)
        )

        self.mid = ResBlock(base_channels * 4, base_channels * 4)

        # Output mean and logvar
        self.conv_out = nn.Conv2d(base_channels * 4, z_channels * 2, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)
        x = self.conv_out(x)

        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar


# ----------------------------
# Decoder
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, z_channels=4):
        super().__init__()

        self.conv_in = nn.Conv2d(z_channels, base_channels * 4, 3, padding=1)

        self.mid = ResBlock(base_channels * 4, base_channels * 4)

        self.up1 = nn.Sequential(
            Upsample(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 2)
        )
        self.up2 = nn.Sequential(
            Upsample(base_channels * 2),
            ResBlock(base_channels * 2, base_channels)
        )
        self.up3 = nn.Sequential(
            Upsample(base_channels),
            ResBlock(base_channels, base_channels)
        )

        self.conv_out = nn.Sequential(
            _group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, z, target_shape=None):
        z = self.conv_in(z)
        z = self.mid(z)
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        if target_shape is not None:
            z = F.interpolate(z, size=target_shape, mode='bilinear', align_corners=False)
        return self.conv_out(z)


# ----------------------------
# VAE Wrapper
# ----------------------------
class VAE(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.component = int(getattr(params, 'component', 8))
        self.base_channels = int(getattr(params, 'base_channels', 2))
        self.z_channels = int(getattr(params, 'z_channels', 4))

        self.encoder = Encoder(in_channels=1, base_channels=self.base_channels, z_channels=self.z_channels)
        self.decoder = Decoder(out_channels=1, base_channels=self.base_channels, z_channels=self.z_channels)

    def _select_surface_component(self, surface_in):
        if surface_in.ndim != 4:
            raise ValueError(f"Expected surface input with shape [B, C, H, W], got {tuple(surface_in.shape)}")
        if self.component < 0 or self.component >= surface_in.shape[1]:
            raise ValueError(
                f"Requested component index {self.component}, but surface input only has {surface_in.shape[1]} channels"
            )
        #print(f"Selected component {self.component} from surface input with shape {surface_in.shape}")
        return surface_in[:, self.component:self.component + 1]

    def _merge_surface_component(self, reconstruction, surface_in, target_surface=None):
        base_surface = target_surface.clone() if target_surface is not None else surface_in.clone()
        base_surface[:, self.component:self.component + 1] = reconstruction
        return base_surface

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    def decode(self, z, target_shape=None):
        return self.decoder(z, target_shape=target_shape)

    def forward(
        self,
        surface_in,
        constant_boundary=None,
        varying_boundary=None,
        upper_air_in=None,
        train=False,
    ):
        #print("Input surface shape:", surface_in.shape)
        x = self._select_surface_component(surface_in) # 2,1 180, 360
        z, mean, logvar = self.encode(x) #2, 2, 23, 45
        x_recon = self.decode(z, target_shape=x.shape[-2:])
        return x_recon, mean, logvar
    
    

    
    
    