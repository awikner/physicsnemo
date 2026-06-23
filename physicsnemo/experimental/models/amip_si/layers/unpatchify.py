# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from /work/nvme/bdiu/awikner/amip @ commit 497827e
# (modules/layers/unpatchify.py) for Phase 8a of the ai-rossby plan.
# Imports rewritten to local layer modules; ``modulate_fused`` is
# inlined from modules/layers/old/fa_basics.py (a jit-scripted
# ``x * (1 + scale) + shift``).

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .conv import SphereConv2d


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """AdaLN modulation (jit-scripted)."""
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, 
                 hidden_size,
                 cond_dim,
                 patch_size, 
                 out_channels,
                 modulate_2d=False,
                 hpx=False):
        super().__init__()
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True))
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            if hpx:
                self.linear = nn.Linear(hidden_size, patch_size**3 * out_channels, bias=True)
            else:
                self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(cond_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            self.modulate_2d = modulate_2d
        
        self.init_params()

    def forward(self, x, c=None):
        if c is None:
            x = self.output_layer(x)
            return x
        else:
            z = self.adaLN_modulation(c) # b, 2*hidden_size
            if self.modulate_2d:
                z = z.unsqueeze(1).unsqueeze(1) # b, 1, 1, 2*hidden_size
            else:
                z = z.unsqueeze(1) # b, 1, 2*hidden_size
            shift, scale = z.chunk(2, dim=-1) # b, 1, hidden_size or b, 1, 1, hidden_size
            x = modulate_fused(self.norm_final(x), shift, scale)
            x = self.linear(x)
            return x
        
    def init_params(self):
        if self.cond_dim is not None:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.adaLN_modulation[0].weight, 0)
            nn.init.constant_(self.adaLN_modulation[0].bias, 0)
            nn.init.constant_(self.linear.weight, 0)
            nn.init.constant_(self.linear.bias, 0)

        else:
            nn.init.constant_(self.output_layer[1].weight, 0)
            nn.init.constant_(self.output_layer[1].bias, 0)


class Unpatchify(nn.Module):
    """
    Unpatchify a tensor.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, grid_size, patch_size, in_dim, out_dim, cond_dim=None):
        super().__init__()
        self.grid_x, self.grid_y = grid_size
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.out_layer = FinalLayer(hidden_size=in_dim,
                                    cond_dim=cond_dim,
                                    patch_size=patch_size[0],
                                    out_channels=out_dim,)
    
    def forward(self, x, cond=None):
        # x in shape [b, nlat//p * nlon//p, dim]
        x = self.out_layer(x, cond) # [batch_size, nlat//p * nlon//p, patch_size * patch_size * out_dim]
        c = self.out_dim
        h, w = self.grid_x, self.grid_y

        assert h * w == x.shape[1]
        ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw)) # [b, c, nlat, nlon]
        imgs = imgs.permute(0, 2, 3, 1) # [b, nlat, nlon, c]

        return imgs

# borrowed from 
#https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class SubPixelConvICNR_2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        patch_size (tuple[int]): px, py
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, grid_size, patch_size, in_chans, out_chans):
        super().__init__()
        assert patch_size[0] == patch_size[1], 'mismatch'
        
        self.grid_x, self.grid_y = grid_size
        self.conv = nn.Conv2d(in_chans, 
                              out_chans*patch_size[0]**2, 
                              kernel_size=1, 
                              stride=1, 
                              padding=0, 
                              bias=False)
        
        self.pixelshuffle = nn.PixelShuffle(patch_size[0])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[0])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x, t):
        x = rearrange(x, 'b (ny nx) c -> b c ny nx', ny=self.grid_x, nx=self.grid_y)

        # x in shape [b, in_chans, h, w], where h, w are the height and width of the patchified feature map
        output = self.conv(x)
        
        output = self.pixelshuffle(output)

        x = rearrange(output, 'b c h w -> b h w c') # [b, nlat, nlon, out_chans]

        return output

def sphere_pad(input, padding) -> torch.Tensor:
    """

    Args:
        input: Input tensor of shape (B, C, H, W)

    Returns:
        Padded tensor with spherical boundary conditions
    """
    assert input.dim() == 4, (
        "Input tensor must be 4D (batch, channels, height, width)"
    )
    assert input.shape[3] % 2 == 0, (
        "Width of the input tensor must be even for proper shperical padding"
    )
    half_width = input.shape[3] // 2

    left_pad, right_pad, top_pad, bottom_pad= padding[0], padding[1], padding[2], padding[3]

    if top_pad > 0:
        top_rows = input[:, :, : top_pad, :]
        top_rows = torch.roll(top_rows, shifts=half_width, dims=3)
        top_rows = torch.flip(top_rows, dims=[2])
    else:
        top_rows = torch.empty(0, device=input.device, dtype=input.dtype)
    if bottom_pad > 0:
        bottom_rows = input[:, :, -bottom_pad :, :]
        bottom_rows = torch.roll(bottom_rows, shifts=half_width, dims=3)
        bottom_rows = torch.flip(bottom_rows, dims=[2])
    else:
        bottom_rows = torch.empty(0, device=input.device, dtype=input.dtype)
    input = torch.cat([top_rows, input, bottom_rows], dim=2)

    return F.pad(input, (left_pad, right_pad, 0, 0), mode="circular")

class PolarPad2d(nn.Module):
    """
    Padding for convolutions on a 2D grid over the pole.

    Args:
        pad: (size of top padding, size of bottom padding)
        x: Image with shape (n_batches, n_channels, lat, lon)
    """
    def __init__(self, pad, num_lat = None):
        super().__init__()
        self.pad_top = pad[0]
        self.pad_bottom = pad[1]
        self.num_lat = num_lat if num_lat is not None else 45
        self.pad_idxs = torch.cat((torch.arange(self.pad_top), torch.arange(self.pad_top+1, self.num_lat+self.pad_top+1),
                                    torch.arange(self.num_lat+self.pad_top+2, self.num_lat+self.pad_top+self.pad_bottom+2))).long()
        self.pad_idxs.requires_grad_(requires_grad = False)

    def forward(self, x):
        # x in shape b c nlat nlon

        # first pad 1 pixel on top and bottom with constant 0, then pad self.pad_top pixels on top and self.pad_bottom pixels on bottom with reflect, then select the padded latitudes according to self.pad_idxs
        padded_x = nn.functional.pad(nn.functional.pad(x, (0, 0, 1, 1), mode = 'constant', value = 0.),
                                    (0, 0, self.pad_top, self.pad_bottom), mode = 'reflect')[..., self.pad_idxs, :]
        padded_x[..., :self.pad_top, :] = torch.roll(padded_x[..., :self.pad_top, :], padded_x.shape[-1] // 2, dims = -1)
        padded_x[..., -self.pad_bottom:, :] = torch.roll(padded_x[..., -self.pad_bottom:, :], padded_x.shape[-1] // 2, dims = -1)
        return padded_x

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False, periodic_dim=None):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
            periodic_dim (int, optional): dimension index along which to apply
                periodic boundary conditions before interpolating. If None,
                no periodic padding is applied.
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.periodic_dim = periodic_dim

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        if self.periodic_dim is not None:
            dim = self.periodic_dim
            # Pad one slice on each side along the periodic dimension
            x = torch.cat([x.select(dim, -1).unsqueeze(dim),
                            x,
                            x.select(dim, 0).unsqueeze(dim)], dim=dim)

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        if self.periodic_dim is not None:
            # Crop the periodically padded region, which expanded by scale_factor
            sf = self.scale_factor
            ndim = x.ndim
            dim = self.periodic_dim if self.periodic_dim >= 0 else ndim + self.periodic_dim
            # scale_factor may be a scalar or a sequence aligned to spatial dims
            # spatial dims start at index 2, so spatial index = dim - 2
            if isinstance(sf, (tuple, list)):
                crop = int(sf[dim - 2])
            else:
                crop = int(sf)
            slices = [slice(None)] * ndim
            slices[dim] = slice(crop, -crop)
            x = x[tuple(slices)]

        return x

class PatchInterpolate2D(nn.Module):
    """
    Patch Interpolation to 2D Image.
    """
    def __init__(self, grid_size, patch_size, in_chans, out_chans, hidden_dim = None):
        super().__init__()
        self.grid_x, self.grid_y = grid_size
        self.patch_size = patch_size[0]
        self.in_chans = in_chans
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_chans // 2
        self.adaLN_shift_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_chans, 2 * in_chans, bias=True)
        )
        nn.init.zeros_(self.adaLN_shift_scale[-1].weight)
        nn.init.zeros_(self.adaLN_shift_scale[-1].bias)
        self.conv = nn.Conv2d(in_chans, self.hidden_dim, kernel_size=1, stride=1, padding=0)
        # x is rearranged to [b, c, h(lat), w(lon)] below, so longitude is the
        # last (periodic) axis. Pad/crop along dim=-1 so the bilinear upsample
        # wraps in longitude and does not leave a left/right seam.
        self.interp = Interpolate(scale_factor=patch_size, mode="bilinear", align_corners=True, periodic_dim=-1)

        self.head = nn.Sequential(
                SphereConv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.GELU(),
                SphereConv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, out_chans, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x, condition_embed):
        # reshape x to [b, in_chans, h, w]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.grid_x, w=self.grid_y)

        shift, scale = self.adaLN_shift_scale(condition_embed).chunk(2, dim=1)
        shift = shift[:, :, None, None]
        scale = scale[:, :, None, None]
        x = modulate_fused(x, shift, scale)

        x = self.conv(x) # b, hidden_dim, h, w
        x = self.interp(x) # b, hidden_dim, h*patch_size, w*patch_size
        x = self.head(x) # b, out_chans, h*patch_size, w*patch_size

        x = rearrange(x, 'b c h w -> b h w c') # b, nlat, nlon, out_chans

        return x

