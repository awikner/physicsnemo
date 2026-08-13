'''
Pseudocode of Pangu-Weather
'''
# The pseudocode can be implemented using deep learning libraries, e.g., Pytorch and Tensorflow or other high-level APIs

# Basic operations used in our model, namely Linear, Conv3d, Conv2d, ConvTranspose3d and ConvTranspose2d
# Linear: Linear transformation, available in all deep learning libraries
# Conv3d and Con2d: Convolution with 2 or 3 dimensions, available in all deep learning libraries
# ConvTranspose3d, ConvTranspose2d: transposed convolution with 2 or 3 dimensions, see Pytorch API or Tensorflow API
#from Your_AI_Library import Linear, Conv3d, Conv2d, ConvTranspose3d, ConvTranspose2d

# Functions in the networks, namely GeLU, DropOut, DropPath, LayerNorm, and SoftMax
# GeLU: the GeLU activation function, see Pytorch API or Tensorflow API
# DropOut: the dropout function, available in all deep learning libraries
# DropPath: the DropPath function, see the implementation of vision-transformer, see timm pakage of Pytorch
# A possible implementation of DropPath: from timm.models.layers import DropPath
# LayerNorm: the layer normalization function, see Pytorch API or Tensorflow API
# Softmax: softmax function, see Pytorch API or Tensorflow API
#from Your_AI_Library import GeLU, DropOut, DropPath, LayerNorm, SoftMax

# Common functions for roll, pad, and crop, depends on the data structure of your software environment
#from Your_AI_Library import roll3D, pad3D, pad2D, Crop3D, Crop2D

# Common functions for reshaping and changing the order of dimensions
# reshape: change the shape of the data with the order unchanged, see Pytorch API or Tensorflow API
# TransposeDimensions: change the order of the dimensions, see Pytorch API or Tensorflow API
#from Your_AI_Library import reshape, TransposeDimensions

# Common functions for creating new tensors
# ConstructTensor: create a new tensor with an arbitrary shape
# TruncatedNormalInit: Initialize the tensor with Truncate Normalization distribution
# RangeTensor: create a new tensor like range(a, b)
#from Your_AI_Library import ConstructTensor, TruncatedNormalInit, RangeTensor

# Common operations for the data, you may design it or simply use deep learning APIs default operations
# LinearSpace: a tensor version of numpy.linspace
# MeshGrid: a tensor version of numpy.meshgrid
# Stack: a tensor version of numpy.stack
# Flatten: a tensor version of numpy.ndarray.flatten
# TensorSum: a tensor version of numpy.sum
# TensorAbs: a tensor version of numpy.abs
# Concatenate: a tensor version of numpy.concatenate
#from Your_AI_Library import LinearSpace, MeshGrid, Stack, Flatten, TensorSum, TensorAbs, Concatenate

# Common functions for training models
# LoadModel and SaveModel: Load and save the model, some APIs may require further adaptation to hardwares
# Backward: Gradient backward to calculate the gratitude of each parameters
# UpdateModelParametersWithAdam: Use Adam to update parameters, e.g., torch.optim.Adam
#from Your_AI_Library import LoadModel, Backward, UpdateModelParametersWithAdam, SaveModel

# Custom functions to read your data from the disc
# LoadData: Load the ERA5 data
# LoadConstantMask: Load constant masks, e.g., soil type
# LoadStatic: Load mean and std of the ERA5 training data, every fields such as T850 is treated as an image and calculate the mean and std
#from Your_AI_Library import LoadData, LoadConstantMask, LoadStatic
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from dataclasses import dataclass
from physicsnemo.core.module import Module
from physicsnemo.core.meta import ModelMetaData
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

# Global flag for using Transformer Engine, initially set to True
USE_TE = False




# # Conditional imports
# if USE_TE:
#     import transformer_engine.pytorch as te
#     from transformer_engine.common import recipe
#     from torch.cuda import amp

#     fp8_recipe = recipe.DelayedScaling(
#         fp8_format=recipe.Format.HYBRID,
#         amax_history_len=16,
#         amax_compute_algo="max"
#     )

@dataclass
class MetaData(ModelMetaData):
    name: str = "PanguModel_Plasim"
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True


class PanguModel_Plasim(Module):
    """
    Pangu-Weather backbone extended for Plasim/ERA5 S2S forecasting.

    Subclasses physicsnemo.Module for checkpoint save/load and registry support.
    """

    def __init__(self, params, num_heads=(6, 12, 12, 6), drop_path=None, land_mask=None, mask_fill=None,
                 surface_ff_std=None, surface_delta_std=None, upper_air_ff_std=None, upper_air_delta_std=None):
        super().__init__(meta=MetaData()) 
        #####
        global USE_TE
        USE_TE = params.use_transformer_engine
        self.checkpointing = 0
        self.use_reentrant = False
        if hasattr(params, 'checkpointing'):
            self.checkpointing = params.checkpointing
        if hasattr(params, 'use_reentrant'):
            self.use_reentrant = params.use_reentrant

        if USE_TE:
            global te, recipe, amp
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe
            from torch.cuda import amp

            global fp8_recipe
            fp8_recipe = recipe.DelayedScaling(
                fp8_format=recipe.Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max"
            )

        #if hasattr(params, 'embed_dim'):
        #   embed_dim = params.embed_dim
        # else:
        #     embed_dim = 192
        embed_dim = 240
        print(f"Embedding Dimensions are {embed_dim}")

        #drop_path = np.linspace(0, 0.2, 8).tolist()
        if not drop_path:
            drop_path = np.append(np.linspace(0, 0.2, np.sum(params.depths[:2])), np.linspace(0.2, 0, np.sum(params.depths[2:]))).tolist()

        # Masked output
        if hasattr(params, 'mask_output'):
            self.mask_output = params.mask_output
        else:
            self.mask_output = False
        self.has_land = False
        self.has_ocean = False
        self.num_land_vars = 0
        self.num_ocean_vars = 0
        self.diagnostic_vars = []
        self.num_diagnostic_vars = 0
        params["patch_size"] = (2,2,2)
        params["window_size"] = (2,6,10)

        if hasattr(params, 'land_variables'):
            if len(params.land_variables) > 0:
                self.has_land = True
                self.num_land_vars = len(params.land_variables)
                if not self.mask_output:
                    print('mask_output is False. Land variables output will not be masked.')
                else:
                    if land_mask is None:
                        lm_file        = os.path.join(params.data_dir, params.boundary_dir, 'lsm.nc')                
                        land_mask_ds   = xr.open_dataset(lm_file)
                        land_mask = torch.from_numpy(land_mask_ds.lsm.values).to(torch.float32)
                        nans = torch.isnan(land_mask)
                        land_mask = land_mask.masked_fill_(nans, 0.)
                        land_mask_ds.close()
                    if self.predict_delta:
                        self.land_mask = Mask(land_mask)
                    else:
                        land_mask_fill = torch.stack([(1. - land_mask) * mask_fill[var] for var in params.land_variables])
                        self.land_mask = Mask(land_mask, land_mask_fill)

        if hasattr(params, 'ocean_variables'):
            if len(params.ocean_variables) > 0:
                self.has_ocean = True
                self.num_ocean_vars = len(params.ocean_variables)
                if not self.mask_output:
                    print('mask_output is False. Ocean variables output will not be masked.')
                else:
                    if self.has_land:
                        ocean_mask = (1. - land_mask)
                    elif land_mask is None:
                        lm_file        = os.path.join(params.data_dir, params.boundary_dir, 'lsm.nc')                
                        land_mask_ds   = xr.open_dataset(lm_file)
                        land_mask = torch.from_numpy(land_mask_ds.lsm.values).to(torch.float32)
                        nans = torch.isnan(land_mask)
                        land_mask = land_mask.masked_fill_(nans, 0.)
                        land_mask_ds.close()
                        ocean_mask = (1. - land_mask)
                    if self.predict_delta:
                        self.ocean_mask = Mask(ocean_mask)
                    else:
                        ocean_mask_fill = torch.stack([(1. - ocean_mask) * mask_fill[var] for var in params.ocean_variables])
                        self.ocean_mask = Mask(ocean_mask, ocean_mask_fill)

        if hasattr(params, 'diagnostic_variables'):
            self.diagnostic_vars = params.diagnostic_variables
            self.num_diagnostic_vars = len(self.diagnostic_vars)
        print(f'Num diagnostic vars: {self.num_diagnostic_vars}')
        if hasattr(params, "drop_rate"):
            if params.drop_rate > 0.:
                drop_path = np.zeros(np.sum(params.depths)).tolist()
        else:
            params['drop_rate'] = 0.
        
        self.num_surface_vars = len(params.surface_variables)
        self.num_atmo_vars = len(params.upper_air_variables)
        self.num_boundary_vars = len(params.constant_boundary_variables) + len(params.varying_boundary_variables)
        self.atmo_resolution = [len(params.levels)] + params.horizontal_resolution
        depths_cumsum = np.cumsum(params.depths).astype(int)
        self.predict_delta = params.predict_delta
        self.surface_prognostic_idxs = torch.cat((torch.arange(self.num_surface_vars).long(), 
                                                  torch.arange(self.num_surface_vars + self.num_diagnostic_vars, self.num_surface_vars + self.num_diagnostic_vars + self.num_land_vars + self.num_ocean_vars).long()))


        self.window_size = params.window_size
        self.vertical_windowing=params.vertical_windowing
        self.embed_dim = embed_dim
        self.updown_scale_factor = params.updown_scale_factor
        self.subpixed_deconv = False
        grid_has_poles = False
        self.polar_pad = False
        self.recovery_head = False
        self.diagnostic_head = False
        if hasattr(params, 'subpixel_deconv'):
            self.subpixel_deconv = params.subpixel_deconv
            if hasattr(params, 'polar_pad'):
                self.polar_pad = params.polar_pad
                if self.polar_pad and hasattr(params, 'grid_has_poles'):
                    grid_has_poles = params.grid_has_poles
                elif self.polar_pad:
                    print('Polar padding for patch recovery is enabled, but grid_has_poles is unspecified. If grid has poles, this will lead to patch artifacts.')
            if hasattr(params, 'recovery_head'):
                self.recovery_head = params.recovery_head
            if hasattr(params, 'diagnostic_head'):
                self.diagnostic_head = params.diagnostic_head
        
        

        self.upper_air_boundary = params.upper_air_boundary
        self.varying_boundary_variables = params.varying_boundary_variables
        self.num_varying_boundary_vars = len(params.varying_boundary_variables)
        self.idx_upper_air_var_bound= self.varying_boundary_variables.index('toa_incident_solar_radiation') # careful, if change, change also the self.patchembed2d_upper_air_boundary and patchembed2d
        self.idx_surface_var_bound = [i for i in range(self.num_varying_boundary_vars) if i != self.idx_upper_air_var_bound]
    
        
        #####

        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)        
        if self.upper_air_boundary:
            self.patchembed2d_upper_air_boundary = PatchEmbed2D(
                img_size=params.horizontal_resolution,
                patch_size=params.patch_size[1:],
                in_chans=1, # to be changed if inncluding more variables in the future
                embed_dim=embed_dim,)
        

        self.patchembed2d = PatchEmbed2D(
            img_size=params.horizontal_resolution,
            patch_size=params.patch_size[1:],
            in_chans=self.num_surface_vars + self.num_land_vars + self.num_ocean_vars + self.num_boundary_vars - 1*self.upper_air_boundary,
            embed_dim=embed_dim,)
            
        
        self.patchembed3d = PatchEmbed3D(
            img_size=self.atmo_resolution,
            patch_size=params.patch_size,
            in_chans=self.num_atmo_vars,
            embed_dim=embed_dim)
        
        EST_input_resolution = (self.patchembed3d.output_size[0]+1+1*self.upper_air_boundary, self.patchembed3d.output_size[1], self.patchembed3d.output_size[2])

        
        downscale_resolution = (self.patchembed3d.output_size[0]+1+1*self.upper_air_boundary,
                                (self.patchembed2d.output_size[0] - self.patchembed2d.output_size[0] % params.updown_scale_factor) \
                                // params.updown_scale_factor + self.patchembed2d.output_size[0] % params.updown_scale_factor,
                                (self.patchembed2d.output_size[1] - self.patchembed2d.output_size[1] % params.updown_scale_factor) \
                                // params.updown_scale_factor + self.patchembed2d.output_size[1] % params.updown_scale_factor)

        self.downscale_resolution = downscale_resolution
        # print("downscale_resolution", downscale_resolution)
        self.EST_input_resolution = EST_input_resolution
        # print("EST_input_resolution", EST_input_resolution)
        if not self.vertical_windowing:
            self.window_size[0] = EST_input_resolution[0]
        # print("EST_input_resolution", EST_input_resolution)

        self.layer1 = EarthSpecificLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=params.depths[0],
            num_heads=num_heads[0],
            window_size=self.window_size,
            drop_path=drop_path[:depths_cumsum[0]],
            vertical_windowing=params.vertical_windowing,
            checkpointing = self.checkpointing,
            use_reentrant = self.use_reentrant)
        
        print('Embed Dim:')
        print(embed_dim)
        print('Input resolution:')
        print(EST_input_resolution)
        print('depth')
        print(params.depths[0])
        print('num_heads')
        print(num_heads[0])
        print('drop_path:')
        print(drop_path[:depths_cumsum[0]])
        
        self.downsample = DownSample(in_dim=embed_dim, input_resolution=EST_input_resolution, output_resolution=downscale_resolution, 
                                     downsample_factor=params.updown_scale_factor)
        
        self.layer2 = EarthSpecificLayer(
            dim=embed_dim * params.updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=params.depths[1],
            num_heads=num_heads[1],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[0]:depths_cumsum[1]],
            vertical_windowing=params.vertical_windowing,
            drop=params.dropout_rate,
            checkpointing = self.checkpointing,
            use_reentrant = self.use_reentrant)
        
        self.layer3 = EarthSpecificLayer(
            dim=embed_dim * params.updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=params.depths[2],
            num_heads=num_heads[2],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[1]:depths_cumsum[2]],
            vertical_windowing=params.vertical_windowing,
            drop=params.dropout_rate,
            checkpointing = self.checkpointing,
            use_reentrant = self.use_reentrant)
        
        

        self.upsample = UpSample(embed_dim * params.updown_scale_factor, embed_dim, downscale_resolution, 
                                 (self.patchembed3d.output_size[0]+1+1*self.upper_air_boundary, self.patchembed3d.output_size[1], self.patchembed3d.output_size[2]))
        
        self.layer4 = EarthSpecificLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=params.depths[3],
            num_heads=num_heads[3],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[2]:],
            vertical_windowing=params.vertical_windowing,
            checkpointing = self.checkpointing,
            use_reentrant = self.use_reentrant)
        
        
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        
        if self.subpixel_deconv:
            if self.recovery_head:
                from utils.patch_recovery import SubPixelConvICNR_2D_wHead as SubPixelConv_2D
                from utils.patch_recovery import SubPixelConvICNR_3D_wHead as SubPixelConv_3D

                if self.diagnostic_head:
                    self.patchrecovery2d = SubPixelConv_2D(params.horizontal_resolution, params.patch_size[1:], 2 * embed_dim, 
                                                           self.num_surface_vars, diagnostic_variables=self.num_diagnostic_vars, diagnostic_head=self.diagnostic_head,
                                                           land_variables=self.num_land_vars, ocean_variables=self.num_ocean_vars,
                                                           num_lat = self.atmo_resolution[1], polar_pad=self.polar_pad, grid_has_poles = grid_has_poles)
                else:
                    self.patchrecovery2d = SubPixelConv_2D(params.horizontal_resolution, params.patch_size[1:], 2 * embed_dim, 
                                                           self.num_surface_vars + self.num_diagnostic_vars, diagnostic_variables=0, diagnostic_head=self.diagnostic_head,
                                                           land_variables=self.num_land_vars, ocean_variables=self.num_ocean_vars,
                                                           num_lat = self.atmo_resolution[1], polar_pad=self.polar_pad, grid_has_poles = grid_has_poles)
                
                self.patchrecovery3d = SubPixelConv_3D(self.atmo_resolution, params.patch_size, 2 * embed_dim, self.num_atmo_vars, padded_front = self.patchembed3d.padded_front,
                                                        num_lat = self.atmo_resolution[1], polar_pad=self.polar_pad, grid_has_poles = grid_has_poles)
            else:
                from utils.patch_recovery import SubPixelConvICNR_2D as SubPixelConv_2D
                from utils.patch_recovery import SubPixelConvICNR_3D as SubPixelConv_3D
                self.patchrecovery2d = SubPixelConv_2D(params.horizontal_resolution, params.patch_size[1:], 2 * embed_dim, 
                                                    self.num_surface_vars + self.num_diagnostic_vars + self.num_land_vars + self.num_ocean_vars,
                                                        num_lat = self.atmo_resolution[1], polar_pad=self.polar_pad, grid_has_poles = grid_has_poles)
                self.patchrecovery3d = SubPixelConv_3D(self.atmo_resolution, params.patch_size, 2 * embed_dim, self.num_atmo_vars, padded_front = self.patchembed3d.padded_front,
                                                        num_lat = self.atmo_resolution[1], polar_pad=self.polar_pad, grid_has_poles = grid_has_poles)
            
        else:
            self.patchrecovery2d = PatchRecovery2D(params.horizontal_resolution, params.patch_size[1:], 2 * embed_dim, self.num_surface_vars + self.num_diagnostic_vars + self.num_land_vars + self.num_ocean_vars)
            self.patchrecovery3d = PatchRecovery3D(self.atmo_resolution, params.patch_size, 2 * embed_dim, self.num_atmo_vars)


    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in, train = False, vae_inputs=None):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        
        if self.upper_air_boundary:
            upper_air_varying_boundary = varying_boundary[:,self.idx_upper_air_var_bound, :, :].unsqueeze(1)
            surface_varying_boundary = varying_boundary[:,self.idx_surface_var_bound, :, :]
            surface = torch.cat([surface_in, constant_boundary, surface_varying_boundary], dim=1)
            surface = self.patchembed2d(surface)
            upper_air_varying_boundary = self.patchembed2d_upper_air_boundary(upper_air_varying_boundary)
            upper_air = self.patchembed3d(upper_air_in)
            x = torch.cat([upper_air_varying_boundary.unsqueeze(2), upper_air, surface.unsqueeze(2)], dim=2)
        else:
            surface = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)
            surface = self.patchembed2d(surface)
            upper_air = self.patchembed3d(upper_air_in)
            x = torch.concat([upper_air, surface.unsqueeze(2)], dim=2)


       
        # print(x.size())
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        # x = self.layer1(x)
        # skip = x

        # x = self.downsample(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.upsample(x)
        # x = self.layer4(x)

        #OPTIMIZATION

        if USE_TE:
            # with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            if self.checkpointing == 2 and train:
                x = checkpoint(self.layer1, x, use_reentrant=self.use_reentrant)
            else:
                x = self.layer1(x, train)
            #print(f'Shape after layer 1: {x.shape}')
            #print(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim)[0,-1,:,0,0])
            #print(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim)[0,-1,:,0,0])
            #print(torch.std(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim), dim=(0,3,4)))
            skip = x
            x = self.downsample(x)
            #print(f'Shape after downsample: {x.shape}')
            #print(x.reshape(1, (self.downscale_resolution[0]), self.downscale_resolution[1], 
            #                      self.downscale_resolution[2], self.embed_dim*self.updown_scale_factor)[0,-1,:,0,0])
            #print(x.reshape(1, (self.downscale_resolution[0]), self.downscale_resolution[1], 
            #                      self.downscale_resolution[2], self.embed_dim*self.updown_scale_factor)[0,-1,:,0,0])
            #print(torch.std(x))
            if self.checkpointing == 2 and train:
                x = checkpoint(self.layer2, x, use_reentrant=self.use_reentrant)
                x = checkpoint(self.layer3, x, use_reentrant=self.use_reentrant)
                x = checkpoint(self.upsample, x, use_reentrant=self.use_reentrant)
                x = checkpoint(self.layer4, x, use_reentrant=self.use_reentrant)
            else:
                x = self.layer2(x, train)
                x = self.layer3(x, train)
                x = self.upsample(x)
                #print(f'Shape after upsample: {x.shape}')
                #print(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim)[0,-1,:,0,0])
                #print(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim)[0,-1,:,0,0])
                #print(torch.std(x))
                x = self.layer4(x, train)
        else:
            # with amp.autocast(enabled=True):
            if self.checkpointing == 2 and train:
                x = checkpoint(self.layer1, x, use_reentrant=self.use_reentrant)
            else:
                x = self.layer1(x, train)
            #print(f'Shape after layer 1: {x.shape}')
            #print(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim)[0,-1,:,0,0])
            #print(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim)[0,-1,:,0,0])
            #print(torch.std(x.reshape(1, (self.EST_input_resolution[0]), self.EST_input_resolution[1], self.EST_input_resolution[2], self.embed_dim), dim=(0,3,4)))
            skip = x
            x = self.downsample(x)
            if self.checkpointing == 2 and train:
                x = checkpoint(self.layer2, x, use_reentrant=self.use_reentrant)
                x = checkpoint(self.layer3, x, use_reentrant=self.use_reentrant)
                print("After layer 3 shape:", x.shape)
                x = checkpoint(self.upsample, x, use_reentrant=self.use_reentrant)
                x = checkpoint(self.layer4, x, use_reentrant=self.use_reentrant)
            else:
                x = self.layer2(x, train)
                x = self.layer3(x, train)
                print("After layer 3 shape:", x.shape) #2, 40500, 480

                x = self.upsample(x)
                x = self.layer4(x, train)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)

        if self.predict_delta:
            output_surface_delta  = output[:, :, -1, :, :]
            if self.upper_air_boundary:
                output_upper_air_delta = output[:, :, 1:-1, :, :]
            else:
                output_upper_air_delta = output[:, :, :-1, :, :]
            if self.checkpointing > 0 and train:
                output_2D = checkpoint(self.patchrecovery2d, output_surface_delta, use_reentrant=self.use_reentrant)
            else:
                output_2D = self.patchrecovery2d(output_surface_delta)
            output_surface = output_2D[:, self.surface_prognostic_idxs]
            if self.has_land and self.mask_output:
                output_surface[:, self.num_surface_vars: self.num_surface_vars + self.num_land_vars] = \
                    self.land_mask(output_surface[:, self.num_surface_vars: self.num_surface_vars + self.num_land_vars]).to(output_surface.dtype)
            if self.has_ocean and self.mask_output:
                output_surface[:, self.num_surface_vars + self.num_land_vars:] = \
                    self.land_mask(output_surface[:, self.num_surface_vars + self.num_land_vars:]).to(output_surface.dtype)
            if self.checkpointing > 0 and train:
                output_upper_air = checkpoint(self.patchrecovery3d, output_upper_air_delta, use_reentrant=self.use_reentrant)
            else:
                output_upper_air = self.patchrecovery3d(output_upper_air_delta)
        else:
            output_surface = output[:, :, -1, :, :]
            if self.upper_air_boundary:
                output_upper_air = output[:, :, 1:-1, :, :]
            else:
                output_upper_air = output[:, :, :-1, :, :]
            if self.checkpointing > 0 and train:
                output_2D = checkpoint(self.patchrecovery2d, output_surface, use_reentrant=self.use_reentrant)
            else:
                output_2D = self.patchrecovery2d(output_surface)
            output_surface = output_2D[:, self.surface_prognostic_idxs]
            if self.has_land and self.mask_output:
                output_surface[:, self.num_surface_vars : self.num_surface_vars + self.num_land_vars] = \
                    self.land_mask(output_surface[:, self.num_surface_vars: self.num_surface_vars + self.num_land_vars]).to(output_surface.dtype)
            if self.has_ocean and self.mask_output:
                output_surface[:, self.num_surface_vars + self.num_land_vars:] = \
                    self.land_mask(output_surface[:, self.num_surface_vars + self.num_land_vars:]).to(output_surface.dtype)

            if self.checkpointing > 0 and train:
                output_upper_air = checkpoint(self.patchrecovery3d, output_upper_air, use_reentrant=self.use_reentrant)
            else:
                output_upper_air = self.patchrecovery3d(output_upper_air)
        if self.num_diagnostic_vars > 0:
            output_diagnostic = output_2D[:, self.num_surface_vars:self.num_surface_vars + self.num_diagnostic_vars].reshape(
                output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1])
            return output_surface, output_upper_air, output_diagnostic
        else:
            return output_surface, output_upper_air
        
"""
        
    def integrate(self, surface, upper_air, surface_dx, upper_air_dx):
        if not self.predict_delta:
            raise ValueError('Model is set to predict full field. integrate cannot be called.')
        else:
            output_surface = self.delta_integrator(surface, surface_dx * (self.surface_delta_std / self.surface_ff_std).reshape(1, -1, 1, 1), 1.)
            output_upper_air = self.delta_integrator(upper_air, 
                                                        upper_air_dx * (self.upper_air_delta_std / self.upper_air_ff_std).reshape(1, -1, self.atmo_resolution[0], 1, 1),
                                                        1.)
            return output_surface, output_upper_air
"""
        

'''
PatchEmbed2D and PatchEmbed3D from utils.patch_embed
class PatchEmbedding:
  def __init__(self, patch_size, dim):
    ###Patch embedding operation###
    # Here we use convolution to partition data into cubes
    self.conv = Conv3d(input_dims=5, output_dims=dim, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = Conv2d(input_dims=7, output_dims=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

    # Load constant masks from the disc
    self.land_mask, self.soil_type, self.topography = LoadConstantMask()
    
  def forward(self, input, input_surface):
    # Zero-pad the input
    input = Pad3D(input)
    input_surface = Pad2D(input_surface)

    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
    input = self.conv(input)

    # Add three constant fields to the surface fields
    input_surface =  Concatenate(input_surface, self.land_mask, self.soil_type, self.topography)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    input_surface = self.conv_surface(input_surface)

    # Concatenate the input in the pressure level, i.e., in Z dimension
    x = Concatenate(input, input_surface)

    # Reshape x for calculation of linear projections
    x = TransposeDimensions(x, (0, 2, 3, 4, 1))
    x = reshape(x, target_shape=(x.shape[0], 8*360*181, x.shape[-1]))
    return x
'''


'''
PatchRecovery2D and PatchRecovery3D from utils.patch_recovery
class PatchRecovery:
  def __init__(self, dim):
    ###Patch recovery operation###
    # Hear we use two transposed convolutions to recover data
    self.conv = ConvTranspose3d(input_dims=dim, output_dims=5, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = ConvTranspose2d(input_dims=dim, output_dims=4, kernel_size=patch_size[1:], stride=patch_size[1:])
    
  def forward(self, x, Z, H, W):
    # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
    # Reshape x back to three dimensions
    x = TransposeDimensions(x, (0, 2, 1))
    x = reshape(x, target_shape=(x.shape[0], x.shape[1], Z, H, W))

    # Call the transposed convolution
    output = self.conv(x[:, :, 1:, :, :])
    output_surface = self.conv_surface(x[:, :, 0, :, :])

    # Crop the output to remove zero-paddings
    output = Crop3D(output)
    output_surface = Crop2D(output_surface)
    return output, output_surface
'''

class Mask(nn.Module):
    def __init__(self, mask, mask_fill = None):
        super().__init__() 
        self.mask = nn.parameter.Parameter(mask.unsqueeze(0).unsqueeze(0), requires_grad=False)
        if type(mask_fill) is not type(None):
            self.mask_fill = nn.parameter.Parameter(mask_fill.unsqueeze(0), requires_grad=False)
        else:
            self.mask_fill = None

    def forward(self, x):
        if type(self.mask_fill) is not type(None):
            return x * self.mask + self.mask_fill
        else:
            return x * self.mask
        

class DownSample(nn.Module):
    """
    Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor

        if USE_TE:
            self.linear = te.Linear(in_dim * (self.downsample_factor ** 2), in_dim * self.downsample_factor, bias=False)
            self.norm = te.LayerNorm((self.downsample_factor ** 2) * in_dim)
        else:
            self.linear = nn.Linear(in_dim * (self.downsample_factor ** 2), in_dim * self.downsample_factor, bias=False)
            self.norm = nn.LayerNorm((self.downsample_factor ** 2) * in_dim)


        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * self.downsample_factor - in_lat
        w_pad = out_lon * self.downsample_factor - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, self.downsample_factor, out_lon, self.downsample_factor, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, (self.downsample_factor ** 2) * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class UpSample(nn.Module):
    """
    Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution, upsample_factor=2):
        super().__init__()
        self.upsample_factor = upsample_factor

        if USE_TE:
            self.linear1 = te.Linear(in_dim, out_dim * (upsample_factor ** 2), bias=False)
            self.linear2 = te.Linear(out_dim, out_dim, bias=False)
            self.norm = te.LayerNorm(out_dim)
        else:
            self.linear1 = nn.Linear(in_dim, out_dim * (upsample_factor ** 2), bias=False)
            self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
            self.norm = nn.LayerNorm(out_dim)

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, self.upsample_factor, self.upsample_factor, C // self.upsample_factor).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * self.upsample_factor, in_lon * self.upsample_factor, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * self.upsample_factor - out_lat
        pad_w = in_lon * self.upsample_factor - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top: self.upsample_factor * in_lat - pad_bottom, pad_left: self.upsample_factor * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x
    

  
class EarthSpecificLayer(nn.Module): #BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer = nn.LayerNorm, vertical_windowing = True, checkpointing = 0,
                 use_reentrant = False): # Using TE here is not working. 
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        norm_layer = te.LayerNorm if USE_TE else nn.LayerNorm
        self.checkpointing = checkpointing
        self.use_reentrant = use_reentrant


        self.blocks = nn.ModuleList([
            EarthSpecificBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer, vertical_windowing = vertical_windowing)
            for i in range(depth)
        ])

    def forward(self, x, train = False):
        for blk in self.blocks:
            if self.checkpointing > 2 and train:
                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            else:
                x = blk(x)
        return x



# class EarthSpecificBlock(nn.Module):
#     """
#     3D Transformer Block
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
#         shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     # CHANGE NORM LAYER BACK TO nn.LayerNorm if not faster

#     def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
#                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
#                  norm_layer=te.LayerNorm): # TE Change
#         super().__init__()
#         window_size = (2, 6, 12) if window_size is None else window_size
#         shift_size = (1, 3, 6) if shift_size is None else shift_size
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio

#         self.norm1 = norm_layer(dim)
#         padding = get_pad3d(input_resolution, window_size)
#         self.pad = nn.ZeroPad3d(padding)

#         pad_resolution = list(input_resolution)
#         pad_resolution[0] += (padding[-1] + padding[-2])
#         pad_resolution[1] += (padding[2] + padding[3])
#         pad_resolution[2] += (padding[0] + padding[1])

#         self.attn = EarthAttention3D(
#             dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
#             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
#         )

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         shift_pl, shift_lat, shift_lon = self.shift_size
#         self.roll = shift_pl and shift_lon and shift_lat

#         if self.roll:
#             attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask)

#     def forward(self, x: torch.Tensor):
#         Pl, Lat, Lon = self.input_resolution
#         B, L, C = x.shape
#         assert L == Pl * Lat * Lon, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, Pl, Lat, Lon, C)

#         # start pad
#         x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

#         _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

#         shift_pl, shift_lat, shift_lon = self.shift_size
#         if self.roll:
#             shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
#             x_windows = window_partition(shifted_x, self.window_size)
#             # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
#         else:
#             shifted_x = x
#             x_windows = window_partition(shifted_x, self.window_size)
#             # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

#         win_pl, win_lat, win_lon = self.window_size
#         x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)
#         # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

#         attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)

#         if self.roll:
#             shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
#             # B * Pl * Lat * Lon * C
#             x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
#         else:
#             shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
#             x = shifted_x

#         # crop, end pad
#         x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

#         x = x.reshape(B, Pl * Lat * Lon, C)
#         x = shortcut + self.drop_path(x)

#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x


# CHANGE SO THAT I CAN REPLACE THE EARTHSPECIFIC LAYER NORMALIZATION SCHEME WITH TE. Must be contiguous before applying the normalization. 

class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer = nn.LayerNorm,
                 vertical_windowing = True): 
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        if vertical_windowing:
            shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2) if shift_size is None else shift_size
        else:
            shift_size = (0, window_size[1] // 2, window_size[2] // 2) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        norm_layer = te.LayerNorm if USE_TE else nn.LayerNorm



        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        shift_pl, shift_lat, shift_lon = self.shift_size
        if vertical_windowing:
            self.roll = shift_pl and shift_lon and shift_lat
        else:
            self.roll = shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        # Ensure x is contiguous before normalization (TE CHANGE)
        if USE_TE:
            x = self.norm1(x.contiguous())
        else:
            x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)

        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            # B * Pl * Lat * Lon * C
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        if USE_TE:
            x = x + self.drop_path(self.mlp(self.norm2(x.contiguous())))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


    
class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros((window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                        self.type_of_windows, num_heads)
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH



        earth_position_index = get_earth_position_index(window_size)  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        if USE_TE:
            self.qkv = te.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = te.Linear(dim, dim)
            # self.dpa = DotProductAttention(num_attention_heads=num_heads, kv_channels=dim // num_heads, num_gqa_groups=num_heads)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        # Mem efficient attention doesn't have permute
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = torch.unbind(qkv, 0)
        L = q.shape[-1]

        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows, -1
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1).contiguous().unsqueeze(0)  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        #with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        if mask is not None:
            nLon = mask.shape[0]
            x = F.scaled_dot_product_attention(q.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                                                    k.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                                                    v.view(B_ // nLon, nLon, self.num_heads, nW_, N, L),
                                                    attn_mask=earth_position_bias.unsqueeze(0) + \
                                                        mask.unsqueeze(1).unsqueeze(0),
                                                    scale = self.scale)
            x = x.view(-1, self.num_heads, nW_, N, L)
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=earth_position_bias, scale=self.scale)

        x = x.permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

  
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if USE_TE:
            self.fc1 = te.Linear(in_features, hidden_features)
            self.fc2 = te.Linear(hidden_features, out_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

'''
def PerlinNoise():
  ###Generate random Perlin noise: we follow https://github.com/pvigier/perlin-numpy/ to calculate the perlin noise.###
  # Define number of noise
  octaves = 3
  # Define the scaling factor of noise
  noise_scale = 0.2
  # Define the number of periods of noise along the axis
  period_number = 12
  # The size of an input slice
  H, W = 721, 1440
  # Scaling factor between two octaves
  persistence = 0.5
  # see https://github.com/pvigier/perlin-numpy/ for the implementation of GenerateFractalNoise (e.g., from perlin_numpy import generate_fractal_noise_3d)
  perlin_noise = noise_scale*GenerateFractalNoise((H, W), (period_number, period_number), octaves, persistence)
  return perlin_noise
  '''


class NoiseConv(nn.Module):
    """
    Input:  (B, 10350, 384)
    Output: (B, 40500, 480)

    Uses ConvTranspose1d with:
      stride=4, kernel_size=4, padding=450, output_padding=0
    which gives:
      L_out = (L_in-1)*4 - 2*450 + 4 = 40500
    """

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels=384,
            out_channels=480,
            kernel_size=4,
            stride=4,
            padding=450,
            output_padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 384:
            raise ValueError(f"Expected (B, L, 384), got {tuple(x.shape)}")

        x = x.transpose(1, 2)        # (B, 384, L)
        x = self.up(x)               # (B, 480, 40500)
        x = x.transpose(1, 2)        # (B, 40500, 480)
        return x