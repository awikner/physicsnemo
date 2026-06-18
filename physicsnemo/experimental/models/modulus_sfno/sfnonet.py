# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Tuple

import torch
import torch.nn as nn

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch.utils.checkpoint import checkpoint

from .initialization import trunc_normal_ 

# wrap fft, to unify interface to spectral transforms
# import global convolution and non-linear spectral layers
# helpers
from .layers import MLP, DropPath, RealFFT2, SpectralAttention2d
from .s2convolutions import SpectralAttentionS2, SpectralConvS2 

# layer normalization
try:
    from apex.normalization import FusedLayerNorm

    apex_imported = True
except ImportError:
    apex_imported = False


class SpectralFilterLayer(nn.Module):
    """Spectral filter layer"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="block-diagonal",
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        hidden_size_factor=1,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        drop_rate=0.0,
        spectral_contraction_fp16=False,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                operator_type=operator_type,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
                spectral_contraction_fp16=spectral_contraction_fp16,
            )

        elif filter_type == "non-linear" and (isinstance(forward_transform, RealFFT2)):
            self.filter = SpectralAttention2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        # spectral transform is passed to the module
        elif filter_type == "linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=True,
                use_tensorly=False if factorization is None else True,
                spectral_contraction_fp16=spectral_contraction_fp16,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator Block"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        use_mlp=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=0,
        spectral_contraction_fp16=False,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
        self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate,
            spectral_contraction_fp16=spectral_contraction_fp16,
        )

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()

        if use_mlp == True:
            MLPH = MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def forward(self, x):
        # Use float32 for zero tensor to avoid implicit float16 downcast from norm layers
        # (norm layers return float32 under AMP autocast; assigning into float16 can overflow)
        x_norm = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = self.norm0(
            x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]]
        )
        x, residual = self.filter(x_norm)

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x_norm = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        x_norm[
            ..., : self.output_shape_loc[0], : self.output_shape_loc[1]
        ] = self.norm1(x[..., : self.output_shape_loc[0], : self.output_shape_loc[1]])
        x = x_norm

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(torch.nn.Module):
    """
    Spherical Fourier Neural Operator Network

    Parameters
    ----------
    params : dict
        Dictionary of parameters
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    filter_type : str, optional
        Type of filter to use ('linear', 'non-linear'), by default "non-linear"
    operator_type : str, optional
        Type of operator to use ('diaginal', 'dhconv'), by default "diagonal"
    img_shape : tuple, optional
        Shape of the input channels, by default (721, 1440)
    scale_factor : int, optional
        Scale factor to use, by default 16
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 12
    use_mlp : int, optional
        Whether to use MLP, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    num_blocks : int, optional
        Number of blocks in the network, by default 16
    sparsity_threshold : float, optional
        Threshold for sparsity, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding to apply, by default 1.0
    use_complex_kernels : bool, optional
        Whether to use complex kernels, by default True
    big_skip : bool, optional
        Whether to use big skip connections, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    complex_network : bool, optional
        Whether to use a complex network architecture, by default True
    complex_activation : str, optional
        Type of complex activation function to use, by default "real"
    spectral_layers : int, optional
        Number of spectral layers, by default 3
    checkpointing : int, optional
        Number of checkpointing segments, by default 0

    Example:
    --------
    >>> from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet as SFNO
    >>> model = SFNO(
    ...         params={},
    ...         img_shape=(8, 16),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=2,
    ...         encoder_layers=1,
    ...         num_blocks=4,
    ...         spectral_layers=2,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 8, 16)).shape
    torch.Size([1, 2, 8, 16])
    """

    def __init__(
        self,
        params,
        spectral_transform: str = "sht",
        filter_type: str = "non-linear",
        operator_type: str = "diagonal",
        img_shape: Tuple[int, int] = (721, 1440),
        scale_factor: int = 16,
        in_chans: int = 2,
        out_chans: int = 2,
        embed_dim: int = 256,
        num_layers: int = 12,
        use_mlp: int = True,
        mlp_ratio: float = 2.0,
        activation_function: str = "gelu",
        encoder_layers: int = 1,
        pos_embed: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.0,
        normalization_layer: str = "instance_norm",
        hard_thresholding_fraction: float = 1.0,
        use_complex_kernels: bool = True,
        big_skip: bool = True,
        rank: float = 1.0,
        factorization: Any = None,
        separable: bool = False,
        complex_network: bool = True,
        complex_activation: str = "real",
        spectral_layers: int = 3,
        checkpointing: int = 0,
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.params = params
        self.spectral_transform = (
            params.spectral_transform
            if hasattr(params, "spectral_transform")
            else spectral_transform
        )
        self.filter_type = (
            params.filter_type if hasattr(params, "filter_type") else filter_type
        )
        self.operator_type = (
            params.operator_type if hasattr(params, "operator_type") else operator_type
        )
        self.img_shape = (
            (params.img_shape_x, params.img_shape_y)
            if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y")
            else img_shape
        )
        self.scale_factor = (
            params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        )
        self.in_chans = (
            params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        )
        self.out_chans = (
            params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        )
        self.embed_dim = self.num_features = (
            params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        )
        self.num_layers = (
            params.num_layers if hasattr(params, "num_layers") else num_layers
        )
        self.num_blocks = (
            params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        )
        self.hard_thresholding_fraction = (
            params.hard_thresholding_fraction
            if hasattr(params, "hard_thresholding_fraction")
            else hard_thresholding_fraction
        )
        self.normalization_layer = (
            params.normalization_layer
            if hasattr(params, "normalization_layer")
            else normalization_layer
        )
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.activation_function = (
            params.activation_function
            if hasattr(params, "activation_function")
            else activation_function
        )
        self.encoder_layers = (
            params.encoder_layers
            if hasattr(params, "encoder_layers")
            else encoder_layers
        )
        self.pos_embed = params.pos_embed if hasattr(params, "pos_embed") else pos_embed
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.factorization = (
            params.factorization if hasattr(params, "factorization") else factorization
        )
        self.separable = params.separable if hasattr(params, "separable") else separable
        self.complex_network = (
            params.complex_network
            if hasattr(params, "complex_network")
            else complex_network
        )
        self.complex_activation = (
            params.complex_activation
            if hasattr(params, "complex_activation")
            else complex_activation
        )
        self.spectral_layers = (
            params.spectral_layers
            if hasattr(params, "spectral_layers")
            else spectral_layers
        )
        self.checkpointing = (
            params.checkpointing if hasattr(params, "checkpointing") else checkpointing
        )
        # When True the spectral weight contraction runs under
        # autocast(dtype=fp16); when False it runs under autocast(enabled=False)
        # so the contraction stays in fp32.  Defaults to False; the SHT/iSHT
        # always run in fp32 regardless of this flag.
        self.spectral_contraction_fp16 = bool(getattr(params, "spectral_contraction_fp16", False))
        data_grid = params.data_grid if hasattr(params, "data_grid") else "equiangular"
        
        print('data_grid',data_grid)
        # self.pretrain_encoding = params.pretrain_encoding if hasattr(params, "pretrain_encoding") else False

        # compute the downscaled image size
        self.h = int(self.img_shape[0] // self.scale_factor)
        self.w = int(self.img_shape[1] // self.scale_factor)

        # Compute the maximum frequencies in h and in w
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

        # no global padding because we removed the horizontal distributed code
        self.padding = (0, 0)

        # prepare the spectral transforms
        if self.spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # set up
            self.trans_down = sht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
            ).float()
            self.itrans_up = isht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
            ).float()
            self.trans = sht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()
            self.itrans = isht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()

        elif self.spectral_transform == "fft":
            fft_handle = th.RealFFT2
            ifft_handle = th.InverseRealFFT2

            # effective image size:
            self.img_shape_eff = (
                self.img_shape[0] + self.padding[0],
                self.img_shape[1] + self.padding[1],
            )
            self.img_shape_loc = (
                self.img_shape_eff[0],
                self.img_shape_eff[1],
            )

            self.trans_down = fft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = ifft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.trans = fft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans = ifft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        self.img_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
        self.img_shape_eff = (self.trans_down.nlat, self.trans_down.nlon)
        self.h_loc = self.itrans.nlat
        self.w_loc = self.itrans.nlon

        # determine activation function
        if self.activation_function == "relu":
            self.activation_function = nn.ReLU
        elif self.activation_function == "gelu":
            self.activation_function = nn.GELU
        elif self.activation_function == "silu":
            self.activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {self.activation_function}")

        # encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(
                nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            )
            encoder_modules.append(self.activation_function())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, self.embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(
                nn.LayerNorm,
                normalized_shape=(self.img_shape_loc[0], self.img_shape_loc[1]),
                eps=1e-6,
            )
            norm_layer1 = partial(
                nn.LayerNorm, normalized_shape=(self.h_loc, self.w_loc), eps=1e-6
            )
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(
                nn.InstanceNorm2d,
                num_features=self.embed_dim,
                eps=1e-6,
                affine=True,
                track_running_stats=False,
            )
            norm_layer1 = norm_layer0
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(
                f"Error, normalization {self.normalization_layer} not implemented."
            )

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear"
            outer_skip = "identity"

            if first_layer:
                norm_layer = (norm_layer0, norm_layer1)
            elif last_layer:
                norm_layer = (norm_layer1, norm_layer0)
            else:
                norm_layer = (norm_layer1, norm_layer1)

            filter_type = self.filter_type

            operator_type = self.operator_type

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                rank=self.rank,
                factorization=self.factorization,
                separable=self.separable,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
                spectral_contraction_fp16=self.spectral_contraction_fp16,
            )

            self.blocks.append(block)

        # decoder
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(
                nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            )
            decoder_modules.append(self.activation_function())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv2d(current_dim, self.out_chans, 1, bias=False))
        self.decoder = nn.Sequential(*decoder_modules)

        # learned position embedding
        if self.pos_embed:
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.embed_dim, self.img_shape_loc[0], self.img_shape_loc[1]
                )
            )
            # self.pos_embed = nn.Parameter( torch.zeros(1, self.embed_dim, self.img_shape_eff[0], self.img_shape_eff[1]) )
            self.pos_embed.is_shared_mp = ["matmul"]
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or (
            apex_imported and isinstance(m, FusedLayerNorm)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):  # pragma: no cover
        """Helper"""
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):
        for blk in self.blocks:
            if self.checkpointing >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return x

    def forward(self, x, return_latent=False):
        # save big skip
        if self.big_skip:
            residual = x

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):
            if self.img_shape_loc == self.img_shape_eff:
                # Shapes match: simple addition in float32
                x = x + self.pos_embed
            else:
                # Shapes differ: zero-pad and add in float32 to prevent overflow
                xp = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
                xp[..., : self.img_shape_loc[0], : self.img_shape_loc[1]] = (
                    x[..., : self.img_shape_loc[0], : self.img_shape_loc[1]].float()
                    + self.pos_embed
                )
                x = xp

        # maybe clean the padding just in case

        x = self.pos_drop(x)

        latent = x.detach().clone() if return_latent else None

        x = self._forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)

        if return_latent:
            return x, latent
        return x


class Params:
    def __init__(self, grid):
        self.data_grid = grid


class SphericalFourierNeuralOperatorNet_v2(SphericalFourierNeuralOperatorNet):
    """
    Same model class as SphericalFourierNeuralOperatorNet with 2 changes:
    (1) takes in params structure from trainer during initialization
    (2) reshapes input before feeding to model.forward call and reshapes model 
        output before outputting final answer.
    """
    def __init__(
        self,
        params_trainer,
        dataset
    ):

        if getattr(params_trainer, 'use_legacy_data_grid', False):
            grid_type = Params('legendre-gauss')
        else:
            grid_type = Params(getattr(params_trainer, 'data_grid', 'equiangular'))
        img_shape = (len(params_trainer['lat']), len(params_trainer['lon']))

        # Calculate in_chans and out_chans
        # Use a copy to avoid mutating the dataset's variable_list_in
        vars_in = list(dataset.variable_list_in)
        vars_in.extend(dataset.constant_boundary_variables)
        vars_out = dataset.variable_list_out
        in_chans, out_chans = len(vars_in), len(vars_out)

        self.has_diagnostic = params_trainer.has_diagnostic

        # Initialize SphericalFourierNeuralOperatorNet class
        super(SphericalFourierNeuralOperatorNet_v2, self).__init__(
            params=grid_type,
            spectral_transform = params_trainer.spectral_transform,
            filter_type = params_trainer.filter_type,
            operator_type = params_trainer.operator_type,
            img_shape = img_shape,
            scale_factor = params_trainer.scale_factor,
            in_chans = in_chans,
            out_chans = out_chans,
            embed_dim = params_trainer.embed_dim,
            num_layers = params_trainer.num_layers,
            use_mlp = params_trainer.use_mlp,
            mlp_ratio = params_trainer.mlp_ratio,
            activation_function = params_trainer.activation_function,
            encoder_layers = params_trainer.encoder_layers,
            pos_embed = params_trainer.pos_embed,
            drop_rate = params_trainer.drop_rate,
            drop_path_rate = params_trainer.drop_path_rate,
            num_blocks = params_trainer.num_blocks,
            sparsity_threshold = params_trainer.sparsity_threshold,
            normalization_layer = params_trainer.normalization_layer,
            hard_thresholding_fraction = params_trainer.hard_thresholding_fraction,
            use_complex_kernels = params_trainer.use_complex_kernels,
            big_skip = params_trainer.big_skip,
            rank = params_trainer.rank,
            factorization = params_trainer.factorization,
            separable = params_trainer.separable,
            complex_network = params_trainer.complex_network,
            complex_activation = params_trainer.complex_activation,
            spectral_layers = params_trainer.spectral_layers,
            checkpointing = params_trainer.checkpointing,
        )

    def forward(self, surface, c_boundary, v_boundary, upper_air, train=None, return_latent=False):
        # surface/c_boundary/v_boundary [b, c, x, y]
        # upper_air [b, c, z, x, y]

        b, nlat, nlon = surface.shape[0], surface.shape[-2], surface.shape[-1]

        #print(f'surface.shape: {surface.shape}')
        surface_vars = surface.shape[1]
        c_boundary_vars = c_boundary.shape[1]
        v_boundary_vars = v_boundary.shape[1]
        upper_air_vars = upper_air.shape[1]
        upper_air_levels = upper_air.shape[2]
        #print(f'c_boundary.shape: {c_boundary.shape}')
        #print(f'v_boundary.shape: {v_boundary.shape}')
        #print(f'upper_air.shape: {upper_air.shape}')

        # Reshape input to [b, c, x, y]
        upper_air = upper_air.view(b, upper_air_vars * upper_air_levels, nlat, nlon)
        #print(f'upper_air reshaped into: {upper_air.shape}')
        x = torch.cat((surface, c_boundary[:surface.shape[0]], v_boundary, upper_air), dim=1)
        #print(f'input shape into base SFNO: {x.shape}')

        if return_latent:
            x, latent = super(SphericalFourierNeuralOperatorNet_v2, self).forward(x, return_latent=True)
        else:
            x = super(SphericalFourierNeuralOperatorNet_v2, self).forward(x)
            latent = None
        #print(f'output shape of base SFNO: {x.shape}')

        # Reshape output
        surface = x[:, :surface_vars, :, :]
        upper_air = x[:, surface_vars:(surface_vars+upper_air_vars*upper_air_levels), :, :]
        upper_air = upper_air.view(b, upper_air_vars, upper_air_levels, nlat, nlon)
        #print(f'surface.shape: {surface.shape}')
        #print(f'upper_air.shape: {upper_air.shape}')
        if self.has_diagnostic:
            diagnostic = x[:, (surface_vars+upper_air_vars*upper_air_levels):, :, :]
            result = (surface, upper_air, diagnostic, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        else:
            result = (surface, upper_air, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

        if return_latent:
            return result + (latent,)
        return result