# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

r"""Faithful port of `PanguWeather v2.0
<https://github.com/198808xc/Pangu-Weather>`_ ``PanguModel_Plasim`` (the
current VAE-augmented 3D Swin / Earth-Specific transformer) into a
:class:`~physicsnemo.core.module.Module`.

Every ``nn.Module`` submodule name, parameter, and buffer name matches the
source so the Phase-5 checkpoint translator can load original PanguWeather
weights 1:1. The intentional departures from the source are:

* constructor takes explicit JSON-serializable keyword arguments instead of a
  ``params``/``YParams`` blob (required by :class:`~physicsnemo.core.module.Module`);
* land/ocean ``Mask`` buffers — persisted in the checkpoint — are zero
  placeholders here, populated by the checkpoint load or by
  :meth:`PanguPlasim.set_land_ocean_masks`;
* the broken/dead ``USE_TE`` opt-in (used ``te.Linear`` without importing
  ``transformer_engine``) is omitted — see :mod:`.layers`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import nn
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

from ._pangu_utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)
from .layers import DownSample, EarthSpecificLayer, Mask, UpSample


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False  # dynamic control flow + activation checkpointing
    amp: bool = True
    amp_gpu: bool = True
    bf16: bool = True
    # Inference
    onnx: bool = False


class PanguPlasim(Module):
    r"""Pangu_Plasim weather emulator (faithful port of PanguWeather v2.0).

    Pangu-Weather-style 3D Swin / Earth-Specific transformer with a
    training-only VAE dual-encoder. Surface and upper-air fields are processed
    as separate streams; constant and time-varying boundary conditions are
    concatenated into the surface stream, with the top-of-atmosphere (TOA)
    solar-radiation varying boundary routed into the upper-air stream when
    ``upper_air_boundary=True``.

    Based on `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate
    Global Weather Forecast <https://arxiv.org/abs/2211.02556>`_; ported from
    `the PanguWeather v2.0 fork
    <https://github.com/198808xc/Pangu-Weather>`_. See
    :class:`PanguPlasimLegacy` for the no-VAE predecessor architecture.

    Parameters
    ----------
    surface_variables : list of str
        Prognostic surface variable names; sets the surface input channel order.
    upper_air_variables : list of str
        Prognostic upper-air variable names; sets the upper-air input channel order.
    constant_boundary_variables : list of str
        Static boundary variable names (e.g., land-sea mask, topography).
    varying_boundary_variables : list of str
        Time-varying boundary variable names. Must contain a solar-radiation
        field named ``"rsdt"`` or ``"toa_incident_solar_radiation"``.
    levels : list
        Vertical levels for the upper-air stream; ``len(levels)`` sets the 3D depth.
    horizontal_resolution : list of int
        ``[n_lat, n_lon]`` of the lat-lon grid.
    patch_size : list of int
        Patch size ``[p_level, p_lat, p_lon]``.
    depths : list of int, optional, default=(2, 6, 6, 2)
        Transformer depths per stage.
    num_heads : tuple of int, optional, default=(6, 12, 12, 6)
        Attention heads per stage.
    embed_dim : int, optional, default=192
        Patch-embedding dimension.
    window_size : list of int, optional, default=(2, 6, 12)
        ``[w_level, w_lat, w_lon]`` 3D attention-window size.
    updown_scale_factor : int, optional, default=2
        Spatial down/up-sample factor between stages.
    vertical_windowing : bool, optional, default=True
        Whether windows shift along the vertical (pressure) axis.
    upper_air_boundary : bool, optional, default=False
        Route the solar-radiation varying boundary into the upper-air stream.
    predict_delta : bool, optional, default=False
        Predict normalized tendencies (to be integrated externally) instead of
        full fields.
    land_variables : list of str, optional, default=()
        Land-only variable names.
    ocean_variables : list of str, optional, default=()
        Ocean-only variable names.
    diagnostic_variables : list of str, optional, default=()
        Output-only diagnostic variable names.
    has_diagnostic : bool or None, optional, default=None
        Override for whether diagnostics are produced. Defaults to
        ``len(diagnostic_variables) > 0``.
    mask_output : bool, optional, default=False
        Apply land/ocean masking to the corresponding output channels.
    mask_fill : dict or None, optional, default=None
        Per-variable additive fill values for the land/ocean masks
        (used by :meth:`set_land_ocean_masks` for from-scratch training; ignored
        when loading a translated checkpoint).
    drop_rate : float, optional, default=0.0
        Dropout rate. When ``> 0`` the stochastic-depth schedule is forced to
        zero (matches the original source).
    drop_path : list of float or None, optional, default=None
        Stochastic-depth schedule. When ``None`` a piecewise-linear default is
        used.
    subpixel_deconv : bool, optional, default=False
        Use sub-pixel-convolution patch-recovery heads instead of the linear
        patch-recovery.
    polar_pad : bool, optional, default=False
        Use polar padding in the patch-recovery heads.
    grid_has_poles : bool, optional, default=False
        Whether the grid includes pole latitudes.
    recovery_head : bool, optional, default=False
        Use the ``_wHead`` variant of the sub-pixel-conv patch recovery.
    diagnostic_head : bool, optional, default=False
        Give the diagnostic outputs their own recovery head.
    checkpointing : int, optional, default=0
        Activation-checkpointing depth (0 disables).
    use_reentrant : bool, optional, default=False
        ``use_reentrant`` flag for :func:`torch.utils.checkpoint.checkpoint`.

    Forward
    -------
    surface_in : torch.Tensor
        Prognostic surface fields of shape :math:`(B, C_s, H, W)`.
    constant_boundary : torch.Tensor
        Static boundary fields of shape :math:`(C_b^c, H, W)` or
        :math:`(B, C_b^c, H, W)`. The leading batch dim is added if absent.
    varying_boundary : torch.Tensor
        Time-varying boundary fields of shape :math:`(B, C_b^v, H, W)`. Must
        include the solar-radiation channel.
    upper_air_in : torch.Tensor
        Prognostic upper-air fields of shape :math:`(B, C_u, L, H, W)`, where
        :math:`L=` ``len(levels)``.
    target_surface : torch.Tensor, optional
        Surface targets of shape :math:`(B, C_s, H, W)` for the VAE encoder-2
        branch; required when ``train=True``.
    target_upper_air : torch.Tensor, optional
        Upper-air targets of shape :math:`(B, C_u, L, H, W)` for the VAE
        encoder-2 branch; required when ``train=True``.
    train : bool, optional, default=False
        Activates the VAE encoder-2 branch (uses ``target_surface``,
        ``target_upper_air``); independent of ``self.training``.
    return_latent : bool, optional, default=False
        Append the post-downsample bottleneck latent to the return tuple.

    Outputs
    -------
    tuple of torch.Tensor
        ``(out_surface, out_upper_air[, out_diagnostic], mu, sigma, mu2, sigma2)``
        — the ``out_diagnostic`` element is present only when
        ``diagnostic_variables`` is non-empty. ``mu``, ``sigma`` are the
        encoder-1 VAE statistics; ``mu2``, ``sigma2`` carry the encoder-2
        statistics in ``train`` mode and are zero-tensor placeholders at
        evaluation. When ``return_latent=True`` the bottleneck latent is
        appended at the end.

    Notes
    -----
    The model is **stochastic in eval mode by design**: :meth:`reparameterize`
    calls :func:`torch.randn_like` unconditionally, so repeated forward passes
    on the same input draw different VAE samples. Tests asserting checkpoint
    fidelity must seed ``torch.manual_seed()`` identically before each forward.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.pangu_plasim import PanguPlasim
    >>> model = PanguPlasim(
    ...     surface_variables=["t2m", "u10", "v10"],
    ...     upper_air_variables=["t", "u", "v", "q", "z"],
    ...     constant_boundary_variables=["lsm"],
    ...     varying_boundary_variables=["rsdt"],
    ...     levels=[200, 300, 500, 700, 850, 925, 1000, 1015],
    ...     horizontal_resolution=[32, 64],
    ...     patch_size=[2, 4, 4],
    ...     depths=[1, 1, 1, 1],
    ...     num_heads=[2, 4, 4, 2],
    ...     embed_dim=64,
    ...     window_size=[2, 4, 8],
    ... ).eval()
    >>> surface_in = torch.randn(1, 3, 32, 64)
    >>> constant_boundary = torch.randn(1, 32, 64)
    >>> varying_boundary = torch.randn(1, 1, 32, 64)
    >>> upper_air_in = torch.randn(1, 5, 8, 32, 64)
    >>> with torch.no_grad():
    ...     out = model(surface_in, constant_boundary, varying_boundary, upper_air_in)
    >>> out[0].shape, out[1].shape
    (torch.Size([1, 3, 32, 64]), torch.Size([1, 5, 8, 32, 64]))
    """

    __model_checkpoint_version__ = "1.0"

    def __init__(
        self,
        *,
        surface_variables: list,
        upper_air_variables: list,
        constant_boundary_variables: list,
        varying_boundary_variables: list,
        levels: list,
        horizontal_resolution: list,
        patch_size: list,
        depths: list = (2, 6, 6, 2),
        num_heads: tuple = (6, 12, 12, 6),
        embed_dim: int = 192,
        window_size: list = (2, 6, 12),
        updown_scale_factor: int = 2,
        vertical_windowing: bool = True,
        upper_air_boundary: bool = False,
        predict_delta: bool = False,
        land_variables: list = (),
        ocean_variables: list = (),
        diagnostic_variables: list = (),
        has_diagnostic: bool = None,
        mask_output: bool = False,
        mask_fill: dict = None,
        drop_rate: float = 0.0,
        drop_path: list = None,
        subpixel_deconv: bool = False,
        polar_pad: bool = False,
        grid_has_poles: bool = False,
        recovery_head: bool = False,
        diagnostic_head: bool = False,
        checkpointing: int = 0,
        use_reentrant: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())

        self.checkpointing = checkpointing
        self.use_reentrant = use_reentrant
        self.embed_dim = embed_dim

        depths = list(depths)
        depths_cumsum = np.cumsum(depths).astype(int)

        if not drop_path:
            drop_path = np.append(
                np.linspace(0, 0.2, int(np.sum(depths[:2]))),
                np.linspace(0.2, 0, int(np.sum(depths[2:]))),
            ).tolist()
        if drop_rate > 0.0:
            drop_path = np.zeros(int(np.sum(depths))).tolist()

        # --- channel / resolution bookkeeping (set predict_delta early so the
        # land/ocean mask construction below can branch on it) ---
        self.predict_delta = predict_delta
        self.mask_output = mask_output

        self.num_surface_vars = len(surface_variables)
        self.num_atmo_vars = len(upper_air_variables)
        self.num_boundary_vars = len(constant_boundary_variables) + len(
            varying_boundary_variables
        )
        self.atmo_resolution = [len(levels)] + list(horizontal_resolution)

        self.diagnostic_vars = list(diagnostic_variables)
        self.num_diagnostic_vars = len(self.diagnostic_vars)
        if has_diagnostic is None:
            self.has_diagnostic = self.num_diagnostic_vars > 0
        else:
            self.has_diagnostic = has_diagnostic

        self.has_land = len(land_variables) > 0
        self.num_land_vars = len(land_variables)
        self.has_ocean = len(ocean_variables) > 0
        self.num_ocean_vars = len(ocean_variables)
        self._land_variables = list(land_variables)
        self._ocean_variables = list(ocean_variables)

        # Non-persistent buffer: moves to the model's device but is kept out of
        # the state_dict (the original stored this as a plain CPU attribute, so
        # excluding it preserves checkpoint-key compatibility).
        self.register_buffer(
            "surface_prognostic_idxs",
            torch.cat(
                (
                    torch.arange(self.num_surface_vars).long(),
                    torch.arange(
                        self.num_surface_vars + self.num_diagnostic_vars,
                        self.num_surface_vars
                        + self.num_diagnostic_vars
                        + self.num_land_vars
                        + self.num_ocean_vars,
                    ).long(),
                )
            ),
            persistent=False,
        )

        # --- land / ocean output masks -------------------------------------
        # ``Mask`` stores its (non-trainable) buffers in the state_dict, so they
        # arrive from the checkpoint. Here we build correctly-shaped zero
        # placeholders; ``set_land_ocean_masks`` populates real values for
        # from-scratch training.
        n_lat, n_lon = horizontal_resolution
        if self.has_land and self.mask_output:
            if self.predict_delta:
                self.land_mask = Mask(torch.zeros(n_lat, n_lon))
            else:
                self.land_mask = Mask(
                    torch.zeros(n_lat, n_lon),
                    torch.zeros(self.num_land_vars, n_lat, n_lon),
                )
        if self.has_ocean and self.mask_output:
            if self.predict_delta:
                self.ocean_mask = Mask(torch.zeros(n_lat, n_lon))
            else:
                self.ocean_mask = Mask(
                    torch.zeros(n_lat, n_lon),
                    torch.zeros(self.num_ocean_vars, n_lat, n_lon),
                )
        self._mask_fill = mask_fill

        # --- geometry / windowing ------------------------------------------
        self.window_size = list(window_size)
        self.vertical_windowing = vertical_windowing
        self.updown_scale_factor = updown_scale_factor
        self.subpixel_deconv = subpixel_deconv
        self.polar_pad = polar_pad
        self._grid_has_poles = grid_has_poles
        self.recovery_head = recovery_head
        self.diagnostic_head = diagnostic_head

        # --- varying-boundary routing (solar radiation -> 3D stream) --------
        self.upper_air_boundary = upper_air_boundary
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.num_varying_boundary_vars = len(varying_boundary_variables)
        _solar_names = ("rsdt", "toa_incident_solar_radiation")
        self.idx_upper_air_var_bound = next(
            (
                self.varying_boundary_variables.index(n)
                for n in _solar_names
                if n in self.varying_boundary_variables
            ),
            None,
        )
        if self.idx_upper_air_var_bound is None:
            raise ValueError(
                f"varying_boundary_variables {self.varying_boundary_variables} "
                f"must contain one of {_solar_names}"
            )
        self.idx_surface_var_bound = [
            i
            for i in range(self.num_varying_boundary_vars)
            if i != self.idx_upper_air_var_bound
        ]

        # --- patch embeddings ----------------------------------------------
        if self.upper_air_boundary:
            self.patchembed2d_upper_air_boundary = PatchEmbed2D(
                img_size=horizontal_resolution,
                patch_size=patch_size[1:],
                in_chans=1,
                embed_dim=embed_dim,
            )

        self.patchembed2d = PatchEmbed2D(
            img_size=horizontal_resolution,
            patch_size=patch_size[1:],
            in_chans=self.num_surface_vars
            + self.num_land_vars
            + self.num_ocean_vars
            + self.num_boundary_vars
            - 1 * self.upper_air_boundary,
            embed_dim=embed_dim,
        )

        self.patchembed3d = PatchEmbed3D(
            img_size=self.atmo_resolution,
            patch_size=patch_size,
            in_chans=self.num_atmo_vars,
            embed_dim=embed_dim,
        )

        self.in_chans = (
            self.num_surface_vars
            + self.num_land_vars
            + self.num_ocean_vars
            + self.num_boundary_vars
            - 1 * self.upper_air_boundary
            + self.num_atmo_vars
            + (1 if self.upper_air_boundary else 0)
        )
        self.out_chans = (
            self.num_surface_vars
            + self.num_diagnostic_vars
            + self.num_land_vars
            + self.num_ocean_vars
            + self.num_atmo_vars
        )

        EST_input_resolution = (
            self.patchembed3d.output_size[0] + 1 + 1 * self.upper_air_boundary,
            self.patchembed3d.output_size[1],
            self.patchembed3d.output_size[2],
        )
        downscale_resolution = (
            self.patchembed3d.output_size[0] + 1 + 1 * self.upper_air_boundary,
            (self.patchembed2d.output_size[0] - self.patchembed2d.output_size[0] % updown_scale_factor)
            // updown_scale_factor
            + self.patchembed2d.output_size[0] % updown_scale_factor,
            (self.patchembed2d.output_size[1] - self.patchembed2d.output_size[1] % updown_scale_factor)
            // updown_scale_factor
            + self.patchembed2d.output_size[1] % updown_scale_factor,
        )
        self.downscale_resolution = downscale_resolution
        self.EST_input_resolution = EST_input_resolution

        if not self.vertical_windowing:
            self.window_size[0] = EST_input_resolution[0]

        # --- encoder 1 ------------------------------------------------------
        self.layer1 = EarthSpecificLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.window_size,
            drop_path=drop_path[: depths_cumsum[0]],
            vertical_windowing=vertical_windowing,
        )
        self.downsample = DownSample(
            in_dim=embed_dim,
            input_resolution=EST_input_resolution,
            output_resolution=downscale_resolution,
            downsample_factor=updown_scale_factor,
        )
        self.layer2 = EarthSpecificLayer(
            dim=embed_dim * updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[0] : depths_cumsum[1]],
            vertical_windowing=vertical_windowing,
            drop=drop_rate,
        )
        self.layer3 = EarthSpecificLayer(
            dim=embed_dim * updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[1] : depths_cumsum[2]],
            vertical_windowing=vertical_windowing,
            drop=drop_rate,
        )

        # --- VAE part (encoder 1) ------------------------------------------
        self.layer_mu = nn.Conv3d(
            in_channels=self.embed_dim * updown_scale_factor,
            out_channels=self.embed_dim,
            kernel_size=1,
        )
        self.layer_sigma = nn.Conv3d(
            in_channels=self.embed_dim * updown_scale_factor,
            out_channels=self.embed_dim,
            kernel_size=1,
        )
        self.layer_purturbation = nn.Conv3d(
            in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=1
        )
        self.layer_perturbation2 = nn.Conv3d(
            in_channels=embed_dim + embed_dim * updown_scale_factor,
            out_channels=embed_dim * updown_scale_factor,
            kernel_size=1,
        )

        # --- 2nd encoder (training-only VAE branch) ------------------------
        self.layer1_e2 = EarthSpecificLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.window_size,
            drop_path=drop_path[: depths_cumsum[0]],
            vertical_windowing=vertical_windowing,
        )
        self.layer2_e2 = EarthSpecificLayer(
            dim=embed_dim * updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[0] : depths_cumsum[1]],
            vertical_windowing=vertical_windowing,
            drop=drop_rate,
        )
        self.layer3_e3 = EarthSpecificLayer(
            dim=embed_dim * updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[1] : depths_cumsum[2]],
            vertical_windowing=vertical_windowing,
            drop=drop_rate,
        )
        self.downsample_e2 = DownSample(
            in_dim=embed_dim,
            input_resolution=EST_input_resolution,
            output_resolution=downscale_resolution,
            downsample_factor=updown_scale_factor,
        )
        self.layer_mu_e2 = nn.Conv3d(
            in_channels=self.embed_dim * updown_scale_factor,
            out_channels=self.embed_dim,
            kernel_size=1,
        )
        self.layer_sigma_e2 = nn.Conv3d(
            in_channels=self.embed_dim * updown_scale_factor,
            out_channels=self.embed_dim,
            kernel_size=1,
        )
        self.layer_purturbation_e2 = nn.Conv3d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=1
        )

        # --- decoder --------------------------------------------------------
        self.upsample = UpSample(
            embed_dim * updown_scale_factor,
            embed_dim,
            downscale_resolution,
            (
                self.patchembed3d.output_size[0] + 1 + 1 * self.upper_air_boundary,
                self.patchembed3d.output_size[1],
                self.patchembed3d.output_size[2],
            ),
        )
        self.layer4 = EarthSpecificLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=self.window_size,
            drop_path=drop_path[depths_cumsum[2] :],
            vertical_windowing=vertical_windowing,
        )

        # --- patch recovery -------------------------------------------------
        if self.subpixel_deconv:
            if self.recovery_head:
                from ._pangu_utils import (
                    SubPixelConvICNR_2D_wHead as SubPixelConv_2D,
                )
                from ._pangu_utils import (
                    SubPixelConvICNR_3D_wHead as SubPixelConv_3D,
                )

                if self.diagnostic_head:
                    self.patchrecovery2d = SubPixelConv_2D(
                        horizontal_resolution,
                        patch_size[1:],
                        2 * embed_dim,
                        self.num_surface_vars,
                        diagnostic_variables=self.num_diagnostic_vars,
                        diagnostic_head=self.diagnostic_head,
                        land_variables=self.num_land_vars,
                        ocean_variables=self.num_ocean_vars,
                        num_lat=self.atmo_resolution[1],
                        polar_pad=self.polar_pad,
                        grid_has_poles=grid_has_poles,
                    )
                else:
                    self.patchrecovery2d = SubPixelConv_2D(
                        horizontal_resolution,
                        patch_size[1:],
                        2 * embed_dim,
                        self.num_surface_vars + self.num_diagnostic_vars,
                        diagnostic_variables=0,
                        diagnostic_head=self.diagnostic_head,
                        land_variables=self.num_land_vars,
                        ocean_variables=self.num_ocean_vars,
                        num_lat=self.atmo_resolution[1],
                        polar_pad=self.polar_pad,
                        grid_has_poles=grid_has_poles,
                    )
                self.patchrecovery3d = SubPixelConv_3D(
                    self.atmo_resolution,
                    patch_size,
                    2 * embed_dim,
                    self.num_atmo_vars,
                    padded_front=self.patchembed3d.padded_front,
                    num_lat=self.atmo_resolution[1],
                    polar_pad=self.polar_pad,
                    grid_has_poles=grid_has_poles,
                )
            else:
                from ._pangu_utils import SubPixelConvICNR_2D as SubPixelConv_2D
                from ._pangu_utils import SubPixelConvICNR_3D as SubPixelConv_3D

                self.patchrecovery2d = SubPixelConv_2D(
                    horizontal_resolution,
                    patch_size[1:],
                    2 * embed_dim,
                    self.num_surface_vars
                    + self.num_diagnostic_vars
                    + self.num_land_vars
                    + self.num_ocean_vars,
                    num_lat=self.atmo_resolution[1],
                    polar_pad=self.polar_pad,
                    grid_has_poles=grid_has_poles,
                )
                self.patchrecovery3d = SubPixelConv_3D(
                    self.atmo_resolution,
                    patch_size,
                    2 * embed_dim,
                    self.num_atmo_vars,
                    padded_front=self.patchembed3d.padded_front,
                    num_lat=self.atmo_resolution[1],
                    polar_pad=self.polar_pad,
                    grid_has_poles=grid_has_poles,
                )
        else:
            self.patchrecovery2d = PatchRecovery2D(
                horizontal_resolution,
                patch_size[1:],
                2 * embed_dim,
                self.num_surface_vars
                + self.num_diagnostic_vars
                + self.num_land_vars
                + self.num_ocean_vars,
            )
            self.patchrecovery3d = PatchRecovery3D(
                self.atmo_resolution, patch_size, 2 * embed_dim, self.num_atmo_vars
            )

    @torch.no_grad()
    def set_land_ocean_masks(
        self,
        land_mask: Float[torch.Tensor, "lat lon"],
        mask_fill: Optional[dict] = None,
    ) -> None:
        r"""Populate the land/ocean ``Mask`` buffers for from-scratch training.

        Reproduces the original PanguWeather mask construction: ``land_mask`` is
        a :math:`(H, W)` land-sea field, and (in non-delta mode) the per-variable
        fill is :math:`(1 - \text{mask}) \cdot \text{mask\_fill}[v]`. When
        loading a translated checkpoint this call is unnecessary — the buffers
        arrive from the checkpoint.

        Parameters
        ----------
        land_mask : torch.Tensor
            Land-sea field of shape :math:`(H, W)`. ``NaN`` entries are zeroed.
        mask_fill : dict or None, optional, default=None
            Per-variable additive fill values. When ``None``, the constructor's
            ``mask_fill`` argument is used.

        Returns
        -------
        None
        """
        mask_fill = mask_fill if mask_fill is not None else self._mask_fill
        land_mask = land_mask.to(torch.float32)
        land_mask = land_mask.masked_fill_(torch.isnan(land_mask), 0.0)
        if self.has_land and self.mask_output:
            self.land_mask.mask.copy_(land_mask.unsqueeze(0).unsqueeze(0))
            if not self.predict_delta:
                fill = torch.stack(
                    [(1.0 - land_mask) * mask_fill[v] for v in self._land_variables]
                )
                self.land_mask.mask_fill.copy_(fill.unsqueeze(0))
        if self.has_ocean and self.mask_output:
            ocean_mask = 1.0 - land_mask
            self.ocean_mask.mask.copy_(ocean_mask.unsqueeze(0).unsqueeze(0))
            if not self.predict_delta:
                fill = torch.stack(
                    [(1.0 - ocean_mask) * mask_fill[v] for v in self._ocean_variables]
                )
                self.ocean_mask.mask_fill.copy_(fill.unsqueeze(0))

    def reparameterize(
        self,
        mu: Float[torch.Tensor, "batch d level lat lon"],
        sigma: Float[torch.Tensor, "batch d level lat lon"],
    ) -> Float[torch.Tensor, "batch d level lat lon"]:
        r"""Sample from the VAE latent distribution via the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian, shape :math:`(B, D, L', H', W')`.
        sigma : torch.Tensor
            Log-variance of the latent Gaussian, shape :math:`(B, D, L', H', W')`.

        Returns
        -------
        torch.Tensor
            Sample of shape :math:`(B, D, L', H', W')`. **Always stochastic**
            (calls :func:`torch.randn_like` regardless of ``self.training``);
            seed externally for reproducibility.
        """
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        surface_in: Float[torch.Tensor, "batch n_surface lat lon"],
        constant_boundary: torch.Tensor,
        varying_boundary: Float[torch.Tensor, "batch n_varying lat lon"],
        upper_air_in: Float[torch.Tensor, "batch n_upper n_levels lat lon"],
        target_surface: Optional[Float[torch.Tensor, "batch n_surface lat lon"]] = None,
        target_upper_air: Optional[
            Float[torch.Tensor, "batch n_upper n_levels lat lon"]
        ] = None,
        train: bool = False,
        return_latent: bool = False,
    ):
        r"""See class docstring (:class:`PanguPlasim`) for the full description.

        ``constant_boundary`` accepts shape :math:`(C_b^c, H, W)` or
        :math:`(B, C_b^c, H, W)`; a missing batch dim is added automatically.
        """
        ### Input validation (skipped under torch.compile per MOD-005).
        if not torch.compiler.is_compiling():
            n_lat, n_lon = self.atmo_resolution[1], self.atmo_resolution[2]
            n_levels = self.atmo_resolution[0]
            if surface_in.ndim != 4:
                raise ValueError(
                    f"Expected 4D surface_in (B, C_s, H, W), got shape {tuple(surface_in.shape)}"
                )
            # Same fix as PanguPlasimLegacy: the patchembed2d packs
            # surface + land + ocean together, so the validation must
            # accept ``num_surface_vars + num_land_vars + num_ocean_vars``
            # channels rather than just ``num_surface_vars``.
            expected_surface_channels = (
                self.num_surface_vars + self.num_land_vars + self.num_ocean_vars
            )
            if surface_in.shape[1] != expected_surface_channels:
                raise ValueError(
                    f"Expected surface_in with {expected_surface_channels} "
                    f"channels (= num_surface_vars={self.num_surface_vars} + "
                    f"num_land_vars={self.num_land_vars} + "
                    f"num_ocean_vars={self.num_ocean_vars}), got shape "
                    f"{tuple(surface_in.shape)}"
                )
            B = surface_in.shape[0]
            if surface_in.shape[-2:] != (n_lat, n_lon):
                raise ValueError(
                    f"Expected surface_in spatial dims ({n_lat}, {n_lon}), "
                    f"got shape {tuple(surface_in.shape)}"
                )
            if constant_boundary.ndim not in (3, 4):
                raise ValueError(
                    "Expected constant_boundary of shape (C_b^c, H, W) or "
                    f"(B, C_b^c, H, W), got shape {tuple(constant_boundary.shape)}"
                )
            if constant_boundary.shape[-2:] != (n_lat, n_lon):
                raise ValueError(
                    f"Expected constant_boundary spatial dims ({n_lat}, {n_lon}), "
                    f"got shape {tuple(constant_boundary.shape)}"
                )
            if varying_boundary.shape != (B, self.num_varying_boundary_vars, n_lat, n_lon):
                raise ValueError(
                    f"Expected varying_boundary of shape "
                    f"({B}, {self.num_varying_boundary_vars}, {n_lat}, {n_lon}), "
                    f"got shape {tuple(varying_boundary.shape)}"
                )
            if upper_air_in.shape != (B, self.num_atmo_vars, n_levels, n_lat, n_lon):
                raise ValueError(
                    f"Expected upper_air_in of shape "
                    f"({B}, {self.num_atmo_vars}, {n_levels}, {n_lat}, {n_lon}), "
                    f"got shape {tuple(upper_air_in.shape)}"
                )
            if train:
                if target_surface is None or target_upper_air is None:
                    raise ValueError(
                        "train=True requires both target_surface and target_upper_air"
                    )
                if target_surface.shape != surface_in.shape:
                    raise ValueError(
                        f"Expected target_surface shape {tuple(surface_in.shape)}, "
                        f"got shape {tuple(target_surface.shape)}"
                    )
                if target_upper_air.shape != upper_air_in.shape:
                    raise ValueError(
                        f"Expected target_upper_air shape {tuple(upper_air_in.shape)}, "
                        f"got shape {tuple(target_upper_air.shape)}"
                    )

        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)

        # ----- data preparation for encoder 1 -----
        if self.upper_air_boundary:
            upper_air_varying_boundary = varying_boundary[
                :, self.idx_upper_air_var_bound, :, :
            ].unsqueeze(1)
            surface_varying_boundary = varying_boundary[:, self.idx_surface_var_bound, :, :]
            surface = torch.cat(
                [
                    surface_in,
                    constant_boundary[: surface_in.shape[0]],
                    surface_varying_boundary,
                ],
                dim=1,
            )
            surface = self.patchembed2d(surface)
            upper_air_varying_boundary = self.patchembed2d_upper_air_boundary(
                upper_air_varying_boundary
            )
            upper_air = self.patchembed3d(upper_air_in)
            x = torch.cat(
                [upper_air_varying_boundary.unsqueeze(2), upper_air, surface.unsqueeze(2)],
                dim=2,
            )
        else:
            surface = torch.concat(
                [surface_in, constant_boundary[: surface_in.shape[0]], varying_boundary],
                dim=1,
            )
            surface = self.patchembed2d(surface)
            upper_air = self.patchembed3d(upper_air_in)
            x = torch.concat([upper_air, surface.unsqueeze(2)], dim=2)

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        if train:
            # ----- data preparation for encoder 2 -----
            if self.upper_air_boundary:
                surface_target = torch.cat(
                    [target_surface, constant_boundary, surface_varying_boundary], dim=1
                )
                surface_target = self.patchembed2d(surface)
                target_upper_air = self.patchembed3d(target_upper_air)
                x_target = torch.cat(
                    [
                        upper_air_varying_boundary.unsqueeze(2),
                        target_upper_air,
                        surface_target.unsqueeze(2),
                    ],
                    dim=2,
                )
            else:
                surface_target = torch.concat(
                    [target_surface, constant_boundary, varying_boundary], dim=1
                )
                surface_target = self.patchembed2d(surface_target)
                target_upper_air = self.patchembed3d(target_upper_air)
                x_target = torch.concat(
                    [target_upper_air, surface_target.unsqueeze(2)], dim=2
                )
            x_target = x_target.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)
        if train:
            x_e2 = self.layer1_e2(x_target)
            x_e2 = self.downsample_e2(x_e2)

        skip = x
        x = self.downsample(x)
        latent = x.detach().clone() if return_latent else None
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(
            B,
            self.downscale_resolution[0],
            self.downscale_resolution[1],
            self.downscale_resolution[2],
            -1,
        ).permute(0, 4, 1, 2, 3)

        x_vae = x
        # ----- VAE encoder 1 -----
        mu = self.layer_mu(x_vae)
        sigma = self.layer_sigma(x_vae)
        norm = self.reparameterize(mu, sigma)
        x_purb = self.layer_purturbation(norm)

        if train:
            # ----- VAE encoder 2 -----
            x_e2 = checkpoint(self.layer2_e2, x_e2, use_reentrant=self.use_reentrant)
            x_e2 = checkpoint(self.layer3_e3, x_e2, use_reentrant=self.use_reentrant)
            x_e2_vae = x_e2.reshape(
                B,
                self.downscale_resolution[0],
                self.downscale_resolution[1],
                self.downscale_resolution[2],
                -1,
            ).permute(0, 4, 1, 2, 3)
            mu_e2 = self.layer_mu_e2(x_e2_vae)
            sigma_e2 = self.layer_sigma_e2(x_e2_vae)
            norm_e2 = self.reparameterize(mu_e2, sigma_e2)  # noqa: F841

        # ----- decoder -----
        x = x_purb + x
        x = x.permute(0, 2, 3, 4, 1).reshape(
            B, -1, self.embed_dim * self.updown_scale_factor
        )
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)

        if self.predict_delta:
            output_surface_delta = output[:, :, -1, :, :]
            if self.upper_air_boundary:
                output_upper_air_delta = output[:, :, 1:-1, :, :]
            else:
                output_upper_air_delta = output[:, :, :-1, :, :]
            if self.checkpointing > 0 and train:
                output_2D = checkpoint(
                    self.patchrecovery2d, output_surface_delta, use_reentrant=self.use_reentrant
                )
            else:
                output_2D = self.patchrecovery2d(output_surface_delta)
            output_surface = output_2D[:, self.surface_prognostic_idxs]
            if self.has_land and self.mask_output:
                output_surface[:, self.num_surface_vars : self.num_surface_vars + self.num_land_vars] = self.land_mask(
                    output_surface[:, self.num_surface_vars : self.num_surface_vars + self.num_land_vars]
                ).to(output_surface.dtype)
            if self.has_ocean and self.mask_output:
                output_surface[:, self.num_surface_vars + self.num_land_vars :] = self.land_mask(
                    output_surface[:, self.num_surface_vars + self.num_land_vars :]
                ).to(output_surface.dtype)
            if self.checkpointing > 0 and train:
                output_upper_air = checkpoint(
                    self.patchrecovery3d, output_upper_air_delta, use_reentrant=self.use_reentrant
                )
            else:
                output_upper_air = self.patchrecovery3d(output_upper_air_delta)
        else:
            output_surface = output[:, :, -1, :, :]
            if self.upper_air_boundary:
                output_upper_air = output[:, :, 1:-1, :, :]
            else:
                output_upper_air = output[:, :, :-1, :, :]
            if self.checkpointing > 0 and train:
                output_2D = checkpoint(
                    self.patchrecovery2d, output_surface, use_reentrant=self.use_reentrant
                )
            else:
                output_2D = self.patchrecovery2d(output_surface)
            output_surface = output_2D[:, self.surface_prognostic_idxs]
            if self.has_land and self.mask_output:
                output_surface[:, self.num_surface_vars : self.num_surface_vars + self.num_land_vars] = self.land_mask(
                    output_surface[:, self.num_surface_vars : self.num_surface_vars + self.num_land_vars]
                ).to(output_surface.dtype)
            if self.has_ocean and self.mask_output:
                output_surface[:, self.num_surface_vars + self.num_land_vars :] = self.land_mask(
                    output_surface[:, self.num_surface_vars + self.num_land_vars :]
                ).to(output_surface.dtype)
            if self.checkpointing > 0 and train:
                output_upper_air = checkpoint(
                    self.patchrecovery3d, output_upper_air, use_reentrant=self.use_reentrant
                )
            else:
                output_upper_air = self.patchrecovery3d(output_upper_air)

        if self.num_diagnostic_vars > 0:
            output_diagnostic = output_2D[
                :, self.num_surface_vars : self.num_surface_vars + self.num_diagnostic_vars
            ].reshape(
                output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1]
            )
            if train:
                result = (output_surface, output_upper_air, output_diagnostic, mu, sigma, mu_e2, sigma_e2)
            else:
                result = (
                    output_surface,
                    output_upper_air,
                    output_diagnostic,
                    mu,
                    sigma,
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                )
        else:
            if train:
                result = (output_surface, output_upper_air, mu, sigma, mu_e2, sigma_e2)
            else:
                result = (
                    output_surface,
                    output_upper_air,
                    mu,
                    sigma,
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                )

        if return_latent:
            return result + (latent,)
        return result
