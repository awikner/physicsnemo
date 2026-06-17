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

"""Faithful, weight-compatible port of PanguWeather v2.0 ``pangu_legacy.py``
(the predecessor no-VAE Pangu_Plasim architecture) into a ``physicsnemo.Module``.

This is the no-VAE sibling of :class:`PanguPlasim` (see ``pangu_plasim.py``):

* Same Earth-Specific dual-stream encoder/decoder (surface + upper-air) with the
  same constant + varying boundary conditioning and TOA-solar routing.
* **No VAE dual-encoder** — there is no second encoder branch, no
  ``layer_mu``/``layer_sigma``/``layer_purturbation``/``layer_perturbation2``
  submodules, and no ``reparameterize`` step. The forward returns the same
  six- (or seven- with diagnostics) tuple shape as :class:`PanguPlasim` in eval
  mode, but with **all four latent slots zero-filled** — matching the original
  ``pangu_legacy.py`` source — so downstream code can be written once against
  the common return shape.

The fidelity contract carries over from :class:`PanguPlasim`: every ``nn.Module``
submodule name, parameter/buffer name, and tensor shape is preserved so that
checkpoints trained with the original PanguWeather repo load via the Phase-5
translation script. Constructor arguments are JSON-serializable (a requirement
of :class:`physicsnemo.Module`); internal computation and submodule structure
mirror the source.

Adapted from the Pangu-Weather architecture
(https://github.com/198808xc/Pangu-Weather).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

from . import layers as _layers
from .layers import DownSample, EarthSpecificLayer, Mask, UpSample
from ._pangu_utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)


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


class PanguPlasimLegacy(Module):
    r"""Pangu_Plasim weather emulator — predecessor no-VAE architecture.

    Faithful port of PanguWeather v2.0 ``pangu_legacy.py``. Same constructor
    contract as :class:`PanguPlasim` (so configs are interchangeable), but with
    no VAE dual-encoder submodules and a simplified forward (no
    ``layer1_e2``/``layer2_e2``/``layer3_e3``/``downsample_e2``, no ``layer_mu``,
    ``layer_sigma``, ``layer_purturbation``, ``layer_perturbation2``, no
    ``reparameterize``).

    Constructor arguments and forward signature mirror :class:`PanguPlasim`.

    Returns
    -------
    tuple
        ``(out_surface, out_upper_air[, out_diagnostic], 0, 0, 0, 0)`` —
        a six- (no diagnostics) or seven-element tuple, where the trailing four
        elements are zero-valued ``torch.tensor(0.)`` placeholders preserving
        positional compatibility with :class:`PanguPlasim`'s eval-mode return.
        When ``return_latent=True``, the post-downsample latent is appended.
    """

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
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())

        _layers.USE_TE = use_transformer_engine
        self.use_transformer_engine = use_transformer_engine

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

        self.window_size = list(window_size)
        self.vertical_windowing = vertical_windowing
        self.updown_scale_factor = updown_scale_factor
        self.subpixel_deconv = subpixel_deconv
        self.polar_pad = polar_pad
        self._grid_has_poles = grid_has_poles
        self.recovery_head = recovery_head
        self.diagnostic_head = diagnostic_head

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

        # --- encoder (single branch — no VAE encoder-2) --------------------
        self.layer1 = EarthSpecificLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.window_size,
            drop_path=drop_path[: depths_cumsum[0]],
            vertical_windowing=vertical_windowing,
            checkpointing=self.checkpointing,
            use_reentrant=self.use_reentrant,
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
            checkpointing=self.checkpointing,
            use_reentrant=self.use_reentrant,
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
            checkpointing=self.checkpointing,
            use_reentrant=self.use_reentrant,
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
            checkpointing=self.checkpointing,
            use_reentrant=self.use_reentrant,
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
    def set_land_ocean_masks(self, land_mask: torch.Tensor, mask_fill: dict = None):
        """Populate the land/ocean ``Mask`` buffers for from-scratch training.

        Same semantics as :meth:`PanguPlasim.set_land_ocean_masks`.
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

    def forward(
        self,
        surface_in,
        constant_boundary,
        varying_boundary,
        upper_air_in,
        target_surface=None,
        target_upper_air=None,
        train=False,
        return_latent=False,
    ):
        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)

        # ----- data preparation -----
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

        # ----- encoder + decoder (no VAE) -----
        x = self.layer1(x)
        skip = x
        x = self.downsample(x)
        latent = x.detach().clone() if return_latent else None
        x = self.layer2(x)
        x = self.layer3(x)
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

        # Match the original pangu_legacy.py source: same tuple positions as the
        # VAE port's eval-mode return, with zero-tensor placeholders in all
        # four latent slots.
        zero = torch.tensor(0.0)
        if self.num_diagnostic_vars > 0:
            output_diagnostic = output_2D[
                :, self.num_surface_vars : self.num_surface_vars + self.num_diagnostic_vars
            ].reshape(
                output_surface.shape[0], -1, output_surface.shape[-2], output_surface.shape[-1]
            )
            result = (output_surface, output_upper_air, output_diagnostic, zero, zero, zero, zero)
        else:
            result = (output_surface, output_upper_air, zero, zero, zero, zero)

        if return_latent:
            return result + (latent,)
        return result
