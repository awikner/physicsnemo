# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""PLASIM-routed Spherical Fourier Neural Operator wrapper.

Faithful port of PanguWeather v2.0's
``networks/modulus_sfno/sfnonet.py::SphericalFourierNeuralOperatorNet_v2``.
The base SFNO is dataset-agnostic (takes a flat ``(in_chans, out_chans)``);
this wrapper bridges that to the PLASIM-style variable-routing convention
shared with :class:`PanguPlasim` / :class:`PanguPlasimLegacy`:

* Constructor takes explicit variable-group lists (surface, constant
  boundary, varying boundary, upper-air pressure, upper-air sigma,
  diagnostic).
* Forward takes the same four input tensors PanguPlasim does
  (``surface_in``, ``constant_boundary``, ``varying_boundary``,
  ``upper_air_in``) and returns the same shape tuple so the trainer can
  swap models without touching ``train_step``.

The base SFNO and its building blocks live in
:mod:`physicsnemo.experimental.models.modulus_sfno`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

from ..modulus_sfno import SphericalFourierNeuralOperatorNet


@dataclass
class MetaData(ModelMetaData):
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    amp_gpu: bool = True
    bf16: bool = True
    onnx: bool = False


class _GridParams:
    """Small shim — the base SFNO reads ``params.data_grid``."""

    def __init__(self, grid: str) -> None:
        self.data_grid = grid


class SfnoPlasim(Module):
    r"""PLASIM-style Spherical Fourier Neural Operator.

    Wraps the vendored
    :class:`physicsnemo.experimental.models.modulus_sfno.SphericalFourierNeuralOperatorNet`
    with the surface / constant-boundary / varying-boundary / upper-air
    routing convention used by :class:`PanguPlasim` and
    :class:`PanguPlasimLegacy`. Constructor + forward stay drop-in compatible
    so the same training recipe handles all three model families.

    Faithful to PanguWeather v2.0 ``networks/modulus_sfno/sfnonet.py``
    (``SphericalFourierNeuralOperatorNet_v2``): the base SFNO has no VAE,
    no dual-encoder, no diagnostic head; ``surface``, optional
    ``diagnostic``, and ``upper_air`` are split off the flat
    ``out_chans`` output by channel position.

    Parameters
    ----------
    surface_variables : list of str
        Surface prognostic channel names. Channel order is preserved in the
        output ``out_surface``.
    upper_air_variables : list of str
        Upper-air prognostic channel names (sigma + pressure concatenated
        in the convention used by :class:`PlasimClimateDataset`). All vars
        share the same ``levels`` count.
    constant_boundary_variables : list of str
        Static boundary channel names (e.g. ``lsm``).
    varying_boundary_variables : list of str
        Time-varying boundary channel names (e.g. ``sst``, ``rsdt``,
        ``sic``).
    diagnostic_variables : list of str, optional, default=()
        Diagnostic channel names. When non-empty the wrapper splits a
        ``out_diagnostic`` tensor off the end of the SFNO output.
    levels : list of float
        Vertical level coordinates (sigma OR pressure). The wrapper only
        reads ``len(levels)``; values are kept for record-keeping.
    horizontal_resolution : list of int
        ``[lat, lon]`` grid shape.

    spectral_transform : {"sht", "fft"}, optional, default="sht"
        Underlying spectral transform (spherical harmonic vs Cartesian FFT).
        PanguWeather v2.0 SFNO_PLASIM uses ``"sht"``.
    filter_type : {"linear", "non-linear"}, optional, default="linear"
    operator_type : {"diagonal", "dhconv"}, optional, default="dhconv"
    scale_factor : int, optional, default=1
        Downsample-then-upsample factor between the embedding layer and the
        spectral blocks. SFNO_PLASIM uses 1.
    embed_dim : int, optional, default=256
    num_layers : int, optional, default=12
    use_mlp : bool, optional, default=True
    mlp_ratio : float, optional, default=2.0
    activation_function : str, optional, default="gelu"
    encoder_layers : int, optional, default=1
    pos_embed : bool, optional, default=False
    drop_rate : float, optional, default=0.0
    drop_path_rate : float, optional, default=0.0
    num_blocks : int, optional, default=8
    sparsity_threshold : float, optional, default=0.0
    normalization_layer : {"instance_norm", "layer_norm", "none"}, optional, default="instance_norm"
    hard_thresholding_fraction : float, optional, default=1.0
    use_complex_kernels : bool, optional, default=True
    big_skip : bool, optional, default=True
    rank : float, optional, default=1.0
    factorization : str or None, optional, default=None
    separable : bool, optional, default=False
    complex_network : bool, optional, default=True
    complex_activation : str, optional, default="real"
    spectral_layers : int, optional, default=3
    checkpointing : int, optional, default=0
    data_grid : str, optional, default="equiangular"
        Spherical grid type for the SHT. PanguWeather uses ``"equiangular"``.

    Forward
    -------
    surface_in : torch.Tensor
        Shape ``(B, n_surface, lat, lon)``.
    constant_boundary : torch.Tensor
        Shape ``(n_const, lat, lon)`` or ``(B, n_const, lat, lon)``.
    varying_boundary : torch.Tensor
        Shape ``(B, n_varying, lat, lon)``.
    upper_air_in : torch.Tensor
        Shape ``(B, n_upper, n_levels, lat, lon)``.
    target_surface, target_upper_air : torch.Tensor, optional
        Unused — accepted for signature compatibility with
        :class:`PanguPlasim`. SFNO has no encoder-2 branch.
    train : bool, optional, default=False
        Accepted for signature compatibility; not consulted by the
        underlying SFNO.
    return_latent : bool, optional, default=False
        Append the bottleneck latent (post-encoder, pre-decoder) to the
        return tuple. Useful for validation / introspection.

    Outputs
    -------
    tuple of torch.Tensor
        ``(out_surface, out_upper_air[, out_diagnostic], 0, 0, 0, 0)``. The
        trailing four zero placeholders match PanguPlasim's eval-mode
        return so downstream ``train_step`` treats all model families
        uniformly (the placeholders gate the VAE-KL branch off — see
        ``examples/weather/ai_rossby/train_loop.py``).
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
        diagnostic_variables: list = (),
        spectral_transform: str = "sht",
        filter_type: str = "linear",
        operator_type: str = "dhconv",
        scale_factor: int = 1,
        embed_dim: int = 256,
        num_layers: int = 12,
        use_mlp: bool = True,
        mlp_ratio: float = 2.0,
        activation_function: str = "gelu",
        encoder_layers: int = 1,
        pos_embed: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.0,
        normalization_layer: str = "instance_norm",
        hard_thresholding_fraction: float = 1.0,
        use_complex_kernels: bool = True,
        big_skip: bool = True,
        rank: float = 1.0,
        factorization: Optional[str] = None,
        separable: bool = False,
        complex_network: bool = True,
        complex_activation: str = "real",
        spectral_layers: int = 3,
        checkpointing: int = 0,
        data_grid: str = "equiangular",
    ) -> None:
        super().__init__(meta=MetaData())

        # Variable groups + level geometry.
        self.surface_variables = list(surface_variables)
        self.upper_air_variables = list(upper_air_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.levels = list(levels)
        self.horizontal_resolution = list(horizontal_resolution)
        self.has_diagnostic = len(self.diagnostic_variables) > 0

        # in_chans = surface + constant_boundary + varying_boundary
        #            + upper_air × levels
        n_surface = len(self.surface_variables)
        n_const = len(self.constant_boundary_variables)
        n_varying = len(self.varying_boundary_variables)
        n_diag = len(self.diagnostic_variables)
        n_upper = len(self.upper_air_variables)
        n_levels = len(self.levels)

        self._n_surface = n_surface
        self._n_upper = n_upper
        self._n_levels = n_levels
        self._n_diag = n_diag

        in_chans = n_surface + n_const + n_varying + n_upper * n_levels
        out_chans = n_surface + n_diag + n_upper * n_levels

        # Build the underlying SFNO. The base reads `params.data_grid` for
        # the SHT grid setup; everything else is via kwargs.
        self.sfno = SphericalFourierNeuralOperatorNet(
            params=_GridParams(data_grid),
            spectral_transform=spectral_transform,
            filter_type=filter_type,
            operator_type=operator_type,
            img_shape=tuple(self.horizontal_resolution),
            scale_factor=scale_factor,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dim=embed_dim,
            num_layers=num_layers,
            use_mlp=use_mlp,
            mlp_ratio=mlp_ratio,
            activation_function=activation_function,
            encoder_layers=encoder_layers,
            pos_embed=pos_embed,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            normalization_layer=normalization_layer,
            hard_thresholding_fraction=hard_thresholding_fraction,
            use_complex_kernels=use_complex_kernels,
            big_skip=big_skip,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            checkpointing=checkpointing,
        )

    @property
    def upper_air_variable_names(self) -> list:
        """Convenience — same shape as PanguPlasim's property of the same name."""
        return list(self.upper_air_variables)

    def forward(
        self,
        surface_in: torch.Tensor,
        constant_boundary: torch.Tensor,
        varying_boundary: torch.Tensor,
        upper_air_in: torch.Tensor,
        target_surface: Optional[torch.Tensor] = None,
        target_upper_air: Optional[torch.Tensor] = None,
        train: bool = False,
        return_latent: bool = False,
    ):
        """See class docstring for the contract.

        The two ``target_*`` tensors and ``train`` are accepted for signature
        compatibility with :class:`PanguPlasim` and ignored — SFNO has no
        encoder-2 branch.
        """
        del target_surface, target_upper_air, train  # accepted but unused

        if constant_boundary.ndim == 3:
            constant_boundary = constant_boundary.unsqueeze(0).expand(
                surface_in.shape[0], -1, -1, -1
            )
        elif constant_boundary.shape[0] != surface_in.shape[0]:
            constant_boundary = constant_boundary[: surface_in.shape[0]]

        b, _, n_lat, n_lon = surface_in.shape
        # Flatten upper_air levels into channels: (B, C_u, L, H, W) → (B, C_u*L, H, W).
        upper_air_flat = upper_air_in.view(b, self._n_upper * self._n_levels, n_lat, n_lon)

        x = torch.cat((surface_in, constant_boundary, varying_boundary, upper_air_flat), dim=1)

        if return_latent:
            x, latent = self.sfno(x, return_latent=True)
        else:
            x = self.sfno(x)
            latent = None

        # Split outputs back into the PLASIM shape:
        # [:, :n_surface] → surface
        # [:, n_surface : n_surface + n_upper*n_levels] → upper_air
        # [:, n_surface + n_upper*n_levels :] → diagnostic (when present)
        out_surface = x[:, : self._n_surface]
        out_upper_air_flat = x[
            :, self._n_surface : self._n_surface + self._n_upper * self._n_levels
        ]
        out_upper_air = out_upper_air_flat.view(b, self._n_upper, self._n_levels, n_lat, n_lon)

        # Placeholder zero tensors for (mu, logvar, mu_e2, logvar_e2) so the
        # PLASIM trainer's train_step takes the legacy path (no VAE KL).
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.has_diagnostic:
            out_diag = x[:, self._n_surface + self._n_upper * self._n_levels :]
            result = (out_surface, out_upper_air, out_diag, zero, zero, zero, zero)
        else:
            result = (out_surface, out_upper_air, zero, zero, zero, zero)

        if return_latent:
            return result + (latent,)
        return result
