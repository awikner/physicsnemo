# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SFNO emulator for PlaSim with a prognostic/diagnostic channel split.

The piece neither PhysicsNeMo nor stock Makani provides is the split itself:
the network predicts 52 prognostic channels *and* a set of diagnostic
distribution parameters, but only the 52 prognostic channels are fed back into
the autoregressive loop. Stock Makani has no ``n_diagnostic_channels`` concept
(its "unpredicted" variables are forcings, i.e. inputs that are never
predicted), so the slicing is done here.

SFNO itself comes from Makani, which registers into PhysicsNeMo's model
registry. Its spherical harmonic transforms are used on a ``legendre-gauss``
grid, which is exactly the grid PlaSim's T42 output lives on - the source files
are literally named ``*_gaussian.nc`` - so no regridding is needed and the poles
and dateline carry no seam.
"""

import torch
import torch.nn as nn

from channels import GRID_SHAPE, N_DIAG_PARAMS, N_DIAGNOSTIC, N_FORCING, N_STATE


def _build_sfno(inp_chans: int, out_chans: int, **kw) -> nn.Module:
    from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet

    cfg = dict(
        inp_shape=GRID_SHAPE,
        out_shape=GRID_SHAPE,
        inp_chans=inp_chans,
        out_chans=out_chans,
        # PlaSim T42 output is on a Gaussian grid; match it exactly.
        model_grid_type="legendre-gauss",
        sht_grid_type="legendre-gauss",
        spectral_transform="sht",
        filter_type="linear",
        operator_type="dhconv",
        scale_factor=1,
        embed_dim=256,
        num_layers=12,
        use_mlp=True,
        mlp_ratio=2.0,
        normalization_layer="instance_norm",
        activation_function="gelu",
        pos_embed="none",
        big_skip=True,
    )
    cfg.update(kw)
    return SphericalFourierNeuralOperatorNet(**cfg)


class PlasimEmulator(nn.Module):
    """One 6-hour step: (state, forcing, noise) -> (next state, diagnostic params).

    Parameters
    ----------
    n_noise : int
        Number of white-noise input channels, resampled at every autoregressive
        step. This is what makes a rollout a genuine sample rather than a
        deterministic function of the initial condition, so AI-RES can draw
        distinct trajectories from the same restart. Set 0 for a deterministic
        model.
    probabilistic : bool
        If True the diagnostic head emits hurdle-Gamma parameters (3 per
        diagnostic); if False it emits a point value per diagnostic, which is the
        like-for-like baseline the probabilistic model must beat.
    """

    def __init__(
        self,
        n_noise: int = 2,
        probabilistic: bool = True,
        n_state: int = N_STATE,
        n_forcing: int = N_FORCING,
        **sfno_kwargs,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_forcing = n_forcing
        self.n_noise = n_noise
        self.probabilistic = probabilistic
        self.n_diag_out = N_DIAG_PARAMS if probabilistic else N_DIAGNOSTIC

        self.inp_chans = n_state + n_forcing + n_noise
        self.out_chans = n_state + self.n_diag_out
        self.net = _build_sfno(self.inp_chans, self.out_chans, **sfno_kwargs)

    def draw_noise(self, batch: int, device, dtype, generator=None) -> torch.Tensor:
        if self.n_noise == 0:
            return torch.empty(batch, 0, *GRID_SHAPE, device=device, dtype=dtype)
        return torch.randn(
            batch, self.n_noise, *GRID_SHAPE,
            device=device, dtype=dtype, generator=generator,
        )

    def forward(self, state, forcing, noise=None, generator=None):
        """
        state   : [B, 52, H, W] z-scored
        forcing : [B,  8, H, W] z-scored
        noise   : [B, n_noise, H, W] or None (drawn internally)

        Returns (state_next [B,52,H,W], diag_raw [B, n_diag_out, H, W]).
        """
        if noise is None:
            noise = self.draw_noise(
                state.shape[0], state.device, state.dtype, generator
            )
        x = torch.cat([state, forcing, noise], dim=1)
        y = self.net(x)
        # THE SPLIT: only the first n_state channels ever feed back.
        return y[:, : self.n_state], y[:, self.n_state :]


class DiagnosticHead(nn.Module):
    """Model B: a separate, smaller SFNO mapping state+forcing -> diagnostics.

    Trained on top of a frozen prognostic backbone. Kept as an SFNO (rather than
    the AFNO the PhysicsNeMo diagnostic example uses) so the head shares the
    backbone's spherical geometry.
    """

    def __init__(
        self,
        probabilistic: bool = True,
        n_state: int = N_STATE,
        n_forcing: int = N_FORCING,
        embed_dim: int = 128,
        num_layers: int = 4,
        **sfno_kwargs,
    ):
        super().__init__()
        self.n_diag_out = N_DIAG_PARAMS if probabilistic else N_DIAGNOSTIC
        self.net = _build_sfno(
            n_state + n_forcing,
            self.n_diag_out,
            embed_dim=embed_dim,
            num_layers=num_layers,
            **sfno_kwargs,
        )

    def forward(self, state, forcing):
        return self.net(torch.cat([state, forcing], dim=1))


def rollout(model, state0, forcing_seq, generator=None):
    """Autoregressive rollout with fresh noise at every step.

    forcing_seq : [B, T, 8, H, W] forcing for each target step.
    Returns states [B, T, 52, H, W] and diag params [B, T, n_diag_out, H, W].

    `model` may be a bare PlasimEmulator or a DDP-wrapped one. DDP does not proxy
    attribute access, so `draw_noise` is taken from the underlying module while
    the forward call stays on the wrapper (which is what synchronises gradients).
    """
    inner = model.module if hasattr(model, "module") else model
    B, T = forcing_seq.shape[0], forcing_seq.shape[1]
    state = state0
    states, diags = [], []
    for t in range(T):
        noise = inner.draw_noise(B, state.device, state.dtype, generator)
        state, diag = model(state, forcing_seq[:, t], noise)
        states.append(state)
        diags.append(diag)
    return torch.stack(states, dim=1), torch.stack(diags, dim=1)
