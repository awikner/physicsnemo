# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss functions for Pangu_Plasim training.

Faithful to PanguWeather v2.0's loss formulation:

* Pixel-wise MSE or MAE residuals on the model's prognostic outputs
  (``surface_in``, ``upper_air_in``, optional ``diagnostic``).
* Lat-weighted reduction (cos(lat) over the latitude dimension).
* Per-variable weights (uniform if not provided).
* Diagnostic loss is a separately-weighted term added at the end.

The PanguPlasimLegacy (no-VAE) variant uses just :class:`PanguPlasimLoss`;
the VAE-enabled PanguPlasim additionally pulls
:func:`vae_kl_loss` over its dual-encoder ``(mu, logvar)`` outputs.
Recipes sum the two as ``total = task_loss + kl_weight * vae_kl_loss``.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch


def cos_lat_weights(num_lat: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r"""Standard cos(lat) weight for a Gaussian-ish lat grid.

    Produces a (num_lat,) tensor where each entry is :math:`\cos(\phi_i)`
    with :math:`\phi_i` the equally-spaced lat in radians from +pi/2 to -pi/2.
    Normalized so the mean weight is 1.
    """
    phi = torch.linspace(
        torch.pi / 2 - torch.pi / (2 * num_lat),
        -torch.pi / 2 + torch.pi / (2 * num_lat),
        num_lat,
        device=device,
        dtype=dtype,
    )
    w = torch.cos(phi)
    return w / w.mean()


def _per_var_weight_tensor(
    names: list[str],
    weights: Optional[Mapping[str, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map a per-variable weight dict to a tensor in `names` order. Missing = 1."""
    if not weights:
        return torch.ones(len(names), device=device, dtype=dtype)
    return torch.tensor(
        [float(weights.get(n, 1.0)) for n in names],
        device=device,
        dtype=dtype,
    )


def lat_weighted_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    *,
    loss_type: str = "l1",
) -> torch.Tensor:
    r"""Mean over (B, …, lat, lon) of cos(lat)-weighted |pred - target|^p.

    Parameters
    ----------
    pred, target : torch.Tensor
        Same shape ``(B, C, [L,] H, W)``.
    lat_weights : torch.Tensor
        Shape ``(H,)``; broadcast across the rest.
    loss_type : {"l1", "l2"}
        ``"l1"`` = MAE, ``"l2"`` = MSE.

    Returns
    -------
    torch.Tensor
        Scalar loss reduced as the **mean** over all dims (channels included).
    """
    if loss_type == "l1":
        resid = (pred - target).abs()
    elif loss_type == "l2":
        resid = (pred - target).pow(2)
    else:
        raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")
    # H is always the second-to-last dim of `resid` (B, C, [L,] H, W); broadcast
    # lat_weights to (1, ..., 1, H, 1) so it aligns with the right axis.
    shape = [1] * resid.ndim
    shape[-2] = lat_weights.shape[0]
    return (resid * lat_weights.view(shape)).mean()


def per_var_lat_weighted_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    per_var: torch.Tensor,
    *,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Per-variable-weighted lat-weighted residual.

    ``pred``/``target`` have shape ``(B, C, [L,] H, W)``; ``per_var`` has
    shape ``(C,)`` and broadcasts over the channel axis.
    """
    if loss_type == "l1":
        resid = (pred - target).abs()
    elif loss_type == "l2":
        resid = (pred - target).pow(2)
    else:
        raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")
    # Weight broadcasting: per_var on dim 1 (C); lat_weights on the second-to-last dim (H).
    pv = per_var.view(1, -1, *([1] * (resid.ndim - 2)))
    lw_shape = [1] * resid.ndim
    lw_shape[-2] = lat_weights.shape[0]
    lw = lat_weights.view(lw_shape)
    return (resid * pv * lw).mean()


def vae_kl_loss(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: Optional[torch.Tensor] = None,
    logvar_p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""KL divergence between two diagonal-covariance Gaussians.

    Faithful to PanguWeather v2.0's ``Kl_divergence_gaussians`` in
    ``utils/losses.py``. PanguPlasim's two encoder heads emit
    ``(mu_q, logvar_q)`` and ``(mu_p, logvar_p)`` — passing both pairs computes
    :math:`KL(q \,\|\, p)` between the encoder posteriors. With ``mu_p`` /
    ``logvar_p`` left ``None``, the prior is the standard normal
    :math:`p \sim \mathcal{N}(0, I)`.

    Parameters
    ----------
    mu_q, logvar_q : torch.Tensor
        Mean and log-variance of the posterior :math:`q`, same shape.
    mu_p, logvar_p : torch.Tensor or None, optional
        Mean and log-variance of the prior :math:`p`. ``None`` → standard
        normal.

    Returns
    -------
    torch.Tensor
        Scalar tensor — the mean over all elements of

        .. math::

            \tfrac{1}{2}\bigl[\log\sigma_p^2 - \log\sigma_q^2
            + (\sigma_q^2 + (\mu_q - \mu_p)^2) / \sigma_p^2 - 1\bigr]

    Notes
    -----
    PanguPlasim's model stores ``logvar`` under the attribute name ``sigma``
    — the model code outputs :math:`\log(\sigma^2)`, not :math:`\sigma`.
    The trainer passes those values straight through here.
    """
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    )
    return kl.mean()


class PanguPlasimLoss(torch.nn.Module):
    r"""Task loss for Pangu_Plasim training (deterministic variant).

    Computes a weighted sum of the surface, upper-air, and (optional)
    diagnostic residuals. Each residual is per-variable-weighted +
    lat-weighted. The terms are summed with configurable surface / upper-air /
    diagnostic weights.

    Parameters
    ----------
    surface_variables : list of str
        Channel-order names of the surface prognostic vars.
    upper_air_variable_names : list of str
        Channel-order names of the upper-air prognostic vars (concat order:
        sigma vars first, then pressure vars).
    diagnostic_variables : list of str
        Channel-order names of the diagnostic vars; empty if none.
    num_lat : int
        Latitude resolution (for the cos-lat weight tensor).
    loss_type : {"l1", "l2"}, optional, default="l1"
        Residual norm to use; "l1" = MAE (matches PANGU_PLASIM_H5_DERECHO_0514).
    surface_weight, upper_air_weight, diagnostic_weight : float, optional
        Scalar weights summed across terms. Defaults: 1, 1, 1.
    surface_var_weights, upper_air_var_weights, diagnostic_var_weights : dict, optional
        Per-variable weight dicts; missing entries default to 1.

    Forward
    -------
    out_surface, out_upper_air : torch.Tensor
        Model predictions; shapes as for :class:`PanguPlasim.forward` outputs.
    target_surface, target_upper_air : torch.Tensor
        Target tensors (delta or full-state, set by the dataset/normalizer).
    out_diagnostic, target_diagnostic : torch.Tensor, optional
        Diagnostic prediction + target. ``None`` if model has no diagnostic head.

    Outputs
    -------
    dict
        ``{"loss": scalar, "surface": scalar, "upper_air": scalar,
        "diagnostic": scalar}`` — the loss is the weighted sum.
    """

    def __init__(
        self,
        *,
        surface_variables: list[str],
        upper_air_variable_names: list[str],
        diagnostic_variables: list[str],
        num_lat: int,
        loss_type: str = "l1",
        surface_weight: float = 1.0,
        upper_air_weight: float = 1.0,
        diagnostic_weight: float = 1.0,
        surface_var_weights: Optional[Mapping[str, float]] = None,
        upper_air_var_weights: Optional[Mapping[str, float]] = None,
        diagnostic_var_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")
        self.loss_type = loss_type
        self.surface_weight = float(surface_weight)
        self.upper_air_weight = float(upper_air_weight)
        self.diagnostic_weight = float(diagnostic_weight)
        self._surface_names = list(surface_variables)
        self._upper_names = list(upper_air_variable_names)
        self._diag_names = list(diagnostic_variables)
        self._surface_var_weights = dict(surface_var_weights or {})
        self._upper_var_weights = dict(upper_air_var_weights or {})
        self._diag_var_weights = dict(diagnostic_var_weights or {})
        self._num_lat = int(num_lat)
        self._cached: dict[tuple, torch.Tensor] = {}

    def _weights_for(self, kind: str, device, dtype) -> torch.Tensor:
        key = (kind, device, dtype)
        if key in self._cached:
            return self._cached[key]
        if kind == "surface":
            t = _per_var_weight_tensor(self._surface_names, self._surface_var_weights, device, dtype)
        elif kind == "upper_air":
            t = _per_var_weight_tensor(self._upper_names, self._upper_var_weights, device, dtype)
        elif kind == "diagnostic":
            t = _per_var_weight_tensor(self._diag_names, self._diag_var_weights, device, dtype)
        elif kind == "lat":
            t = cos_lat_weights(self._num_lat, device, dtype)
        else:
            raise KeyError(kind)
        self._cached[key] = t
        return t

    def forward(
        self,
        out_surface: torch.Tensor,
        out_upper_air: torch.Tensor,
        target_surface: torch.Tensor,
        target_upper_air: torch.Tensor,
        out_diagnostic: Optional[torch.Tensor] = None,
        target_diagnostic: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        device, dtype = out_surface.device, out_surface.dtype
        lat = self._weights_for("lat", device, dtype)

        loss_surface = per_var_lat_weighted_residual(
            out_surface,
            target_surface,
            lat,
            self._weights_for("surface", device, dtype),
            loss_type=self.loss_type,
        )
        loss_upper_air = per_var_lat_weighted_residual(
            out_upper_air,
            target_upper_air,
            lat,
            self._weights_for("upper_air", device, dtype),
            loss_type=self.loss_type,
        )
        loss_diag = torch.zeros((), device=device, dtype=dtype)
        if out_diagnostic is not None and target_diagnostic is not None and self._diag_names:
            loss_diag = per_var_lat_weighted_residual(
                out_diagnostic,
                target_diagnostic,
                lat,
                self._weights_for("diagnostic", device, dtype),
                loss_type=self.loss_type,
            )

        total = (
            self.surface_weight * loss_surface
            + self.upper_air_weight * loss_upper_air
            + self.diagnostic_weight * loss_diag
        )
        return {
            "loss": total,
            "surface": loss_surface.detach(),
            "upper_air": loss_upper_air.detach(),
            "diagnostic": loss_diag.detach(),
        }
