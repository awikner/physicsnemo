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

r"""Per-channel z-score normalization for :class:`PlasimClimateDataset` samples.

PanguWeather v2.0 ships per-variable mean/std NetCDF files (e.g.,
``data_12-132_mean_sigma.nc`` / ``..._std_sigma.nc``). The stats file carries
two level coordinates (``Z`` for pressure, ``Z_2`` for sigma) and one variable
per data array. This module loads those files, subsets ``Z`` to the model's
configured pressure levels, and assembles the (mean, std) tensors aligned with
the channel order :class:`PlasimClimateDataset` produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr


class PlasimNormalizer:
    r"""Per-channel z-score normalizer for PLASIM samples.

    Loads PanguWeather-style per-variable mean and std NetCDF files (with
    ``Z`` = pressure levels, ``Z_2`` = sigma levels), assembles them onto
    tensors aligned with the channel ordering
    :class:`PlasimClimateDataset` produces, and exposes a callable that takes
    a sample dict and returns a dict of the same shape with the prognostic and
    target tensors z-scored. Constant boundaries and diagnostic fields are
    left untouched by default (toggle with
    ``normalize_constant_boundary`` / ``normalize_diagnostic``).

    Stats-file Z coord is subset to the dataset's ``pressure_levels`` via
    nearest-match (with a hard tolerance) — accommodating the common case
    where stats are computed on more pressure levels than the model uses.

    Parameters
    ----------
    mean_path : str or pathlib.Path
        Path to the per-variable mean NetCDF file.
    std_path : str or pathlib.Path
        Path to the per-variable std NetCDF file.
    surface_variables : sequence of str
        Surface variable names in the dataset's channel order.
    varying_boundary_variables : sequence of str
        Varying-boundary variable names.
    sigma_upper_air_variables : sequence of str
        Sigma-level upper-air variable names.
    pressure_upper_air_variables : sequence of str
        Pressure-level upper-air variable names.
    sigma_levels : sequence of float
        Sigma levels from the dataset (used to subset stats ``Z_2``).
    pressure_levels : sequence of float
        Pressure levels (Pa) from the dataset (used to subset stats ``Z``).
    constant_boundary_variables : sequence of str, optional, default=()
        Channel-order list for constant boundaries (used only if
        ``normalize_constant_boundary=True``).
    diagnostic_variables : sequence of str, optional, default=()
        Channel-order list for diagnostics (used only if
        ``normalize_diagnostic=True``).
    normalize_constant_boundary : bool, optional, default=False
        Whether to z-score constant boundaries.
    normalize_diagnostic : bool, optional, default=False
        Whether to z-score diagnostics.

    Forward
    -------
    sample : dict of torch.Tensor
        A :class:`PlasimClimateDataset` sample dict.

    Outputs
    -------
    dict of torch.Tensor
        Same shape as the input dict with the configured channel groups
        z-scored.

    Notes
    -----
    Means/stds are kept as ``float32`` tensors registered as buffers; calling
    :meth:`to` on the normalizer moves them to the target device, so the
    transform fuses cleanly into a data loader pipeline that already moves
    samples to CUDA.
    """

    def __init__(
        self,
        mean_path: str | Path,
        std_path: str | Path,
        *,
        surface_variables: Sequence[str],
        varying_boundary_variables: Sequence[str],
        sigma_upper_air_variables: Sequence[str],
        pressure_upper_air_variables: Sequence[str],
        sigma_levels: Sequence[float],
        pressure_levels: Sequence[float],
        constant_boundary_variables: Sequence[str] = (),
        diagnostic_variables: Sequence[str] = (),
        normalize_constant_boundary: bool = False,
        normalize_diagnostic: bool = False,
    ) -> None:
        self._normalize_constant_boundary = normalize_constant_boundary
        self._normalize_diagnostic = normalize_diagnostic

        mean = xr.open_dataset(mean_path)
        std = xr.open_dataset(std_path)

        # Surface / varying boundary / diagnostic / constant boundary are scalars.
        self.surface_mean, self.surface_std = self._stack_scalars(
            mean, std, surface_variables
        )  # (C_s, 1, 1)
        self.varying_mean, self.varying_std = self._stack_scalars(
            mean, std, varying_boundary_variables
        )
        self.constant_mean, self.constant_std = self._stack_scalars(
            mean, std, constant_boundary_variables
        )
        self.diagnostic_mean, self.diagnostic_std = self._stack_scalars(
            mean, std, diagnostic_variables
        )

        # Upper-air vars: sigma (Z_2-dim) + pressure (Z-dim), concatenated along
        # the variable axis matching PlasimClimateDataset.upper_air_variable_names.
        sigma_mean, sigma_std = self._stack_levels(
            mean, std, sigma_upper_air_variables, "Z_2", sigma_levels
        )
        pressure_mean, pressure_std = self._stack_levels(
            mean, std, pressure_upper_air_variables, "Z", pressure_levels
        )
        # (n_sigma + n_pressure, L, 1, 1)
        self.upper_air_mean = self._cat_upper_air(sigma_mean, pressure_mean)
        self.upper_air_std = self._cat_upper_air(sigma_std, pressure_std)

        # Targets share the same stats as inputs (predict_delta=False; the
        # PredictDeltaTransform separately applies delta_std for delta mode).
        self.target_surface_mean = self.surface_mean
        self.target_surface_std = self.surface_std
        self.target_upper_air_mean = self.upper_air_mean
        self.target_upper_air_std = self.upper_air_std

    # ------------------------------------------------------------------ #
    # Stat-file loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stack_scalars(
        mean: xr.Dataset, std: xr.Dataset, names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not names:
            return None, None
        try:
            m = np.array([float(mean[v]) for v in names], dtype="float32")
            s = np.array([float(std[v]) for v in names], dtype="float32")
        except KeyError as exc:
            raise KeyError(
                f"Stats file is missing variable {exc.args[0]!r}; "
                f"available vars: {sorted(mean.data_vars)}"
            ) from exc
        # Broadcast shape (C, 1, 1) so it composes with sample tensors (C, H, W).
        return (
            torch.from_numpy(m).view(-1, 1, 1),
            torch.from_numpy(s).view(-1, 1, 1),
        )

    @staticmethod
    def _stack_levels(
        mean: xr.Dataset,
        std: xr.Dataset,
        names: Sequence[str],
        dim: str,
        target_levels: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not names:
            return None, None
        target = np.asarray(target_levels, dtype="float32")
        per_var_mean: list[np.ndarray] = []
        per_var_std: list[np.ndarray] = []
        for v in names:
            if dim not in mean[v].dims:
                raise ValueError(
                    f"Stats var {v!r} expected dim {dim}; got dims {mean[v].dims}"
                )
            stats_levels = mean.coords[dim].values.astype("float32")
            idx = _nearest_indices(stats_levels, target)
            per_var_mean.append(mean[v].values.astype("float32")[idx])
            per_var_std.append(std[v].values.astype("float32")[idx])
        m = np.stack(per_var_mean)  # (n_vars, L)
        s = np.stack(per_var_std)
        # Broadcast shape (C, L, 1, 1) so it composes with sample tensors
        # (C, L, H, W).
        return (
            torch.from_numpy(m).unsqueeze(-1).unsqueeze(-1),
            torch.from_numpy(s).unsqueeze(-1).unsqueeze(-1),
        )

    @staticmethod
    def _cat_upper_air(
        sigma_t: torch.Tensor | None, pressure_t: torch.Tensor | None
    ) -> torch.Tensor:
        parts = [t for t in (sigma_t, pressure_t) if t is not None]
        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------ #
    # Device + transform
    # ------------------------------------------------------------------ #
    def to(self, device: torch.device | str) -> "PlasimNormalizer":
        for name in (
            "surface_mean",
            "surface_std",
            "varying_mean",
            "varying_std",
            "constant_mean",
            "constant_std",
            "diagnostic_mean",
            "diagnostic_std",
            "upper_air_mean",
            "upper_air_std",
            "target_surface_mean",
            "target_surface_std",
            "target_upper_air_mean",
            "target_upper_air_std",
        ):
            v = getattr(self, name, None)
            if v is not None:
                setattr(self, name, v.to(device))
        return self

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(sample)
        if "surface_in" in out and self.surface_mean is not None:
            out["surface_in"] = (out["surface_in"] - self.surface_mean) / self.surface_std
        if "target_surface" in out and self.target_surface_mean is not None:
            out["target_surface"] = (
                out["target_surface"] - self.target_surface_mean
            ) / self.target_surface_std
        if "varying_boundary" in out and self.varying_mean is not None:
            out["varying_boundary"] = (
                out["varying_boundary"] - self.varying_mean
            ) / self.varying_std
        if "upper_air_in" in out and self.upper_air_mean is not None:
            out["upper_air_in"] = (
                out["upper_air_in"] - self.upper_air_mean
            ) / self.upper_air_std
        if "target_upper_air" in out and self.target_upper_air_mean is not None:
            out["target_upper_air"] = (
                out["target_upper_air"] - self.target_upper_air_mean
            ) / self.target_upper_air_std
        if (
            self._normalize_constant_boundary
            and "constant_boundary" in out
            and self.constant_mean is not None
        ):
            out["constant_boundary"] = (
                out["constant_boundary"] - self.constant_mean
            ) / self.constant_std
        if (
            self._normalize_diagnostic
            and "diagnostic" in out
            and self.diagnostic_mean is not None
        ):
            out["diagnostic"] = (
                out["diagnostic"] - self.diagnostic_mean
            ) / self.diagnostic_std
        return out

    @classmethod
    def from_dataset(
        cls,
        dataset,
        mean_path: str | Path,
        std_path: str | Path,
        **kwargs,
    ) -> "PlasimNormalizer":
        """Build a normalizer aligned with a :class:`PlasimClimateDataset`'s layout."""
        return cls(
            mean_path,
            std_path,
            surface_variables=dataset.layout.surface_variables,
            varying_boundary_variables=dataset.layout.varying_boundary_variables,
            sigma_upper_air_variables=dataset.layout.sigma_upper_air_variables,
            pressure_upper_air_variables=dataset.layout.pressure_upper_air_variables,
            sigma_levels=dataset.sigma_levels,
            pressure_levels=dataset.pressure_levels,
            constant_boundary_variables=dataset.layout.constant_boundary_variables,
            diagnostic_variables=dataset.layout.diagnostic_variables,
            **kwargs,
        )


def _nearest_indices(
    haystack: np.ndarray, needles: np.ndarray, atol: float = 1e-3
) -> np.ndarray:
    """Return indices into ``haystack`` matching each element of ``needles``
    by nearest value, raising if any needle has no near match (within ``atol``
    relative to the needle's magnitude, plus a small floor).
    """
    haystack = np.asarray(haystack, dtype="float64")
    needles = np.asarray(needles, dtype="float64")
    out = np.empty(needles.shape, dtype="int64")
    for i, n in enumerate(needles):
        d = np.abs(haystack - n)
        j = int(np.argmin(d))
        # Tolerance: max(atol, atol * |needle|) covers both pressure (Pa,
        # ~1e4-1e5) and sigma (~0-1) ranges with the same scalar.
        tol = max(atol, atol * abs(float(n)))
        if d[j] > tol:
            raise ValueError(
                f"No near match for level {n} in stats file (closest {haystack[j]}, "
                f"distance {d[j]}, tol {tol}). Stats levels: {haystack.tolist()}."
            )
        out[i] = j
    return out
