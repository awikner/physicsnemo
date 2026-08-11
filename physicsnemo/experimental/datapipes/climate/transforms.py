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

r"""Per-channel z-score normalization for :class:`ClimateZarrDataset` samples.

PanguWeather v2.0 ships per-variable mean/std NetCDF files (e.g.,
``data_12-132_mean_sigma.nc`` / ``..._std_sigma.nc``). The stats file carries
two level coordinates (``Z`` for pressure, ``Z_2`` for sigma) and one variable
per data array. This module loads those files, subsets ``Z`` to the model's
configured pressure levels, and assembles the (mean, std) tensors aligned with
the channel order :class:`ClimateZarrDataset` produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr


class ClimateNormalizer:
    r"""Per-channel z-score normalizer for PLASIM samples.

    Loads PanguWeather-style per-variable mean and std NetCDF files (with
    ``Z`` = pressure levels, ``Z_2`` = sigma levels), assembles them onto
    tensors aligned with the channel ordering
    :class:`ClimateZarrDataset` produces, and exposes a callable that takes
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
        A :class:`ClimateZarrDataset` sample dict.

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
        predict_delta: bool = False,
        delta_std_path: Optional[str | Path] = None,
    ) -> None:
        self._normalize_constant_boundary = normalize_constant_boundary
        self._normalize_diagnostic = normalize_diagnostic
        self._predict_delta = bool(predict_delta)
        if self._predict_delta and delta_std_path is None:
            raise ValueError(
                "predict_delta=True requires delta_std_path (the tendency-std "
                "NetCDF). Generate one via tools/data/plasim/compute_delta_stats.py."
            )

        # Force cftime decoding for consistency with the dataset's open_zarr;
        # the per-variable mean/std .nc files have no time coord so this is a
        # no-op at the data layer but keeps the open kwargs uniform.
        _cftime_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        mean = xr.open_dataset(mean_path, decode_times=_cftime_decoder)
        std = xr.open_dataset(std_path, decode_times=_cftime_decoder)

        # PanguWeather's normalization_pangu_s2s.zarr packs mean+std into one
        # file with a 'stat' coord {'mean','std'}; mean_path == std_path in
        # that case. Slice each side here so downstream lookups behave like
        # the per-var .nc files PLASIM uses.
        if "stat" in mean.coords:
            mean = mean.sel(stat="mean", drop=True)
        if "stat" in std.coords:
            std = std.sel(stat="std", drop=True)

        # Surface / varying boundary / diagnostic / constant boundary are scalars.
        self.surface_mean, self.surface_std = self._stack_scalars(
            mean, std, surface_variables
        )  # (C_s, 1, 1)
        self.varying_mean, self.varying_std = self._stack_scalars(
            mean, std, varying_boundary_variables
        )
        # Skip constant-boundary stats lookup unless we'll actually normalize
        # them — the SFNO_S2S stats zarr doesn't ship stats for land_sea_mask
        # / geopotential_at_surface (they're truly constant).
        if normalize_constant_boundary:
            self.constant_mean, self.constant_std = self._stack_scalars(
                mean, std, constant_boundary_variables
            )
        else:
            self.constant_mean, self.constant_std = None, None
        if normalize_diagnostic:
            self.diagnostic_mean, self.diagnostic_std = self._stack_scalars(
                mean, std, diagnostic_variables
            )
        else:
            self.diagnostic_mean, self.diagnostic_std = None, None

        # Upper-air vars: sigma (Z_2-dim) + pressure (Z-dim), concatenated along
        # the variable axis matching ClimateZarrDataset.upper_air_variable_names.
        sigma_mean, sigma_std = self._stack_levels(
            mean, std, sigma_upper_air_variables, "Z_2", sigma_levels
        )
        pressure_mean, pressure_std = self._stack_levels(
            mean, std, pressure_upper_air_variables, "Z", pressure_levels
        )
        # (n_sigma + n_pressure, L, 1, 1)
        self.upper_air_mean = self._cat_upper_air(sigma_mean, pressure_mean)
        self.upper_air_std = self._cat_upper_air(sigma_std, pressure_std)

        # Targets share the same stats as inputs in the predict_full-field case.
        # In predict_delta mode, the target tensors carry tendencies normalized
        # by delta_std (no mean subtraction — PanguWeather convention).
        if self._predict_delta:
            delta = xr.open_dataset(delta_std_path, decode_times=_cftime_decoder)
            # delta_std for surface vars is scalar per var; broadcast as (C, 1, 1).
            self.target_surface_delta_std = self._stack_scalars(
                delta, delta, surface_variables
            )[0]  # mean unused; same call returns (m, s) but for delta_std we just want vals
            # Upper-air delta_std mirrors the sigma+pressure stacking.
            sigma_d, _ = self._stack_levels(
                delta, delta, sigma_upper_air_variables, "Z_2", sigma_levels
            )
            pressure_d, _ = self._stack_levels(
                delta, delta, pressure_upper_air_variables, "Z", pressure_levels
            )
            self.target_upper_air_delta_std = self._cat_upper_air(sigma_d, pressure_d)
            # Sanity: deltas should be > 0 everywhere; zero std means the var
            # didn't change across timesteps in the source.
            if (self.target_surface_delta_std == 0).any():
                raise ValueError(
                    f"delta_std contains zero entries for some surface variables; "
                    f"check {delta_std_path}."
                )
            if (self.target_upper_air_delta_std == 0).any():
                raise ValueError(
                    f"delta_std contains zero entries for some upper-air variables; "
                    f"check {delta_std_path}."
                )
        else:
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
        # ERA5-style stats use 'pressure_level' / 'sigma_level' rather than the
        # PLASIM-native 'Z' / 'Z_2'. AMIP's per-variable normalize_mean.nc
        # uses the shorter 'level' spelling. Accept any of the three so
        # upstream stats files drop in unchanged.
        _aliases = {
            "Z": ("pressure_level", "level"),
            "Z_2": ("sigma_level",),
            "pressure_level": ("level",),
        }
        per_var_mean: list[np.ndarray] = []
        per_var_std: list[np.ndarray] = []
        for v in names:
            var_dims = mean[v].dims
            actual_dim = None
            if dim in var_dims:
                actual_dim = dim
            else:
                for alias in _aliases.get(dim, ()):
                    if alias in var_dims:
                        actual_dim = alias
                        break
            if actual_dim is None:
                raise ValueError(
                    f"Stats var {v!r} expected dim {dim} (or {_aliases.get(dim)}); "
                    f"got dims {var_dims}"
                )
            stats_levels = mean.coords[actual_dim].values.astype("float32")
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
    def to(self, device: torch.device | str) -> "ClimateNormalizer":
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
            "target_surface_delta_std",
            "target_upper_air_delta_std",
        ):
            v = getattr(self, name, None)
            if v is not None:
                setattr(self, name, v.to(device))
        return self

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(sample)
        # In predict_delta mode the target tensors must see the RAW start state
        # for the tendency computation, so handle targets BEFORE we mutate
        # surface_in / upper_air_in.
        if self._predict_delta:
            if "target_surface" in out and "surface_in" in out:
                out["target_surface"] = (
                    out["target_surface"] - out["surface_in"]
                ) / self.target_surface_delta_std
            if "target_upper_air" in out and "upper_air_in" in out:
                out["target_upper_air"] = (
                    out["target_upper_air"] - out["upper_air_in"]
                ) / self.target_upper_air_delta_std
        else:
            if "target_surface" in out and self.target_surface_mean is not None:
                out["target_surface"] = (
                    out["target_surface"] - self.target_surface_mean
                ) / self.target_surface_std
            if "target_upper_air" in out and self.target_upper_air_mean is not None:
                out["target_upper_air"] = (
                    out["target_upper_air"] - self.target_upper_air_mean
                ) / self.target_upper_air_std
        # Now normalize the state inputs (mutates the values the targets used).
        if "surface_in" in out and self.surface_mean is not None:
            out["surface_in"] = (out["surface_in"] - self.surface_mean) / self.surface_std
        if "upper_air_in" in out and self.upper_air_mean is not None:
            out["upper_air_in"] = (
                out["upper_air_in"] - self.upper_air_mean
            ) / self.upper_air_std
        # ArchesWeather previous-state inputs (t - k) share the current-state
        # stats (they are the same physical fields one step earlier). Present
        # only when the datapipe was built with prev_state_steps > 0.
        if "surface_prev_in" in out and self.surface_mean is not None:
            out["surface_prev_in"] = (
                out["surface_prev_in"] - self.surface_mean
            ) / self.surface_std
        if "upper_air_prev_in" in out and self.upper_air_mean is not None:
            out["upper_air_prev_in"] = (
                out["upper_air_prev_in"] - self.upper_air_mean
            ) / self.upper_air_std
        if "varying_boundary" in out and self.varying_mean is not None:
            out["varying_boundary"] = (
                out["varying_boundary"] - self.varying_mean
            ) / self.varying_std
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

        # Sequence (multi-step rollout) keys carry an extra leading time dim:
        # surface_in_seq:        (B, T+1, C_s, H, W)
        # upper_air_in_seq:      (B, T+1, C_u, L, H, W)
        # varying_boundary_seq:  (B, T+1, C_b, H, W)
        # diagnostic_seq:        (B, T+1, C_d, H, W)
        # Stats broadcast naturally — (C_s, 1, 1) right-aligns over the
        # trailing (C_s, H, W) of every time step. No reshape needed; the
        # predict_delta mode is intentionally not supported on `_seq` keys
        # (delta semantics belong to the single-step training path).
        if "surface_in_seq" in out and self.surface_mean is not None:
            out["surface_in_seq"] = (
                out["surface_in_seq"] - self.surface_mean
            ) / self.surface_std
        if "upper_air_in_seq" in out and self.upper_air_mean is not None:
            out["upper_air_in_seq"] = (
                out["upper_air_in_seq"] - self.upper_air_mean
            ) / self.upper_air_std
        if "varying_boundary_seq" in out and self.varying_mean is not None:
            out["varying_boundary_seq"] = (
                out["varying_boundary_seq"] - self.varying_mean
            ) / self.varying_std
        if (
            self._normalize_diagnostic
            and "diagnostic_seq" in out
            and self.diagnostic_mean is not None
        ):
            out["diagnostic_seq"] = (
                out["diagnostic_seq"] - self.diagnostic_mean
            ) / self.diagnostic_std
        return out

    def denormalize_state(
        self,
        *,
        surface: Optional[torch.Tensor] = None,
        upper_air: Optional[torch.Tensor] = None,
        diagnostic: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        r"""Map normalized state tensors back to physical units.

        Inverse of :meth:`__call__` for the prognostic state channels.
        Inputs may have any leading dims (batch, ensemble, frame, …) as
        long as the trailing dims align with the registered stats
        (``(C, H, W)`` for surface / diagnostic, ``(C, L, H, W)`` for
        upper-air). ``None`` inputs are skipped.

        Only the ``predict_delta=False`` semantics are inverted —
        tendency-mode predictions would need the *previous* state to add
        back, which is a different kind of operation than this helper
        provides. The forecast path (this codepath's only caller)
        always runs in non-delta mode at inference time.

        Diagnostic: if ``normalize_diagnostic=False`` the diagnostic
        channel was never z-scored at ``__call__`` time, so this method
        returns it unchanged (an idempotent passthrough).
        """
        if self._predict_delta:
            raise RuntimeError(
                "denormalize_state cannot invert tendency-mode predictions; "
                "predict_delta=True needs the previous physical-units state "
                "to add back. Use predict_delta=False for inference."
            )
        out: dict[str, torch.Tensor] = {}
        if surface is not None and self.surface_mean is not None:
            out["surface"] = surface * self.surface_std + self.surface_mean
        elif surface is not None:
            out["surface"] = surface
        if upper_air is not None and self.upper_air_mean is not None:
            out["upper_air"] = upper_air * self.upper_air_std + self.upper_air_mean
        elif upper_air is not None:
            out["upper_air"] = upper_air
        if diagnostic is not None:
            if self._normalize_diagnostic and self.diagnostic_mean is not None:
                out["diagnostic"] = (
                    diagnostic * self.diagnostic_std + self.diagnostic_mean
                )
            else:
                out["diagnostic"] = diagnostic
        return out

    @classmethod
    def from_dataset(
        cls,
        dataset,
        mean_path: str | Path,
        std_path: str | Path,
        **kwargs,
    ) -> "ClimateNormalizer":
        """Build a normalizer aligned with a :class:`ClimateZarrDataset`'s layout.

        Pass ``predict_delta=True`` + ``delta_std_path`` for tendency-mode
        target normalization.

        Any keyword in ``kwargs`` **overrides** the value derived from the
        dataset layout. This lets a caller pass, e.g., ``pressure_levels=[...]``
        to align the normalizer with a stats store that carries a different
        level set than the data (e.g. a 17-level model trained against an
        18-level archive). Without the override the dataset-derived value is
        used.
        """
        base = dict(
            surface_variables=dataset.layout.surface_variables,
            varying_boundary_variables=dataset.layout.varying_boundary_variables,
            sigma_upper_air_variables=dataset.layout.sigma_upper_air_variables,
            pressure_upper_air_variables=dataset.layout.pressure_upper_air_variables,
            sigma_levels=dataset.sigma_levels,
            pressure_levels=dataset.pressure_levels,
            constant_boundary_variables=dataset.layout.constant_boundary_variables,
            diagnostic_variables=dataset.layout.diagnostic_variables,
        )
        base.update(kwargs)  # caller-supplied values win (e.g. pressure_levels)
        return cls(mean_path, std_path, **base)


class NanFillTransform:
    r"""Replace NaN values in PLASIM boundary fields with configured fill values.

    PLASIM's land-sea-mask (``lsm``), sea-surface temperature (``sst``), and
    sea-ice-concentration (``sic``) carry NaN at high latitudes / over the
    other side of the land-ocean mask by convention. PanguPlasim's attention
    paths propagate NaN catastrophically — we substitute neutral numeric
    values here, BEFORE normalization, so downstream tensors are finite.

    The substitution runs on the CPU at the dataset's ``__getitem__`` boundary
    (transform=… kwarg). Per the Phase-2 contract this is composable with
    :class:`ClimateNormalizer`: chain via ``[NanFillTransform(...), normalizer]``.

    Parameters
    ----------
    constant_boundary_variables : sequence of str
        Channel-order list for ``constant_boundary`` (used to map var name → index).
    varying_boundary_variables : sequence of str
        Channel-order list for ``varying_boundary``.
    fill_values : dict[str, float], optional, default=None
        Per-variable fill values, e.g. ``{"sst": 273.15, "lsm": 0.0}``. Variables
        not in the dict use ``default``.
    default : float, optional, default=0.0
        Fallback fill for any boundary variable not listed in ``fill_values``.
    scan_constant : bool, optional, default=True
        Whether to scan ``constant_boundary`` for NaN.
    scan_varying : bool, optional, default=True
        Whether to scan ``varying_boundary`` for NaN.
    strict : bool, optional, default=False
        If True, raise if any NaN remains after the fill (sentinel for stats
        changes / unexpected source fields). Useful in tests + dev.
    smooth_nan_boundaries : bool, optional, default=False
        Replace the hard constant fill on the **boundary** groups with the
        masked-Gaussian coast fade of :func:`smooth_masked_boundary` (Phase
        12d.14; upstream amip_v2 ``smooth_nan_boundaries``). Valid values are
        preserved exactly and the masked region fades smoothly to the
        per-variable fill, so the land–sea seam does not stamp a step
        discontinuity into e.g. SST. Surface / diagnostic groups keep the
        hard fill either way (matching upstream, which only smooths
        boundaries).
    smooth_sigma : float, optional, default=1.5
        Per-step Gaussian std-dev in pixels (upstream AMIP default).
    smooth_kernel_size : int, optional, default=5
        Gaussian kernel width (upstream AMIP default).
    smooth_n_iters : int, optional, default=10
        Diffusion iterations — larger = wider, softer fade (upstream default).

    Forward
    -------
    sample : dict of torch.Tensor
        Output of :meth:`ClimateZarrDataset.__getitem__`.

    Outputs
    -------
    dict of torch.Tensor
        Same shape; ``constant_boundary`` and/or ``varying_boundary`` have
        their NaN replaced by the per-channel fill.
    """

    def __init__(
        self,
        *,
        constant_boundary_variables: Sequence[str] = (),
        varying_boundary_variables: Sequence[str] = (),
        surface_variables: Sequence[str] = (),
        diagnostic_variables: Sequence[str] = (),
        fill_values: Optional[dict[str, float]] = None,
        default: float = 0.0,
        scan_constant: bool = True,
        scan_varying: bool = True,
        strict: bool = False,
        smooth_nan_boundaries: bool = False,
        smooth_sigma: float = 1.5,
        smooth_kernel_size: int = 5,
        smooth_n_iters: int = 10,
    ) -> None:
        self._scan_constant = scan_constant
        self._scan_varying = scan_varying
        self._strict = strict
        self._default = float(default)
        # Phase 12d.14: masked-Gaussian coast fade instead of a hard constant
        # fill, for the BOUNDARY groups only (upstream ``_fill_mask`` smooths
        # boundaries; surface/diagnostic keep the hard fill).
        self._smooth = bool(smooth_nan_boundaries)
        self._smooth_sigma = float(smooth_sigma)
        self._smooth_kernel_size = int(smooth_kernel_size)
        self._smooth_n_iters = int(smooth_n_iters)
        fill_values = fill_values or {}
        self._const_fill = self._build_fill_tensor(
            constant_boundary_variables, fill_values
        )
        self._varying_fill = self._build_fill_tensor(
            varying_boundary_variables, fill_values
        )
        # Prognostic surface / diagnostic vars can also carry masked NaN (e.g.
        # sea_surface_temperature over land, now a prognostic surface var). Fill
        # them with the SAME per-variable mask values (Pangu v2.0 _fill_mask:
        # sst=270, skin/ts=270, sic/soil=0). (C,1,1) fill broadcasts against both
        # (C,H,W) and (prev,C,H,W) surface tensors.
        self._surface_fill = self._build_fill_tensor(surface_variables, fill_values)
        self._diagnostic_fill = self._build_fill_tensor(diagnostic_variables, fill_values)

    def _build_fill_tensor(
        self,
        names: Sequence[str],
        fill_values: dict[str, float],
    ) -> Optional[torch.Tensor]:
        if not names:
            return None
        return torch.tensor(
            [float(fill_values.get(n, self._default)) for n in names],
            dtype=torch.float32,
        ).view(-1, 1, 1)

    def _fill_boundary(
        self, tensor: torch.Tensor, fill: torch.Tensor
    ) -> torch.Tensor:
        """Hard per-channel fill, or the smooth coast fade when enabled."""
        if not self._smooth:
            return _nan_to_per_channel(tensor, fill)
        nans = torch.isnan(tensor)
        if not bool(nans.any()):
            return tensor
        out = tensor.clone()
        n_channels = tensor.shape[-3]
        for c in range(n_channels):
            ch_nans = nans[..., c, :, :]
            if not bool(ch_nans.any()):
                continue
            out[..., c, :, :] = _smooth_fill_channel(
                tensor[..., c, :, :],
                ch_nans,
                float(fill.reshape(-1)[c]),
                sigma=self._smooth_sigma,
                kernel_size=self._smooth_kernel_size,
                n_iters=self._smooth_n_iters,
            )
        return out

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(sample)
        if self._scan_constant and "constant_boundary" in out and self._const_fill is not None:
            out["constant_boundary"] = self._fill_boundary(
                out["constant_boundary"], self._const_fill
            )
            if self._strict:
                _assert_finite(out["constant_boundary"], "constant_boundary")
        if self._scan_varying and "varying_boundary" in out and self._varying_fill is not None:
            out["varying_boundary"] = self._fill_boundary(
                out["varying_boundary"], self._varying_fill
            )
            if self._strict:
                _assert_finite(out["varying_boundary"], "varying_boundary")
        if self._surface_fill is not None:
            for key in ("surface_in", "surface_prev_in", "target_surface"):
                if key in out and torch.is_tensor(out[key]):
                    out[key] = _nan_to_per_channel(out[key], self._surface_fill)
                    if self._strict:
                        _assert_finite(out[key], key)
        if self._diagnostic_fill is not None:
            for key in ("diagnostic", "target_diagnostic"):
                if key in out and torch.is_tensor(out[key]):
                    out[key] = _nan_to_per_channel(out[key], self._diagnostic_fill)
                    if self._strict:
                        _assert_finite(out[key], key)
        return out

    @classmethod
    def from_dataset(
        cls,
        dataset,
        **kwargs,
    ) -> "NanFillTransform":
        """Build a fill aligned with a :class:`ClimateZarrDataset` layout."""
        return cls(
            constant_boundary_variables=dataset.layout.constant_boundary_variables,
            varying_boundary_variables=dataset.layout.varying_boundary_variables,
            surface_variables=dataset.layout.surface_variables,
            diagnostic_variables=dataset.layout.diagnostic_variables,
            **kwargs,
        )


def _nan_to_per_channel(x: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    """Replace NaN with per-channel fill (broadcast against (C, H, W) or (B, C, H, W))."""
    return torch.where(torch.isnan(x), fill.to(x.dtype).to(x.device), x)


def _gaussian_kernel_2d(
    k: int, sigma: float, device, dtype=torch.float32
) -> torch.Tensor:
    """Separable-Gaussian 2D kernel, ``(1, 1, k, k)``, normalized to sum 1."""
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def smooth_masked_boundary(
    data: torch.Tensor,
    mask: torch.Tensor,
    sigma: float = 1.5,
    kernel_size: int = 5,
    n_iters: int = 10,
    lon_circular: bool = True,
) -> torch.Tensor:
    r"""Extend masked-region values outward with a smooth fade to zero
    (Phase 12d.14 — faithful port of amip_v2 ``data/amip.py``).

    Iterative Dirichlet diffusion: each step blurs the whole field with a
    Gaussian, then resets the inside of the mask to the original values, so
    the interior stays **exact** while the outside grows a smooth, curved
    fade. Used instead of a hard constant fill so the land–ocean seam does
    not stamp a step discontinuity (and hence the land–sea mask itself)
    into an SST-like boundary channel.

    Parameters
    ----------
    data : torch.Tensor
        ``(..., H, W)``. **Must be 0 wherever ``mask == 0``** — the fade is
        toward 0, so callers center the field on their fill value first (see
        :class:`NanFillTransform`).
    mask : torch.Tensor
        ``(..., H, W)`` binary; 1 = valid interior to preserve exactly.
    sigma : float, optional, default=1.5
        Per-step Gaussian std-dev in pixels; controls contour smoothness.
    kernel_size : int, optional, default=5
        Gaussian width. Upstream recommends ``>= 6 * sigma + 1``; the AMIP
        configs run ``sigma=1.5, kernel_size=5`` (narrower, cheaper).
    n_iters : int, optional, default=10
        Diffusion steps. Larger = wider, softer transition.
    lon_circular : bool, optional, default=True
        Treat the last axis as periodic (global longitude); latitude is
        always replicate-padded.

    Outputs
    -------
    torch.Tensor
        Same shape / dtype as ``data``.
    """
    out_dtype = data.dtype
    *batch, H, W = data.shape
    d = data.reshape(-1, 1, H, W).to(torch.float32)
    m = mask.reshape(-1, 1, H, W).to(torch.float32)

    kernel = _gaussian_kernel_2d(kernel_size, sigma, device=d.device)
    p = kernel_size // 2
    inv_m = 1.0 - m
    d_in = d * m

    state = d.clone()
    lon_mode = "circular" if lon_circular else "replicate"
    for _ in range(n_iters):
        s = torch.nn.functional.pad(state, (p, p, 0, 0), mode=lon_mode)
        s = torch.nn.functional.pad(s, (0, 0, p, p), mode="replicate")
        blurred = torch.nn.functional.conv2d(s, kernel)
        state = d_in + blurred * inv_m

    return state.reshape(*batch, H, W).to(out_dtype)


def _smooth_fill_channel(
    channel: torch.Tensor,
    nans: torch.Tensor,
    fill_val: float,
    *,
    sigma: float,
    kernel_size: int,
    n_iters: int,
) -> torch.Tensor:
    """One channel's NaN region filled by a smooth fade to ``fill_val``.

    Faithful to upstream ``AMIPDataset._fill_mask``'s smoothing branch:
    ``smooth_masked_boundary`` fades toward 0, so center the field on
    ``fill_val``, smooth, then add it back — the far field equals
    ``fill_val`` and the blend runs from the true coastal value down to it.
    """
    mask = (~nans).to(torch.float32)
    centered = torch.where(nans, torch.zeros_like(channel), channel - fill_val)
    smoothed = smooth_masked_boundary(
        centered,
        mask,
        sigma=sigma,
        kernel_size=kernel_size,
        n_iters=n_iters,
        lon_circular=True,
    )
    return smoothed + fill_val


def _assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise ValueError(
            f"NanFillTransform(strict=True): non-finite values remain in {name!r} "
            f"after filling. Check stats/source for unexpected NaN."
        )


class ComposeTransform:
    r"""Chain multiple sample-level transforms in order.

    Useful for stacking :class:`NanFillTransform` then :class:`ClimateNormalizer`
    as the ``transform=`` kwarg to :class:`ClimateZarrDataset`.
    """

    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def to(self, device):
        for t in self.transforms:
            if hasattr(t, "to"):
                t.to(device)
        return self


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
