# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

r"""SST boundary rescaling and the global-mean-SST trend scalar (Phase 12g).

Vendored from amip_v2 ``data/sst_forcing.py`` @ ``e0b7b60``.

Why this exists (upstream's reasoning, kept because it is the whole point)
-------------------------------------------------------------------------
SST reaches the model through the normalizer, which z-scores each forcing
channel by one global scalar. For SST that divisor is the std of *absolute*
SST, ~12.3 K — dominated by the equator-to-pole contrast. The signal an AMIP
emulator has to track, the 1979–2022 ocean warming, is +0.40 K in the global
mean. Divided by 12.3 K it arrives as a **0.03 sigma** perturbation of a field
whose seasonal and synoptic variability is O(1 sigma) by construction. The CO2
scalar spans 3.56 sigma over the same years, which is why the CO2 configs
capture the trend and an SST-only run does not: the forcing is present in both,
but only one is legible.

Two independent switches make it legible without changing the physics:

``sst_anomaly_channel``
    Subtract a fixed day-of-year climatology (harmonic fit over the *training*
    years) and divide by the std of the **residual** (~0.6 K, not 12.3 K). The
    same +0.40 K then arrives as ~0.7 sigma. ``append`` keeps the untouched
    absolute channel alongside it — the model retains the mean state that
    drives the atmosphere and gains a well-scaled departure channel;
    ``replace`` substitutes it.

``scalar_forcing: global_mean_sst``
    The ocean-area-weighted mean of that anomaly field, z-scored, routed into
    the third calendar slot — the slot the CO2 configs use, carried by the
    affine head in ``ScalarForcingEmbedder`` that exists to extrapolate past
    the training range. The AMIP-clean analogue of the CO2 scalar.

Both read one artifact, ``sst_climatology.npz``, written by
``tools/data/amip/make_sst_climatology.py``. **The ``.npz`` key set is
upstream's exactly**, so artifacts are interchangeable in both directions.

The climatology is fitted on the **stored** SST field — the one the loader
serves, NaN-filled and coast-smoothed. That is deliberate: over land the
climatology and the field are the same constant-ish surface, so the anomaly is
~0 there and stays continuous across the coast instead of stamping the land-sea
mask into a channel meant to carry ocean variability. The ocean mask in the
artifact is used only for the area weights of the global-mean scalar and for
pooling the anomaly std.

Fork difference: **where the physical SST comes from.** Upstream computes the
anomaly inside ``forcing_from_raw``, which still holds the raw kelvin frame.
This fork's chain is ``NaN-fill -> normalizer -> ForcingAssembler``, and the
rescaler runs inside the assembler (upstream's step 2 ordering, before the CO2
pop), by which point SST is z-scored. So :meth:`SSTForcing.apply` takes the
channel's ``(mean, std)`` and inverts the z-score to recover kelvin. That
round-trip is exact to ~2e-5 K in float32 on a ~300 K field, five orders below
the ~0.6 K residual it has to resolve.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "ANOMALY_MODES",
    "SSTRescaler",
    "DEFAULT_SST_CLIMATOLOGY_CACHE",
    "SST_ANOMALY_CHANNEL_NAME",
    "SST_VARIABLE_NAMES",
    "SSTForcing",
    "YEAR_LENGTH_DAYS",
    "harmonic_design_row",
    "year_fraction",
    "year_fraction_from_calendar",
]

DEFAULT_SST_CLIMATOLOGY_CACHE = "sst_climatology.npz"

#: Days per year for the seasonal phase. Fixed rather than calendar-aware: the
#: fit and the evaluation share this constant, so a leap year only wobbles the
#: phase of a smooth annual cycle by well under a day.
YEAR_LENGTH_DAYS = 365.25

#: SST variable spellings a climatology may apply to. The daily-average archive
#: interpolates monthly SST, so both map to the same field.
SST_VARIABLE_NAMES = (
    "sea_surface_temperature_monthly_interp",
    "sea_surface_temperature",
)

#: Name of the anomaly channel :meth:`SSTForcing.apply` derives. It has no entry
#: in a store's ``varying_boundary_variables`` — it is computed, not stored — so
#: it needs a pseudo-name to be addressable from a config. Also what
#: ``RollingDiTWrapper.ocean_state_variables`` names when the anomaly is one of
#: the predicted ocean channels.
SST_ANOMALY_CHANNEL_NAME = "sea_surface_temperature_anomaly"

ANOMALY_MODES = ("none", "append", "replace")


def year_fraction(second_of_day, day_of_year):
    """Seasonal phase in ``[0, 1)`` from a **1-indexed** ``(sod, doy)`` pair.

    Upstream's semantics, kept bit-identical so an artifact written by either
    repo evaluates the same: ``day_of_year`` counts from 1
    (``AMIPDataset._compute_calendar`` returns ``seconds // 86400 + 1``), so
    Jan 1 at midnight is phase 0.

    .. warning:: **This fork's calendar row is 0-indexed.**
        ``ClimateZarrDataset._decompose_time`` returns ``dayofyr - 1`` and
        ``_calendar_vector`` emits it unchanged, so a calendar row from this
        loader must go through :func:`year_fraction_from_calendar`, not
        straight in here. Passing it raw shifts every evaluation one day in
        phase — small against a smooth annual cycle, invisible in any shape or
        loss, and enough to stop upstream-written artifacts from being
        interchangeable.
    """
    return ((day_of_year - 1) + second_of_day / 86400.0) / YEAR_LENGTH_DAYS


def year_fraction_from_calendar(calendar_row) -> float:
    """Seasonal phase from this fork's ``(second_of_day, day_of_year)`` row.

    The one place the 0-indexed/1-indexed conversion lives (see the warning on
    :func:`year_fraction`). Takes anything indexable — a ``(2,)`` / ``(3,)``
    tensor, a list, a tuple.
    """
    sod = float(calendar_row[0])
    doy_zero_based = float(calendar_row[1])
    return year_fraction(sod, doy_zero_based + 1.0)


def harmonic_design_row(year_frac, n_harmonics: int) -> np.ndarray:
    """``[1, cos(2πt), sin(2πt), cos(4πt), …]`` for one time, float64.

    Shared by the fitting tool and the runtime evaluation, in this module, so
    the two cannot drift apart.
    """
    t = np.atleast_1d(np.asarray(year_frac, dtype=np.float64))
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t))
        cols.append(np.sin(2 * np.pi * k * t))
    return np.stack(cols, axis=-1)


class SSTForcing:
    r"""Day-of-year SST climatology plus the derived anomaly channel and scalar.

    Build with :meth:`from_config`, which returns ``None`` when the config asks
    for neither feature so callers can keep the attribute optional.

    Parameters
    ----------
    path : str
        The ``sst_climatology.npz`` artifact.
    anomaly_mode : {"none", "append", "replace"}, optional, default="none"
        Whether to derive the anomaly channel and, if so, whether it rides
        alongside the absolute SST channel or takes its place.
    global_mean_scalar : bool, optional, default=False
        Derive the ocean-mean anomaly scalar for the calendar row's third slot.
    anomaly_scale, scalar_scale : str or float
        ``"anom_std"`` / ``"gm_std"`` use the fitted statistics; a number is a
        literal divisor in kelvin (for forced +ΔT experiments — see
        :meth:`_resolve_scale`).
    """

    def __init__(
        self,
        path: str,
        anomaly_mode: str = "none",
        global_mean_scalar: bool = False,
        anomaly_scale="anom_std",
        scalar_scale="gm_std",
    ) -> None:
        if anomaly_mode not in ANOMALY_MODES:
            raise ValueError(
                f"sst_anomaly_channel must be one of {ANOMALY_MODES}, "
                f"got {anomaly_mode!r}"
            )
        self.anomaly_mode = anomaly_mode
        self.global_mean_scalar = bool(global_mean_scalar)
        self.path = str(path)

        try:
            z = np.load(self.path, allow_pickle=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"SST climatology not found at {self.path}. Build it once with:\n"
                f"  python tools/data/amip/make_sst_climatology.py "
                f"--zarr <store-root> --year-start 1979 --year-end 2015"
            ) from None

        self.n_harmonics = int(z["n_harmonics"])
        self.coeffs = torch.from_numpy(
            np.asarray(z["harmonic_coeffs"], dtype=np.float32)
        )  # (n_coef, H, W)
        self.ocean_weight = torch.from_numpy(
            np.asarray(z["ocean_weight"], dtype=np.float32)
        )  # (H, W), sums to 1
        self.anom_std = float(z["anom_std"])
        self.gm_mean = float(z["gm_mean"])
        self.gm_std = float(z["gm_std"])
        self.fit_years = (int(z["fit_year_start"]), int(z["fit_year_end"]))

        if self.anom_std <= 0 or self.gm_std <= 0:
            raise ValueError(
                f"{self.path} holds a non-positive scale (anom_std="
                f"{self.anom_std}, gm_std={self.gm_std}); regenerate it."
            )

        self.anomaly_scale = self._resolve_scale(anomaly_scale, "sst_anomaly_scale")
        self.scalar_scale = self._resolve_scale(scalar_scale, "sst_scalar_scale")
        expected_coef = 1 + 2 * self.n_harmonics
        if self.coeffs.shape[0] != expected_coef:
            raise ValueError(
                f"{self.path} holds {self.coeffs.shape[0]} harmonic "
                f"coefficients but n_harmonics={self.n_harmonics} implies "
                f"{expected_coef}."
            )

    def _resolve_scale(self, spec, key: str) -> float:
        """Kelvin per unit for one derived channel.

        ``"anom_std"`` / ``"gm_std"`` take the fitted statistics; a number is a
        literal divisor in kelvin. Which to use is a *conditioning* choice, not
        an information one: dividing a channel by a constant cannot change what
        it carries, only its amplitude at the input layer and — the reason this
        knob exists — how far outside the training envelope a forced
        perturbation lands. The climatology subtraction is what recovers the
        trend signal; the divisor only decides how loudly it arrives.

        The fitted statistics maximise amplitude and are right for historical
        runs. A physical unit (1–2 K) is right when the plan includes forced
        +2 K / +4 K experiments: at the fitted scales a uniform +4 K displaces
        the anomaly channel by +7 and the scalar by +32, against training ranges
        of roughly ±5 and ±2.4.
        """
        if isinstance(spec, str):
            key_map = {"anom_std": self.anom_std, "gm_std": self.gm_std}
            if spec.lower() not in key_map:
                raise ValueError(
                    f"{key} must be 'anom_std', 'gm_std' or a number of "
                    f"kelvin, got {spec!r}"
                )
            return key_map[spec.lower()]
        scale = float(spec)
        if scale <= 0:
            raise ValueError(f"{key} must be positive, got {scale}")
        return scale

    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        requires_scalar: bool = False,
        path: Optional[str] = None,
    ) -> Optional["SSTForcing"]:
        """Build from a dataset config block, or ``None`` when unused.

        ``requires_scalar`` is set by the caller when ``scalar_forcing``
        resolved to ``global_mean_sst``, so a config asking for the scalar alone
        still loads the artifact. ``path`` overrides
        ``cfg.sst_climatology_path`` (the recipes resolve it to an absolute path
        first).
        """
        get = cfg.get if hasattr(cfg, "get") else cfg.__getitem__
        mode = str(get("sst_anomaly_channel", "none") or "none").lower()
        if mode == "none" and not requires_scalar:
            return None
        resolved = path or get("sst_climatology_path", None)
        if not resolved:
            raise ValueError(
                "sst_anomaly_channel / scalar_forcing: global_mean_sst need "
                "dataset.sst_climatology_path to point at the artifact "
                "(tools/data/amip/make_sst_climatology.py writes it)"
            )
        return cls(
            resolved,
            anomaly_mode=mode,
            global_mean_scalar=requires_scalar,
            anomaly_scale=get("sst_anomaly_scale", "anom_std"),
            scalar_scale=get("sst_scalar_scale", "gm_std"),
        )

    def describe(self) -> str:
        return (
            f"SST forcing: anomaly_channel={self.anomaly_mode}, "
            f"global_mean_scalar={self.global_mean_scalar}, climatology fit "
            f"{self.fit_years[0]}-{self.fit_years[1]} ({self.n_harmonics} "
            f"harmonics); anomaly channel / {self.anomaly_scale:.3f} K, scalar "
            f"/ {self.scalar_scale:.3f} K (fitted: anom_std "
            f"{self.anom_std:.3f} K, gm_std {self.gm_std:.3f} K)"
        )

    @property
    def adds_channel(self) -> bool:
        """Whether this adds one channel to the gridded forcing stack."""
        return self.anomaly_mode == "append"

    def grid_forcing_names(self, varying_names: Sequence[str]) -> list[str]:
        """``varying_names`` with the anomaly pseudo-channel inserted.

        Mirrors upstream ``AMIPDataset.grid_forcing_names``: ``append`` puts the
        anomaly immediately after the absolute channel — the pair reads as "SST,
        and how anomalous it is" in every channel listing — and ``replace``
        substitutes it in place. Model configs must list the result, since
        that is what sizes ``c_grid_dim``.
        """
        names = list(varying_names)
        if self.anomaly_mode == "none":
            return names
        idx = self.sst_index(names)
        if self.adds_channel:
            names.insert(idx + 1, SST_ANOMALY_CHANNEL_NAME)
        else:
            names[idx] = SST_ANOMALY_CHANNEL_NAME
        return names

    @staticmethod
    def sst_index(varying_names: Sequence[str]) -> int:
        """Channel of the SST field within a varying-boundary name list."""
        for i, name in enumerate(varying_names):
            if name in SST_VARIABLE_NAMES:
                return i
        raise ValueError(
            f"sst_anomaly_channel / scalar_forcing: global_mean_sst need an SST "
            f"channel in varying_boundary_variables={list(varying_names)}; "
            f"expected one of {SST_VARIABLE_NAMES}"
        )

    # ------------------------------------------------------------------

    def climatology(self, second_of_day, day_of_year) -> torch.Tensor:
        """Fitted SST climatology ``(H, W)`` for one time, in kelvin.

        ``day_of_year`` is **1-indexed** (upstream's convention — see
        :func:`year_fraction`). Callers holding this fork's calendar row should
        use :meth:`climatology_at` instead.
        """
        row = harmonic_design_row(
            year_fraction(second_of_day, day_of_year), self.n_harmonics
        )[0]
        w = torch.from_numpy(row.astype(np.float32)).to(self.coeffs.device)
        return torch.tensordot(w, self.coeffs, dims=([0], [0]))  # (H, W)

    def climatology_at(self, calendar_row) -> torch.Tensor:
        """:meth:`climatology` for one of **this fork's** calendar rows."""
        row = harmonic_design_row(
            year_fraction_from_calendar(calendar_row), self.n_harmonics
        )[0]
        w = torch.from_numpy(row.astype(np.float32)).to(self.coeffs.device)
        return torch.tensordot(w, self.coeffs, dims=([0], [0]))  # (H, W)

    def anomaly(self, sst: torch.Tensor, calendar_row) -> torch.Tensor:
        """Departure of one SST frame (kelvin) from its climatology, in kelvin.

        ``calendar_row`` is this fork's ``(second_of_day, day_of_year[, scalar])``.
        """
        clim = self.climatology_at(calendar_row)
        return sst - clim.to(device=sst.device, dtype=sst.dtype)

    def apply(
        self,
        normalized_boundary: torch.Tensor,
        sst_index: int,
        calendar_row,
        *,
        sst_mean: float,
        sst_std: float,
    ) -> tuple[torch.Tensor, Optional[float]]:
        """Rescale/extend the SST channel and derive the global-mean scalar.

        ``normalized_boundary`` is ``(c_b, H, W)`` **z-scored** (this fork
        normalizes before the assembler runs — see the module docstring), so the
        physical field is recovered as ``normalized * sst_std + sst_mean``.

        Returns ``(boundary, gm_scalar)``; ``gm_scalar`` is the z-scored
        ocean-mean anomaly, or ``None`` when the scalar was not requested.
        """
        sst_k = normalized_boundary[sst_index] * float(sst_std) + float(sst_mean)
        # Checked before the subtraction so the message names the cause; a
        # mismatch would otherwise surface as a broadcast error from inside
        # ``anomaly``, or — worse, if one axis happened to be 1 — broadcast
        # silently.
        if tuple(sst_k.shape) != tuple(self.coeffs.shape[-2:]):
            raise ValueError(
                f"climatology grid is {tuple(self.coeffs.shape[-2:])} but the "
                f"SST field is {tuple(sst_k.shape)}; the artifact was fitted "
                f"on a different grid than this store serves"
            )
        anom = self.anomaly(sst_k, calendar_row)

        gm = None
        if self.global_mean_scalar:
            weight = self.ocean_weight.to(device=anom.device, dtype=anom.dtype)
            if weight.shape != anom.shape:
                raise ValueError(
                    f"climatology ocean_weight is {tuple(weight.shape)} but the "
                    f"SST field is {tuple(anom.shape)}; the artifact is internally "
                    f"inconsistent — regenerate it"
                )
            gm = float((anom * weight).sum())
            gm = (gm - self.gm_mean) / self.scalar_scale

        if self.anomaly_mode != "none":
            scaled = (anom / self.anomaly_scale).unsqueeze(0)
            if self.anomaly_mode == "replace":
                normalized_boundary = torch.cat(
                    [
                        normalized_boundary[:sst_index],
                        scaled,
                        normalized_boundary[sst_index + 1 :],
                    ],
                    dim=0,
                )
            else:
                normalized_boundary = torch.cat(
                    [
                        normalized_boundary[: sst_index + 1],
                        scaled,
                        normalized_boundary[sst_index + 1 :],
                    ],
                    dim=0,
                )

        return normalized_boundary, gm


class SSTRescaler:
    r"""Adapter that makes an :class:`SSTForcing` the assembler's hook.

    :class:`~.forcing.ForcingAssembler` calls its ``sst_rescaler`` as
    ``sample = sst_rescaler(sample)`` before the CO2 pop — upstream's step-2
    ordering. This wraps the stateless math above with the two things it needs
    from the run: where SST sits in the varying stream, and the ``(mean, std)``
    that invert the normalizer's z-score back to kelvin.

    Parameters
    ----------
    forcing : SSTForcing
    varying_boundary_variables : sequence of str
        The store's varying-boundary order, i.e. the channel order of
        ``sample["varying_boundary"]`` as the normalizer leaves it.
    sst_mean, sst_std : float
        The normalizer's statistics for the SST channel
        (``ClimateNormalizer.varying_mean/std`` at :meth:`SSTForcing.sst_index`).
    emit_scalar : bool, optional, default=None
        Append the global-mean anomaly to the calendar row. Defaults to the
        forcing's own ``global_mean_scalar``.

    Notes
    -----
    Per-frame only: this runs inside the dataset transform, before
    :class:`~.sequence.SequenceDataset` stacks anything, so
    ``varying_boundary`` is ``(C, H, W)``. A stacked tensor is refused rather
    than silently mis-indexed.
    """

    def __init__(
        self,
        forcing: SSTForcing,
        varying_boundary_variables: Sequence[str],
        *,
        sst_mean: float,
        sst_std: float,
        emit_scalar: Optional[bool] = None,
    ) -> None:
        self.forcing = forcing
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.sst_index = SSTForcing.sst_index(self.varying_boundary_variables)
        self.sst_mean = float(sst_mean)
        self.sst_std = float(sst_std)
        self.emit_scalar = (
            bool(forcing.global_mean_scalar) if emit_scalar is None else bool(emit_scalar)
        )
        if self.sst_std <= 0:
            raise ValueError(
                f"the SST channel's normalizer std is {self.sst_std}; the "
                f"z-score cannot be inverted"
            )

    @property
    def grid_forcing_names(self) -> list[str]:
        """Varying-channel order after this rescaler runs."""
        return self.forcing.grid_forcing_names(self.varying_boundary_variables)

    @property
    def adds_channel(self) -> bool:
        return self.forcing.adds_channel

    def __call__(self, sample: dict) -> dict:
        vb = sample.get("varying_boundary")
        if vb is None:
            raise KeyError(
                "SSTRescaler needs 'varying_boundary' in the sample; the SST "
                "channel is what it rescales"
            )
        if vb.dim() != 3:
            raise ValueError(
                f"SSTRescaler runs per frame (C, H, W) inside the dataset "
                f"transform, got {tuple(vb.shape)} — it must run before the "
                f"sequence stacking, not after"
            )
        cal = sample.get("calendar")
        if cal is None:
            raise KeyError(
                "SSTRescaler needs 'calendar' in the sample (dataset "
                "emit_calendar=True) to know the day of year"
            )

        out = dict(sample)
        boundary, gm = self.forcing.apply(
            vb,
            self.sst_index,
            cal,
            sst_mean=self.sst_mean,
            sst_std=self.sst_std,
        )
        out["varying_boundary"] = boundary
        if self.emit_scalar:
            if gm is None:
                raise RuntimeError(
                    "SSTRescaler was asked to emit the trend scalar but the "
                    "SSTForcing did not compute one (global_mean_scalar=False)"
                )
            # Third calendar slot — the one the CO2 configs use, which is why a
            # config cannot ask for both (see build_forcing_pipeline).
            out["calendar"] = torch.cat(
                [cal, torch.tensor([gm], dtype=cal.dtype, device=cal.device)],
                dim=-1,
            )
        return out
