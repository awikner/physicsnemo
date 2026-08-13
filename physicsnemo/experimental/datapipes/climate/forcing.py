# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Zhou, Carnegie Mellon University.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 12d.13 — the ai-rossby equivalent of amip_v2's
# ``AMIPDataset.forcing_from_raw`` (``data/amip.py`` @ e0b7b60).

r"""The single place stored boundary channels become model inputs.

Upstream amip_v2 funnels *every* entrypoint (training, validation,
``rollout.py``, ``bias.py``, ``test_diffusion.py``) through one
``forcing_from_raw`` method so "the channel contract cannot fork between
training and a 40-year rollout". This module is that choke point for the
fork, expressed as a composable dataset transform.

Division of labour (the fork's chain vs upstream's single method):

===========================  ============================================
upstream ``forcing_from_raw``  ai-rossby
===========================  ============================================
1. z-score every channel      :class:`~.transforms.ClimateNormalizer`
2. SST rescale / anomaly      ``sst_rescaler`` hook below (Phase 12g)
3. pop the CO2 channel        :class:`ForcingAssembler` (here)
4. emit the calendar row      dataset ``emit_calendar`` + assembler
===========================  ============================================

The recipes compose these in a fixed order via
``examples/weather/ai_rossby/dataset_setup.build_dataset_transform``, which
is the thing that actually enforces the no-fork property — see that module
for the ordering rationale (NaN-fill runs in **physical** units, therefore
strictly before the normalizer).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

__all__ = ["ForcingAssembler"]


class ForcingAssembler:
    r"""Route scalar-valued boundary channels out of ``c_grid`` into ``c_scalar``.

    A boundary channel that is *spatially uniform* — amip_v2's
    ``global_mean_co2`` is the motivating case — carries one number per
    frame. Leaving it in the gridded stream spends a full ``(H, W)`` map and
    a slice of the boundary encoder's width on a constant, which is why
    upstream pops it and appends it to the calendar row instead
    (``scalar_dim`` 2 → 3, ``c_grid_dim`` 6 → 5 in ``ERDM_co2.yaml``).

    This transform performs step 3 + 4 of the upstream contract: it removes
    the named channels from ``varying_boundary`` and appends their scalar
    values to ``calendar``, so the sample a model wrapper packs already has
    the final channel contract.

    .. note:: **Ordering.** Runs *after* the normalizer, so the appended
        scalar is in the same z-scored frame as the rest of the boundary
        stream (upstream reads its CO2 scalar off the normalized tensor too:
        ``scalar = boundary[0, 0, 0].item()`` *after*
        ``boundary_transform``). Placement is handled for you by
        ``dataset_setup.build_dataset_transform``.

    Parameters
    ----------
    varying_boundary_variables : sequence of str
        Channel-order list for the stored ``varying_boundary`` tensor.
    constant_boundary_variables : sequence of str, optional, default=()
        Channel-order list for ``constant_boundary``. Never routed to the
        scalar slot (they are time-invariant maps); used only so
        :attr:`c_grid_dim` can report the packed width.
    scalar_routed_variables : sequence of str, optional, default=()
        Names to pop out of ``varying_boundary`` and append to ``calendar``,
        in the order they should occupy the trailing scalar slots. Empty
        makes this transform a no-op passthrough.
    calendar_dim : int, optional, default=2
        Width of the dataset's calendar vector before routing
        (``(second_of_day, day_of_year)`` = 2, or ``(month, hour)`` = 2).
    sst_rescaler : callable, optional, default=None
        **Phase 12g hook.** Invoked as ``sample = sst_rescaler(sample)``
        before the pop, i.e. while SST is still a gridded channel, mirroring
        upstream's ordering (``sst_forcing.apply`` runs before the CO2 pop).
        A rescaler that needs *physical* units must arrange for its own raw
        copy — the normalizer has already run at this point.
    reduce : {"mean", "first"}, optional, default="mean"
        How a routed channel's map collapses to a scalar. ``"mean"`` is
        robust to the tiny per-gridpoint jitter a coarsened / interpolated
        store can carry; ``"first"`` reproduces upstream's literal
        ``boundary[0, 0, 0]`` read.
    uniform_atol : float, optional, default=1e-3
        Tolerance for the uniformity check on a routed channel (in the
        channel's own units — normalized, given the ordering above).
    strict : bool, optional, default=True
        Raise when a routed channel is missing, when the sample carries no
        ``calendar`` to append to, or when a routed channel is not uniform
        within ``uniform_atol``. With ``strict=False`` each of those degrades
        to a warning (and a non-uniform channel is still reduced).

    Forward
    -------
    sample : dict of torch.Tensor
        A :class:`~.dataset.ClimateZarrDataset` sample. Reads
        ``varying_boundary`` ``(C_v, H, W)`` and ``calendar``
        ``(calendar_dim,)``.

    Outputs
    -------
    dict of torch.Tensor
        ``varying_boundary`` narrowed to ``(C_v - n_routed, H, W)`` and
        ``calendar`` extended to ``(calendar_dim + n_routed,)``.

    Examples
    --------
    >>> import torch
    >>> asm = ForcingAssembler(
    ...     varying_boundary_variables=["global_mean_co2", "sst"],
    ...     constant_boundary_variables=["lsm"],
    ...     scalar_routed_variables=["global_mean_co2"],
    ... )
    >>> asm.c_grid_dim, asm.scalar_dim
    (2, 3)
    >>> sample = {
    ...     "varying_boundary": torch.stack(
    ...         [torch.full((4, 8), 1.5), torch.zeros(4, 8)]
    ...     ),
    ...     "calendar": torch.tensor([0.0, 12.0]),
    ... }
    >>> out = asm(sample)
    >>> tuple(out["varying_boundary"].shape), out["calendar"].tolist()
    ((1, 4, 8), [0.0, 12.0, 1.5])
    """

    def __init__(
        self,
        *,
        varying_boundary_variables: Sequence[str],
        constant_boundary_variables: Sequence[str] = (),
        scalar_routed_variables: Sequence[str] = (),
        calendar_dim: int = 2,
        sst_rescaler: Optional[Callable[[dict], dict]] = None,
        reduce: str = "mean",
        uniform_atol: float = 1e-3,
        strict: bool = True,
    ) -> None:
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.scalar_routed_variables = list(scalar_routed_variables)
        self.calendar_dim = int(calendar_dim)
        self.sst_rescaler = sst_rescaler
        if reduce not in ("mean", "first"):
            raise ValueError(f"reduce must be 'mean' or 'first', got {reduce!r}")
        self.reduce = reduce
        self.uniform_atol = float(uniform_atol)
        self.strict = bool(strict)

        missing = [
            v
            for v in self.scalar_routed_variables
            if v not in self.varying_boundary_variables
        ]
        if missing:
            msg = (
                f"scalar_routed_variables {missing} are not in "
                f"varying_boundary_variables {self.varying_boundary_variables}"
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning("%s — ignoring them", msg)
            self.scalar_routed_variables = [
                v for v in self.scalar_routed_variables if v not in missing
            ]

        # Channel bookkeeping, derived once (never restated by a config —
        # the amip_v2 ``state_layout`` lesson).
        #
        # Indices are taken against the order the pop actually SEES, which is
        # after the SST rescaler has run (upstream's step 2 before step 3): with
        # ``sst_anomaly_channel: append`` the derived channel is inserted right
        # after SST, so every stored channel beyond SST shifts by one. Indexing
        # the stored order instead silently drops the last channel — sea ice, in
        # the shipped AMIP list — with no shape error, because the widths still
        # add up.
        #
        # Not an upstream bug: amip_v2 pins CO2 to index 0 (``has_co2`` only
        # fires when it is the FIRST varying entry) and pops it positionally
        # with ``boundary[1:]``, and the anomaly is inserted at ``sst_index + 1``
        # with ``sst_index >= 1``, so an insertion can never disturb the channel
        # its pop removes. The exposure here is the price of this class's
        # generalization — routing *any* named channel from *any* position —
        # which is worth keeping, so the fix is to resolve the names later
        # rather than to pin the order.
        self._input_names = list(self.varying_boundary_variables)
        if self.sst_rescaler is not None:
            derived = getattr(self.sst_rescaler, "grid_forcing_names", None)
            if derived:
                self._input_names = list(derived)
        self._routed_idx = [
            self._input_names.index(v) for v in self.scalar_routed_variables
        ]
        self._keep_idx = [
            i
            for i in range(len(self._input_names))
            if i not in set(self._routed_idx)
        ]
        self.varying_boundary_variables_out = [
            self._input_names[i] for i in self._keep_idx
        ]
        self._warned_no_calendar = False

    # ------------------------------------------------------------------ #
    # Derived contract — what a model wrapper must be sized for.
    # ------------------------------------------------------------------ #

    @property
    def active(self) -> bool:
        """Whether any channel is actually routed (else a passthrough)."""
        return bool(self.scalar_routed_variables)

    @property
    def c_grid_dim(self) -> int:
        """Packed boundary width after routing (constant + kept varying).

        Includes the Phase-12g SST anomaly channel when the rescaler appends one
        — it is derived, not stored, but it *is* a channel the model sees, so a
        model config using ``sst_anomaly_channel: append`` must list it (see
        ``SSTForcing.grid_forcing_names``). ``ForcingPipeline.assert_matches``
        compares this against the wrapper's own count, so a config that lists it
        in one place and not the other fails at construction.
        """
        return len(self.constant_boundary_variables) + len(
            self.varying_boundary_variables_out
        )

    @property
    def scalar_dim(self) -> int:
        """Calendar-row width after routing.

        The Phase-12g ``global_mean_sst`` scalar occupies the same third slot the
        CO2 route uses, so it adds one exactly when CO2 does not — the two are
        mutually exclusive by construction (``resolve_scalar_forcing``).
        """
        emits_sst_scalar = bool(getattr(self.sst_rescaler, "emit_scalar", False))
        return (
            self.calendar_dim
            + len(self.scalar_routed_variables)
            + int(emits_sst_scalar)
        )

    # ------------------------------------------------------------------ #

    def _scalar_of(self, channel: torch.Tensor, name: str) -> torch.Tensor:
        """Collapse one routed channel's map to a scalar, checking uniformity."""
        flat = channel.reshape(-1)
        if self.strict or logger.isEnabledFor(logging.WARNING):
            spread = float(flat.max() - flat.min())
            if spread > self.uniform_atol:
                msg = (
                    f"scalar-routed boundary channel {name!r} is not spatially "
                    f"uniform (max-min={spread:.3e} > uniform_atol="
                    f"{self.uniform_atol:.3e}); routing it to the scalar slot "
                    f"would discard structure"
                )
                if self.strict:
                    raise ValueError(msg)
                logger.warning("%s — reducing anyway (reduce=%s)", msg, self.reduce)
        return flat.mean() if self.reduce == "mean" else flat[0]

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.sst_rescaler is not None:
            sample = self.sst_rescaler(sample)
        if not self.active:
            # No CO2-style routing: the rescaler (which may already have
            # appended the SST trend scalar to the calendar itself) is all there
            # was to do.
            return sample

        out = dict(sample)
        vb = out.get("varying_boundary")
        if vb is None:
            if self.strict:
                raise KeyError(
                    "ForcingAssembler: sample has no 'varying_boundary' to route "
                    f"{self.scalar_routed_variables} out of"
                )
            return out
        cal = out.get("calendar")
        if cal is None:
            # Upstream: a dataset that hands out no calendar hands out no
            # scalar either, and leaves the CO2 channel in the grid.
            msg = (
                "ForcingAssembler: scalar_routed_variables "
                f"{self.scalar_routed_variables} require a 'calendar' in the "
                "sample (dataset emit_calendar=True) — leaving the channels "
                "in varying_boundary"
            )
            if self.strict:
                raise KeyError(msg)
            if not self._warned_no_calendar:
                logger.warning(msg)
                self._warned_no_calendar = True
            return out

        scalars = [
            self._scalar_of(vb[..., i, :, :], name)
            for i, name in zip(self._routed_idx, self.scalar_routed_variables)
        ]
        keep = torch.as_tensor(self._keep_idx, device=vb.device)
        out["varying_boundary"] = vb.index_select(-3, keep)
        out["calendar"] = torch.cat(
            [
                cal,
                torch.stack(scalars).to(dtype=cal.dtype, device=cal.device),
            ],
            dim=-1,
        )
        return out

    def to(self, device) -> "ForcingAssembler":  # noqa: D401 - transform protocol
        """No buffers to move; present for :class:`ComposeTransform` parity."""
        return self

    @classmethod
    def from_dataset(
        cls, dataset, *, scalar_routed_variables: Sequence[str] = (), **kwargs
    ) -> "ForcingAssembler":
        """Build an assembler aligned with a dataset's stored boundary layout."""
        return cls(
            varying_boundary_variables=dataset.layout.varying_boundary_variables,
            constant_boundary_variables=dataset.layout.constant_boundary_variables,
            scalar_routed_variables=scalar_routed_variables,
            **kwargs,
        )
