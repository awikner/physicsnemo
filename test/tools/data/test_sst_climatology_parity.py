# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0

"""Phase 12g/12h — our SST climatology vs upstream's committed artifact.

On 2026-08-13 our fit (Polaris job 7448676) was compared against
``amip_v2:norm_stats/sst_climatology.npz`` — which is checked into the public
upstream repo — and every array came out **bitwise identical**:

    harmonic_coeffs (7, 180, 360)   max |diff| 0.000e+00, array_equal True
    ocean_weight / ocean_mask       array_equal True (40,191 ocean cells)
    gm_series (13,149 frames)       max |diff| 0.000e+00
    anom_std_map                    max |diff| 0.000e+00
    anom_std  0.5752863884          identical
    gm_std    0.1256771386          identical
    gm_mean   -2.6e-17 vs -9.5e-17  both floating-point zero (zero by
                                    construction: the intercept absorbs it)

That is a stronger result than parity of statistics. Our fit read a *different
storage backend* (per-year Zarr vs their single memmap), derived the ocean mask a
*different way* (reading the boundary store's NaN vs probing the original HDF5),
re-applied the loader's fill itself, and ran an independently written fitting
loop — so agreeing to the bit exercises the whole chain: the 12c HDF5→Zarr
conversion, 12d's NaN fill and coast smoothing, the fitting math, and the
day-of-year phase. The phase is the subtle one: this fork's calendar is
0-indexed and upstream's is 1-indexed, and a one-day error in
``year_fraction_from_calendar`` would shift the design-matrix rows and perturb
coefficients 1..6. They are exactly zero.

The scalars below are therefore pinned as regression constants — they hold
without either file present. The array comparison runs only when both artifacts
are reachable (ours is built on Polaris and not checked in at 2.5 MB); point
``AMIP_SST_CLIMATOLOGY`` / ``AMIP_V2_REPO`` at them to exercise it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Measured 2026-08-13 against upstream's committed artifact (see the module
# docstring). Fitted 1979-2015, 3 harmonics, stride 4, 13,149 daily frames.
UPSTREAM_ANOM_STD = 0.5752863884
UPSTREAM_GM_STD = 0.1256771386
UPSTREAM_OCEAN_CELLS = 40191
UPSTREAM_N_HARMONICS = 3
UPSTREAM_FIT_YEARS = (1979, 2015)
UPSTREAM_FRAMES = 13149
UPSTREAM_COEFF_SHAPE = (7, 180, 360)


def _find(env: str, *candidates: Path):
    p = os.environ.get(env)
    if p and Path(p).exists():
        return Path(p)
    for c in candidates:
        if c.exists():
            return c
    return None


def _ours():
    return _find(
        "AMIP_SST_CLIMATOLOGY",
        Path("/eagle/lighthouse-uchicago/physicsnemo-zarr/norm_stats/sst_climatology.npz"),
    )


def _upstream():
    repo = os.environ.get("AMIP_V2_REPO")
    if repo and (Path(repo) / "norm_stats" / "sst_climatology.npz").exists():
        return Path(repo) / "norm_stats" / "sst_climatology.npz"
    return None


def test_the_pinned_statistics_are_self_consistent():
    """Holds with no artifact present: the numbers we claim upstream parity on.

    A future refit that changes the window, the harmonic count or the stride
    changes these, and should update them deliberately rather than by accident —
    the fit is only valid over the TRAINING years, so silently widening it would
    put part of the ocean warming into the reference climatology.
    """
    assert UPSTREAM_FIT_YEARS == (1979, 2015)
    assert UPSTREAM_COEFF_SHAPE[0] == 1 + 2 * UPSTREAM_N_HARMONICS
    # 36 years at one frame per day, 9 of them leap.
    assert UPSTREAM_FRAMES == 36 * 365 + 9
    # The residual std must be far below the ~12.3 K absolute-SST std — that
    # ratio is the entire point of the artifact (0.03 sigma -> ~0.7 sigma).
    assert UPSTREAM_ANOM_STD < 1.0
    assert 12.3 / UPSTREAM_ANOM_STD > 20


@pytest.mark.skipif(_ours() is None, reason="our sst_climatology.npz not reachable")
def test_our_artifact_matches_the_pinned_statistics():
    z = np.load(_ours(), allow_pickle=False)
    assert int(z["n_harmonics"]) == UPSTREAM_N_HARMONICS
    assert (int(z["fit_year_start"]), int(z["fit_year_end"])) == UPSTREAM_FIT_YEARS
    assert tuple(z["harmonic_coeffs"].shape) == UPSTREAM_COEFF_SHAPE
    assert float(z["anom_std"]) == pytest.approx(UPSTREAM_ANOM_STD, abs=1e-9)
    assert float(z["gm_std"]) == pytest.approx(UPSTREAM_GM_STD, abs=1e-9)
    assert int((z["ocean_weight"] > 0).sum()) == UPSTREAM_OCEAN_CELLS
    assert len(z["gm_series"]) == UPSTREAM_FRAMES
    # Fork-only provenance keys: the fill the fit actually saw.
    assert float(z["fill_value"]) == 270.0
    assert bool(z["fill_smoothed"])


@pytest.mark.skipif(
    _ours() is None or _upstream() is None,
    reason="needs both our artifact and an amip_v2 checkout (AMIP_V2_REPO)",
)
def test_bitwise_identical_to_upstreams_artifact():
    """The full comparison, when both files are reachable."""
    up = np.load(_upstream(), allow_pickle=False)
    ou = np.load(_ours(), allow_pickle=False)

    # Every key upstream reads must be present; ours adds provenance keys only.
    assert set(up.files) <= set(ou.files)

    for key in ("harmonic_coeffs", "ocean_weight", "ocean_mask", "anom_std_map",
                "gm_series"):
        assert np.array_equal(up[key], ou[key]), f"{key} differs"
    for key in ("anom_std", "gm_std"):
        assert float(up[key]) == float(ou[key]), f"{key} differs"
    # gm_mean is zero by construction; only its last bits may differ.
    assert abs(float(up["gm_mean"])) < 1e-12
    assert abs(float(ou["gm_mean"])) < 1e-12
