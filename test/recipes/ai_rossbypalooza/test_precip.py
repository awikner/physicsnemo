# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for precip conversion (datapipes/precip.py)."""

from __future__ import annotations

import numpy as np
import pytest

from datapipes.precip import PrecipSpec


def _slabs(*values, shape=(3, 4)):
    return np.stack([np.full(shape, v, dtype=np.float32) for v in values])


# --------------------------------------------------------------------------- #
# lead_values
# --------------------------------------------------------------------------- #


def test_lead_values_6h():
    spec = PrecipSpec("tp", axis="6h", kind="accum", units="m")
    assert spec.lead_values(8) == [174, 180, 186, 192]  # tau*24-18 ... tau*24


def test_lead_values_daily_accum_and_cumulative():
    accum = PrecipSpec("tp", axis="daily", kind="accum", units="mm")
    assert accum.lead_values(8) == [8]
    cum = PrecipSpec("tp", axis="daily", kind="cumulative", units="m")
    assert cum.lead_values(8) == [7, 8]


def test_lead_values_day_offset_shifts():
    spec = PrecipSpec("tp", axis="6h", kind="accum", units="m", day_offset=-1)
    assert spec.lead_values(8) == [150, 156, 162, 168]
    daily = PrecipSpec("tp", axis="daily", kind="accum", units="mm", day_offset=1)
    assert daily.lead_values(8) == [9]


# --------------------------------------------------------------------------- #
# to_mm_per_day, one case per (axis, kind, units) combination in use
# --------------------------------------------------------------------------- #


def test_6h_accum_metres_sums_and_converts():
    spec = PrecipSpec("tp", axis="6h", kind="accum", units="m")
    out = spec.to_mm_per_day(_slabs(0.001, 0.002, 0.0, 0.003))
    np.testing.assert_allclose(out, 6.0, rtol=1e-6)  # 0.006 m -> 6 mm


def test_6h_accum_mm():
    spec = PrecipSpec("tp", axis="6h", kind="accum", units="mm")
    out = spec.to_mm_per_day(_slabs(1.0, 2.0, 3.0, 4.0))
    np.testing.assert_allclose(out, 10.0, rtol=1e-6)


def test_6h_rate_kg_m2_s():
    spec = PrecipSpec("tp", axis="6h", kind="rate", units="kg m-2 s-1")
    # 1e-4 mm/s sustained over four 6h windows = 1e-4 * 86400 = 8.64 mm
    out = spec.to_mm_per_day(_slabs(1e-4, 1e-4, 1e-4, 1e-4))
    np.testing.assert_allclose(out, 8.64, rtol=1e-5)


def test_daily_accum_metres():
    spec = PrecipSpec("tp", axis="daily", kind="accum", units="m")
    out = spec.to_mm_per_day(_slabs(0.012))
    np.testing.assert_allclose(out, 12.0, rtol=1e-6)


def test_daily_cumulative_diff_and_clip():
    spec = PrecipSpec("tp", axis="daily", kind="cumulative", units="m")
    out = spec.to_mm_per_day(_slabs(0.010, 0.0125))
    np.testing.assert_allclose(out, 2.5, rtol=1e-6)
    # Float noise producing a negative diff is clipped at 0.
    noisy = spec.to_mm_per_day(_slabs(0.0100001, 0.0100000))
    assert (noisy >= 0.0).all()
    np.testing.assert_allclose(noisy, 0.0, atol=1e-9)


def test_daily_rate():
    spec = PrecipSpec("tp", axis="daily", kind="rate", units="kg m-2 s-1")
    out = spec.to_mm_per_day(_slabs(2e-4))
    np.testing.assert_allclose(out, 17.28, rtol=1e-5)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def test_wrong_slab_count_raises():
    spec = PrecipSpec("tp", axis="6h", kind="accum", units="m")
    with pytest.raises(ValueError, match="needs 4 slabs"):
        spec.to_mm_per_day(_slabs(1.0, 2.0))


def test_invalid_specs_raise():
    with pytest.raises(ValueError, match="daily axis"):
        PrecipSpec("tp", axis="6h", kind="cumulative", units="m")
    with pytest.raises(ValueError, match="rate precip units"):
        PrecipSpec("tp", axis="6h", kind="rate", units="m")
    with pytest.raises(ValueError, match="depth precip units"):
        PrecipSpec("tp", axis="daily", kind="accum", units="kg m-2 s-1")
    with pytest.raises(ValueError, match="must divide 24"):
        PrecipSpec("tp", axis="6h", kind="accum", units="m", step_hours=7)


# --------------------------------------------------------------------------- #
# LogPrecipTransform (model v1)
# --------------------------------------------------------------------------- #


def test_log_transform_roundtrip_numpy_and_torch():
    import torch

    from datapipes.precip import LogPrecipTransform

    t = LogPrecipTransform(epsilon=1e-3, units="m")
    mm = np.array([0.0, 0.5, 5.0, 50.0, 200.0], dtype=np.float64)
    y = t.forward(mm)
    # log(1e-3 + P[m]): dry day -> log(1e-3), 50 mm -> log(0.051).
    np.testing.assert_allclose(y[0], np.log(1e-3), rtol=1e-12)
    np.testing.assert_allclose(y[3], np.log(1e-3 + 0.05), rtol=1e-12)
    np.testing.assert_allclose(t.inverse(y), mm, rtol=1e-10, atol=1e-10)
    tt = torch.tensor(mm)
    ty = t.forward(tt)
    np.testing.assert_allclose(ty.numpy(), y, rtol=1e-10)
    torch.testing.assert_close(t.inverse(ty), tt, rtol=1e-8, atol=1e-8)
    # Inverse clamps below-dry-floor values at 0 mm.
    assert t.inverse(np.array([-20.0]))[0] == 0.0
    assert t.inverse(torch.tensor([-20.0]))[0] == 0.0


def test_log_transform_validation():
    from datapipes.precip import LogPrecipTransform

    with pytest.raises(ValueError, match="units"):
        LogPrecipTransform(units="inches")
    with pytest.raises(ValueError, match="epsilon"):
        LogPrecipTransform(epsilon=0.0)
